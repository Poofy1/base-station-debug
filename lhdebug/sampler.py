"""The engine: pulls frames from a Source, runs detection and attribution,
records the session, and publishes messages for the dashboard."""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from typing import Callable

from .attribution import Attributor
from .detectors import DetectorBank
from .heatmap import Heatmap
from .model import DeviceClass, Frame, Incident, LogEvent, Source, to_json
from .rangeprofile import RangeProfile
from .store import Store
from .world import World

STATE_HZ = 10.0
BASELINE_HZ = 10.0
FINDINGS_EVERY_S = 5.0
BURST_PRE_S = 2.0
BURST_POST_S = 2.0
KEEP_INCIDENTS = 2000
KEEP_LOG = 2000
DEFAULT_WINDOW_S = 900.0   # rolling buffer: anything older is dropped
MIN_WINDOW_S, MAX_WINDOW_S = 60.0, 4 * 3600.0
PRUNE_EVERY_S = 10.0


class Engine:
    def __init__(self, source: Source, log_source=None, store: Store | None = None,
                 mode: str = "live", voxel: float = 0.2) -> None:
        self.source = source
        self.log_source = log_source          # anything with drain() -> list[LogEvent]
        self.store = store
        self.mode = mode
        self.started = time.time()
        self.world = World()
        self.detectors = DetectorBank(self.world)
        self.heatmap = Heatmap(voxel)
        self.attributor = Attributor(self.world)
        self.range = RangeProfile()
        self._next_range = 0.0
        self.play_area: dict | None = None
        self.incidents: deque[Incident] = deque(maxlen=KEEP_INCIDENTS)
        self.log: deque[LogEvent] = deque(maxlen=KEEP_LOG)
        self.lock = threading.RLock()         # guards everything above for the HTTP threads
        self.connected = False
        self.frames = 0
        self._subs: list[Callable[[dict], None]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._next_state = 0.0
        self._next_baseline = 0.0
        self._next_findings = 0.0
        self._bursts: dict[str, float] = {}   # serial -> record full rate until t
        self._devices_sig: tuple = ()
        self._findings_cache: list = []
        self.window_s = DEFAULT_WINDOW_S
        self.recording = True
        self.paused_at: float | None = None
        self._next_prune = 0.0
        if store:
            store.set_meta("window_s", str(self.window_s))

    # ---- pub/sub -------------------------------------------------------
    def subscribe(self, fn: Callable[[dict], None]) -> None:
        self._subs.append(fn)

    def unsubscribe(self, fn: Callable[[dict], None]) -> None:
        if fn in self._subs:
            self._subs.remove(fn)

    def _publish(self, msg: dict) -> None:
        for fn in list(self._subs):
            try:
                fn(msg)
            except Exception:
                pass

    # ---- lifecycle -----------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self.run, name="lhdebug-engine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def run(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    frame = self.source.read()
                except StopIteration:
                    break
                self.step(frame)
        finally:
            self.finish()

    def finish(self) -> None:
        with self.lock:
            for inc in self.detectors.flush():
                self._on_incident(inc)
            self._findings_cache = self._compute_findings()
            if self.store:
                self.store.set_meta("findings", json.dumps(to_json(self._findings_cache)))
                self.store.set_meta("ended", str(time.time()))
                self.store.flush()
        self.source.close()

    # ---- per-frame -----------------------------------------------------
    def step(self, frame: Frame | None) -> None:
        out: list[dict] = []
        with self.lock:
            devices = self.source.devices()
            self.world.set_devices(devices)
            events = list(self.source.drain_log())
            if self.log_source is not None:
                events += self.log_source.drain()
            for ev in events:
                self.world.add_log(ev)
                if not self.recording:
                    continue
                self.log.append(ev)
                if self.store:
                    self.store.put_log(ev)
                out.append({"type": "log", "event": to_json(ev)})

            area = getattr(self.source, "play_area", lambda: None)()
            if area != self.play_area:
                self.play_area = area
                out.append({"type": "session", "session": self.session_info()})

            self.connected = frame is not None
            now = frame.t if frame is not None else time.time()
            if frame is not None:
                self.frames += 1
                self.world.add_frame(frame)
                # detectors always run so device health stays live; while recording
                # is stopped their incidents are discarded and nothing is stored
                closed = self.detectors.feed(frame)
                if self.recording:
                    for inc in closed:
                        out.append(self._on_incident(inc))
                    for s in frame.samples:
                        info = devices.get(s.serial)
                        if info is None or info.device_class is DeviceClass.BASE:
                            continue
                        self.heatmap.add(s, self.detectors.metrics(s.serial))
                    if now >= self._next_range:
                        self._next_range = now + 0.1
                        self._range_tick(frame, devices)
                    self._record(frame, devices)
                    if now >= self._next_prune:
                        self._next_prune = now + PRUNE_EVERY_S
                        self._prune(now)

            if now >= self._next_state:
                self._next_state = now + (1.0 / STATE_HZ if frame is not None else 1.0)
                out.append(self.state_message(now))
            if now >= self._next_findings and self.recording:
                self._next_findings = now + FINDINGS_EVERY_S
                self._findings_cache = self._compute_findings()
                payload = to_json(self._findings_cache)
                out.append({"type": "findings", "findings": payload})
                if self.store:      # keep the recording usable if the window is just closed
                    self.store.set_meta("findings", json.dumps(payload))
        for msg in out:
            self._publish(msg)

    def _compute_findings(self) -> list:
        findings = list(self.attributor.findings())
        far = self.range.finding()
        if far is not None:
            findings.append(far)
            findings.sort(key=lambda f: -f.score)
        return findings

    def _range_tick(self, frame: Frame, devices) -> None:
        bases = self.world.bases()
        base_pos = {b.serial: p.pos for b in bases
                    if (p := self.world.base_pose(b.serial)) is not None and p.valid}
        if not base_pos:
            return
        by_slot = {f"S-{b.channel}": b.serial for b in bases if b.channel is not None}
        for s in frame.samples:
            info = devices.get(s.serial)
            if (info is None or not s.connected
                    or info.device_class in (DeviceClass.BASE, DeviceClass.OTHER)):
                continue
            seen = None
            vis = self.world.visible.get(s.serial)
            if vis:
                seen = set()
                for slot, bid in vis.items():
                    serial = by_slot.get(slot)
                    if serial is None and bid:
                        serial = next((x for x in base_pos if x.upper().endswith(bid.upper())), None)
                    if serial:
                        seen.add(serial)
                seen = seen or None      # mapping failed: treat as unknown
            self.range.add(s, self.detectors.metrics(s.serial), base_pos, seen)

    def _on_incident(self, inc: Incident) -> dict:
        self.attributor.attribute(inc)
        self.heatmap.add_incident(inc)
        self.incidents.append(inc)
        if self.store:
            self.store.put_incident(inc)
            pre = [s for s in self.world.recent(inc.serial) if s.t >= inc.t_start - BURST_PRE_S]
            if pre:
                self.store.put_samples(pre, burst=True)
            self._bursts[inc.serial] = inc.t_end + BURST_POST_S
        return {"type": "incident", "incident": to_json(inc)}

    def _record(self, frame: Frame, devices) -> None:
        if not self.store:
            return
        if frame.t >= self._next_baseline:
            self._next_baseline = frame.t + 1.0 / BASELINE_HZ
            self.store.put_samples(frame.samples)
            sig = tuple((d.serial, d.connected, d.role, d.dongle, d.channel,
                         round(d.battery, 2) if d.battery is not None else None)
                        for d in devices.values())
            if sig != self._devices_sig:
                self._devices_sig = sig
                self.store.put_devices(devices)
        if self._bursts:
            live = {s: t for s, t in self._bursts.items() if t >= frame.t}
            self._bursts = live
            burst = [s for s in frame.samples if s.serial in live]
            if burst:
                self.store.put_samples(burst, burst=True)

    # ---- rolling window / controls -------------------------------------
    def _prune(self, now: float) -> None:
        cutoff = now - self.window_s
        while self.incidents and self.incidents[0].t_end < cutoff:
            self.incidents.popleft()
        while self.log and self.log[0].t < cutoff:
            self.log.popleft()
        if self.store:
            self.store.prune(cutoff)

    def set_recording(self, on: bool) -> None:
        with self.lock:
            if on == self.recording:
                return
            self.recording = on
            self.paused_at = None if on else (self.world.t or time.time())
            if self.store:
                self.store.flush()
        self._publish({"type": "session", "session": self.session_info()})

    def set_window(self, seconds: float) -> None:
        with self.lock:
            self.window_s = min(MAX_WINDOW_S, max(MIN_WINDOW_S, float(seconds)))
            if self.store:
                self.store.set_meta("window_s", str(self.window_s))
            if self.recording:
                self._prune(self.world.t or time.time())
        self._publish({"type": "session", "session": self.session_info()})

    def clear(self) -> None:
        """Forget everything recorded so far: incidents, log, heatmap, findings."""
        with self.lock:
            self.incidents.clear()
            self.log.clear()
            self.heatmap = Heatmap(self.heatmap.voxel)
            self.attributor = Attributor(self.world)
            self.range = RangeProfile()
            self._findings_cache = []
            self._bursts.clear()
            if self.paused_at is not None:
                self.paused_at = self.world.t or time.time()
            if self.store:
                self.store.clear()
                self.store.set_meta("findings", "[]")
        self._publish({"type": "cleared"})
        self._publish({"type": "session", "session": self.session_info()})

    # ---- views ---------------------------------------------------------
    def session_info(self) -> dict:
        return {"id": self.store.path.stem if self.store else "unrecorded", "mode": self.mode,
                "started": self.started, "steamvr": self.connected, "voxel": self.heatmap.voxel,
                "recording": self.recording, "paused_at": self.paused_at, "window_s": self.window_s,
                "play_area": self.play_area}

    def state_message(self, now: float) -> dict:
        open_by_serial = {i.serial: i.kind.value for i in self.detectors.open_incidents()}
        devs, bases = [], []
        for info in self.world.devices.values():
            s = self.world.last.get(info.serial)
            pos = list(s.pos) if s else [0, 0, 0]
            quat = list(s.quat) if s else [1, 0, 0, 0]
            if info.device_class is DeviceClass.BASE:
                bases.append({"serial": info.serial, "channel": info.channel, "pos": pos, "quat": quat,
                              "connected": bool(s and s.connected), "valid": bool(s and s.valid)})
                continue
            if info.device_class is DeviceClass.OTHER:
                continue
            m = self.detectors.metrics(info.serial)
            devs.append({
                "serial": info.serial, "class": info.device_class.value, "model": info.model,
                "role": info.role, "label": info.label(),
                "connected": bool(s and s.connected) and info.connected,
                "valid": bool(s and s.valid), "result": s.result if s else 0, "ok": bool(s and s.ok),
                "pos": pos, "quat": quat,
                "battery": info.battery, "charging": info.charging,
                "dongle": self.world.dongle_of.get(info.serial, info.dongle),
                "update_hz": m.update_hz, "jitter_mm": m.jitter_mm, "jitter_deg": m.jitter_deg,
                "ok_frac": m.ok_frac, "speed": m.speed,
                "bases_seen": self.world.visible_bases(info.serial),
                "incident": open_by_serial.get(info.serial),
            })
        return {"type": "state", "t": now, "steamvr": self.connected, "recording": self.recording,
                "devices": devs, "bases": bases}

    def heatmap_snapshot(self, metric: str, serial: str | None) -> dict:
        with self.lock:
            if metric == "coverage":
                poses = [(s.pos, s.quat) for b in self.world.bases()
                         if (s := self.world.base_pose(b.serial)) is not None and s.valid]
                return self.heatmap.coverage(poses)
            return self.heatmap.snapshot(metric, serial)

    def findings(self) -> list:
        with self.lock:
            return to_json(self._findings_cache)

    def range_snapshot(self) -> dict:
        with self.lock:
            return self.range.snapshot()
