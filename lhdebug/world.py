"""Shared rolling context: what the detectors and attribution engine look at.

The engine (sampler.py) owns one World and feeds it every frame and every log
event. Detectors/attribution read from it; they never talk to a Source.
"""
from __future__ import annotations

from collections import deque
from typing import Iterable

from .model import DeviceClass, DeviceInfo, Frame, LogEvent, LogKind, Sample

SAMPLE_WINDOW_S = 6.0      # full-rate pose history kept per device
LOG_WINDOW_S = 900.0       # log event history kept


class World:
    def __init__(self) -> None:
        self.t: float = 0.0
        self.devices: dict[str, DeviceInfo] = {}
        self.samples: dict[str, deque[Sample]] = {}
        self.last: dict[str, Sample] = {}
        self.log: deque[LogEvent] = deque()
        # serial -> {slot "S-n": base id or None}; maintained from SOB add/drop lines
        self.visible: dict[str, dict[str, str | None]] = {}
        # device serial -> dongle serial, from "Connected to receiver" lines / properties
        self.dongle_of: dict[str, str] = {}

    # ---- feeding -------------------------------------------------------
    def set_devices(self, devices: dict[str, DeviceInfo]) -> None:
        self.devices = devices
        for serial, info in devices.items():
            if info.dongle:
                self.dongle_of[serial] = info.dongle

    def add_frame(self, frame: Frame) -> None:
        self.t = frame.t
        cutoff = frame.t - SAMPLE_WINDOW_S
        for s in frame.samples:
            buf = self.samples.get(s.serial)
            if buf is None:
                buf = self.samples[s.serial] = deque()
            buf.append(s)
            while buf and buf[0].t < cutoff:
                buf.popleft()
            self.last[s.serial] = s

    def add_log(self, ev: LogEvent) -> None:
        self.log.append(ev)
        cutoff = ev.t - LOG_WINDOW_S
        while self.log and self.log[0].t < cutoff:
            self.log.popleft()
        if ev.serial is None:
            return
        if ev.kind in (LogKind.SOB_ADD, LogKind.SOB_DROP):
            seen = {e["slot"]: e.get("base") for e in ev.data.get("seeing", [])}
            if ev.kind is LogKind.SOB_ADD:
                seen.setdefault(ev.data["base"], None)
            else:
                seen.pop(ev.data["base"], None)
            self.visible[ev.serial] = seen
        elif ev.kind is LogKind.RECEIVER_CONNECTED:
            self.dongle_of[ev.serial] = ev.data["dongle"]
        elif ev.kind is LogKind.TRACKING_RESET:
            self.visible[ev.serial] = {}

    # ---- queries -------------------------------------------------------
    def bases(self) -> list[DeviceInfo]:
        return [d for d in self.devices.values() if d.device_class is DeviceClass.BASE]

    def tracked(self) -> list[DeviceInfo]:
        """Everything that moves: HMD, controllers, trackers."""
        return [d for d in self.devices.values()
                if d.device_class in (DeviceClass.HMD, DeviceClass.CONTROLLER, DeviceClass.TRACKER)]

    def base_pose(self, serial: str) -> Sample | None:
        return self.last.get(serial)

    def recent(self, serial: str, seconds: float | None = None) -> list[Sample]:
        buf = self.samples.get(serial)
        if not buf:
            return []
        if seconds is None:
            return list(buf)
        cutoff = self.t - seconds
        return [s for s in buf if s.t >= cutoff]

    def log_between(self, t0: float, t1: float, serial: str | None = None,
                    kinds: Iterable[LogKind] | None = None) -> list[LogEvent]:
        kindset = set(kinds) if kinds is not None else None
        return [e for e in self.log
                if t0 <= e.t <= t1
                and (serial is None or e.serial == serial)
                and (kindset is None or e.kind in kindset)]

    def visible_bases(self, serial: str) -> list[str]:
        """Slots ("S-n") the driver last said this device was seeing."""
        return sorted(self.visible.get(serial, {}))

    def devices_on_dongle(self, dongle: str) -> list[str]:
        return [s for s, d in self.dongle_of.items() if d == dongle]
