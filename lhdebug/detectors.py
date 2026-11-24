"""Per-frame anomaly detectors.

`DetectorBank.feed` runs once per frame for every device, so the hot path is
plain scalar Python with O(1) state per device (exponential averages rather
than windows). Incidents are episodes with hysteresis: a condition has to hold
for `open_s` of accumulated time to open one and be absent for `close_s` to
close it, so one physical event is reported once, when it is over.

What counts as a pose *update*: OpenVR may extrapolate the last driver update
to "now", so a stalled radio link can still yield slightly different positions
each frame. A sample is a new update only if its velocity or angular velocity
changed, or its pose changed in a way constant-velocity extrapolation does not
explain. Exactly repeated poses are the degenerate case.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from .model import (DeviceClass, DeviceMetrics, Frame, Incident, IncidentKind,
                    Sample)
from .world import World

UNIVERSE_SERIAL = "universe"     # Incident.serial used for UNIVERSE_SHIFT


@dataclass
class DetectorConfig:
    # dropout: connected but pose invalid / not Running_OK
    dropout_open_s: float = 0.03
    dropout_close_s: float = 1.0
    # disconnect: device reports not connected
    disconnect_open_s: float = 0.03
    disconnect_close_s: float = 1.0
    # jump: position step vs. what the reported velocity predicts
    jump_m: float = 0.015            # residual that counts as a jump
    jump_rel: float = 0.25           # + this fraction of the step length
    jump_max_gap_s: float = 0.05     # do not judge steps across longer update gaps
    jump_close_s: float = 1.5        # jumps closer together than this are one incident
    # jitter: high-frequency residual RMS (3D), robust to smooth fast motion
    jitter_mm: float = 2.0
    jitter_deg: float = 0.5
    jitter_tau_s: float = 0.5        # averaging time constant
    jitter_release: float = 0.6      # stays open until below release * threshold
    jitter_open_s: float = 0.1
    jitter_close_s: float = 1.5
    # stale: distinct pose updates per second collapsed while tracking is "ok"
    rate_window_s: float = 1.0
    stale_min_hz: float = 60.0       # absolute floor ...
    stale_ratio: float = 0.5         # ... or this fraction of the sampling rate, whichever is lower
    stale_gap_s: float = 0.25        # no update at all for this long
    stale_open_s: float = 0.2
    stale_close_s: float = 1.0
    # base shift: a base station's reported pose moves
    base_shift_m: float = 0.005
    base_shift_deg: float = 0.25
    base_settle_s: float = 1.0       # pose must hold still this long to close
    # universe shift: every base (or, with no bases known, every device) moves at once
    universe_window_s: float = 0.5
    universe_min_devices: int = 3
    universe_frac: float = 0.75
    ok_tau_s: float = 2.0            # averaging for DeviceMetrics.ok_frac


class _Ep:
    """One hysteresis episode tracker (a potential / open incident)."""
    __slots__ = ("kind", "open_s", "close_s", "start", "last", "is_open", "pos",
                 "n", "bad_s", "peak", "peak2", "id")

    def __init__(self, kind: IncidentKind, open_s: float, close_s: float) -> None:
        self.kind = kind
        self.open_s = open_s
        self.close_s = close_s
        self.start: float | None = None
        self.last = 0.0
        self.is_open = False
        self.pos = (0.0, 0.0, 0.0)
        self.n = 0
        self.bad_s = 0.0
        self.peak = 0.0
        self.peak2 = 0.0
        self.id = 0


class _Dev:
    __slots__ = ("serial", "m", "seen_connected", "prev", "last_fresh", "last_ok_pos",
                 "ok_since", "fresh_t", "chain", "v1", "v2", "w1", "w2", "jm2", "jr2",
                 "disc", "drop", "jump", "jitter", "stale", "eps", "busy")

    def __init__(self, serial: str, cfg: DetectorConfig) -> None:
        self.serial = serial
        self.m = DeviceMetrics()
        self.seen_connected = False
        self.prev: Sample | None = None          # previous sample if it was ok
        self.last_fresh: Sample | None = None    # last sample that was a real update
        self.last_ok_pos = None
        self.ok_since = 0.0
        self.fresh_t: deque[float] = deque()
        self.chain = 0                           # consecutive evenly spaced updates
        self.v1 = self.v2 = self.w1 = self.w2 = (0.0, 0.0, 0.0)
        self.jm2 = 0.0                           # mean square jitter, m^2
        self.jr2 = 0.0                           # mean square rotational jitter, rad^2
        K = IncidentKind
        self.disc = _Ep(K.DISCONNECT, cfg.disconnect_open_s, cfg.disconnect_close_s)
        self.drop = _Ep(K.DROPOUT, cfg.dropout_open_s, cfg.dropout_close_s)
        self.jump = _Ep(K.JUMP, 0.0, cfg.jump_close_s)
        self.jitter = _Ep(K.JITTER, cfg.jitter_open_s, cfg.jitter_close_s)
        self.stale = _Ep(K.STALE, cfg.stale_open_s, cfg.stale_close_s)
        self.eps = (self.disc, self.drop, self.jump, self.jitter, self.stale)
        self.busy = False                        # any episode has start set


class _Base:
    __slots__ = ("serial", "ref_pos", "ref_quat", "cand_pos", "cand_quat",
                 "move_start", "last_change", "id")

    def __init__(self, serial: str) -> None:
        self.serial = serial
        self.ref_pos = None
        self.ref_quat = None
        self.cand_pos = None
        self.cand_quat = None
        self.move_start: float | None = None
        self.last_change = 0.0
        self.id = 0


def _quat_angle_deg(a, b) -> float:
    d = abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3])
    return math.degrees(2.0 * math.acos(d)) if d < 1.0 else 0.0


def _dist(a, b) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


class DetectorBank:
    def __init__(self, world: World, config: DetectorConfig | None = None) -> None:
        self.world = world
        self.cfg = config or DetectorConfig()
        self._devs: dict[str, _Dev] = {}
        self._bases: dict[str, _Base] = {}
        self._next_id = 1
        self._last_t: float | None = None
        self._dt = 0.0                 # smoothed frame interval
        self._bases_moving = 0
        self._recent_jumps: list[tuple[float, str, tuple[float, float, float]]] = []
        self._suppress_jumps_until = 0.0

    # ---- public --------------------------------------------------------------
    def feed(self, frame: Frame) -> list[Incident]:
        t = frame.t
        closed: list[Incident] = []
        if self._last_t is not None:
            d = t - self._last_t
            if 0.0 < d < 1.0:
                self._dt = d if self._dt == 0.0 else self._dt + 0.05 * (d - self._dt)
        self._last_t = t
        devs = self._devs
        bases = self._bases
        feed_dev = self._feed_dev
        for s in frame.samples:
            st = devs.get(s.serial)
            if st is None:
                b = bases.get(s.serial)
                if b is None:
                    info = self.world.devices.get(s.serial)
                    if info is not None and info.device_class is DeviceClass.BASE:
                        b = bases[s.serial] = _Base(s.serial)
                    else:
                        st = devs[s.serial] = _Dev(s.serial, self.cfg)
                if b is not None:
                    if s.pos != b.ref_pos or s.quat != b.ref_quat or b.move_start is not None:
                        self._feed_base(b, s, t)
                    continue
            feed_dev(st, s, t, closed)
        if self._bases_moving:
            self._check_bases(t, closed)
        if self._recent_jumps:
            self._check_universe_jumps(t, closed)
        return closed

    def metrics(self, serial: str) -> DeviceMetrics:
        """Rolling metrics. The returned object is live (updated in place)."""
        st = self._devs.get(serial)
        return st.m if st is not None else DeviceMetrics()

    def open_incidents(self) -> list[Incident]:
        out = []
        for st in self._devs.values():
            if st.busy:
                for ep in st.eps:
                    if ep.start is not None and ep.is_open:
                        out.append(self._incident(st.serial, ep))
        for b in self._bases.values():
            if b.move_start is not None:
                out.append(self._base_incident(b, IncidentKind.BASE_SHIFT))
        return out

    def flush(self) -> list[Incident]:
        out: list[Incident] = []
        for st in self._devs.values():
            for ep in st.eps:
                if ep.start is not None and ep.is_open:
                    out.append(self._incident(st.serial, ep))
                ep.start = None
                ep.is_open = False
            st.busy = False
        if self._bases_moving:
            self._check_bases(float("inf"), out)
        return out

    # ---- tracked devices -----------------------------------------------------
    def _feed_dev(self, st: _Dev, s: Sample, t: float, closed: list[Incident]) -> None:
        cfg = self.cfg
        m = st.m
        connected = s.connected
        ok = connected and s.valid and s.result == 200
        dt = self._dt
        m.ok_frac += min(1.0, dt / cfg.ok_tau_s) * ((1.0 if ok else 0.0) - m.ok_frac)
        if connected:
            st.seen_connected = True
        stale = False
        jump = 0.0
        jitter = False
        if not ok:
            st.prev = None
            st.chain = 0
            m.speed = 0.0
            ft = st.fresh_t
            if ft:
                cut = t - cfg.rate_window_s
                while ft and ft[0] < cut:
                    ft.popleft()
                m.update_hz = len(ft) / cfg.rate_window_s
            else:
                m.update_hz = 0.0
        else:
            prev = st.prev
            st.prev = s
            st.last_ok_pos = s.pos
            vel = s.vel
            if prev is None:
                # first good sample of a streak: start afresh
                fresh = True
                st.ok_since = t
                st.fresh_t.clear()
                st.last_fresh = None
            elif vel != prev.vel or s.angvel != prev.angvel:
                fresh = True
            elif s.pos == prev.pos:
                fresh = s.quat != prev.quat and s.angvel == (0.0, 0.0, 0.0)
            else:
                # same velocity, different position: extrapolated or new?
                h = t - prev.t
                p = s.pos
                q = prev.pos
                ex = p[0] - q[0] - vel[0] * h
                ey = p[1] - q[1] - vel[1] * h
                ez = p[2] - q[2] - vel[2] * h
                fresh = ex * ex + ey * ey + ez * ez > 2.5e-11      # (5 um)^2
            ft = st.fresh_t
            if fresh:
                ft.append(t)
                pf = st.last_fresh
                st.last_fresh = s
                m.speed = math.sqrt(vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2])
                if pf is None:
                    st.chain = 0
                else:
                    dtf = t - pf.t
                    if dtf <= 0.0 or dtf > cfg.jump_max_gap_s:
                        st.chain = 0
                    else:
                        p = s.pos
                        q = pf.pos
                        dx = p[0] - q[0]
                        dy = p[1] - q[1]
                        dz = p[2] - q[2]
                        pv = pf.vel
                        h = 0.5 * dtf
                        rx = dx - (vel[0] + pv[0]) * h
                        ry = dy - (vel[1] + pv[1]) * h
                        rz = dz - (vel[2] + pv[2]) * h
                        r2 = rx * rx + ry * ry + rz * rz
                        if r2 > cfg.jump_m * cfg.jump_m:
                            thr = cfg.jump_m + cfg.jump_rel * math.sqrt(dx * dx + dy * dy + dz * dz)
                            if r2 > thr * thr:
                                jump = math.sqrt(r2)
                        if jump:
                            st.chain = 0
                            if t < self._suppress_jumps_until:
                                jump = 0.0
                            else:
                                self._recent_jumps.append((t, st.serial, (rx, ry, rz)))
                        elif dtf > 1.6 * dt:
                            st.chain = 0          # missed updates: residual would alias motion
                        else:
                            inv = 1.0 / dtf
                            v0 = (dx * inv, dy * inv, dz * inv)
                            a = s.quat
                            b = pf.quat
                            qw = b[0] * a[0] + b[1] * a[1] + b[2] * a[2] + b[3] * a[3]
                            k = 2.0 * inv if qw >= 0.0 else -2.0 * inv
                            w0 = ((b[0] * a[1] - b[1] * a[0] - b[2] * a[3] + b[3] * a[2]) * k,
                                  (b[0] * a[2] + b[1] * a[3] - b[2] * a[0] - b[3] * a[1]) * k,
                                  (b[0] * a[3] - b[1] * a[2] + b[2] * a[1] - b[3] * a[0]) * k)
                            if st.chain >= 2:
                                # second difference of pose-derived rates = third
                                # difference of pose: kills smooth motion, keeps noise
                                v1 = st.v1
                                v2 = st.v2
                                ex = v0[0] - 2.0 * v1[0] + v2[0]
                                ey = v0[1] - 2.0 * v1[1] + v2[1]
                                ez = v0[2] - 2.0 * v1[2] + v2[2]
                                g = dtf * dtf * 0.05          # /20: noise gain of a third difference
                                al = dtf / cfg.jitter_tau_s
                                st.jm2 += al * ((ex * ex + ey * ey + ez * ez) * g - st.jm2)
                                w1 = st.w1
                                w2 = st.w2
                                ex = w0[0] - 2.0 * w1[0] + w2[0]
                                ey = w0[1] - 2.0 * w1[1] + w2[1]
                                ez = w0[2] - 2.0 * w1[2] + w2[2]
                                st.jr2 += al * ((ex * ex + ey * ey + ez * ez) * g - st.jr2)
                                m.jitter_mm = math.sqrt(st.jm2) * 1000.0
                                m.jitter_deg = math.sqrt(st.jr2) * 57.29577951308232
                            st.v2 = st.v1
                            st.v1 = v0
                            st.w2 = st.w1
                            st.w1 = w0
                            st.chain += 1
            cut = t - cfg.rate_window_s
            while ft and ft[0] < cut:
                ft.popleft()
            age = t - st.ok_since
            if age >= cfg.rate_window_s:
                hz = m.update_hz = len(ft) / cfg.rate_window_s
                limit = cfg.stale_ratio / dt if dt > 0.0 else cfg.stale_min_hz
                if limit > cfg.stale_min_hz:
                    limit = cfg.stale_min_hz
                stale = hz < limit
            else:
                m.update_hz = len(ft) / age if age > 0.1 else (1.0 / dt if dt > 0.0 else 0.0)
            lf = st.last_fresh
            if not stale and lf is not None and t - lf.t > cfg.stale_gap_s:
                stale = True
            jm = m.jitter_mm
            jd = m.jitter_deg
            if jm > cfg.jitter_mm * cfg.jitter_release or jd > cfg.jitter_deg * cfg.jitter_release:
                if st.jitter.is_open:
                    jitter = True
                else:
                    jitter = jm > cfg.jitter_mm or jd > cfg.jitter_deg

        if not st.seen_connected:
            return
        disc = not connected
        lost = connected and not ok
        if not (disc or lost or stale or jump or jitter or st.busy):
            return                                   # healthy fast path
        ep_update = self._ep
        busy = ep_update(st, st.disc, disc, t, s, 0.0, 0.0, closed)
        busy |= ep_update(st, st.drop, lost, t, s, 0.0, 0.0, closed)
        busy |= ep_update(st, st.jump, jump > 0.0, t, s, jump, 0.0, closed)
        busy |= ep_update(st, st.jitter, jitter, t, s, m.jitter_mm, m.jitter_deg, closed)
        busy |= ep_update(st, st.stale, stale, t, s, -m.update_hz, 0.0, closed)
        st.busy = busy

    def _ep(self, st: _Dev, ep: _Ep, cond: bool, t: float, s: Sample,
            v1: float, v2: float, closed: list[Incident]) -> bool:
        if cond:
            if ep.start is None:
                ep.start = t
                ep.pos = st.last_ok_pos or s.pos
                ep.n = 0
                ep.bad_s = 0.0
                ep.peak = v1
                ep.peak2 = v2
                ep.is_open = False
            else:
                if v1 > ep.peak:
                    ep.peak = v1
                if v2 > ep.peak2:
                    ep.peak2 = v2
            ep.last = t
            ep.n += 1
            ep.bad_s += self._dt
            if not ep.is_open and ep.bad_s >= ep.open_s:
                ep.is_open = True
                ep.id = self._next_id
                self._next_id += 1
            return True
        if ep.start is None:
            return False
        if t - ep.last >= ep.close_s:
            if ep.is_open:
                closed.append(self._incident(st.serial, ep))
            ep.start = None
            ep.is_open = False
            return False
        return True

    def _incident(self, serial: str, ep: _Ep) -> Incident:
        K = IncidentKind
        kind = ep.kind
        dur = ep.last - ep.start
        cfg = self.cfg
        rate = 1.0 / self._dt if self._dt > 0 else 0.0
        if kind is K.JUMP:
            metrics = {"jump_m": round(ep.peak, 4), "count": float(ep.n)}
            sev = min(1.0, ep.peak / 0.10)
        elif kind is K.JITTER:
            metrics = {"jitter_mm": round(ep.peak, 3), "jitter_deg": round(ep.peak2, 4)}
            sev = min(1.0, max(ep.peak / cfg.jitter_mm, ep.peak2 / cfg.jitter_deg) / 5.0)
        elif kind is K.STALE:
            hz = -ep.peak
            metrics = {"update_hz": round(hz, 1), "expected_hz": round(rate, 1)}
            sev = min(1.0, max(0.0, 1.0 - hz / rate)) if rate > 0 else 1.0
        else:
            metrics = {"lost_s": round(ep.bad_s, 3)}
            sev = min(1.0, 0.3 + 0.7 * dur / 5.0)
        return Incident(id=ep.id, kind=kind, serial=serial, t_start=ep.start, t_end=ep.last,
                        pos=tuple(ep.pos), severity=round(sev, 3), metrics=metrics)

    # ---- base stations -------------------------------------------------------
    def _feed_base(self, b: _Base, s: Sample, t: float) -> None:
        if not (s.connected and s.valid):
            return
        if b.ref_pos is None:
            b.ref_pos = b.cand_pos = s.pos
            b.ref_quat = b.cand_quat = s.quat
            return
        cfg = self.cfg
        if b.move_start is None:
            if (_dist(s.pos, b.ref_pos) < cfg.base_shift_m
                    and _quat_angle_deg(s.quat, b.ref_quat) < cfg.base_shift_deg):
                return
            b.move_start = t
            b.last_change = t
            b.cand_pos = s.pos
            b.cand_quat = s.quat
            b.id = self._next_id
            self._next_id += 1
            self._bases_moving += 1
        elif (_dist(s.pos, b.cand_pos) > 0.2 * cfg.base_shift_m
              or _quat_angle_deg(s.quat, b.cand_quat) > 0.2 * cfg.base_shift_deg):
            b.last_change = t
            b.cand_pos = s.pos
            b.cand_quat = s.quat

    def _base_incident(self, b: _Base, kind: IncidentKind) -> Incident:
        shift = _dist(b.cand_pos, b.ref_pos)
        ang = _quat_angle_deg(b.cand_quat, b.ref_quat)
        return Incident(id=b.id, kind=kind, serial=b.serial, t_start=b.move_start,
                        t_end=b.last_change, pos=tuple(b.ref_pos),
                        severity=round(min(1.0, max(shift / 0.05, ang / 2.0)), 3),
                        metrics={"shift_m": round(shift, 4), "shift_deg": round(ang, 3)})

    def _check_bases(self, t: float, closed: list[Incident]) -> None:
        cfg = self.cfg
        moving = [b for b in self._bases.values() if b.move_start is not None]
        if any(t - b.last_change < cfg.base_settle_s for b in moving):
            return
        known = [b for b in self._bases.values() if b.ref_pos is not None]
        starts = [b.move_start for b in moving]
        if (len(known) >= 2 and len(moving) == len(known)
                and max(starts) - min(starts) <= cfg.universe_window_s):
            t0 = min(starts)
            t1 = max(b.last_change for b in moving)
            shift = max(_dist(b.cand_pos, b.ref_pos) for b in moving)
            ang = max(_quat_angle_deg(b.cand_quat, b.ref_quat) for b in moving)
            closed.append(Incident(
                id=moving[0].id, kind=IncidentKind.UNIVERSE_SHIFT, serial=UNIVERSE_SERIAL,
                t_start=t0, t_end=t1, pos=(0.0, 0.0, 0.0), severity=1.0,
                metrics={"shift_m": round(shift, 4), "shift_deg": round(ang, 3),
                         "bases": float(len(moving))}))
            self._absorb_jumps(t0 - cfg.universe_window_s, t1 + cfg.universe_window_s)
        else:
            for b in moving:
                closed.append(self._base_incident(b, IncidentKind.BASE_SHIFT))
        for b in moving:
            b.ref_pos = b.cand_pos
            b.ref_quat = b.cand_quat
            b.move_start = None
        self._bases_moving = 0

    # ---- universe shift ------------------------------------------------------
    def _absorb_jumps(self, t0: float, t1: float) -> int:
        """Drop per-device jump episodes that are just the universe moving."""
        n = 0
        for st in self._devs.values():
            ep = st.jump
            if ep.start is not None and ep.start >= t0 and ep.last <= t1:
                ep.start = None
                ep.is_open = False
                n += 1
        return n

    def _check_universe_jumps(self, t: float, closed: list[Incident]) -> None:
        """Fallback when no base poses are available: every device jumping by
        the same vector within a moment is the universe moving, not N jumps."""
        cfg = self.cfg
        jumps = self._recent_jumps
        if t - jumps[0][0] < 0.1:
            return
        self._recent_jumps = []
        if self._bases:
            return                       # base poses decide (see _check_bases)
        by_dev: dict[str, tuple[float, float, float]] = {}
        for _, serial, vec in jumps:
            by_dev.setdefault(serial, vec)
        n_ok = sum(1 for st in self._devs.values() if st.prev is not None)
        need = max(cfg.universe_min_devices, math.ceil(cfg.universe_frac * n_ok))
        if len(by_dev) < need:
            return
        vecs = list(by_dev.values())
        mean = tuple(sum(v[i] for v in vecs) / len(vecs) for i in range(3))
        norm = math.sqrt(mean[0] ** 2 + mean[1] ** 2 + mean[2] ** 2)
        if norm <= 0.0 or any(_dist(v, mean) > 0.35 * norm for v in vecs):
            return
        t0 = jumps[0][0]
        t1 = jumps[-1][0]
        self._absorb_jumps(t0 - 0.05, t1 + 0.05)
        self._suppress_jumps_until = t1 + cfg.universe_window_s
        closed.append(Incident(
            id=self._next_id, kind=IncidentKind.UNIVERSE_SHIFT, serial=UNIVERSE_SERIAL,
            t_start=t0, t_end=t1, pos=(0.0, 0.0, 0.0), severity=1.0,
            metrics={"shift_m": round(norm, 4), "devices": float(len(by_dev))}))
        self._next_id += 1
