"""Synthetic full-body rig with injectable tracking faults.

A person walks around a ~3x3 m playspace wearing an HMD, two controllers and
three trackers (waist, feet), watched by three base stations in upper corners.
Faults produce both the pose symptoms and the driver log events a real
SteamVR lighthouse driver would emit, via `drain_log()`.

Fault times are seconds since session start. `region` is a floor circle
`(x, z, radius)` in metres.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any

from ..heatmap import base_sees, quat_mul, quat_rotate, sight_blocked
from ..model import (DeviceClass, DeviceInfo, Frame, LogEvent, LogKind, Quat,
                     Sample, TrackingResult, Vec3)

FAULT_KINDS = ("occlusion", "radio", "base_shift", "reflection", "imu", "usb", "battery",
               "universe")

HMD = "LHR-A1B2C3D0"
LEFT_HAND = "LHR-C0111E01"
RIGHT_HAND = "LHR-C0111E02"
WAIST = "LHR-29C03CCA"
LEFT_FOOT = "LHR-5F2A91B3"
RIGHT_FOOT = "LHR-7D440E6F"
BASES = ("LHB-4F091D9A", "LHB-B3316995", "LHB-8E21C7F4")
DONGLE_WAIST = "40E4AC9C61"
DONGLE_FEET = "7B19D0F2A3"          # multi-device dongle shared by both feet

DEMO_PERIOD = 300.0

_OK = int(TrackingResult.RUNNING_OK)
_OUT_OF_RANGE = int(TrackingResult.RUNNING_OUT_OF_RANGE)
_UNINIT = int(TrackingResult.UNINITIALIZED)
_ZERO = (0.0, 0.0, 0.0)


@dataclass
class Fault:
    """One injected problem.

    kind      one of FAULT_KINDS
    t_start   seconds since session start
    t_end     seconds since session start (base_shift: when the driver notices
              and re-solves the base pose; universe: ignored)
    serial    target device (defaults per kind); for usb/universe unused
    base      target base station serial (base_shift)
    region    (x, z, radius): fault only acts while the device is inside
              (occlusion, reflection)
    params    occlusion: drift_s
              radio: variant "frozen"|"extrapolate", stale_s, disconnect (bool),
                     whole_dongle (bool), dongle (serial)
              base_shift: rot_deg, shift_m
              reflection: jump_m (lo, hi)
              imu: noise_m, noise_rad, period_s, burst_s
              battery: level (start), level_end
              universe: shift (x, y, z)
    """
    kind: str
    t_start: float
    t_end: float
    serial: str | None = None
    base: str | None = None
    region: tuple[float, float, float] | None = None
    params: dict[str, Any] = field(default_factory=dict)


class _Fx:
    """What the active faults do to one device on this frame."""
    __slots__ = ("mode", "off", "dvel", "npos", "nrot", "p", "variant")

    def __init__(self) -> None:
        self.mode = "ok"                 # "ok" | "lost" | "disc"
        self.off = _ZERO
        self.dvel = _ZERO
        self.npos = 0.0
        self.nrot = 0.0
        self.p = 1.0                     # probability a pose update arrives this frame
        self.variant = "frozen"


class _Dev:
    __slots__ = ("info", "serial", "idx", "last", "emitted", "vis", "true_vis", "optical",
                 "pending", "b0", "batt_fault", "batt_release")

    def __init__(self, info: DeviceInfo, idx: int) -> None:
        self.info = info
        self.serial = info.serial
        self.idx = idx
        self.last: Sample | None = None       # last real update
        self.emitted: Sample | None = None
        self.vis: list[str] = []              # slots announced through the log
        self.true_vis: list[str] = []         # slots with geometric line of sight
        self.optical = True
        self.pending: dict[str, int] = {}     # debounce counters for visibility flips
        self.b0 = info.battery or 1.0
        self.batt_fault = False
        self.batt_release = 0.0


class _Active:
    __slots__ = ("f", "st")

    def __init__(self, f: Fault) -> None:
        self.f = f
        self.st: dict[str, Any] = {}


def _ypr(yaw: float, pitch: float, roll: float) -> Quat:
    """Quaternion for yaw about +Y, then pitch about X, then roll about Z."""
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    w1, x1, y1, z1 = cy * cp, cy * sp, sy * cp, -sy * sp
    return (w1 * cr - z1 * sr, x1 * cr + y1 * sr, y1 * cr - x1 * sr, w1 * sr + z1 * cr)


def look_at(pos: Vec3, target: Vec3) -> Quat:
    """Orientation whose local -Z points from pos to target, +Y up."""
    fx, fy, fz = target[0] - pos[0], target[1] - pos[1], target[2] - pos[2]
    n = math.sqrt(fx * fx + fy * fy + fz * fz)
    zx, zy, zz = -fx / n, -fy / n, -fz / n            # local +Z
    xx, xy, xz = zz, 0.0, -zx                         # up(0,1,0) x z
    n = math.sqrt(xx * xx + xz * xz) or 1.0
    xx, xz = xx / n, xz / n
    yx, yy, yz = zy * xz - zz * xy, zz * xx - zx * xz, zx * xy - zy * xx   # z x x
    m00, m01, m02 = xx, yx, zx
    m10, m11, m12 = xy, yy, zy
    m20, m21, m22 = xz, yz, zz
    tr = m00 + m11 + m22
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return (0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s)
    if m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        return ((m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s)
    if m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        return ((m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s)
    s = math.sqrt(1.0 + m22 - m00 - m11) * 2
    return ((m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s)


def base_hex(serial: str) -> str:
    return serial[4:] if serial.startswith("LHB-") else serial


def demo_faults(cycle: int) -> list[Fault]:
    """The built-in looping scenario: one of everything every DEMO_PERIOD."""
    o = cycle * DEMO_PERIOD
    sign = 1.0 if cycle % 2 == 0 else -1.0
    return [
        Fault("occlusion", o + 20, o + 32, serial=LEFT_FOOT),
        Fault("radio", o + 60, o + 68, serial=LEFT_FOOT,
              params={"whole_dongle": True, "variant": "frozen" if cycle % 2 == 0 else "extrapolate"}),
        Fault("base_shift", o + 100, o + 103, base=BASES[cycle % 3],
              params={"rot_deg": 1.5 * sign, "shift_m": 0.03 * sign}),
        Fault("reflection", o + 125, o + 175, serial=RIGHT_FOOT, region=(0.7, -0.5, 0.8)),
        Fault("imu", o + 190, o + 200, serial=RIGHT_HAND),
        Fault("usb", o + 220, o + 221.5),
        Fault("battery", o + 240, o + 290, serial=WAIST),
    ]


class SimSource:
    """Simulated `Source`. See module docstring."""

    def __init__(self, rate: float = 200.0, realtime: bool = True, duration: float | None = None,
                 faults: str | list[Fault] | None = "demo", seed: int = 0,
                 t0: float | None = None, motion: str = "normal") -> None:
        self.rate = float(rate)
        self.dt = 1.0 / self.rate
        self.realtime = realtime
        self.t0 = time.time() if t0 is None else float(t0)
        self._n = 0
        self._n_max = None if duration is None else int(round(duration * self.rate))
        self._rng = random.Random(seed)
        self._closed = False
        self._wall0: float | None = None
        self._fast = motion == "fast"
        self._log: list[LogEvent] = []

        self._demo = faults == "demo"
        self._demo_cycle = 0
        if self._demo or not faults:
            flist: list[Fault] = []
        else:
            flist = list(faults)  # type: ignore[arg-type]
        for f in flist:
            if f.kind not in FAULT_KINDS:
                raise ValueError(f"unknown fault kind {f.kind!r}")
        self._pending = sorted(flist, key=lambda f: f.t_start)
        self._active: list[_Active] = []

        def dev(serial, cls, model, role, idx, battery, dongle):
            return DeviceInfo(serial=serial, device_class=cls, model=model, role=role, index=idx,
                              connected=True, battery=battery,
                              charging=False if battery is not None else None,
                              dongle=dongle, firmware="sim")
        infos = [
            dev(HMD, DeviceClass.HMD, "Index", "head", 0, None, None),
            dev(LEFT_HAND, DeviceClass.CONTROLLER, "Knuckles Left", "left_hand", 1, 0.86, HMD[4:] + "-LYM"),
            dev(RIGHT_HAND, DeviceClass.CONTROLLER, "Knuckles Right", "right_hand", 2, 0.91, HMD[4:] + "-RYB"),
            dev(WAIST, DeviceClass.TRACKER, "VIVE Tracker 3.0 MV", "waist", 3, 0.82, DONGLE_WAIST),
            dev(LEFT_FOOT, DeviceClass.TRACKER, "VIVE Tracker 3.0 MV", "left_foot", 4, 0.77, DONGLE_FEET),
            dev(RIGHT_FOOT, DeviceClass.TRACKER, "VIVE Tracker 3.0 MV", "right_foot", 5, 0.69, DONGLE_FEET),
        ]
        self._devs = [_Dev(info, i) for i, info in enumerate(infos)]
        self._by_serial = {d.serial: d for d in self._devs}
        self._infos: dict[str, DeviceInfo] = {i.serial: i for i in infos}

        base_pos = ((-1.9, 2.2, -1.9), (1.9, 2.2, -1.9), (-1.9, 2.2, 1.9))
        self._base_slot: dict[str, str] = {}
        self._base_true: dict[str, tuple[Vec3, Quat]] = {}       # used for line of sight
        self._base_reported: dict[str, tuple[Vec3, Quat]] = {}   # what the driver believes
        for i, (serial, pos) in enumerate(zip(BASES, base_pos)):
            self._infos[serial] = DeviceInfo(serial=serial, device_class=DeviceClass.BASE,
                                             model="Valve SR Imp", index=6 + i, connected=True,
                                             firmware="sim", channel=i + 1)
            self._base_slot[serial] = f"S-{i + 1}"
            pose = (pos, look_at(pos, (0.0, 0.9, 0.0)))
            self._base_true[serial] = pose
            self._base_reported[serial] = pose
        self._slot_base = {v: k for k, v in self._base_slot.items()}
        self._universe_off = _ZERO
        self._vis_every = max(1, int(round(self.rate / 20.0)))
        self._truth_prev = self._truth_cur = self._truth_next = None

    # ---- Source protocol -----------------------------------------------------
    def devices(self) -> dict[str, DeviceInfo]:
        return self._infos

    def drain_log(self) -> list[LogEvent]:
        out = self._log
        self._log = []
        return out

    def play_area(self) -> dict | None:
        r = 1.5
        corners = [[r, 0.0, -r], [-r, 0.0, -r], [-r, 0.0, r], [r, 0.0, r]]
        return {"rect": corners, "bounds": [[-1.8, 0.0, -1.7], [-1.7, 0.0, 1.8], [1.8, 0.0, 1.9], [1.9, 0.0, -1.6]],
                "height": 2.4}

    def close(self) -> None:
        self._closed = True

    def read(self) -> Frame | None:
        if self._closed or (self._n_max is not None and self._n >= self._n_max):
            raise StopIteration
        trel = self._n * self.dt
        if self.realtime:
            now = time.monotonic()
            if self._wall0 is None:
                self._wall0 = now
            wait = self._wall0 + trel - now
            if wait > 0:
                time.sleep(wait)
        frame = self._step(trel)
        self._n += 1
        return frame

    # ---- helpers for tests / callers ----------------------------------------
    def serial_for(self, role: str) -> str:
        for d in self._devs:
            if d.info.role == role:
                return d.serial
        raise KeyError(role)

    @property
    def base_serials(self) -> tuple[str, ...]:
        return BASES

    # ---- motion ----------------------------------------------------------------
    def _skeleton(self, t: float) -> list[tuple[Vec3, Quat]]:
        """True pose of each tracked device at session time t (device order)."""
        sin, cos = math.sin, math.cos
        if self._fast:
            k, hf, ha = 3.0, 10.0, 0.45       # body speed-up, hand frequency, hand amplitude
        else:
            k, hf, ha = 1.0, 1.0, 0.15
        tb = t * k
        rx = 0.9 * sin(0.23 * tb + 0.4) + 0.25 * sin(0.71 * tb)
        rz = 0.9 * sin(0.17 * tb + 1.3) + 0.25 * sin(0.53 * tb + 2.0)
        yaw = 1.4 * sin(0.13 * tb) + 0.6 * sin(0.41 * tb + 1.0)
        cy, sy = cos(yaw), sin(yaw)
        g = 1.9 * tb
        th = t * hf
        sg, cg = sin(g), cos(g)
        out = []
        # head
        lz = 0.02 * sin(0.8 * tb)
        out.append(((rx + sy * lz, 1.68 + 0.015 * sin(1.9 * tb), rz + cy * lz),
                    _ypr(yaw + 0.3 * sin(0.5 * tb), 0.2 * sin(0.37 * tb), 0.05 * sin(0.9 * tb))))
        # hands
        for side, ph in ((-1.0, 0.0), (1.0, 1.7)):
            lx = 0.3 * side
            ly = 1.05 + ha * sin(1.3 * th + ph)
            lz = -0.25 + ha * sin(1.7 * th + ph)
            out.append(((rx + cy * lx + sy * lz, ly, rz - sy * lx + cy * lz),
                        _ypr(yaw + 0.5 * sin(1.1 * th + ph), 0.6 * sin(1.3 * th), 0.4 * sin(0.9 * th + ph))))
        # waist
        lz = -0.08
        out.append(((rx + sy * lz, 1.0 + 0.012 * sin(1.9 * tb + 0.5), rz + cy * lz),
                    _ypr(yaw, 0.05 * sin(0.7 * tb), 0.03 * sin(1.9 * tb))))
        # feet (opposite phase)
        for side, s_, c_ in ((-1.0, sg, cg), (1.0, -sg, -cg)):
            lx = 0.13 * side
            ly = 0.10 + 0.04 * (1.0 + s_)
            lz = 0.18 * c_
            out.append(((rx + cy * lx + sy * lz, ly, rz - sy * lx + cy * lz),
                        _ypr(yaw + 0.1 * s_, 0.3 * c_, 0.0)))
        return out

    def _advance(self, trel: float) -> list[tuple[Vec3, Quat, Vec3, Vec3]]:
        dt = self.dt
        if self._truth_cur is None:
            self._truth_prev = self._skeleton(trel - dt)
            self._truth_cur = self._skeleton(trel)
        else:
            self._truth_prev = self._truth_cur
            self._truth_cur = self._truth_next
        self._truth_next = nxt = self._skeleton(trel + dt)
        inv = 0.5 / dt
        inv_w = 1.0 / dt
        out = []
        prev = self._truth_prev
        for i, (pos, quat) in enumerate(self._truth_cur):
            pp, pq = prev[i]
            np_, nq = nxt[i]
            vel = ((np_[0] - pp[0]) * inv, (np_[1] - pp[1]) * inv, (np_[2] - pp[2]) * inv)
            aw, ax, ay, az = nq
            bw, bx, by, bz = pq
            w = aw * bw + ax * bx + ay * by + az * bz
            s = inv_w if w >= 0 else -inv_w
            ang = ((-aw * bx + ax * bw - ay * bz + az * by) * s,
                   (-aw * by + ax * bz + ay * bw - az * bx) * s,
                   (-aw * bz - ax * by + ay * bx + az * bw) * s)
            out.append((pos, quat, vel, ang))
        return out

    # ---- log -----------------------------------------------------------------
    def _emit(self, t: float, kind: LogKind, serial: str | None, data: dict, raw: str,
              level: str = "Info") -> None:
        self._log.append(LogEvent(t=t, kind=kind, serial=serial, data=data, raw=raw, level=level))

    def _seeing(self, slots: list[str]) -> list[dict]:
        return [{"slot": s, "base": base_hex(self._slot_base[s])} for s in slots]

    def _sob(self, t: float, d: _Dev, slot: str, add: bool) -> None:
        if add:
            if slot not in d.vis:
                d.vis.append(slot)
                d.vis.sort()
        elif slot in d.vis:
            d.vis.remove(slot)
        verb = "Adding" if add else "Dropping"
        self._emit(t, LogKind.SOB_ADD if add else LogKind.SOB_DROP, d.serial,
                   {"base": slot, "seeing": self._seeing(d.vis)},
                   f"lighthouse: {d.serial} H: {verb} sync-on-beam base {slot} "
                   f"({base_hex(self._slot_base[slot])}), now seeing {len(d.vis)}")

    def _reacquire(self, t: float, d: _Dev) -> None:
        d.optical = True
        d.pending.clear()
        for slot in d.true_vis:
            self._sob(t, d, slot, True)

    def _update_visibility(self, t: float, truth) -> None:
        head, waist, lf, rf = truth[0][0], truth[3][0], truth[4][0], truth[5][0]
        capsules = [(head, waist, 0.17), (waist, lf, 0.09), (waist, rf, 0.09)]
        for d in self._devs:
            pos = truth[d.idx][0]
            now = [self._base_slot[b] for b, (bp, bq) in self._base_true.items()
                   if base_sees(bp, bq, pos) and not sight_blocked(pos, bp, capsules)]
            d.true_vis = now
            if not d.optical:
                continue
            pend = d.pending
            for slot in self._slot_base:
                want = slot in now
                if want == (slot in d.vis):
                    pend.pop(slot, None)
                    continue
                c = pend.get(slot, 0) + 1
                if c >= 3:                       # debounce: 3 ticks = 150 ms
                    pend.pop(slot, None)
                    self._sob(t, d, slot, want)
                else:
                    pend[slot] = c

    # ---- frame ---------------------------------------------------------------
    def _step(self, trel: float) -> Frame:
        t = self.t0 + trel
        rng = self._rng
        truth = self._advance(trel)

        if self._n == 0:
            self._emit(t, LogKind.SERVER_START, None, {"version": "sim"}, "VR server (sim) starting up")
            for d in self._devs:
                if d.info.dongle:
                    self._emit(t, LogKind.RECEIVER_CONNECTED, d.serial, {"dongle": d.info.dongle},
                               f"lighthouse: {d.serial}: Connected to receiver {d.info.dongle}")
        if self._n % self._vis_every == 0:
            self._update_visibility(t, truth)
        if self._n % 200 == 0:
            for d in self._devs:
                if d.batt_fault and d.batt_release and trel >= d.batt_release:
                    d.batt_fault = False              # swapped / put on charge
                if d.info.battery is not None and not d.batt_fault:
                    d.info.battery = max(0.2, d.b0 - trel / 18000.0)

        if self._demo and trel >= self._demo_cycle * DEMO_PERIOD - 1.0:
            self._pending.extend(demo_faults(self._demo_cycle))
            self._demo_cycle += 1
        pending = self._pending
        while pending and pending[0].t_start <= trel:
            self._active.append(_Active(pending.pop(0)))

        fx: dict[str, _Fx] = {}
        if self._active:
            keep = []
            for fa in self._active:
                ending = trel >= fa.f.t_end
                getattr(self, "_f_" + fa.f.kind)(fa, trel, t, truth, fx, ending)
                if not ending:
                    keep.append(fa)
            self._active = keep

        uo = self._universe_off
        shifted = uo != _ZERO
        gauss = rng.gauss
        samples: list[Sample] = []
        for d in self._devs:
            pos, quat, vel, ang = truth[d.idx]
            e = fx.get(d.serial)
            npos, nrot = 0.0003, 0.0003
            if e is not None:
                if e.mode != "ok":
                    last = d.emitted
                    lp = last.pos if last else pos
                    lq = last.quat if last else quat
                    if e.mode == "disc":
                        s = Sample(t, d.serial, False, False, _UNINIT, lp, lq)
                    else:
                        s = Sample(t, d.serial, True, False, _OUT_OF_RANGE, lp, lq)
                    d.emitted = s
                    d.last = None
                    samples.append(s)
                    continue
                if e.p < 1.0 and d.last is not None and rng.random() >= e.p:
                    last = d.last
                    if e.variant == "extrapolate":
                        h = t - last.t
                        lv = last.vel
                        lp = last.pos
                        s = Sample(t, d.serial, True, True, _OK,
                                   (lp[0] + lv[0] * h, lp[1] + lv[1] * h, lp[2] + lv[2] * h),
                                   last.quat, lv, last.angvel)
                    else:
                        s = Sample(t, d.serial, True, True, _OK, last.pos, last.quat, last.vel, last.angvel)
                    d.emitted = s
                    samples.append(s)
                    continue
                if e.npos:
                    npos += e.npos
                    nrot += e.nrot
                off = e.off
                if off != _ZERO:
                    pos = (pos[0] + off[0], pos[1] + off[1], pos[2] + off[2])
                dv = e.dvel
                if dv != _ZERO:
                    vel = (vel[0] + dv[0], vel[1] + dv[1], vel[2] + dv[2])
            if shifted:
                pos = (pos[0] + uo[0], pos[1] + uo[1], pos[2] + uo[2])
            pos = (pos[0] + gauss(0.0, npos), pos[1] + gauss(0.0, npos), pos[2] + gauss(0.0, npos))
            hx, hy, hz = gauss(0.0, nrot) * 0.5, gauss(0.0, nrot) * 0.5, gauss(0.0, nrot) * 0.5
            w, x, y, z = quat
            quat = (w - x * hx - y * hy - z * hz,
                    x + w * hx + y * hz - z * hy,
                    y + w * hy + z * hx - x * hz,
                    z + w * hz + x * hy - y * hx)
            nv = 0.002 + npos * 2.0
            vel = (vel[0] + gauss(0.0, nv), vel[1] + gauss(0.0, nv), vel[2] + gauss(0.0, nv))
            ang = (ang[0] + gauss(0.0, 0.005), ang[1] + gauss(0.0, 0.005), ang[2] + gauss(0.0, 0.005))
            s = Sample(t, d.serial, True, True, _OK, pos, quat, vel, ang)
            d.last = d.emitted = s
            samples.append(s)
        for serial, (bp, bq) in self._base_reported.items():
            if shifted:
                bp = (bp[0] + uo[0], bp[1] + uo[1], bp[2] + uo[2])
            samples.append(Sample(t, serial, True, True, _OK, bp, bq))
        return Frame(t, samples)

    # ---- faults ----------------------------------------------------------------
    def _fx(self, fx: dict[str, _Fx], serial: str) -> _Fx:
        e = fx.get(serial)
        if e is None:
            e = fx[serial] = _Fx()
        return e

    @staticmethod
    def _set_mode(e: _Fx, mode: str) -> None:
        if mode == "disc" or e.mode == "ok":
            e.mode = mode

    @staticmethod
    def _in_region(region, pos: Vec3) -> bool:
        if region is None:
            return True
        return (pos[0] - region[0]) ** 2 + (pos[2] - region[1]) ** 2 <= region[2] ** 2

    def _rand_dir(self) -> Vec3:
        g = self._rng.gauss
        while True:
            x, y, z = g(0, 1), g(0, 1) * 0.5, g(0, 1)
            n = math.sqrt(x * x + y * y + z * z)
            if n > 1e-3:
                return (x / n, y / n, z / n)

    def _f_occlusion(self, fa, trel, t, truth, fx, ending) -> None:
        f, st = fa.f, fa.st
        d = self._by_serial[f.serial or LEFT_FOOT]
        phase = st.get("phase", "idle")
        inside = not ending and self._in_region(f.region, truth[d.idx][0])
        if phase == "idle":
            if not inside:
                return
            phase = st["phase"] = "drift"
            st["t_in"] = trel
            b = self._rand_dir()
            st["bias"] = (b[0] * 0.5, b[1] * 0.5, b[2] * 0.5)
            st["drops"] = list(d.vis)
            st["next_drop"] = trel
            d.optical = False
        if not inside:
            st["phase"] = "idle"
            self._reacquire(t, d)
            return
        tau = trel - st["t_in"]
        if st["drops"] and trel >= st["next_drop"]:
            self._sob(t, d, st["drops"].pop(0), False)
            st["next_drop"] = trel + 0.05
        drift_s = f.params.get("drift_s", 0.4)
        if phase == "drift":
            if tau < drift_s:
                b = st["bias"]
                e = self._fx(fx, d.serial)
                h = 0.5 * tau * tau
                e.off = (e.off[0] + b[0] * h, e.off[1] + b[1] * h, e.off[2] + b[2] * h)
                e.dvel = (e.dvel[0] + b[0] * tau, e.dvel[1] + b[1] * tau, e.dvel[2] + b[2] * tau)
                return
            st["phase"] = "lost"
            while st["drops"]:
                self._sob(t, d, st["drops"].pop(0), False)
            ms = int(drift_s * 1000)
            self._emit(t, LogKind.TRACKING_RESET, d.serial,
                       {"reason": "no_optical", "ms": ms,
                        "text": f"Resetting tracking: no optical samples for {ms} ms"},
                       f"lighthouse: {d.serial} H: Resetting tracking: no optical samples for {ms} ms",
                       "Warning")
        self._set_mode(self._fx(fx, d.serial), "lost")

    def _f_radio(self, fa, trel, t, truth, fx, ending) -> None:
        f, st, p = fa.f, fa.st, fa.f.params
        serial = f.serial or LEFT_FOOT
        dongle = p.get("dongle") or self._infos[serial].dongle
        if p.get("whole_dongle") or p.get("dongle"):
            targets = [d for d in self._devs if d.info.dongle == dongle]
        else:
            targets = [self._by_serial[serial]]
        whole = len(targets) > 1 or bool(p.get("whole_dongle"))
        disconnect = p.get("disconnect", True)
        dur = f.t_end - f.t_start
        stale_s = p.get("stale_s", min(2.0, 0.5 * dur) if disconnect else min(1.0, 0.3 * dur))
        tau = trel - f.t_start
        if ending:
            if st.get("disc"):
                for d in targets:
                    self._emit(t, LogKind.RECEIVER_CONNECTED, d.serial, {"dongle": d.info.dongle},
                               f"lighthouse: {d.serial}: Connected to receiver {d.info.dongle}")
                    self._reacquire(t, d)
            return
        if disconnect and tau >= stale_s:
            if not st.get("disc"):
                st["disc"] = True
                for d in targets:
                    d.optical = False
                    d.vis = []
                    self._emit(t, LogKind.WIRELESS_DISCONNECTED, d.serial,
                               {"dongle": d.info.dongle, "device": d.serial},
                               f"lighthouse: {d.serial}: Lost wireless connection to receiver "
                               f"{d.info.dongle}", "Warning")
                if whole:
                    self._emit(t, LogKind.RECEIVER_DISCONNECTED, None, {"dongle": dongle},
                               f"lighthouse: Receiver {dongle} disconnected", "Warning")
            for d in targets:
                self._set_mode(self._fx(fx, d.serial), "disc")
            return
        frac = min(1.0, tau / stale_s) if stale_s > 0 else 1.0
        prob = 0.5 * (1.0 - frac) + 0.03
        for d in targets:
            e = self._fx(fx, d.serial)
            e.p = min(e.p, prob)
            e.variant = p.get("variant", "frozen")

    def _f_base_shift(self, fa, trel, t, truth, fx, ending) -> None:
        f, st, p = fa.f, fa.st, fa.f.params
        serial = f.base or BASES[0]
        slot = self._base_slot[serial]
        bp, bq = self._base_reported[serial]
        rot = math.radians(p.get("rot_deg", 1.5))
        shift = p.get("shift_m", 0.03)
        dq = (math.cos(rot * 0.5), 0.0, math.sin(rot * 0.5), 0.0)       # yaw about +Y
        if ending:
            self._base_reported[serial] = ((bp[0] + shift, bp[1], bp[2]), quat_mul(dq, bq))
            hexid = base_hex(serial)
            deg = abs(p.get("rot_deg", 1.5))
            self._emit(t, LogKind.TILT_UPDATE, None, {"base": hexid, "delta_deg": deg},
                       f"lighthouse: Base {hexid} updated tilt, changed by {deg:.2f} degrees", "Warning")
            self._emit(t, LogKind.BASE_CALIBRATED, None, {"base": hexid, "pitch": 31.0, "roll": 0.4},
                       f"lighthouse: Base {hexid} calibrated: pitch 31.00 roll 0.40")
            return
        # The base physically moved but the driver still believes the old pose:
        # everything solved through this base is displaced.
        for d in self._devs:
            if slot not in d.true_vis:
                continue
            pos = truth[d.idx][0]
            rel = (pos[0] - bp[0], pos[1] - bp[1], pos[2] - bp[2])
            r = quat_rotate(dq, rel)
            w = 1.0 / len(d.true_vis)
            e = self._fx(fx, d.serial)
            e.off = (e.off[0] + (r[0] - rel[0] + shift) * w,
                     e.off[1] + (r[1] - rel[1]) * w,
                     e.off[2] + (r[2] - rel[2]) * w)

    def _f_reflection(self, fa, trel, t, truth, fx, ending) -> None:
        f, st = fa.f, fa.st
        d = self._by_serial[f.serial or RIGHT_FOOT]
        region = f.region if f.region is not None else (0.7, -0.5, 0.8)
        if ending or not self._in_region(region, truth[d.idx][0]):
            st["off"] = None
            return
        rng = self._rng
        if trel >= st.get("next_log", 0.0):
            st["next_log"] = trel + 1.0
            data = {"rejected": rng.randint(60, 160), "backfacing": rng.randint(30, 90),
                    "non_clustered": rng.randint(5, 30)}
            self._emit(t, LogKind.BACKFACING_HITS, d.serial, data,
                       f"lighthouse: {d.serial} H: Rejected {data['rejected']} hits: "
                       f"{data['backfacing']} back-facing, {data['non_clustered']} non-clustered")
        off = st.get("off")
        if off is not None:
            if trel < st["until"]:
                e = self._fx(fx, d.serial)
                e.off = (e.off[0] + off[0], e.off[1] + off[1], e.off[2] + off[2])
            else:
                st["off"] = None
                st["next"] = trel + rng.uniform(0.3, 0.9)
        elif trel >= st.get("next", 0.0):
            lo, hi = f.params.get("jump_m", (0.02, 0.05))
            mag = rng.uniform(lo, hi)
            u = self._rand_dir()
            st["off"] = (u[0] * mag, u[1] * mag, u[2] * mag)
            st["until"] = trel + rng.uniform(0.1, 0.4)

    def _f_imu(self, fa, trel, t, truth, fx, ending) -> None:
        f, st, p = fa.f, fa.st, fa.f.params
        if ending:
            return
        d = self._by_serial[f.serial or RIGHT_HAND]
        tau = trel - f.t_start
        if tau % p.get("period_s", 1.2) >= p.get("burst_s", 0.6):
            return
        e = self._fx(fx, d.serial)
        e.npos = max(e.npos, p.get("noise_m", 0.0025))
        e.nrot = max(e.nrot, p.get("noise_rad", 0.01))
        if trel >= st.get("next_log", 0.0):
            st["next_log"] = trel + 0.1
            self._emit(t, LogKind.IMU_OFF_SCALE, d.serial, {},
                       f"lighthouse: {d.serial} H: IMU went off scale.", "Warning")

    def _f_usb(self, fa, trel, t, truth, fx, ending) -> None:
        st = fa.st
        if not st.get("started"):
            st["started"] = True
            for iface in ("IMU", "Optical", "VrController"):
                self._emit(t, LogKind.HID_ERROR, None, {"interface": iface},
                           f"lighthouse: HID read error on {iface} interface: device not connected", "Error")
            for d in self._devs:
                d.optical = False
                d.vis = []
        if ending:
            for d in self._devs:
                if d.info.dongle:
                    self._emit(t, LogKind.RECEIVER_CONNECTED, d.serial, {"dongle": d.info.dongle},
                               f"lighthouse: {d.serial}: Connected to receiver {d.info.dongle}")
                self._reacquire(t, d)
            return
        for d in self._devs:
            self._set_mode(self._fx(fx, d.serial), "disc")

    def _f_battery(self, fa, trel, t, truth, fx, ending) -> None:
        f, st, p = fa.f, fa.st, fa.f.params
        d = self._by_serial[f.serial or WAIST]
        if ending:
            d.batt_release = trel + 10.0          # stays flat for a while after the fault
            if st.get("down"):
                self._reacquire(t, d)
            return
        rng = self._rng
        frac = (trel - f.t_start) / max(1e-6, f.t_end - f.t_start)
        lvl0 = p.get("level", 0.08)
        d.batt_fault = True
        d.batt_release = 0.0
        d.info.battery = round(lvl0 + (p.get("level_end", 0.01) - lvl0) * frac, 4)
        if "next" not in st:
            st["next"] = trel + rng.uniform(1.0, 2.0)
        if st.get("down"):
            if trel >= st["until"]:
                st["down"] = False
                st["next"] = trel + (6.0 - 4.5 * frac) * rng.uniform(0.7, 1.3)
                self._reacquire(t, d)
                return
            self._set_mode(self._fx(fx, d.serial), "disc")
        elif trel >= st["next"]:
            st["down"] = True
            st["until"] = trel + rng.uniform(0.3, 0.8)
            d.optical = False
            d.vis = []
            if rng.random() < 0.5:
                self._emit(t, LogKind.WIRELESS_DISCONNECTED, d.serial,
                           {"dongle": d.info.dongle, "device": d.serial},
                           f"lighthouse: {d.serial}: Lost wireless connection to receiver "
                           f"{d.info.dongle}", "Warning")
            self._set_mode(self._fx(fx, d.serial), "disc")

    def _f_universe(self, fa, trel, t, truth, fx, ending) -> None:
        st = fa.st
        if st.get("done"):
            return
        st["done"] = True
        sh = fa.f.params.get("shift", (0.06, 0.0, 0.04))
        uo = self._universe_off
        self._universe_off = (uo[0] + sh[0], uo[1] + sh[1], uo[2] + sh[2])
        self._emit(t, LogKind.UNIVERSE_CHANGED, None, {"universe": "sim-universe-2"},
                   "lighthouse: Universe changed to sim-universe-2", "Warning")
