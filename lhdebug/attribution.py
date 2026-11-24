"""Rule-based root-cause attribution.

`attribute(inc)` gathers evidence around one incident (driver log lines, what
other devices were doing, dongle sharing, base geometry and body occlusion,
battery, recurrence in the same spot) and turns it into per-cause scores.
Scores are additive evidence weights; confidences are the scores normalised
together with an "unknown" share that wins when nothing reaches ~0.3.

`findings()` groups everything attributed so far into a ranked, plain-language
session summary with one concrete fix per finding.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

from .heatmap import FOV_H_DEG, FOV_V_DEG, RANGE_M, base_view, sight_blocked
from .model import (Cause, DeviceClass, Finding, Incident, IncidentKind, LogKind,
                    Vec3)
from .world import World

C = Cause
K = IncidentKind

UNKNOWN_FLOOR = 0.3            # best score below this -> "unknown" wins
LOW_BATTERY = 0.15
SAME_SPOT_M = 0.5
MAX_HISTORY = 5000

_DIGITS = re.compile(r"\d+(\.\d+)?")
_CONNECTIVITY = (K.DISCONNECT, K.DROPOUT, K.STALE)
_OPTICAL = (K.DROPOUT, K.JUMP, K.JITTER)


class _Rec:
    __slots__ = ("inc", "scores", "ev", "battery", "dongle", "seen", "base")

    def __init__(self, inc: Incident) -> None:
        self.inc = inc
        self.scores: dict[Cause, float] = defaultdict(float)
        self.ev: list[tuple[float, str]] = []
        self.battery: float | None = None
        self.dongle: str | None = None
        self.seen: list[str] | None = None
        self.base: str | None = None          # base serial implicated, if any

    def add(self, cause: Cause, weight: float, text: str) -> None:
        self.scores[cause] += weight
        self.ev.append((weight, text))

    def note(self, text: str) -> None:
        self.ev.append((0.0, text))

    def finish(self) -> None:
        scores = {c: s for c, s in self.scores.items() if s > 0.0}
        best = max(scores.values(), default=0.0)
        unknown = UNKNOWN_FLOOR if best < UNKNOWN_FLOOR else UNKNOWN_FLOOR * max(0.0, 1.0 - best)
        total = sum(scores.values()) + unknown
        causes = {c.value: round(s / total, 3) for c, s in scores.items() if s / total >= 0.03}
        if unknown / total >= 0.03:
            causes[C.UNKNOWN.value] = round(unknown / total, 3)
        self.inc.causes = dict(sorted(causes.items(), key=lambda kv: -kv[1]))
        ev = [text for _, text in sorted(self.ev, key=lambda e: -e[0])][:10]
        if best < UNKNOWN_FLOOR:
            ev.append("not enough evidence to name a cause")
        self.inc.evidence = ev


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _dur(s: float) -> str:
    return f"{s:.1f} s" if s < 60 else f"{s / 60:.1f} min"


def _join(items: list[str]) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


class Attributor:
    def __init__(self, world: World) -> None:
        self.world = world
        self._history: list[_Rec] = []

    # ---- naming --------------------------------------------------------------
    def _label(self, serial: str) -> str:
        info = self.world.devices.get(serial)
        if info is None:
            return "the tracking universe" if serial == "universe" else serial
        if info.device_class is DeviceClass.BASE:
            return self._base_name(serial)
        return info.label()

    def _base_name(self, serial: str) -> str:
        info = self.world.devices.get(serial)
        if info is not None and info.channel is not None:
            return f"base station S-{info.channel} ({serial})"
        return f"base station {serial}"

    def _resolve_base(self, token) -> str | None:
        """Map a log token ("S-2", bare hex id, or serial) to a base serial."""
        if not token:
            return None
        tok = str(token).upper()
        for b in self.world.bases():
            ser = b.serial.upper()
            if tok == ser or ser.endswith(tok) or tok.endswith(ser.split("-")[-1]):
                return b.serial
            if b.channel is not None and tok == f"S-{b.channel}":
                return b.serial
        return None

    def _where(self, pos: Vec3) -> str:
        x, y, z = pos
        r = math.hypot(x, z)
        height = "near the floor" if y < 0.45 else "at waist height" if y < 1.3 else "at head height"
        if r < 0.5:
            place = "near the centre of the playspace"
        else:
            place = None
            best = 0.75
            for b in self.world.bases():
                s = self.world.base_pose(b.serial)
                if s is None:
                    continue
                br = math.hypot(s.pos[0], s.pos[2])
                if br < 1e-6:
                    continue
                dot = (x * s.pos[0] + z * s.pos[2]) / (r * br)
                if dot > best:
                    best = dot
                    name = f"S-{b.channel}" if b.channel is not None else b.serial
                    place = f"about {r:.1f} m from the centre toward base station {name}"
            if place is None:
                side = ("right" if x > 0 else "left") if abs(x) > 0.3 else ""
                depth = ("back" if z > 0 else "front") if abs(z) > 0.3 else ""
                place = f"about {r:.1f} m from the centre, {'-'.join(p for p in (depth, side) if p)} side"
        return f"{place} (x {x:+.1f} m, z {z:+.1f} m, {height})"

    # ---- evidence gathering --------------------------------------------------
    def _seen_at(self, serial: str, t: float) -> list[str] | None:
        """Base slots the driver said the device was seeing at time t."""
        last = None
        for e in self.world.log:
            if e.t > t:
                break
            if e.serial == serial and e.kind in (LogKind.SOB_ADD, LogKind.SOB_DROP,
                                                 LogKind.TRACKING_RESET):
                last = e
        if last is None:
            return None
        if last.kind is LogKind.TRACKING_RESET:
            return []
        slots = [s.get("slot") for s in last.data.get("seeing", []) if s.get("slot")]
        if last.kind is LogKind.SOB_DROP and last.data.get("base") in slots:
            slots.remove(last.data["base"])
        return sorted(slots)

    def _pos_at(self, serial: str, t: float) -> Vec3 | None:
        buf = self.world.samples.get(serial)
        if not buf:
            return None
        best = None
        for s in reversed(buf):
            if s.connected and s.valid and s.result == 200:
                best = s.pos
                if s.t <= t:
                    break
        return best

    def _capsules(self, t: float) -> list[tuple[Vec3, Vec3, float]]:
        """Body approximation: head-waist torso and waist-foot legs."""
        by_role: dict[str, Vec3] = {}
        for d in self.world.tracked():
            role = d.role or ("head" if d.device_class is DeviceClass.HMD else "")
            if role in ("head", "waist", "left_foot", "right_foot"):
                p = self._pos_at(d.serial, t)
                if p is not None:
                    by_role[role] = p
        head, waist = by_role.get("head"), by_role.get("waist")
        if head is not None and waist is None:
            waist = (head[0], head[1] * 0.6, head[2])
        elif head is not None:
            # the tracker sits on the body surface; the torso axis is up to
            # ~10 cm further in, towards the line below the head
            dx, dz = head[0] - waist[0], head[2] - waist[2]
            d = math.hypot(dx, dz)
            k = min(1.0, 0.10 / d) if d > 1e-6 else 0.0
            waist = (waist[0] + dx * k, waist[1], waist[2] + dz * k)
        out = []
        if head is not None and waist is not None:
            out.append((head, waist, 0.17))
        for foot in ("left_foot", "right_foot"):
            if waist is not None and foot in by_role:
                out.append((waist, by_role[foot], 0.09))
        if head is not None and len(out) == 1:
            out.append((waist, (head[0], 0.0, head[2]), 0.14))      # no foot trackers: one leg column
        return out

    def _others(self, serial: str, t0: float, t1: float) -> dict[str, set[str]]:
        """What every other tracked device was doing during [t0, t1]:
        flags among "disc", "lost", "stale", "jump"."""
        out: dict[str, set[str]] = {}
        lo, hi = t0 - 0.3, t1 + 0.3
        for d in self.world.tracked():
            if d.serial == serial:
                continue
            buf = self.world.samples.get(d.serial)
            if not buf:
                continue
            n_disc = n_lost = n_ok = n_fresh = 0
            jump = False
            prev = fresh_prev = None
            for s in buf:
                if s.t < lo:
                    continue
                if s.t > hi:
                    break
                if not s.connected:
                    n_disc += 1
                    prev = fresh_prev = None
                elif not (s.valid and s.result == 200):
                    n_lost += 1
                    prev = fresh_prev = None
                else:
                    n_ok += 1
                    if prev is None or s.vel != prev.vel or s.angvel != prev.angvel:
                        n_fresh += 1
                        if fresh_prev is not None and 0 < s.t - fresh_prev.t < 0.05:
                            h = 0.5 * (s.t - fresh_prev.t)
                            dx = s.pos[0] - fresh_prev.pos[0]
                            dy = s.pos[1] - fresh_prev.pos[1]
                            dz = s.pos[2] - fresh_prev.pos[2]
                            rx = dx - (s.vel[0] + fresh_prev.vel[0]) * h
                            ry = dy - (s.vel[1] + fresh_prev.vel[1]) * h
                            rz = dz - (s.vel[2] + fresh_prev.vel[2]) * h
                            thr = 0.015 + 0.25 * math.sqrt(dx * dx + dy * dy + dz * dz)
                            if rx * rx + ry * ry + rz * rz > thr * thr:
                                jump = True
                        fresh_prev = s
                    prev = s
            flags: set[str] = set()
            if n_disc >= 3:
                flags.add("disc")
            if n_lost >= 3:
                flags.add("lost")
            if n_ok >= 40 and n_fresh < 0.5 * n_ok:
                flags.add("stale")
            if jump:
                flags.add("jump")
            out[d.serial] = flags
        return out

    def _geometry(self, pos: Vec3, t: float) -> tuple[list[str], list[str]]:
        """(bases with a clear view of pos, reasons the others do not have one)."""
        capsules = self._capsules(t)
        clear, reasons = [], []
        for b in self.world.bases():
            s = self.world.base_pose(b.serial)
            if s is None or not s.valid:
                continue
            name = f"S-{b.channel}" if b.channel is not None else b.serial
            dist, az, el = base_view(s.pos, s.quat, pos)
            if dist > RANGE_M:
                reasons.append(f"{name} is {dist:.1f} m away")
            elif abs(az) > FOV_H_DEG / 2 or abs(el) > FOV_V_DEG / 2:
                reasons.append(f"outside {name}'s sweep")
            elif sight_blocked(pos, s.pos, capsules):
                reasons.append(f"{name} blocked by the body")
            else:
                clear.append(name)
        return clear, reasons

    # ---- per-incident --------------------------------------------------------
    def attribute(self, inc: Incident) -> Incident:
        rec = _Rec(inc)
        if inc.kind is K.BASE_SHIFT:
            self._attr_base_shift(rec)
        elif inc.kind is K.UNIVERSE_SHIFT:
            self._attr_universe(rec)
        else:
            self._attr_device(rec)
        rec.finish()
        self._history.append(rec)
        if len(self._history) > MAX_HISTORY:
            del self._history[: MAX_HISTORY // 10]
        return inc

    def _attr_universe(self, rec: _Rec) -> None:
        inc = rec.inc
        mm = inc.metrics.get("shift_m", 0.0) * 1000
        rec.add(C.CALIBRATION, 0.7,
                f"everything moved together by {mm:.0f} mm at the same instant: "
                "the room calibration (universe) changed, not one device")
        logs = self.world.log_between(inc.t_start - 10, inc.t_end + 10,
                                      kinds=(LogKind.UNIVERSE_CHANGED, LogKind.UNIVERSE_TILT))
        if logs:
            rec.add(C.CALIBRATION, 0.25, "driver log: universe / tilt calibration changed")

    def _attr_base_shift(self, rec: _Rec) -> None:
        inc = rec.inc
        rec.base = inc.serial
        name = self._base_name(inc.serial)
        rec.add(C.BASE_STATION, 0.7,
                f"{name}'s reported pose moved by {inc.metrics.get('shift_m', 0) * 1000:.0f} mm / "
                f"{inc.metrics.get('shift_deg', 0):.1f} deg mid-session")
        tilts = self.world.log_between(inc.t_start - 10, inc.t_end + 10, kinds=(LogKind.TILT_UPDATE,))
        mine = [e for e in tilts if self._resolve_base(e.data.get("base")) == inc.serial]
        if mine:
            rec.add(C.BASE_STATION, 0.25,
                    f"driver log: it reported a tilt change of {mine[-1].data.get('delta_deg', 0):.1f} deg "
                    "(it was bumped or its mount is slipping)")
        # device trouble just before the driver noticed is explained by this
        hit = []
        for other in self._history:
            o = other.inc
            if (o.kind in _OPTICAL and o.t_end >= inc.t_start - 10 and o.t_start <= inc.t_end + 2
                    and other.base is None):
                other.base = inc.serial
                other.add(C.BASE_STATION, 0.5, f"{name}'s pose moved moments later")
                other.scores[C.REFLECTION] *= 0.5
                other.finish()
                hit.append(self._label(o.serial))
        if hit:
            rec.note(f"{_join(sorted(set(hit)))} misbehaved just before the driver corrected the pose")

    def _attr_device(self, rec: _Rec) -> None:
        inc = rec.inc
        w = self.world
        serial = inc.serial
        t0, t1 = inc.t_start, inc.t_end
        kind = inc.kind
        info = w.devices.get(serial)
        label = self._label(serial)
        dongle = rec.dongle = w.dongle_of.get(serial) or (info.dongle if info else None)
        add = rec.add
        m = inc.metrics

        # what happened
        if kind is K.DROPOUT:
            add(C.OCCLUSION, 0.2, f"{label} stayed connected but lost its pose for {_dur(inc.duration)}")
        elif kind is K.DISCONNECT:
            add(C.RADIO, 0.25, f"{label} dropped off its radio link for {_dur(inc.duration)}")
        elif kind is K.STALE:
            add(C.RADIO, 0.5, f"pose updates from {label} fell to {m.get('update_hz', 0):.0f}/s "
                              f"(normally ~{m.get('expected_hz', 0):.0f}) while tracking still reported OK")
        elif kind is K.JUMP:
            add(C.REFLECTION, 0.2, f"{label} jumped {m.get('jump_m', 0) * 100:.1f} cm "
                                   f"({int(m.get('count', 1))} jumps) while tracking stayed valid")
        elif kind is K.JITTER:
            rec.note(f"{label} jitter rose to {m.get('jitter_mm', 0):.1f} mm / {m.get('jitter_deg', 0):.2f} deg")

        # driver log around the incident
        logs = w.log_between(t0 - 3.0, t1 + 1.5)
        mine = [e for e in logs if e.serial == serial or e.data.get("device") == serial]
        n_reset = [e for e in mine if e.kind is LogKind.TRACKING_RESET]
        no_opt = [e for e in n_reset if e.data.get("reason") == "no_optical"]
        if no_opt and kind in _OPTICAL:
            ms = no_opt[0].data.get("ms")
            add(C.OCCLUSION, 0.35, "driver log: tracking reset because no base station light was "
                                   "received" + (f" for {ms} ms" if ms else ""))
        drops = {e.data.get("base") for e in mine
                 if e.kind is LogKind.SOB_DROP and t0 - 1.5 <= e.t <= t0 + 0.5}
        if drops and kind is K.DROPOUT:
            add(C.OCCLUSION, 0.15, f"driver log: lost sight of {len(drops)} base station(s) just before")
        if any(e.data.get("reason") == "imu_misalignment" for e in n_reset):
            add(C.IMU, 0.3, "driver log: tracking reset for IMU misalignment")
        if any(e.kind is LogKind.WIRELESS_DISCONNECTED for e in mine):
            add(C.RADIO, 0.35, "driver log: wireless connection to the receiver was lost")
        if dongle and any(e.kind is LogKind.RECEIVER_DISCONNECTED and e.data.get("dongle") == dongle
                          for e in logs):
            add(C.RADIO, 0.3, f"driver log: receiver {dongle} disconnected")
        hid = [e for e in logs if e.kind is LogKind.HID_ERROR]
        if hid:
            ifaces = sorted({str(e.data.get("interface", "?")) for e in hid})
            add(C.USB, 0.5, f"driver log: {len(hid)} USB HID error(s) ({', '.join(ifaces)})")
        n_imu = sum(1 for e in mine if e.kind is LogKind.IMU_OFF_SCALE)
        if n_imu:
            add(C.IMU, 0.5 + (0.2 if n_imu >= 3 else 0.0),
                f"driver log: IMU went off scale {n_imu} time(s) (shock or vibration beyond its range)")
        backfacing = sum(int(e.data.get("backfacing", 0)) for e in mine if e.kind is LogKind.BACKFACING_HITS)
        if backfacing >= 20 and kind in (K.JUMP, K.JITTER, K.DROPOUT):
            add(C.REFLECTION, 0.4, f"driver log: {backfacing} back-facing hits rejected "
                                   "(laser light arriving from an impossible direction)")
        if kind in _OPTICAL:
            wide = w.log_between(t0 - 8.0, t1 + 8.0,
                                 kinds=(LogKind.TILT_UPDATE, LogKind.UNIVERSE_CHANGED, LogKind.UNIVERSE_TILT))
            tilted = sorted({self._resolve_base(e.data.get("base")) or str(e.data.get("base"))
                             for e in wide if e.kind is LogKind.TILT_UPDATE})
            if tilted:
                rec.base = tilted[0]
                add(C.BASE_STATION, 0.5, f"driver log: {_join([self._base_name(b) for b in tilted])} "
                                         "reported a tilt change around the same time")
            else:
                for other in reversed(self._history):
                    o = other.inc
                    if o.kind is K.BASE_SHIFT and o.t_start <= t1 + 8 and o.t_end >= t0 - 8:
                        rec.base = o.serial
                        add(C.BASE_STATION, 0.5, f"{self._base_name(o.serial)}'s pose moved around the same time")
                        break
            if kind is K.JUMP and any(e.kind is not LogKind.TILT_UPDATE for e in wide):
                add(C.CALIBRATION, 0.5, "driver log: the universe / room calibration changed at that moment")

        # what the other devices were doing
        others = self._others(serial, t0, t1)
        n_others = len(others)
        down = {s for s, f in others.items() if f & {"disc", "lost"}}
        trouble = down | {s for s, f in others.items() if "stale" in f}
        jumped = {s for s, f in others.items() if "jump" in f}
        all_down = n_others >= 2 and len(down) >= max(2, math.ceil(0.8 * n_others))
        if kind in _CONNECTIVITY:
            if all_down:
                add(C.USB, 0.4 if hid else 0.35,
                    f"{len(down)} of {n_others} other devices dropped at the same moment")
                hmds = [d.serial for d in w.tracked() if d.device_class is DeviceClass.HMD]
                if any(h in down or h == serial for h in hmds):
                    add(C.USB, 0.1, "the wired headset dropped too, so this is not a wireless problem")
            elif trouble:
                shared = sorted(s for s in trouble if dongle and w.dongle_of.get(s) == dongle)
                if shared:
                    add(C.RADIO, 0.3, f"{_join([self._label(s) for s in shared])} on the same dongle "
                                      f"({dongle}) had trouble at the same time")
                else:
                    add(C.RADIO, 0.1, f"{len(trouble)} other device(s) on other receivers had trouble "
                                      "at the same time (possible 2.4 GHz interference)")
            else:
                rec.note("no other device was affected")
                if kind is K.DROPOUT:
                    add(C.OCCLUSION, 0.05, "only this device lost tracking")
        elif kind is K.JUMP:
            if len(jumped) >= 2:
                add(C.BASE_STATION, 0.3, f"{len(jumped)} other devices jumped at the same moment "
                                         "(something they share moved: a base station or the calibration)")
            elif not jumped:
                add(C.REFLECTION, 0.05, "only this device jumped")

        # geometry: could the bases see it?
        if kind in _OPTICAL and w.bases():
            clear, reasons = self._geometry(inc.pos, t0)
            if not clear and reasons:
                add(C.OCCLUSION, 0.3, "no base station had a clear line of sight: " + "; ".join(reasons))
            elif len(clear) == 1 and reasons:
                add(C.OCCLUSION, 0.15, f"only {clear[0]} had a clear line of sight"
                                       + (" (" + "; ".join(reasons) + ")" if reasons else ""))
            seen = rec.seen = self._seen_at(serial, t0 - 1.0)
            if seen is not None and len(seen) <= 1:
                add(C.OCCLUSION, 0.1, "the driver was only seeing "
                    + (f"base {seen[0]}" if seen else "no base at all") + " shortly before")

        # recurrence in the same spot
        if kind in (K.DROPOUT, K.JUMP):
            again = sum(1 for o in self._history
                        if o.inc.serial == serial and o.inc.kind is kind and t0 - o.inc.t_end > 2.0
                        and math.dist(o.inc.pos, inc.pos) <= SAME_SPOT_M)
            if again:
                if kind is K.DROPOUT:
                    add(C.OCCLUSION, min(0.15, 0.05 * again),
                        f"{again} earlier dropout(s) of {label} in the same spot")
                else:
                    add(C.REFLECTION, min(0.25, 0.1 * again),
                        f"{again} earlier jump incident(s) of {label} in the same spot")

        # battery
        battery = rec.battery = info.battery if info else None
        low = (battery is not None and battery <= LOW_BATTERY and not (info and info.charging)
               and kind in _CONNECTIVITY)
        if low:
            add(C.BATTERY, 0.3 + 0.4 * (1.0 - battery / LOW_BATTERY), f"{label} battery was at {_pct(battery)}")
            if not trouble:
                add(C.BATTERY, 0.1, "only the device with the low battery was affected")
            earlier = [o for o in self._history
                       if o.inc.serial == serial and o.inc.kind in _CONNECTIVITY
                       and t0 - o.inc.t_end < 120.0 and o.battery is not None and o.battery > battery]
            if len(earlier) >= 2:
                add(C.BATTERY, 0.15, f"{len(earlier) + 1} dropouts in two minutes as the charge fell")

        # causes that explain each other away
        sc = rec.scores
        if hid and all_down:
            sc[C.RADIO] *= 0.4
            sc[C.OCCLUSION] *= 0.3
            sc[C.BATTERY] *= 0.3
        elif all_down:
            sc[C.RADIO] *= 0.6
            sc[C.OCCLUSION] *= 0.3
        if low and battery <= 0.10 and not trouble:
            sc[C.RADIO] *= 0.6
        if sc[C.BASE_STATION] >= 0.5 or sc[C.IMU] >= 0.5 or sc[C.CALIBRATION] >= 0.5:
            sc[C.REFLECTION] *= 0.5
        if kind is K.DISCONNECT:
            sc[C.OCCLUSION] *= 0.3

    # ---- session summary -----------------------------------------------------
    def findings(self) -> list[Finding]:
        groups: dict[tuple, list[_Rec]] = defaultdict(list)
        for rec in self._history:
            cause = Cause(rec.inc.top_cause)
            if cause in (C.OCCLUSION, C.REFLECTION, C.IMU, C.BATTERY):
                key = (cause, rec.inc.serial)
            elif cause is C.RADIO:
                key = (cause, rec.dongle or rec.inc.serial)
            elif cause is C.BASE_STATION:
                key = (cause, rec.base or "?")
            else:
                key = (cause, "")
            groups[key].append(rec)
        out = [self._finding(cause, key, recs) for (cause, key), recs in groups.items()]
        follow = self._base_follow_finding()
        if follow is not None:
            out.append(follow)
        out.sort(key=lambda f: -f.score)
        return out

    def _spot(self, recs: list[_Rec]) -> tuple[str, bool]:
        """Describe where a group of incidents happened; True if localised."""
        pts = [r.inc.pos for r in recs]
        if len(pts) == 1:
            return "at one spot " + self._where(pts[0]), True
        med = tuple(sorted(p[i] for p in pts)[len(pts) // 2] for i in range(3))
        near = sum(1 for p in pts if math.hypot(p[0] - med[0], p[2] - med[2]) <= 1.0)
        if near >= 0.6 * len(pts):
            return f"{near} of {len(pts)} times {self._where(med)}", True
        return "all over the playspace", False

    def _finding(self, cause: Cause, key: str, recs: list[_Rec]) -> Finding:
        n = len(recs)
        dur = sum(r.inc.duration for r in recs)
        conf = sum(r.inc.causes.get(cause.value, 0.0) for r in recs) / n
        strength = 1.0 - math.exp(-(n + dur / 3.0) / 4.0)
        score = conf * (0.4 + 0.6 * strength)
        serials = sorted({r.inc.serial for r in recs if r.inc.kind not in (K.BASE_SHIFT, K.UNIVERSE_SHIFT)})
        names = _join([self._label(s) for s in serials]) or "devices"
        bases = sorted({r.base for r in recs if r.base})
        kinds = defaultdict(int)
        for r in recs:
            kinds[r.inc.kind.value] += 1
        what = ", ".join(f"{c} {k}" for k, c in sorted(kinds.items(), key=lambda kv: -kv[1]))
        top_ev: dict[str, tuple[float, str]] = {}
        for r in recs:
            for wgt, text in r.ev:
                if wgt > 0 and text.startswith("driver log"):
                    top_ev[_DIGITS.sub("N", text)] = (top_ev.get(_DIGITS.sub("N", text), (0.0, ""))[0] + wgt,
                                                      text.split(":", 1)[1].strip())
        log_bits = [t for _, t in sorted(top_ev.values(), key=lambda v: -v[0])[:2]]
        log_txt = (" Driver log: " + "; ".join(log_bits) + ".") if log_bits else ""
        plural = "s" if n != 1 else ""
        head = f"{n} incident{plural} ({what}), {_dur(dur)} in total."

        if cause is C.OCCLUSION:
            spot, local = self._spot(recs)
            title = f"{names} keeps losing sight of the base stations" if n > 1 else \
                f"{names} lost sight of the base stations"
            detail = f"{head} It stayed connected but received no laser light; it happened {spot}.{log_txt}"
            fix = (f"Give {names} a clear view of at least two base stations "
                   + ("there" if local else "everywhere you play")
                   + ": raise or re-aim a base (or add one on the opposite side) so it sees that spot "
                     "past your body, and move anything that blocks it (desk, chair, blanket, loose clothing "
                     "over the tracker).")
        elif cause is C.RADIO:
            shared = len(serials) > 1
            title = (f"Radio link to {names} is dropping out"
                     + (f" (shared dongle {key})" if shared else ""))
            detail = (f"{head} Pose updates slow down or stop and the device disconnects, optical and IMU "
                      f"data together"
                      + (", and the devices on that dongle fail at the same time" if shared else "")
                      + f".{log_txt}")
            fix = (f"Move the dongle for {names} onto a USB 2.0 port or a short USB extension so it sits in "
                   "the open with line of sight to the trackers, at least 20 cm from other dongles, USB 3 "
                   "devices and Wi-Fi routers; then re-pair the device if it keeps happening.")
        elif cause is C.BASE_STATION:
            bname = _join([self._base_name(b) for b in bases]) or "a base station"
            title = f"{bname[0].upper()}{bname[1:]} moved during the session"
            detail = (f"{head} Its pose changed and devices relying on it"
                      + (f" ({names})" if serials else "") + f" jumped or shifted.{log_txt}")
            fix = (f"Check that {bname} is rigidly mounted (tighten the bracket, no wobbling stand, not on "
                   "a wall that shakes or a shelf that gets bumped), then re-run room setup. If it keeps "
                   "reporting tilt changes while untouched, replace it.")
        elif cause is C.REFLECTION:
            spot, local = self._spot(recs)
            title = f"Reflections are making {names} jump"
            detail = (f"{head} Position jumps by a few centimetres while tracking stays valid; it happened "
                      f"{spot}.{log_txt}")
            fix = ("Cover or remove shiny surfaces that face the base stations "
                   + ("near that spot" if local else "around the room")
                   + ": mirrors, windows, glass picture frames, glossy screens, polished floors or furniture.")
        elif cause is C.IMU:
            title = f"{names} is being shaken beyond what its motion sensor can measure"
            detail = f"{head} Jitter spikes on this device only.{log_txt}"
            fix = (f"Tighten the strap or mount of {names} so it cannot rattle or slap against the body, "
                   "and avoid hard impacts. If it also happens while the device is at rest, its IMU is "
                   "faulty: recalibrate or replace it.")
        elif cause is C.USB:
            title = "USB connection to the headset/receivers glitched: every device dropped at once"
            detail = f"{head} Affected: {names}.{log_txt}"
            fix = ("Plug the headset or link box into a different USB controller (a rear motherboard port, "
                   "not a hub or front panel), turn off USB selective suspend / power saving, and replace "
                   "worn cables.")
        elif cause is C.BATTERY:
            levels = [r.battery for r in recs if r.battery is not None]
            lvl = f" (down to {_pct(min(levels))})" if levels else ""
            title = f"{names} is running out of battery{lvl}"
            trend = ""
            if n >= 4:
                ts = sorted(r.inc.t_start for r in recs)
                gaps = [b - a for a, b in zip(ts, ts[1:])]
                half = len(gaps) // 2
                a, b = sum(gaps[:half]) / max(1, half), sum(gaps[half:]) / max(1, len(gaps) - half)
                if b < a:
                    trend = f" Dropouts are getting more frequent: every {a:.0f} s at first, every {b:.0f} s lately."
            detail = f"{head} Only this device drops out and its charge is very low.{trend}{log_txt}"
            fix = f"Charge {names} or swap in a charged one; keep it above about 20% during sessions."
        elif cause is C.CALIBRATION:
            title = "The room calibration changed mid-session"
            detail = f"{head} All devices and base stations shifted together.{log_txt}"
            fix = ("Re-run SteamVR room setup once the base stations are fixed in place, and avoid changing "
                   "the base layout or chaperone while a session is running.")
        else:
            title = f"{n} incident{plural} without a clear cause"
            detail = f"{head} Affected: {names}. The evidence did not point at one cause."
            fix = ("Leave lhdebug running: more incidents (and where they happen) usually make the pattern "
                   "clear. Check the incident list for details.")
            score *= 0.5
        return Finding(cause=cause, score=round(min(1.0, score), 3), title=title, detail=detail, fix=fix,
                       devices=serials, bases=bases, incident_count=n)

    def _base_follow_finding(self) -> Finding | None:
        """Problems that follow one base: optical incidents that mostly happen
        while devices rely on that base alone, across several devices."""
        sole: dict[str, list[_Rec]] = defaultdict(list)
        total = 0
        for r in self._history:
            if r.inc.kind in _OPTICAL and r.seen is not None and len(r.seen) == 1:
                sole[r.seen[0]].append(r)
                total += 1
        if total < 4:
            return None
        slot, recs = max(sole.items(), key=lambda kv: len(kv[1]))
        serials = sorted({r.inc.serial for r in recs})
        if len(recs) < 4 or len(recs) < 0.7 * total or len(serials) < 2:
            return None
        base = self._resolve_base(slot)
        bname = self._base_name(base) if base else f"base station {slot}"
        names = _join([self._label(s) for s in serials])
        return Finding(
            cause=C.BASE_STATION, score=round(0.35 + 0.3 * (1 - math.exp(-len(recs) / 6)), 3),
            title=f"Tracking problems follow {bname}",
            detail=(f"{len(recs)} of {total} optical incidents ({names}) happened while the device could "
                    f"only see {bname}; the other bases do not show this."),
            fix=(f"Clean the front of {bname}, check its status LED and that nothing vibrates it, make sure "
                 "no other base uses its channel, and update its firmware. If devices still misbehave when "
                 "it is their only base, swap it with another to confirm and replace it."),
            devices=serials, bases=[base] if base else [], incident_count=len(recs))
