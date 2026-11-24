"""Reader for SteamVR's ``config/lighthouse/lighthousedb.json``.

The file is the lighthouse driver's persistent memory: every base station it
has ever seen (factory calibration, firmware, channel, measured tilt) and every
"universe" (a solved arrangement of bases). We only ever read it.

Identity
--------
A base station has three spellings in the wild:

* OpenVR serial        ``LHB-B3316995``
* vrserver.txt id      ``B3316995`` (sometimes lower case, ``b3316995``)
* lighthousedb.json    ``3006359957`` (the same 32-bit number, in decimal)

`base_id` gives the bare upper-case hex id, `base_serial` the ``LHB-`` form.
Everything in `LighthouseDB` is keyed by the ``LHB-`` serial.

Units / frames (worked out from the real file)
----------------------------------------------
* ``tilt.pitch`` / ``tilt.roll`` are radians in the file; exposed in degrees so
  they compare directly with the log's "pitch 36.23 deg roll 8.82 deg".
* ``sobChannel`` is zero based; the log's slot ``S-n`` is ``sobChannel + 1``.
  `BaseRecord.channel` is the 1-based one, `BaseRecord.slot` the "S-n" string.
* A universe pose is 7 floats ``[qx, qy, qz, qw, x, y, z]`` expressed in the
  frame of the universe's primary base (which therefore has the identity
  pose). It is NOT the OpenVR standing universe: no gravity alignment and no
  chaperone transform has been applied. Distances between bases are still
  directly comparable with the log's RELATIONSHIP lines. Exposed as
  ``pos=(x, y, z)`` and ``quat=(w, x, y, z)`` per the project convention.
* A base listed in a universe with an identity pose and
  ``target_serial_number`` equal to its own serial has never been located in
  that universe (``located=False``); the identity-pose base with
  ``target_serial_number == 0`` is the primary. (Inferred from the data; it
  agrees with the log's "Selected existing universe … (B3316995 is primary)".)
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from math import degrees, dist
from pathlib import Path
from typing import Any

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]

_HEX8 = re.compile(r"[0-9A-Fa-f]{8}\Z")
_FALLBACK_STEAM = Path(r"C:\Program Files (x86)\Steam")


# ---- identity helpers ------------------------------------------------------
def base_id(serial: str | int) -> str:
    """Any spelling of a base station -> bare upper-case hex id.

    ``"LHB-4F091D9A"`` / ``"4f091d9a"`` / ``1326062490`` / ``"1326062490"``
    all give ``"4F091D9A"``. Idempotent. A string of exactly 8 hex digits is
    always taken as hex (so ``"40512413"`` is an id, not a decimal number);
    any other all-digit string is the decimal form lighthousedb.json uses.
    Pass an ``int`` when the value is known to be the decimal form.
    Unrecognised input is returned stripped and upper-cased.
    """
    if isinstance(serial, int):
        return f"{serial & 0xFFFFFFFF:08X}"
    s = str(serial).strip()
    if s[:4].upper() == "LHB-":
        s = s[4:]
    if _HEX8.match(s):
        return s.upper()
    if s.isdigit():
        return f"{int(s) & 0xFFFFFFFF:08X}"
    return s.upper()


def base_serial(ident: str | int) -> str:
    """Any spelling of a base station -> ``"LHB-XXXXXXXX"``."""
    return "LHB-" + base_id(ident)


def steam_dir() -> Path:
    """Steam install dir: HKCU\\Software\\Valve\\Steam SteamPath, else Program Files (x86)."""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam") as key:
            value, _ = winreg.QueryValueEx(key, "SteamPath")
        if value:
            p = Path(os.path.normpath(str(value)))
            if p.is_dir():
                return p
    except Exception:
        pass
    return _FALLBACK_STEAM


def default_path() -> Path:
    return steam_dir() / "config" / "lighthouse" / "lighthousedb.json"


# ---- records ---------------------------------------------------------------
@dataclass
class BaseRecord:
    serial: str                       # "LHB-B3316995"
    id: str                           # "B3316995" (as in vrserver.txt)
    serial_number: int                # decimal form used in the json
    channel: int | None = None        # 1-based: the n of the log's "S-n"
    slot: str | None = None           # "S-n"
    model_id: int | None = None
    firmware: int | None = None
    mode: int | None = None           # basestation_mode
    faults: int = 0
    reset_count: int | None = None
    gravity: Vec3 | None = None       # accelerometer vector in the base's own frame
    tilt_pitch: float | None = None   # degrees
    tilt_roll: float | None = None    # degrees
    tilt_variance: float | None = None  # ~39.5 is the driver's "never measured" default
    last_seen: float | None = None    # unix epoch
    channels_seen: list[int] = field(default_factory=list)  # every channel it has used, oldest first
    states: int = 0                   # number of dynamic_states entries (config/firmware/channel changes)


@dataclass
class UniverseBase:
    serial: str                       # "LHB-…"
    pos: Vec3 = (0.0, 0.0, 0.0)       # metres, primary-base frame (see module doc)
    quat: Quat = (1.0, 0.0, 0.0, 0.0)  # (w, x, y, z)
    variance: float = 0.0
    located: bool = True              # False: listed but never placed in this universe
    primary: bool = False             # the base that defines the universe origin


@dataclass
class UniverseRecord:
    id: str
    bases: list[str] = field(default_factory=list)          # LHB serials
    poses: dict[str, UniverseBase] = field(default_factory=dict)  # keyed by LHB serial
    tilt_pitch: float | None = None   # degrees
    tilt_roll: float | None = None    # degrees
    tilt_variance: float | None = None
    last_chaperone_commit: float | None = None   # unix epoch of the last room setup using it
    previous_id: str | None = None

    @property
    def primary(self) -> str | None:
        for b in self.poses.values():
            if b.primary:
                return b.serial
        return None

    def located(self) -> list[str]:
        return [s for s in self.bases if self.poses[s].located]

    def distance(self, a: str | int, b: str | int) -> float | None:
        """Metres between two located bases, None if either is not placed."""
        pa, pb = self.poses.get(base_serial(a)), self.poses.get(base_serial(b))
        if pa is None or pb is None or not (pa.located and pb.located):
            return None
        return dist(pa.pos, pb.pos)


class LighthouseDB:
    def __init__(self) -> None:
        self.path: Path | None = None
        self.revision: int | None = None
        self.error: str | None = None            # why loading failed, if it did
        self.bases: dict[str, BaseRecord] = {}
        self.universes: list[UniverseRecord] = []
        self.known_objects: dict[str, str] = {}  # tracked-device serial -> deviceClass

    # ---- queries -------------------------------------------------------
    def base(self, ident: str | int) -> BaseRecord | None:
        return self.bases.get(base_serial(ident))

    def universe(self, universe_id: str | int) -> UniverseRecord | None:
        uid = str(universe_id)
        for u in self.universes:
            if u.id == uid:
                return u
        return None

    def current_universe(self) -> UniverseRecord | None:
        """Best guess at the universe in use: the most recently committed one.

        The log's UNIVERSE_CHANGED events name the id authoritatively; prefer
        `universe(id)` when one has been seen.
        """
        best = None
        for u in self.universes:
            if best is None or (u.last_chaperone_commit or 0.0) >= (best.last_chaperone_commit or 0.0):
                best = u
        return best

    def by_slot(self, slot: str | int, universe_id: str | int | None = None) -> BaseRecord | None:
        """Base on a channel ("S-5" or 5): the most recently seen match.

        Channels get reused over the years so stale records can share a slot;
        restricting to a universe narrows it further.
        """
        try:
            n = int(str(slot).upper().removeprefix("S-"))
        except ValueError:
            return None
        pool = list(self.bases.values())
        if universe_id is not None:
            u = self.universe(universe_id)
            members = set(u.bases) if u else set()
            pool = [b for b in pool if b.serial in members]
        hits = [b for b in pool if b.channel == n]
        return max(hits, key=lambda b: b.last_seen or 0.0) if hits else None

    def summary(self) -> dict:
        """JSON-able view for GET /api/lighthousedb."""
        cur = self.current_universe()
        universes = []
        for u in self.universes:
            located = u.located()
            universes.append({
                "id": u.id,
                "current": cur is not None and u.id == cur.id,
                "primary": u.primary,
                "bases": list(u.bases),
                "located": located,
                "frame": "primary_base",
                "poses": {s: {"pos": list(p.pos), "quat": list(p.quat), "variance": p.variance,
                              "located": p.located, "primary": p.primary}
                          for s, p in u.poses.items()},
                "distances": [{"a": a, "b": b, "distance": round(dist(u.poses[a].pos, u.poses[b].pos), 3)}
                              for i, a in enumerate(located) for b in located[i + 1:]],
                "tilt": {"pitch": u.tilt_pitch, "roll": u.tilt_roll, "variance": u.tilt_variance},
                "last_chaperone_commit": u.last_chaperone_commit,
                "previous_id": u.previous_id,
            })
        return {
            "path": str(self.path) if self.path else None,
            "ok": self.error is None,
            "error": self.error,
            "revision": self.revision,
            "bases": {s: {**asdict(b), "gravity": list(b.gravity) if b.gravity else None}
                      for s, b in self.bases.items()},
            "universes": universes,
            "known_objects": dict(self.known_objects),
        }


# ---- loading ---------------------------------------------------------------
def _num(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _deg(v: Any) -> float | None:
    f = _num(v)
    return None if f is None else degrees(f)


def _vec3(v: Any) -> Vec3 | None:
    if isinstance(v, list) and len(v) == 3:
        try:
            return (float(v[0]), float(v[1]), float(v[2]))
        except (TypeError, ValueError):
            return None
    return None


def _parse_base(raw: dict) -> BaseRecord | None:
    cfg = raw.get("config") or {}
    try:
        number = int(cfg["serialNumber"])
    except (KeyError, TypeError, ValueError):
        return None
    rec = BaseRecord(serial=base_serial(number), id=base_id(number), serial_number=number,
                     model_id=cfg.get("modelId"))
    states = [s for s in (raw.get("dynamic_states") or []) if isinstance(s, dict)]
    rec.states = len(states)
    for st in states:
        ch = (st.get("dynamic_state") or {}).get("sobChannel")
        if isinstance(ch, int) and (not rec.channels_seen or rec.channels_seen[-1] != ch + 1):
            rec.channels_seen.append(ch + 1)
    if states:
        # entries are appended chronologically, but trust the timestamp over the order
        last = max(states, key=lambda s: _num(s.get("time_last_seen")) or 0.0)
        dyn = last.get("dynamic_state") or {}
        ch = dyn.get("sobChannel")
        if isinstance(ch, int):
            rec.channel, rec.slot = ch + 1, f"S-{ch + 1}"
        rec.firmware = dyn.get("firmware_version")
        rec.mode = dyn.get("basestation_mode")
        rec.faults = dyn.get("faults") or 0
        rec.reset_count = dyn.get("reset_count")
        rec.gravity = _vec3(dyn.get("gravity_vector"))
        tilt = last.get("tilt") or {}
        rec.tilt_pitch, rec.tilt_roll = _deg(tilt.get("pitch")), _deg(tilt.get("roll"))
        rec.tilt_variance = _num(tilt.get("variance"))
        rec.last_seen = _num(last.get("time_last_seen"))
    return rec


_IDENTITY = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]


def _parse_universe(raw: dict) -> UniverseRecord:
    tilt = raw.get("tilt") or {}
    u = UniverseRecord(id=str(raw.get("id", "")),
                       tilt_pitch=_deg(tilt.get("pitch")), tilt_roll=_deg(tilt.get("roll")),
                       tilt_variance=_num(tilt.get("variance")),
                       last_chaperone_commit=_num(raw.get("lastChaperoneCommit")),
                       previous_id=str(raw["previous_id"]) if raw.get("previous_id") else None)
    for b in raw.get("base_stations") or []:
        try:
            number = int(b["base_serial_number"])
        except (KeyError, TypeError, ValueError):
            continue
        serial = base_serial(number)
        target = b.get("target_pose") or {}
        pose = target.get("pose")
        try:
            pose = [float(x) for x in pose] if isinstance(pose, list) and len(pose) == 7 else list(_IDENTITY)
        except (TypeError, ValueError):
            pose = list(_IDENTITY)
        qx, qy, qz, qw, x, y, z = pose
        identity = pose == _IDENTITY
        tsn = target.get("target_serial_number") or 0
        u.bases.append(serial)
        u.poses[serial] = UniverseBase(serial=serial, pos=(x, y, z), quat=(qw, qx, qy, qz),
                                       variance=_num(target.get("variance")) or 0.0,
                                       located=not (identity and tsn == number),
                                       primary=identity and tsn == 0)
    return u


def load(path: Path | str | None = None) -> LighthouseDB:
    """Read lighthousedb.json. Never raises: a missing/corrupt file gives an
    empty DB with `.error` set (SteamVR rewrites the file while running)."""
    db = LighthouseDB()
    db.path = Path(path) if path is not None else default_path()
    try:
        with open(db.path, "r", encoding="utf-8", errors="replace") as fh:
            raw = json.load(fh)
        if not isinstance(raw, dict):
            raise ValueError("top level is not an object")
    except (OSError, ValueError) as exc:
        db.error = f"{type(exc).__name__}: {exc}"
        return db
    rev = raw.get("revision")
    db.revision = rev if isinstance(rev, int) else None
    for b in raw.get("base_stations") or []:
        rec = _parse_base(b) if isinstance(b, dict) else None
        if rec is not None:
            db.bases[rec.serial] = rec
    for u in raw.get("known_universes") or []:
        if isinstance(u, dict):
            db.universes.append(_parse_universe(u))
    for o in raw.get("known_objects") or []:
        if isinstance(o, dict) and o.get("serialNumber"):
            db.known_objects[str(o["serialNumber"])] = str(o.get("deviceClass", ""))
    return db


def _main(argv: list[str]) -> int:
    import time

    db = load(argv[0] if argv else None)
    print(f"{db.path}  revision {db.revision}" + (f"  ERROR {db.error}" if db.error else ""))
    for b in sorted(db.bases.values(), key=lambda r: -(r.last_seen or 0)):
        seen = time.strftime("%Y-%m-%d %H:%M", time.localtime(b.last_seen)) if b.last_seen else "never"
        tilt = f"{b.tilt_pitch:6.2f} {b.tilt_roll:6.2f}" if b.tilt_pitch is not None else "     -      -"
        print(f"  {b.serial}  {b.slot or '-':5} fw {b.firmware}  faults {b.faults}  resets {b.reset_count}  "
              f"tilt {tilt}  last seen {seen}")
    cur = db.current_universe()
    for u, s in zip(db.universes, db.summary()["universes"]):
        print(f"universe {u.id}{' (current)' if u is cur else ''}: primary {u.primary}, "
              f"{len(s['located'])}/{len(u.bases)} bases located, "
              f"tilt {u.tilt_pitch or 0:.2f}/{u.tilt_roll or 0:.2f} deg")
        for row in s["distances"]:
            print(f"    {row['a']} <-> {row['b']}  {row['distance']:.2f} m")
    return 1 if db.error else 0


if __name__ == "__main__":
    import sys
    raise SystemExit(_main(sys.argv[1:]))
