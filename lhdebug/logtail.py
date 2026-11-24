"""vrserver.txt -> typed `LogEvent`s.

`parse_line` turns one log line into an event (or None for noise),
`LogTailer` follows the live file across SteamVR restarts, `scan_file` parses a
whole file, and ``python -m lhdebug.logtail [path]`` prints a per-kind /
per-device count table for offline diagnosis.

Line anatomy (all formats here were taken from real logs)::

    Mon Nov 23 2026 20:00:07.705 [Info] - lighthouse: LHR-329ADD4A H: SOB: add S-7
    `-- local time, ms ---------' level   subsystem   device       |  message
                                                                   `- tracking context tag
                                                                      (H = HMD, C = controller/
                                                                      tracker); often absent

Conventions for `LogEvent.data`
-------------------------------
* The keys documented on each `LogKind` in model.py are always present.
* Base stations are bare upper-case hex ids ("EE93C79B") whatever case the log
  used; "S-n" slot strings are kept as written.
* Extra keys are additive. Notable ones:
    - SOB_ADD/SOB_DROP: ``base_id`` (hex id of the added/dropped base when the
      line gives it, else None), ``generation_changed`` (bool).
    - TILT_UPDATE: ``old_pitch``, ``old_roll``, ``new_pitch``, ``new_roll``,
      ``variance``.  ``delta_deg`` is the number inside ``<- 12.90 deg ->``.
    - TRACKING_RESET: ``base`` (for "no optical samples from base X"),
      ``angles``/``sigma`` (IMU misalignment), ``pos`` (position out of bounds).
    - BASE_ADD_FAILED: ``stage`` ("secondary" | "start"), ``hits`` ([a, b]).
    - BASE_BOOTSTRAPPED: ``pitch``, ``roll``; ``mode`` can also be "best".
      ``velocity_missing`` is True for the variant with no velocity in the line
      (``velocity`` is then 0.0).
    - UNIVERSE_CHANGED: ``action`` ("selected" | "stopped"), and for selected
      ``base`` + ``role`` ("primary" | "secondary").
    - WIRELESS_DISCONNECTED / DONGLE_ENUMERATED: see the regex table.
    - OTHER: ``what`` names lines that are recognised but have no LogKind of
      their own ("base_moved", "calibration_failed", "no_optical_frames",
      "optical_frames_resumed", "no_bases_seen", "best_tilt",
      "late_optical_data", "base_moved_tracking_pop"), plus their parsed
      fields. Truly unparsed lines have no ``what``.
"""
from __future__ import annotations

import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

from .lighthousedb import steam_dir
from .model import LogEvent, LogKind

__all__ = ["parse_line", "LogTailer", "scan_file", "default_path", "summarize"]

# ---- line envelope ---------------------------------------------------------
_LINE = re.compile(
    r"^[A-Z][a-z]{2} ([A-Z][a-z]{2}) (\d{1,2}) (\d{4}) (\d{2}):(\d{2}):(\d{2})\.(\d{3}) "
    r"\[(\w+)\] - (.*)$")
_MONTHS = {m: i + 1 for i, m in enumerate(
    "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split())}
_LH = "lighthouse: "
_F = r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"          # a float
_B = r"([0-9A-Fa-f]{8})"                            # a base id
_DEVICE = re.compile(r"^(LH[RB]-[0-9A-F]{8})(?: ([A-Z]))?: (.*)$")
_SERVER_START = re.compile(r"^vrserver (\S+) startup with PID=(\d+)")

_epoch_cache: dict[tuple, float] = {}


def _epoch(mon: str, day: str, year: str, hh: str, mm: str, ss: str, ms: str) -> float:
    """Local wall-clock time -> unix epoch (the log has no zone; it is local)."""
    key = (year, mon, day, hh, mm)
    base = _epoch_cache.get(key)
    if base is None:
        if len(_epoch_cache) > 512:
            _epoch_cache.clear()
        base = time.mktime((int(year), _MONTHS[mon], int(day), int(hh), int(mm), 0, 0, 0, -1))
        _epoch_cache[key] = base
    return base + int(ss) + int(ms) / 1000.0


def _up(s: str) -> str:
    return s.upper()


# ---- device-scoped messages ("lighthouse: LHR-xxxxxxxx [H|C]: <msg>") -------
_SOB = re.compile(
    r"^SOB: (add|drop) (S-\d+)(?: \(" + _B + r"\))?( \(generation changed\))?"
    r"(?: (?:also )?seeing(.*))?$")
_SOB_SEEN = re.compile(r"(S-\d+)(?: \(" + _B + r"\))?")
_DROPPED = re.compile(r"^Dropped (.+) during the previous tracking session")
_DROPPED_PART = re.compile(r"(\d+) (rejected updates?|back-facing hits?|non-clustered hits?)")
_HITS = re.compile(r"\(hits: (\d+)\+(\d+)\)")

Handler = Callable[[re.Match], tuple[LogKind, dict[str, Any]]]


def _sob(m: re.Match) -> tuple[LogKind, dict[str, Any]]:
    seeing = [{"slot": s, "base": b.upper() if b else None}
              for s, b in _SOB_SEEN.findall(m.group(5) or "")]
    kind = LogKind.SOB_ADD if m.group(1) == "add" else LogKind.SOB_DROP
    return kind, {"base": m.group(2), "seeing": seeing,
                  "base_id": m.group(3).upper() if m.group(3) else None,
                  "generation_changed": bool(m.group(4))}


def _dropped(m: re.Match) -> tuple[LogKind, dict[str, Any]]:
    data = {"rejected": 0, "backfacing": 0, "non_clustered": 0}
    for n, what in _DROPPED_PART.findall(m.group(1)):
        key = {"r": "rejected", "b": "backfacing", "n": "non_clustered"}[what[0]]
        data[key] = int(n)
    return LogKind.BACKFACING_HITS, data


def _reset(m: re.Match) -> tuple[LogKind, dict[str, Any]]:
    text = m.group(1).strip()
    data: dict[str, Any] = {"reason": "other", "ms": None, "text": text}
    mm = re.match(r"no optical samples(?: from base " + _B + r")? for (\d+)\s*ms", text)
    if mm:
        data.update(reason="no_optical", ms=int(mm.group(2)))
        if mm.group(1):
            data["base"] = mm.group(1).upper()
        return LogKind.TRACKING_RESET, data
    mm = re.match(r"IMU misalignment unreasonably large \(([^)]*)\) deg(?: sigma " + _F + ")?", text)
    if mm:
        data["reason"] = "imu_misalignment"
        data["angles"] = _floats(mm.group(1))
        if mm.group(2):
            data["sigma"] = float(mm.group(2))
        return LogKind.TRACKING_RESET, data
    mm = re.match(r"position out of bounds \(([^)]*)\)", text)
    if mm:
        data["pos"] = _floats(mm.group(1))
        data["detail"] = "position_out_of_bounds"
    return LogKind.TRACKING_RESET, data


def _floats(csv: str) -> list[float]:
    out = []
    for part in csv.split(","):
        try:
            out.append(float(part))
        except ValueError:
            pass
    return out


def _add_failed(m: re.Match) -> tuple[LogKind, dict[str, Any]]:
    reason = m.group(3).strip()
    data: dict[str, Any] = {"base": m.group(2).upper(), "reason": reason,
                            "stage": "secondary" if m.group(1).startswith("add") else "start"}
    hits = _HITS.search(reason)
    if hits:
        data["hits"] = [int(hits.group(1)), int(hits.group(2))]
    return LogKind.BASE_ADD_FAILED, data


def _calibration_failed(m: re.Match) -> tuple[LogKind, dict[str, Any]]:
    text = m.group(1).strip()
    data: dict[str, Any] = {"what": "calibration_failed", "text": text}
    mm = re.match(r"no optical samples from base " + _B + r" for (\d+)\s*ms", text)
    if mm:
        data.update(base=mm.group(1).upper(), ms=int(mm.group(2)))
    mm = re.match(r"position out of bounds \(([^)]*)\)", text)
    if mm:
        data["pos"] = _floats(mm.group(1))
    return LogKind.OTHER, data


def _assertion(m: re.Match) -> tuple[LogKind, dict[str, Any]]:
    text = m.group(1).strip()
    if "sync" in text.lower():
        return LogKind.SYNC_ASSERT, {"text": text}
    return LogKind.OTHER, {"what": "assertion", "text": text}


_DEVICE_RULES: list[tuple[re.Pattern, Handler]] = [
    (_SOB, _sob),
    (re.compile(r"^-+ RELATIONSHIP bases " + _B + r" <-> " + _B + r" distance " + _F + r"m, angle " + _F + r" deg"),
     lambda m: (LogKind.BASE_RELATIONSHIP, {"a": _up(m.group(1)), "b": _up(m.group(2)),
                                            "distance": float(m.group(3)), "angle": float(m.group(4))})),
    (re.compile(r"^-+ CALIBRATED base " + _B + r" at pitch " + _F + r" deg roll " + _F + r" deg"),
     lambda m: (LogKind.BASE_CALIBRATED, {"base": _up(m.group(1)), "pitch": float(m.group(2)),
                                          "roll": float(m.group(3))})),
    (re.compile(r"^-+ SECONDARY base " + _B + r" distance " + _F + r"m"),
     lambda m: (LogKind.BASE_SECONDARY, {"base": _up(m.group(1)), "distance": float(m.group(2))})),
    (re.compile(r"^-+ BOOTSTRAPPED base " + _B + r" \((\w+)\) distance " + _F + r"m(?: velocity " + _F
                + r"m/s)?(?: base pitch ~" + _F + r" deg roll ~" + _F + r" deg)?"),
     lambda m: (LogKind.BASE_BOOTSTRAPPED, {
         "base": _up(m.group(1)), "distance": float(m.group(3)),
         # one real variant omits the velocity; keep the key a float as documented
         "velocity": float(m.group(4)) if m.group(4) else 0.0,
         "velocity_missing": m.group(4) is None,
         "mode": m.group(2),
         "pitch": float(m.group(5)) if m.group(5) else None,
         "roll": float(m.group(6)) if m.group(6) else None})),
    (re.compile(r"^IMU went off scale"), lambda m: (LogKind.IMU_OFF_SCALE, {})),
    (re.compile(r"^Resetting tracking: (.*)$"), _reset),
    (_DROPPED, _dropped),
    (re.compile(r"^Trying to (add a secondary|start tracking from) base " + _B + r": (.*)$"), _add_failed),
    (re.compile(r"^Connected to receiver (\S+)"),
     lambda m: (LogKind.RECEIVER_CONNECTED, {"dongle": m.group(1)})),
    (re.compile(r"^Disconnected from receiver (\S+)"),
     lambda m: (LogKind.RECEIVER_DISCONNECTED, {"dongle": m.group(1)})),
    (re.compile(r"^Updated IMU calibration: Accel bias change " + _F + r"\s*m/s/s"),
     lambda m: (LogKind.IMU_CALIBRATION, {"accel_bias": float(m.group(1))})),
    (re.compile(r"^Assertion failed:\s*(.*)$"), _assertion),
    # --- recognised, but no LogKind of their own: OTHER with a `what` tag ---
    (re.compile(r"^Calibration failed: (.*)$"), _calibration_failed),
    (re.compile(r"^No optical frames in past " + _F + r" seconds"),
     lambda m: (LogKind.OTHER, {"what": "no_optical_frames", "seconds": float(m.group(1))})),
    (re.compile(r"^Optical frames received again"),
     lambda m: (LogKind.OTHER, {"what": "optical_frames_resumed"})),
    (re.compile(r"^No base stations seen"),
     lambda m: (LogKind.OTHER, {"what": "no_bases_seen"})),
    (re.compile(r"^LighthouseDirect: Optical data at time " + _F + r" < filter at " + _F),
     lambda m: (LogKind.OTHER, {"what": "late_optical_data",
                                "late_ms": round((float(m.group(2)) - float(m.group(1))) * 1000.0, 3)})),
]

# Start-up / housekeeping chatter about a device: dropped (None), not OTHER.
_DEVICE_NOISE = re.compile(
    r"^(?:Transition to protocol version|Protocol versioning supported|Checking to see if display"
    r"|Triggered keepalive|Packet received after|Optical subprotocol changed"
    r"|Controller subprotocol changed|Successfully fetched gyro|Read config of|Firmware Version"
    r"|Applying scale|SwSyncDetect|Source: auto disambiguator|Saw demod rate|TDM sensorType"
    r"|Warning: TDM sensorType|Attempting HID Open|Bootstrapped setting tilt)")

# ---- driver-wide messages ("lighthouse: <msg>", no tracked-device prefix) ---
# Each handler returns (kind, serial, data).
GHandler = Callable[[re.Match], tuple[LogKind, "str | None", dict[str, Any]]]

_GLOBAL_RULES: list[tuple[re.Pattern, GHandler]] = [
    (re.compile(r"^Updating tilt for " + _B + r":\s*Old=\s*" + _F + r",\s*" + _F + r",\s*" + _F
                + r"\s*<-\s*" + _F + r" deg\s*->\s*new=\s*" + _F + r",\s*" + _F + r",\s*" + _F),
     lambda m: (LogKind.TILT_UPDATE, None, {
         "base": _up(m.group(1)), "delta_deg": float(m.group(5)),
         "old_pitch": float(m.group(2)), "old_roll": float(m.group(3)),
         "new_pitch": float(m.group(6)), "new_roll": float(m.group(7)),
         "variance": float(m.group(8))})),
    (re.compile(r"^Setting universe tilt from " + _B + r" .*pitch " + _F + r" deg roll " + _F + r" deg"),
     lambda m: (LogKind.UNIVERSE_TILT, None, {"base": _up(m.group(1)), "pitch": float(m.group(2)),
                                              "roll": float(m.group(3))})),
    # "<dongle>: Wireless controller [LHR-…] disconnected" (device is blank when
    # the dongle had nothing paired through it at shutdown)
    (re.compile(r"^(\S+): Wireless controller\s+(?:(LH[RB]-[0-9A-F]{8})\s+)?disconnected"),
     lambda m: (LogKind.WIRELESS_DISCONNECTED, m.group(2), {"dongle": m.group(1), "device": m.group(2)})),
    (re.compile(r"^Lighthouse (\w+) HID device error"),
     lambda m: (LogKind.HID_ERROR, None, {"interface": m.group(1)})),
    (re.compile(r"^Stopped tracking with universe (\d+)"),
     lambda m: (LogKind.UNIVERSE_CHANGED, None, {"universe": m.group(1), "action": "stopped"})),
    (re.compile(r"^Selected existing universe (\d+)(?: \(" + _B + r" is (\w+)\))?"),
     lambda m: (LogKind.UNIVERSE_CHANGED, None, {
         "universe": m.group(1), "action": "selected",
         "base": _up(m.group(2)) if m.group(2) else None, "role": m.group(3)})),
    (re.compile(r"^Device (LH[RB]-[0-9A-F]{8}) powering off upon entering standby"),
     lambda m: (LogKind.DEVICE_STANDBY, m.group(1), {})),
    # "    28DE, 2101: s/n E518136DA0: Watchman Dongle (interface 0)"
    (re.compile(r"^\s*([0-9A-Fa-f]{4}), ([0-9A-Fa-f]{4}): s/n (\S+): "
                r"((Watchman Dongle|Valve VR Radio)[^(]*?)\s*\(interface (\d+)\)"),
     lambda m: (LogKind.DONGLE_ENUMERATED, None, {
         "dongle": m.group(3), "kind": m.group(5), "product": m.group(4),
         "vid": m.group(1).upper(), "pid": m.group(2).upper(), "interface": int(m.group(6))})),
    # --- recognised, but no LogKind of their own: OTHER with a `what` tag ---
    (re.compile(r"^Moving base " + _B + r" " + _F + r"mm and " + _F + r" deg because of relationship with " + _B),
     lambda m: (LogKind.OTHER, None, {"what": "base_moved", "base": _up(m.group(1)),
                                      "mm": float(m.group(2)), "deg": float(m.group(3)),
                                      "reference": _up(m.group(4))})),
    (re.compile(r"^\.\.\. Moving the base for tracking too"),
     lambda m: (LogKind.OTHER, None, {"what": "base_moved_tracking_pop"})),
    (re.compile(r"^Best tilt from (LH[RB]-[0-9A-F]{8}) for base " + _B + r" pitch " + _F + r" roll " + _F
                + r"(?: score " + _F + r")?"),
     lambda m: (LogKind.OTHER, m.group(1), {
         "what": "best_tilt", "base": _up(m.group(2)), "pitch": float(m.group(3)),
         "roll": float(m.group(4)), "score": float(m.group(5)) if m.group(5) else None})),
]


# ---- parse_line ------------------------------------------------------------
def parse_line(line: str) -> LogEvent | None:
    """One vrserver.txt line -> event, or None for non-lighthouse lines / noise.

    Never raises on malformed input.
    """
    m = _LINE.match(line.rstrip("\r\n"))
    if m is None:
        return None
    mon, day, year, hh, mi, ss, ms, level, body = m.groups()
    if body.startswith(_LH):
        hit = _parse_lighthouse(body[len(_LH):])
    elif body.startswith("vrserver "):
        sm = _SERVER_START.match(body)
        hit = (LogKind.SERVER_START, None,
               {"version": sm.group(1), "pid": int(sm.group(2))}) if sm else None
    else:
        hit = None
    if hit is None or mon not in _MONTHS:
        return None
    try:
        t = _epoch(mon, day, year, hh, mi, ss, ms)
    except (ValueError, OverflowError, OSError):
        return None
    kind, serial, data = hit
    return LogEvent(t=t, kind=kind, serial=serial, data=data, raw=line.rstrip("\r\n"), level=level)


def _parse_lighthouse(msg: str) -> tuple[LogKind, str | None, dict[str, Any]] | None:
    dm = _DEVICE.match(msg)
    if dm is not None:
        serial, _ctx, rest = dm.groups()
        for rx, handler in _DEVICE_RULES:
            mm = rx.match(rest)
            if mm is not None:
                kind, data = handler(mm)
                return kind, serial, data
        if _DEVICE_NOISE.match(rest):
            return None
        return LogKind.OTHER, serial, {}
    for rx, ghandler in _GLOBAL_RULES:
        mm = rx.match(msg)
        if mm is not None:
            return ghandler(mm)
    return None        # monitor enumeration, dongle keepalives, HID opens, …


# ---- files -----------------------------------------------------------------
def default_path() -> Path:
    """<Steam>/logs/vrserver.txt (registry SteamPath, else Program Files (x86))."""
    return steam_dir() / "logs" / "vrserver.txt"


def _decode(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace")


def scan_file(path: Path | str | None = None) -> list[LogEvent]:
    """Parse a whole log file. Missing file -> []."""
    p = Path(path) if path is not None else default_path()
    try:
        with open(p, "rb") as fh:
            blob = fh.read()
    except OSError:
        return []
    out = []
    for line in _decode(blob).splitlines():
        ev = parse_line(line)
        if ev is not None:
            out.append(ev)
    return out


class LogTailer:
    """Non-blocking follower of vrserver.txt.

    The file is opened, read and closed inside each `drain()`; no handle is
    held between calls, so SteamVR is never blocked from renaming the file on
    restart (Windows refuses to rename a file someone holds open without
    FILE_SHARE_DELETE).

    Rotation: on restart SteamVR renames vrserver.txt to vrserver.previous.txt
    and starts a new file. Detected by any of: file identity (st_dev, st_ino)
    changing, size dropping below our read offset, or the first bytes of the
    file changing. On rotation the unread tail of the old file is recovered
    from ``<name>.previous<ext>`` when that is verifiably the same file, then
    the new file is read from its start.
    """

    _HEAD = 64            # bytes of file start remembered as a fingerprint
    _MAX_READ = 8 << 20   # cap per drain so a huge backlog cannot stall a frame

    def __init__(self, path: Path | str | None = None, from_start: bool = False) -> None:
        self.path = Path(path) if path is not None else default_path()
        self._pos = 0
        self._buf = b""                 # partial last line carried to the next drain
        self._ident: tuple[int, int] | None = None
        self._head = b""
        self._skip_fragment = False     # started mid-line: discard up to the first newline
        self.rotations = 0
        if not from_start:
            self._seek_end()

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _identity(st: os.stat_result) -> tuple[int, int] | None:
        return (st.st_dev, st.st_ino) if st.st_ino else None

    def _seek_end(self) -> None:
        try:
            with open(self.path, "rb") as fh:
                st = os.fstat(fh.fileno())
                self._head = fh.read(self._HEAD)
                self._ident = self._identity(st)
                self._pos = st.st_size
                if st.st_size:
                    fh.seek(st.st_size - 1)
                    self._skip_fragment = fh.read(1) != b"\n"
        except OSError:
            pass   # not there yet: everything that appears later is new

    def _reset(self) -> None:
        self._pos = 0
        self._buf = b""
        self._ident = None
        self._head = b""
        self._skip_fragment = False

    def _previous_path(self) -> Path:
        return self.path.with_name(self.path.stem + ".previous" + self.path.suffix)

    def _recover_rotated_tail(self) -> bytes:
        """Unread remainder of the file we were following, if it was renamed."""
        if self._ident is None:
            return b""
        try:
            with open(self._previous_path(), "rb") as fh:
                st = os.fstat(fh.fileno())
                if self._identity(st) != self._ident or st.st_size <= self._pos:
                    return b""
                fh.seek(self._pos)
                return fh.read(self._MAX_READ)
        except OSError:
            return b""

    # -- API -------------------------------------------------------------
    def drain(self) -> list[LogEvent]:
        """New events since the last call. Never blocks, never raises on I/O."""
        chunks: list[bytes] = []
        try:
            fh = open(self.path, "rb")
        except OSError:
            # Missing (or mid-rename). If we were following a file, pick up
            # whatever was appended before it was moved away.
            if self._ident is not None or self._pos:
                tail = self._recover_rotated_tail()
                events = self._consume(tail, final=True)
                self._reset()
                self.rotations += 1
                return events
            return []
        events: list[LogEvent] = []
        try:
            with fh:
                st = os.fstat(fh.fileno())
                ident = self._identity(st)
                head = fh.read(self._HEAD)
                rotated = (
                    st.st_size < self._pos
                    or (ident is not None and self._ident is not None and ident != self._ident)
                    or (self._head and head[:len(self._head)] != self._head)
                )
                if rotated:
                    events += self._consume(self._recover_rotated_tail(), final=True)
                    self._reset()
                    self.rotations += 1
                self._ident = ident
                if len(head) > len(self._head):
                    self._head = head
                if st.st_size > self._pos:
                    fh.seek(self._pos)
                    data = fh.read(min(st.st_size - self._pos, self._MAX_READ))
                    self._pos += len(data)
                    chunks.append(data)
        except OSError:
            return events
        for data in chunks:
            events += self._consume(data)
        return events

    def _consume(self, data: bytes, final: bool = False) -> list[LogEvent]:
        """Split bytes into complete lines; keep a trailing partial line."""
        if not data and not (final and self._buf):
            return []
        data = self._buf + data
        self._buf = b""
        if self._skip_fragment:
            nl = data.find(b"\n")
            if nl < 0:
                return []
            data = data[nl + 1:]
            self._skip_fragment = False
        lines = data.split(b"\n")
        tail = lines.pop()
        if final:
            if tail:
                lines.append(tail)
        else:
            self._buf = tail
        out = []
        for raw in lines:
            ev = parse_line(_decode(raw))
            if ev is not None:
                out.append(ev)
        return out


# ---- offline summary / CLI -------------------------------------------------
_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:e-?\d+)?")
_RAW_DEVICE = re.compile(r" - lighthouse: LH[RB]-[0-9A-F]{8}(?: [A-Z])?: ")


def _shape(ev: LogEvent) -> str:
    """Collapse a raw line to its shape so unparsed lines can be grouped."""
    msg = ev.raw.split(" - lighthouse: ", 1)[-1]
    msg = re.sub(r"^LH[RB]-[0-9A-F]{8}(?: [A-Z])?: ", "", msg)
    msg = re.sub(r"\b[0-9A-Fa-f]{8}\b", "<base>", msg)
    return _NUM.sub("N", msg)[:110]


def summarize(path: Path | str | None = None) -> dict[str, Any]:
    """Counts for one log file: per kind, per device x kind, OTHER breakdown."""
    p = Path(path) if path is not None else default_path()
    try:
        with open(p, "rb") as fh:
            lines = _decode(fh.read()).splitlines()
    except OSError as exc:
        return {"path": str(p), "error": str(exc)}
    events = [ev for ev in map(parse_line, lines) if ev is not None]
    device_lines = sum(1 for ln in lines if _RAW_DEVICE.search(ln))
    other = [e for e in events if e.kind is LogKind.OTHER]
    untagged = [e for e in other if e.serial and "what" not in e.data]
    by_device: dict[str, Counter] = {}
    for e in events:
        by_device.setdefault(e.serial or "(driver)", Counter())[e.kind.value] += 1
    return {
        "path": str(p),
        "lines": len(lines),
        "lighthouse_lines": sum(1 for ln in lines if " - lighthouse: " in ln),
        "device_lines": device_lines,
        "events": len(events),
        "span": (events[0].t, events[-1].t) if events else None,
        "by_kind": dict(Counter(e.kind.value for e in events).most_common()),
        "by_device": {k: dict(v) for k, v in sorted(by_device.items())},
        "other_tagged": dict(Counter(e.data["what"] for e in other if "what" in e.data).most_common()),
        "other_device": sum(1 for e in other if e.serial and _RAW_DEVICE.search(e.raw)),
        "other_untagged": len(untagged),
        "unparsed_shapes": Counter(_shape(e) for e in untagged).most_common(10),
    }


def _print_summary(s: dict[str, Any]) -> None:
    print(f"== {s['path']}")
    if "error" in s:
        print(f"   cannot read: {s['error']}")
        return
    span = ""
    if s["span"]:
        fmt = lambda t: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))  # noqa: E731
        span = f"   {fmt(s['span'][0])} .. {fmt(s['span'][1])}"
    print(f"   {s['lines']} lines, {s['lighthouse_lines']} lighthouse, "
          f"{s['device_lines']} about a tracked device, {s['events']} events{span}")
    kinds = list(s["by_kind"])
    if not kinds:
        return
    print()
    print(f"   {'kind':<24}{'count':>7}")
    for k in kinds:
        print(f"   {k:<24}{s['by_kind'][k]:>7}")
    print()
    cols = [k for k in kinds if any(k in v for v in s["by_device"].values())]
    abbr = {k: "".join(w[:3] for w in k.split("_"))[:7] for k in cols}
    print("   " + f"{'device':<14}" + "".join(f"{abbr[k]:>8}" for k in cols))
    for dev, counts in s["by_device"].items():
        print("   " + f"{dev:<14}" + "".join(f"{counts.get(k, 0) or '.':>8}" for k in cols))
    print("   columns: " + ", ".join(f"{abbr[k]}={k}" for k in cols))
    print()
    dl = s["device_lines"] or 1
    print(f"   OTHER on device lines: {s['other_device']} / {s['device_lines']} "
          f"({100.0 * s['other_device'] / dl:.1f}%), of which unparsed (no 'what'): "
          f"{s['other_untagged']} ({100.0 * s['other_untagged'] / dl:.1f}%)")
    if s["other_tagged"]:
        print("   tagged OTHER: " + ", ".join(f"{k}={v}" for k, v in s["other_tagged"].items()))
    for shape, n in s["unparsed_shapes"]:
        print(f"   {n:>5}  {shape}")


def main(argv: Iterable[str] | None = None) -> int:
    import sys
    args = list(sys.argv[1:] if argv is None else argv)
    paths = [Path(a) for a in args] or [default_path()]
    for i, p in enumerate(paths):
        if i:
            print()
        _print_summary(summarize(p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
