"""Shared data contracts. Every module talks in these types.

Conventions
-----------
* Time `t` is unix epoch seconds (float).
* Space is the OpenVR *standing* universe: metres, right-handed, +Y up,
  +X right, -Z forward. three.js uses the same convention, so no conversion.
* Quaternions are (w, x, y, z).
* Devices are identified by serial number ("LHR-29C03CCA", "LHB-4F091D9A").
  Log lines name base stations by the bare hex id ("B3316995"); helpers in
  `lighthousedb` map between the two.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum, IntEnum
from typing import Any, Protocol

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]


class DeviceClass(str, Enum):
    HMD = "hmd"
    CONTROLLER = "controller"
    TRACKER = "tracker"
    BASE = "base"
    OTHER = "other"


class TrackingResult(IntEnum):
    """Mirrors openvr ETrackingResult."""
    UNINITIALIZED = 1
    CALIBRATING_IN_PROGRESS = 100
    CALIBRATING_OUT_OF_RANGE = 101
    RUNNING_OK = 200
    RUNNING_OUT_OF_RANGE = 201
    FALLBACK_ROTATION_ONLY = 300


@dataclass(slots=True)
class Sample:
    """One pose reading for one device."""
    t: float
    serial: str
    connected: bool
    valid: bool
    result: int                       # TrackingResult value
    pos: Vec3 = (0.0, 0.0, 0.0)
    quat: Quat = (1.0, 0.0, 0.0, 0.0)
    vel: Vec3 = (0.0, 0.0, 0.0)       # m/s
    angvel: Vec3 = (0.0, 0.0, 0.0)    # rad/s

    @property
    def ok(self) -> bool:
        return self.connected and self.valid and self.result == TrackingResult.RUNNING_OK


@dataclass(slots=True)
class Frame:
    """All devices sampled at one instant."""
    t: float
    samples: list[Sample]


@dataclass
class DeviceInfo:
    serial: str
    device_class: DeviceClass
    model: str = ""
    role: str = ""                    # "waist", "left_foot", "right_foot", "left_hand", "right_hand", "head", ""
    index: int = -1                   # OpenVR device index, -1 if n/a
    connected: bool = True
    battery: float | None = None      # 0..1
    charging: bool | None = None
    dongle: str | None = None         # receiver serial this device is paired through
    firmware: str = ""
    channel: int | None = None        # base stations only: lighthouse channel (the "S-n" in logs)

    def label(self) -> str:
        return self.role or self.model or self.serial


class LogKind(str, Enum):
    """Typed vrserver.txt events. `data` keys are listed per kind."""
    SOB_ADD = "sob_add"                    # base: str(S-n), seeing: list[{"slot": "S-n", "base": "B33.."|None}]
    SOB_DROP = "sob_drop"                  # same keys as SOB_ADD
    TRACKING_RESET = "tracking_reset"      # reason: "no_optical"|"imu_misalignment"|"other", ms: int|None, text: str
    IMU_OFF_SCALE = "imu_off_scale"        # {}
    BACKFACING_HITS = "backfacing_hits"    # rejected: int, backfacing: int, non_clustered: int
    TILT_UPDATE = "tilt_update"            # base: str, delta_deg: float
    UNIVERSE_TILT = "universe_tilt"        # base: str, pitch: float, roll: float
    BASE_CALIBRATED = "base_calibrated"    # base: str, pitch: float, roll: float
    BASE_BOOTSTRAPPED = "base_bootstrapped"  # base: str, distance: float, velocity: float, mode: "immediate"|"delayed"
    BASE_SECONDARY = "base_secondary"      # base: str, distance: float
    BASE_RELATIONSHIP = "base_relationship"  # a: str, b: str, distance: float, angle: float
    BASE_ADD_FAILED = "base_add_failed"    # base: str, reason: str
    RECEIVER_CONNECTED = "receiver_connected"        # dongle: str
    RECEIVER_DISCONNECTED = "receiver_disconnected"  # dongle: str
    WIRELESS_DISCONNECTED = "wireless_disconnected"  # dongle: str, device: str|None
    HID_ERROR = "hid_error"                # interface: "IMU"|"Optical"|"VrController"|str
    DONGLE_ENUMERATED = "dongle_enumerated"  # dongle: str, kind: "Watchman Dongle"|"Valve VR Radio"
    DEVICE_STANDBY = "device_standby"      # {}
    IMU_CALIBRATION = "imu_calibration"    # accel_bias: float
    SYNC_ASSERT = "sync_assert"            # text: str
    UNIVERSE_CHANGED = "universe_changed"  # universe: str
    SERVER_START = "server_start"          # version: str
    OTHER = "other"                        # unparsed lighthouse line, kept raw


@dataclass
class LogEvent:
    t: float
    kind: LogKind
    serial: str | None = None         # the device the line is about (LHR-…), else None
    data: dict[str, Any] = field(default_factory=dict)
    raw: str = ""
    level: str = "Info"               # Info | Warning | Error


class IncidentKind(str, Enum):
    DROPOUT = "dropout"               # pose invalid / not Running_OK while connected
    DISCONNECT = "disconnect"         # device reports not connected
    JUMP = "jump"                     # position discontinuity while "valid"
    JITTER = "jitter"                 # sustained high-frequency noise
    STALE = "stale"                   # pose update rate collapsed (repeated poses)
    BASE_SHIFT = "base_shift"         # a base station's pose moved
    UNIVERSE_SHIFT = "universe_shift" # every device jumped together


class Cause(str, Enum):
    OCCLUSION = "occlusion"
    RADIO = "radio"
    BASE_STATION = "base_station"
    REFLECTION = "reflection"
    IMU = "imu"
    USB = "usb"
    BATTERY = "battery"
    CALIBRATION = "calibration"       # universe / tilt / room setup changed
    UNKNOWN = "unknown"


@dataclass
class Incident:
    id: int
    kind: IncidentKind
    serial: str
    t_start: float
    t_end: float
    pos: Vec3                         # where the device was when it started
    severity: float = 0.0             # 0..1
    metrics: dict[str, float] = field(default_factory=dict)   # kind-specific, e.g. {"jump_m": 0.12}
    causes: dict[str, float] = field(default_factory=dict)    # Cause value -> confidence 0..1 (filled by attribution)
    evidence: list[str] = field(default_factory=list)         # human-readable reasons (filled by attribution)

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start

    @property
    def top_cause(self) -> str:
        if not self.causes:
            return Cause.UNKNOWN.value
        return max(self.causes.items(), key=lambda kv: kv[1])[0]


@dataclass
class Finding:
    """A session-level conclusion shown in the diagnosis panel."""
    cause: Cause
    score: float                      # 0..1, used for ranking
    title: str                        # one line, plain language
    detail: str                       # what was observed
    fix: str                          # what to try
    devices: list[str] = field(default_factory=list)   # serials
    bases: list[str] = field(default_factory=list)     # serials
    incident_count: int = 0


@dataclass
class DeviceMetrics:
    """Rolling per-device health numbers (last few seconds)."""
    update_hz: float = 0.0            # distinct poses per second
    jitter_mm: float = 0.0            # high-frequency position residual RMS
    jitter_deg: float = 0.0           # high-frequency rotation residual RMS
    ok_frac: float = 1.0              # fraction of recent samples with Sample.ok
    speed: float = 0.0                # m/s


class Source(Protocol):
    """A stream of pose frames. Implemented by openvr, replay and sim sources."""

    def read(self) -> Frame | None:
        """Block until the next frame (the source does its own pacing).

        Returns None when not connected / no data yet; the source sleeps
        briefly itself in that case so callers can loop without spinning.
        Raises StopIteration when a finite source (replay, bounded sim) ends.
        """
        ...

    def devices(self) -> dict[str, DeviceInfo]:
        """Current device registry keyed by serial (includes base stations)."""
        ...

    def drain_log(self) -> list[LogEvent]:
        """Log events produced by the source itself (sim/replay). Live returns []."""
        ...

    def close(self) -> None: ...


def to_json(obj: Any) -> Any:
    """Dataclass -> plain JSON-able structure (enums become their values)."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: to_json(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {(k.value if isinstance(k, Enum) else k): to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json(v) for v in obj]
    return obj
