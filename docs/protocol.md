# Module and wire contracts

Everything here is a contract between independently written parts. Types are in
`lhdebug/model.py`; rolling context is `lhdebug/world.py`.

## Python interfaces

### `lhdebug/logtail.py`

```python
def parse_line(line: str) -> LogEvent | None
    # One vrserver.txt line -> event. None for lines that are not from the
    # lighthouse driver or are pure noise (monitor enumeration, keepalives,
    # HID open attempts). Unrecognised lighthouse lines -> LogKind.OTHER.

class LogTailer:
    def __init__(self, path: Path | None = None, from_start: bool = False)
        # path defaults to <Steam>/logs/vrserver.txt (find via registry
        # HKCU\Software\Valve\Steam SteamPath, fallback Program Files (x86)).
    def drain(self) -> list[LogEvent]
        # Non-blocking. New events since the last call. Must survive the file
        # being rotated/truncated when SteamVR restarts, and not existing yet.
```

### `lhdebug/lighthousedb.py`

```python
def base_id(serial: str) -> str          # "LHB-4F091D9A" -> "4F091D9A"; idempotent
def load(path: Path | None = None) -> LighthouseDB
class LighthouseDB:
    bases: dict[str, BaseRecord]         # keyed by LHB serial
    universes: list[UniverseRecord]      # id, base serials, per-base pose, tilt
    def summary(self) -> dict            # JSON-able, for GET /api/lighthousedb
```

### `lhdebug/detectors.py`

```python
class DetectorBank:
    def __init__(self, world: World, config: DetectorConfig | None = None)
    def feed(self, frame: Frame) -> list[Incident]
        # Called once per frame AFTER world.add_frame(frame). Returns incidents
        # that closed on this frame (ids assigned, causes empty).
    def metrics(self, serial: str) -> DeviceMetrics
    def open_incidents(self) -> list[Incident]     # currently ongoing
    def flush(self) -> list[Incident]              # close everything (shutdown)
```

### `lhdebug/heatmap.py`

```python
class Heatmap:
    def __init__(self, voxel: float = 0.2)
    def add(self, sample: Sample, metrics: DeviceMetrics) -> None   # moving devices only
    def add_incident(self, inc: Incident) -> None
    def snapshot(self, metric: str, serial: str | None = None) -> dict
        # metric: "samples" | "dropout" | "jitter" | "jumps" | "update_rate"
    def coverage(self, bases: list[tuple[Vec3, Quat]], bounds=None) -> dict
        # predicted number of bases with line of sight per voxel
# Both return: {"voxel": 0.2, "metric": str, "min": float, "max": float,
#               "cells": [[ix, iy, iz, value, n], ...]}
# Cell centre = ((ix + .5) * voxel, (iy + .5) * voxel, (iz + .5) * voxel).
```

### `lhdebug/attribution.py`

```python
class Attributor:
    def __init__(self, world: World)
    def attribute(self, inc: Incident) -> Incident   # fills .causes and .evidence
    def findings(self) -> list[Finding]              # ranked session summary
```

### Engine order per frame (`lhdebug/sampler.py`)

```
frame = source.read()
world.set_devices(source.devices())
for ev in log_source.drain(): world.add_log(ev); store; broadcast
world.add_frame(frame)
for inc in detectors.feed(frame): attributor.attribute(inc); heatmap.add_incident(inc); store; broadcast
for s in frame.samples (non-base): heatmap.add(s, detectors.metrics(s.serial))
```

## HTTP

Server listens on `http://127.0.0.1:8777`. Static dashboard at `/`.

| Route | Returns |
|---|---|
| `GET /api/session` | `{"id", "mode": "live"\|"sim"\|"replay", "started", "steamvr": bool, "voxel"}` |
| `GET /api/heatmap?metric=&serial=` | heatmap snapshot (shape above); `metric=coverage` for predicted coverage |
| `GET /api/incidents?limit=200` | `{"incidents": [Incident…]}` newest first |
| `GET /api/log?limit=300` | `{"events": [LogEvent…]}` newest first |
| `GET /api/findings` | `{"findings": [Finding…]}` ranked |
| `GET /api/lighthousedb` | `LighthouseDB.summary()` |
| `GET /api/sessions` | `{"sessions": [{"id", "mode", "started", "size"}]}` |

## WebSocket `/ws`

Server → client JSON messages, each with a `type`:

```jsonc
{"type": "hello", "session": {/* same as /api/session */}}

// 10 Hz
{"type": "state", "t": 1795511961.12, "steamvr": true,
 "devices": [{
   "serial": "LHR-29C03CCA", "class": "tracker", "model": "VIVE Tracker 3.0 MV",
   "role": "waist", "label": "waist",
   "connected": true, "valid": true, "result": 200, "ok": true,
   "pos": [x, y, z], "quat": [w, x, y, z],
   "battery": 0.82, "charging": false, "dongle": "40E4AC9C61",
   "update_hz": 248.0, "jitter_mm": 0.4, "jitter_deg": 0.05, "ok_frac": 1.0, "speed": 0.3,
   "bases_seen": ["S-1", "S-5"],
   "incident": "dropout"            // kind of the currently open incident, or null
 }],
 "bases": [{"serial": "LHB-4F091D9A", "channel": 1, "pos": [x, y, z], "quat": [w, x, y, z],
            "connected": true, "valid": true}]}

{"type": "incident", "incident": {/* Incident, with causes + evidence */}}
{"type": "log", "event": {/* LogEvent */}}
{"type": "findings", "findings": [/* Finding… */]}      // every ~5 s
```

Enums are sent as their string values. Incident `causes` is `{cause: confidence}`.
