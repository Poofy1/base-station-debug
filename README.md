# lhdebug — Lighthouse tracking debugger

Runs in the background while you are in VR and works out *why* SteamVR Lighthouse
(Base Station 2.0) tracking is bad: occlusion, a bumped or failing base station,
reflections, a weak dongle/radio link, USB trouble, IMU faults or low battery.

It fuses two sources:

- **Pose sampling** through OpenVR at 200 Hz (headset, controllers, trackers, base stations).
- **The lighthouse driver log** (`vrserver.txt`), which says what the driver itself saw go wrong.

and shows the result in a local web dashboard: a 3D room with a voxel heatmap,
per-device health, a timeline of incidents, and a ranked diagnosis with suggested fixes.

## Quick start (Windows)

1. Install [Python 3.11+](https://www.python.org/downloads/) (tick *Add python.exe to PATH*).
2. Double-click **`run.bat`**.

The first launch creates `.venv` and installs dependencies (needs internet once).
It then waits for SteamVR — it never launches SteamVR itself — and opens the dashboard at
<http://127.0.0.1:8777/>. Leave it running while you play; view the dashboard on the
desktop or in-headset through any desktop-window overlay.

| Command | What it does |
|---|---|
| `run.bat` | Attach to SteamVR, record, open the dashboard |
| `run.bat simulate` | Demo with a simulated rig and injected faults (no headset needed) |
| `run.bat report` | Print incidents and findings of the latest recorded session |
| `run.bat sessions` | List recorded sessions |
| `.venv\Scripts\python -m lhdebug.logtail` | Summarise the current SteamVR driver log on its own |

Options (after the command): `--port 8777`, `--rate 200`, `--open`, `--no-record` (run), `--clean` / `--record` (simulate).

Recordings go to `sessions/` as SQLite files: 10 Hz for the whole session plus
full-rate bursts around each incident. Expect tens of MB per hour, more when many
incidents fire.

## What it detects

| Incident | Signal |
|---|---|
| dropout | pose invalid / out of range while still connected |
| disconnect | device reports not connected |
| jump | position step not explained by the reported velocity |
| jitter | high-frequency noise in position or rotation |
| stale | pose update rate collapses (radio packet loss) |
| base_shift | a base station's pose moves |
| universe_shift | every device jumps together |

Each incident is attributed to a cause with a confidence and the evidence behind it.
When the evidence is thin the cause is `unknown` rather than a guess.

## Layout

```
lhdebug/
  model.py, world.py      shared types and rolling context
  sources/                openvr_source (live), sim_source (synthetic rig + faults)
  logtail.py              vrserver.txt parser and tailer
  lighthousedb.py         reader for SteamVR's lighthousedb.json
  detectors.py            incident detection and per-device metrics
  attribution.py          evidence scoring and session findings
  heatmap.py              voxel accumulators and predicted base coverage
  sampler.py              the engine tying it together
  store.py                SQLite session recording
  server.py               HTTP + WebSocket API (see docs/protocol.md)
web/                      static dashboard (three.js vendored, no build step)
tests/                    pytest suite
```

## Development

```
.venv\Scripts\python -m pip install pytest
.venv\Scripts\python -m pytest
```

`web/?mock=1` runs the dashboard on browser-generated data with no backend.

## Status and limits

- The attribution rules are validated against the simulator and against recorded
  driver logs, not yet against live tracking faults. Treat findings as leads.
- The log parser's patterns come from SteamVR 2.18 logs; unknown lines are kept as
  raw events rather than dropped.
- `tests/fixtures/` contains excerpts of real driver logs, including device serial
  numbers.
- Not built yet: guided test procedure, dashboard replay of recorded sessions,
  auto-start with SteamVR, Bluetooth base-station isolation test.
