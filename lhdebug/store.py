"""SQLite session recording. One file per session under sessions/."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from .model import (DeviceClass, DeviceInfo, Frame, Incident, IncidentKind, LogEvent,
                    LogKind, Sample, to_json)

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS devices (serial TEXT PRIMARY KEY, info TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS samples (
    t REAL NOT NULL, serial TEXT NOT NULL,
    connected INTEGER, valid INTEGER, result INTEGER,
    px REAL, py REAL, pz REAL, qw REAL, qx REAL, qy REAL, qz REAL,
    vx REAL, vy REAL, vz REAL, wx REAL, wy REAL, wz REAL,
    burst INTEGER NOT NULL DEFAULT 0   -- 0 = 10 Hz baseline, 1 = full-rate burst around an incident
);
CREATE INDEX IF NOT EXISTS samples_t ON samples (t);
CREATE TABLE IF NOT EXISTS incidents (
    id INTEGER PRIMARY KEY, kind TEXT, serial TEXT, t_start REAL, t_end REAL, body TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS log_events (
    t REAL NOT NULL, kind TEXT, serial TEXT, level TEXT, data TEXT, raw TEXT
);
CREATE INDEX IF NOT EXISTS log_t ON log_events (t);
"""

_SAMPLE_COLS = ("t, serial, connected, valid, result, px, py, pz, qw, qx, qy, qz, "
                "vx, vy, vz, wx, wy, wz, burst")


def _row(s: Sample, burst: int) -> tuple:
    return (s.t, s.serial, int(s.connected), int(s.valid), int(s.result),
            *s.pos, *s.quat, *s.vel, *s.angvel, burst)


def _sample(r: tuple) -> Sample:
    return Sample(t=r[0], serial=r[1], connected=bool(r[2]), valid=bool(r[3]), result=r[4],
                  pos=tuple(r[5:8]), quat=tuple(r[8:12]), vel=tuple(r[12:15]), angvel=tuple(r[15:18]))


class Store:
    def __init__(self, path: str | Path, read_only: bool = False) -> None:
        self.path = Path(path)
        if not read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        uri = f"file:{self.path.as_posix()}" + ("?mode=ro" if read_only else "")
        self.db = sqlite3.connect(uri, uri=True, check_same_thread=False)
        if not read_only:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=NORMAL")
            self.db.executescript(SCHEMA)
        self._pending = 0

    @classmethod
    def new_session(cls, root: str | Path = "sessions", mode: str = "live") -> "Store":
        name = time.strftime("%Y%m%d-%H%M%S") + f"-{mode}.sqlite"
        store = cls(Path(root) / name)
        store.set_meta("mode", mode)
        store.set_meta("started", str(time.time()))
        return store

    # ---- writes --------------------------------------------------------
    def set_meta(self, key: str, value: str) -> None:
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, value))
        self.db.commit()

    def put_devices(self, devices: dict[str, DeviceInfo]) -> None:
        self.db.executemany("INSERT OR REPLACE INTO devices VALUES (?, ?)",
                            [(s, json.dumps(to_json(d))) for s, d in devices.items()])
        self._bump(len(devices))

    def put_samples(self, samples: list[Sample], burst: bool = False) -> None:
        self.db.executemany(f"INSERT INTO samples ({_SAMPLE_COLS}) VALUES ({','.join('?' * 19)})",
                            [_row(s, int(burst)) for s in samples])
        self._bump(len(samples))

    def put_incident(self, inc: Incident) -> None:
        self.db.execute("INSERT OR REPLACE INTO incidents VALUES (?, ?, ?, ?, ?, ?)",
                        (inc.id, inc.kind.value, inc.serial, inc.t_start, inc.t_end,
                         json.dumps(to_json(inc))))
        self._bump(1)

    def put_log(self, ev: LogEvent) -> None:
        self.db.execute("INSERT INTO log_events VALUES (?, ?, ?, ?, ?, ?)",
                        (ev.t, ev.kind.value, ev.serial, ev.level, json.dumps(ev.data), ev.raw))
        self._bump(1)

    def prune(self, cutoff: float) -> None:
        """Drop everything older than `cutoff` (rolling window)."""
        self.db.execute("DELETE FROM samples WHERE t < ?", (cutoff,))
        self.db.execute("DELETE FROM incidents WHERE t_end < ?", (cutoff,))
        self.db.execute("DELETE FROM log_events WHERE t < ?", (cutoff,))
        self.flush()

    def clear(self) -> None:
        for table in ("samples", "incidents", "log_events"):
            self.db.execute(f"DELETE FROM {table}")
        self.flush()

    def _bump(self, n: int) -> None:
        self._pending += n
        if self._pending >= 500:
            self.flush()

    def flush(self) -> None:
        self.db.commit()
        self._pending = 0

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self.db.close()

    # ---- reads ---------------------------------------------------------
    def meta(self) -> dict[str, str]:
        return dict(self.db.execute("SELECT key, value FROM meta"))

    def devices(self) -> dict[str, DeviceInfo]:
        out = {}
        for serial, body in self.db.execute("SELECT serial, info FROM devices"):
            d = json.loads(body)
            d["device_class"] = DeviceClass(d["device_class"])
            out[serial] = DeviceInfo(**d)
        return out

    def frames(self, burst: bool | None = False):
        """Yield recorded Frames in time order (baseline rate by default)."""
        where = "" if burst is None else f"WHERE burst = {int(burst)}"
        cur = self.db.execute(f"SELECT {_SAMPLE_COLS} FROM samples {where} ORDER BY t")
        t, batch = None, []
        for r in cur:
            if t is not None and r[0] != t:
                yield Frame(t, batch)
                batch = []
            t = r[0]
            batch.append(_sample(r))
        if batch:
            yield Frame(t, batch)

    def incidents(self) -> list[Incident]:
        out = []
        for (body,) in self.db.execute("SELECT body FROM incidents ORDER BY t_start"):
            d = json.loads(body)
            d["kind"] = IncidentKind(d["kind"])
            d["pos"] = tuple(d["pos"])
            out.append(Incident(**d))
        return out

    def log_events(self) -> list[LogEvent]:
        return [LogEvent(t=t, kind=LogKind(kind), serial=serial, level=level,
                         data=json.loads(data), raw=raw)
                for t, kind, serial, level, data, raw in
                self.db.execute("SELECT t, kind, serial, level, data, raw FROM log_events ORDER BY t")]
