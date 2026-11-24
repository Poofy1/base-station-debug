"""Wires SimSource -> World -> DetectorBank -> Attributor -> Heatmap in the
engine's per-frame order (docs/protocol.md) and times the analysis part."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from lhdebug.attribution import Attributor
from lhdebug.detectors import DetectorBank, DetectorConfig
from lhdebug.heatmap import Heatmap
from lhdebug.model import DeviceClass, Finding, Incident
from lhdebug.sources.sim_source import SimSource
from lhdebug.world import World

T0 = 1_795_000_000.0


@dataclass
class Run:
    source: SimSource
    world: World
    bank: DetectorBank
    attributor: Attributor
    heatmap: Heatmap
    incidents: list[Incident] = field(default_factory=list)
    frames: int = 0
    core_s: float = 0.0          # world + detectors + heatmap time
    attr_s: float = 0.0          # attribution time (only when incidents close)
    max_jitter_mm: dict[str, float] = field(default_factory=dict)

    @property
    def findings(self) -> list[Finding]:
        return self.attributor.findings()

    def of(self, kind=None, serial=None) -> list[Incident]:
        return [i for i in self.incidents
                if (kind is None or i.kind == kind) and (serial is None or i.serial == serial)]


def run(duration: float = 30.0, faults=None, config: DetectorConfig | None = None,
        seed: int = 0, motion: str = "normal", rate: float = 200.0, **kw) -> Run:
    source = SimSource(rate=rate, realtime=False, duration=duration, faults=faults,
                       seed=seed, t0=T0, motion=motion, **kw)
    return drive(source, config)


def drive(source, config: DetectorConfig | None = None) -> Run:
    world = World()
    bank = DetectorBank(world, config)
    r = Run(source, world, bank, Attributor(world), Heatmap())
    heatmap = r.heatmap
    attributor = r.attributor
    perf = time.perf_counter
    base = DeviceClass.BASE
    while True:
        try:
            frame = source.read()
        except StopIteration:
            break
        if frame is None:
            continue
        log = source.drain_log()
        t_a = perf()
        devices = source.devices()
        world.set_devices(devices)
        for ev in log:
            world.add_log(ev)
        world.add_frame(frame)
        closed = bank.feed(frame)
        if closed:
            t_b = perf()
            for inc in closed:
                attributor.attribute(inc)
            r.attr_s += perf() - t_b
            for inc in closed:
                heatmap.add_incident(inc)
                r.incidents.append(inc)
        for s in frame.samples:
            if devices[s.serial].device_class is not base:
                m = bank.metrics(s.serial)
                heatmap.add(s, m)
                if m.jitter_mm > r.max_jitter_mm.get(s.serial, 0.0):
                    r.max_jitter_mm[s.serial] = m.jitter_mm
        r.core_s += perf() - t_a
        r.frames += 1
    r.core_s -= r.attr_s
    for inc in bank.flush():
        attributor.attribute(inc)
        heatmap.add_incident(inc)
        r.incidents.append(inc)
    return r
