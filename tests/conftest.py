import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from harness import run  # noqa: E402
from lhdebug.sources.sim_source import (BASES, LEFT_FOOT, RIGHT_FOOT, RIGHT_HAND,  # noqa: E402
                                        WAIST, Fault)

SCENARIOS = {
    "clean": dict(duration=60),
    "fast": dict(duration=30, motion="fast"),
    "occlusion": dict(duration=25, faults=[Fault("occlusion", 5, 9, serial=LEFT_FOOT)]),
    "radio": dict(duration=25, faults=[Fault("radio", 5, 10, serial=LEFT_FOOT,
                                             params={"whole_dongle": True})]),
    "radio_single": dict(duration=25, faults=[Fault("radio", 5, 10, serial=WAIST)]),
    "stale_frozen": dict(duration=20, faults=[Fault("radio", 5, 9, serial=WAIST,
                                                    params={"disconnect": False, "variant": "frozen"})]),
    "stale_extrapolate": dict(duration=20, faults=[Fault("radio", 5, 9, serial=WAIST,
                                                         params={"disconnect": False,
                                                                 "variant": "extrapolate"})]),
    "base_shift": dict(duration=25, faults=[Fault("base_shift", 5, 8, base=BASES[1])]),
    "reflection": dict(duration=60, faults=[Fault("reflection", 2, 58, serial=RIGHT_FOOT,
                                                  region=(0.7, -0.5, 0.8))]),
    "imu": dict(duration=25, faults=[Fault("imu", 5, 12, serial=RIGHT_HAND)]),
    "usb": dict(duration=20, faults=[Fault("usb", 5, 6.5)]),
    "battery": dict(duration=60, faults=[Fault("battery", 3, 55, serial=WAIST)]),
    "universe": dict(duration=20, faults=[Fault("universe", 5, 6)]),
    "demo": dict(duration=300, faults="demo"),
}

_cache: dict = {}


@pytest.fixture
def scenario():
    """scenario(name) -> harness.Run, simulated once per test session."""
    def get(name: str):
        if name not in _cache:
            _cache[name] = run(**SCENARIOS[name])
        return _cache[name]
    return get
