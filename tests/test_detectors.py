import math

import pytest

from harness import T0, drive, run
from lhdebug.detectors import UNIVERSE_SERIAL, DetectorBank, DetectorConfig
from lhdebug.model import (DeviceClass, DeviceInfo, Frame, IncidentKind as K, Sample)
from lhdebug.sources.sim_source import (BASES, HMD, LEFT_FOOT, LEFT_HAND, RIGHT_FOOT,
                                        RIGHT_HAND, WAIST, Fault)
from lhdebug.world import World

TRACKED = (HMD, LEFT_HAND, RIGHT_HAND, WAIST, LEFT_FOOT, RIGHT_FOOT)


def rel(inc):
    return inc.t_start - T0, inc.t_end - T0


# ---- healthy sessions --------------------------------------------------------
def test_clean_minute_has_no_incidents(scenario):
    r = scenario("clean")
    assert r.frames == 12000
    assert r.incidents == []
    assert r.bank.open_incidents() == []
    for serial in TRACKED:
        m = r.bank.metrics(serial)
        assert m.update_hz == pytest.approx(200.0, abs=2.0)
        assert 0.3 < m.jitter_mm < 1.0            # ~0.3 mm/axis sensor noise, 3D RMS
        assert m.jitter_deg < 0.1
        assert m.ok_frac > 0.999
        assert r.max_jitter_mm[serial] < 1.0


def test_fast_healthy_motion_triggers_nothing(scenario):
    r = scenario("fast")
    hand = [s for s in r.world.recent(RIGHT_HAND)]
    assert max(math.dist((0, 0, 0), s.vel) for s in hand) > 4.0      # genuinely fast
    assert r.of(K.JITTER) == [] and r.of(K.JUMP) == []
    assert r.incidents == []
    # the jitter estimate does not rise with speed
    assert r.max_jitter_mm[RIGHT_HAND] < 1.0
    assert r.bank.metrics(RIGHT_HAND).speed > 0.5


# ---- one incident per physical event -----------------------------------------
def test_dropout(scenario):
    r = scenario("occlusion")
    assert [(i.kind, i.serial) for i in r.incidents] == [(K.DROPOUT, LEFT_FOOT)]
    inc = r.incidents[0]
    t0, t1 = rel(inc)
    assert 5.3 < t0 < 5.6 and 8.9 < t1 < 9.01
    assert inc.id > 0 and 0 < inc.severity <= 1
    assert inc.metrics["lost_s"] == pytest.approx(inc.duration, abs=0.05)
    assert inc.pos[1] < 0.4                       # where the foot was


def test_stale_then_disconnect_on_shared_dongle(scenario):
    r = scenario("radio")
    for serial in (LEFT_FOOT, RIGHT_FOOT):
        assert [i.kind for i in r.of(serial=serial)] == [K.STALE, K.DISCONNECT]
        stale, disc = r.of(serial=serial)
        assert 5.5 < rel(stale)[0] < 7.0 and rel(stale)[1] < 7.01
        assert stale.metrics["update_hz"] < 60 and stale.metrics["expected_hz"] == pytest.approx(200, abs=1)
        assert rel(disc)[0] == pytest.approx(7.0, abs=0.02)
        assert rel(disc)[1] == pytest.approx(10.0, abs=0.02)
    assert len(r.incidents) == 4


@pytest.mark.parametrize("name", ["stale_frozen", "stale_extrapolate"])
def test_stale_detected_for_frozen_and_extrapolated_poses(scenario, name):
    r = scenario(name)
    assert [(i.kind, i.serial) for i in r.incidents] == [(K.STALE, WAIST)]
    inc = r.incidents[0]
    # (the 1 s rate window makes the end lag the fault's by a fraction of a second)
    assert 5.2 < rel(inc)[0] < 6.5 and 8.8 < rel(inc)[1] < 9.5
    assert inc.metrics["update_hz"] < 20


def _feed_synthetic(extrapolate: bool, vel_noise: bool):
    """A device moving at constant velocity, sampled at 200 Hz, whose driver
    only delivers an update every 10th sample after t=3 s."""
    world = World()
    world.set_devices({"LHR-X": DeviceInfo("LHR-X", DeviceClass.TRACKER, role="waist")})
    bank = DetectorBank(world)
    out = []
    upd = None
    for n in range(1600):
        t = T0 + n / 200.0
        fresh = n < 600 or n % 10 == 0
        if fresh:
            v = (0.3 + (1e-4 * (n % 7) if vel_noise else 0.0), 0.0, 0.1)
            upd = Sample(t, "LHR-X", True, True, 200, (0.3 * n / 200.0, 1.0, 0.1 * n / 200.0),
                         (1.0, 0.0, 0.0, 0.0), v, (0.0, 0.0, 0.01 * (n % 5)))
            s = upd
        elif extrapolate:
            h = t - upd.t
            s = Sample(t, "LHR-X", True, True, 200,
                       tuple(p + v * h for p, v in zip(upd.pos, upd.vel)), upd.quat, upd.vel, upd.angvel)
        else:
            s = Sample(t, "LHR-X", True, True, 200, upd.pos, upd.quat, upd.vel, upd.angvel)
        frame = Frame(t, [s])
        world.add_frame(frame)
        out += bank.feed(frame)
    return bank, out + bank.flush()


@pytest.mark.parametrize("extrapolate", [False, True])
def test_update_rate_counts_velocity_changes_not_positions(extrapolate):
    bank, incs = _feed_synthetic(extrapolate, vel_noise=True)
    assert bank.metrics("LHR-X").update_hz == pytest.approx(20.0, abs=1.0)
    assert [i.kind for i in incs] == [K.STALE]
    assert incs[0].t_start - T0 == pytest.approx(3.0 + 0.72, abs=0.15)


def test_constant_velocity_source_without_velocity_noise_is_not_stale():
    # velocity never changes but positions advance in a way extrapolation from
    # the previous sample does not explain (they include their own offset)
    world = World()
    world.set_devices({"LHR-X": DeviceInfo("LHR-X", DeviceClass.TRACKER)})
    bank = DetectorBank(world)
    out = []
    for n in range(800):
        t = T0 + n / 200.0
        wob = 0.001 * math.sin(n * 0.7)
        f = Frame(t, [Sample(t, "LHR-X", True, True, 200, (wob, 1.0, 0.0), (1.0, 0.0, 0.0, 0.0))])
        world.add_frame(f)
        out += bank.feed(f)
    assert out == [] and bank.metrics("LHR-X").update_hz == pytest.approx(200, abs=8)


def test_jump_incidents_and_no_jitter_from_jumps(scenario):
    r = scenario("reflection")
    assert r.incidents and all(i.kind is K.JUMP and i.serial == RIGHT_FOOT for i in r.incidents)
    assert 2 <= len(r.incidents) <= 8                      # clustered per visit, not per jump
    for i in r.incidents:
        assert 0.015 < i.metrics["jump_m"] < 0.07
        assert i.metrics["count"] >= 1
    assert sum(i.metrics["count"] for i in r.incidents) > len(r.incidents)


def test_jitter(scenario):
    r = scenario("imu")
    assert [(i.kind, i.serial) for i in r.of(K.JITTER)] == [(K.JITTER, RIGHT_HAND)]
    inc = r.of(K.JITTER)[0]
    assert 5.0 < rel(inc)[0] < 5.5 and 11.0 < rel(inc)[1] < 13.5
    assert inc.metrics["jitter_mm"] > 2.0 and inc.metrics["jitter_deg"] > 0.5
    assert all(i.serial == RIGHT_HAND for i in r.incidents)


def test_base_shift_reported_once_for_the_base(scenario):
    r = scenario("base_shift")
    shifts = r.of(K.BASE_SHIFT)
    assert [i.serial for i in shifts] == [BASES[1]]
    assert shifts[0].metrics["shift_m"] == pytest.approx(0.03, abs=1e-3)
    assert shifts[0].metrics["shift_deg"] == pytest.approx(1.5, abs=0.05)
    assert rel(shifts[0])[0] == pytest.approx(8.0, abs=0.02)
    assert r.of(K.UNIVERSE_SHIFT) == []
    jumps = r.of(K.JUMP)
    assert len({i.serial for i in jumps}) >= 3             # devices relying on it jumped
    assert all(4.9 < rel(i)[0] < 8.1 for i in jumps)


def test_disconnect_all_devices(scenario):
    r = scenario("usb")
    assert sorted(i.serial for i in r.incidents) == sorted(TRACKED)
    assert all(i.kind is K.DISCONNECT for i in r.incidents)
    assert len({i.t_start for i in r.incidents}) == 1


def test_universe_shift_reported_once_not_per_device(scenario):
    r = scenario("universe")
    assert [(i.kind, i.serial) for i in r.incidents] == [(K.UNIVERSE_SHIFT, UNIVERSE_SERIAL)]
    inc = r.incidents[0]
    assert rel(inc)[0] == pytest.approx(5.0, abs=0.01)
    assert inc.metrics["shift_m"] == pytest.approx(math.hypot(0.06, 0.04), abs=1e-3)


def test_universe_shift_without_base_samples():
    """Sources with no base poses: all devices jumping by one vector is one incident."""
    world = World()
    names = [f"LHR-{i}" for i in range(4)]
    world.set_devices({n: DeviceInfo(n, DeviceClass.TRACKER) for n in names})
    bank = DetectorBank(world)
    out = []
    for n in range(1200):
        t = T0 + n / 200.0
        off = 0.05 if n >= 600 else 0.0
        samples = []
        for k, name in enumerate(names):
            ph = n * 0.01 + k
            samples.append(Sample(t, name, True, True, 200, (k + off + 0.1 * math.sin(ph), 1.0, 0.0),
                                  (1.0, 0.0, 0.0, 0.0), (0.1 * 2.0 * math.cos(ph), 0.0, 0.0),
                                  (0.0, 0.0, 1e-3 * (n % 3))))
        f = Frame(t, samples)
        world.add_frame(f)
        out += bank.feed(f)
    out += bank.flush()
    assert [(i.kind, i.serial) for i in out] == [(K.UNIVERSE_SHIFT, UNIVERSE_SERIAL)]
    assert out[0].metrics["devices"] == 4


def test_single_device_jump_is_not_a_universe_shift():
    world = World()
    names = [f"LHR-{i}" for i in range(4)]
    world.set_devices({n: DeviceInfo(n, DeviceClass.TRACKER) for n in names})
    bank = DetectorBank(world)
    out = []
    for n in range(1200):
        t = T0 + n / 200.0
        samples = [Sample(t, name, True, True, 200,
                          (k + (0.05 if n >= 600 and k == 2 else 0.0), 1.0, 0.0),
                          (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1e-3 * (n % 3)))
                   for k, name in enumerate(names)]
        f = Frame(t, samples)
        world.add_frame(f)
        out += bank.feed(f)
    out += bank.flush()
    assert [(i.kind, i.serial) for i in out] == [(K.JUMP, "LHR-2")]
    assert out[0].metrics["jump_m"] == pytest.approx(0.05, abs=1e-3)


# ---- hysteresis and lifecycle --------------------------------------------------
def _blips(pattern, config=None):
    """pattern: function n -> ok? for one device at 200 Hz."""
    world = World()
    world.set_devices({"LHR-X": DeviceInfo("LHR-X", DeviceClass.TRACKER)})
    bank = DetectorBank(world, config)
    out, opened = [], []
    for n in range(2000):
        t = T0 + n / 200.0
        ok = pattern(n)
        s = Sample(t, "LHR-X", True, ok, 200 if ok else 201, (0.0, 1.0, 0.0), (1.0, 0.0, 0.0, 0.0),
                   (0.0, 0.0, 1e-4 * (n % 5)), (0.0, 0.0, 0.0))
        f = Frame(t, [s])
        world.add_frame(f)
        out += bank.feed(f)
        opened.append(len(bank.open_incidents()))
    return bank, out, opened


def test_flickering_dropout_is_one_incident():
    # 2 s of tracking flickering on and off every 100 ms
    bank, out, opened = _blips(lambda n: not (400 <= n < 800 and (n // 20) % 2 == 0))
    assert [i.kind for i in out] == [K.DROPOUT]
    assert out[0].t_start - T0 == pytest.approx(2.0, abs=0.01)
    assert out[0].t_end - T0 == pytest.approx(3.9, abs=0.02)
    assert out[0].metrics["lost_s"] == pytest.approx(1.0, abs=0.05)
    assert max(opened) == 1 and opened[-1] == 0
    assert opened[410] == 1 and opened[390] == 0           # visible as open while ongoing


def test_single_bad_frame_is_ignored():
    _, out, opened = _blips(lambda n: n != 500)
    assert out == [] and max(opened) == 0


def test_separate_events_are_separate_incidents():
    _, out, _ = _blips(lambda n: not (400 <= n < 500 or 1200 <= n < 1300))
    assert [i.kind for i in out] == [K.DROPOUT, K.DROPOUT]
    assert out[0].id != out[1].id


def test_flush_closes_open_incidents():
    bank, out, opened = _blips(lambda n: n < 1900)
    assert out == [] and opened[-1] == 1
    flushed = bank.flush()
    assert [i.kind for i in flushed] == [K.DROPOUT]
    assert bank.open_incidents() == [] and bank.flush() == []


def test_config_thresholds_are_respected():
    cfg = DetectorConfig(dropout_open_s=1.0)
    _, out, _ = _blips(lambda n: not (400 <= n < 500), cfg)       # 0.5 s outage < 1 s
    assert out == []


def test_device_never_connected_is_not_a_disconnect():
    world = World()
    world.set_devices({"LHR-X": DeviceInfo("LHR-X", DeviceClass.TRACKER)})
    bank = DetectorBank(world)
    out = []
    for n in range(600):
        t = T0 + n / 200.0
        f = Frame(t, [Sample(t, "LHR-X", False, False, 1)])
        world.add_frame(f)
        out += bank.feed(f)
    assert out + bank.flush() == []


def test_metrics_reflect_trouble(scenario):
    r = run(8.0, [Fault("occlusion", 3, 20, serial=LEFT_FOOT)])
    m = r.bank.metrics(LEFT_FOOT)
    assert m.ok_frac < 0.3 and m.update_hz == 0.0 and m.speed == 0.0
    assert r.bank.metrics(RIGHT_FOOT).ok_frac > 0.99
    assert [i.kind for i in r.incidents] == [K.DROPOUT]      # still open at the end: flushed
    assert r.bank.metrics("LHR-UNKNOWN").update_hz == 0.0
