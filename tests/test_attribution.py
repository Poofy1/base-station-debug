import json
import time

import pytest

from harness import T0, run
from lhdebug.attribution import Attributor
from lhdebug.detectors import UNIVERSE_SERIAL
from lhdebug.model import (Cause, DeviceClass, DeviceInfo, Finding, Frame, Incident,
                           IncidentKind as K, LogEvent, LogKind, Sample, to_json)
from lhdebug.sources.sim_source import (BASES, DONGLE_FEET, HMD, LEFT_FOOT, LEFT_HAND,
                                        RIGHT_FOOT, RIGHT_HAND, WAIST, Fault)
from lhdebug.world import World

TRACKED = (HMD, LEFT_HAND, RIGHT_HAND, WAIST, LEFT_FOOT, RIGHT_FOOT)


def finding(r, cause, serial=None):
    for f in r.findings:
        if f.cause is cause and (serial is None or serial in f.devices):
            return f
    raise AssertionError(f"no {cause} finding for {serial}: {[(f.cause, f.devices) for f in r.findings]}")


def check_well_formed(r):
    for inc in r.incidents:
        assert inc.causes and inc.evidence
        assert all(0.0 <= c <= 1.0 for c in inc.causes.values())
        assert sum(inc.causes.values()) == pytest.approx(1.0, abs=0.05)
        assert set(inc.causes) <= {c.value for c in Cause}
        assert all(isinstance(e, str) and e for e in inc.evidence)
    scores = [f.score for f in r.findings]
    assert scores == sorted(scores, reverse=True)
    for f in r.findings:
        assert isinstance(f, Finding) and 0.0 <= f.score <= 1.0
        assert f.title and f.detail and f.fix and f.incident_count >= 1
        assert "LHR-" not in f.title                 # devices are named by label
    assert sum(f.incident_count for f in r.findings) == len(r.incidents)
    json.dumps(to_json(r.findings))
    json.dumps(to_json(r.incidents))


def test_clean_session_has_nothing_to_say(scenario):
    r = scenario("clean")
    assert r.incidents == [] and r.findings == []


# ---- each fault kind: right top cause and a matching finding -------------------
def test_occlusion(scenario):
    r = scenario("occlusion")
    check_well_formed(r)
    inc = r.of(K.DROPOUT, LEFT_FOOT)[0]
    assert inc.top_cause == Cause.OCCLUSION.value and inc.causes["occlusion"] > 0.7
    assert any("no base station light" in e for e in inc.evidence)
    assert any("lost sight of" in e for e in inc.evidence)
    f = finding(r, Cause.OCCLUSION, LEFT_FOOT)
    assert r.findings[0] is not None and r.findings[0].cause is Cause.OCCLUSION
    assert "left_foot" in f.title
    assert "from the centre" in f.detail or "near the centre" in f.detail      # says where
    assert "near the floor" in f.detail
    assert "base station" in f.fix


def test_radio_shared_dongle(scenario):
    r = scenario("radio")
    check_well_formed(r)
    assert len(r.incidents) == 4
    assert all(i.top_cause == Cause.RADIO.value for i in r.incidents)
    disc = r.of(K.DISCONNECT, LEFT_FOOT)[0]
    assert any("same dongle" in e and "right_foot" in e for e in disc.evidence)
    assert any("wireless connection" in e for e in disc.evidence)
    assert any(DONGLE_FEET in e for e in disc.evidence)
    f = finding(r, Cause.RADIO, LEFT_FOOT)
    assert f.devices == sorted([LEFT_FOOT, RIGHT_FOOT]) and f.incident_count == 4
    assert "left_foot" in f.title and "right_foot" in f.title and DONGLE_FEET in f.title
    assert "dongle" in f.fix
    assert len(r.findings) == 1


def test_radio_single_device(scenario):
    r = scenario("radio_single")
    check_well_formed(r)
    assert [i.kind for i in r.incidents] == [K.STALE, K.DISCONNECT]
    assert all(i.top_cause == Cause.RADIO.value for i in r.incidents)
    assert finding(r, Cause.RADIO, WAIST).devices == [WAIST]


@pytest.mark.parametrize("name", ["stale_frozen", "stale_extrapolate"])
def test_stale_link_without_any_log_lines_is_radio(scenario, name):
    r = scenario(name)
    assert [i.top_cause for i in r.incidents] == [Cause.RADIO.value]
    assert any("pose updates" in e for e in r.incidents[0].evidence)
    finding(r, Cause.RADIO, WAIST)


def test_base_shift(scenario):
    r = scenario("base_shift")
    check_well_formed(r)
    shift = r.of(K.BASE_SHIFT)[0]
    assert shift.top_cause == Cause.BASE_STATION.value and shift.causes["base_station"] > 0.8
    assert any("tilt change" in e for e in shift.evidence)
    jumps = r.of(K.JUMP)
    assert jumps and all(i.top_cause == Cause.BASE_STATION.value for i in jumps)
    f = finding(r, Cause.BASE_STATION)
    assert f.bases == [BASES[1]] and BASES[1] in f.title and "S-2" in f.title
    assert f.incident_count == len(r.incidents)
    assert "mounted" in f.fix
    assert r.findings[0] == f


def test_reflection(scenario):
    r = scenario("reflection")
    check_well_formed(r)
    assert r.incidents and all(i.top_cause == Cause.REFLECTION.value for i in r.incidents)
    assert any("back-facing" in e for e in r.incidents[0].evidence)
    assert any("same spot" in e for i in r.incidents for e in i.evidence)       # recurrence noticed
    f = finding(r, Cause.REFLECTION, RIGHT_FOOT)
    assert "right_foot" in f.title
    assert "x +" in f.detail and "z " in f.detail                                # says where in the room
    assert "mirror" in f.fix.lower()


def test_imu(scenario):
    r = scenario("imu")
    check_well_formed(r)
    assert r.of(K.JITTER) and all(i.top_cause == Cause.IMU.value for i in r.incidents)
    assert any("IMU went off scale" in e for e in r.of(K.JITTER)[0].evidence)
    f = finding(r, Cause.IMU, RIGHT_HAND)
    assert "right_hand" in f.title and "strap" in f.fix


def test_usb(scenario):
    r = scenario("usb")
    check_well_formed(r)
    assert len(r.incidents) == 6 and all(i.top_cause == Cause.USB.value for i in r.incidents)
    inc = r.of(serial=WAIST)[0]
    assert any("HID" in e for e in inc.evidence)
    assert any("5 of 5 other devices" in e for e in inc.evidence)
    f = finding(r, Cause.USB)
    assert f.devices == sorted(TRACKED) and f.incident_count == 6 and "USB" in f.fix
    assert len(r.findings) == 1


def test_battery(scenario):
    r = scenario("battery")
    check_well_formed(r)
    assert len(r.incidents) >= 8 and all(i.serial == WAIST and i.kind is K.DISCONNECT for i in r.incidents)
    top = [i.top_cause for i in r.incidents]
    assert top.count(Cause.BATTERY.value) >= 0.9 * len(top)
    assert any("battery was at" in e for e in r.incidents[0].evidence)
    assert any("as the charge fell" in e for i in r.incidents for e in i.evidence)
    f = finding(r, Cause.BATTERY, WAIST)
    assert r.findings[0] == f
    assert "waist" in f.title and "Charge waist" in f.fix
    assert "more frequent" in f.detail


def test_universe_shift_is_calibration(scenario):
    r = scenario("universe")
    check_well_formed(r)
    assert [i.top_cause for i in r.incidents] == [Cause.CALIBRATION.value]
    assert r.incidents[0].serial == UNIVERSE_SERIAL
    assert finding(r, Cause.CALIBRATION).devices == []


def test_demo_cycle_end_to_end(scenario):
    """One full demo cycle: every cause shows up as a finding for the right device."""
    r = scenario("demo")
    check_well_formed(r)
    expect = {Cause.OCCLUSION: LEFT_FOOT, Cause.RADIO: LEFT_FOOT, Cause.REFLECTION: RIGHT_FOOT,
              Cause.IMU: RIGHT_HAND, Cause.USB: HMD, Cause.BATTERY: WAIST}
    for cause, serial in expect.items():
        finding(r, cause, serial)
    assert finding(r, Cause.BASE_STATION).bases == [BASES[0]]
    unknown = [f for f in r.findings if f.cause is Cause.UNKNOWN]
    assert sum(f.incident_count for f in unknown) <= 2
    assert r.findings[-1].score <= r.findings[0].score


# ---- thin evidence -> unknown -------------------------------------------------
def _world_with(samples_fn, n=1200, devices=None):
    world = World()
    devices = devices or {
        "LHR-HEAD": DeviceInfo("LHR-HEAD", DeviceClass.HMD, role="head"),
        "LHR-W": DeviceInfo("LHR-W", DeviceClass.TRACKER, role="waist", battery=0.9, dongle="D1"),
        "LHR-F": DeviceInfo("LHR-F", DeviceClass.TRACKER, role="left_foot", battery=0.9, dongle="D2"),
    }
    world.set_devices(devices)
    for i in range(n):
        t = T0 + i / 200.0
        world.add_frame(Frame(t, samples_fn(i, t)))
    return world


def _steady(i, t, bad=None):
    out = []
    for serial, pos in (("LHR-HEAD", (0.0, 1.7, 0.0)), ("LHR-W", (0.0, 1.0, 0.0)), ("LHR-F", (0.1, 0.1, 0.0))):
        ok = not (bad and bad(serial, i))
        out.append(Sample(t, serial, True, ok, 200 if ok else 201, pos, (1.0, 0.0, 0.0, 0.0),
                          (0.0, 0.0, 1e-4 * (i % 7)), (0.0, 0.0, 0.0)))
    return out


def test_bare_jitter_with_no_evidence_is_unknown():
    world = _world_with(_steady)
    a = Attributor(world)
    inc = a.attribute(Incident(1, K.JITTER, "LHR-W", T0 + 2, T0 + 3, (0.0, 1.0, 0.0),
                               metrics={"jitter_mm": 3.0, "jitter_deg": 0.1}))
    assert inc.top_cause == Cause.UNKNOWN.value
    assert any("not enough evidence" in e for e in inc.evidence)
    f = a.findings()
    assert len(f) == 1 and f[0].cause is Cause.UNKNOWN and f[0].score < 0.3


def test_bare_disconnect_with_healthy_battery_is_unknown():
    world = _world_with(_steady)
    a = Attributor(world)
    inc = a.attribute(Incident(1, K.DISCONNECT, "LHR-W", T0 + 2, T0 + 3, (0.0, 1.0, 0.0)))
    assert inc.top_cause == Cause.UNKNOWN.value
    assert inc.causes.get("radio", 0) > 0          # still the leading hypothesis, just not asserted


def test_same_disconnect_with_low_battery_is_battery():
    devices = {
        "LHR-HEAD": DeviceInfo("LHR-HEAD", DeviceClass.HMD, role="head"),
        "LHR-W": DeviceInfo("LHR-W", DeviceClass.TRACKER, role="waist", battery=0.04, dongle="D1"),
        "LHR-F": DeviceInfo("LHR-F", DeviceClass.TRACKER, role="left_foot", battery=0.9, dongle="D2"),
    }
    a = Attributor(_world_with(_steady, devices=devices))
    inc = a.attribute(Incident(1, K.DISCONNECT, "LHR-W", T0 + 2, T0 + 3, (0.0, 1.0, 0.0)))
    assert inc.top_cause == Cause.BATTERY.value
    devices["LHR-W"].charging = True               # on charge: the level says nothing
    inc = a.attribute(Incident(2, K.DISCONNECT, "LHR-W", T0 + 4, T0 + 5, (0.0, 1.0, 0.0)))
    assert inc.top_cause == Cause.UNKNOWN.value


def test_dropout_geometry_body_blocks_only_base():
    """No log lines at all: a dropout where the body blocks the only base is
    occlusion; with a clear view the same dropout stays unknown."""
    from lhdebug.sources.sim_source import look_at

    def make(base_pos):
        devices = {
            "LHR-HEAD": DeviceInfo("LHR-HEAD", DeviceClass.HMD, role="head"),
            "LHR-W": DeviceInfo("LHR-W", DeviceClass.TRACKER, role="waist", battery=0.9),
            "LHB-1": DeviceInfo("LHB-1", DeviceClass.BASE, channel=1),
        }
        q = look_at(base_pos, (0.0, 1.0, 0.0))

        def fn(i, t):
            return [Sample(t, "LHR-HEAD", True, True, 200, (0.0, 1.7, 0.0), (1.0, 0.0, 0.0, 0.0),
                           (0.0, 0.0, 1e-4 * (i % 7))),
                    Sample(t, "LHR-W", True, True, 200, (0.0, 1.0, 0.25), (1.0, 0.0, 0.0, 0.0),
                           (0.0, 0.0, 1e-4 * (i % 7))),
                    Sample(t, "LHB-1", True, True, 200, base_pos, q)]
        return Attributor(_world_with(fn, devices=devices))

    inc = Incident(1, K.DROPOUT, "LHR-W", T0 + 2, T0 + 3, (0.0, 1.0, 0.25))
    blocked = make((0.0, 1.4, -3.0)).attribute(inc)            # base in front, tracker on the back
    assert blocked.top_cause == Cause.OCCLUSION.value
    assert any("blocked by the body" in e for e in blocked.evidence)
    inc = Incident(2, K.DROPOUT, "LHR-W", T0 + 2, T0 + 3, (0.0, 1.0, 0.25))
    clear = make((0.0, 2.2, 3.0)).attribute(inc)
    assert clear.top_cause == Cause.UNKNOWN.value
    assert not any("blocked" in e for e in clear.evidence)
    inc = Incident(3, K.DROPOUT, "LHR-W", T0 + 2, T0 + 3, (0.0, 1.0, 0.25))
    far = make((0.0, 2.2, 9.0)).attribute(inc)
    assert far.top_cause == Cause.OCCLUSION.value and any("m away" in e for e in far.evidence)


def test_co_occurrence_on_shared_dongle_vs_other_receivers():
    def bad(serial, i):
        return serial in ("LHR-W", "LHR-F") and 400 <= i < 600

    def fn(i, t):
        out = _steady(i, t)
        return [Sample(s.t, s.serial, not bad(s.serial, i), not bad(s.serial, i),
                       1 if bad(s.serial, i) else 200, s.pos, s.quat, s.vel) for s in out]

    def devs(dongle_f):
        return {
            "LHR-HEAD": DeviceInfo("LHR-HEAD", DeviceClass.HMD, role="head"),
            "LHR-W": DeviceInfo("LHR-W", DeviceClass.TRACKER, role="waist", battery=0.9, dongle="D1"),
            "LHR-F": DeviceInfo("LHR-F", DeviceClass.TRACKER, role="left_foot", battery=0.9, dongle=dongle_f),
        }
    shared = Attributor(_world_with(fn, devices=devs("D1")))
    inc = shared.attribute(Incident(1, K.DISCONNECT, "LHR-W", T0 + 2, T0 + 3, (0.0, 1.0, 0.0)))
    assert inc.top_cause == Cause.RADIO.value
    assert any("same dongle (D1)" in e and "left_foot" in e for e in inc.evidence)
    apart = Attributor(_world_with(fn, devices=devs("D2")))
    inc2 = apart.attribute(Incident(1, K.DISCONNECT, "LHR-W", T0 + 2, T0 + 3, (0.0, 1.0, 0.0)))
    assert inc2.causes["radio"] < inc.causes["radio"]
    assert not any("same dongle" in e for e in inc2.evidence)


def test_log_evidence_is_scoped_to_the_device_and_time():
    world = _world_with(_steady)
    world.add_log(LogEvent(T0 + 2.1, LogKind.IMU_OFF_SCALE, "LHR-F"))            # other device
    world.add_log(LogEvent(T0 + 2.2, LogKind.TRACKING_RESET, "LHR-F", {"reason": "no_optical", "ms": 500}))
    a = Attributor(world)
    inc = a.attribute(Incident(1, K.JITTER, "LHR-W", T0 + 2, T0 + 3, (0.0, 1.0, 0.0)))
    assert inc.top_cause == Cause.UNKNOWN.value
    inc = a.attribute(Incident(2, K.JITTER, "LHR-F", T0 + 2, T0 + 3, (0.1, 0.1, 0.0)))
    assert inc.top_cause == Cause.IMU.value


def test_problems_following_one_base_become_a_finding():
    devices = {
        "LHR-HEAD": DeviceInfo("LHR-HEAD", DeviceClass.HMD, role="head"),
        "LHR-W": DeviceInfo("LHR-W", DeviceClass.TRACKER, role="waist", battery=0.9),
        "LHR-F": DeviceInfo("LHR-F", DeviceClass.TRACKER, role="left_foot", battery=0.9),
        "LHB-AAAA0001": DeviceInfo("LHB-AAAA0001", DeviceClass.BASE, channel=1),
        "LHB-BBBB0002": DeviceInfo("LHB-BBBB0002", DeviceClass.BASE, channel=2),
    }
    world = World()
    world.set_devices(devices)
    a = Attributor(world)
    n = 0
    for k in range(6):
        serial = ("LHR-W", "LHR-F")[k % 2]
        t = T0 + 10 * k
        world.add_log(LogEvent(t, LogKind.SOB_DROP, serial,
                               {"base": "S-1", "seeing": [{"slot": "S-2", "base": "BBBB0002"}]}))
        n += 1
        a.attribute(Incident(n, K.JITTER, serial, t + 3, t + 4, (0.5 * k - 1, 1.0, 0.0)))
        world.add_log(LogEvent(t + 6, LogKind.SOB_ADD, serial,
                               {"base": "S-1", "seeing": [{"slot": "S-1", "base": "AAAA0001"},
                                                          {"slot": "S-2", "base": "BBBB0002"}]}))
    follow = [f for f in a.findings() if f.cause is Cause.BASE_STATION]
    assert len(follow) == 1
    assert follow[0].bases == ["LHB-BBBB0002"] and "S-2" in follow[0].title
    assert sorted(follow[0].devices) == ["LHR-F", "LHR-W"]


# ---- cost ----------------------------------------------------------------------
def test_per_frame_cost_of_world_detectors_heatmap(scenario, capsys):
    """Time world + detectors + heatmap over a simulated minute (9 devices at 200 Hz)."""
    best = None
    for _ in range(3):
        r = run(60.0)
        assert r.frames == 12000 and len(r.source.devices()) == 9
        us = r.core_s / r.frames * 1e6
        best = us if best is None else min(best, us)
    with capsys.disabled():
        print(f"\n[perf] world+detectors+heatmap: {best:.1f} us/frame "
              f"({best / 9:.1f} us/device-sample, 9 devices @ 200 Hz = {best * 200 / 1e4:.2f}% of one core)")
    assert best < 500.0          # target: well under 1 ms per frame
    r = scenario("demo")         # with faults flying, attribution included
    with capsys.disabled():
        print(f"[perf] demo cycle (300 s, {len(r.incidents)} incidents): "
              f"{r.core_s / r.frames * 1e6:.1f} us/frame core, "
              f"{r.attr_s / max(1, len(r.incidents)) * 1e3:.2f} ms per attributed incident")
    assert r.core_s / r.frames * 1e6 < 500.0
    t = time.perf_counter()
    r.attributor.findings()
    assert time.perf_counter() - t < 0.2
