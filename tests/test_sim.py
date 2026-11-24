import math
import time

import pytest

from harness import T0
from lhdebug.heatmap import base_sees
from lhdebug.model import DeviceClass, LogKind, TrackingResult
from lhdebug.sources.sim_source import (BASES, DONGLE_FEET, FAULT_KINDS, HMD, LEFT_FOOT,
                                        RIGHT_FOOT, RIGHT_HAND, WAIST, Fault, SimSource)
from lhdebug.world import World


def collect(duration=10.0, faults=None, **kw):
    src = SimSource(realtime=False, duration=duration, faults=faults, t0=T0, **kw)
    frames, log = [], []
    while True:
        try:
            frames.append(src.read())
        except StopIteration:
            break
        log.extend(src.drain_log())
    return src, frames, log


def series(frames, serial):
    return [s for f in frames for s in f.samples if s.serial == serial]


def kinds(log, serial=None):
    return [e.kind for e in log if serial is None or e.serial == serial]


def test_finite_duration_and_timestamps():
    src, frames, _ = collect(2.0)
    assert len(frames) == 400
    assert frames[0].t == T0
    assert frames[-1].t == pytest.approx(T0 + 399 / 200.0)
    with pytest.raises(StopIteration):
        src.read()


def test_rig_layout_and_registry():
    src, frames, _ = collect(30.0)
    devs = src.devices()
    by_class = {}
    for d in devs.values():
        by_class.setdefault(d.device_class, []).append(d)
    assert len(by_class[DeviceClass.HMD]) == 1
    assert len(by_class[DeviceClass.CONTROLLER]) == 2
    assert len(by_class[DeviceClass.TRACKER]) == 3
    assert len(by_class[DeviceClass.BASE]) == 3
    assert {d.role for d in by_class[DeviceClass.TRACKER]} == {"waist", "left_foot", "right_foot"}
    assert all(d.dongle for d in devs.values()
               if d.device_class in (DeviceClass.CONTROLLER, DeviceClass.TRACKER))
    assert devs[LEFT_FOOT].dongle == devs[RIGHT_FOOT].dongle == DONGLE_FEET
    assert sorted(d.channel for d in by_class[DeviceClass.BASE]) == [1, 2, 3]
    assert len(frames[0].samples) == 9

    def mean_y(serial):
        ys = [s.pos[1] for s in series(frames, serial)]
        return sum(ys) / len(ys)
    assert 1.6 < mean_y(HMD) < 1.8
    assert 0.9 < mean_y(WAIST) < 1.1
    assert mean_y(LEFT_FOOT) < 0.3 and mean_y(RIGHT_FOOT) < 0.3
    assert 0.7 < mean_y(RIGHT_HAND) < 1.4
    for s in series(frames, HMD):
        assert abs(s.pos[0]) < 1.6 and abs(s.pos[2]) < 1.6
    xs = [s.pos[0] for s in series(frames, HMD)]
    assert max(xs) - min(xs) > 1.0          # actually walks around
    for b in BASES:
        s = series(frames, b)[0]
        assert 2.0 < s.pos[1] < 2.5
        assert base_sees(s.pos, s.quat, (0.0, 1.0, 0.0))
        assert not base_sees(s.pos, s.quat, (s.pos[0] * 2, 2.2, s.pos[2] * 2))


def test_clean_rig_is_always_ok_with_fresh_velocity_and_small_noise():
    _, frames, log = collect(10.0)
    for serial in (HMD, WAIST, LEFT_FOOT, RIGHT_HAND):
        ss = series(frames, serial)
        assert all(s.ok for s in ss)
        assert all(a.vel != b.vel for a, b in zip(ss, ss[1:]))
        # reported velocity explains the motion to within sensor noise
        res = [math.dist(b.pos, a.pos) - math.dist((0, 0, 0), b.vel) * (b.t - a.t)
               for a, b in zip(ss, ss[1:])]
        assert max(abs(r) for r in res) < 0.004
        for s in ss[::50]:
            assert sum(c * c for c in s.quat) == pytest.approx(1.0, abs=1e-5)
    bad = {LogKind.TRACKING_RESET, LogKind.WIRELESS_DISCONNECTED, LogKind.RECEIVER_DISCONNECTED,
           LogKind.HID_ERROR, LogKind.IMU_OFF_SCALE, LogKind.TILT_UPDATE, LogKind.BACKFACING_HITS}
    assert not bad & set(kinds(log))
    assert LogKind.SOB_ADD in kinds(log)


def test_deterministic_per_seed():
    _, a, _ = collect(1.0, seed=3)
    _, b, _ = collect(1.0, seed=3)
    _, c, _ = collect(1.0, seed=4)
    assert a[-1].samples[0].pos == b[-1].samples[0].pos
    assert a[-1].samples[0].pos != c[-1].samples[0].pos


def test_battery_drains():
    src, _, _ = collect(60.0, rate=50.0)
    assert src.devices()[HMD].battery is None
    assert 0.80 < src.devices()[WAIST].battery < 0.82


def test_log_events_keep_world_context_in_sync():
    src = SimSource(realtime=False, duration=5.0, faults=None, t0=T0)
    world = World()
    while True:
        try:
            frame = src.read()
        except StopIteration:
            break
        world.set_devices(src.devices())
        for ev in src.drain_log():
            world.add_log(ev)
        world.add_frame(frame)
    assert world.dongle_of[WAIST] == src.devices()[WAIST].dongle
    assert set(world.devices_on_dongle(DONGLE_FEET)) == {LEFT_FOOT, RIGHT_FOOT}
    assert 1 <= len(world.visible_bases(HMD)) <= 3
    assert all(slot.startswith("S-") for slot in world.visible_bases(HMD))
    assert world.base_pose(BASES[0]).pos[1] > 2.0


def test_realtime_paces_itself():
    src = SimSource(rate=100.0, realtime=True, duration=0.2, faults=None)
    t = time.monotonic()
    n = 0
    while True:
        try:
            src.read()
            n += 1
        except StopIteration:
            break
    assert n == 20
    assert time.monotonic() - t >= 0.15


def test_unknown_fault_kind_rejected():
    with pytest.raises(ValueError):
        SimSource(faults=[Fault("gremlins", 1, 2)])


# ---- faults: pose symptoms and matching log events ---------------------------
def test_occlusion_fault():
    _, frames, log = collect(12.0, [Fault("occlusion", 3, 6, serial=LEFT_FOOT)])
    ss = series(frames, LEFT_FOOT)
    lost = [s for s in ss if not s.ok]
    assert lost and all(s.connected and not s.valid for s in lost)
    assert all(s.result == TrackingResult.RUNNING_OUT_OF_RANGE for s in lost)
    assert T0 + 3.3 < lost[0].t < T0 + 3.6          # drifts on IMU for a moment first
    assert lost[-1].t < T0 + 6.01
    assert all(s.ok for s in series(frames, RIGHT_FOOT))
    mine = [e for e in log if e.serial == LEFT_FOOT and e.t >= T0 + 3]
    ks = [e.kind for e in mine]
    assert LogKind.SOB_DROP in ks
    reset = next(e for e in mine if e.kind is LogKind.TRACKING_RESET)
    assert reset.data["reason"] == "no_optical" and reset.data["ms"] > 0
    assert ks.index(LogKind.SOB_DROP) < ks.index(LogKind.TRACKING_RESET)
    adds = [e for e in mine if e.kind is LogKind.SOB_ADD and e.t >= T0 + 6]
    assert adds and adds[-1].data["seeing"] and "slot" in adds[-1].data["seeing"][0]


def test_occlusion_region_only_inside():
    region = (0.7, -0.5, 0.8)
    _, frames, _ = collect(40.0, [Fault("occlusion", 0, 40, serial=WAIST, region=region)])
    ss = series(frames, WAIST)
    assert any(not s.ok for s in ss) and any(s.ok for s in ss)
    far_ok = [s for s in ss if math.hypot(s.pos[0] - region[0], s.pos[2] - region[1]) > 1.2]
    assert far_ok and all(s.ok for s in far_ok)


@pytest.mark.parametrize("variant", ["frozen", "extrapolate"])
def test_radio_fault_goes_stale_then_disconnects(variant):
    _, frames, log = collect(12.0, [Fault("radio", 3, 8, serial=LEFT_FOOT,
                                          params={"whole_dongle": True, "variant": variant})])
    for serial in (LEFT_FOOT, RIGHT_FOOT):          # same dongle: both hit
        ss = series(frames, serial)
        stale_phase = [s for s in ss if T0 + 4 <= s.t < T0 + 5]
        assert all(s.ok for s in stale_phase)
        pairs = list(zip(stale_phase, stale_phase[1:]))
        same_vel = [(a, b) for a, b in pairs if a.vel == b.vel and a.angvel == b.angvel]
        assert len(same_vel) > 0.6 * len(pairs)     # update rate has collapsed
        if variant == "frozen":
            assert all(a.pos == b.pos and a.quat == b.quat for a, b in same_vel)
        else:
            assert all(a.pos != b.pos for a, b in same_vel)
            a, b = same_vel[0]
            for i in range(3):
                assert b.pos[i] - a.pos[i] == pytest.approx(a.vel[i] * (b.t - a.t), abs=1e-9)
        gone = [s for s in ss if not s.connected]
        assert gone and T0 + 4.9 < gone[0].t < T0 + 5.1 and gone[-1].t < T0 + 8.01
        assert LogKind.WIRELESS_DISCONNECTED in kinds(log, serial)
    assert all(s.ok for s in series(frames, WAIST))
    rd = [e for e in log if e.kind is LogKind.RECEIVER_DISCONNECTED]
    assert rd and rd[0].data["dongle"] == DONGLE_FEET
    assert any(e.kind is LogKind.RECEIVER_CONNECTED and e.t >= T0 + 8 for e in log)


def test_base_shift_fault():
    _, frames, log = collect(10.0, [Fault("base_shift", 3, 5, base=BASES[1])])
    bs = series(frames, BASES[1])
    assert bs[0].pos == bs[999 - 1].pos                 # unchanged until the driver notices
    assert math.dist(bs[0].pos, bs[-1].pos) == pytest.approx(0.03, abs=1e-6)
    assert bs[0].quat != bs[-1].quat
    assert series(frames, BASES[0])[0].pos == series(frames, BASES[0])[-1].pos
    tilt = [e for e in log if e.kind is LogKind.TILT_UPDATE]
    assert len(tilt) == 1 and tilt[0].data["base"] == BASES[1][4:]
    assert tilt[0].data["delta_deg"] == pytest.approx(1.5)
    assert tilt[0].t == pytest.approx(T0 + 5.0)
    # devices relying on it are displaced while the driver has the wrong pose
    _, clean, _ = collect(10.0)
    disp = [math.dist(a.pos, b.pos) for a, b in zip(series(frames, HMD), series(clean, HMD))]
    assert max(disp[:590]) < 0.005 and max(disp[1010:]) < 0.005
    assert max(disp[600:1000]) > 0.01
    assert all(s.ok for s in series(frames, HMD))


def test_reflection_fault():
    region = (0.7, -0.5, 0.8)
    f = [Fault("reflection", 2, 58, serial=RIGHT_FOOT, region=region)]
    _, frames, log = collect(60.0, f)
    _, clean, _ = collect(60.0)
    ss = series(frames, RIGHT_FOOT)
    assert all(s.ok for s in ss)                         # tracking stays valid
    disp = [(math.dist(a.pos, b.pos), b) for a, b in zip(ss, series(clean, RIGHT_FOOT))]
    big = [(d, s) for d, s in disp if d > 0.015]
    assert big and max(d for d, _ in big) < 0.06         # a few centimetres
    for _, s in big:                                     # confined to the region
        assert math.hypot(s.pos[0] - region[0], s.pos[2] - region[1]) < region[2] + 0.1
    hits = [e for e in log if e.kind is LogKind.BACKFACING_HITS]
    assert hits and all(e.serial == RIGHT_FOOT and e.data["backfacing"] >= 20 for e in hits)
    assert {"rejected", "backfacing", "non_clustered"} <= set(hits[0].data)


def test_imu_fault():
    _, frames, log = collect(10.0, [Fault("imu", 3, 7, serial=RIGHT_HAND)])
    _, clean, _ = collect(10.0)
    ev = [e for e in log if e.kind is LogKind.IMU_OFF_SCALE]
    assert len(ev) > 5 and all(e.serial == RIGHT_HAND and T0 + 3 <= e.t <= T0 + 7 for e in ev)
    d = [math.dist(a.pos, b.pos) for a, b in zip(series(frames, RIGHT_HAND), series(clean, RIGHT_HAND))]
    assert max(d[:590]) < 0.003 and max(d[1410:]) < 0.003
    assert max(d[600:1400]) > 0.005
    assert all(s.ok for s in series(frames, RIGHT_HAND))


def test_usb_fault_drops_everything_at_once():
    src, frames, log = collect(8.0, [Fault("usb", 3, 4)])
    hid = [e for e in log if e.kind is LogKind.HID_ERROR]
    assert hid and {e.data["interface"] for e in hid} >= {"IMU", "Optical"}
    tracked = [d.serial for d in src.devices().values() if d.device_class is not DeviceClass.BASE]
    firsts = set()
    for serial in tracked:
        gone = [s for s in series(frames, serial) if not s.connected]
        assert gone
        firsts.add(gone[0].t)
        assert gone[-1].t < T0 + 4.01
    assert len(firsts) == 1
    assert all(s.ok for s in frames[-1].samples)


def test_battery_fault_low_charge_and_rising_dropouts():
    src = SimSource(realtime=False, duration=60.0, faults=[Fault("battery", 3, 58, serial=WAIST)], t0=T0)
    levels, starts, prev_conn = [], [], True
    while True:
        try:
            frame = src.read()
        except StopIteration:
            break
        s = next(x for x in frame.samples if x.serial == WAIST)
        if prev_conn and not s.connected:
            starts.append(frame.t - T0)
        prev_conn = s.connected
        if 3 < frame.t - T0 < 58:
            levels.append(src.devices()[WAIST].battery)
    assert max(levels) <= 0.08 and min(levels) < 0.02
    assert levels == sorted(levels, reverse=True)
    assert len(starts) >= 8
    assert sum(1 for t in starts if t > 30.5) > sum(1 for t in starts if t <= 30.5)


def test_demo_cycles_through_every_fault_kind():
    src, frames, log = collect(300.0, "demo", rate=50.0)
    ks = set(kinds(log))
    assert {LogKind.SOB_DROP, LogKind.TRACKING_RESET, LogKind.WIRELESS_DISCONNECTED,
            LogKind.RECEIVER_DISCONNECTED, LogKind.TILT_UPDATE, LogKind.BACKFACING_HITS,
            LogKind.IMU_OFF_SCALE, LogKind.HID_ERROR} <= ks
    assert any(not x.connected for x in series(frames, HMD))          # the usb glitch
    assert set(FAULT_KINDS) >= {"occlusion", "radio", "base_shift", "reflection", "imu", "usb", "battery"}
    # and loops: the second cycle is scheduled
    src2 = SimSource(rate=20.0, realtime=False, duration=None, faults="demo", t0=T0)
    for _ in range(int(335 * 20)):
        src2.read()
    assert any(e.kind is LogKind.TRACKING_RESET and e.t > T0 + 300 for e in src2.drain_log())
