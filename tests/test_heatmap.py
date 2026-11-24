import json
import math

import pytest

from lhdebug.heatmap import (Heatmap, base_sees, base_view, quat_rotate, segment_distance,
                             sight_blocked)
from lhdebug.model import DeviceMetrics, Incident, IncidentKind, Sample
from lhdebug.sources.sim_source import LEFT_FOOT, RIGHT_FOOT, look_at

IDENT = (1.0, 0.0, 0.0, 0.0)


def sample(serial, pos, ok=True, t=0.0):
    return Sample(t, serial, True, ok, 200 if ok else 201, pos, IDENT)


def cells(snap):
    return {(c[0], c[1], c[2]): (c[3], c[4]) for c in snap["cells"]}


def test_snapshot_shape_and_cell_indexing():
    h = Heatmap(voxel=0.2)
    h.add(sample("A", (0.25, 1.01, -0.05)), DeviceMetrics(update_hz=200, jitter_mm=0.4))
    snap = h.snapshot("samples")
    assert set(snap) == {"voxel", "metric", "min", "max", "cells"}
    assert snap["voxel"] == 0.2 and snap["metric"] == "samples"
    assert snap["cells"] == [[1, 5, -1, 1.0, 1]]
    ix, iy, iz = snap["cells"][0][:3]
    centre = ((ix + .5) * 0.2, (iy + .5) * 0.2, (iz + .5) * 0.2)
    assert all(abs(c - p) <= 0.1 + 1e-9 for c, p in zip(centre, (0.25, 1.01, -0.05)))
    json.dumps(snap)


def test_metrics_per_device_and_combined():
    h = Heatmap()
    good = DeviceMetrics(update_hz=200.0, jitter_mm=0.5)
    for _ in range(30):
        h.add(sample("A", (0.1, 0.1, 0.1)), good)
    for _ in range(10):
        h.add(sample("B", (0.1, 0.1, 0.1)), DeviceMetrics(update_hz=100.0, jitter_mm=2.5))
    for _ in range(10):
        h.add(sample("A", (1.1, 0.1, 0.1)), good)
    # A loses tracking: booked where it was last seen, whatever the stale pose says
    for _ in range(10):
        h.add(sample("A", (9.0, 9.0, 9.0), ok=False), DeviceMetrics())

    assert cells(h.snapshot("samples")) == {(0, 0, 0): (40.0, 40), (5, 0, 0): (20.0, 20)}
    assert cells(h.snapshot("samples", "B")) == {(0, 0, 0): (10.0, 10)}
    assert cells(h.snapshot("dropout", "A")) == {(0, 0, 0): (0.0, 30), (5, 0, 0): (0.5, 20)}
    assert cells(h.snapshot("dropout", "B"))[(0, 0, 0)][0] == 0.0
    assert cells(h.snapshot("jitter"))[(0, 0, 0)][0] == pytest.approx(1.0)       # (30*.5 + 10*2.5)/40
    assert cells(h.snapshot("jitter", "A"))[(5, 0, 0)][0] == pytest.approx(0.5)  # bad samples excluded
    assert cells(h.snapshot("update_rate"))[(0, 0, 0)][0] == pytest.approx(175.0)
    snap = h.snapshot("dropout")
    assert snap["min"] == 0.0 and snap["max"] == 0.5
    assert h.snapshot("samples", "nobody")["cells"] == []
    with pytest.raises(ValueError):
        h.snapshot("bogus")


def test_lost_sample_before_any_good_one_is_skipped():
    h = Heatmap()
    h.add(sample("A", (0.0, 0.0, 0.0), ok=False), DeviceMetrics())
    assert h.snapshot("samples")["cells"] == []


def test_incidents_feed_the_jump_map():
    h = Heatmap()
    h.add_incident(Incident(1, IncidentKind.JUMP, "A", 0.0, 1.0, (0.5, 0.1, 0.5), metrics={"count": 4.0}))
    h.add_incident(Incident(2, IncidentKind.JUMP, "B", 0.0, 1.0, (0.5, 0.1, 0.5), metrics={"jump_m": 0.03}))
    h.add_incident(Incident(3, IncidentKind.DROPOUT, "A", 0.0, 1.0, (0.5, 0.1, 0.5)))
    h.add_incident(Incident(4, IncidentKind.BASE_SHIFT, "LHB-1", 0.0, 1.0, (2.0, 2.2, 2.0)))
    assert cells(h.snapshot("jumps")) == {(2, 0, 2): (5.0, 0)}
    assert cells(h.snapshot("jumps", "A")) == {(2, 0, 2): (4.0, 0)}
    assert cells(h.snapshot("incidents")) == {(2, 0, 2): (3.0, 0)}


def test_end_to_end_dropout_shows_where_the_fault_was(scenario):
    r = scenario("occlusion")
    drop = cells(r.heatmap.snapshot("dropout", LEFT_FOOT))
    assert max(v for v, _ in drop.values()) > 0.5
    hot = [k for k, (v, _) in drop.items() if v > 0.5]
    inc = r.incidents[0]
    for k in hot:        # the hot voxels are next to where the incident started
        assert math.dist([(i + .5) * 0.2 for i in k], inc.pos) < 0.5
    assert max(v for v, _ in cells(r.heatmap.snapshot("dropout", RIGHT_FOOT)).values()) == 0.0
    assert sum(n for _, n in cells(r.heatmap.snapshot("samples")).values()) == r.frames * 6
    assert r.heatmap.snapshot("jitter", LEFT_FOOT)["max"] < 1.5


# ---- geometry ----------------------------------------------------------------
def test_quat_rotate_and_look_at():
    q = look_at((0.0, 2.0, 0.0), (3.0, 2.0, 0.0))
    fwd = quat_rotate(q, (0.0, 0.0, -1.0))
    assert fwd == pytest.approx((1.0, 0.0, 0.0), abs=1e-9)
    assert quat_rotate(q, (0.0, 1.0, 0.0)) == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)
    q = look_at((2.0, 2.0, 2.0), (0.0, 0.0, 0.0))
    assert quat_rotate(q, (0.0, 0.0, -1.0)) == pytest.approx([-1 / math.sqrt(3)] * 3, abs=1e-9)


def test_base_field_of_view():
    # identity orientation: looking along -Z from the origin
    assert base_sees((0, 0, 0), IDENT, (0, 0, -3))
    assert not base_sees((0, 0, 0), IDENT, (0, 0, 3))               # behind
    assert not base_sees((0, 0, 0), IDENT, (0, 0, -6.5))            # beyond ~6 m
    d, az, el = base_view((0, 0, 0), IDENT, (1.0, 0.0, -1.0))
    assert (d, az, el) == pytest.approx((math.sqrt(2), 45.0, 0.0))

    def at(az_deg, el_deg, r=2.0):
        a, e = math.radians(az_deg), math.radians(el_deg)
        return (r * math.cos(e) * math.sin(a), r * math.sin(e), -r * math.cos(e) * math.cos(a))
    assert base_sees((0, 0, 0), IDENT, at(70, 0)) and not base_sees((0, 0, 0), IDENT, at(80, 0))    # 150 deg wide
    assert base_sees((0, 0, 0), IDENT, at(-70, 0)) and not base_sees((0, 0, 0), IDENT, at(-80, 0))
    assert base_sees((0, 0, 0), IDENT, at(0, 50)) and not base_sees((0, 0, 0), IDENT, at(0, 60))    # 110 deg tall
    assert base_sees((0, 0, 0), IDENT, at(0, -50)) and not base_sees((0, 0, 0), IDENT, at(0, -60))


def test_segment_distance_and_body_blocking():
    assert segment_distance((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)) == pytest.approx(1.0)
    assert segment_distance((0, 0, -1), (0, 0, 1), (-1, 0.5, 0), (1, 0.5, 0)) == pytest.approx(0.5)
    assert segment_distance((0, 0, 0), (1, 0, 0), (3, 0, 0), (4, 0, 0)) == pytest.approx(2.0)
    assert segment_distance((0, 0, 0), (0, 0, 0), (1, 1, 0), (1, -1, 0)) == pytest.approx(1.0)
    torso = [((0.0, 1.7, 0.0), (0.0, 1.0, 0.0), 0.17)]
    # tracker on the back of the waist, base in front: the torso is in the way
    assert sight_blocked((0.0, 1.0, 0.25), (0.0, 1.4, -3.0), torso)
    assert not sight_blocked((0.0, 1.0, 0.25), (0.0, 2.2, 3.0), torso)        # base behind sees it
    assert not sight_blocked((0.0, 1.0, 0.1), (0.0, 2.2, 3.0), torso)         # own body part is ignored
    assert not sight_blocked((0.6, 1.0, 0.0), (0.6, 2.2, -3.0), torso)        # off to the side


# ---- coverage ------------------------------------------------------------------
def test_coverage_single_base():
    h = Heatmap(voxel=0.5)
    base = ((0.0, 2.0, 0.0), look_at((0.0, 2.0, 0.0), (0.0, 2.0, -1.0)))       # level, facing -Z
    snap = h.coverage([base], bounds=((-8, 0, -8), (8, 3, 8)))
    assert snap["metric"] == "coverage" and snap["voxel"] == 0.5
    assert snap["min"] == 0.0 and snap["max"] == 1.0
    cov = cells(snap)
    assert len(cov) == 32 * 6 * 32

    def at(x, y, z):
        return cov[(math.floor(x / 0.5), math.floor(y / 0.5), math.floor(z / 0.5))][0]
    assert at(0.1, 1.9, -3.0) == 1          # straight ahead
    assert at(0.1, 1.9, 3.0) == 0           # behind
    assert at(0.1, 1.9, -7.0) == 0          # out of range
    assert at(2.9, 1.9, -1.1) == 1          # ~69 deg off axis: inside 150 deg
    assert at(4.1, 1.9, -0.6) == 0          # ~80 deg off axis
    assert at(0.1, 0.1, -0.6) == 0          # right underneath: steeper than 55 deg down
    assert at(0.1, 0.1, -3.1) == 1
    for (ix, iy, iz), (v, n) in cov.items():        # agrees with the scalar test everywhere
        p = ((ix + .5) * 0.5, (iy + .5) * 0.5, (iz + .5) * 0.5)
        assert v == (1.0 if base_sees(base[0], base[1], p) else 0.0)
        assert n == 1


def test_coverage_of_the_simulated_room(scenario):
    r = scenario("clean")
    bases = [(r.world.base_pose(b.serial).pos, r.world.base_pose(b.serial).quat) for b in r.world.bases()]
    assert len(bases) == 3
    snap = r.heatmap.coverage(bases)                 # default bounds: where the devices went
    cov = cells(snap)
    assert snap["max"] == 3.0
    visited = cells(r.heatmap.snapshot("samples"))
    assert set(visited) <= set(cov)
    # bases in three upper corners aimed at the centre: the whole walked volume is covered by >= 2
    assert min(cov[k][0] for k in visited) >= 2
    assert sum(1 for k in visited if cov[k][0] == 3) > 0.8 * len(visited)
    assert all(n == 3 for _, n in cov.values())
    json.dumps(snap)
    # with no devices seen yet it falls back to the box spanned by the bases
    fresh = Heatmap().coverage(bases)
    ys = [c[1] for c in fresh["cells"]]
    assert min(ys) == 0 and max(ys) >= 10 and fresh["max"] == 3.0
    assert Heatmap().coverage([])["cells"] == []
