from lhdebug.model import Cause, DeviceMetrics, Sample
from lhdebug.rangeprofile import NEAREST, RangeProfile

BASES = {"LHB-A": (0.0, 2.0, 0.0), "LHB-B": (10.0, 2.0, 0.0)}


def feed(p, x, n, ok=True, jitter=0.5, seen=None, serial="LHR-1"):
    for i in range(n):
        s = Sample(t=i * 0.1, serial=serial, connected=True, valid=ok, result=200 if ok else 201,
                   pos=(x, 2.0, 0.0))
        p.add(s, DeviceMetrics(jitter_mm=jitter), BASES, seen)


def test_bins_by_distance_and_respects_seen_bases():
    p = RangeProfile()
    feed(p, 2.2, 100, jitter=0.4, seen={"LHB-A"})
    snap = p.snapshot()["series"]
    assert [r["d"] for r in snap["LHB-A"]] == [2.25]
    assert snap["LHB-A"][0]["jitter_mm"] == 0.4
    assert "LHB-B" not in snap                   # not a base the device was using
    assert snap[NEAREST][0]["ok_frac"] == 1.0


def test_small_bins_are_hidden():
    p = RangeProfile()
    feed(p, 1.0, 10)
    assert p.snapshot()["series"] == {}


def test_loss_is_counted_at_last_tracked_position():
    p = RangeProfile()
    feed(p, 5.0, 100)
    feed(p, 0.0, 100, ok=False)                  # invalid poses report garbage positions
    row = p.snapshot()["series"][NEAREST][0]
    assert row["d"] == 5.25 and row["n"] == 200 and row["ok_frac"] == 0.5


def test_no_finding_when_healthy_or_thin():
    p = RangeProfile()
    feed(p, 2.0, 400, jitter=0.4)
    assert p.finding() is None                   # no far data at all
    feed(p, 5.0, 400, jitter=1.0)                # 2.5x the distance, 2.5x the jitter: expected
    assert p.finding() is None


def test_finding_on_loss_far_from_bases():
    p = RangeProfile()
    feed(p, 2.0, 400)
    feed(p, 5.0, 300)
    feed(p, 5.0, 100, ok=False)
    f = p.finding()
    assert f is not None and f.cause is Cause.RANGE and f.score >= 0.5
    assert "25%" in f.detail


def test_finding_on_excess_jitter_far_from_bases():
    p = RangeProfile()
    feed(p, 2.0, 400, jitter=0.4)
    feed(p, 5.0, 400, jitter=4.0)
    f = p.finding()
    assert f is not None and "jitter" in f.detail
