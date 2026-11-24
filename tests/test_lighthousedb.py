import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lhdebug import lighthousedb  # noqa: E402
from lhdebug.lighthousedb import LighthouseDB, base_id, base_serial, load  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "lighthousedb_sample.json"


@pytest.fixture(scope="module")
def db():
    d = load(FIXTURE)
    assert d.error is None
    return d


# ---- identity ----------------------------------------------------------------
@pytest.mark.parametrize("given, expected", [
    ("LHB-4F091D9A", "4F091D9A"),
    ("4F091D9A", "4F091D9A"),
    ("lhb-4f091d9a", "4F091D9A"),
    ("ee93c79b", "EE93C79B"),
    (" B3316995 ", "B3316995"),
    (1079059475, "40512413"),          # decimal int, as stored in lighthousedb.json
    ("1079059475", "40512413"),        # same, as a string (10 digits -> decimal)
    ("3006359957", "B3316995"),
    ("40512413", "40512413"),          # 8 hex digits that happen to be all decimal: an id
    ("LHB-40512413", "40512413"),
    (979930530, "3A688DA2"),
])
def test_base_id(given, expected):
    assert base_id(given) == expected
    assert base_id(base_id(given)) == expected          # idempotent
    assert base_serial(given) == "LHB-" + expected
    assert base_serial(base_serial(given)) == "LHB-" + expected


def test_base_id_small_number_is_zero_padded():
    assert base_id(255) == "000000FF"


# ---- load ----------------------------------------------------------------------
def test_bases_keyed_by_lhb_serial(db):
    assert set(db.bases) == {
        "LHB-3A688DA2", "LHB-40512413", "LHB-4FD91D9A", "LHB-987626CC", "LHB-A0CD6092",
        "LHB-B3316995", "LHB-C6C0FFCD", "LHB-EB557902", "LHB-EDF5C3E1", "LHB-EE93C79B"}
    for serial, b in db.bases.items():
        assert b.serial == serial and serial == "LHB-" + b.id
        assert base_id(b.serial_number) == b.id


def test_base_record_fields(db):
    b = db.bases["LHB-B3316995"]
    assert b.serial_number == 3006359957
    # sobChannel 0 in the file == slot "S-1" in vrserver.txt
    assert b.channel == 1 and b.slot == "S-1"
    assert b.firmware == 3008 and b.faults == 0 and b.model_id == 15
    assert b.tilt_pitch == pytest.approx(36.10, abs=0.05)     # radians in file -> degrees
    assert b.tilt_roll == pytest.approx(7.95, abs=0.05)
    assert b.last_seen and b.last_seen > 1.7e9
    assert b.gravity is not None and len(b.gravity) == 3
    assert db.bases["LHB-EE93C79B"].slot == "S-5"
    assert db.bases["LHB-A0CD6092"].slot == "S-7"
    assert db.bases["LHB-987626CC"].faults == 4


def test_lookup_helpers(db):
    assert db.base("b3316995") is db.bases["LHB-B3316995"]
    assert db.base(3006359957) is db.bases["LHB-B3316995"]
    assert db.base("LHB-00000000") is None
    # S-1 is shared by a stale record; the most recently seen one wins
    assert db.by_slot("S-1").serial == "LHB-B3316995"
    assert db.by_slot(5, universe_id="1749950692").serial == "LHB-EE93C79B"
    assert db.by_slot("S-7").serial == "LHB-A0CD6092"
    assert db.by_slot("S-16") is None and db.by_slot("bogus") is None


def test_universes(db):
    assert [u.id for u in db.universes] == ["1637272552", "1691467899", "1728616698", "1749950692"]
    u = db.universe(1749950692)
    assert u is db.current_universe()
    assert u.primary == "LHB-B3316995"
    assert set(u.located()) == {"LHB-B3316995", "LHB-EE93C79B", "LHB-A0CD6092"}
    assert len(u.bases) == 8 and set(u.bases) <= set(db.bases)
    p = u.poses["LHB-B3316995"]
    assert p.primary and p.located and p.pos == (0.0, 0.0, 0.0) and p.quat == (1.0, 0.0, 0.0, 0.0)
    stale = u.poses["LHB-40512413"]
    assert not stale.located and not stale.primary
    for s in u.located():
        q = u.poses[s].quat
        assert sum(c * c for c in q) == pytest.approx(1.0, abs=1e-4)
    assert u.tilt_pitch == pytest.approx(36.14, abs=0.05)
    assert u.tilt_roll == pytest.approx(8.14, abs=0.05)
    assert u.last_chaperone_commit == 1795231387.0
    assert db.universe("1691467899").previous_id == "1637272552"
    assert db.universe("nope") is None


def test_universe_distances_match_the_log(db):
    """vrserver.txt RELATIONSHIP lines for this rig say 2.74-2.83 / 4.02-4.22 / 2.88-3.00 m."""
    u = db.universe("1749950692")
    assert u.distance("B3316995", "ee93c79b") == pytest.approx(2.75, abs=0.1)
    assert u.distance("LHB-B3316995", "LHB-A0CD6092") == pytest.approx(4.14, abs=0.1)
    assert u.distance("A0CD6092", "EE93C79B") == pytest.approx(2.55, abs=0.1)
    assert u.distance("B3316995", "40512413") is None       # not located
    assert u.distance("B3316995", "FFFFFFFF") is None       # not in the universe


def test_summary_is_json_able(db):
    s = db.summary()
    text = json.dumps(s)
    assert json.loads(text) == s
    assert s["ok"] is True and s["error"] is None and s["revision"] == db.revision
    assert set(s["bases"]) == set(db.bases)
    assert s["bases"]["LHB-B3316995"]["slot"] == "S-1"
    cur = [u for u in s["universes"] if u["current"]]
    assert len(cur) == 1 and cur[0]["id"] == "1749950692" and cur[0]["primary"] == "LHB-B3316995"
    assert len(cur[0]["distances"]) == 3
    pose = cur[0]["poses"]["LHB-EE93C79B"]
    assert len(pose["pos"]) == 3 and len(pose["quat"]) == 4 and pose["located"] is True
    assert s["known_objects"]


# ---- robustness ------------------------------------------------------------------
def test_missing_file(tmp_path):
    d = load(tmp_path / "nope.json")
    assert isinstance(d, LighthouseDB)
    assert d.bases == {} and d.universes == [] and d.error
    s = d.summary()
    assert s["ok"] is False and s["bases"] == {} and s["universes"] == []
    json.dumps(s)
    assert d.current_universe() is None


@pytest.mark.parametrize("content", ["", "{ not json", "[1, 2]", '{"base_stations": [{"config": {}}, 7], '
                                     '"known_universes": [{"id": 5, "base_stations": [{}]}]}'])
def test_corrupt_or_odd_file(tmp_path, content):
    p = tmp_path / "lighthousedb.json"
    p.write_text(content, encoding="utf-8")
    d = load(p)
    json.dumps(d.summary())
    assert d.bases == {}


def test_default_path():
    p = lighthousedb.default_path()
    assert p.name == "lighthousedb.json" and p.parent.name == "lighthouse"
    assert lighthousedb.steam_dir().name.lower() == "steam"
    assert isinstance(load(), LighthouseDB)      # never raises, whatever is installed
