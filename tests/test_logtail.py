import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lhdebug import logtail  # noqa: E402
from lhdebug.logtail import LogTailer, parse_line, scan_file  # noqa: E402
from lhdebug.model import LogKind, to_json  # noqa: E402
from lhdebug.world import World  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "vrserver_sample.txt"
P = "Mon Nov 23 2026 20:00:07.705 [Info] - "
LH = P + "lighthouse: "


def ev(msg, prefix=LH):
    e = parse_line(prefix + msg)
    assert e is not None, msg
    return e


# ---- envelope ---------------------------------------------------------------
def test_timestamp_is_local_time_epoch():
    e = ev("LHR-329ADD4A H: SOB: add S-7")
    expected = time.mktime((2026, 11, 23, 20, 0, 7, 0, 0, -1)) + 0.705
    assert e.t == pytest.approx(expected, abs=1e-6)
    assert e.level == "Info"
    assert e.raw == LH + "LHR-329ADD4A H: SOB: add S-7"


def test_level_and_trailing_newline():
    e = parse_line("Tue Nov 24 2026 01:38:20.251 [Error] - lighthouse: Lighthouse IMU HID device error\r\n")
    assert e.level == "Error" and not e.raw.endswith("\n")


@pytest.mark.parametrize("line", [
    "",
    "garbage",
    "{",
    '   "controller_type" : null',
    P + "//==============================================================",
    P + "[Input] No legacy bindings found for app system.generated.foo",
    P + "lighthouse: MONITOR\\MSI3CB0\\{4d36e96e-e325-11ce-bfc1-08002be10318}\\0002: 0, 0, 2560, 1440",
    P + "lighthouse: Enumerating displays",
    P + "lighthouse: Attempting HID Open IMU: LHR-329ADD4A",
    P + "lighthouse: Attempting HID Open IMU: 40E4AC9C61",
    P + "lighthouse: Lighthouse IMU HID opened",
    P + "lighthouse: E518136DA0: Triggered keepalive (succeeded)",
    P + "lighthouse: 0B48A6C31B-RYB: Packet received after 4.752s, keepalive (0/1)",
    P + "lighthouse: LHR-D9015EEF: Triggered keepalive (failed)",
    P + "lighthouse: LHR-29C03CCA: Transition to protocol version 1",
    P + "lighthouse: LHR-329ADD4A H: Source: auto disambiguator (right sync) -- Preferred basestation 00000000",
    P + "lighthouse: Saved LighthouseDB rev 61946 to C:\\Steam\\config\\lighthouse\\lighthousedb.json",
    "Xxx Foo 99 2026 20:00:07.705 [Info] - lighthouse: LHR-329ADD4A H: SOB: add S-7",
])
def test_noise_is_dropped(line):
    assert parse_line(line) is None


# ---- SOB --------------------------------------------------------------------
def test_sob_add_bare():
    e = ev("LHR-329ADD4A H: SOB: add S-7")
    assert e.kind is LogKind.SOB_ADD and e.serial == "LHR-329ADD4A"
    assert e.data["base"] == "S-7" and e.data["seeing"] == []
    assert e.data["base_id"] is None and e.data["generation_changed"] is False


def test_sob_add_also_seeing_without_ids():
    e = ev("LHR-329ADD4A H: SOB: add S-5 also seeing S-1 S-7")
    assert e.data["base"] == "S-5"
    assert e.data["seeing"] == [{"slot": "S-1", "base": None}, {"slot": "S-7", "base": None}]


def test_sob_add_with_ids():
    e = ev("LHR-29C03CCA C: SOB: add S-5 also seeing S-1 (B3316995) S-7 (ee93c79b)")
    assert e.serial == "LHR-29C03CCA"
    assert e.data["seeing"] == [{"slot": "S-1", "base": "B3316995"}, {"slot": "S-7", "base": "EE93C79B"}]


def test_sob_add_mixed_ids_and_generation_changed():
    e = ev("LHR-F290968A C: SOB: add S-1 (generation changed) also seeing S-5 (EE93C79B) S-7")
    assert e.kind is LogKind.SOB_ADD and e.data["base"] == "S-1"
    assert e.data["generation_changed"] is True
    assert e.data["seeing"] == [{"slot": "S-5", "base": "EE93C79B"}, {"slot": "S-7", "base": None}]


def test_sob_drop_variants():
    e = ev("LHR-29C03CCA C: SOB: drop S-7 (A0CD6092) seeing S-1 (B3316995) S-5 (EE93C79B)")
    assert e.kind is LogKind.SOB_DROP
    assert e.data["base"] == "S-7" and e.data["base_id"] == "A0CD6092"
    assert e.data["seeing"] == [{"slot": "S-1", "base": "B3316995"}, {"slot": "S-5", "base": "EE93C79B"}]
    e = ev("LHR-BA34F722 C: SOB: drop S-7")
    assert e.data == {"base": "S-7", "seeing": [], "base_id": None, "generation_changed": False}
    e = ev("LHR-BA34F722 C: SOB: drop S-5 seeing S-1 S-7 (A0CD6092)")
    assert e.data["seeing"] == [{"slot": "S-1", "base": None}, {"slot": "S-7", "base": "A0CD6092"}]


def test_sob_sequence_through_world():
    w = World()
    seq = [
        ("LHR-29C03CCA C: SOB: add S-5", ["S-5"]),
        ("LHR-29C03CCA C: SOB: add S-1 also seeing S-5", ["S-1", "S-5"]),
        ("LHR-29C03CCA C: SOB: add S-7 also seeing S-1 S-5", ["S-1", "S-5", "S-7"]),
        ("LHR-29C03CCA C: SOB: drop S-7 (A0CD6092) seeing S-1 (B3316995) S-5 (EE93C79B)", ["S-1", "S-5"]),
        ("LHR-29C03CCA C: SOB: drop S-5 (EE93C79B) seeing S-1 (B3316995)", ["S-1"]),
        ("LHR-29C03CCA C: SOB: add S-7 also seeing S-1 (B3316995)", ["S-1", "S-7"]),
        ("LHR-29C03CCA C: Resetting tracking: no optical samples for 2001ms", []),
        ("LHR-29C03CCA C: SOB: add S-5 (generation changed) also seeing S-7", ["S-5", "S-7"]),
    ]
    for msg, expect in seq:
        w.add_log(ev(msg))
        assert w.visible_bases("LHR-29C03CCA") == expect, msg
    assert w.visible["LHR-29C03CCA"] == {"S-5": None, "S-7": None}
    assert w.visible_bases("LHR-329ADD4A") == []


def test_fixture_sob_sequence_through_world():
    """The real play-session SOB lines for one controller, replayed in order."""
    w = World()
    states = []
    for e in scan_file(FIXTURE):
        w.add_log(e)
        if e.serial == "LHR-29C03CCA" and e.kind in (LogKind.SOB_ADD, LogKind.SOB_DROP):
            states.append(w.visible_bases("LHR-29C03CCA"))
            # an add always leaves the added slot visible, a drop never does
            assert (e.data["base"] in states[-1]) == (e.kind is LogKind.SOB_ADD)
    assert states[:4] == [["S-5"], ["S-1", "S-5"], ["S-1", "S-5", "S-7"], ["S-1", "S-5"]]
    assert len(states) > 10
    for slots in w.visible.values():
        assert set(slots) <= {"S-1", "S-5", "S-7"}
    # ids named next to slots are consistent across the whole session
    ids = {}
    for e in scan_file(FIXTURE):
        for s in e.data.get("seeing", []) if e.kind in (LogKind.SOB_ADD, LogKind.SOB_DROP) else []:
            if s["base"]:
                ids.setdefault(s["slot"], set()).add(s["base"])
    assert ids == {"S-1": {"B3316995"}, "S-5": {"EE93C79B"}, "S-7": {"A0CD6092"}}


# ---- base solving -----------------------------------------------------------
def test_relationship_normalises_case():
    e = ev("LHR-329ADD4A H: ----- RELATIONSHIP bases B3316995 <-> ee93c79b distance 2.80m, angle 40.48 deg -----")
    assert e.kind is LogKind.BASE_RELATIONSHIP and e.serial == "LHR-329ADD4A"
    assert e.data == {"a": "B3316995", "b": "EE93C79B", "distance": 2.80, "angle": 40.48}


def test_calibrated():
    e = ev("LHR-329ADD4A H: ----- CALIBRATED base B3316995 at pitch 36.23 deg roll -8.82 deg -----")
    assert e.kind is LogKind.BASE_CALIBRATED
    assert e.data == {"base": "B3316995", "pitch": 36.23, "roll": -8.82}


def test_secondary():
    e = ev("LHR-329ADD4A H: ----- SECONDARY base EE93C79B distance 2.98m -----")
    assert e.kind is LogKind.BASE_SECONDARY and e.data == {"base": "EE93C79B", "distance": 2.98}


def test_bootstrapped_modes():
    e = ev("LHR-329ADD4A H: ----- BOOTSTRAPPED base B3316995 (immediate) distance 3.32m velocity 0.03m/s "
           "base pitch ~34.6 deg roll ~5.5 deg -----")
    assert e.kind is LogKind.BASE_BOOTSTRAPPED
    assert e.data["base"] == "B3316995" and e.data["mode"] == "immediate"
    assert e.data["distance"] == 3.32 and e.data["velocity"] == 0.03
    assert e.data["pitch"] == 34.6 and e.data["roll"] == 5.5 and e.data["velocity_missing"] is False
    e = ev("LHR-BA34F722 C: ----- BOOTSTRAPPED base EE93C79B (best) distance 3.08m velocity 0.48m/s "
           "base pitch ~27.8 deg roll ~-2.9 deg -----")
    assert e.data["mode"] == "best" and e.data["roll"] == -2.9
    e = ev("LHR-29C03CCA C: ----- BOOTSTRAPPED base A0CD6092 (delayed) distance 3.47m velocity 0.02m/s "
           "base pitch ~33.5 deg roll ~0.8 deg -----")
    assert e.data["mode"] == "delayed"


def test_bootstrapped_without_velocity():
    e = ev("LHR-72E8FAE5 C: ----- BOOTSTRAPPED base B3316995 (immediate) distance 2.21m "
           "base pitch ~34.6 deg roll ~5.5 deg -----")
    assert e.kind is LogKind.BASE_BOOTSTRAPPED
    assert e.data["distance"] == 2.21 and e.data["velocity"] == 0.0 and e.data["velocity_missing"] is True
    assert e.data["pitch"] == 34.6


def test_base_add_failed():
    e = ev("LHR-29C03CCA C: Trying to add a secondary base A0CD6092: "
           "Available samples not sufficient for booting (hits: 2+0)")
    assert e.kind is LogKind.BASE_ADD_FAILED and e.serial == "LHR-29C03CCA"
    assert e.data["base"] == "A0CD6092" and e.data["stage"] == "secondary" and e.data["hits"] == [2, 0]
    assert e.data["reason"] == "Available samples not sufficient for booting (hits: 2+0)"
    e = ev("LHR-F19C1C47 C: Trying to start tracking from base b3316995: "
           "Samples didn't yield successful bootstrap pose")
    assert e.kind is LogKind.BASE_ADD_FAILED and e.data["stage"] == "start"
    assert e.data["base"] == "B3316995" and "hits" not in e.data


# ---- tracking health ----------------------------------------------------------
def test_imu_off_scale():
    e = ev("LHR-F19C1C47 C: IMU went off scale.")
    assert e.kind is LogKind.IMU_OFF_SCALE and e.serial == "LHR-F19C1C47" and e.data == {}


def test_tracking_reset_no_optical():
    e = ev("LHR-BA34F722 C: Resetting tracking: no optical samples for 2001ms")
    assert e.kind is LogKind.TRACKING_RESET
    assert e.data["reason"] == "no_optical" and e.data["ms"] == 2001
    assert e.data["text"] == "no optical samples for 2001ms" and "base" not in e.data
    e = ev("LHR-72E8FAE5 C: Resetting tracking: no optical samples from base A0CD6092 for 2000ms")
    assert e.data["reason"] == "no_optical" and e.data["ms"] == 2000 and e.data["base"] == "A0CD6092"


def test_tracking_reset_imu_misalignment():
    e = ev("LHR-72E8FAE5 C: Resetting tracking: IMU misalignment unreasonably large (-8.5, -17, 2.1) deg sigma 0.81")
    assert e.data["reason"] == "imu_misalignment" and e.data["ms"] is None
    assert e.data["angles"] == [-8.5, -17.0, 2.1] and e.data["sigma"] == 0.81


def test_tracking_reset_other():
    e = ev("LHR-BA34F722 C: Resetting tracking: position out of bounds (3.7, -1.8, -5.7)")
    assert e.kind is LogKind.TRACKING_RESET
    assert e.data["reason"] == "other" and e.data["ms"] is None
    assert e.data["pos"] == [3.7, -1.8, -5.7]
    e = ev("LHR-BA34F722 C: Resetting tracking: some future reason")
    assert e.data == {"reason": "other", "ms": None, "text": "some future reason"}


def test_backfacing_hits_all_shapes():
    e = ev("LHR-29C03CCA C: Dropped 3 rejected updates, 109934 back-facing hits, 123 non-clustered hits "
           "during the previous tracking session")
    assert e.kind is LogKind.BACKFACING_HITS
    assert e.data == {"rejected": 3, "backfacing": 109934, "non_clustered": 123}
    e = ev("LHR-329ADD4A H: Dropped 244598 back-facing hits during the previous tracking session")
    assert e.data == {"rejected": 0, "backfacing": 244598, "non_clustered": 0}
    e = ev("LHR-F19C1C47 C: Dropped 3 non-clustered hits during the previous tracking session")
    assert e.data == {"rejected": 0, "backfacing": 0, "non_clustered": 3}
    e = ev("LHR-F19C1C47 C: Dropped 1 rejected update, 156640 back-facing hits, 98 non-clustered hits "
           "during the previous tracking session")
    assert e.data == {"rejected": 1, "backfacing": 156640, "non_clustered": 98}


def test_imu_calibration_without_context_tag():
    e = ev("LHR-329ADD4A: Updated IMU calibration: Accel bias change 0.10m/s/s")
    assert e.kind is LogKind.IMU_CALIBRATION and e.serial == "LHR-329ADD4A"
    assert e.data == {"accel_bias": 0.10}


def test_sync_assert():
    e = ev("LHR-29C03CCA C: Assertion failed:  Explicit right sync does not match pending frame")
    assert e.kind is LogKind.SYNC_ASSERT
    assert e.data == {"text": "Explicit right sync does not match pending frame"}


# ---- tilt / universe ------------------------------------------------------------
def test_tilt_update_delta_and_extras():
    e = ev("Updating tilt for B3316995:  Old= 36.56, -5.00, 1.70864e-05 <- 12.90 deg -> "
           "new= 36.23, 8.82, 5.98581e-06   result=36.23, 8.82, 5.98581e-06")
    assert e.kind is LogKind.TILT_UPDATE and e.serial is None
    assert e.data["base"] == "B3316995" and e.data["delta_deg"] == 12.90
    assert (e.data["old_pitch"], e.data["old_roll"]) == (36.56, -5.00)
    assert (e.data["new_pitch"], e.data["new_roll"]) == (36.23, 8.82)
    assert e.data["variance"] == pytest.approx(5.98581e-06)


def test_universe_tilt():
    e = ev("Setting universe tilt from B3316995 via transform to global: pitch 36.23 deg roll 8.82 deg")
    assert e.kind is LogKind.UNIVERSE_TILT and e.serial is None
    assert e.data == {"base": "B3316995", "pitch": 36.23, "roll": 8.82}


def test_universe_changed():
    e = ev("Stopped tracking with universe 1749950692")
    assert e.kind is LogKind.UNIVERSE_CHANGED
    assert e.data == {"universe": "1749950692", "action": "stopped"}
    e = ev("Selected existing universe 1749950692 (ee93c79b is secondary)")
    assert e.kind is LogKind.UNIVERSE_CHANGED
    assert e.data == {"universe": "1749950692", "action": "selected", "base": "EE93C79B", "role": "secondary"}


# ---- radio / usb ------------------------------------------------------------------
def test_receiver_connect_disconnect():
    e = ev("LHR-29C03CCA: Connected to receiver 40E4AC9C61")
    assert e.kind is LogKind.RECEIVER_CONNECTED and e.serial == "LHR-29C03CCA"
    assert e.data == {"dongle": "40E4AC9C61"}
    e = ev("LHR-F19C1C47: Disconnected from receiver 0A6DB34FE3-LYM")
    assert e.kind is LogKind.RECEIVER_DISCONNECTED and e.data == {"dongle": "0A6DB34FE3-LYM"}
    w = World()
    w.add_log(ev("LHR-29C03CCA: Connected to receiver 40E4AC9C61"))
    assert w.devices_on_dongle("40E4AC9C61") == ["LHR-29C03CCA"]


def test_wireless_disconnected_with_and_without_device():
    e = ev("0B48A6C31B-RYB: Wireless controller  disconnected")
    assert e.kind is LogKind.WIRELESS_DISCONNECTED and e.serial is None
    assert e.data == {"dongle": "0B48A6C31B-RYB", "device": None}
    e = ev("40E4AC9C61: Wireless controller LHR-29C03CCA disconnected")
    assert e.serial == "LHR-29C03CCA"
    assert e.data == {"dongle": "40E4AC9C61", "device": "LHR-29C03CCA"}


@pytest.mark.parametrize("iface", ["IMU", "Optical", "VrController"])
def test_hid_error(iface):
    e = ev(f"Lighthouse {iface} HID device error")
    assert e.kind is LogKind.HID_ERROR and e.serial is None and e.data == {"interface": iface}


def test_dongle_enumerated():
    e = ev("    28DE, 2101: s/n E518136DA0: Watchman Dongle (interface 0)")
    assert e.kind is LogKind.DONGLE_ENUMERATED and e.serial is None
    assert e.data["dongle"] == "E518136DA0" and e.data["kind"] == "Watchman Dongle"
    assert (e.data["vid"], e.data["pid"], e.data["interface"]) == ("28DE", "2101", 0)
    e = ev("    28DE, 2102: s/n 0BDBF733E7-LYX: Valve VR Radio (interface 0)")
    assert e.data["dongle"] == "0BDBF733E7-LYX" and e.data["kind"] == "Valve VR Radio"
    e = ev("    28DE, 2102: s/n 0A6DB34FE3-LYM: Valve VR Radio & HMD Mic (interface 0)")
    assert e.data["kind"] == "Valve VR Radio" and e.data["product"] == "Valve VR Radio & HMD Mic"
    # the HMD's own interfaces are not dongles
    assert parse_line(LH + "    28DE, 2300: s/n LHR-329ADD4A: Optical (interface 1)") is None


def test_device_standby():
    e = ev("Device LHR-29C03CCA powering off upon entering standby.")
    assert e.kind is LogKind.DEVICE_STANDBY and e.serial == "LHR-29C03CCA" and e.data == {}


def test_server_start():
    e = ev("vrserver 2.18.4 startup with PID=30980, config=C:\\Steam\\config, runtime=C:\\SteamVR, arch=win64",
           prefix=P)
    assert e.kind is LogKind.SERVER_START and e.serial is None
    assert e.data["version"] == "2.18.4" and e.data["pid"] == 30980


# ---- OTHER ------------------------------------------------------------------------
def test_unrecognised_device_line_is_other_with_raw():
    e = ev("LHR-29C03CCA C: Some brand new complaint 42")
    assert e.kind is LogKind.OTHER and e.serial == "LHR-29C03CCA"
    assert e.data == {} and e.raw.endswith("Some brand new complaint 42")


def test_tagged_other_lines():
    e = ev("Moving base EE93C79B 56mm and 0.8 deg because of relationship with B3316995, "
           "which is closer to the origin")
    assert e.kind is LogKind.OTHER and e.serial is None
    assert e.data == {"what": "base_moved", "base": "EE93C79B", "mm": 56.0, "deg": 0.8, "reference": "B3316995"}
    e = ev("LHR-BA34F722 C: Calibration failed: no optical samples from base A0CD6092 for 2000ms")
    assert e.kind is LogKind.OTHER and e.data["what"] == "calibration_failed"
    assert e.data["base"] == "A0CD6092" and e.data["ms"] == 2000
    e = ev("LHR-329ADD4A H: No optical frames in past 5 seconds")
    assert e.data == {"what": "no_optical_frames", "seconds": 5.0}
    assert ev("LHR-329ADD4A H: Optical frames received again...").data["what"] == "optical_frames_resumed"
    assert ev("LHR-329ADD4A H: No base stations seen...").data["what"] == "no_bases_seen"
    e = ev("Best tilt from LHR-329ADD4A for base B3316995 pitch 36.23 roll 8.82 score 2.56")
    assert e.serial == "LHR-329ADD4A" and e.data["what"] == "best_tilt" and e.data["score"] == 2.56
    e = ev("LHR-29C03CCA C: LighthouseDirect: Optical data at time 582.938088 < filter at 582.952171")
    assert e.data["what"] == "late_optical_data" and e.data["late_ms"] == pytest.approx(14.083)


# ---- whole fixture -------------------------------------------------------------------
EMITTED = set(LogKind)


def test_fixture_covers_every_kind_and_is_json_able():
    import json
    events = scan_file(FIXTURE)
    assert len(events) > 200
    assert {e.kind for e in events} == EMITTED
    assert [e.t for e in events] == sorted(e.t for e in events)
    for e in events:
        json.dumps(to_json(e))
        assert e.raw and e.level in ("Info", "Warning", "Error")
        if e.serial is not None:
            assert e.serial.startswith("LHR-")
    # every base id anywhere is upper-case 8-hex
    for e in events:
        for key in ("base", "a", "b", "base_id", "reference"):
            v = e.data.get(key)
            if isinstance(v, str) and not v.startswith("S-"):
                assert len(v) == 8 and v == v.upper(), (key, v, e.raw)


def test_fixture_unparsed_device_lines_are_rare():
    events = scan_file(FIXTURE)
    untagged = [e for e in events if e.kind is LogKind.OTHER and "what" not in e.data]
    assert untagged == []


def test_scan_missing_file(tmp_path):
    assert scan_file(tmp_path / "nope.txt") == []


def test_summarize_and_cli(capsys):
    s = logtail.summarize(FIXTURE)
    assert s["by_kind"]["sob_add"] > 0 and s["device_lines"] > 0
    assert s["by_device"]["LHR-29C03CCA"]["sob_add"] > 0
    assert logtail.main([str(FIXTURE)]) == 0
    out = capsys.readouterr().out
    assert "sob_add" in out and "LHR-29C03CCA" in out and "OTHER on device lines" in out
    assert logtail.main([str(FIXTURE.with_name("missing.txt"))]) == 0
    assert "cannot read" in capsys.readouterr().out


# ---- tailer ---------------------------------------------------------------------------
L1 = LH + "LHR-329ADD4A H: SOB: add S-7\n"
L2 = LH + "LHR-329ADD4A H: SOB: add S-5 also seeing S-7\n"
L3 = LH + "LHR-F19C1C47 C: IMU went off scale.\n"
NOISE = LH + "Enumerating displays\n"


def append(path, text):
    with open(path, "ab") as fh:
        fh.write(text.encode("utf-8"))


def kinds(events):
    return [e.kind for e in events]


def test_tailer_starts_at_end_by_default(tmp_path):
    p = tmp_path / "vrserver.txt"
    append(p, L1 + L2)
    t = LogTailer(p)
    assert t.drain() == []
    append(p, NOISE + L3)
    assert kinds(t.drain()) == [LogKind.IMU_OFF_SCALE]
    assert t.drain() == []


def test_tailer_from_start(tmp_path):
    p = tmp_path / "vrserver.txt"
    append(p, L1 + L2)
    t = LogTailer(p, from_start=True)
    assert kinds(t.drain()) == [LogKind.SOB_ADD, LogKind.SOB_ADD]
    assert t.drain() == []


def test_tailer_partial_last_line(tmp_path):
    p = tmp_path / "vrserver.txt"
    append(p, L1)
    t = LogTailer(p)
    append(p, L2[:60])
    assert t.drain() == []
    append(p, L2[60:-1])          # still no newline
    assert t.drain() == []
    append(p, "\n" + L3[:10])
    got = t.drain()
    assert kinds(got) == [LogKind.SOB_ADD]
    assert got[0].data["base"] == "S-5" and got[0].data["seeing"] == [{"slot": "S-7", "base": None}]
    append(p, L3[10:])
    assert kinds(t.drain()) == [LogKind.IMU_OFF_SCALE]


def test_tailer_started_mid_line_skips_the_fragment(tmp_path):
    p = tmp_path / "vrserver.txt"
    append(p, L1 + L2[:60])
    t = LogTailer(p)
    append(p, L2[60:] + L3)
    assert kinds(t.drain()) == [LogKind.IMU_OFF_SCALE]


def test_tailer_crlf_and_bad_bytes(tmp_path):
    p = tmp_path / "vrserver.txt"
    t = LogTailer(p, from_start=True)
    with open(p, "ab") as fh:
        fh.write(L1.replace("\n", "\r\n").encode() + b"\xff\xfe garbage\r\n" + L3.encode())
    got = t.drain()
    assert kinds(got) == [LogKind.SOB_ADD, LogKind.IMU_OFF_SCALE]
    assert not got[0].raw.endswith("\r")


def test_tailer_file_not_existing_yet(tmp_path):
    p = tmp_path / "logs" / "vrserver.txt"
    t = LogTailer(p)               # from_start=False, but nothing exists
    assert t.drain() == []
    assert t.drain() == []
    p.parent.mkdir()
    append(p, L1)
    assert kinds(t.drain()) == [LogKind.SOB_ADD]   # everything in a new file is new


def test_tailer_rotation_by_rename(tmp_path):
    p = tmp_path / "vrserver.txt"
    prev = tmp_path / "vrserver.previous.txt"
    append(p, L1 * 50)
    t = LogTailer(p)
    append(p, L2)
    assert kinds(t.drain()) == [LogKind.SOB_ADD]
    # SteamVR writes one last line, exits, restarts: rename + new file
    append(p, L3)
    os.replace(p, prev)
    append(p, L1 + L2)             # new file, smaller than the old one
    got = t.drain()
    assert kinds(got) == [LogKind.IMU_OFF_SCALE, LogKind.SOB_ADD, LogKind.SOB_ADD]
    assert t.rotations == 1
    append(p, L3)
    assert kinds(t.drain()) == [LogKind.IMU_OFF_SCALE]
    assert t.rotations == 1


def test_tailer_rotation_to_larger_file(tmp_path):
    """New file already bigger than our offset: only identity/fingerprint reveals it."""
    p = tmp_path / "vrserver.txt"
    append(p, L1)
    t = LogTailer(p)
    assert t.drain() == []
    os.replace(p, tmp_path / "vrserver.previous.txt")
    append(p, L3 + L2 * 5)
    got = t.drain()
    assert kinds(got) == [LogKind.IMU_OFF_SCALE] + [LogKind.SOB_ADD] * 5
    assert t.rotations == 1


def test_tailer_rotation_seen_while_file_missing(tmp_path):
    p = tmp_path / "vrserver.txt"
    append(p, L1)
    t = LogTailer(p)
    append(p, L2)
    os.replace(p, tmp_path / "vrserver.previous.txt")
    assert kinds(t.drain()) == [LogKind.SOB_ADD]   # tail recovered from .previous
    assert t.drain() == []
    append(p, L3)
    assert kinds(t.drain()) == [LogKind.IMU_OFF_SCALE]
    assert t.rotations == 1


def test_tailer_truncation_in_place(tmp_path):
    p = tmp_path / "vrserver.txt"
    append(p, L1 * 20)
    t = LogTailer(p)
    assert t.drain() == []
    with open(p, "wb") as fh:      # same file identity, size shrinks
        fh.write(L3.encode())
    assert kinds(t.drain()) == [LogKind.IMU_OFF_SCALE]
    assert t.rotations == 1
    append(p, L2)
    assert kinds(t.drain()) == [LogKind.SOB_ADD]


def test_tailer_does_not_hold_the_file_open(tmp_path):
    p = tmp_path / "vrserver.txt"
    append(p, L1)
    t = LogTailer(p, from_start=True)
    t.drain()
    os.replace(p, tmp_path / "moved.txt")   # would raise on Windows if a handle were held
    assert t.drain() == []


def test_tailer_default_path():
    t = LogTailer()
    assert t.path.name == "vrserver.txt" and t.path.parent.name == "logs"
    assert isinstance(t.drain(), list)
