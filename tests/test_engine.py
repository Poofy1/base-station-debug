"""Engine recording controls: rolling window, stop/resume, clear."""
from lhdebug.sampler import Engine
from lhdebug.sources.sim_source import SimSource
from lhdebug.store import Store

T0 = 1_800_000_000.0


def make(tmp_path, duration=150.0):
    store = Store(tmp_path / "s.sqlite")
    src = SimSource(realtime=False, duration=duration, faults="demo", t0=T0)
    return Engine(src, store=store, mode="sim"), store


def run(engine, seconds):
    msgs = []
    engine.subscribe(msgs.append)
    end = None
    while True:
        try:
            frame = engine.source.read()
        except StopIteration:
            break
        end = end or frame.t + seconds
        engine.step(frame)
        if frame.t >= end:
            break
    engine.unsubscribe(msgs.append)
    return msgs


def test_rolling_window_drops_old_data(tmp_path):
    engine, store = make(tmp_path)
    engine.set_window(60)
    run(engine, 140)
    now = engine.world.t
    assert now - T0 > 130
    slack = 60 + 11          # window + prune interval
    assert engine.incidents and engine.log
    assert all(i.t_end >= now - slack for i in engine.incidents)
    assert all(e.t >= now - slack for e in engine.log)
    oldest_sample = store.db.execute("SELECT MIN(t) FROM samples").fetchone()[0]
    assert oldest_sample >= now - slack
    assert all(i.t_end >= now - slack for i in store.incidents())
    # the demo scenario has incidents in its first minute; they must be gone
    assert min(i.t_start for i in engine.incidents) > T0 + 40


def test_stop_freezes_and_resume_continues(tmp_path):
    engine, store = make(tmp_path)
    run(engine, 40)
    engine.set_recording(False)
    assert engine.session_info()["recording"] is False
    assert engine.paused_at is not None
    n_inc, n_log = len(engine.incidents), len(engine.log)
    n_rows = store.db.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
    msgs = run(engine, 50)
    kinds = {m["type"] for m in msgs}
    assert "state" in kinds                      # live device state keeps flowing
    assert not kinds & {"incident", "log", "findings"}
    assert all(m["recording"] is False for m in msgs if m["type"] == "state")
    assert (len(engine.incidents), len(engine.log)) == (n_inc, n_log)
    assert store.db.execute("SELECT COUNT(*) FROM samples").fetchone()[0] == n_rows

    engine.set_recording(True)
    assert engine.paused_at is None
    run(engine, 40)
    assert store.db.execute("SELECT COUNT(*) FROM samples").fetchone()[0] > n_rows


def test_clear_forgets_everything(tmp_path):
    engine, store = make(tmp_path)
    run(engine, 100)
    assert engine.incidents and engine.findings()
    assert engine.heatmap_snapshot("samples", None)["cells"]
    msgs = []
    engine.subscribe(msgs.append)
    engine.clear()
    assert [m["type"] for m in msgs] == ["cleared", "session"]
    assert not engine.incidents and not engine.log and engine.findings() == []
    assert engine.heatmap_snapshot("samples", None)["cells"] == []
    assert store.incidents() == [] and store.log_events() == []
    assert store.db.execute("SELECT COUNT(*) FROM samples").fetchone()[0] == 0
    run(engine, 5)                               # still records afterwards
    assert store.db.execute("SELECT COUNT(*) FROM samples").fetchone()[0] > 0
