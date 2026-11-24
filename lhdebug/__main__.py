"""lhdebug command line.

  python -m lhdebug run        attach to SteamVR and record in the background
  python -m lhdebug simulate   synthetic rig with injected faults (no headset needed)
  python -m lhdebug report S   print the findings of a recorded session
  python -m lhdebug sessions   list recorded sessions
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import webbrowser
from pathlib import Path


def _serve(engine, args) -> int:
    import uvicorn
    from .server import create_app

    app = create_app(engine, Path(args.sessions))
    engine.start()
    url = f"http://{args.host}:{args.port}/"
    print(f"lhdebug [{engine.mode}] dashboard: {url}")
    if engine.store:
        print(f"recording to {engine.store.path}")
    if args.open:
        webbrowser.open(url)
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        engine.stop()
        if engine.store:
            engine.store.close()
    return 0


def cmd_run(args) -> int:
    from .logtail import LogTailer
    from .sampler import Engine
    from .sources.openvr_source import OpenVRSource
    from .store import Store

    store = None if args.no_record else Store.new_session(args.sessions, "live")
    tailer = LogTailer()
    engine = Engine(OpenVRSource(rate=args.rate), log_source=tailer, store=store, mode="live")
    # Replay what the driver already logged this SteamVR run so we know which bases
    # each device sees and which dongle it is on. Context only: none of it is shown
    # as new events.
    try:
        from .logtail import scan_file
        from .model import LogKind
        events = scan_file(tailer.path)
        starts = [i for i, ev in enumerate(events) if ev.kind is LogKind.SERVER_START]
        for ev in events[starts[-1] if starts else 0:]:
            engine.world.add_log(ev)
    except Exception as e:      # no log yet, unreadable: carry on without history
        print(f"(could not preload the SteamVR log: {e})")
    print("waiting for SteamVR (this never launches it)...")
    return _serve(engine, args)


def cmd_simulate(args) -> int:
    from .sampler import Engine
    from .sources.sim_source import SimSource
    from .store import Store

    store = Store.new_session(args.sessions, "sim") if args.record else None
    faults = None if args.clean else "demo"
    engine = Engine(SimSource(rate=args.rate, faults=faults), store=store, mode="sim")
    return _serve(engine, args)


def _resolve(sessions: str, name: str) -> Path:
    p = Path(name)
    if p.is_file():
        return p
    root = Path(sessions)
    if name == "latest":
        found = sorted(root.glob("*.sqlite"))
        if found:
            return found[-1]
    for cand in (root / name, root / f"{name}.sqlite"):
        if cand.is_file():
            return cand
    raise SystemExit(f"no such session: {name}")


def cmd_report(args) -> int:
    from .store import Store

    path = _resolve(args.sessions, args.session)
    st = Store(path, read_only=True)
    meta, devices, incidents = st.meta(), st.devices(), st.incidents()
    st.close()
    started = float(meta.get("started", 0) or 0)
    ended = float(meta.get("ended", 0) or 0)
    print(f"session {path.stem}  mode={meta.get('mode', '?')}  "
          f"started {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(started))}"
          + (f"  length {(ended - started) / 60:.1f} min" if ended else "  (not cleanly closed)"))
    label = {s: d.label() for s, d in devices.items()}

    print(f"\nIncidents: {len(incidents)}")
    tally: dict[tuple[str, str, str], list] = {}
    for inc in incidents:
        tally.setdefault((label.get(inc.serial, inc.serial), inc.kind.value, inc.top_cause), []).append(inc)
    for (dev, kind, cause), items in sorted(tally.items(), key=lambda kv: -len(kv[1])):
        total = sum(i.duration for i in items)
        print(f"  {len(items):4d}x  {dev:<14} {kind:<15} -> {cause:<13} ({total:.1f}s total)")

    findings = json.loads(meta.get("findings", "[]"))
    print(f"\nFindings: {len(findings)}")
    for i, f in enumerate(findings, 1):
        print(f"\n{i}. [{f['cause']}  {f['score']:.0%}] {f['title']}")
        print(f"   {f['detail']}")
        print(f"   Fix: {f['fix']}")
    return 0


def cmd_sessions(args) -> int:
    for p in sorted(Path(args.sessions).glob("*.sqlite")):
        print(f"{p.stem}  {p.stat().st_size / 1e6:.1f} MB")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="lhdebug", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", default="sessions", help="directory for session recordings")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def serving(p):
        p.add_argument("--host", default="127.0.0.1")
        p.add_argument("--port", type=int, default=8777)
        p.add_argument("--rate", type=float, default=200.0, help="pose sampling rate (Hz)")
        p.add_argument("--open", action="store_true", help="open the dashboard in a browser")

    p = sub.add_parser("run", help="attach to SteamVR and record")
    serving(p)
    p.add_argument("--no-record", action="store_true")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("simulate", help="synthetic rig with injected faults")
    serving(p)
    p.add_argument("--clean", action="store_true", help="no injected faults")
    p.add_argument("--record", action="store_true")
    p.set_defaults(fn=cmd_simulate)

    p = sub.add_parser("report", help="print findings of a recorded session")
    p.add_argument("session", nargs="?", default="latest")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("sessions", help="list recorded sessions")
    p.set_defaults(fn=cmd_sessions)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
