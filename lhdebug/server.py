"""HTTP + WebSocket front for an Engine. See docs/protocol.md."""
from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import Body, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .model import to_json
from .sampler import Engine
from .store import Store

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
HEATMAP_METRICS = {"samples", "dropout", "jitter", "jumps", "update_rate", "coverage"}
# state is high-rate and disposable: a slow client drops frames instead of queueing them
DROPPABLE = {"state"}


def create_app(engine: Engine, sessions_dir: Path) -> FastAPI:
    app = FastAPI(title="lhdebug", docs_url=None, redoc_url=None)

    @app.get("/api/session")
    def session():
        return engine.session_info()

    @app.post("/api/control")
    def control(cmd: dict = Body(...)):
        action = cmd.get("action")
        if action == "stop":
            engine.set_recording(False)
        elif action == "resume":
            engine.set_recording(True)
        elif action == "clear":
            engine.clear()
        elif action is not None:
            return JSONResponse({"error": f"unknown action {action!r}"}, status_code=400)
        if cmd.get("window_s") is not None:
            try:
                engine.set_window(float(cmd["window_s"]))
            except (TypeError, ValueError):
                return JSONResponse({"error": "window_s must be a number"}, status_code=400)
        return engine.session_info()

    @app.get("/api/heatmap")
    def heatmap(metric: str = "dropout", serial: str | None = None):
        if metric not in HEATMAP_METRICS:
            return JSONResponse({"error": f"unknown metric {metric!r}"}, status_code=400)
        return engine.heatmap_snapshot(metric, serial or None)

    @app.get("/api/incidents")
    def incidents(limit: int = 200):
        with engine.lock:
            items = list(engine.incidents)[-limit:]
        return {"incidents": to_json(items[::-1])}

    @app.get("/api/log")
    def log(limit: int = 300):
        with engine.lock:
            items = list(engine.log)[-limit:]
        return {"events": to_json(items[::-1])}

    @app.get("/api/findings")
    def findings():
        return {"findings": engine.findings()}

    @app.get("/api/range")
    def range_profile():
        return engine.range_snapshot()

    @app.get("/api/lighthousedb")
    def lighthousedb():
        try:
            from . import lighthousedb as ldb
            return ldb.load().summary()
        except Exception as e:   # missing Steam install, unreadable file
            return JSONResponse({"error": str(e)}, status_code=404)

    @app.get("/api/sessions")
    def sessions():
        out = []
        for p in sorted(sessions_dir.glob("*.sqlite"), reverse=True):
            try:
                st = Store(p, read_only=True)
                meta = st.meta()
                st.close()
            except Exception:
                meta = {}
            out.append({"id": p.stem, "mode": meta.get("mode", "?"),
                        "started": float(meta.get("started", 0) or 0), "size": p.stat().st_size})
        return {"sessions": out}

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=400)

        def enqueue(msg: dict) -> None:
            if queue.full():
                if msg["type"] in DROPPABLE:
                    return
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(msg)

        def on_msg(msg: dict) -> None:      # called from the engine thread
            if msg["type"] in DROPPABLE and queue.qsize() > 5:
                return
            loop.call_soon_threadsafe(enqueue, msg)

        engine.subscribe(on_msg)
        try:
            await sock.send_json({"type": "hello", "session": engine.session_info()})
            await sock.send_json({"type": "findings", "findings": engine.findings()})
            while True:
                await sock.send_json(await queue.get())
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            engine.unsubscribe(on_msg)

    if WEB_DIR.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app
