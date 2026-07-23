"""FastAPI application — runs, steps and the SSE log stream.

The UI (htmx + Alpine) is served from ``/`` and talks to these JSON/SSE endpoints.
Steps run in a background thread so the SSE stream can report progress live.
"""
from __future__ import annotations

import re
import threading
import unicodedata
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from cutforge.api import events
from cutforge.config.channels import list_channels
from cutforge.models.project import VideoProject, list_runs
from cutforge.pipeline import runner
from cutforge.pipeline.steps import wizard_state
from cutforge.services import song_service

_HERE = Path(__file__).resolve().parent
_UI = _HERE.parent / "ui"
templates = Jinja2Templates(directory=str(_UI / "templates"))


def _slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[-\s]+", "-", text) or "run"


def create_app() -> FastAPI:
    app = FastAPI(title="CutForge")
    app.mount("/static", StaticFiles(directory=str(_UI / "static")), name="static")

    # ---- Pages ----
    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        return templates.TemplateResponse(request, "index.html", {
            "runs": list_runs(),
            "channels": list_channels(),
        })

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    def run_page(request: Request, run_id: str):
        project = VideoProject.load(run_id)
        return templates.TemplateResponse(request, "wizard.html", {
            "project": project,
            "steps": wizard_state(project, runner.running_ids(run_id), runner.error_ids(run_id)),
        })

    # ---- Run lifecycle ----
    @app.post("/api/runs")
    def create_run(
        character: str = Form(...),
        anime: str = Form(""),
        mood: str = Form(""),
        topic: str = Form(""),
        language: str = Form("en"),
        channel_slug: str = Form("zenkai-beats"),
    ):
        date = datetime.now().strftime("%Y%m%d")
        run_id = f"{date}-{_slugify(character or topic)}"
        project = VideoProject.create(
            run_id=run_id, character=character, anime=anime, mood=mood,
            topic=topic or character, language=language, channel_slug=channel_slug,
        )
        return JSONResponse({"run_id": project.run_id})

    @app.get("/api/runs/{run_id}/state")
    def run_state(run_id: str):
        project = VideoProject.load(run_id)
        return {
            "run_id": run_id,
            "title": project.title,
            "steps": wizard_state(project, runner.running_ids(run_id),
                                  runner.error_ids(run_id)),
        }

    # ---- Lyrics: genre suggestions (synchronous — quick) ----
    @app.post("/api/runs/{run_id}/suggest-genres")
    def suggest_genres(run_id: str):
        project = VideoProject.load(run_id)
        suggestions = song_service.suggest_genres(project)
        return suggestions.model_dump()

    # ---- Run a step (background thread; progress via SSE) ----
    @app.post("/api/runs/{run_id}/steps/{step_id}")
    def run_step(run_id: str, step_id: str, payload: dict | None = None):
        params = payload or {}
        force = bool(params.pop("force", False))
        log = events.make_logger(run_id)

        def worker():
            try:
                runner.run_step(run_id, step_id, params, force=force, log=log)
            finally:
                events.publish(run_id, events.DONE)

        threading.Thread(target=worker, daemon=True).start()
        return {"status": "started", "step": step_id}

    # ---- Upload the Suno track.mp3 ----
    @app.post("/api/runs/{run_id}/track")
    async def upload_track(run_id: str, file: UploadFile):
        project = VideoProject.load(run_id)
        project.audio_dir.mkdir(parents=True, exist_ok=True)
        data = await file.read()
        project.track_path.write_bytes(data)
        return {"status": "ok", "size": len(data)}

    # ---- SSE log stream ----
    @app.get("/api/runs/{run_id}/events")
    async def stream_events(run_id: str, request: Request):
        q = events.subscribe(run_id)

        async def gen():
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        msg = q.get(timeout=1.0)
                    except Exception:
                        continue
                    if msg == events.DONE:
                        yield {"event": "done", "data": "1"}
                    else:
                        yield {"event": "log", "data": msg}
            finally:
                events.unsubscribe(run_id, q)

        return EventSourceResponse(gen())

    return app


app = create_app()
