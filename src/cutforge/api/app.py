"""FastAPI application — runs, steps and the SSE log stream.

The UI (htmx + Alpine) is served from ``/`` and talks to these JSON/SSE endpoints.
Steps run in a background thread so the SSE stream can report progress live.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import threading
import unicodedata
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from cutforge.api import events
from cutforge.config.channels import list_channels
from cutforge.config.settings import get_settings
from cutforge.models.project import VideoProject, list_runs, list_runs_summary
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
            "runs": list_runs_summary(),
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
        channel_slug: str = Form("enkai"),
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

    @app.delete("/api/runs/{run_id}")
    def delete_run(run_id: str):
        project = VideoProject.load(run_id)
        run_dir = project.run_dir.resolve()
        output_dir = get_settings().output_dir.resolve()
        # Guard against path traversal — only delete inside the output dir.
        if output_dir not in run_dir.parents:
            return JSONResponse({"error": "invalid run path"}, status_code=400)
        shutil.rmtree(run_dir, ignore_errors=True)
        return {"status": "deleted", "run_id": run_id}

    # ---- Lyrics: genre suggestions (synchronous — quick) ----
    @app.post("/api/runs/{run_id}/suggest-genres")
    def suggest_genres(run_id: str, payload: dict | None = None):
        project = VideoProject.load(run_id)
        blend = (payload or {}).get("content_blend")
        suggestions = song_service.suggest_genres(project, content_blend=blend)
        return suggestions.model_dump()

    # ---- Mood suggestion (stateless — used on the create screen) ----
    @app.post("/api/suggest-mood")
    def suggest_mood(payload: dict | None = None):
        data = payload or {}
        character = (data.get("character") or "").strip()
        if not character:
            return JSONResponse({"error": "character is required"}, status_code=400)
        mood = song_service.suggest_mood(character, data.get("anime", ""))
        return {"mood": mood}

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

    # ---- Step output previews ----
    @app.get("/api/runs/{run_id}/output/lyrics")
    def output_lyrics(run_id: str):
        project = VideoProject.load(run_id)
        if not project.lyrics_path.exists():
            return JSONResponse({"error": "not found"}, status_code=404)
        suno = {}
        if project.suno_prompt_path.exists():
            suno = json.loads(project.suno_prompt_path.read_text(encoding="utf-8"))
        return {
            "lyrics": project.lyrics_path.read_text(encoding="utf-8"),
            "title": suno.get("title", ""),
            "style": suno.get("style", ""),
            "exclude": suno.get("exclude", ""),
            "suno_tips": suno.get("suno_tips", ""),
        }

    @app.get("/api/runs/{run_id}/output/metadata")
    def output_metadata(run_id: str):
        project = VideoProject.load(run_id)
        if not project.metadata_path.exists():
            return JSONResponse({"error": "not found"}, status_code=404)
        return json.loads(project.metadata_path.read_text(encoding="utf-8"))

    @app.get("/api/runs/{run_id}/output/reference")
    def output_reference(run_id: str):
        project = VideoProject.load(run_id)
        if not project.reference_profile_path.exists():
            return JSONResponse({"error": "not found"}, status_code=404)
        return json.loads(project.reference_profile_path.read_text(encoding="utf-8"))

    @app.get("/api/runs/{run_id}/output/thumbnail")
    def output_thumbnail(run_id: str):
        project = VideoProject.load(run_id)
        if not project.thumbnail_path.exists():
            return JSONResponse({"error": "not found"}, status_code=404)
        return FileResponse(str(project.thumbnail_path), media_type="image/jpeg")

    # ---- Open output folder in OS file explorer ----
    @app.post("/api/runs/{run_id}/open-folder")
    def open_folder(run_id: str):
        project = VideoProject.load(run_id)
        folder = str(project.run_dir)
        system = platform.system()
        if system == "Windows":
            os.startfile(folder)  # noqa: S606
        elif system == "Darwin":
            subprocess.Popen(["open", folder])
        else:
            subprocess.Popen(["xdg-open", folder])
        return {"status": "ok"}

    # ---- Run summary (for index page) ----
    @app.get("/api/runs/{run_id}/summary")
    def run_summary(run_id: str):
        try:
            project = VideoProject.load(run_id)
        except FileNotFoundError:
            return JSONResponse({"error": "not found"}, status_code=404)
        steps = wizard_state(project, runner.running_ids(run_id), runner.error_ids(run_id))
        done_count = sum(1 for s in steps if s["status"] == "done")
        return {
            "run_id": run_id,
            "character": project.character,
            "anime": project.anime,
            "language": project.language,
            "title": project.title,
            "done_count": done_count,
            "total_steps": len(steps),
        }

    return app


app = create_app()
