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
        suggestions = song_service.suggest_genres(project)
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

    # ---- References list ----
    @app.get("/api/runs/{run_id}/references")
    def list_references(run_id: str):
        project = VideoProject.load(run_id)
        from cutforge.services import reference_service
        profiles = reference_service.load_all_reference_profiles(project)
        return {
            "urls": project.reference_urls,
            "profiles": [
                {"index": i, "source_title": p.get("source_title", ""),
                 "bpm": p.get("bpm"), "source_url": p.get("source_url", ""),
                 "transcript": p.get("transcript", ""),
                 "lyrics_source": p.get("lyrics_source", "whisper")}
                for i, p in enumerate(profiles)
            ],
        }

    @app.post("/api/runs/{run_id}/references/{index}/lyrics")
    def set_reference_lyrics(run_id: str, index: int, payload: dict | None = None):
        data = payload or {}
        lyrics = (data.get("lyrics") or "").strip()
        source = (data.get("source") or "manual").strip() or "manual"
        project = VideoProject.load(run_id)
        from cutforge.services import reference_service
        profile = reference_service.set_reference_lyrics(project, index, lyrics, source=source)
        if profile is None:
            return JSONResponse({"error": "reference not analyzed yet"}, status_code=404)
        return {
            "index": index,
            "source_title": profile.get("source_title", ""),
            "transcript": profile.get("transcript", ""),
            "lyrics_source": profile.get("lyrics_source", source),
        }

    @app.get("/api/runs/{run_id}/references/{index}/subtitles")
    def fetch_reference_subtitles(run_id: str, index: int, lang: str = "en", url: str = ""):
        project = VideoProject.load(run_id)
        # ``url`` may be passed directly when fetching BEFORE the reference is analyzed
        # (at add time). Otherwise fall back to the stored URL for that index.
        target = (url or "").strip()
        if not target:
            urls = project.reference_urls
            if index >= len(urls) or not urls[index]:
                return JSONResponse({"error": "reference not found"}, status_code=404)
            target = urls[index]
        import tempfile
        from cutforge.integrations import youtube_dl
        log = events.make_logger(run_id)
        with tempfile.TemporaryDirectory() as tmp:
            text = youtube_dl.download_subtitles(
                target, Path(tmp), lang=(lang or "en").strip(), on_log=log)
        return {"text": text, "available": bool(text.strip())}

    @app.get("/api/runs/{run_id}/references/{index}/subtitle-langs")
    def list_reference_subtitle_langs(run_id: str, index: int, url: str = ""):
        project = VideoProject.load(run_id)
        target = (url or "").strip()
        if not target:
            urls = project.reference_urls
            if index >= len(urls) or not urls[index]:
                return JSONResponse({"error": "reference not found"}, status_code=404)
            target = urls[index]
        from cutforge.integrations import youtube_dl
        try:
            langs = youtube_dl.list_manual_subtitles(target)
        except Exception as exc:  # noqa: BLE001 — surface probe failures to the UI
            return JSONResponse({"error": str(exc)}, status_code=502)
        return {"langs": langs}

    @app.delete("/api/runs/{run_id}/references/{index}")
    def delete_reference(run_id: str, index: int):
        project = VideoProject.load(run_id)
        from cutforge.services import reference_service
        reference_service.remove_reference(project, index)
        return {"status": "deleted", "index": index}

    # ---- Lore profile (character knowledge mined from reference) ----
    @app.get("/api/runs/{run_id}/lore-profile")
    def get_lore_profile(run_id: str):
        project = VideoProject.load(run_id)
        from cutforge.services import lore_service
        profiles = lore_service.load_all_lore_profiles(project)
        merged = lore_service.merge_lore_profiles(profiles)
        if not merged:
            return JSONResponse({"error": "not found"}, status_code=404)
        return merged.model_dump()

    @app.post("/api/runs/{run_id}/lore-profile/refresh")
    def refresh_lore_profile(run_id: str, payload: dict | None = None):
        data = payload or {}
        index = int(data.get("index", 0))
        instruction = (data.get("instruction") or "").strip()
        project = VideoProject.load(run_id)
        from cutforge.services import lore_service
        profile = lore_service.mine_reference_lore(
            project, index=index, refresh=True, user_instruction=instruction)
        if not profile:
            return JSONResponse({"error": "no reference to mine"}, status_code=404)
        # Return merged view so the UI always sees the full picture.
        all_profiles = lore_service.load_all_lore_profiles(project)
        merged = lore_service.merge_lore_profiles(all_profiles)
        return merged.model_dump() if merged else profile.model_dump()

    # ---- Narrative structure (proven skeleton synthesized from the reference[s]) ----
    @app.get("/api/runs/{run_id}/structure-profile")
    def get_structure_profile(run_id: str):
        project = VideoProject.load(run_id)
        from cutforge.services import structure_service
        profile = structure_service.load_structure_profile(project)
        if not profile:
            return JSONResponse({"error": "not found"}, status_code=404)
        return profile.model_dump()

    @app.post("/api/runs/{run_id}/structure-profile/refresh")
    def refresh_structure_profile(run_id: str, payload: dict | None = None):
        data = payload or {}
        instruction = (data.get("instruction") or "").strip()
        project = VideoProject.load(run_id)
        from cutforge.services import structure_service
        profile = structure_service.extract_structure_profile(
            project, refresh=True, user_instruction=instruction)
        if not profile:
            return JSONResponse({"error": "no reference to analyze"}, status_code=404)
        return profile.model_dump()

    # ---- Shared story (the story to retell — "rewrite the story" mode) ----
    @app.get("/api/runs/{run_id}/story-profile")
    def get_story_profile(run_id: str):
        project = VideoProject.load(run_id)
        from cutforge.services import story_service
        profile = story_service.load_story_profile(project)
        if not profile:
            return JSONResponse({"error": "not found"}, status_code=404)
        return profile.model_dump()

    @app.post("/api/runs/{run_id}/story-profile/refresh")
    def refresh_story_profile(run_id: str, payload: dict | None = None):
        data = payload or {}
        instruction = (data.get("instruction") or "").strip()
        project = VideoProject.load(run_id)
        from cutforge.services import story_service
        profile = story_service.extract_story_profile(
            project, refresh=True, user_instruction=instruction)
        if not profile:
            return JSONResponse({"error": "no reference to analyze"}, status_code=404)
        return profile.model_dump()

    # ---- Creative direction (original-song brief) ----
    @app.get("/api/runs/{run_id}/creative-direction")
    def get_creative_direction(run_id: str):
        project = VideoProject.load(run_id)
        if not project.creative_direction_path.exists():
            return JSONResponse({"error": "not found"}, status_code=404)
        import json as _json
        return _json.loads(project.creative_direction_path.read_text(encoding="utf-8"))

    @app.post("/api/runs/{run_id}/creative-direction/refresh")
    def refresh_creative_direction(run_id: str, payload: dict | None = None):
        data = payload or {}
        genre = (data.get("genre") or "").strip()
        if not genre:
            return JSONResponse({"error": "genre is required"}, status_code=400)
        ref_index = int(data.get("ref_index", 0))
        instruction = (data.get("instruction") or "").strip()
        project = VideoProject.load(run_id)
        from cutforge.services import lore_service, reference_service
        music_profile = reference_service.load_reference_profile(project, index=ref_index)
        all_lores = lore_service.load_all_lore_profiles(project)
        lore_profile = lore_service.merge_lore_profiles(all_lores)
        direction = song_service.plan_creative_direction(
            project, genre,
            music_profile=music_profile, lore_profile=lore_profile,
            user_instruction=instruction,
            refresh=True,
        )
        return direction.model_dump()

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

    # ---- Thumbnail style-reference images (optional, per run) ----
    _IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

    @app.post("/api/runs/{run_id}/thumbnail-refs")
    async def upload_thumbnail_refs(run_id: str, files: list[UploadFile]):
        project = VideoProject.load(run_id)
        refs_dir = project.thumbnail_refs_dir
        refs_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        # Continue numbering after any existing refs so multiple uploads accumulate.
        existing = len(project.thumbnail_ref_paths())
        for file in files:
            ext = Path(file.filename or "").suffix.lower()
            if ext not in _IMG_EXTS:
                continue
            data = await file.read()
            (refs_dir / f"ref_{existing + saved + 1:02d}{ext}").write_bytes(data)
            saved += 1
        return {"status": "ok", "count": len(project.thumbnail_ref_paths()), "added": saved}

    @app.get("/api/runs/{run_id}/thumbnail-refs")
    def list_thumbnail_refs(run_id: str):
        project = VideoProject.load(run_id)
        return {"count": len(project.thumbnail_ref_paths())}

    @app.delete("/api/runs/{run_id}/thumbnail-refs")
    def clear_thumbnail_refs(run_id: str):
        project = VideoProject.load(run_id)
        for p in project.thumbnail_ref_paths():
            p.unlink(missing_ok=True)
        return {"status": "ok", "count": 0}

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

    @app.get("/api/runs/{run_id}/output/reference-lore")
    def output_reference_lore(run_id: str):
        project = VideoProject.load(run_id)
        if not project.reference_lore_profile_path.exists():
            return JSONResponse({"error": "not found"}, status_code=404)
        return json.loads(project.reference_lore_profile_path.read_text(encoding="utf-8"))

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
