"""Step runner — executes one pipeline step, streaming log lines to a callback.

Idempotent: a step whose output already exists is skipped unless ``force=True``. The
runner tracks which steps are running/errored per run so the UI can reflect live state.
Each step maps to a service call; extra parameters (genre, url, color...) arrive via
``params``.
"""
from __future__ import annotations

import threading
import traceback
from typing import Callable

from cutforge.models.project import VideoProject
from cutforge.pipeline.steps import STEP_BY_ID

LogFn = Callable[[str], None]

# Per-run live state (module-level; single-process desktop app).
_running: dict[str, set[str]] = {}   # run_id -> {step_id}
_errors: dict[str, set[str]] = {}    # run_id -> {step_id}
_lock = threading.Lock()


def running_ids(run_id: str) -> set[str]:
    with _lock:
        return set(_running.get(run_id, set()))


def error_ids(run_id: str) -> set[str]:
    with _lock:
        return set(_errors.get(run_id, set()))


def _mark(run_id: str, step_id: str, *, running: bool | None = None,
          error: bool | None = None) -> None:
    with _lock:
        if running is True:
            _running.setdefault(run_id, set()).add(step_id)
        elif running is False:
            _running.get(run_id, set()).discard(step_id)
        if error is True:
            _errors.setdefault(run_id, set()).add(step_id)
        elif error is False:
            _errors.get(run_id, set()).discard(step_id)


def _execute(step_id: str, project: VideoProject, params: dict, log: LogFn):
    """Dispatch a step id to its service call. Returns a small result dict."""
    from cutforge.services import (
        song_service, footage_service, alignment_service,
        caption_service, thumbnail_service, metadata_service, premiere_service,
        reference_service,
    )

    if step_id == "reference":
        url = params.get("url")
        if not url:
            raise ValueError("Missing 'url' for reference analysis.")
        index = int(params.get("index", 0))
        profile = reference_service.analyze_reference(
            project, url, index, refresh=params.get("refresh", False),
            manual_lyrics=params.get("lyrics", ""),
            lyrics_source=params.get("lyrics_source", "manual"), on_log=log)
        return {"bpm": profile.get("bpm"), "title": profile.get("source_title"), "index": index}

    if step_id == "lyrics":
        genre = params.get("genre")
        if not genre:
            raise ValueError("Missing 'genre' — pick a genre direction first.")
        pkg = song_service.generate_package(
            project, genre, is_vs=params.get("is_vs", False),
            ref_index=int(params.get("ref_index", 0)),
            mode=params.get("mode", "original"),
            follow_structure=params.get("follow_structure", False),
            new_hook=params.get("new_hook", True),
            refresh=params.get("refresh", False), on_log=log)
        log(f"Song: {pkg.title}")
        return {"title": pkg.title, "style": pkg.style}

    if step_id == "footage":
        url = params.get("url")
        if not url:
            raise ValueError("Missing 'url' for footage download.")
        meta = footage_service.download_footage(project, url, on_log=log)
        return {"title": meta.get("title"), "duration": meta.get("duration")}

    if step_id == "align":
        alignment = alignment_service.align_project(
            project,
            refresh=params.get("refresh", False),
            backend=params.get("backend", "stable"),
            on_log=log)
        return {"lines": alignment.line_count, "words": alignment.word_count}

    if step_id == "captions":
        caption_service.generate_captions(
            project,
            max_chunk_duration=float(params.get("max_chunk_duration", 1.5)),
            on_log=log)
        return {"ok": True}

    if step_id == "thumbnail":
        path = thumbnail_service.generate_thumbnail(
            project, genre_badge=params.get("genre_badge"), on_log=log)
        return {"path": path}

    if step_id == "metadata":
        meta = metadata_service.generate_metadata(project, on_log=log)
        return {"title": meta.get("title")}

    if step_id == "premiere":
        path = premiere_service.build_project(project, on_log=log)
        return {"path": str(path)}

    raise ValueError(f"Unknown step id: {step_id}")


def run_step(run_id: str, step_id: str, params: dict | None = None, *,
             force: bool = False, log: LogFn | None = None) -> dict:
    """Run one step synchronously. Returns {status, result?, error?}.

    ``log`` receives progress lines (also used to feed the SSE stream).
    """
    params = params or {}
    log = log or (lambda _m: None)
    if step_id not in STEP_BY_ID:
        raise ValueError(f"Unknown step id: {step_id}")

    project = VideoProject.load(run_id)
    step = STEP_BY_ID[step_id]

    # The reference step's is_done only knows about index 0, so adding a NEW reference
    # (index 1, 2, …) would otherwise be short-circuited by the generic gate below.
    # A request for an index whose profile doesn't exist yet is not "done" — let it run.
    if step_id == "reference" and not force:
        idx = int(params.get("index", 0))
        if not project.ref_profile_path(idx).exists():
            force = True

    if step.is_done(project) and not force:
        log(f"[{step_id}] já concluído — pulando (use force para refazer).")
        return {"status": "skipped"}

    if not step.can_run(project) and not force:
        log(f"[{step_id}] bloqueado: {step.requires_hint}")
        return {"status": "locked", "hint": step.requires_hint}

    _mark(run_id, step_id, running=True, error=False)
    log(f"[{step_id}] iniciando…")
    try:
        # A forced re-run should regenerate cached sub-artifacts (creative direction,
        # and — when explicitly asked — the mined lore) rather than reuse stale ones.
        if force and "refresh" not in params:
            params = {**params, "refresh": True}
        result = _execute(step_id, project, params, log)
        log(f"[{step_id}] concluído.")
        return {"status": "done", "result": result}
    except Exception as exc:  # noqa: BLE001 — surface any failure to the UI
        _mark(run_id, step_id, error=True)
        log(f"[{step_id}] ERRO: {exc}")
        log(traceback.format_exc())
        return {"status": "error", "error": str(exc)}
    finally:
        _mark(run_id, step_id, running=False)
