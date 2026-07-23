"""OpenAI Whisper word-timestamp transcription, with on-disk cache.

Ported from ``align_lyrics.py``. Whisper mishears words, so downstream we only trust
its *timing* and map the user's clean lyrics onto it. Raw output is cached so the
mapping step can be re-run without paying for another transcription.
"""
from __future__ import annotations

import json
from pathlib import Path

from cutforge.config.settings import get_settings


def transcribe_words(
    audio_path: Path,
    *,
    cache_path: Path | None = None,
    refresh: bool = False,
    on_log=None,
) -> list[dict]:
    """Return ``[{word, start, end}]`` with word-level timestamps.

    Caches raw Whisper output to ``cache_path`` when given.
    """
    if cache_path and cache_path.exists() and not refresh:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if on_log:
            on_log(f"Using cached Whisper transcript ({len(cached)} words)")
        return cached

    from openai import OpenAI

    settings = get_settings()
    client = OpenAI(api_key=settings.require("openai_api_key"))
    if on_log:
        on_log(f"Transcribing {audio_path.name} with Whisper (word timestamps)...")

    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model=settings.whisper_model,
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["word"],
        )

    words = getattr(result, "words", None) or []
    out: list[dict] = []
    for w in words:
        if isinstance(w, dict):
            out.append({"word": w["word"], "start": float(w["start"]), "end": float(w["end"])})
        else:
            out.append({"word": w.word, "start": float(w.start), "end": float(w.end)})

    if on_log:
        on_log(f"Whisper returned {len(out)} timed words.")
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
