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
    prompt: str | None = None,
    on_log=None,
) -> list[dict]:
    """Return ``[{word, start, end}]`` with word-level timestamps.

    Caches raw Whisper output to ``cache_path`` when given. ``prompt`` is a decoding hint
    (typically the clean lyrics) — Whisper recognises processed/effect-heavy sung vocals
    far better with the expected words as a prior, instead of dropping or hallucinating
    over them. The cache key includes a hash of the prompt so a changed hint re-transcribes.
    """
    prompt_tag = None
    if prompt:
        import hashlib
        prompt_tag = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:10]

    if cache_path and cache_path.exists() and not refresh:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        # New cache format: {"prompt_tag": ..., "words": [...]}. Old format: bare list.
        if isinstance(cached, dict):
            if cached.get("prompt_tag") == prompt_tag:
                words = cached.get("words", [])
                if on_log:
                    on_log(f"Using cached Whisper transcript ({len(words)} words)")
                return words
            if on_log:
                on_log("Cached transcript prompt differs — re-transcribing with new hint.")
        elif prompt_tag is None:
            # Legacy bare-list cache: reuse only when no hint is requested.
            if on_log:
                on_log(f"Using cached Whisper transcript ({len(cached)} words)")
            return cached

    from openai import OpenAI

    settings = get_settings()
    client = OpenAI(api_key=settings.require("openai_api_key"))
    if on_log:
        on_log(f"Transcribing {audio_path.name} with Whisper (word timestamps"
               f"{', lyrics-hinted' if prompt else ''})...")

    kwargs = dict(
        model=settings.whisper_model,
        response_format="verbose_json",
        timestamp_granularities=["word"],
    )
    if prompt:
        # Whisper's prompt is capped near ~224 tokens; keep the hint compact.
        kwargs["prompt"] = prompt[:1000]
    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(file=f, **kwargs)

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
        cache_path.write_text(
            json.dumps({"prompt_tag": prompt_tag, "words": out}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return out
