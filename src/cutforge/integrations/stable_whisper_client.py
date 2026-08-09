"""stable-ts force-alignment — phonetic word timestamps against known lyrics.

Uses ``model.align()`` instead of ``model.transcribe()``: we pass the exact lyrics
text so every word gets a timestamp, instead of relying on Whisper to transcribe
sung vocals correctly (it typically misses ~30% of words in musical tracks).
"""
from __future__ import annotations

import json
from pathlib import Path


def align_words(
    audio_path: Path,
    text: str,
    *,
    cache_path: Path | None = None,
    refresh: bool = False,
    language: str = "en",
    model_name: str = "medium",
    on_log=None,
) -> list[dict]:
    """Force-align ``text`` to ``audio_path``. Returns ``[{word, start, end}]``.

    Uses stable-ts ``model.align()`` which phonetically anchors every word in
    ``text`` to the audio — no words are dropped or hallucinated. Much more
    accurate than transcribe() on musical/vocal tracks.
    """
    cache_key = f"stable-align-{model_name}-{language}"

    if cache_path and cache_path.exists() and not refresh:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(cached, dict) and cached.get("backend") == cache_key:
            words = cached.get("words", [])
            if on_log:
                on_log(f"Using cached stable-ts alignment ({len(words)} words)")
            return words

    try:
        import stable_whisper
    except ImportError:
        raise RuntimeError("stable-ts is not installed. Run: pip install -U stable-ts")

    if on_log:
        on_log(f"Force-aligning {audio_path.name} with stable-ts ({model_name})...")

    model = stable_whisper.load_model(model_name)
    result = model.align(str(audio_path), text, language=language, suppress_silence=True)

    out: list[dict] = []
    for seg in result.segments:
        for w in (seg.words or []):
            word_text = getattr(w, "word", "").strip()
            if not word_text:
                continue
            out.append({
                "word": word_text,
                "start": round(float(w.start), 3),
                "end": round(float(w.end), 3),
            })

    if on_log:
        on_log(f"stable-ts aligned {len(out)} words.")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"backend": cache_key, "words": out}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return out


def transcribe_words(
    audio_path: Path,
    *,
    cache_path: Path | None = None,
    refresh: bool = False,
    language: str | None = None,
    model_name: str = "medium",
    on_log=None,
) -> list[dict]:
    """Transcribe ``audio_path`` from scratch (unknown lyrics). Returns ``[{word, start, end}]``.

    Uses stable-ts ``model.transcribe()`` — NOT ``align()`` — because here we do not yet
    know the lyrics (this is the reference-analysis path). Whisper's cloud ``whisper-1`` is
    prone to *repetition loops* on musical tracks: on instrumental/low-vocal sections it
    gets stuck echoing the last recognised phrase (e.g. an intro channel promo), which both
    spams the transcript and swallows the real lyrics. We disable
    ``condition_on_previous_text`` and enable Silero ``vad`` so silent/instrumental spans are
    masked instead of hallucinated over.

    ``language=None`` lets Whisper auto-detect (reference raps may be PT/EN/etc.).
    """
    cache_key = f"stable-transcribe-{model_name}-{language or 'auto'}"

    if cache_path and cache_path.exists() and not refresh:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(cached, dict) and cached.get("backend") == cache_key:
            words = cached.get("words", [])
            if on_log:
                on_log(f"Using cached stable-ts transcript ({len(words)} words)")
            return words

    try:
        import stable_whisper
    except ImportError:
        raise RuntimeError("stable-ts is not installed. Run: pip install -U stable-ts")

    if on_log:
        on_log(f"Transcribing {audio_path.name} with stable-ts ({model_name}, anti-loop)...")

    model = stable_whisper.load_model(model_name)
    result = model.transcribe(
        str(audio_path),
        language=language,
        # Anti-hallucination: don't feed prior output back as a prompt, so the model
        # can't lock into a repetition loop on instrumental/silent sections.
        condition_on_previous_text=False,
        # Mask non-speech spans (Silero VAD) instead of transcribing over them.
        vad=True,
        suppress_silence=True,
        only_voice_freq=True,
    )

    out: list[dict] = []
    for seg in result.segments:
        for w in (seg.words or []):
            word_text = getattr(w, "word", "").strip()
            if not word_text:
                continue
            out.append({
                "word": word_text,
                "start": round(float(w.start), 3),
                "end": round(float(w.end), 3),
            })

    if on_log:
        on_log(f"stable-ts transcribed {len(out)} words.")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"backend": cache_key, "words": out}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return out
