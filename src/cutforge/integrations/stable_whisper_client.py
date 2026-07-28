"""stable-whisper transcription — word-level timestamps with phonetic alignment.

Produces word-level timing using stable_whisper (https://github.com/jianfch/stable-whisper),
which force-aligns the audio against its own transcription for much tighter timestamps
than the OpenAI Whisper API (~50ms accuracy vs ~500ms).

Results are cached to avoid re-running the model on every pipeline re-run.
"""
from __future__ import annotations

import json
from pathlib import Path


def transcribe_words(
    audio_path: Path,
    *,
    cache_path: Path | None = None,
    refresh: bool = False,
    language: str = "en",
    model_name: str = "small",
    on_log=None,
) -> list[dict]:
    """Return ``[{word, start, end}]`` with stable-whisper word-level timestamps.

    Uses ``mel_first=True`` which stabilizes alignment on musical/vocal tracks.
    Results are cached to ``cache_path``; pass ``refresh=True`` to force re-run.
    """
    cache_key = f"stable-{model_name}-{language}"

    if cache_path and cache_path.exists() and not refresh:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(cached, dict) and cached.get("backend") == cache_key:
            words = cached.get("words", [])
            if on_log:
                on_log(f"Using cached stable-whisper transcript ({len(words)} words)")
            return words

    try:
        import stable_whisper
    except ImportError:
        raise RuntimeError(
            "stable-whisper is not installed. Run: pip install stable-whisper"
        )

    if on_log:
        on_log(f"Transcribing {audio_path.name} with stable-whisper ({model_name})...")

    model = stable_whisper.load_model(model_name)
    result = model.transcribe(str(audio_path), language=language, mel_first=True)

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
        on_log(f"stable-whisper returned {len(out)} timed words.")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"backend": cache_key, "words": out}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return out


def transcribe_segments(
    audio_path: Path,
    *,
    language: str = "en",
    model_name: str = "small",
    on_log=None,
) -> list[dict]:
    """Return segment-level results ``[{text, start, end, words: [{word, start, end}]}]``.

    Used when we want to drive SRT directly from stable-whisper's own segmentation
    instead of re-aligning against lyrics.txt.
    """
    try:
        import stable_whisper
    except ImportError:
        raise RuntimeError(
            "stable-whisper is not installed. Run: pip install stable-whisper"
        )

    if on_log:
        on_log(f"Transcribing segments from {audio_path.name} with stable-whisper ({model_name})...")

    model = stable_whisper.load_model(model_name)
    result = model.transcribe(str(audio_path), language=language, mel_first=True)

    out: list[dict] = []
    for seg in result.segments:
        words = []
        for w in (seg.words or []):
            word_text = getattr(w, "word", "").strip()
            if not word_text:
                continue
            words.append({
                "word": word_text,
                "start": round(float(w.start), 3),
                "end": round(float(w.end), 3),
            })
        out.append({
            "text": seg.text.strip(),
            "start": round(float(seg.start), 3),
            "end": round(float(seg.end), 3),
            "words": words,
        })

    if on_log:
        on_log(f"stable-whisper returned {len(out)} segments.")
    return out
