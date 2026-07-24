"""Rhythm analysis via librosa — BPM, beat grid and onset density.

Lazy-imports librosa (like ``whisper_client``/``anthropic_client`` lazy-import their
SDKs) so the app boots without paying librosa's heavy import cost until this step runs.
Output is cached on disk so re-running the reference step doesn't recompute.
"""
from __future__ import annotations

import json
from pathlib import Path


def analyze_rhythm(
    audio_path: Path,
    *,
    cache_path: Path | None = None,
    refresh: bool = False,
    on_log=None,
) -> dict:
    """Return rhythm features for ``audio_path``.

    Keys: ``bpm``, ``beat_count``, ``beat_times``, ``duration_sec``,
    ``estimated_time_signature``, ``onset_rate_per_sec``. Caches to ``cache_path``.
    """
    if cache_path and cache_path.exists() and not refresh:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if on_log:
            on_log(f"Using cached rhythm analysis (bpm={cached.get('bpm')})")
        return cached

    if on_log:
        on_log("Loading rhythm analyzer (librosa)...")
    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - only in a build missing librosa
        raise RuntimeError(
            "librosa is not available in this build — cannot analyze rhythm. "
            "Install with `pip install librosa soundfile`."
        ) from exc

    if on_log:
        on_log(f"Analyzing rhythm of {audio_path.name} (first run is slow - numba warmup)...")

    y, sr = librosa.load(str(audio_path), mono=True)
    duration_sec = float(librosa.get_duration(y=y, sr=sr))

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo) if not hasattr(tempo, "__len__") else float(tempo[0])
    beat_times = librosa.frames_to_time(beats, sr=sr).tolist()

    onsets = librosa.onset.onset_detect(y=y, sr=sr)
    onset_rate = float(len(onsets) / duration_sec) if duration_sec > 0 else 0.0

    result = {
        "bpm": round(bpm, 1),
        "beat_count": len(beat_times),
        "beat_times": [round(t, 3) for t in beat_times],
        "duration_sec": round(duration_sec, 2),
        "estimated_time_signature": 4,  # assume common time; librosa doesn't detect this
        "onset_rate_per_sec": round(onset_rate, 2),
    }

    if on_log:
        on_log(f"Rhythm: {result['bpm']} BPM, {result['beat_count']} beats, "
               f"{result['onset_rate_per_sec']} onsets/s")
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
