"""Reference-rap service — download a YouTube rap, transcribe it, analyze its rhythm.

Produces a small "reference profile" (lyrics transcript + BPM + flow metrics) that the
lyrics-generation step (``song_service.generate_package``) detects on disk and uses to
write an ORIGINAL, non-infringing song with a matching flow/energy/structure.

Reuses ``youtube_dl.download_audio`` (audio-only fetch), ``whisper_client.transcribe_words``
(word timestamps, cached) and ``librosa_client.analyze_rhythm`` (BPM/beat, cached).
"""
from __future__ import annotations

import json

from cutforge.integrations import librosa_client, whisper_client, youtube_dl
from cutforge.models.project import VideoProject


def _reconstruct_lyrics(words: list[dict]) -> str:
    """Rebuild a plain transcript from Whisper word tokens."""
    if not words:
        return ""
    # Whisper word tokens usually carry no leading space; join with spaces.
    return " ".join(w["word"].strip() for w in words if w.get("word", "").strip())


def analyze_reference(project: VideoProject, url: str, *, refresh: bool = False,
                      on_log=None) -> dict:
    """Download, transcribe and rhythm-analyze the reference rap. Returns the profile."""
    log = on_log or (lambda _m: None)

    # 1. Download audio-only.
    meta = youtube_dl.download_audio(url, project.reference_audio_path, on_log=log)

    # 2. Transcribe (cached).
    words = whisper_client.transcribe_words(
        project.reference_audio_path,
        cache_path=project.reference_whisper_path,
        refresh=refresh,
        on_log=log,
    )
    transcript = _reconstruct_lyrics(words)
    project.reference_lyrics_path.parent.mkdir(parents=True, exist_ok=True)
    project.reference_lyrics_path.write_text(transcript, encoding="utf-8")

    # 3. Rhythm analysis (cached).
    rhythm = librosa_client.analyze_rhythm(
        project.reference_audio_path,
        cache_path=project.reference_rhythm_path,
        refresh=refresh,
        on_log=log,
    )

    # 4. Flow metrics from Whisper timing.
    bpm = rhythm.get("bpm", 0) or 0
    word_count = len(words)
    span = 0.0
    if words:
        span = max(0.0, float(words[-1]["end"]) - float(words[0]["start"]))
    words_per_sec = round(word_count / span, 2) if span > 0 else 0.0
    syllables_per_beat = round(words_per_sec * 60.0 / bpm, 2) if bpm > 0 else 0.0

    # 5. Assemble the profile (omit the large beat_times array from the prompt payload).
    profile = {
        "source_url": url,
        "source_title": meta.get("title", ""),
        "duration_sec": rhythm.get("duration_sec"),
        "bpm": rhythm.get("bpm"),
        "time_signature": rhythm.get("estimated_time_signature"),
        "onset_rate_per_sec": rhythm.get("onset_rate_per_sec"),
        "flow": {
            "word_count": word_count,
            "words_per_sec": words_per_sec,
            "syllables_per_beat": syllables_per_beat,
        },
        "transcript": transcript,
    }

    project.reference_profile_path.write_text(
        json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    project.reference_url = url
    project.save()

    log(f"Reference profile saved: {profile['bpm']} BPM, {word_count} words.")
    return profile


def load_reference_profile(project: VideoProject) -> dict | None:
    """Return the saved reference profile, or None if this run has no reference."""
    if not project.reference_profile_path.exists():
        return None
    return json.loads(project.reference_profile_path.read_text(encoding="utf-8"))
