"""Reference-rap service — download a YouTube rap, transcribe it, analyze its rhythm.

Produces a small "reference profile" (lyrics transcript + BPM + flow metrics) that the
lyrics-generation step (``song_service.generate_package``) detects on disk and uses to
write an ORIGINAL, non-infringing song with a matching flow/energy/structure.

Multiple references are supported. Each is stored under ``reference/{index}/`` (e.g.
``reference/0/``, ``reference/1/``). The first reference (index 0) is the "primary" —
its sonic DNA (BPM/flow) drives the style. All references contribute lore.

Legacy runs that used the old flat layout (``reference/reference_profile.json``) are
transparently read as a single reference at index 0.
"""
from __future__ import annotations

import json

from cutforge.integrations import librosa_client, stable_whisper_client, youtube_dl
from cutforge.models.project import VideoProject


def _reconstruct_lyrics(words: list[dict]) -> str:
    """Rebuild a plain transcript from Whisper word tokens."""
    if not words:
        return ""
    return " ".join(w["word"].strip() for w in words if w.get("word", "").strip())


def analyze_reference(project: VideoProject, url: str, index: int = 0,
                      *, refresh: bool = False, manual_lyrics: str = "",
                      lyrics_source: str = "manual", on_log=None) -> dict:
    """Download, transcribe and rhythm-analyze one reference rap. Returns the profile.

    ``index`` selects the sub-folder (0 = primary, 1 = second reference, …).
    The result is written to ``reference/{index}/reference_profile.json``.
    ``project.reference_urls`` is updated and saved.

    ``manual_lyrics``: when non-empty, these lyrics are used verbatim as the reference
    transcript and Whisper transcription is skipped entirely. Whisper only runs as a
    fallback when no manual lyrics are supplied (its rap transcription is unreliable).
    The audio is still downloaded and rhythm-analyzed either way — BPM/flow come from
    librosa, not Whisper.

    ``lyrics_source``: how ``manual_lyrics`` was obtained — ``"manual"`` (pasted by hand)
    or ``"subtitles"`` (fetched from the channel's manual YouTube subtitles). Recorded on
    the profile for the UI badge. Ignored when Whisper runs (then it's ``"whisper"``).
    """
    log = on_log or (lambda _m: None)

    audio_path = project.ref_audio_path(index)
    whisper_path = project.ref_whisper_path(index)
    rhythm_path = project.ref_rhythm_path(index)
    lyrics_path = project.ref_lyrics_path(index)
    profile_path = project.ref_profile_path(index)

    audio_path.parent.mkdir(parents=True, exist_ok=True)

    meta = youtube_dl.download_audio(url, audio_path, on_log=log)

    rhythm = librosa_client.analyze_rhythm(
        audio_path,
        cache_path=rhythm_path,
        refresh=refresh,
        on_log=log,
    )
    bpm = rhythm.get("bpm", 0) or 0

    manual_lyrics = (manual_lyrics or "").strip()
    if manual_lyrics:
        log(f"Reference {index}: using pasted lyrics ({len(manual_lyrics.split())} words) — skipping Whisper.")
        transcript = manual_lyrics
        lyrics_path.write_text(transcript, encoding="utf-8")
        word_count = len(transcript.split())
        # No word-level timings without Whisper — estimate flow from duration + BPM.
        span = float(rhythm.get("duration_sec") or 0.0)
    else:
        words = stable_whisper_client.transcribe_words(
            audio_path,
            cache_path=whisper_path,
            refresh=refresh,
            language=None,
            on_log=log,
        )
        transcript = _reconstruct_lyrics(words)
        lyrics_path.write_text(transcript, encoding="utf-8")
        word_count = len(words)
        span = 0.0
        if words:
            span = max(0.0, float(words[-1]["end"]) - float(words[0]["start"]))

    words_per_sec = round(word_count / span, 2) if span > 0 else 0.0
    syllables_per_beat = round(words_per_sec * 60.0 / bpm, 2) if bpm > 0 else 0.0

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
        "lyrics_source": lyrics_source if manual_lyrics else "whisper",
    }

    profile_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")

    # Keep reference_urls in sync.
    urls = list(project.reference_urls)
    while len(urls) <= index:
        urls.append("")
    urls[index] = url
    project.reference_urls = urls
    project.save()

    log(f"Reference {index} profile saved: {profile['bpm']} BPM, {word_count} words.")
    return profile


def load_reference_profile(project: VideoProject, index: int = 0) -> dict | None:
    """Return the saved profile for one reference, or None if it does not exist.

    Falls back to the legacy flat layout (``reference/reference_profile.json``) when
    the indexed sub-folder does not exist and ``index == 0``.
    """
    indexed_path = project.ref_profile_path(index)
    if indexed_path.exists():
        return json.loads(indexed_path.read_text(encoding="utf-8"))
    # Legacy flat layout (runs created before multi-reference support).
    if index == 0 and project.reference_profile_path.exists():
        return json.loads(project.reference_profile_path.read_text(encoding="utf-8"))
    return None


def load_all_reference_profiles(project: VideoProject) -> list[dict]:
    """Return all reference profiles for this run, in index order.

    The legacy flat profile (if present and no indexed sub-folders exist) is returned
    as a single-element list at index 0.
    """
    profiles = []
    for i in range(max(len(project.reference_urls), 1)):
        p = load_reference_profile(project, index=i)
        if p is not None:
            profiles.append(p)
    return profiles


def set_reference_lyrics(project: VideoProject, index: int, lyrics: str,
                         *, source: str = "manual", on_log=None) -> dict | None:
    """Overwrite one already-analyzed reference's transcript with corrected lyrics.

    ``source`` records where the text came from (``"manual"`` = pasted by hand,
    ``"subtitles"`` = fetched from the channel's manual YouTube subtitles) and is stored
    as ``lyrics_source`` on the profile so the UI can badge it.

    Use this to correct Whisper's mistranscription without re-downloading or re-analyzing
    the audio. Recomputes flow from the new word count against the cached duration/BPM,
    updates ``reference_lyrics.txt`` and the profile, and invalidates the derived caches
    (per-reference lore and the per-run narrative structure) so they re-mine from the
    corrected text on next use. Returns the updated profile, or None if the reference
    hasn't been analyzed yet.
    """
    log = on_log or (lambda _m: None)

    profile = load_reference_profile(project, index)
    if profile is None:
        return None

    lyrics = (lyrics or "").strip()
    profile_path = project.ref_profile_path(index)
    lyrics_path = project.ref_lyrics_path(index)

    profile["transcript"] = lyrics
    profile["lyrics_source"] = source

    word_count = len(lyrics.split())
    bpm = profile.get("bpm", 0) or 0
    span = float(profile.get("duration_sec") or 0.0)
    words_per_sec = round(word_count / span, 2) if span > 0 else 0.0
    syllables_per_beat = round(words_per_sec * 60.0 / bpm, 2) if bpm > 0 else 0.0
    profile["flow"] = {
        "word_count": word_count,
        "words_per_sec": words_per_sec,
        "syllables_per_beat": syllables_per_beat,
    }

    lyrics_path.parent.mkdir(parents=True, exist_ok=True)
    lyrics_path.write_text(lyrics, encoding="utf-8")
    profile_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")

    # The lore and the shared narrative structure were mined from the old transcript.
    lore_path = project.ref_lore_profile_path(index)
    if lore_path.exists():
        lore_path.unlink()
    if project.narrative_structure_path.exists():
        project.narrative_structure_path.unlink()
    if project.story_content_path.exists():
        project.story_content_path.unlink()

    log(f"Reference {index} lyrics updated: {word_count} words. Lore/structure caches cleared.")
    return profile


def remove_reference(project: VideoProject, index: int) -> None:
    """Remove one reference: delete its sub-folder and update project.reference_urls.

    Also invalidates the per-run narrative structure blueprint, since the shared skeleton
    was synthesized from the (now changed) set of references and would be stale.
    """
    import shutil
    ref_dir = project.ref_dir(index)
    if ref_dir.exists():
        shutil.rmtree(ref_dir)
    urls = list(project.reference_urls)
    if index < len(urls):
        urls.pop(index)
    project.reference_urls = urls
    project.save()

    # The narrative structure is a cross-reference synthesis — stale once refs change.
    if project.narrative_structure_path.exists():
        project.narrative_structure_path.unlink()
    if project.story_content_path.exists():
        project.story_content_path.unlink()
