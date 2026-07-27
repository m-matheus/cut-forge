"""Caption service — build Premiere transcript and SRT from a lyric alignment."""
from __future__ import annotations

import json

from cutforge.models.alignment import Alignment, LyricLine
from cutforge.models.project import VideoProject


def seconds_to_srt_time(seconds: float) -> str:
    """SRT timestamp: HH:MM:SS,mmm (comma decimal, milliseconds)."""
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = round((seconds - int(seconds)) * 1000)
    if ms == 1000:
        s, ms = s + 1, 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt(lines: list[LyricLine], *, words_per_group: int = 3) -> str:
    """Build an SRT segmented into short UPPERCASE phrase groups for Premiere's importer."""
    groups: list[dict] = []
    for line in lines:
        words = line.words
        if not words:
            continue
        for g in range(0, len(words), words_per_group):
            chunk = words[g:g + words_per_group]
            text = " ".join(w.word for w in chunk).upper().strip()
            if not text:
                continue
            groups.append({
                "start": chunk[0].start,
                "word_end": chunk[-1].end,
                "text": text,
            })

    MAX_HOLD_AFTER_WORD = 1.5

    out = []
    for i, grp in enumerate(groups):
        start = grp["start"]
        word_end = grp["word_end"]
        if i + 1 < len(groups):
            raw_end = max(groups[i + 1]["start"], start + 0.3)
        else:
            raw_end = max(word_end, start + 0.8)
        end = min(raw_end, word_end + MAX_HOLD_AFTER_WORD)
        end = max(end, start + 0.3)
        out.append(
            f"{i + 1}\n"
            f"{seconds_to_srt_time(start)} --> {seconds_to_srt_time(end)}\n"
            f"{grp['text']}"
        )
    return "\n\n".join(out) + "\n"


_SPEAKER_ID = "9f1e6c00-4a2b-4c3d-8e5f-0a1b2c3d4e5f"


def build_premiere_transcript(lines: list[LyricLine], *, language: str = "en-us",
                              speaker_name: str = "Vocals") -> dict:
    """Build an Adobe Premiere transcript (schema v1.0.0) for Text panel > Import transcript."""
    _PREMIERE_LANGS = {
        "en": "en-us", "es": "es-es", "pt": "pt-br",
    }
    lang = _PREMIERE_LANGS.get(language, language if "-" in language else "??-??")

    segments = []
    for line in lines:
        words = line.words
        if not words:
            continue
        word_objs = []
        for i, w in enumerate(words):
            start = max(0.0, float(w.start))
            dur = max(0.0, float(w.end) - start)
            word_objs.append({
                "confidence": 1.0,
                "duration": round(dur, 3),
                "eos": i == len(words) - 1,
                "start": round(start, 3),
                "tags": [],
                "text": w.word,
                "type": "word",
            })
        seg_start = max(0.0, float(words[0].start))
        seg_end = max(seg_start, float(words[-1].end))
        segments.append({
            "duration": round(seg_end - seg_start, 3),
            "language": lang,
            "speaker": _SPEAKER_ID,
            "start": round(seg_start, 3),
            "words": word_objs,
        })

    return {
        "language": lang,
        "segments": segments,
        "speakers": [{"id": _SPEAKER_ID, "name": speaker_name}],
    }


def generate_captions(project: VideoProject, alignment: Alignment | None = None, *,
                      words_per_group: int = 3, on_log=None) -> None:
    """Write premiere_transcript.json and captions.srt for the run."""
    if alignment is None:
        if not project.alignment_path.exists():
            raise FileNotFoundError("lyrics_alignment.json not found — run alignment first.")
        data = json.loads(project.alignment_path.read_text(encoding="utf-8"))
        alignment = Alignment(**data)

    project.audio_dir.mkdir(parents=True, exist_ok=True)

    srt = build_srt(alignment.lines, words_per_group=words_per_group)
    project.captions_srt_path.write_text(srt, encoding="utf-8")

    transcript = build_premiere_transcript(alignment.lines, language=project.language)
    project.premiere_transcript_path.write_text(
        json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")

    if on_log:
        on_log(f"Captions saved: {project.captions_srt_path.name} + "
               f"{project.premiere_transcript_path.name} "
               f"({alignment.line_count} lines, {words_per_group} words/group)")
