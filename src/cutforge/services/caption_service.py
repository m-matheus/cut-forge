"""Caption service — build Premiere transcript and SRT from a lyric alignment."""
from __future__ import annotations

import json
import re

from cutforge.models.alignment import Alignment, LyricLine
from cutforge.models.project import VideoProject

# Function words that should NOT be left stranded at the end of a caption chunk.
# When the last word of a chunk is one of these, it is moved to the start of the
# next chunk within the same lyric line — so "one whiff of betrayal, I /
# close your file no rehearsal" becomes "one whiff of betrayal, /
# I close your file no rehearsal".
_GLUE_WORDS = frozenset({
    "i", "a", "an", "the", "to", "so", "but", "and", "or", "nor",
    "in", "on", "at", "of", "my", "your", "his", "her", "its", "our",
    "their", "we", "you", "it", "as", "for", "not", "no", "do", "did",
    "be", "is", "was", "by", "up", "if", "oh", "ah",
})


def _is_glue(word: str) -> bool:
    return re.sub(r"[^\w]", "", word).lower() in _GLUE_WORDS


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


def build_srt(lines: list[LyricLine], *, max_chunk_duration: float = 1.5,
              max_words: int = 5) -> str:
    """Build an SRT respecting line boundaries, grouping words by time window.

    Each lyric line is never split across groups — a new group always starts at
    a new line. Within a line, words accumulate until the chunk spans
    max_chunk_duration seconds OR reaches max_words words (whichever comes first).

    Two post-processing passes improve quality:
    - Glue-word repair: if the last word of a chunk is a short function word
      (I, a, the, to, and, …) it is moved to the start of the next chunk within
      the same line, so phrase openings stay with their phrase.
    - Section-break hold: when the gap to the next caption is large (>2s the
      caption holds for only 0.5s after the last word, giving a clean blank
      screen during instrumental breaks instead of a lingering subtitle.
    """
    # --- Phase 1: build per-line word-level chunks ---
    line_chunk_lists: list[list[list]] = []
    for line in lines:
        words = line.words
        if not words:
            line_chunk_lists.append([])
            continue
        chunks: list[list] = []
        chunk: list = []
        for word in words:
            chunk.append(word)
            duration = chunk[-1].end - chunk[0].start
            if duration >= max_chunk_duration or len(chunk) >= max_words:
                chunks.append(chunk)
                chunk = []
        if chunk:
            chunks.append(chunk)

        # --- Phase 2: glue-word repair within this line ---
        for i in range(len(chunks) - 1):
            if chunks[i] and _is_glue(chunks[i][-1].word):
                moved = chunks[i].pop()
                chunks[i + 1].insert(0, moved)

        line_chunk_lists.append([c for c in chunks if c])

    # --- Phase 3: flatten to groups ---
    groups: list[dict] = []
    for chunks in line_chunk_lists:
        for chunk in chunks:
            text = " ".join(w.word for w in chunk).upper().strip()
            if text:
                groups.append({
                    "start": chunk[0].start,
                    "word_end": chunk[-1].end,
                    "text": text,
                })

    # --- Phase 4: compute SRT end times ---
    # During flowing lyrics (gap to next < 2s) hold up to 1.5s so captions
    # scroll smoothly. At section breaks (instrumental, gap ≥ 2s) hold only
    # 0.5s so the screen clears quickly before the silence.
    MAX_HOLD = 1.5
    SECTION_BREAK_HOLD = 0.5
    SECTION_BREAK_GAP = 2.0

    out = []
    for i, grp in enumerate(groups):
        start = grp["start"]
        word_end = grp["word_end"]
        if i + 1 < len(groups):
            gap_to_next = groups[i + 1]["start"] - word_end
            raw_end = max(groups[i + 1]["start"], start + 0.3)
            hold = SECTION_BREAK_HOLD if gap_to_next > SECTION_BREAK_GAP else MAX_HOLD
        else:
            raw_end = max(word_end, start + 0.8)
            hold = MAX_HOLD
        end = min(raw_end, word_end + hold)
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
                      max_chunk_duration: float = 1.5, on_log=None) -> None:
    """Write premiere_transcript.json and captions.srt for the run."""
    if alignment is None:
        if not project.alignment_path.exists():
            raise FileNotFoundError("lyrics_alignment.json not found — run alignment first.")
        data = json.loads(project.alignment_path.read_text(encoding="utf-8"))
        alignment = Alignment(**data)

    project.audio_dir.mkdir(parents=True, exist_ok=True)

    srt = build_srt(alignment.lines, max_chunk_duration=max_chunk_duration)
    project.captions_srt_path.write_text(srt, encoding="utf-8")

    transcript = build_premiere_transcript(alignment.lines, language=project.language)
    project.premiere_transcript_path.write_text(
        json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")

    if on_log:
        on_log(f"Captions saved: {project.captions_srt_path.name} + "
               f"{project.premiere_transcript_path.name} "
               f"({alignment.line_count} lines, max_chunk={max_chunk_duration}s)")
