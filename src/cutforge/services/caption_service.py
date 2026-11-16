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


# Lyric-sheet annotations ([Spoken], [Chorus], [Verse 1], …) are performance
# notes for the singer, not lyrics — they must never reach the screen.
_TAG_RE = re.compile(r"^[\[(][^\]\)]*[\]\)][.,!?…]*$")
# Em/en dashes read as noise in a caption (they arrive as their own token from
# the aligner, e.g. "BANKAI! — THAT'S THE WARNING"). Plain hyphens are kept,
# so "GHOST-WHITE" survives intact.
_DASH_RE = re.compile(r"[—–]+")
# Punctuation a caption block should never end on. "!" and "?" are kept —
# they carry delivery, a period or comma only carries grammar.
_TRAILING_PUNCT = " .,;:…"


def _is_glue(word: str) -> bool:
    return re.sub(r"[^\w]", "", word).lower() in _GLUE_WORDS


def _is_caption_word(word: str) -> bool:
    """False for tokens that are markup rather than something the singer sings."""
    token = _DASH_RE.sub("", word).strip()
    return bool(token) and not _TAG_RE.match(token)


def _is_comma_orphan(chunk: list) -> bool:
    """True when a chunk ends with a comma followed by a single stray word.

    "RENJI BLOCKS MY PATH, KENPACHI" reads better as "RENJI BLOCKS MY PATH"
    with "KENPACHI" leading the next block. The word moves forward and the
    now-trailing comma is dropped by _clean_text. Chunks of fewer than three
    words are left alone so the move never strands a single word on screen.
    """
    return len(chunk) >= 3 and chunk[-2].word.rstrip().endswith(",")


def _clean_text(text: str) -> str:
    """Strip dashes and final punctuation, and collapse the resulting gaps."""
    text = _DASH_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.rstrip(_TRAILING_PUNCT)


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

    Post-processing passes improve quality:
    - Markup drop: lyric-sheet tags ([Spoken], [Chorus], …) and bare em/en
      dashes are removed before chunking, so they never take up screen time.
    - Glue-word repair: if the last word of a chunk is a short function word
      (I, a, the, to, and, …) it is moved to the start of the next chunk within
      the same line, so phrase openings stay with their phrase.
    - Comma-orphan repair: a chunk ending in ", <one word>" hands that word to
      the next chunk — "RENJI BLOCKS MY PATH, KENPACHI" / "GRINS FOR THE KILL"
      becomes "RENJI BLOCKS MY PATH" / "KENPACHI GRINS FOR THE KILL".
    - Text clean-up: dashes and final punctuation (. , ; : …) are stripped;
      "!" and "?" stay because they carry delivery.
    - Tight end times: each caption clears shortly after its last sung word
      (TAIL_HOLD) instead of lingering, but consecutive lines chain seamlessly
      when the next one starts almost immediately (CHAIN_GAP).
    - No overlaps: a caption never starts before the previous one had MIN_VISIBLE
      on screen, and never ends after the next one starts.
    """
    # --- Phase 1: build per-line word-level chunks ---
    line_chunk_lists: list[list[list]] = []
    for line in lines:
        words = [w for w in line.words if _is_caption_word(w.word)]
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

        # --- Phase 2: hand stray trailing words to the next chunk of this line ---
        # Repeat per chunk: moving a glue word can expose a comma orphan behind
        # it (". . . ribs, sparked the" -> "the" then "sparked" both move on).
        for i in range(len(chunks) - 1):
            while chunks[i] and (_is_glue(chunks[i][-1].word)
                                 or _is_comma_orphan(chunks[i])):
                chunks[i + 1].insert(0, chunks[i].pop())

        line_chunk_lists.append([c for c in chunks if c])

    # --- Phase 3: flatten to groups ---
    groups: list[dict] = []
    for chunks in line_chunk_lists:
        for chunk in chunks:
            text = _clean_text(" ".join(w.word for w in chunk).upper())
            if text:
                groups.append({
                    "start": chunk[0].start,
                    "word_end": chunk[-1].end,
                    "text": text,
                })

    # --- Phase 4: compute SRT end times ---
    # End each caption tightly after its last sung word (like Premiere's own
    # transcript) instead of letting it linger. Knobs:
    #   TAIL_HOLD — how long a caption stays after the last word before a pause,
    #               so it clears quickly instead of dragging into the silence.
    #   CHAIN_GAP — if the next caption starts within this, hold until it so
    #               back-to-back lines chain seamlessly (no blank flicker).
    #   MIN_DUR   — floor so very short chunks stay readable (never overlapping next).
    #   MIN_VISIBLE — every caption gets this long before the next one replaces it.
    TAIL_HOLD = 0.2
    CHAIN_GAP = 0.7
    MIN_DUR = 0.8
    MIN_VISIBLE = 0.3

    # Words the aligner squeezed onto one instant give several groups the same start.
    # Overlapping blocks import into Premiere stacked on top of each other, so each
    # group waits its turn; the delay disappears at the next real gap.
    earliest = 0.0
    for grp in groups:
        grp["start"] = max(grp["start"], earliest)
        earliest = grp["start"] + MIN_VISIBLE

    out = []
    for i, grp in enumerate(groups):
        start = grp["start"]
        word_end = grp["word_end"]
        if i + 1 < len(groups):
            next_start = groups[i + 1]["start"]
            if next_start - word_end <= CHAIN_GAP:
                end = next_start                              # chain into the next line
            else:
                end = word_end + TAIL_HOLD                    # clear quickly before the pause
                end = max(end, min(start + MIN_DUR, next_start))
            end = min(end, next_start)                        # never overlap the next caption
        else:
            end = max(word_end + TAIL_HOLD, start + MIN_DUR)
        end = max(end, start + MIN_VISIBLE)                   # absolute floor
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
        words = [w for w in line.words if _is_caption_word(w.word)]
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
