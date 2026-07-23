"""Lyric alignment service.

Ports the CRITICAL alignment logic from the old ``align_lyrics.py`` verbatim: clean
lyrics are mapped onto Whisper's timed words via SequenceMatcher, spurious jump anchors
(from repeated choruses) are rejected, and gaps are interpolated. This logic was tuned
against real mixes (the Naruto ES bug where "Kyuubi" froze for 92s) — keep it 1:1.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

from cutforge.integrations import whisper_client
from cutforge.models.alignment import Alignment, LyricLine, LyricWord
from cutforge.models.project import VideoProject

SECTION_MARKER = re.compile(r"^\s*[\[(].*[\])]\s*$")  # [Verse], (Chorus), etc.


def normalize(word: str) -> str:
    """Lowercase and strip punctuation for matching only (not for display)."""
    return re.sub(r"[^\w']", "", word.lower())


def parse_lyrics(lyrics_text: str) -> list[list[str]]:
    """Parse lyrics into display lines (list of words). Section markers/blanks dropped."""
    lines: list[list[str]] = []
    for raw_line in lyrics_text.splitlines():
        stripped = raw_line.strip()
        if not stripped or SECTION_MARKER.match(stripped):
            continue
        words = stripped.split()
        if words:
            lines.append(words)
    return lines


def align(clean_words: list[str], timed_words: list[dict]) -> list[dict]:
    """Map clean lyric words onto Whisper's timed words via sequence matching."""
    clean_norm = [normalize(w) for w in clean_words]
    timed_norm = [normalize(w["word"]) for w in timed_words]

    matcher = SequenceMatcher(a=clean_norm, b=timed_norm, autojunk=False)
    aligned: list[dict | None] = [None] * len(clean_words)

    for a0, b0, size in matcher.get_matching_blocks():
        for k in range(size):
            tw = timed_words[b0 + k]
            aligned[a0 + k] = {
                "word": clean_words[a0 + k],
                "start": tw["start"],
                "end": tw["end"],
            }

    _reject_jump_anchors(aligned)
    _interpolate_gaps(aligned, clean_words, timed_words)
    return aligned  # type: ignore[return-value]


def _reject_jump_anchors(aligned: list, max_rate_factor: float = 6.0,
                         min_jump_seconds: float = 3.0) -> None:
    """Drop spurious anchors that force an implausible forward time jump (in place)."""
    idxs = [i for i, a in enumerate(aligned) if a is not None]
    if len(idxs) < 3:
        return

    rates = []
    for p, q in zip(idxs, idxs[1:]):
        dt = aligned[q]["start"] - aligned[p]["start"]
        rate = dt / (q - p)
        if dt >= 0:
            rates.append(rate)
    if not rates:
        return
    rates.sort()
    median_rate = rates[len(rates) // 2] or 0.01
    cap = max(median_rate * max_rate_factor, 0.5)

    changed = True
    while changed:
        changed = False
        idxs = [i for i, a in enumerate(aligned) if a is not None]
        for p, q in zip(idxs, idxs[1:]):
            dt = aligned[q]["start"] - aligned[p]["start"]
            span = dt / (q - p)
            if dt > min_jump_seconds and span > cap:
                aligned[q] = None
                changed = True
                break


def _interpolate_gaps(aligned: list, clean_words: list[str], timed_words: list[dict]) -> None:
    """Fill None entries by evenly spreading time between matched neighbors (in place)."""
    n = len(aligned)
    if n == 0:
        return

    song_start = timed_words[0]["start"] if timed_words else 0.0
    song_end = timed_words[-1]["end"] if timed_words else 0.0

    i = 0
    while i < n:
        if aligned[i] is not None:
            i += 1
            continue
        j = i
        while j < n and aligned[j] is None:
            j += 1
        prev_end = aligned[i - 1]["end"] if i > 0 else song_start
        next_start = aligned[j]["start"] if j < n else song_end
        if next_start < prev_end:
            next_start = prev_end
        count = j - i
        span = (next_start - prev_end) / count if count else 0.0
        for k in range(count):
            s = prev_end + span * k
            e = prev_end + span * (k + 1)
            aligned[i + k] = {"word": clean_words[i + k], "start": round(s, 3), "end": round(e, 3)}
        i = j


def build_lines(display_lines: list[list[str]], aligned_flat: list[dict]) -> list[LyricLine]:
    """Regroup the flat aligned word list back into display lines."""
    lines: list[LyricLine] = []
    idx = 0
    for words in display_lines:
        n = len(words)
        chunk = aligned_flat[idx:idx + n]
        idx += n
        if not chunk:
            continue
        lines.append(LyricLine(
            start=round(chunk[0]["start"], 3),
            end=round(chunk[-1]["end"], 3),
            words=[LyricWord(word=w["word"], start=round(w["start"], 3), end=round(w["end"], 3))
                   for w in chunk],
        ))
    return lines


def enforce_monotonic(lines: list[LyricLine]) -> None:
    """Ensure word/line times never go backwards (in place) — karaoke tags require it."""
    last = 0.0
    for line in lines:
        for w in line.words:
            if w.start < last:
                w.start = last
            if w.end < w.start:
                w.end = w.start
            last = w.end
        if line.words:
            line.start = line.words[0].start
            line.end = line.words[-1].end


def align_project(project: VideoProject, *, refresh: bool = False, on_log=None) -> Alignment:
    """Align the run's lyrics.txt to its track.mp3 and write lyrics_alignment.json."""
    if not project.track_path.exists():
        raise FileNotFoundError(
            f"track.mp3 not found at {project.track_path} — add the Suno song first."
        )
    if not project.lyrics_path.exists():
        raise FileNotFoundError(f"lyrics.txt not found at {project.lyrics_path}")

    display_lines = parse_lyrics(project.lyrics_path.read_text(encoding="utf-8"))
    if not display_lines:
        raise ValueError("No lyric lines found in lyrics.txt")
    clean_words = [w for line in display_lines for w in line]
    if on_log:
        on_log(f"Parsed {len(display_lines)} lines, {len(clean_words)} words from lyrics.txt")

    timed_words = whisper_client.transcribe_words(
        project.track_path, cache_path=project.whisper_cache_path,
        refresh=refresh, on_log=on_log,
    )
    if not timed_words:
        raise RuntimeError("Whisper returned no words — cannot align.")

    aligned_flat = align(clean_words, timed_words)
    lines = build_lines(display_lines, aligned_flat)
    enforce_monotonic(lines)

    alignment = Alignment(audio=str(project.track_path), lines=lines)

    import json
    project.audio_dir.mkdir(parents=True, exist_ok=True)
    project.alignment_path.write_text(
        json.dumps(alignment.to_json_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if on_log:
        matched = sum(1 for w in aligned_flat if w is not None)
        on_log(f"Aligned {alignment.line_count} lines, {alignment.word_count} words "
               f"({matched} directly matched to Whisper)")
    return alignment
