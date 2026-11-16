"""Lyric alignment service.

Force-aligns known lyrics to the track with stable-ts (``model.align()``), which
returns word timestamps already 1:1 in-order with our lyrics. Clean lyrics are mapped
onto those timed words via SequenceMatcher; any word stable-ts failed to emit is filled
by ``_interpolate_gaps``.

Note: the old chorus-misbinding guard (``_reject_jump_anchors``) and the OpenAI Whisper
transcribe path were removed — they existed for the cloud *transcribe* backend where the
word↔time mapping was unknown. With force-alignment the mapping is exact, so that guard
only deleted correct post-gap timestamps.
"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from cutforge.integrations import stable_whisper_client
from cutforge.models.alignment import Alignment, LyricLine, LyricWord
from cutforge.models.project import VideoProject
from cutforge.services import catalog_service
from cutforge.services.song_service import canonical_lyrics, suno_lyrics

SECTION_MARKER = re.compile(r"^\s*[\[(].*[\])]\s*$")  # [Verse], (Chorus), etc.
# Shorter than this, a word was squeezed onto an instant rather than timed.
COLLAPSED_WORD_DUR = 0.1
# Above this share of collapsed words the text is not what is being sung: the worst run
# aligned to its real lyrics stays well below it, a run aligned to another song's far above.
MAX_COLLAPSED_RATIO = 0.5


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


def embedded_lyrics(audio_path: Path) -> str:
    """The lyrics Suno stamped into the MP3 (ID3 USLT frame), or "" when it has none."""
    from mutagen import MutagenError
    from mutagen.id3 import ID3

    try:
        frames = ID3(str(audio_path)).getall("USLT")
    except MutagenError:
        return ""
    return frames[0].text.strip() if frames else ""


def _with_section_breaks(sheet: str) -> str:
    """Suno drops blank lines from the tag; put back the one before each section."""
    out: list[str] = []
    for line in sheet.splitlines():
        if out and out[-1].strip() and catalog_service.is_section_tag(line):
            out.append("")
        out.append(line)
    return "\n".join(out)


def sync_lyrics_with_track(project: VideoProject, *, on_log=None) -> bool:
    """Make lyrics.txt the sheet the track was made from. Returns True if it changed.

    The tag holds what Suno was actually given, so a mismatch means the sheet changed on
    the way there: a line edited in Suno, or another take of the song pasted in. Aligning
    the wrong text raises no error, it just yields captions of a song nobody sings, and
    the metadata and release lyrics read this file too.

    A lyrics.txt newer than the track is left alone: that is a hand correction of what
    Suno really sang, or a new song still waiting for its track.
    """
    sheet = embedded_lyrics(project.track_path)
    if not sheet or project.lyrics_path.stat().st_mtime >= project.track_path.stat().st_mtime:
        return False

    current = project.lyrics_path.read_text(encoding="utf-8")
    pronunciations = {}
    if project.suno_prompt_path.exists():
        suno = json.loads(project.suno_prompt_path.read_text(encoding="utf-8"))
        pronunciations = suno.get("pronunciations") or {}
    # Pasting the respelled sheet the wizard offers is not a change of lyrics.
    if parse_lyrics(sheet) in (parse_lyrics(current),
                               parse_lyrics(suno_lyrics(current, pronunciations))):
        return False

    project.previous_lyrics_path.write_text(current, encoding="utf-8")
    project.lyrics_path.write_text(
        _with_section_breaks(canonical_lyrics(sheet, pronunciations)), encoding="utf-8")
    if on_log:
        on_log(f"lyrics.txt differs from the sheet inside {project.track_path.name}: using the "
               f"track's (old text kept in {project.previous_lyrics_path.name})")
    return True


def align(clean_words: list[str], timed_words: list[dict]) -> list[dict]:
    """Map clean lyric words onto stable-ts's timed words via sequence matching.

    ``model.align()`` was fed the exact lyrics, so ``timed_words`` is already 1:1 in
    order with ``clean_words`` — SequenceMatcher just carries the real timestamps onto
    the display words, and ``_interpolate_gaps`` fills only the rare word stable-ts
    failed to emit. No chorus-misbinding guard is needed (or wanted): there is no
    cross-repeat misbinding to catch when the exact text was the alignment input.
    """
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

    _interpolate_gaps(aligned, clean_words, timed_words)
    return aligned  # type: ignore[return-value]


def _median_matched_rate(aligned: list, default: float = 0.35) -> float:
    """Median seconds-per-word between consecutive matched anchors (a sane singing pace)."""
    idxs = [i for i, a in enumerate(aligned) if a is not None]
    rates = []
    for p, q in zip(idxs, idxs[1:]):
        dt = aligned[q]["start"] - aligned[p]["start"]
        if dt > 0:
            rates.append(dt / (q - p))
    if not rates:
        return default
    rates.sort()
    return rates[len(rates) // 2] or default


def _interpolate_gaps(aligned: list, clean_words: list[str], timed_words: list[dict]) -> None:
    """Fill None entries with timing between matched neighbours (in place).

    Even-spreading a gap works when the missed words were sung continuously. But when a
    long INSTRUMENTAL break falls inside the gap (Whisper transcribed nothing across it),
    even-spreading smears the words across the silence, so a caption lights up long before
    it is actually sung. Fix: cap each interpolated word to a plausible singing pace
    (median matched rate). If the words fit the gap at that pace, keep the natural even
    spread. If the gap is far larger than the words need (an instrumental break), PACK the
    words to END at the next anchor — clustering them right before the vocal resumes and
    leaving the slack as a silent lead-in rather than early-firing captions.
    """
    n = len(aligned)
    if n == 0:
        return

    song_start = timed_words[0]["start"] if timed_words else 0.0
    song_end = timed_words[-1]["end"] if timed_words else 0.0
    median_rate = _median_matched_rate(aligned)
    # A word may run a bit longer than the median when sung slowly; allow headroom before
    # treating the remaining span as an instrumental break to skip over.
    per_word_cap = max(median_rate * 1.5, 0.5)

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
        total = next_start - prev_end
        even_span = total / count if count else 0.0

        if even_span <= per_word_cap or count == 0:
            # Words fit at a natural pace — spread evenly from prev_end.
            for k in range(count):
                s = prev_end + even_span * k
                e = prev_end + even_span * (k + 1)
                aligned[i + k] = {"word": clean_words[i + k], "start": round(s, 3), "end": round(e, 3)}
        else:
            # Gap far exceeds what the words need → an instrumental break sits inside it.
            # Pack the words at per_word_cap pace to END at next_start, so captions fire
            # just before the vocal resumes instead of over the silence.
            block = per_word_cap * count
            block_start = max(prev_end, next_start - block)
            for k in range(count):
                s = block_start + per_word_cap * k
                e = block_start + per_word_cap * (k + 1)
                aligned[i + k] = {"word": clean_words[i + k], "start": round(s, 3), "end": round(e, 3)}
        i = j


def pull_back_dragged_tails(display_lines: list[list[str]], aligned_flat: list[dict], *,
                            min_gap: float = 2.0, glue_gap: float = 0.5,
                            max_tail_words: int = 3) -> int:
    """Re-seat trailing words that force-alignment dragged onto the next vocal onset.

    stable-ts must emit every word in order, so when a phrase's last word is held
    over an instrumental break it tends to park that word at the *next* vocal onset
    instead of where it was actually sung. "Let the vinyl crack, let the steel start
    speakin'." came back with "speakin'." at 38.69s — 15s after "start" and 0.04s
    before the next section's first word — so the caption surfaced glued to the
    wrong chorus.

    The tell is the pair of gaps around the tail: a long silence *before* it and
    essentially none *after* it. Only then is the tail shifted back to sit directly
    after its own line's previous word, durations and internal spacing preserved.

    This is deliberately narrower than the old ``_reject_jump_anchors`` guard the
    module docstring mentions: that one dropped any post-gap timestamp and so
    destroyed correct ones. Here a word is only moved when it is provably welded to
    the following phrase's onset, and it keeps its timing rather than being deleted.

    Returns the number of lines repaired. Mutates ``aligned_flat`` in place.
    """
    repaired = 0
    idx = 0
    for words in display_lines:
        start, idx = idx, idx + len(words)
        chunk = aligned_flat[start:min(idx, len(aligned_flat))]
        if len(chunk) < 2:
            continue
        # Everything after the line's last oversized gap is the candidate tail.
        split = next((k for k in range(len(chunk) - 1, 0, -1)
                      if chunk[k]["start"] - chunk[k - 1]["end"] > min_gap), None)
        if split is None or len(chunk) - split > max_tail_words:
            continue
        following = aligned_flat[idx] if idx < len(aligned_flat) else None
        if following is None or following["start"] - chunk[-1]["end"] > glue_gap:
            continue  # stands on its own after the break — genuinely sung there
        shift = chunk[split - 1]["end"] - chunk[split]["start"]
        for word in chunk[split:]:
            word["start"] = round(word["start"] + shift, 3)
            word["end"] = round(word["end"] + shift, 3)
        repaired += 1
    return repaired


def spread_collapsed_words(aligned_flat: list[dict], *, max_word_dur: float = 0.5) -> int:
    """Give words the aligner squeezed onto one instant a real share of time (in place).

    stable-ts must place every word, so the ones it could not resolve come back with no
    duration, piled on one timestamp, and would reach Premiere as captions stacked on top
    of each other. They were sung, just not separated:

    - With room before the next real word, they take no more than ``max_word_dur`` each. A
      lone word keeps its stamp: it closes a phrase before a break, like Luffy's "light".
    - Several words in that room lean on the next real word, because the stamp only marks
      where the aligner gave up: Yhwach had 29 words at 28.1s, sung right up to the next
      word at 50.6s, and spreading them from the stamp ran the captions 10s early.
    - With no room, the next real word swallowed them, and the words split its span evenly:
      "Nine... One more." came back as "more." holding the whole 2.9s.

    Returns the number of collapsed words moved.
    """
    def collapsed(word: dict) -> bool:
        return word["end"] - word["start"] < COLLAPSED_WORD_DUR

    moved = 0
    i, n = 0, len(aligned_flat)
    while i < n:
        if not collapsed(aligned_flat[i]):
            i += 1
            continue
        # Two collapsed words a phrase apart are two piles, not one.
        j = i + 1
        while (j < n and collapsed(aligned_flat[j])
               and aligned_flat[j]["start"] - aligned_flat[j - 1]["end"] < COLLAPSED_WORD_DUR):
            j += 1
        start = aligned_flat[i]["start"]
        # Take in following words until the pile fits at a sane pace.
        k = j
        while k < n and aligned_flat[k]["start"] - start < (k - i) * COLLAPSED_WORD_DUR:
            k += 1
        if k == j and k < n:
            pace = min((aligned_flat[k]["start"] - start) / (k - i), max_word_dur)
            if j - i > 1:
                start = aligned_flat[k]["start"] - pace * (k - i)
        else:
            pace = (aligned_flat[k - 1]["end"] - start) / (k - i)
        if pace > 0:
            for offset, word in enumerate(aligned_flat[i:k]):
                word["start"] = round(start + pace * offset, 3)
                word["end"] = round(start + pace * (offset + 1), 3)
            moved += j - i
        i = k
    return moved


def build_lines(display_lines: list[list[str]], aligned_flat: list[dict],
                gap_threshold: float = 2.0) -> list[LyricLine]:
    """Regroup the flat aligned word list back into display lines.

    Lines are split further if consecutive words within a display line are
    separated by more than ``gap_threshold`` seconds.  This happens when
    ``_interpolate_gaps`` packed words after a long instrumental break, leaving
    an isolated early-matched word (e.g. "I'm" at 10s) followed by the rest of
    the phrase packed at 42s.  Without splitting, the resulting LyricLine would
    span 10s–46s and every caption builder would show the text across the entire
    instrumental break.  After splitting, each LyricLine covers only a
    contiguous sung segment, and the silence between segments is genuinely blank.
    """
    lines: list[LyricLine] = []
    idx = 0
    for words in display_lines:
        n = len(words)
        chunk = aligned_flat[idx:idx + n]
        idx += n
        if not chunk:
            continue
        # Walk the chunk and emit a new LyricLine whenever there is a large gap
        # between consecutive words.
        segment_start = 0
        for k in range(1, len(chunk)):
            gap = chunk[k]["start"] - chunk[k - 1]["end"]
            if gap > gap_threshold:
                seg = chunk[segment_start:k]
                lines.append(LyricLine(
                    start=round(seg[0]["start"], 3),
                    end=round(seg[-1]["end"], 3),
                    words=[LyricWord(word=w["word"], start=round(w["start"], 3),
                                     end=round(w["end"], 3)) for w in seg],
                ))
                segment_start = k
        seg = chunk[segment_start:]
        if seg:
            lines.append(LyricLine(
                start=round(seg[0]["start"], 3),
                end=round(seg[-1]["end"], 3),
                words=[LyricWord(word=w["word"], start=round(w["start"], 3),
                                  end=round(w["end"], 3)) for w in seg],
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
    """Align the run's lyrics.txt to its track.mp3 and write lyrics_alignment.json.

    Uses stable-ts force-alignment (local, precise ~50ms timestamps) with Demucs vocal
    isolation + VAD, passing the exact lyrics text to ``model.align()`` so every word is
    anchored and none are dropped.
    """
    if not project.track_path.exists():
        raise FileNotFoundError(f"track.mp3 not found at {project.track_path}")
    if not project.lyrics_path.exists():
        raise FileNotFoundError(f"lyrics.txt not found at {project.lyrics_path}")

    sync_lyrics_with_track(project, on_log=on_log)
    display_lines = parse_lyrics(project.lyrics_path.read_text(encoding="utf-8"))
    if not display_lines:
        raise ValueError("No lyric lines found in lyrics.txt")
    clean_words = [w for line in display_lines for w in line]
    if on_log:
        on_log(f"Parsed {len(display_lines)} lines, {len(clean_words)} words from lyrics.txt")

    # Pass the full lyrics as text so model.align() anchors every word — no words dropped.
    lyrics_text = " ".join(clean_words)
    timed_words = stable_whisper_client.align_words(
        project.track_path,
        lyrics_text,
        cache_path=project.whisper_cache_path,
        refresh=refresh,
        language=project.language,
        on_log=on_log,
    )
    if not timed_words:
        raise RuntimeError("stable-ts returned no words — cannot align.")
    collapsed = sum(w["end"] - w["start"] < COLLAPSED_WORD_DUR for w in timed_words)
    if collapsed / len(timed_words) > MAX_COLLAPSED_RATIO:
        raise ValueError(
            f"{collapsed} of {len(timed_words)} words found no place in the audio: "
            f"lyrics.txt does not look like the song in {project.track_path.name}.")

    aligned_flat = align(clean_words, timed_words)
    dragged = pull_back_dragged_tails(display_lines, aligned_flat)
    if dragged and on_log:
        on_log(f"Pulled {dragged} line tail(s) back off the next section's vocal onset")
    spread = spread_collapsed_words(aligned_flat)
    if spread and on_log:
        on_log(f"Spread {spread} word(s) the aligner had collapsed onto a single instant")
    lines = build_lines(display_lines, aligned_flat)
    enforce_monotonic(lines)

    alignment = Alignment(audio=str(project.track_path), lines=lines)
    project.audio_dir.mkdir(parents=True, exist_ok=True)
    project.alignment_path.write_text(
        json.dumps(alignment.to_json_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if on_log:
        matched = sum(1 for w in aligned_flat if w is not None)
        on_log(f"Aligned {alignment.line_count} lines, {alignment.word_count} words "
               f"({matched}/{len(clean_words)} directly matched via stable-ts force-alignment)")
    return alignment
