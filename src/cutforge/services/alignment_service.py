"""Lyric alignment service.

Ports the CRITICAL alignment logic from the old ``align_lyrics.py`` verbatim: clean
lyrics are mapped onto Whisper's timed words via SequenceMatcher, spurious jump anchors
(from repeated choruses) are rejected, and gaps are interpolated. This logic was tuned
against real mixes (the Naruto ES bug where "Kyuubi" froze for 92s) — keep it 1:1.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

from cutforge.integrations import stable_whisper_client, whisper_client
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


def align(clean_words: list[str], timed_words: list[dict], *,
          reject_jumps: bool = True) -> list[dict]:
    """Map clean lyric words onto Whisper's timed words via sequence matching.

    ``reject_jumps`` controls the chorus-misbinding guard (``_reject_jump_anchors``).
    Keep it True on the OpenAI *transcribe* path: there the word↔time mapping is
    unknown, so SequenceMatcher can bind an early lyric word to a timestamp from a
    LATER chorus repeat, and that guard is essential. Set it False on the stable-ts
    *force-align* path: ``model.align()`` was fed the exact lyrics, so ``timed_words``
    is already 1:1 in order with ``clean_words`` — there is no misbinding to catch, and
    the guard's constant-pace drift test instead deletes correct post-gap timestamps
    (rap intros/breaks legitimately break a constant words-per-second), which
    ``_interpolate_gaps`` then refills with synthetic timing. That was the source of
    the verse/post-break drift.
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

    if reject_jumps:
        _reject_jump_anchors(aligned)
    _interpolate_gaps(aligned, clean_words, timed_words)
    return aligned  # type: ignore[return-value]


def _reject_jump_anchors(aligned: list, drift_threshold: float = 0.30) -> None:
    """Drop spurious anchors whose time-position runs far ahead of their lyric-position.

    When lyrics repeat (choruses), Whisper transcribes each repeat and the sequence
    matcher can bind an EARLY clean word to a timestamp from a LATER repeat. Because
    matches stay monotonic, that bad anchor drags everything after it forward — the
    highlight freezes on the jumped word for tens of seconds and the tail lines get
    crammed into whatever time is left.

    The clean discriminator between a chorus misbinding and a legitimate long
    instrumental break is DRIFT: for each matched anchor compare its position in the
    lyrics (index / total) to its position in time (start / span). A chorus misbind
    drifts far (early lyric word landing deep in the song); an instrumental break only
    drifts modestly. Any anchor whose |time_frac - lyric_frac| exceeds ``drift_threshold``
    is dropped so ``_interpolate_gaps`` re-spreads it between trustworthy neighbours.
    Iterated to stability because dropping one outlier can re-expose the reference pace.
    """
    n = len(aligned)
    if n < 3:
        return

    changed = True
    while changed:
        changed = False
        idxs = [i for i, a in enumerate(aligned) if a is not None]
        if len(idxs) < 3:
            return
        t0 = aligned[idxs[0]]["start"]
        t1 = aligned[idxs[-1]]["start"]
        span_t = (t1 - t0) or 1.0
        span_i = (idxs[-1] - idxs[0]) or 1

        worst_i, worst_drift = None, drift_threshold
        for i in idxs:
            lyric_frac = (i - idxs[0]) / span_i
            time_frac = (aligned[i]["start"] - t0) / span_t
            drift = abs(time_frac - lyric_frac)
            if drift > worst_drift:
                worst_drift = drift
                worst_i = i
        if worst_i is not None:
            aligned[worst_i] = None
            changed = True


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


def align_project(project: VideoProject, *, refresh: bool = False,
                  backend: str = "stable", on_log=None) -> Alignment:
    """Align the run's lyrics.txt to its track.mp3 and write lyrics_alignment.json.

    backend='stable' uses stable-whisper (local, precise ~50ms timestamps).
    backend='openai' uses the OpenAI Whisper API (cloud, ~500ms timestamps).
    """
    if backend == "stable":
        return _align_stable(project, refresh=refresh, on_log=on_log)
    return _align_openai(project, refresh=refresh, on_log=on_log)


def _align_stable(project: VideoProject, *, refresh: bool = False, on_log=None) -> Alignment:
    """Alignment via stable-ts force-alignment: passes exact lyrics text to model.align()."""
    if not project.track_path.exists():
        raise FileNotFoundError(f"track.mp3 not found at {project.track_path}")
    if not project.lyrics_path.exists():
        raise FileNotFoundError(f"lyrics.txt not found at {project.lyrics_path}")

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

    # Force-align output is already 1:1 in-order with our lyrics, so skip the
    # chorus-misbinding guard: it would delete correct post-gap timestamps.
    aligned_flat = align(clean_words, timed_words, reject_jumps=False)
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
               f"({matched}/{len(clean_words)} directly matched via stable-ts force-alignment)")
    return alignment


def _align_openai(project: VideoProject, *, refresh: bool = False, on_log=None) -> Alignment:
    """Original alignment via OpenAI Whisper API."""
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
        refresh=refresh, prompt=" ".join(clean_words), on_log=on_log,
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
