"""Tests for the critical lyric-alignment logic (jump-anchor rejection + interpolation)."""
from cutforge.services.alignment_service import align, parse_lyrics


def test_parse_lyrics_drops_section_markers():
    text = "[Verse 1]\nShadow rises now\n\n(Chorus)\nArise my king\n"
    lines = parse_lyrics(text)
    assert lines == [["Shadow", "rises", "now"], ["Arise", "my", "king"]]


def test_align_fills_all_words():
    clean = ["shadow", "monarch", "rise"]
    timed = [
        {"word": "shadow", "start": 1.0, "end": 1.5},
        {"word": "monarch", "start": 1.5, "end": 2.0},
        {"word": "rise", "start": 2.0, "end": 2.5},
    ]
    result = align(clean, timed)
    assert len(result) == 3
    assert all(r is not None for r in result)
    assert [r["word"] for r in result] == clean


def test_align_rejects_jump_anchor_from_repeated_chorus():
    # "shadow monarch" repeats; Whisper's 2nd repeat is at 50s. The matcher may bind
    # the first clean "rise" region forward — the jump-anchor rejecter should prevent
    # a multi-second freeze by dropping the bad anchor and interpolating instead.
    clean = ["shadow", "monarch", "rise", "again", "shadow", "monarch", "fall"]
    timed = [
        {"word": "shadow", "start": 1.0, "end": 1.5},
        {"word": "monarch", "start": 1.5, "end": 2.0},
        {"word": "rise", "start": 2.0, "end": 2.5},
        {"word": "again", "start": 2.5, "end": 3.0},
        {"word": "shadow", "start": 50.0, "end": 50.5},
        {"word": "monarch", "start": 50.5, "end": 51.0},
        {"word": "fall", "start": 51.0, "end": 51.5},
    ]
    result = align(clean, timed)
    # Monotonic non-decreasing starts, no freeze longer than the song.
    starts = [r["start"] for r in result]
    assert starts == sorted(starts)
    assert all(r is not None for r in result)


def test_interpolated_words_pack_before_vocal_after_instrumental_break():
    # Words sung ~0-2s, then a long instrumental break, vocal resumes at 40s. The two
    # unmatched middle words must NOT smear across the silence (appearing early); they
    # should pack just before 40s so captions fire when singing actually resumes.
    clean = ["intro", "line", "bridge", "word", "verse", "back"]
    timed = [
        {"word": "intro", "start": 0.0, "end": 0.5},
        {"word": "line", "start": 0.5, "end": 1.0},
        # "bridge" and "word" have no Whisper match (instrumental break swallowed them)
        {"word": "verse", "start": 40.0, "end": 40.5},
        {"word": "back", "start": 40.5, "end": 41.0},
    ]
    result = align(clean, timed)
    assert all(r is not None for r in result)
    # The two gap words should sit close to the 40s resume, not spread from ~1s.
    bridge, word = result[2], result[3]
    assert bridge["start"] > 30.0, f"gap word fired too early at {bridge['start']}s"
    assert word["start"] < 40.0
