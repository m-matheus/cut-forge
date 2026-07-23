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
