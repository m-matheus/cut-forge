"""Tests for the critical lyric-alignment logic (jump-anchor rejection + interpolation)."""
import json
import os
import sys
from types import SimpleNamespace

import pytest
from mutagen.id3 import ID3, USLT

from cutforge.integrations import stable_whisper_client
from cutforge.models.project import VideoProject
from cutforge.services import alignment_service
from cutforge.services.alignment_service import (
    COLLAPSED_WORD_DUR, align, align_project, build_lines, embedded_lyrics, parse_lyrics,
    pull_back_dragged_tails, spread_collapsed_words, sync_lyrics_with_track,
)


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


def _timed(*spans) -> list[dict]:
    return [{"word": w, "start": s, "end": e} for w, s, e in spans]


def test_held_final_word_is_pulled_off_the_next_sections_onset():
    # stable-ts parked the chorus's held last word ("speakin'") at the next
    # section's vocal onset, 15s after the word before it and 0.04s before the
    # verse. It belongs right after "start".
    display_lines = [["let", "the", "steel", "start", "speakin'"],
                     ["a", "ghost-white", "edge"]]
    flat = _timed(
        ("let", 22.69, 22.87), ("the", 22.87, 23.17), ("steel", 23.17, 23.41),
        ("start", 23.41, 23.65), ("speakin'", 38.69, 39.45),
        ("a", 39.49, 39.60), ("ghost-white", 39.60, 39.99), ("edge", 39.99, 40.31),
    )
    assert pull_back_dragged_tails(display_lines, flat) == 1
    assert flat[4]["start"] == 23.65          # seated right after "start"
    assert round(flat[4]["end"] - flat[4]["start"], 2) == 0.76  # duration preserved
    assert [w["start"] for w in flat[5:]] == [39.49, 39.60, 39.99]  # next line untouched

    # And the line now survives build_lines intact instead of being split in two.
    lines = build_lines(display_lines, flat)
    assert len(lines) == 2
    assert [w.word for w in lines[0].words] == display_lines[0]


def test_tail_standing_on_its_own_after_a_break_is_left_alone():
    # Same long gap, but the word is NOT welded to what follows — it really is
    # sung after the break, so its timing must be preserved.
    display_lines = [["let", "the", "steel", "start", "speakin'"], ["next", "line"]]
    flat = _timed(
        ("let", 22.69, 22.87), ("the", 22.87, 23.17), ("steel", 23.17, 23.41),
        ("start", 23.41, 23.65), ("speakin'", 38.69, 39.45),
        ("next", 44.00, 44.30), ("line", 44.30, 44.60),
    )
    assert pull_back_dragged_tails(display_lines, flat) == 0
    assert flat[4]["start"] == 38.69


def test_whole_phrase_parked_after_a_break_is_not_dragged_back():
    # More than max_tail_words sit after the gap — that is a phrase genuinely sung
    # later, not a single held word, so leave it where the aligner put it.
    display_lines = [["a", "b", "c", "d", "e", "f"], ["next"]]
    flat = _timed(
        ("a", 1.0, 1.2), ("b", 1.2, 1.4),
        ("c", 30.0, 30.2), ("d", 30.2, 30.4), ("e", 30.4, 30.6), ("f", 30.6, 30.8),
        ("next", 30.9, 31.1),
    )
    assert pull_back_dragged_tails(display_lines, flat) == 0
    assert flat[2]["start"] == 30.0


def test_collapsed_words_share_the_span_of_the_word_that_swallowed_them():
    flat = _timed(("nine...", 17.112, 17.112), ("one", 17.112, 17.112), ("more.", 17.112, 20.032),
                  ("enkai.", 22.252, 27.852))
    assert spread_collapsed_words(flat) == 2
    assert [w["start"] for w in flat[:3]] == [17.112, 18.085, 19.059]
    assert flat[2]["end"] == 20.032
    assert flat[3] == {"word": "enkai.", "start": 22.252, "end": 27.852}  # next phrase untouched


def test_piled_words_lean_on_the_next_word_at_a_singing_pace():
    flat = _timed(("a", 42.0, 42.0), ("b", 42.0, 42.0), ("c", 50.0, 50.6))
    assert spread_collapsed_words(flat) == 2
    assert [(w["start"], w["end"]) for w in flat[:2]] == [(49.0, 49.5), (49.5, 50.0)]
    assert flat[2]["start"] == 50.0


def test_a_verse_piled_at_the_start_of_a_break_is_not_shown_during_the_break():
    # Yhwach: 29 words stamped 28.096s, then "nothing" at 50.64s. The vocal stem and a blind
    # transcription both put the verse right before that word, from about 42s.
    flat = _timed(*[(f"w{i}", 28.096, 28.096) for i in range(29)], ("nothing", 50.64, 50.9))
    assert spread_collapsed_words(flat) == 29
    assert flat[28]["end"] == 50.64
    assert flat[0]["start"] >= 36.0
    assert [w["start"] for w in flat] == sorted(w["start"] for w in flat)


def test_a_lone_collapsed_word_keeps_its_stamp_before_a_break():
    flat = _timed(("a", 1.0, 1.5), ("light", 42.0, 42.0), ("sun", 50.0, 50.6))
    assert spread_collapsed_words(flat) == 1
    assert (flat[1]["start"], flat[1]["end"]) == (42.0, 42.5)


def test_a_pile_with_no_room_takes_in_as_many_words_as_it_needs():
    flat = _timed(*[(f"w{i}", 10.0, 10.0) for i in range(5)],
                  ("x", 10.0, 10.2), ("y", 10.2, 10.4), ("z", 10.4, 11.0), ("next", 20.0, 20.5))
    assert spread_collapsed_words(flat) == 5
    durations = [round(w["end"] - w["start"], 3) for w in flat[:8]]
    assert len(set(durations)) == 1 and durations[0] >= COLLAPSED_WORD_DUR
    assert flat[7]["end"] == 11.0
    assert flat[8]["start"] == 20.0


def test_collapsed_words_a_phrase_apart_stay_in_their_own_phrases():
    # Luffy: "light" ends one phrase at 137s and "Sun" opens the next at 148s, both with
    # no duration. Treated as one pile, "Sun" got dragged 11s back into the wrong phrase.
    flat = _timed(("light", 137.06, 137.06), ("sun", 148.48, 148.48), ("god", 148.48, 148.96))
    assert spread_collapsed_words(flat) == 2
    assert (flat[0]["start"], flat[0]["end"]) == (137.06, 137.56)
    assert (flat[1]["start"], flat[1]["end"]) == (148.48, 148.72)
    assert flat[2]["end"] == 148.96


def test_well_timed_words_are_not_touched():
    flat = _timed(("a", 1.0, 1.2), ("b", 1.2, 1.5), ("c", 3.0, 3.1))
    before = [dict(w) for w in flat]
    assert spread_collapsed_words(flat) == 0
    assert flat == before


def test_pile_at_the_very_end_of_the_song_is_left_for_the_captions_to_handle():
    flat = _timed(("a", 1.0, 1.5), ("b", 282.0, 282.0), ("c", 282.0, 282.0))
    assert spread_collapsed_words(flat) == 0
    assert flat[2]["start"] == 282.0


@pytest.fixture
def project(tmp_path, monkeypatch):
    from cutforge.config import settings as settings_mod
    monkeypatch.setattr(settings_mod.Settings, "output_dir",
                        property(lambda self: tmp_path / "output"))
    settings_mod.get_settings.cache_clear()
    p = VideoProject.create(run_id="20261103-yhwach", character="Yhwach", anime="Bleach")
    p.lyrics_path.write_text("[Verse 1]\nthe old song\n\n[Chorus]\nthe old hook\n",
                             encoding="utf-8")
    return p


def _stamp_track(project: VideoProject, sheet: str | None, *, age: float = -100) -> None:
    """Write a track with ``sheet`` in its tag, ``age`` seconds away from lyrics.txt's mtime."""
    project.audio_dir.mkdir(parents=True, exist_ok=True)
    project.track_path.write_bytes(b"\xff\xfb\x90\x00" * 64)
    if sheet is not None:
        tags = ID3()
        tags.add(USLT(encoding=3, lang="eng", desc="", text=sheet))
        tags.save(str(project.track_path))
    mtime = project.lyrics_path.stat().st_mtime - age
    os.utime(project.track_path, (mtime, mtime))


def test_embedded_lyrics_reads_the_tag_and_tolerates_a_track_without_one(project):
    _stamp_track(project, "[Verse 1]\nthe new song")
    assert embedded_lyrics(project.track_path) == "[Verse 1]\nthe new song"
    _stamp_track(project, None)
    assert embedded_lyrics(project.track_path) == ""


def test_track_with_another_sheet_replaces_lyrics_and_keeps_the_old_text(project):
    old = project.lyrics_path.read_text(encoding="utf-8")
    _stamp_track(project, "[Intro: organ]\nnine\n[Spoken]\nenkai\n[Verse 1: calm]\nthe new song")

    assert sync_lyrics_with_track(project) is True

    # Blank lines go back before sections only, not before delivery tags like [Spoken].
    assert project.lyrics_path.read_text(encoding="utf-8") == (
        "[Intro: organ]\nnine\n[Spoken]\nenkai\n\n[Verse 1: calm]\nthe new song")
    assert project.previous_lyrics_path.read_text(encoding="utf-8") == old
    # Having synced, lyrics.txt is now the newer file: running again changes nothing.
    assert sync_lyrics_with_track(project) is False


def test_same_words_with_different_tags_are_not_a_change(project):
    original = project.lyrics_path.read_text(encoding="utf-8")
    _stamp_track(project, "[Verse 1: calm]\nthe old song\n[Chorus: sung]\nthe old hook")
    assert sync_lyrics_with_track(project) is False
    assert project.lyrics_path.read_text(encoding="utf-8") == original
    assert not project.previous_lyrics_path.exists()


def test_pasting_the_respelled_sheet_keeps_the_canonical_names(project):
    project.lyrics_path.write_text("[Drop]\nAuswählen. Come home.", encoding="utf-8")
    project.suno_prompt_path.write_text(
        json.dumps({"pronunciations": {"Auswählen": "Ows-vay-len"}}), encoding="utf-8")
    _stamp_track(project, "[Drop]\nOws-vay-len. Come home.")
    assert sync_lyrics_with_track(project) is False
    assert "Auswählen" in project.lyrics_path.read_text(encoding="utf-8")


def test_a_line_edited_in_suno_comes_back_with_the_canonical_names(project):
    project.lyrics_path.write_text("[Drop]\nAuswählen. Come home.", encoding="utf-8")
    project.suno_prompt_path.write_text(
        json.dumps({"pronunciations": {"Auswählen": "Ows-vay-len"}}), encoding="utf-8")
    _stamp_track(project, "[Drop]\nOws-vay-len. Come home now.")
    assert sync_lyrics_with_track(project) is True
    assert project.lyrics_path.read_text(encoding="utf-8") == "[Drop]\nAuswählen. Come home now."


def test_lyrics_corrected_by_hand_after_the_track_win_over_the_tag(project):
    original = project.lyrics_path.read_text(encoding="utf-8")
    _stamp_track(project, "[Verse 1]\nwhat suno was given", age=100)
    assert sync_lyrics_with_track(project) is False
    assert project.lyrics_path.read_text(encoding="utf-8") == original


def test_track_without_a_tag_keeps_lyrics_txt(project):
    _stamp_track(project, None)
    assert sync_lyrics_with_track(project) is False


def test_align_project_refuses_lyrics_the_audio_does_not_contain(project, monkeypatch):
    _stamp_track(project, None)
    words = "the old song the old hook".split()
    monkeypatch.setattr(alignment_service.stable_whisper_client, "align_words",
                        lambda *a, **k: [{"word": w, "start": 28.0, "end": 28.0} for w in words])
    with pytest.raises(ValueError, match="does not look like the song"):
        align_project(project)
    assert not project.alignment_path.exists()


def test_align_project_aligns_the_sheet_inside_the_track(project, monkeypatch):
    _stamp_track(project, "[Verse 1]\nthe new song")
    seen = {}

    def fake_align(_path, text, **_kwargs):
        seen["text"] = text
        return [{"word": w, "start": i * 1.0, "end": i + 0.5} for i, w in enumerate(text.split())]

    monkeypatch.setattr(alignment_service.stable_whisper_client, "align_words", fake_align)
    alignment = align_project(project)
    assert seen["text"] == "the new song"
    assert alignment.word_count == 3


def test_cached_timestamps_are_not_reused_for_another_sheet(project, monkeypatch):
    _stamp_track(project, None)
    cache = project.whisper_cache_path
    fresh = [{"word": "new", "start": 1.0, "end": 2.0}]

    class _Word:
        word, start, end = " new", 1.0, 2.0

    class _Result:
        segments = [SimpleNamespace(words=[_Word()])]

    calls = []
    model = SimpleNamespace(align=lambda *a, **k: (calls.append(a), _Result())[1])
    monkeypatch.setitem(sys.modules, "stable_whisper",
                        SimpleNamespace(load_model=lambda _name: model))

    assert stable_whisper_client.align_words(project.track_path, "new", cache_path=cache) == fresh
    assert stable_whisper_client.align_words(project.track_path, "new", cache_path=cache) == fresh
    assert len(calls) == 1  # same pair: served from the cache
    stable_whisper_client.align_words(project.track_path, "other", cache_path=cache)
    assert len(calls) == 2  # another sheet: aligned again


def test_alignment_does_not_skip_long_silences(project, monkeypatch):
    # stable-ts' default cut the audio window at a break and piled a whole verse before it.
    _stamp_track(project, None)
    options = {}
    model = SimpleNamespace(align=lambda *a, **k: (options.update(k), SimpleNamespace(segments=[]))[1])
    monkeypatch.setitem(sys.modules, "stable_whisper",
                        SimpleNamespace(load_model=lambda _name: model))

    stable_whisper_client.align_words(project.track_path, "new")
    assert options["nonspeech_skip"] is None


def test_uploading_a_track_drops_captions_timed_against_the_old_one(project):
    from fastapi.testclient import TestClient

    from cutforge.api.app import create_app

    project.audio_dir.mkdir(parents=True, exist_ok=True)
    stale = (project.alignment_path, project.captions_srt_path, project.premiere_transcript_path)
    for path in stale:
        path.write_text("{}", encoding="utf-8")

    response = TestClient(create_app()).post(f"/api/runs/{project.run_id}/track",
                                             files={"file": ("take2.mp3", b"mp3", "audio/mpeg")})

    assert response.status_code == 200
    assert project.track_path.read_bytes() == b"mp3"
    assert not any(path.exists() for path in stale)
