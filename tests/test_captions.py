"""Tests for caption ASS generation — karaoke and kinetic styles."""
from cutforge.models.alignment import Alignment, LyricLine, LyricWord
from cutforge.services import caption_service


def _sample_lines() -> list[LyricLine]:
    # Two lines, 4 words each — enough to exercise phrase grouping.
    def word(w, s):
        return LyricWord(word=w, start=s, end=s + 0.4)
    return [
        LyricLine(start=0.0, end=2.0, words=[
            word("shadow", 0.0), word("monarch", 0.5),
            word("rise", 1.0), word("now", 1.5),
        ]),
        LyricLine(start=2.0, end=4.0, words=[
            word("arise", 2.0), word("my", 2.5),
            word("king", 3.0), word("again", 3.5),
        ]),
    ]


def test_karaoke_one_dialogue_per_line():
    ass = caption_service.build_ass_karaoke(_sample_lines(), color="cyan", unsung="white")
    dialogues = [l for l in ass.splitlines() if l.startswith("Dialogue:")]
    assert len(dialogues) == 2  # one per lyric line
    assert "Karaoke" in ass
    assert "\\k" in ass  # per-word highlight tags


def test_kinetic_groups_words():
    lines = _sample_lines()  # 8 words total, 4 per line
    ass = caption_service.build_ass_music_kinetic(
        lines, color="yellow", unsung="white", words_per_group=3)
    dialogues = [l for l in ass.splitlines() if l.startswith("Dialogue:")]
    # 4 words/line, 3/group, never crossing a line => 2 groups per line => 4 total.
    assert len(dialogues) == 4
    assert "KineticCenter" in ass
    assert "\\pos" in ass and "\\fscx" in ass  # centered + pop-in bounce


def test_srt_segments_into_short_uppercase_groups():
    srt = caption_service.build_srt(_sample_lines(), words_per_group=3)
    blocks = [b for b in srt.strip().split("\n\n") if b]
    # 2 lines x 2 groups each (3+1 words) = 4 SRT blocks.
    assert len(blocks) == 4
    assert "SHADOW MONARCH RISE" in srt   # uppercase, grouped
    assert "-->" in srt and "," in srt    # SRT comma-decimal timestamps


def test_premiere_transcript_matches_adobe_schema():
    t = caption_service.build_premiere_transcript(_sample_lines(), language="en")
    assert set(t.keys()) == {"language", "segments", "speakers"}
    assert t["language"] == "en-us"
    assert len(t["speakers"]) == 1 and t["speakers"][0]["name"]
    speaker_id = t["speakers"][0]["id"]
    assert len(t["segments"]) == 2
    seg = t["segments"][0]
    assert set(seg.keys()) == {"duration", "language", "speaker", "start", "words"}
    assert seg["speaker"] == speaker_id      # linkage by UUID
    w = seg["words"][0]
    # Every Adobe-required word field present, times in seconds.
    assert set(w.keys()) == {"confidence", "duration", "eos", "start", "tags", "text", "type"}
    assert w["type"] == "word" and w["tags"] == []
    assert seg["words"][-1]["eos"] is True   # last word ends the sentence

