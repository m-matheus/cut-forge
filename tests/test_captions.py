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
