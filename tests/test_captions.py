"""Tests for caption generation — SRT and Premiere transcript."""
from cutforge.models.alignment import LyricLine, LyricWord
from cutforge.services import caption_service


def _sample_lines() -> list[LyricLine]:
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


def test_srt_segments_into_short_uppercase_groups():
    srt = caption_service.build_srt(_sample_lines(), words_per_group=3)
    blocks = [b for b in srt.strip().split("\n\n") if b]
    # 2 lines x 2 groups each (3+1 words) = 4 SRT blocks.
    assert len(blocks) == 4
    assert "SHADOW MONARCH RISE" in srt
    assert "-->" in srt and "," in srt


def test_premiere_transcript_matches_adobe_schema():
    t = caption_service.build_premiere_transcript(_sample_lines(), language="en")
    assert set(t.keys()) == {"language", "segments", "speakers"}
    assert t["language"] == "en-us"
    assert len(t["speakers"]) == 1 and t["speakers"][0]["name"]
    speaker_id = t["speakers"][0]["id"]
    assert len(t["segments"]) == 2
    seg = t["segments"][0]
    assert set(seg.keys()) == {"duration", "language", "speaker", "start", "words"}
    assert seg["speaker"] == speaker_id
    w = seg["words"][0]
    assert set(w.keys()) == {"confidence", "duration", "eos", "start", "tags", "text", "type"}
    assert w["type"] == "word" and w["tags"] == []
    assert seg["words"][-1]["eos"] is True
