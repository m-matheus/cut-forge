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


def _line(*words: str, start: float = 0.0) -> LyricLine:
    """A line of back-to-back words, short enough that max_words drives chunking."""
    ws = [LyricWord(word=w, start=start + i * 0.2, end=start + i * 0.2 + 0.2)
          for i, w in enumerate(words)]
    return LyricLine(start=ws[0].start, end=ws[-1].end, words=ws)


def _texts(srt: str) -> list[str]:
    """The caption text of every SRT block, in order."""
    return [b.split("\n")[2] for b in srt.strip().split("\n\n") if b]


def test_srt_segments_into_short_uppercase_groups():
    srt = caption_service.build_srt(_sample_lines(), max_words=3)
    blocks = [b for b in srt.strip().split("\n\n") if b]
    # 2 lines x 2 groups each (3+1 words) = 4 SRT blocks.
    assert len(blocks) == 4
    assert "SHADOW MONARCH RISE" in srt
    assert "-->" in srt and "," in srt


def test_srt_drops_lyric_sheet_tags():
    srt = caption_service.build_srt([_line("[Spoken]", "enkai", "again...")])
    assert _texts(srt) == ["ENKAI AGAIN"]


def test_srt_drops_em_dashes_but_keeps_hyphens():
    srt = caption_service.build_srt([
        _line("bankai!", "—", "that's", "the", "warning"),
        _line("a", "ghost-white", "edge", start=4.0),
    ])
    assert _texts(srt) == ["BANKAI! THAT'S THE WARNING", "A GHOST-WHITE EDGE"]


def test_srt_strips_final_punctuation_but_keeps_delivery_marks():
    srt = caption_service.build_srt([
        _line("who", "bled", "into", "a", "blade."),
        _line("test", "me", "again?", start=4.0),
    ])
    assert _texts(srt) == ["WHO BLED INTO A BLADE", "TEST ME AGAIN?"]


def test_srt_moves_word_stranded_after_a_comma_to_the_next_block():
    # "renji blocks my path, kenpachi / grins for the kill" should read as
    # "RENJI BLOCKS MY PATH" / "KENPACHI GRINS FOR THE KILL".
    srt = caption_service.build_srt(
        [_line("renji", "blocks", "my", "path,", "kenpachi",
               "grins", "for", "the", "kill")],
        max_words=5,
    )
    assert _texts(srt) == ["RENJI BLOCKS MY PATH", "KENPACHI GRINS FOR THE KILL"]


def _blocks(srt: str) -> list[tuple[str, str, str]]:
    """(start, end, text) of every SRT block, in order."""
    out = []
    for block in srt.strip().split("\n\n"):
        _, times, text = block.split("\n")
        start, end = times.split(" --> ")
        out.append((start, end, text))
    return out


def test_srt_never_stacks_captions_when_words_collapse_onto_one_instant():
    # The real shape stable-ts returned: whole lines with no duration on one timestamp.
    def collapsed(*words: str) -> LyricLine:
        return LyricLine(start=28.096, end=28.096,
                         words=[LyricWord(word=w, start=28.096, end=28.096) for w in words])
    lines = [collapsed("awake."), collapsed("shaped", "me", "first."),
             collapsed("one", "war."), _line("other", "door.", start=38.72)]

    blocks = _blocks(caption_service.build_srt(lines))

    assert [b[2] for b in blocks] == ["AWAKE", "SHAPED ME FIRST", "ONE WAR", "OTHER DOOR"]
    assert blocks[0][0] == "00:00:28,096"
    for current, following in zip(blocks, blocks[1:]):
        assert current[1] <= following[0], f"{current} overlaps {following}"
    assert all(start < end for start, end, _ in blocks)
    # The pile-up is paid back at the next real gap instead of delaying the rest of the song.
    assert blocks[3][0] == "00:00:38,720"


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
