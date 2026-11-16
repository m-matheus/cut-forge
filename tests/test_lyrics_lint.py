"""Tests for the deterministic checks that run on a lyric sheet before it reaches Suno."""
import pytest

from cutforge.services import alignment_service, catalog_service, lyrics_lint_service as lint
from cutforge.services import release_service


# The first two lists are real parentheticals from the channel's own songs.
@pytest.mark.parametrize("text", [
    "engine sputters", "stone grinding, dust settling", "strained breath", "dust settling",
    "organ swells", "steel cracking", "stone cracking", "wind", "chains rattle", "bell tolls",
    "piano fades", "fading",
])
def test_sound_effects_are_recognised(text):
    assert lint._is_sound_effect(text)


@pytest.mark.parametrize("text", [
    "Rise up", "uh", "pull it", "one more time", "ha", "still holdin'", "open", "bury 'em",
    "all of 'em", "nothing", "shh", "I wake", "yeah", "let's go", "boom",
    # A sound word in someone's mouth is still a sung line.
    "hear my breath", "let it rain", "you crack",
])
def test_sung_adlibs_are_left_alone(text):
    assert not lint._is_sound_effect(text)


def test_whole_line_effect_becomes_a_short_tag():
    result = lint.lint_lyrics("[Intro]\nOne more.\n(organ swells)\n\n[Verse 1]\nBroken.")
    assert result.lyrics == "[Intro]\nOne more.\n[Organ swells]\n\n[Verse 1]\nBroken."
    assert result.fixes == ["(organ swells) virou [Organ swells]"]


def test_inline_effect_moves_to_its_own_line_before_the_lyric():
    result = lint.lint_lyrics("Broken (steel cracking) now")
    assert result.lyrics == "[Steel cracking]\nBroken now"


def test_effect_too_long_for_a_tag_is_removed():
    result = lint.lint_lyrics("Line one\n(the distant wind slowly dies away)\nLine two")
    assert result.lyrics == "Line one\nLine two"
    assert "removido" in result.fixes[0]


def test_sung_adlibs_stay_and_are_reported():
    lyrics = "Burn it all (yeah)\n\n(one more time)"
    result = lint.lint_lyrics(lyrics)
    assert result.lyrics == lyrics
    assert result.fixes == []
    assert result.adlibs == ["yeah", "one more time"]


def test_parentheses_inside_a_tag_are_not_touched():
    lyrics = "[Build (drums rising)]\nLine"
    assert lint.lint_lyrics(lyrics).lyrics == lyrics


def test_tag_word_count_ignores_the_section_label():
    fine = "[Final Chorus: belted, full choir, biggest moment]\nLine"
    assert lint.lint_lyrics(fine).warnings == []
    long = "[Pipe organ only, tender child-soft delivery, almost no drums]\nLine"
    warnings = lint.lint_lyrics(long).warnings
    assert len(warnings) == 1 and "9 palavras" in warnings[0]


def _rap_section(header, short, full):
    return "\n".join(header + ["One war."] * short + ["your blades came through my people"] * full)


@pytest.mark.parametrize("header", [
    ["[Verse 1: aggressive rap]"],
    ["[Verse 1]", "[Calm half-spoken rap, sparse hats]"],
])
def test_choppy_rap_is_flagged_for_both_tag_formats(header):
    warnings = lint.lint_lyrics(_rap_section(header, 3, 5)).warnings
    assert len(warnings) == 1
    assert "3 de 8" in warnings[0] and "38%" in warnings[0]


def test_short_lines_at_the_ratio_limit_or_outside_rap_are_fine():
    assert lint.lint_lyrics(_rap_section(["[Verse 1: rap]"], 2, 6)).warnings == []
    assert lint.lint_lyrics(_rap_section(["[Chorus: sung, emotional]"], 8, 0)).warnings == []
    assert lint.lint_lyrics(_rap_section(["[Verse 1: rap]"], 2, 1)).warnings == []   # too few


def test_dense_rap_is_flagged():
    dense = "captains on their knees with the bankai pulled out from their spines"
    fine = "your blades came through my people"
    assert lint.lint_lyrics("\n".join(["[Verse 1: rap]"] + [dense] * 2 + [fine] * 6)).warnings == []
    warnings = lint.lint_lyrics("\n".join(["[Verse 1: rap]"] + [dense] * 3 + [fine] * 5)).warnings
    assert len(warnings) == 1
    assert "3 de 8" in warnings[0] and "mais de 10 palavras" in warnings[0]


def test_a_verse_counts_as_rap_unless_its_tag_says_otherwise():
    assert lint.lint_lyrics(_rap_section(["[Verse 4: slow, imperious, wide]"], 3, 5)).warnings
    assert lint.lint_lyrics(_rap_section(["[Verse 4]", "[Organ only, tender]"], 3, 5)).warnings
    assert not lint.lint_lyrics(_rap_section(["[Verse 4: sung, soft]"], 3, 5)).warnings
    assert not lint.lint_lyrics(_rap_section(["[Verse 4]", "[Whispered]"], 3, 5)).warnings
    assert not lint.lint_lyrics(_rap_section(["[Bridge: slow, imperious]"], 3, 5)).warnings


def test_delivery_tags_switch_rap_on_and_off_but_effects_do_not():
    spoken = "[Verse 1: rap]\n" + "\n".join(["your blades came through my people"] * 6) \
        + "\n[Spoken, flat]\nOne war.\nOne future.\nNo door.\nFine."
    assert lint.lint_lyrics(spoken).warnings == []
    effect = "[Verse 1: rap]\nyour blades came through my people\n[Steel cracking]\n" \
        + "\n".join(["One war."] * 4)
    assert "4 de 5" in lint.lint_lyrics(effect).warnings[0]


def test_length_warning_suggests_the_section_boundary_closest_to_the_middle():
    assert lint.check_length("[Verse 1]\nshort") == []
    verse = "x" * 1600
    sheet = f"[Verse 1]\n{verse}\n[Chorus]\n{verse}\n[Drop]\n{verse}\n[Chorus]\n{verse}"
    warning = lint.check_length(sheet)[0]
    assert f"{len(sheet)} caracteres, 4 seções" in warning
    assert "antes de [Drop]" in warning and "Extend" in warning


def test_length_warning_names_which_repeat_of_a_section():
    verse = "x" * 1600
    sheet = f"[Verse 1]\n{verse}\n[Chorus]\n{verse}\n[Chorus]\n{verse}\n[Outro]\n{verse}"
    assert "antes do 2º [Chorus]" in lint.check_length(sheet)[0]


def test_too_many_sections_warns_even_when_short():
    sheet = "\n".join(f"[Verse {n}]\nline" for n in range(lint.MAX_SECTIONS + 1))
    assert "17 seções" in lint.check_length(sheet)[0]


def test_style_checks():
    assert lint.check_style("dark trap, deep 808, male rap verses, 140 BPM") == []
    assert "201 caracteres" in lint.check_style("x" * 201)[0]
    assert "Rustage" in lint.check_style("orchestral trap like Rustage, 150 BPM")[0]


def test_inline_section_tags_work_with_every_lyrics_consumer():
    """The short tag format must not leak into captions, the catalog or the release lyrics."""
    sheet = "[Intro: soft piano]\nEnkai.\n\n[Verse 2: aggressive rap]\nFirst line\n" \
            "[Bell tolls]\nSecond line\n\n[Final Chorus: belted, full choir]\nThe hook"

    assert alignment_service.parse_lyrics(sheet) == [
        ["Enkai."], ["First", "line"], ["Second", "line"], ["The", "hook"]]
    assert [name for name, _ in catalog_service.sections(sheet)] == ["intro", "verse", "chorus"]
    assert catalog_service.section(sheet, "chorus") == "The hook"
    assert catalog_service.replace_section(sheet, "intro", "Enkai, awake.").startswith(
        "[Intro: soft piano]\nEnkai, awake.")
    assert release_service.clean_lyrics(sheet) == \
        "Enkai.\n\nFirst line\nSecond line\n\nThe hook"
