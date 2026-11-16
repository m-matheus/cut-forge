"""Deterministic checks on a lyric sheet, right before it goes to Suno.

The writer gets the same rules in its prompt, but a prompt is no guarantee. This module
fixes what is safe to fix (sound effects in parentheses, which Suno would sing as a
backing vocal) and reports what needs a human ear: oversized tags, a sheet too long for
one generation, rap sections that are choppy or rushed and a style string off the rules.

The thresholds were calibrated on the channel's own songs: the line ratios flag the choppy
verses of Yhwach and the dense verses that lost the Obito A/B test, and no song so far
passed 14 sections.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from cutforge.services import catalog_service

# Suno's Lyrics box holds ~5,000 characters on v4.5+ and cuts the rest silently. That
# figure comes from third-party guides (Suno does not publish it), hence the margin.
SUNO_LYRICS_MAX_CHARS = 4500
MAX_SECTIONS = 16
MAX_TAG_WORDS = 5
STYLE_MAX_CHARS = 200
MAX_EFFECT_TAG_WORDS = 4   # a longer effect is dropped instead of becoming a tag
SHORT_LINE_WORDS = 3
LONG_LINE_WORDS = 10       # roughly the 14-syllable ceiling for rap lines
MAX_LINE_RATIO = 0.25      # share of a rap section allowed outside those two bounds
MIN_RAP_LINES = 4          # below this the ratio says nothing

_PARENTHETICAL = re.compile(r"\(([^)]*)\)")
_TAG = re.compile(r"\[([^\]]*)\]")
_TAG_LINE = re.compile(r"^\[[^\]]*\]$")
_VERSE_TAG = re.compile(r"^\[verse\b", re.IGNORECASE)
_RAP = re.compile(r"\brap(?:s|ped|ping)?\b", re.IGNORECASE)
# A tag that changes HOW a passage is delivered. Others ([Organ swells]) leave it as it was.
_DELIVERY = re.compile(r"\b(?:rap\w*|sung|sing\w*|spoken|whisper\w*|shout\w*|chant\w*|belt\w*)\b",
                       re.IGNORECASE)

# What separates a sound effect from a sung ad-lib: it names a sound or an instrument and
# nobody is speaking. The list can't be complete, so lint_lyrics reports what it kept.
_SOUND_CUE = re.compile(
    r"\b(?:swell|crack|toll|sputter|creak|hiss|clang|clank|shatter|echo|crash|gasp|howl|roar|"
    r"sigh|laugh|sob)(?:s|es|ed|ing)?\b"
    r"|\b(?:fad|rattl|settl|rumbl)(?:e|es|ed|ing)\b"
    r"|\bgrind(?:s|ing)\b"
    r"|\b(?:organ|piano|choir|drums?|strings|bass|guitar|808s?|thunder|wind|rain|breath|"
    r"footsteps?|heartbeat|static|silence|sfx|explosion|instrumental|spoken|whispered|"
    r"whispering|shouted)\b",
    re.IGNORECASE,
)
_PRONOUN = re.compile(r"\b(?:i|me|my|mine|you|your|we|us|our|he|him|his|she|her|they|them|"
                      r"their|it|its)\b|'em\b", re.IGNORECASE)

# The lanes named in the writer's prompt: the artist names most likely to leak into a style.
_ARTIST_NAME = re.compile(r"\b(?:basara|m4rkim|anirap|rustage|7 minutoz|divide music)\b",
                          re.IGNORECASE)


@dataclass
class LintResult:
    lyrics: str                                       # the sheet after the automatic fixes
    fixes: list[str] = field(default_factory=list)    # what was changed
    adlibs: list[str] = field(default_factory=list)   # parentheses kept, Suno will sing them
    warnings: list[str] = field(default_factory=list)


def _is_sound_effect(text: str) -> bool:
    return bool(_SOUND_CUE.search(text)) and not _PRONOUN.search(text)


def _fix_sound_effects(lyrics: str) -> LintResult:
    """Move sound effects out of parentheses: a short tag on its own line, or nothing."""
    result = LintResult(lyrics)
    out: list[str] = []
    for line in lyrics.splitlines():
        if _TAG_LINE.match(line.strip()):
            out.append(line)
            continue
        tags: list[str] = []

        def swap(match: re.Match) -> str:
            text = match.group(1).strip()
            if not _is_sound_effect(text):
                result.adlibs.append(text)
                return match.group(0)
            if len(text.split()) > MAX_EFFECT_TAG_WORDS:
                result.fixes.append(f"({text}) removido")
            else:
                tags.append(f"[{text[0].upper()}{text[1:]}]")
                result.fixes.append(f"({text}) virou {tags[-1]}")
            return ""

        rest = _PARENTHETICAL.sub(swap, line)
        out += tags
        if rest == line:
            out.append(line)
        elif rest.strip():
            out.append(" ".join(rest.split()))
    if result.fixes:
        result.lyrics = "\n".join(out)
    return result


def _oversized_tags(lyrics: str) -> list[str]:
    """Only the description counts: "[Final Chorus: belted, full choir]" has 3 words."""
    warnings = []
    for match in _TAG.finditer(lyrics):
        words = len(match.group(1).split(":", 1)[-1].split())
        if words > MAX_TAG_WORDS:
            warnings.append(f"Tag longa ({words} palavras): {match.group(0)}. O Suno pode cantar "
                            "descrições longas; deixe em até 4 palavras.")
    return warnings


def _rap_lines(lyrics: str) -> list[tuple[str, list[str]]]:
    """``[(section_tag, lines delivered as rap)]`` in order, for sections that have any.

    On this channel a verse is rap unless its tag says otherwise, so a verse tagged only
    with a mood ("[Verse 4: slow, imperious]") still counts.
    """
    blocks: list[tuple[str, list[str]]] = []
    rap = False
    for raw in lyrics.splitlines():
        line = raw.strip()
        if not line:
            continue
        if catalog_service.is_section_tag(line):
            blocks.append((line, []))
            rap = bool(_RAP.search(line)) or (bool(_VERSE_TAG.match(line))
                                              and not _DELIVERY.search(line))
        elif _TAG_LINE.match(line):
            if _DELIVERY.search(line):
                rap = bool(_RAP.search(line))
        elif rap and blocks:
            blocks[-1][1].append(line)
    return [block for block in blocks if block[1]]


def _rap_flow_warnings(lyrics: str) -> list[str]:
    warnings = []
    for tag, lines in _rap_lines(lyrics):
        if len(lines) < MIN_RAP_LINES:
            continue
        words = [len(_PARENTHETICAL.sub("", ln).split()) for ln in lines]
        short = sum(n <= SHORT_LINE_WORDS for n in words)
        long = sum(n > LONG_LINE_WORDS for n in words)
        if short / len(lines) > MAX_LINE_RATIO:
            warnings.append(f"{tag}: {short} de {len(lines)} linhas de rap têm até "
                            f"{SHORT_LINE_WORDS} palavras ({short / len(lines):.0%}). O flow "
                            "tende a sair picotado.")
        if long / len(lines) > MAX_LINE_RATIO:
            warnings.append(f"{tag}: {long} de {len(lines)} linhas de rap têm mais de "
                            f"{LONG_LINE_WORDS} palavras ({long / len(lines):.0%}). O Suno tende "
                            "a atropelar linhas densas.")
    return warnings


def _split_point(lyrics: str) -> str:
    """Describe the section boundary closest to the middle of the sheet."""
    offset = 0
    candidates: list[tuple[str, int]] = []
    for line in lyrics.splitlines(keepends=True):
        if offset and catalog_service.is_section_tag(line):
            candidates.append((line.strip(), offset))
        offset += len(line)
    if not candidates:
        return "Divida em partes e continue com o Extend."
    tag, before = min(candidates, key=lambda c: abs(c[1] - offset / 2))
    same = [at for name, at in candidates if name == tag]
    label = f"do {same.index(before) + 1}º {tag}" if len(same) > 1 else f"de {tag}"
    return (f"Gere até antes {label} ({before} caracteres) e continue o restante "
            f"({offset - before} caracteres) com o Extend.")


def check_length(lyrics: str) -> list[str]:
    """Warn when the text pasted into Suno won't fit a single generation."""
    count = len(catalog_service.sections(lyrics))
    if len(lyrics) <= SUNO_LYRICS_MAX_CHARS and count <= MAX_SECTIONS:
        return []
    return [f"Letra longa para uma geração só do Suno ({len(lyrics)} caracteres, {count} "
            f"seções). {_split_point(lyrics)}"]


def check_style(style: str) -> list[str]:
    warnings = []
    if len(style) > STYLE_MAX_CHARS:
        warnings.append(f"Estilo com {len(style)} caracteres (máximo ~{STYLE_MAX_CHARS}): o Suno "
                        "dilui o som em listas longas.")
    for match in _ARTIST_NAME.finditer(style):
        warnings.append(f"Estilo cita um artista ({match.group(0)}): descreva o som em vez disso.")
    return warnings


def lint_lyrics(lyrics: str) -> LintResult:
    """Apply the automatic fixes and report what is left for a human to judge."""
    result = _fix_sound_effects(lyrics)
    result.warnings = _oversized_tags(result.lyrics) + _rap_flow_warnings(result.lyrics)
    return result
