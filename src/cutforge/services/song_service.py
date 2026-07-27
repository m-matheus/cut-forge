"""Song generation service — suggests genre directions and generates the full Suno package.

System prompts ported from the old ``generate_song_prompt.py``, extended with a
``language`` parameter so a run can be produced in EN, ES or PT (one per run).
"""
from __future__ import annotations

import json

from cutforge.integrations import anthropic_client
from cutforge.models.project import VideoProject
from cutforge.models.song import GenreDirection, GenreSuggestions, SongPackage

_LANG_NAMES = {"en": "English", "es": "Spanish", "pt": "Brazilian Portuguese"}


SUGGEST_SYSTEM_PROMPT = """\
You are a music director for an anime music channel (think 7 Minutoz / Rustage / Sensei Beats).
Given a character or matchup, propose distinct GENRE/VIBE directions for an original song
that matches that character's personality and power fantasy.

Study the range these channels cover:
- Dark, calculating villains -> aggressive drill or dark trap, menacing 808s, sparse and cold
- Unstoppable heroes / power-ups -> epic orchestral trap, cinematic strings + hard drums
- Hot-blooded shonen fighters -> high-energy rap rock, electric guitars, chant hooks
- Tragic / emotional characters -> melodic trap, piano, emotional but still hard-hitting
- Ancient / godlike beings -> cinematic hybrid orchestral, choir, war drums

For the given character(s), propose 3 directions. Each must genuinely fit THAT character —
tie the genre to their personality, not generic "epic anime song".

OUTPUT FORMAT
Return a single valid JSON object. No markdown fences, no commentary.

{
  "character_read": "1-2 sentences on the character's vibe that should drive the music",
  "directions": [
    {
      "label": "Short genre name (e.g. 'Cold Drill')",
      "style": "The Suno style string — comma-separated genre/mood/instrumentation/tempo descriptors",
      "why": "One sentence: why this fits the character"
    }
  ]
}

Rules:
- Exactly 3 directions, each clearly different from the others.
- style must be Suno-ready: genres + mood + key instruments + tempo/energy, comma-separated.
- All descriptors in English (Suno reads English style strings best).
"""


GENERATE_SYSTEM_PROMPT = """\
You are a songwriter for an anime music channel (7 Minutoz / Rustage / Sensei Beats style).
You write an original song about a specific anime character (or matchup) that will be
generated on Suno AI and played over AMV footage on YouTube.

You produce a COMPLETE Suno package: a style string, structured lyrics, and a title.

TONE & CONTENT
- The song is FROM or ABOUT the character — capture their personality, powers, and arc
  through vivid, specific imagery. Not a plot summary — a hype anthem / character piece.
- Reference the character's actual abilities, iconic moments, and personality traits, but
  through metaphor and attitude, not exposition.
- Match the requested genre's energy in the word choice and rhythm.
- Keep it hype and quotable — the hook should be something viewers scream in the comments.
- Do NOT use trademarked catchphrases verbatim; evoke them instead.

LYRICS STRUCTURE (for a ~2.5-3.5 minute song)
Use Suno section tags on their own lines. A strong structure:
  [Intro]        — short, sets the mood (2-4 lines or an atmospheric line)
  [Verse 1]      — establish the character, their world, their attitude (6-8 lines)
  [Chorus]       — the hook. Punchy, repeatable, the emotional core (4 lines)
  [Verse 2]      — escalate: their power, a defining moment, a turn (6-8 lines)
  [Chorus]       — repeat the hook
  [Bridge]       — shift energy: a threat, a vow, or a quiet-before-storm moment (4-6 lines)
  [Outro]        — final hook variation or a hard closing line (2-4 lines)
For a "vs" / matchup song, alternate perspectives and make the chorus the clash.

LINE RULES (these lyrics also become on-screen karaoke captions)
- Each line should be a natural caption length — roughly 4-9 words. Avoid very long lines.
- One idea per line. Clean, punchy phrasing reads better on screen.
- Real words only — no "yeah yeah" filler padding beyond the occasional intentional hook ad-lib.

OUTPUT FORMAT
Return a single valid JSON object. No markdown fences, no commentary.

{
  "title": "Song title — short and evocative (just the song name, e.g. 'Shadow Sovereign')",
  "style": "Suno style string — the chosen genre plus mood/instruments/tempo, comma-separated. MUST begin with the required vocal-language tag.",
  "lyrics": "Full structured lyrics as a single string. Use \\n for line breaks and put each [Section] tag on its own line.",
  "suno_tips": "One short line of advice for the Suno generation"
}

Rules:
- lyrics MUST include section tags ([Intro], [Verse 1], [Chorus], etc.) each on its own line.
- Target 2.5-3.5 minutes of lyrics — enough sections, not padded.
"""


def _lang_directive(language: str) -> str:
    lang = _LANG_NAMES.get(language, "English")
    return (
        f"LANGUAGE: Write the LYRICS in {lang}. "
        f"The Suno style string MUST start with the vocal-language tag "
        f"'{lang} vocals, sung in {lang}, ' followed by the genre descriptors. "
        f"Keep genre/instrument descriptors in English."
    )


SUGGEST_MOOD_SYSTEM_PROMPT = """\
You are a creative director for an anime music channel (7 Minutoz / Rustage / Sensei Beats style).
Given an anime character (or matchup), describe the MOOD / VIBE that an original song about them
should have — the emotional atmosphere that drives the music, caption colors and thumbnail.

Return a SINGLE short line: 3-6 comma-separated descriptors capturing the character's energy
(e.g. "dark, cold, ominous power, shadow army" or "hot-blooded, explosive, heroic, hype").
No commentary, no quotes, no markdown — just the descriptors line.
"""


# --- Reference-inspiration addenda, keyed by how much CONTENT may be borrowed ---
# Every level shares one hard, non-negotiable anti-plagiarism floor; the levels differ
# only in how much theme / imagery / hook-shape / phrasing may be reused.

_REFERENCE_HARD_FLOOR = """\
HARD FLOOR (applies at EVERY level — never violate):
- NEVER reproduce any line, hook, or distinctive phrase from the reference transcript
  verbatim, and never merely swap a word or two — that is still copying.
- NEVER reuse the reference's proper nouns, character names, or brand catchphrases.
- The transcript is auto-transcribed and may contain mis-heard words.
ALWAYS match the reference's RHYTHM:
- The tempo/energy feel (target roughly the given BPM) and rhythmic density.
- The flow: line length and syllables-per-bar. Faster BPM / higher words-per-second =>
  shorter, denser lines; slower => more spacious lines.
Include a tempo tag near the target BPM in the Suno style string.
"""

_REFERENCE_CONTENT_RULES = {
    "rhythm": """\
CONTENT: Use the reference ONLY as a rhythm/energy signal. Do NOT borrow its themes,
imagery, message, or hook structure — write the character piece entirely from your own
ideas. The transcript is vibe-only; treat its words as off-limits.""",
    "light": """\
CONTENT (light borrow): You MAY echo 1-2 of the reference's core themes and its overall
tone, but expressed in completely fresh words. Do NOT copy its specific images or
metaphors — invent your own that fit the CutForge character.""",
    "moderate": """\
CONTENT (moderate borrow): You MAY reuse the reference's core themes, its semantic field
of imagery/metaphors, and its hook shape (e.g. a short repeating chorus, call-and-response,
a chant) — all rewritten around the CutForge character with fresh wording. No verbatim
phrases; the images should clearly rhyme with the reference's world without quoting it.""",
    "strong": """\
CONTENT (strong borrow): You MAY take the reference's themes, imagery, message and hook
structure, follow its section arc, AND lift its most striking phrases as inspiration —
but you MUST heavily rewrite each one (swap the vocabulary, the referents, and the angle
so it reads as a new line about the CutForge character). Aim for "unmistakably inspired by,
never a copy." The hard floor above still holds: no literal text, no proper nouns reused.""",
}


def _reference_addendum(level: str) -> str:
    """Return the reference system-prompt addendum for a content-blend level."""
    rules = _REFERENCE_CONTENT_RULES.get(level, _REFERENCE_CONTENT_RULES["rhythm"])
    return (
        "\n\nREFERENCE INSPIRATION\n"
        "You are given a reference rap the user admires: its transcript, BPM, and flow "
        "metrics. Produce a STANDALONE, original composition about the CutForge character.\n"
        f"{_REFERENCE_HARD_FLOOR}\n{rules}"
    )


def suggest_mood(character: str, anime: str = "") -> str:
    """Return a short mood/vibe descriptor line for a character (stateless)."""
    who = f"{character} from {anime}" if anime else character
    user_prompt = (
        f"Character / matchup: {who}\n\n"
        "Give the mood/vibe line for an original anime song about this. One line only."
    )
    text = anthropic_client.complete_text(SUGGEST_MOOD_SYSTEM_PROMPT, user_prompt)
    if not text:
        return ""
    return text.strip().strip('"').splitlines()[0]


def suggest_genres(project: VideoProject) -> GenreSuggestions:
    """Return 3 genre directions for the project's character/topic."""
    topic = project.topic or project.character
    user_prompt = (
        f"Character / matchup: {topic}\n\n"
        "Propose 3 genre/vibe directions for an original anime song about this. "
        "Return only valid JSON."
    )
    data = anthropic_client.complete_json(SUGGEST_SYSTEM_PROMPT, user_prompt)
    directions = [GenreDirection(**d) for d in data.get("directions", [])]
    return GenreSuggestions(character_read=data.get("character_read", ""), directions=directions)


def generate_package(project: VideoProject, genre: str, *, is_vs: bool = False,
                     reference_profile: dict | None = None,
                     content_blend: str = "rhythm") -> SongPackage:
    """Generate the full Suno package and write lyrics.txt + suno_prompt.json.

    If ``reference_profile`` is None, auto-loads a saved profile for this run (produced
    by the optional ``reference`` step). When present, the song is inspired by the
    reference rap; ``content_blend`` controls HOW MUCH of the reference's CONTENT
    (themes, imagery, hook shape, phrasing) may be borrowed — from "rhythm" (rhythm
    only, the strict default) up to "strong" (heavy thematic borrow, still non-verbatim).
    """
    # Auto-load a saved reference profile (lazy import avoids a circular dependency).
    if reference_profile is None:
        from cutforge.services import reference_service
        reference_profile = reference_service.load_reference_profile(project)

    topic = project.topic or project.character
    lines = [
        f"Character / matchup: {topic}",
        f"Chosen genre / style direction: {genre}",
    ]
    if project.character:
        lines.append(f"Character name(s): {project.character}")
    if project.anime:
        lines.append(f"Anime: {project.anime}")
    if is_vs:
        lines.append("This is a VS / matchup song — alternate perspectives and make the "
                     "chorus their clash.")
    lines.append("")
    lines.append(_lang_directive(project.language))

    system_prompt = GENERATE_SYSTEM_PROMPT
    if reference_profile:
        system_prompt = GENERATE_SYSTEM_PROMPT + _reference_addendum(content_blend)
        flow = reference_profile.get("flow", {})
        rhythm_only = content_blend == "rhythm"
        header = ("REFERENCE (style signal only — DO NOT COPY):" if rhythm_only
                  else f"REFERENCE (content inspiration, blend='{content_blend}' — never copy verbatim):")
        transcript_label = ("- Reference transcript (reference only — do not copy any line):"
                            if rhythm_only
                            else "- Reference transcript (content inspiration — rewrite, never copy verbatim):")
        lines.append("")
        lines.append(header)
        lines.append(f"- BPM: {reference_profile.get('bpm')}")
        lines.append(f"- Time signature: {reference_profile.get('time_signature')}/4")
        lines.append(f"- Onset density: {reference_profile.get('onset_rate_per_sec')} onsets/s")
        lines.append(f"- Flow: {flow.get('words_per_sec')} words/s, "
                     f"~{flow.get('syllables_per_beat')} syllables/beat")
        lines.append(transcript_label)
        lines.append(reference_profile.get("transcript", ""))

    lines.append("")
    lines.append("Write the complete Suno package. Return only valid JSON.")
    user_prompt = "\n".join(lines)

    data = anthropic_client.complete_json(system_prompt, user_prompt)
    package = SongPackage(
        title=data.get("title", ""),
        style=data.get("style", ""),
        lyrics=data.get("lyrics", ""),
        suno_tips=data.get("suno_tips", ""),
        topic=topic,
        character=project.character,
        anime=project.anime,
    )

    # Persist: lyrics.txt (paste into Suno) + suno_prompt.json (metadata for later steps)
    project.run_dir.mkdir(parents=True, exist_ok=True)
    project.lyrics_path.write_text(package.lyrics, encoding="utf-8")
    project.suno_prompt_path.write_text(
        json.dumps({
            "title": package.title,
            "style": package.style,
            "suno_tips": package.suno_tips,
            "topic": package.topic,
            "character": package.character,
            "anime": package.anime,
            "language": project.language,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Cache the title back onto the project for downstream steps.
    if package.title and not project.title:
        project.title = package.title
        project.save()

    return package
