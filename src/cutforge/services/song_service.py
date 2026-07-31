"""Song generation service — suggests genre directions and generates the full Suno package.

The system prompts are tuned to how Suno AI (v4.5/v5) actually reads its two fields:
- STYLE = the sonic world only (audio descriptors), never visual/video terms.
- LYRICS = words + bracketed structure/delivery tags that Suno obeys.

When a reference track is present it drives the SONIC WORLD (genre, BPM, instruments,
energy). The CHARACTER drives the LYRICAL IDENTITY (attitude, delivery, what is said).
These two axes are intentionally kept separate so the song sounds like the character
rapping over the reference's lane — not a generic orchestral-trap template.
"""
from __future__ import annotations

import json

from cutforge.integrations import anthropic_client
from cutforge.models.project import VideoProject
from cutforge.models.song import GenreDirection, GenreSuggestions, SongPackage


# ---------------------------------------------------------------------------
# Shared Suno craft rules — embedded in every prompt so style and lyrics stay
# in sync. Two separate sections: one for the style field, one for lyrics.
# ---------------------------------------------------------------------------

_SUNO_STYLE_RULES = """\
HOW TO WRITE THE SUNO "STYLE" STRING
Suno generates AUDIO, not video. The style field describes ONLY what you HEAR.

Formula — order matters, Suno weights the leading tokens most:
  <dominant subgenre>, <mood/energy>, <vocal: gender + delivery>, <2–4 signature
  instruments with qualities>, <production word>, <one BPM number> BPM

Rules:
- Lead with a SPECIFIC subgenre ("dark trap", "UK drill", "boom bap", "melodic trap",
  "orchestral trap") — never bare "hip-hop" or "rap".
- 8–14 comma-separated descriptors, ~120–200 characters.
- Exactly ONE BPM number (e.g. "140 BPM"), never a range.
- Vocal: character + delivery: "aggressive rapped male vocal", "anthemic sung hook",
  "confident rap flow", "deadpan delivery". For rap add "rap vocals" or "spoken flow".
- Instruments: 2–4 named + described: "distorted 808 bass", "dark minor piano",
  "sliding drill hi-hats", "crisp trap hats", "warm guitar sample".

BANNED IN STYLE — these are VISUAL / meta terms Suno cannot hear:
  AMV, anime edit, anime rap, "cinematic anime rap", anime AMV, music video,
  montage, edit, clip, scene, 4K, visuals, "epic anime", any character/anime name,
  "make it hard", "best song", narrative content about the plot.
Translate that intent into SOUND. "Cinematic anime rap" → "orchestral trap, epic
choir, soaring strings, booming 808s". The "anime" feel comes from the sonic palette,
not from the word "anime". No contradictory genres ("trap, lofi chill").

FORCE RAP (Suno defaults to melodic singing):
Put "aggressive rap vocals" / "spoken flow" / "confident rap flow" in style, AND
[Rap]/[Rapped] inline in the lyrics. Only use "autotune"/"melodic"/"sung" when you
genuinely want sung sections.
"""

_SUNO_LYRICS_RULES = """\
LYRICS FIELD — STRUCTURE AND DELIVERY TAGS
Suno obeys bracketed tags in the Lyrics field (structure tags most reliably).
- [Rap] / [Rapped] on each verse → forces rapping, not singing.
- [Male Vocal] / [Female Vocal] → pins the voice gender.
- [Gang Vocals] / [Chant] on the hook → anthemic group shout.
- [Fast Rap] / [Double Time] / [Slow Flow] → cadence control.
- Ad-libs in (parentheses) at line ends: (yeah) (uh) (gang).
- Anything NOT bracketed or in parentheses WILL be sung as lyrics.
- Section-length caps (over-long → Suno rushes delivery): verses 4–8 lines,
  choruses 4–6, bridges 2–4. Put the strongest line FIRST in each section.
"""

# ---------------------------------------------------------------------------
# Archetype range — used only when there is NO reference. When a reference
# is present this block is replaced by the reference's own sonic read.
# ---------------------------------------------------------------------------

_ARCHETYPE_RANGE_NO_REF = """\
GENRE SELECTION (no reference — pick the archetype that fits the character):

1. Dark villain        → dark trap / drill / phonk      | 130–150 BPM
   distorted 808, sliding hi-hats, dark violin stabs, minor piano
   | deep menacing rapped male vocal, reverb ad-libs           (lane: M4RKIM)

2. Unstoppable hero    → orchestral / epic hybrid trap   | 140–160 BPM
   booming 808s, crisp trap hats, soaring strings, epic choir  (lane: Rustage)

3. Hot-blooded shonen  → rap rock / trap metal           | 150–170 BPM
   distorted guitars, double-kick, heavy 808              (lane: 7 Minutoz)

4. Tragic / emotional  → melodic trap / emo rap          | 130–150 BPM
   sad piano, clean guitar, warm pads, soft 808            (lane: Divide Music)

5. Godlike / ancient   → cinematic hybrid orchestral     | 90 BPM or 150 half-time
   war drums/taiko, full orchestra, epic choir, braams

Scene sweet spot: ~140–160 BPM. What makes these songs HIT: character-POV writing
with real lore; the biggest drop timed to the character's signature moment; an
anthemic, sing-along hook.
"""

# ---------------------------------------------------------------------------
# Reference-based addenda
# ---------------------------------------------------------------------------

_REFERENCE_HARD_FLOOR = """\
HARD FLOOR — never violate:
- NEVER reproduce any line, hook, or distinctive phrase from the reference transcript
  verbatim, and never merely swap a word or two — that is still copying.
- NEVER lift the reference's rhyme scheme, metaphor, or hook phrasing as-is — rewrite
  completely so it reads as a new composition.
- The transcript is auto-transcribed and may contain mis-heard words.
Match the reference's RHYTHM:
- Target roughly the given BPM; put ONE number in the style string.
- Flow: line length and syllables-per-bar. Faster / denser reference → shorter,
  denser lines; slower → more spacious lines.

LORE MINING (always do this, regardless of content_blend level):
The reference transcript was written about the same character — it is a goldmine of
character-specific lore. Read it carefully and extract:
  - Power names, techniques, abilities mentioned
  - Iconic story moments or turning points referenced
  - Personality traits expressed through the lyrics
  - Thematic threads (sacrifice, loneliness, burning will, etc.)
  - Any numbers, titles, or symbolic imagery tied to the character
Then USE these lore facts as raw material for the new song's imagery and references.
The facts belong to the CHARACTER, not to the reference composer — rewrite them in
completely fresh language. This is the primary way the reference enriches the lyrics.
"""

_REFERENCE_CONTENT_RULES = {
    "rhythm": """\
CONTENT: Use the reference ONLY as a rhythm/energy signal. Do NOT borrow its themes,
imagery, message, or hook structure — write the character piece entirely from scratch.
The transcript is vibe-only; treat its words as off-limits.""",
    "light": """\
CONTENT (light borrow): Echo 1–2 of the reference's core themes and its overall tone,
expressed in completely fresh words. Do NOT copy its specific images or metaphors.""",
    "moderate": """\
CONTENT (moderate borrow): Reuse the reference's core themes, its semantic field of
imagery/metaphors, and its hook shape — all rewritten around the character with fresh
wording. No verbatim phrases.""",
    "strong": """\
CONTENT (strong borrow): Take the reference's themes, imagery, message and hook
structure, follow its section arc, and lift its most striking phrases as inspiration —
but heavily rewrite each one (swap vocabulary, referents, angle). "Unmistakably
inspired by, never a copy." The hard floor still holds.""",
}


def _reference_addendum(level: str) -> str:
    rules = _REFERENCE_CONTENT_RULES.get(level, _REFERENCE_CONTENT_RULES["rhythm"])
    return (
        "\n\nREFERENCE INSPIRATION\n"
        "You have a reference rap the user admires. Use it to set the SONIC WORLD of "
        "the new song — genre, tempo, beat energy, vocal delivery style. The CHARACTER "
        "shapes only the LYRICAL IDENTITY: what the song says, the attitude, the "
        "imagery, the flow personality. Do NOT impose orchestral/strings/choir unless "
        "the reference actually has those sounds. If the reference is a trap/drill beat, "
        "keep it trap/drill. If it is boom-bap, keep it boom-bap.\n"
        f"{_REFERENCE_HARD_FLOOR}\n{rules}"
    )


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SUGGEST_SYSTEM_PROMPT_BASE = f"""\
You are a producer for an anime-rap channel in the lane of BASARA, M4RKIM, ANIRAP,
Rustage and 7 Minutoz. Given a character (or matchup), propose distinct GENRE/VIBE
directions for an original song.

{_SUNO_STYLE_RULES}

OUTPUT FORMAT — a single valid JSON object, no markdown fences, no commentary:
{{
  "character_read": "1–2 sentences on the character's vibe that should drive the music",
  "directions": [
    {{
      "label": "Short genre name (e.g. 'Cold Drill')",
      "style": "A Suno-ready style string following the formula above",
      "why": "One sentence: why this fits the character"
    }}
  ]
}}

Rules:
- Exactly 3 directions, each clearly different from the others.
- Each style must follow the formula and BANNED-terms rule above.
- All descriptors in English.
"""

SUGGEST_SYSTEM_PROMPT_NO_REF = SUGGEST_SYSTEM_PROMPT_BASE + f"""
{_ARCHETYPE_RANGE_NO_REF}
"""

SUGGEST_SYSTEM_PROMPT_WITH_REF = SUGGEST_SYSTEM_PROMPT_BASE + """
GENRE FOLLOWS THE REFERENCE — when a reference is provided the 3 directions must be
anchored to the reference's lane and BPM. They are VARIATIONS within that lane (e.g.
harder vs. more melodic, sparser vs. more orchestral), not three unrelated genres.
Use the reference's transcript to infer: subgenre, energy, beat character, vocal style.
Do NOT add orchestral strings / choir unless the reference itself has them.
Every direction must still genuinely fit the character.
"""

GENERATE_SYSTEM_PROMPT_BASE = f"""\
You are an expert songwriter and music producer for an anime-rap channel in the lane
of BASARA, M4RKIM, ANIRAP, Rustage and 7 Minutoz. You write an original song about a
specific anime character (or matchup) that will be generated on Suno AI.

You know exactly how Suno reads its Style and Lyrics fields and you exploit that to
get a clean, on-genre track — not generic mush.

TWO AXES — keep them separate:
  SONIC WORLD  → driven by the REFERENCE (genre, BPM, instruments, energy, beat feel)
  LYRICAL IDENTITY → driven by the CHARACTER (attitude, powers, arc, delivery persona)

The goal is to imagine that CHARACTER rapping in the REFERENCE'S lane.
Does the song sound like this character performing in this style? That is the test.

{_SUNO_STYLE_RULES}

{_SUNO_LYRICS_RULES}

THE "EXCLUDE" FIELD
Return a short comma-separated Exclude Styles list to stop Suno from drifting.
For a hard rap track: "singing, melodic vocals, auto-tune, sung chorus".
For a melodic track: "harsh, distorted, aggressive, screaming".
Keep it short (3–6 terms).

LYRICS STRUCTURE
Write AT LEAST ~2.5 minutes of material. A reliable arrangement:
  [Intro] → [Verse 1] → [Chorus] → [Verse 2] → [Chorus] → [Bridge] → [Verse 3] → [Outro]
This is a STARTING POINT — adapt to the genre/reference's own arrangement:
  - Drop-driven beat → [Build] then [Drop] timed to the character's signature moment.
  - Emotional / melodic → fewer, sung-heavy hooks; maybe [Instrumental Break]; drop the
    gang-vocal chant.
  - Cypher / anthem → verse-per-character feel, [Gang Vocals] + [Chant] on the hook.
  - Boom-bap / rap rock → posse-trade verses, sing-along chant hook.
Section caps: verses 4–8 lines, choruses 4–6, bridges 2–4. Strongest line FIRST.
For a vs / matchup: alternate perspectives, make the chorus the clash.

TONE & CONTENT
- The song is FROM or ABOUT the character — capture their personality, powers and arc
  through vivid, specific imagery. Not a plot summary — a hype anthem / character piece.
- Reference their actual abilities, iconic moments and traits through metaphor and
  attitude, not exposition.
- Do NOT use trademarked catchphrases verbatim; evoke them instead.
- Keep it hype and quotable — the hook should be something viewers scream in the comments.

LINE RULES (lyrics also become on-screen karaoke captions)
- Each line: roughly 4–9 words — natural caption length.
- One idea per line. Clean, punchy.
- No "yeah yeah" filler beyond intentional (parenthetical) ad-libs.

OUTPUT FORMAT — a single valid JSON object, no markdown fences, no commentary:
{{
  "title": "Song title — short and evocative (e.g. 'Shadow Sovereign')",
  "style": "The Suno STYLE string — audio only, following the formula and banned-terms rule",
  "exclude": "Short comma-separated Suno Exclude-Styles list",
  "lyrics": "Full structured lyrics as one string. \\n for line breaks; each [tag] on its own line.",
  "suno_tips": "One short line of advice for the Suno generation"
}}
"""

GENERATE_SYSTEM_PROMPT_NO_REF = GENERATE_SYSTEM_PROMPT_BASE + f"""
{_ARCHETYPE_RANGE_NO_REF}
"""

SUGGEST_MOOD_SYSTEM_PROMPT = """\
You are a creative director for an anime-rap channel (BASARA / M4RKIM / Rustage lane).
Given an anime character (or matchup), describe the MOOD / VIBE that an original song
about them should have.

Return a SINGLE short line: 3–6 comma-separated descriptors capturing the character's
energy (e.g. "dark, cold, ominous power, shadow army" or "hot-blooded, explosive,
heroic, hype"). No commentary, no quotes, no markdown — just the descriptors line.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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


def suggest_genres(project: VideoProject, *, content_blend: str | None = None) -> GenreSuggestions:
    """Return 3 genre directions for the project's character/topic.

    When a reference rap is present the genre always follows the reference's lane.
    ``content_blend`` is accepted for signature compatibility but only governs lyrical
    content at generation time — it no longer gates genre anchoring.
    """
    from cutforge.services import reference_service

    topic = project.topic or project.character
    profile = reference_service.load_reference_profile(project)

    lines = [f"Character / matchup: {topic}", ""]

    if profile:
        system = SUGGEST_SYSTEM_PROMPT_WITH_REF
        lines += [
            "REFERENCE (derive the genre/lane from this — stay in its lane):",
            f"- BPM: {profile.get('bpm')} — use this one number in every style string",
            f"- Onset density: {profile.get('onset_rate_per_sec')} onsets/s",
            f"- Title: {profile.get('source_title')}",
            f"- Transcript (infer subgenre, energy, beat character — do NOT copy words): "
            f"{profile.get('transcript', '')[:1200]}",
            "",
            "Propose 3 directions. All 3 must stay in the reference's lane and use its BPM. "
            "Vary within the lane (e.g. harder vs. more melodic). Every direction must also "
            "fit the character's personality.",
        ]
    else:
        system = SUGGEST_SYSTEM_PROMPT_NO_REF
        lines.append(
            "Propose 3 distinct genre/vibe directions for an original anime song about this."
        )

    lines.append("Return only valid JSON.")
    user_prompt = "\n".join(lines)

    data = anthropic_client.complete_json(system, user_prompt)
    directions = [GenreDirection(**d) for d in data.get("directions", [])]
    return GenreSuggestions(character_read=data.get("character_read", ""), directions=directions)


def generate_package(project: VideoProject, genre: str, *, is_vs: bool = False,
                     reference_profile: dict | None = None,
                     content_blend: str = "rhythm") -> SongPackage:
    """Generate the full Suno package and write lyrics.txt + suno_prompt.json.

    The reference (when present) drives the SONIC WORLD; the character drives the
    LYRICAL IDENTITY. ``content_blend`` controls how much of the reference's lyrical
    content may be borrowed: rhythm (none) → light → moderate → strong.
    """
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
        lines.append(
            "This is a VS / matchup song — alternate perspectives and make the chorus the clash."
        )

    if reference_profile:
        system_prompt = GENERATE_SYSTEM_PROMPT_BASE + _reference_addendum(content_blend)
        flow = reference_profile.get("flow", {})
        rhythm_only = content_blend == "rhythm"
        header = (
            "REFERENCE (sonic world — derive style from this; do NOT copy any words):"
            if rhythm_only else
            f"REFERENCE (sonic world + content inspiration, blend='{content_blend}' — never copy verbatim):"
        )
        transcript_note = (
            "- Transcript (derive genre/energy/beat feel; words are off-limits):"
            if rhythm_only else
            "- Transcript (content inspiration — rewrite thoroughly, never copy):"
        )
        lines += [
            "",
            header,
            f"- BPM: {reference_profile.get('bpm')} (put this ONE number in the style string)",
            f"- Time signature: {reference_profile.get('time_signature')}/4",
            f"- Onset density: {reference_profile.get('onset_rate_per_sec')} onsets/s",
            f"- Flow: {flow.get('words_per_sec')} words/s, ~{flow.get('syllables_per_beat')} syllables/beat",
            transcript_note,
            reference_profile.get("transcript", ""),
            "",
            "Build the style string from what you HEAR in this reference's lane. "
            "Do NOT add orchestral strings, choir, or cinematic elements unless they are "
            "present in the reference. Stay in the reference's genre lane.",
        ]
    else:
        system_prompt = GENERATE_SYSTEM_PROMPT_NO_REF

    lines += ["", "Write the complete Suno package. Return only valid JSON."]
    user_prompt = "\n".join(lines)

    data = anthropic_client.complete_json(system_prompt, user_prompt)
    package = SongPackage(
        title=data.get("title", ""),
        style=data.get("style", ""),
        exclude=data.get("exclude", ""),
        lyrics=data.get("lyrics", ""),
        suno_tips=data.get("suno_tips", ""),
        topic=topic,
        character=project.character,
        anime=project.anime,
    )

    project.run_dir.mkdir(parents=True, exist_ok=True)
    project.lyrics_path.write_text(package.lyrics, encoding="utf-8")
    project.suno_prompt_path.write_text(
        json.dumps({
            "title": package.title,
            "style": package.style,
            "exclude": package.exclude,
            "suno_tips": package.suno_tips,
            "topic": package.topic,
            "character": package.character,
            "anime": package.anime,
            "language": project.language,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if package.title and not project.title:
        project.title = package.title
        project.save()

    return package
