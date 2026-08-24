"""Song generation service — original composition from character lore + reference DNA.

The system prompts are tuned to how Suno AI (v4.5/v5) actually reads its two fields:
- STYLE = the sonic world only (audio descriptors), never visual/video terms.
- LYRICS = words + bracketed structure/delivery tags that Suno obeys.

CORE PRINCIPLE — the reference is split into two independent signals:
- reference MUSIC profile (BPM/flow/onset — the SONIC DNA) → drives the style only.
- reference LORE profile (facts/abilities/easter eggs mined from the transcript) →
  feeds the writer as KNOWLEDGE about the character.

The reference's lyrical EXPRESSION is never reused. Every song is written from scratch
as an ORIGINAL composition: new concept, new hook, new metaphors, new rhymes. A
CreativeDirection brief is planned first so the writer commits to a new angle instead
of drifting back toward the reference.
"""
from __future__ import annotations

import json

from cutforge.integrations import anthropic_client
from cutforge.models.lore import ReferenceLoreProfile
from cutforge.models.project import VideoProject
from cutforge.models.song import (
    CreativeDirection,
    GenreDirection,
    GenreSuggestions,
    SongPackage,
)
from cutforge.models.story import StoryContentProfile
from cutforge.models.structure import NarrativeStructureProfile


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

N_GENRE_SUGGESTIONS = 5

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
# Reference-based addenda — SONIC DNA only. The reference's lyrical content is
# handled entirely through the mined lore profile, never through borrowing.
# ---------------------------------------------------------------------------

_REFERENCE_SONIC_RULES = """\
REFERENCE = SONIC DNA ONLY
A reference track is provided. It contributes EXACTLY ONE thing to this song: the sonic
world — genre/subgenre, BPM, energy, beat character, instrument palette, vocal delivery
and flow density. Nothing else.
- Target roughly the given BPM; put ONE number in the style string.
- Flow: match line length and syllables-per-bar to the reference. Faster/denser
  reference → shorter, denser lines; slower → more spacious lines.
- Do NOT add orchestral strings, choir or cinematic elements unless the reference
  itself has them. Trap stays trap; drill stays drill; boom-bap stays boom-bap.
- The reference's WORDS, themes, hooks and metaphors are OFF-LIMITS for the style. Any
  character knowledge from the reference reaches you only through the LORE section
  below — never by reading the reference's lyrics.
"""

_ORIGINALITY_RULES = """\
ORIGINALITY — NON-NEGOTIABLE
You are writing a COMPLETELY ORIGINAL song. It is not a translation, adaptation,
cover or rewrite of the reference. If a reference exists, you never saw its lyrics —
you only received its sonic DNA and a list of extracted character facts.

DO NOT:
- translate the reference;
- paraphrase the reference;
- mirror the reference line-by-line or follow its lyrical sequence;
- reproduce its distinctive phrases, punchlines or hook;
- reuse its metaphors or imagery;
- intentionally reproduce its rhyme scheme;
- preserve its lyrical structure.

DO:
- use the extracted lore purely as factual/creative source material;
- invent a NEW narrative angle, hook, metaphors, imagery and rhyme choices;
- follow the CREATIVE DIRECTION brief below;
- turn specific lore/easter eggs into fresh lines, metaphors and punchlines of your own.

The test: the final song must read as a brand-new composition that merely happens to be
about the same character in a similar sonic lane — never as a version of the reference.
"""


_STRUCTURE_FOLLOW_RULES = """\
FOLLOW THE PROVEN STRUCTURE — WITH 100% ORIGINAL EXPRESSION
This song deliberately FOLLOWS a proven narrative skeleton reverse-engineered from
reference songs about this character (the NARRATIVE STRUCTURE block below). Many
successful channels reuse the same formula — same story beats, same order, hook in the
same place, same emotional arc — changing only the words. You are doing exactly that.

DO:
- follow the skeleton's beat ORDER, section arrangement, hook placement and emotional arc;
- honour the intensity/cadence shape (where the energy and density rise and fall);
- fill each beat with the KIND of lore its slot calls for, using the mined lore below;
- write EVERY line, rhyme, metaphor and the hook WORDING from scratch — 100% original.

DO NOT:
- copy, paraphrase or translate any words, phrases, punchlines or hook wording from the
  reference — you never saw its lyrics, only its SHAPE;
- reuse the reference's specific metaphors or imagery;
- intentionally reproduce its rhyme scheme.

The test: same skeleton, brand-new words. A listener who knows the references should
recognise the STRUCTURE but hear nothing lifted — every line is your own.
"""


_REWRITE_RULES = """\
REWRITE THE STORY — SAME STORY, NEW EXPRESSION
This song deliberately RE-TELLS a story reverse-engineered from reference songs about
this character (the STORY TO RETELL block below). The user sent 1–3 raps that tell the
SAME story with different rhymes and rhythms, and wants a NEW rap that keeps that story
and its ideas but is written fresh. You are doing exactly that: same story, new words.

DO:
- preserve the logline, the ORDER of the story points, and the key ideas and imagery;
- keep the same narrative meaning at each point — say the SAME thing, differently;
- re-write EVERY line with NEW rhymes, a different rhythm/cadence and your own phrasing;
- honour the hook instruction below (keep the reference's hook CONCEPT, or invent a new one).

DO NOT:
- copy, quote, translate or lightly paraphrase any line, rhyme, punchline or hook WORDING
  from the reference — you never saw its lyrics, only a neutral description of its story;
- mirror the reference line-by-line or preserve its rhyme scheme;
- change WHAT the story says — only HOW it is said.

The test: a listener who knows the references should recognise the SAME story and ideas,
but hear NOTHING lifted — every line, rhyme and cadence is your own re-writing.
"""



# ---------------------------------------------------------------------------
# Lore + creative-direction prompt formatting
# ---------------------------------------------------------------------------

def _format_lore_for_prompt(lore: ReferenceLoreProfile) -> str:
    """Render a mined lore profile as a compact, writer-facing knowledge block."""
    parts: list[str] = [
        "CHARACTER LORE (mined from the reference — this is KNOWLEDGE, not lyrics to "
        "reuse). Use these as raw material for NEW lines, metaphors and punchlines. "
        "Never quote or paraphrase how the reference expressed them.",
    ]
    if lore.character:
        parts.append(f"- Character: {lore.character}")
    if lore.facts:
        parts.append("- Facts:")
        parts += [f"    • [{f.category}/{f.confidence}] {f.fact}" for f in lore.facts]
    if lore.abilities:
        parts.append("- Abilities / techniques:")
        parts += [
            f"    • {a.name}: {a.description}".rstrip(": ").rstrip() for a in lore.abilities
        ]
    if lore.events:
        parts.append("- Story events / turning points:")
        parts += [f"    • [{e.importance}] {e.event}" for e in lore.events]
    if lore.relationships:
        parts.append("- Relationships:")
        parts += [
            f"    • {' & '.join(r.characters)}: {r.relationship}".strip()
            for r in lore.relationships
        ]
    if lore.easter_eggs:
        parts.append("- Easter eggs (specific anime/manga references — gold for punchlines):")
        for egg in lore.easter_eggs:
            meaning = f" → {egg.interpreted_meaning}" if egg.interpreted_meaning else ""
            related = f" (related: {egg.related_lore})" if egg.related_lore else ""
            parts.append(f"    • {egg.reference}{meaning}{related}")
    if lore.themes:
        parts.append(f"- Themes: {', '.join(lore.themes)}")
    if lore.personality_traits:
        parts.append(f"- Personality traits: {', '.join(lore.personality_traits)}")
    if lore.author_interpretations:
        parts.append(
            "- The reference's THEMATIC UNIVERSE — the artistic lens it uses to see this "
            "character. Use it ONLY to understand the character's themes; do NOT copy its "
            "metaphor domain. Your own metaphors must still be sourced from the character's "
            "in-world reality (powers, techniques, weapons, the anime's own imagery), NEVER "
            "from abstract real-world domains (legal/estate/will, finance/debt/ledger/rent/"
            "lease, corporate, sports, cooking, gambling): "
            + "; ".join(lore.author_interpretations)
        )
    if lore.uncertain_items:
        parts.append(
            "- Uncertain / possibly mis-transcribed (verify against what you know before "
            "relying on these): " + "; ".join(lore.uncertain_items)
        )
    return "\n".join(parts)


def _format_direction_for_prompt(direction: CreativeDirection) -> str:
    """Render the creative-direction brief as a writer-facing block."""
    parts = ["CREATIVE DIRECTION (the NEW song to write — follow this brief):"]
    if direction.core_theme:
        parts.append(f"- Core theme: {direction.core_theme}")
    if direction.narrative_angle:
        parts.append(f"- Narrative angle: {direction.narrative_angle}")
    if direction.emotional_arc:
        parts.append(f"- Emotional arc: {direction.emotional_arc}")
    if direction.hook_concept:
        parts.append(f"- Hook concept: {direction.hook_concept}")
    if direction.key_lore_points:
        parts.append("- Key lore points to weave in: " + "; ".join(direction.key_lore_points))
    if direction.original_metaphor_direction:
        parts.append(f"- Original metaphor direction: {direction.original_metaphor_direction}")
    if direction.delivery_personality:
        parts.append(f"- Delivery personality: {direction.delivery_personality}")
    if direction.things_to_avoid:
        parts.append("- Things to avoid: " + "; ".join(direction.things_to_avoid))
    return "\n".join(parts)


def _format_structure_for_prompt(structure: NarrativeStructureProfile) -> str:
    """Render the narrative skeleton as a writer-facing block — SHAPE only, no phrasing."""
    parts = [
        "NARRATIVE STRUCTURE (the PROVEN skeleton to FOLLOW — this is SHAPE, not lyrics. "
        "Follow the beat order/arrangement/arc/hook placement, but invent ALL wording).",
    ]
    if structure.overall_shape:
        parts.append(f"- Overall shape: {structure.overall_shape}")
    if structure.section_arrangement:
        parts.append("- Section arrangement: " + " → ".join(structure.section_arrangement))
    if structure.beats:
        parts.append("- Story beats (follow this order; fill each with your own words):")
        for b in sorted(structure.beats, key=lambda x: x.order):
            slot = f" ← fill with: {b.maps_to_lore}" if b.maps_to_lore else ""
            parts.append(
                f"    {b.order}. [{b.section} · {b.function} · {b.intensity}] {b.beat}{slot}"
            )
    if structure.hook_placement:
        parts.append(f"- Hook placement: {structure.hook_placement}")
    if structure.emotional_arc:
        parts.append(f"- Emotional arc: {structure.emotional_arc}")
    if structure.flow_cadence_notes:
        parts.append(f"- Flow / cadence: {structure.flow_cadence_notes}")
    if structure.shared_pattern_notes:
        parts.append(f"- Proven-formula notes: {structure.shared_pattern_notes}")
    return "\n".join(parts)


def _format_story_for_prompt(story: StoryContentProfile, *, new_hook: bool) -> str:
    """Render the shared story as a writer-facing block — CONTENT to preserve, no phrasing.

    ``new_hook`` controls the hook instruction: ``False`` keeps the reference's hook
    CONCEPT (rewording only), ``True`` asks for a brand-new hook.
    """
    parts = [
        "STORY TO RETELL (the SAME story to preserve — this is CONTENT, not lyrics. Keep "
        "WHAT is said and its order; re-write HOW it is said with new rhymes and rhythm).",
    ]
    if story.logline:
        parts.append(f"- Logline (the whole story to keep): {story.logline}")
    if story.key_ideas:
        parts.append("- Key ideas to preserve: " + "; ".join(story.key_ideas))
    if story.themes:
        parts.append(f"- Themes: {', '.join(story.themes)}")
    if story.story_points:
        parts.append("- Story points (keep this order and meaning; re-write the wording):")
        for p in sorted(story.story_points, key=lambda x: x.order):
            imgs = f" — keep imagery: {', '.join(p.key_images)}" if p.key_images else ""
            parts.append(f"    {p.order}. [{p.section} · {p.function}] {p.point}{imgs}")
    if story.emotional_arc:
        parts.append(f"- Emotional arc: {story.emotional_arc}")
    if story.shared_pattern_notes:
        parts.append(f"- Shared-story notes: {story.shared_pattern_notes}")
    if new_hook:
        parts.append(
            "- HOOK: invent a BRAND-NEW hook/refrain concept and wording. Do NOT reuse the "
            "reference's hook idea."
        )
    elif story.hook_concept:
        parts.append(
            f"- HOOK: keep the reference's hook CONCEPT ({story.hook_concept}) but re-write "
            "its wording — same idea, new words and rhyme."
        )
    else:
        parts.append(
            "- HOOK: keep the reference's hook concept, but re-write its wording — same "
            "idea, new words and rhyme."
        )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SUGGEST_SYSTEM_PROMPT_BASE = f"""\
You are a producer for an anime-rap channel in the lane of BASARA, MHRAP, Tauz M4RKIM, ANIRAP and 7 Minutoz. Given a character (or matchup), propose distinct GENRE/VIBE
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
- Exactly {N_GENRE_SUGGESTIONS} directions, each clearly different from the others.
- Each style must follow the formula and BANNED-terms rule above.
- All descriptors in English.
"""

SUGGEST_SYSTEM_PROMPT_NO_REF = SUGGEST_SYSTEM_PROMPT_BASE + f"""
{_ARCHETYPE_RANGE_NO_REF}
"""

SUGGEST_SYSTEM_PROMPT_WITH_REF = SUGGEST_SYSTEM_PROMPT_BASE + f"""
GENRE FOLLOWS THE REFERENCE — when a reference is provided the {N_GENRE_SUGGESTIONS} directions must be
anchored to the reference's lane and BPM. They are VARIATIONS within that lane (e.g.
harder vs. more melodic, sparser vs. more orchestral), not three unrelated genres.
Infer subgenre, energy, beat character and vocal style from the reference's SONIC DNA
(BPM / onset density). Do NOT add orchestral strings / choir unless the reference itself
has them. Every direction must still genuinely fit the character.
"""

# Creative-direction planner — the brief-writer that answers "what is the NEW song?".
CREATIVE_DIRECTION_SYSTEM_PROMPT = """\
You are a creative director for an anime-rap channel (BASARA / M4RKIM / Rustage lane).
Your job is to design the brief for a COMPLETELY ORIGINAL song about a specific anime
character (or matchup), BEFORE any lyrics are written.

You answer ONE question: "What is the NEW song we are trying to make?"
You NEVER answer "How do we rewrite the reference?".

Inputs you may receive:
- the character / anime and the user's topic;
- a reference MUSIC profile (BPM/energy) — sonic direction only;
- mined CHARACTER LORE (facts, abilities, events, easter eggs) — knowledge to draw from.

Design an original direction: pick a fresh core theme and narrative angle, an emotional
arc, an original hook concept, the specific lore points worth weaving in, a direction for
NEW metaphors (not the reference's), and the delivery personality. In "things_to_avoid",
explicitly include reusing the reference's expression (its hooks, metaphors, phrasing,
structure) and any clichés that would make the song generic.

Prefer specific, character-grounded ideas over generic hype. If strong easter eggs /
obscure lore exist, prioritise them — they make the song feel authentic and specific.

NON-NEGOTIABLE — METAPHOR SOURCING: The metaphor world must be built from the character's
OWN universe — their powers, techniques, forms, weapons, gear, creatures, locations and the
source material's own imagery. You may NOT translate the character's story into an unrelated
real-world domain. Specifically BANNED as metaphor systems (unless the character literally
lives in that world): legal/courtroom (wills, estates, deeds, inheritance law, verdicts,
contracts), finance/accounting (debt, ledgers, rent, leases, invoices, interest),
corporate/business, sports, cooking, and gambling. Abstract themes like 'inheritance',
'legacy', 'debt of hatred' or 'destiny' are allowed as IDEAS but must be voiced through
concrete in-world imagery — blood, the awakened eye, the clan's fire, the scratched
headband — never through paperwork, money, property or contracts.

OUTPUT FORMAT — a single valid JSON object, no markdown fences, no commentary:
{
  "core_theme": "the central idea of the NEW song, in one line",
  "narrative_angle": "the fresh POV/approach (e.g. 'first-person vow at his lowest moment')",
  "emotional_arc": "how the feeling evolves across the song",
  "hook_concept": "an ORIGINAL hook idea — the concept, not finished lyrics",
  "key_lore_points": ["specific facts/abilities/easter eggs to weave in", "..."],
  "original_metaphor_direction": "A NEW angle within the reference's thematic universe, but the IMAGERY must be sourced ONLY from the character's actual in-world reality — their canonical powers, techniques, weapons, transformations, creatures, locations and the anime's own objects and visuals. Every governing image MUST literally exist in the character's world. HARD BAN: never build the metaphor system out of abstract real-world domains the character does not inhabit — no legal/court/estate/will/inheritance-law, no finance/debt/ledger/rent/lease/accounting, no corporate/business, no sports, no cooking, no gambling — unless the character literally comes from that world. Render even abstract themes (legacy, inheritance, destiny) through concrete in-world objects, not through paperwork, money or contracts. One concrete in-world metaphor system in 1-2 sentences.",
  "delivery_personality": "the rap persona/attitude and cadence feel",
  "things_to_avoid": ["reference's hook/metaphors/phrasing", "generic clichés", "..."]
}
Return ONLY the JSON object.
"""

# Rewrite-the-story planner — the EXPRESSION-only brief-writer. Unlike the default
# planner (which forbids preserving the reference's story), this one treats the story as
# FIXED and designs only how to re-express it: new metaphors, rhyme/cadence approach,
# delivery, and the hook decision.
CREATIVE_DIRECTION_REWRITE_SYSTEM_PROMPT = """\
You are a creative director for an anime-rap channel (BASARA / M4RKIM / Rustage lane).
The channel has 1–3 reference raps that tell the SAME story about a character, and wants
a NEW rap that KEEPS that story and its ideas but is re-written with different rhymes, a
different rhythm and (optionally) a different hook.

You answer ONE question: "How do we RE-EXPRESS this fixed story?".
The STORY, its points, its ideas and its emotional arc are GIVEN and FIXED (in the STORY
block below) — you do NOT invent a new story or change what it says. You design only the
EXPRESSION layer used to re-write it.

Inputs you may receive:
- the character / anime and the user's topic;
- a reference MUSIC profile (BPM/energy) — sonic direction only;
- the SHARED STORY to retell (logline, story points, ideas, arc, hook concept);
- mined CHARACTER LORE — extra knowledge to draw richer wording from.

Design the re-expression: keep the given core theme and narrative (do not replace them),
then choose a FRESH metaphor world, a rhyme/cadence approach that differs from the
references, the delivery personality, and the hook decision (respect whether a new hook
was requested). In "things_to_avoid", explicitly include lifting or lightly paraphrasing
the reference's wording, rhymes or hook, and any clichés.

NON-NEGOTIABLE — METAPHOR SOURCING: The metaphor world must be built from the character's
OWN universe — their powers, techniques, forms, weapons, gear, creatures, locations and the
source material's own imagery. You may NOT translate the character's story into an unrelated
real-world domain. Specifically BANNED as metaphor systems (unless the character literally
lives in that world): legal/courtroom (wills, estates, deeds, inheritance law, verdicts,
contracts), finance/accounting (debt, ledgers, rent, leases, invoices, interest),
corporate/business, sports, cooking, and gambling. Abstract themes like 'inheritance',
'legacy', 'debt of hatred' or 'destiny' are allowed as IDEAS but must be voiced through
concrete in-world imagery — blood, the awakened eye, the clan's fire, the scratched
headband — never through paperwork, money, property or contracts.

OUTPUT FORMAT — a single valid JSON object, no markdown fences, no commentary:
{
  "core_theme": "the story's central idea (kept from the STORY block, in one line)",
  "narrative_angle": "the story's POV/approach (kept — how it is TOLD, re-expressed)",
  "emotional_arc": "the arc from the STORY block (kept; may be phrased in your words)",
  "hook_concept": "the hook idea to use — kept-but-reworded, or brand-new if requested",
  "key_lore_points": ["specific facts/ideas to weave in while retelling", "..."],
  "original_metaphor_direction": "A concrete metaphor SYSTEM sourced ONLY from the character's actual in-world reality — their canonical powers, techniques, transformations, weapons, creatures, locations, objects, and the anime's own visual imagery (for a Naruto shinobi: the Sharingan, Amaterasu, chakra, jutsu, kunai/shuriken, the clan crest, the Hidden Leaf). Every governing image MUST be something that literally exists in this character's world. HARD BAN: do NOT build the metaphor system out of abstract real-world domains the character does not literally inhabit — no legal/court/estate/will/inheritance-law imagery, no finance/debt/ledger/rent/lease/accounting imagery, no corporate/business, no sports, no cooking, no gambling. A thematically correct idea such as 'inheritance of hatred' must be expressed through in-world objects (blood, bloodline, the awakened eye, the clan's undying fire) — NEVER through wills, estates, ledgers, rent or leases. Exception: only use a real-world domain if the character literally comes from it (e.g. a chef character may use cooking imagery). Give ONE concrete in-world metaphor system in 1-2 sentences.",
  "delivery_personality": "the rap persona/attitude and cadence feel",
  "things_to_avoid": ["reference's exact wording/rhymes/hook", "line-by-line mirroring", "generic clichés", "..."]
}
Return ONLY the JSON object.
"""

GENERATE_INTRO_ORIGINAL = """\
You are writing a completely original song. When a reference exists, it gives you ONLY
the sonic world (genre/BPM/energy) and a list of extracted character facts — you never
saw its lyrics and you never reproduce its expression."""

GENERATE_INTRO_STRUCTURE = """\
You are writing an original song that deliberately FOLLOWS a proven narrative skeleton
extracted from reference songs about this character. The reference gives you three things:
the sonic world (genre/BPM/energy), a list of extracted character facts, and the STORY
STRUCTURE to follow. You write EVERY word yourself — you never saw the reference's lyrics
and you never reproduce its expression; you follow only its SHAPE."""

GENERATE_INTRO_REWRITE = """\
You are RE-WRITING a rap that already exists. Reference songs told a story about this
character; you keep that STORY and its ideas but re-write the expression. The reference
gives you three things: the sonic world (genre/BPM/energy), a list of extracted character
facts, and the STORY to retell. You preserve WHAT the story says and its order, but you
write EVERY line, rhyme and cadence yourself — you never saw the reference's lyrics (only
a neutral description of its story) and you never lift its wording. Same story, new words."""


def _build_generate_base(rules_block: str, *, intro: str) -> str:
    """Assemble the GENERATE base system prompt from a mode intro + a rules block.

    ``intro`` is the framing paragraph (original vs. follow-structure) and ``rules_block``
    is either ``_ORIGINALITY_RULES`` (max-originality mode) or ``_STRUCTURE_FOLLOW_RULES``
    (follow-structure mode). Everything else (Suno craft rules, output format) is shared.
    """
    return f"""\
You are an expert songwriter and music producer for an anime-rap channel in the lane
of BASARA, M4RKIM, ANIRAP, Rustage and 7 Minutoz. You write a compelling song
about a specific anime character (or matchup) that will be generated on Suno AI.

{intro}

You know exactly how Suno reads its Style and Lyrics fields and you exploit that to
get a clean, on-genre track — not generic mush.

TWO AXES — keep them separate:
  SONIC WORLD  → genre, BPM, instruments, energy, beat feel (from the reference if any)
  LYRICAL IDENTITY → the ORIGINAL composition: character attitude, powers, arc, delivery,
                     following the CREATIVE DIRECTION brief and the mined lore.

{rules_block}

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
[Intro] style — always open with a [Spoken] label/producer shoutout tag: if a "Channel
name" is given in the user prompt, use it (e.g. "Enkai again...", "another one from
Enkai...", "Enkai, let's go..."). Follow with 1-2 short character-specific hype lines
(who this song is about / what they represent). 3-4 lines total — entry moment, NOT a
metaphor section, NOT verse content.
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
  through vivid, specific, ORIGINAL imagery. Not a plot summary — a hype anthem.
- Turn the mined lore, abilities and easter eggs into fresh metaphors and punchlines of
  your own invention — never a restatement of how the reference phrased them.
- Every metaphor and comparison must be drawn from the character's own world (their powers,
  techniques, weapons, and the anime's own objects and imagery). Do NOT reach for abstract
  real-world metaphor domains — legal/estate/will, finance/debt/ledger/rent/lease, corporate,
  sports, cooking, gambling — unless the character literally belongs to that world. If a theme
  is abstract (legacy, inheritance, destiny), picture it as an in-world object before you
  write the line.
- Canonical ability/technique/form names (jutsu, powers, transformations — e.g. "Mangekyou
  Sharingan", "Amaterasu", "Rasengan", "Bankai") MUST appear verbatim — fans expect to hear
  the exact name and it is a mark of authenticity. What you EVOKE instead: the reference's
  own hook phrasing and lyrical catchphrases — rewrite those; keep canonical names.
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


GENERATE_SYSTEM_PROMPT_BASE = _build_generate_base(
    _ORIGINALITY_RULES, intro=GENERATE_INTRO_ORIGINAL
)

# Follow-structure mode base — same Suno craft, but the originality block is replaced by
# the structure-follow rules and the intro reframed around following the skeleton.
GENERATE_SYSTEM_PROMPT_STRUCTURE_BASE = _build_generate_base(
    _STRUCTURE_FOLLOW_RULES, intro=GENERATE_INTRO_STRUCTURE
)

GENERATE_SYSTEM_PROMPT_NO_REF = GENERATE_SYSTEM_PROMPT_BASE + f"""
{_ARCHETYPE_RANGE_NO_REF}
"""

GENERATE_SYSTEM_PROMPT_WITH_REF = GENERATE_SYSTEM_PROMPT_BASE + f"""
{_REFERENCE_SONIC_RULES}
"""

# Follow-structure mode always needs a reference (there is no skeleton without one), so
# there is only a WITH_REF variant.
GENERATE_SYSTEM_PROMPT_STRUCTURE_WITH_REF = GENERATE_SYSTEM_PROMPT_STRUCTURE_BASE + f"""
{_REFERENCE_SONIC_RULES}
"""

# Rewrite-the-story mode base — same Suno craft, but the originality block is replaced by
# the rewrite rules and the intro reframed around re-telling the reference's story.
GENERATE_SYSTEM_PROMPT_REWRITE_BASE = _build_generate_base(
    _REWRITE_RULES, intro=GENERATE_INTRO_REWRITE
)

# Rewrite mode always needs a reference (there is no story to retell without one), so
# there is only a WITH_REF variant.
GENERATE_SYSTEM_PROMPT_REWRITE_WITH_REF = GENERATE_SYSTEM_PROMPT_REWRITE_BASE + f"""
{_REFERENCE_SONIC_RULES}
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


def suggest_genres(project: VideoProject, **_ignored) -> GenreSuggestions:
    """Return genre directions for the project's character/topic.

    When multiple references exist, each one produces its own set of directions
    (anchored to its sonic lane). All sets are returned together so the UI can
    accumulate and let the user pick freely.

    Extra keyword args are accepted and ignored for signature compatibility.
    """
    from cutforge.services import reference_service

    topic = project.topic or project.character
    all_profiles = reference_service.load_all_reference_profiles(project)

    all_directions: list[GenreDirection] = []
    character_read = ""

    if all_profiles:
        for idx, profile in enumerate(all_profiles):
            system = SUGGEST_SYSTEM_PROMPT_WITH_REF
            lines = [f"Character / matchup: {topic}", ""]
            lines += [
                "REFERENCE SONIC DNA (derive the genre/lane from this — sound only):",
                f"- BPM: {profile.get('bpm')} — use this one number in every style string",
                f"- Onset density: {profile.get('onset_rate_per_sec')} onsets/s",
                f"- Title: {profile.get('source_title')}",
                "",
                f"Propose {N_GENRE_SUGGESTIONS} directions. All {N_GENRE_SUGGESTIONS} must stay in the reference's lane and use its BPM. "
                "Vary within the lane (e.g. harder vs. more melodic). Every direction must also "
                "fit the character's personality.",
                "Return only valid JSON.",
            ]
            data = anthropic_client.complete_json(system, "\n".join(lines))
            if not character_read:
                character_read = data.get("character_read", "")
            for d in data.get("directions", []):
                all_directions.append(GenreDirection(**d, ref_index=idx))
    else:
        system = SUGGEST_SYSTEM_PROMPT_NO_REF
        lines = [
            f"Character / matchup: {topic}", "",
            f"Propose {N_GENRE_SUGGESTIONS} distinct genre/vibe directions for an original anime song about this.",
            "Return only valid JSON.",
        ]
        data = anthropic_client.complete_json(system, "\n".join(lines))
        character_read = data.get("character_read", "")
        for d in data.get("directions", []):
            all_directions.append(GenreDirection(**d, ref_index=0))

    return GenreSuggestions(character_read=character_read, directions=all_directions)


def _format_instruction_note(user_instruction: str, *, domain: str) -> str:
    """Return a high-priority steering block for a regeneration, or "" when empty.

    ``domain`` is "direction" or "lore" — only the flavour text differs. The note is
    meant to be prepended to the USER prompt so the model weights it heavily while the
    SYSTEM prompt's non-negotiable rules still bound it.
    """
    instruction = (user_instruction or "").strip()
    if not instruction:
        return ""
    if domain == "lore":
        return (
            "!!! HIGH-PRIORITY USER STEERING (overrides default extraction focus) !!!\n"
            "The user is RE-MINING this reference and gave an explicit instruction. "
            "Prioritise it when deciding what to surface and emphasise, while still "
            "obeying the system rules: extract KNOWLEDGE only (never reproduce the "
            "reference's phrasing/hooks/metaphors as facts), and never invent lore that "
            "is not supported by the transcript. If the instruction narrows or re-weights "
            "focus (e.g. 'focus on his abilities', 'ignore relationships'), follow it.\n"
            f"USER INSTRUCTION: {instruction}\n"
        )
    return (
        "!!! HIGH-PRIORITY USER STEERING (overrides default creative choices) !!!\n"
        "The user is REGENERATING this brief and gave an explicit instruction. Treat it "
        "as the top priority when it conflicts with your default instincts. Obey it while "
        "still respecting the non-negotiable rules in the system prompt (originality vs. "
        "the reference, lore = knowledge not lyrics to reuse). If the instruction says to "
        "avoid a theme/metaphor, do NOT reintroduce it; if it says to emphasise something, "
        "make it the spine of the brief.\n"
        f"USER INSTRUCTION: {instruction}\n"
    )


def plan_creative_direction(
    project: VideoProject,
    genre: str,
    *,
    music_profile: dict | None = None,
    lore_profile: ReferenceLoreProfile | None = None,
    structure_profile: NarrativeStructureProfile | None = None,
    story_profile: StoryContentProfile | None = None,
    new_hook: bool = True,
    is_vs: bool = False,
    user_instruction: str = "",
    refresh: bool = False,
) -> CreativeDirection:
    """Plan the original-song brief (cached to ``creative_direction_path``).

    Answers "what is the NEW song?" from the character, the chosen genre and — when a
    reference exists — its music profile and mined lore. Persisted so a re-run of the
    lyrics step does not re-plan unless ``refresh=True``. ``user_instruction`` is an
    optional free-text steering note (used when regenerating) — it is injected into the
    prompt but never persisted onto the model.

    ``structure_profile`` (follow-structure mode) is fed as a FIXED-shape constraint: the
    arc/arrangement/hook placement come from the proven skeleton, so the brief designs
    fresh CONTENT within that shape instead of fighting it.

    ``story_profile`` (rewrite-the-story mode) flips the planner: the story/points/ideas/
    arc are FIXED and a rewrite-specific system prompt designs only the EXPRESSION layer
    (metaphor world, rhyme/cadence, delivery, hook decision). ``new_hook`` decides whether
    the hook is kept-but-reworded or invented fresh. When ``story_profile`` is given it
    takes precedence over ``structure_profile``.
    """
    rewrite = story_profile is not None and not story_profile.is_empty()
    use_structure = (
        not rewrite and structure_profile is not None and not structure_profile.is_empty()
    )
    effective_mode = "rewrite" if rewrite else "structure" if use_structure else "original"

    if project.creative_direction_path.exists() and not refresh:
        data = json.loads(project.creative_direction_path.read_text(encoding="utf-8"))
        # Reuse the cache ONLY when it was planned for the SAME mode. A mode switch
        # (e.g. original → rewrite) must re-plan, or the cached brief keeps pointing
        # the writer away from the story/structure the new mode is meant to follow.
        if data.get("planned_mode", "original") == effective_mode:
            return CreativeDirection(**data)

    topic = project.topic or project.character
    lines = []
    note = _format_instruction_note(user_instruction, domain="direction")
    if note:
        lines += [note, ""]
    lines += [
        f"Character / matchup: {topic}",
        f"Chosen genre / sonic direction: {genre}",
    ]
    if project.character:
        lines.append(f"Character name(s): {project.character}")
    if project.anime:
        lines.append(f"Anime: {project.anime}")
    if project.mood:
        lines.append(f"Desired mood/vibe: {project.mood}")
    if is_vs:
        lines.append("This is a VS / matchup song — the direction should frame the clash.")

    if music_profile:
        lines += [
            "",
            "REFERENCE SONIC DNA (energy/tempo context — sound only, no lyrics):",
            f"- BPM: {music_profile.get('bpm')}",
            f"- Onset density: {music_profile.get('onset_rate_per_sec')} onsets/s",
        ]
    if lore_profile and not lore_profile.is_empty():
        lines += ["", _format_lore_for_prompt(lore_profile)]

    if rewrite:
        lines += [
            "",
            _format_story_for_prompt(story_profile, new_hook=new_hook),
            "",
            "IMPORTANT: the story, its points, ideas and emotional arc above are FIXED — do "
            "NOT invent a new story or change what it says. Design only the EXPRESSION used "
            "to re-write it (a fresh metaphor world, a rhyme/cadence approach that differs "
            "from the references, delivery, and the hook decision).",
            "",
            "Design the brief to RE-EXPRESS this story. Return only valid JSON.",
        ]
        system_prompt = CREATIVE_DIRECTION_REWRITE_SYSTEM_PROMPT
    else:
        if use_structure:
            lines += [
                "",
                _format_structure_for_prompt(structure_profile),
                "",
                "IMPORTANT: the emotional arc, section arrangement and hook placement are "
                "FIXED by this proven skeleton. Design a fresh core theme, narrative angle, "
                "metaphor world and hook CONCEPT that live WITHIN this shape — do not fight "
                "the structure. Your emotional_arc must be compatible with the skeleton's.",
            ]
        lines += [
            "",
            "Design the brief for a NEW original song. Return only valid JSON.",
        ]
        system_prompt = CREATIVE_DIRECTION_SYSTEM_PROMPT

    user_prompt = "\n".join(lines)

    data = anthropic_client.complete_json(system_prompt, user_prompt)
    direction = CreativeDirection(**data)
    # Stamp the mode this brief was planned for so a later mode switch invalidates it.
    direction.planned_mode = effective_mode

    project.run_dir.mkdir(parents=True, exist_ok=True)
    project.creative_direction_path.write_text(
        direction.model_dump_json(indent=2), encoding="utf-8"
    )
    return direction


def generate_package(project: VideoProject, genre: str, *, is_vs: bool = False,
                     reference_profile: dict | None = None,
                     ref_index: int = 0,
                     mode: str = "original",
                     follow_structure: bool = False,
                     new_hook: bool = True,
                     user_instruction: str = "",
                     refresh: bool = False,
                     refresh_lore: bool = False,
                     refresh_structure: bool = False,
                     on_log=None,
                     **_ignored) -> SongPackage:
    """Generate the full Suno package and write lyrics.txt + suno_prompt.json.

    ``ref_index`` selects which reference provides the sonic DNA (BPM/flow/style).
    All available references contribute lore (merged and deduplicated).

    ``mode`` selects the lyrics-generation mode:
    - ``"original"`` (default): a 100% original composition (only sonic DNA + lore reach
      the writer);
    - ``"structure"``: follows the proven narrative skeleton across the reference(s) while
      keeping 100% original wording;
    - ``"rewrite"``: re-tells the SAME story extracted from the reference(s) with new
      rhymes/rhythm/phrasing. ``new_hook`` decides whether the hook is kept-but-reworded
      (``False``) or invented fresh (``True``).
    Both ``"structure"`` and ``"rewrite"`` need at least one reference; with none, or with
    no usable profile, they fall back to ``"original"``.

    ``follow_structure`` is the legacy boolean — when ``True`` and ``mode`` was left at the
    default, it is treated as ``mode="structure"`` for backward compatibility.

    ``refresh`` re-plans the creative direction; ``refresh_lore`` re-mines all lore;
    ``refresh_structure`` re-extracts the narrative structure / shared story.
    ``user_instruction`` optionally steers the creative-direction re-plan.
    """
    from cutforge.services import (
        lore_service, reference_service, story_service, structure_service,
    )

    log = on_log or (lambda _m: None)

    # Legacy compat: the old boolean maps to the "structure" mode when no explicit mode.
    if follow_structure and mode == "original":
        mode = "structure"
    if mode not in ("original", "structure", "rewrite"):
        mode = "original"

    # Sonic DNA comes from the selected reference only.
    if reference_profile is None:
        reference_profile = reference_service.load_reference_profile(project, index=ref_index)

    # Mine ALL references for lore and merge.
    all_ref_profiles = reference_service.load_all_reference_profiles(project)
    lore_profile = None
    if all_ref_profiles:
        individual_lores = []
        for idx in range(len(all_ref_profiles)):
            lp = lore_service.mine_reference_lore(project, index=idx,
                                                  refresh=refresh_lore, on_log=log)
            if lp:
                individual_lores.append(lp)
        lore_profile = lore_service.merge_lore_profiles(individual_lores)

    # Follow-structure mode: synthesize the shared skeleton across all references.
    structure_profile = None
    if mode == "structure":
        if all_ref_profiles:
            structure_profile = structure_service.extract_structure_profile(
                project, refresh=refresh_structure, on_log=log)
        if not (structure_profile and not structure_profile.is_empty()):
            log("Follow-structure requested but no usable reference skeleton — "
                "falling back to max-originality mode.")
            structure_profile = None
            mode = "original"

    # Rewrite-the-story mode: synthesize the shared story across all references.
    story_profile = None
    if mode == "rewrite":
        if all_ref_profiles:
            story_profile = story_service.extract_story_profile(
                project, refresh=refresh_structure, on_log=log)
        if not (story_profile and not story_profile.is_empty()):
            log("Rewrite-the-story requested but no usable reference story — "
                "falling back to max-originality mode.")
            story_profile = None
            mode = "original"

    direction = plan_creative_direction(
        project, genre,
        music_profile=reference_profile, lore_profile=lore_profile,
        structure_profile=structure_profile,
        story_profile=story_profile, new_hook=new_hook,
        is_vs=is_vs, user_instruction=user_instruction, refresh=refresh,
    )

    topic = project.topic or project.character
    lines = [
        f"Character / matchup: {topic}",
        f"Chosen genre / style direction: {genre}",
    ]
    if project.character:
        lines.append(f"Character name(s): {project.character}")
    if project.anime:
        lines.append(f"Anime: {project.anime}")
    if project.channel_slug:
        lines.append(f"Channel name: {project.channel_slug.replace('-', ' ').title()}")
    if is_vs:
        lines.append(
            "This is a VS / matchup song — alternate perspectives and make the chorus the clash."
        )

    lines += ["", _format_direction_for_prompt(direction)]

    use_structure = structure_profile is not None  # already validated non-empty above
    use_rewrite = story_profile is not None         # already validated non-empty above

    if reference_profile:
        if use_rewrite:
            system_prompt = GENERATE_SYSTEM_PROMPT_REWRITE_WITH_REF
        elif use_structure:
            system_prompt = GENERATE_SYSTEM_PROMPT_STRUCTURE_WITH_REF
        else:
            system_prompt = GENERATE_SYSTEM_PROMPT_WITH_REF
        flow = reference_profile.get("flow", {})
        lines += [
            "",
            "REFERENCE SONIC DNA (build the STYLE from this — sound only, no lyrics):",
            f"- BPM: {reference_profile.get('bpm')} (put this ONE number in the style string)",
            f"- Time signature: {reference_profile.get('time_signature')}/4",
            f"- Onset density: {reference_profile.get('onset_rate_per_sec')} onsets/s",
            f"- Flow: {flow.get('words_per_sec')} words/s, ~{flow.get('syllables_per_beat')} syllables/beat",
            "",
            "Build the style string from the reference's sonic lane. Do NOT add orchestral "
            "strings, choir or cinematic elements unless present in the reference. Stay in "
            "the reference's genre lane. The reference's WORDS are off-limits — write the "
            "lyrics entirely from the creative direction and lore above.",
        ]
        if lore_profile and not lore_profile.is_empty():
            lines += ["", _format_lore_for_prompt(lore_profile)]
        if use_structure:
            lines += ["", _format_structure_for_prompt(structure_profile)]
        elif use_rewrite:
            lines += ["", _format_story_for_prompt(story_profile, new_hook=new_hook)]
    else:
        system_prompt = GENERATE_SYSTEM_PROMPT_NO_REF

    if use_rewrite:
        lines += [
            "",
            "Write the complete Suno package: RE-TELL the story above — keep WHAT it says "
            "and its order, but write EVERY line, rhyme and cadence yourself with new "
            "phrasing (never lift the reference's wording). Handle the hook as instructed. "
            "Return only valid JSON.",
        ]
    elif use_structure:
        lines += [
            "",
            "Write the complete Suno package: FOLLOW the narrative structure above, but "
            "write EVERY word, rhyme and hook yourself — 100% original expression. "
            "Return only valid JSON.",
        ]
    else:
        lines += ["", "Write the complete ORIGINAL Suno package. Return only valid JSON."]
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
