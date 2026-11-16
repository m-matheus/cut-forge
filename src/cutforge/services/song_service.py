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
import re
import statistics

from cutforge.integrations import anthropic_client
from cutforge.models.lore import ReferenceLoreProfile
from cutforge.services import catalog_service, lyrics_lint_service
from cutforge.models.project import VideoProject
from cutforge.models.song import CreativeDirection, SongPackage
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
  <specific subgenre>, <1–2 signature instruments>, <vocal plan>, <mood and dynamics>,
  <one BPM number> BPM

Rules:
- KEEP IT LEAN: 6–8 comma-separated descriptors, ~80–140 characters, never over 200. Every extra
  instrument dilutes the mix — Suno averages a long list ("taiko, koto, choir, strings,
  808, drill hats...") into generic "epic" mush. One sonic idea, 1–2 signature instruments
  ("sad piano, deep 808", "distorted 808 bass, sliding drill hi-hats").
- Lead with a SPECIFIC subgenre ("emotional dark trap", "UK drill", "boom bap", "melodic
  trap", "orchestral trap") — never bare "hip-hop" or "rap".
- Exactly ONE BPM number (e.g. "140 BPM"), never a range.
- Vocal plan — say WHERE it raps and WHERE it sings. Default for emotional, tragic or epic
  characters: "male rap verses, soaring sung chorus". Pure rap ("aggressive rapped male
  vocal") only for a cold/menacing lane whose brief has no sung hook.
- Mood and dynamics — the song's overall mood lives HERE, not in the lyrics tags. Name the
  contrast so the track breathes: "dynamic build and drop", "quiet verses, explosive
  chorus". Never one flat energy ("aggressive" from start to end).

BANNED IN STYLE — these are VISUAL / meta terms Suno cannot hear:
  AMV, anime edit, anime rap, "cinematic anime rap", anime AMV, music video,
  montage, edit, clip, scene, 4K, visuals, "epic anime", any character/anime name,
  any artist, band or producer name (the lanes named in this prompt included),
  "make it hard", "best song", narrative content about the plot.
Translate that intent into SOUND. "Cinematic anime rap" → "orchestral trap, epic
choir, soaring strings, booming 808s". The "anime" feel comes from the sonic palette,
not from the word "anime". No contradictory genres ("trap, lofi chill").

RAP vs SUNG: rapped verses come from "rap verses" in the style + a rap tag on each verse;
a sung hook comes from "sung chorus" in the style + [Sung, ...] on the chorus. The
contrast between the two is what makes the hook land — don't kill it in the Exclude field.
"""

_SUNO_LYRICS_RULES = """\
LYRICS FIELD — STRUCTURE, ENERGY AND DELIVERY TAGS
Suno obeys bracketed tags in the Lyrics field. Use them to drive the ENERGY of each
section, not just to label it.
- Section tags: [Intro] [Verse 1] [Pre-Chorus] [Chorus] [Build] [Break] [Breakdown]
  [Drop] [Bridge] [Final Chorus] [Outro] [End].
- Say how each section sounds INSIDE its tag, in 2–4 words at most: [Intro: soft piano]
  [Verse 1: calm rap] [Verse 2: aggressive rap] [Build: drums rising] [Drop: heavy 808]
  [Verse 3: tender rap, piano] [Chorus: sung, emotional] [Final Chorus: belted, full choir]
  [Outro: spoken, voice breaking]. Suno sometimes SINGS long bracket text, so a tag is never
  a sentence, and the song's overall mood goes in the STYLE string, not in tags.
- Switch delivery mid-section with a short tag on its own line before the line(s) it covers:
  [Shouted] for an outburst, [Spoken, flat] for a cold reply, [Whispered] for a secret.
- [Break] holding ONE short spoken line = the beat cuts out right before an impact. It is
  the strongest moment in the song — use it once.
- [Male Vocal] / [Female Vocal] pin the voice gender; [Gang Vocals] / [Chant] give a group
  shout (anthem-type hooks only).
- A sound effect or scene direction is a short tag on its own line: [Chains rattle]
  [Bell tolls]. NEVER put it in parentheses: Suno SINGS parenthetical text as a backing vocal.
- (Parentheses) are ONLY for ad-libs that MUST be sung: (yeah) (one more time).
- Anything NOT bracketed or in parentheses WILL be sung as lyrics.
"""

_EMOTIONAL_CRAFT_RULES = """\
EMOTIONAL CRAFT — what separates a song that HITS from generic AI rap
These are TECHNIQUES, not expression: you may use them even when the reference does too.
The originality rules apply to wording, never to craft.

1. A LINE IS A BREATH. Suno gives each line its own rhythmic slot: short lines land with
   weight, long lines get rushed into one flat stream. Sung, quiet and peak moments use
   short lines (see FLOW TARGET); rap verses use the length in RAP CRAFT. Split a thought
   that outgrows its line. A one-word line ("Fine." / "War.") is a rare accent for impact,
   never a run of them.
2. DEPTH OVER BREADTH. Dramatize the 4–6 SCENES of the brief, each with room to breathe
   (one scene can fill a whole verse). Drop lore that does not serve the arc — a song that
   touches every fact reads like a wiki page and moves nobody.
3. SCENES, NOT SUMMARIES. Put the listener INSIDE the moment: direct dialogue in quotes
   (a [Shouted] / [Spoken] tag per voice), what the character sees and hears, present-
   tense urgency. A line of the argument itself beats a line describing the argument.
4. PLAIN WORDS AT THE PEAKS. The most emotional lines are the simplest ("Then the mountain
   fell."). Imagery and wordplay belong in the verses; at the peak the character just
   says it.
5. THE SONG MUST BREATHE. Follow the brief's ENERGY MAP: quiet open → rising → a silence
   ([Break]) → impact ([Drop]) → biggest final moment → quiet outro. Never one energy from
   start to end.
6. THE HOOK IS SUNG AND SIMPLE. 6–8 short lines, open singable vowels, a phrase a viewer
   can sing back after ONE listen. Strongest with a double meaning tied to the
   character's core wound. The [Final Chorus] FLIPS it: rewrite 2–4 of its lines so the
   hook pays off the arc — what it meant at the start is answered at the end — keeping the
   rhyme sounds of the original chorus.
7. PLANT AND PAY OFF. Use the brief's CALLBACK: plant the small concrete detail early (a
   habit, an object, a phrase) and bring it back at the end with new meaning.
8. THE OUTRO LANDS IT. The quietest, most direct lines of the song, in the character's
   voice — no new lore, just the last thing they would say.
"""

_RAP_CRAFT_RULES = """\
RAP CRAFT — for every section or passage tagged as rap
- RHYME: every rap verse has a rhyme scheme, with end rhymes and internal rhymes. Free verse
  belongs only in spoken or deliberately poetic passages.
- LENGTH: 8–14 syllables per line, most lines 8–10 (about 6–8 words). More than 10 words is
  too long: Suno rushes dense lines into a flat stream. Split a thought that does not fit
  across two rhymed lines instead of packing it into one.
- NO CHOPPING: never stack runs of 1–3-word lines in a rap verse, they come out as a broken,
  stop-start flow. A short line is a single accent between full lines.
- This length is for RAP only. Sung hooks, [Break], quiet sections and the outro keep their
  short lines.
- When a chorus repeats with variation, keep the rhyme sounds of the original chorus.
"""

_COHERENCE_RULES = """\
COHERENCE — the song must hold up for a fan who knows the story
- FACTS: every reference to an existing character's story stays faithful to the order and
  context of the events. Never confuse WHEN something happened.
- AMBIGUITY: avoid words a listener could take the wrong way (literal vs. figurative).
- CONTRADICTIONS: no section may say or imply the opposite of another section.
- INVENTED TERMS: never coin a word that sounds like it belongs to the universe but does not
  exist in it. Use the canonical name or plain words.
"""

_PRONUNCIATION_RULES = """\
PRONUNCIATION
Suno mispronounces proper names and foreign words. Keep the canonical spelling in "lyrics",
because the lyrics also become on-screen captions, and list each hard word in
"pronunciations" with an English phonetic respelling that Suno reads correctly, syllables
separated by hyphens, e.g. {"Mangekyou": "Mahn-geh-kyoh", "Auswählen": "Ows-vay-len"}. Only
list words Suno would likely get wrong; use {} when there are none.
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
- Flow: the FLOW TARGET gives the line length measured from the reference — match it
  outside the rap verses.
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
        parts.append("- Scenes to dramatize (give each room — depth over breadth):")
        parts += [f"    • {p}" for p in direction.key_lore_points]
    if direction.energy_map:
        parts.append(f"- ENERGY MAP (tag each section accordingly): {direction.energy_map}")
    if direction.callback:
        parts.append(f"- CALLBACK (plant early, pay off at the end): {direction.callback}")
    if direction.original_metaphor_direction:
        parts.append(f"- Recurring imagery: {direction.original_metaphor_direction}")
    if direction.delivery_personality:
        parts.append(f"- Delivery personality: {direction.delivery_personality}")
    if direction.intro_concept:
        parts.append(f"- Intro concept: {direction.intro_concept}")
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


# Line-density defaults (no usable reference lyrics) and clamps, for everything but rap
# verses (see RAP CRAFT). A ~4-word line is one breath; the Obito A/B test showed 9–10-word
# lines get rushed into a flat stream.
_DEFAULT_WORDS_PER_LINE = 4.5
_DEFAULT_TOTAL_WORDS = 600
_WORDS_PER_LINE_RANGE = (3.0, 7.0)
_TOTAL_WORDS_RANGE = (450, 700)
# Mean words per line (rap included) past which the song approaches the density that lost
# that A/B test.
_DENSE_MEAN_WORDS_PER_LINE = 8


def _flow_target(reference_profile: dict | None) -> tuple[float, int, str]:
    """``(words_per_line, total_words, note)`` for the writer's FLOW TARGET.

    Measured from the reference lyrics when they are real lyric lines (pasted or from
    subtitles) — Whisper segments are not lyric lines, so they fall back to the defaults.
    """
    profile = reference_profile or {}
    if profile.get("lyrics_source") in ("manual", "subtitles"):
        counts = [len(ln.split()) for ln in (profile.get("transcript") or "").splitlines()
                  if ln.strip()]
        if len(counts) >= 20:
            median = statistics.median(counts)
            wpl = min(max(float(median), _WORDS_PER_LINE_RANGE[0]), _WORDS_PER_LINE_RANGE[1])
            total = min(max(round(sum(counts) * 0.8), _TOTAL_WORDS_RANGE[0]),
                        _TOTAL_WORDS_RANGE[1])
            note = (f"measured from the reference lyrics: median {median:g} words/line, "
                    f"{len(counts)} lines, {sum(counts)} words")
            return wpl, total, note
    return _DEFAULT_WORDS_PER_LINE, _DEFAULT_TOTAL_WORDS, "channel default"


def _format_flow_target(wpl: float, total: int, note: str) -> str:
    return "\n".join([
        f"FLOW TARGET ({note}):",
        f"- Sung, quiet and peak lines: ~{wpl:g} words per line — one breath.",
        "- Rap verses: 8–14 syllables per line, most 8–10 (about 6–8 words), per RAP CRAFT, "
        "whatever the number above says.",
        f"- ~{total} words total (≈3.5–4 min on Suno). This is a budget, do not exceed it.",
    ])


def _words_per_line(lyrics: str) -> tuple[float, int]:
    """Mean sung words per line and the line count (tags and ad-libs ignored)."""
    lines = catalog_service.sung_text(lyrics).splitlines()
    if not lines:
        return 0.0, 0
    return sum(len(ln.split()) for ln in lines) / len(lines), len(lines)


def suno_lyrics(lyrics: str, pronunciations: dict[str, str]) -> str:
    """The sheet to paste into Suno: hard words respelled, everything else as written.

    ``lyrics.txt`` keeps the canonical names because captions, the release lyrics and the
    metadata read it, so this is derived from it instead of being stored as a second copy.
    """
    return _swap_terms(lyrics, pronunciations)


def canonical_lyrics(sheet: str, pronunciations: dict[str, str]) -> str:
    """Undo ``suno_lyrics``: a sheet read back from Suno, with the canonical names restored."""
    return _swap_terms(sheet, {respelling: term for term, respelling in pronunciations.items()})


def _swap_terms(lyrics: str, swaps: dict[str, str]) -> str:
    """Replace whole words per ``swaps`` (any case), leaving [tag] lines as written."""
    terms = sorted((t for t, new in swaps.items() if t.strip() and new.strip()),
                   key=len, reverse=True)
    if not terms:
        return lyrics
    swap = {t.lower(): swaps[t].strip() for t in terms}
    pattern = re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, terms)) + r")(?!\w)",
                         re.IGNORECASE)
    return "\n".join(
        ln if ln.lstrip().startswith("[")
        else pattern.sub(lambda m: swap[m.group(0).lower()], ln)
        for ln in lyrics.splitlines()
    )


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_SONIC_DIRECTION_RULES = f"""SONIC DIRECTION — you also pick the SOUND of the song ("sonic_direction").
- If the user forced a genre, the sonic_direction is in that genre.
- If a reference is given, stay in its lane and BPM; vary only within it.
- Otherwise pick by the character's archetype (the GENRE SELECTION list, if given).
- If a CHANNEL CATALOG with previous styles is given: do NOT reuse the subgenre of the
  3 most recent songs, nor their vocal-delivery + instrument combination, unless the
  forced genre or the reference lane requires it — then change the signature instruments
  and energy instead. Listeners hear the catalog back to back.
- Write it as a Suno style seed following these rules:

{_SUNO_STYLE_RULES}
"""


# Shared by both planners: plan for emotion (scenes, energy, callback) instead of coverage.
_DESIGN_FOR_EMOTION = """\
DESIGN FOR EMOTION, NOT COVERAGE. A song that touches every fact reads like a wiki page.
- Pick 4–6 SCENES that carry the arc — concrete moments (who, where, what is said) the
  writer can put the listener inside, with dialogue where it fits. Let the rest go.
- Plan the ENERGY MAP: where the song is quiet, where it builds, the ONE silence ([Break])
  right before the biggest impact, the [Drop], the biggest final moment, the quiet outro.
- The hook is SUNG and simple (unless this is a cold, pure-rap lane), ideally a double
  meaning tied to the character's core wound, and the final chorus FLIPS it to pay off
  the arc.
- Plan a CALLBACK: a small concrete detail (a habit, an object, a phrase) planted early
  and paid off at the end with new meaning.

"""

_IMAGERY_SOURCING = """\
IMAGERY SOURCING: imagery comes from the character's OWN universe — powers, techniques,
weapons, creatures, locations, the source material's objects. Never translate the story
into an unrelated real-world domain (legal/courtroom/wills, finance/debt/ledgers,
corporate, sports, cooking, gambling) unless the character literally lives there. Abstract
themes (legacy, destiny, inheritance) are voiced through concrete in-world objects.

"""

# JSON fields shared by both planners' output formats (no trailing comma).
_DIRECTION_CRAFT_FIELDS = """\
  "hook_concept": "the SUNG hook — its concept (original, or the story's concept reworded if the user kept it), the double meaning it carries, and how the FINAL chorus flips it to pay off the arc",
  "key_lore_points": ["4–6 SCENES to dramatize — each one concrete moment (who, where, what is said), not a fact list", "..."],
  "energy_map": "section-by-section energy/delivery plan in one line, e.g. 'Intro whispered over piano → V1 calm rap → Chorus sung → V2 tension rising → Build → Break (one spoken line, silence) → V3 tender, piano only → Drop aggressive → Bridge cold then cracking → Final Chorus belted, choir → Outro spoken, quiet'",
  "callback": "the small concrete detail planted early (a habit, object or phrase) and the moment at the end where it returns with new meaning",
  "original_metaphor_direction": "1–2 recurring in-world images (an object, sound or place from the character's world) the song returns to — NOT an elaborate metaphor system; the emotional peaks stay in plain words. 1–2 sentences.",
  "delivery_personality": "the persona/attitude, and where it raps vs where it sings",
  "intro_concept": "how THIS song opens, in one line: pick a shape (cold open in the character's voice, an in-world sound as a short [tag], a one-word chant, a question to the listener, a scene-setter, or the bare producer tag) and the specific moment/sound/word it uses. It must differ from every intro in the CHANNEL CATALOG if one is given, and never use the frames \\"This one's for...\\" or \\"<Name>, the boy who...\\"\""""


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
arc, an original hook concept, the scenes worth dramatizing, recurring imagery (not the
reference's), and the delivery personality. In "things_to_avoid", include reusing the
reference's expression (its hooks, metaphors, phrasing) and any clichés that would make
the song generic.

Prefer specific, character-grounded ideas over generic hype. If strong easter eggs /
obscure lore exist, prioritise them — they make the song feel authentic and specific.

""" + _DESIGN_FOR_EMOTION + _IMAGERY_SOURCING + _SONIC_DIRECTION_RULES + """
OUTPUT FORMAT — a single valid JSON object, no markdown fences, no commentary:
{
  "sonic_direction": "a Suno-ready style seed in one line, per SONIC DIRECTION above",
  "core_theme": "the central idea of the NEW song, in one line",
  "narrative_angle": "the fresh POV/approach (e.g. 'first-person vow at his lowest moment')",
  "emotional_arc": "how the feeling evolves across the song",
""" + _DIRECTION_CRAFT_FIELDS + """,
  "things_to_avoid": ["reference's hook/metaphors/phrasing (its techniques — dialogue, short lines, silence — are fair game)", "generic clichés", "..."]
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

The story is fixed, but you still choose its SCENES: pick the 4–6 story points that carry
the arc and give them room; the rest can be a single line or skipped.

""" + _DESIGN_FOR_EMOTION + _IMAGERY_SOURCING + _SONIC_DIRECTION_RULES + """
OUTPUT FORMAT — a single valid JSON object, no markdown fences, no commentary:
{
  "sonic_direction": "a Suno-ready style seed in one line, per SONIC DIRECTION above",
  "core_theme": "the story's central idea (kept from the STORY block, in one line)",
  "narrative_angle": "the story's POV/approach (kept — how it is TOLD, re-expressed)",
  "emotional_arc": "the arc from the STORY block (kept; may be phrased in your words)",
""" + _DIRECTION_CRAFT_FIELDS + """,
  "things_to_avoid": ["reference's exact wording/rhymes/hook (its techniques — dialogue, short lines, silence — are fair game)", "line-by-line mirroring", "generic clichés", "..."]
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

{_EMOTIONAL_CRAFT_RULES}

{_RAP_CRAFT_RULES}

{_COHERENCE_RULES}

{_PRONUNCIATION_RULES}

THE "EXCLUDE" FIELD
Return 3–5 comma-separated terms that would wreck THIS song, e.g. "female vocals, heavy
autotune, lo-fi, EDM". Do NOT exclude singing / melodic vocals when the song has a sung
hook (the default). Only a cold, pure-rap song with a rapped hook may exclude "singing,
sung chorus".

LYRICS STRUCTURE
Length: ~3.5–4 minutes — the FLOW TARGET in the user prompt gives the word budget. The whole
sheet, tags included, must stay under {lyrics_lint_service.SUNO_LYRICS_MAX_CHARS} characters or Suno cuts it.
A proven emotional arrangement (adapt it to the brief's ENERGY MAP):
  [Intro] → [Verse 1] → [Chorus] → [Verse 2] → [Build] → [Break] → [Verse 3] →
  [Breakdown] → [Drop] → [Chorus] → [Bridge] → [Final Chorus] → [Outro]
The first [Chorus] lands right after Verse 1 (within the first minute).
Section sizes in lines: rap verses 12–20, chorus 6–8, build 4–8, break 1, breakdown 6–12,
bridge 6–12, outro 4–10. Strongest line FIRST in each section.
[Intro] style — a quick entry moment of 2-6 short lines, NOT a metaphor section, NOT verse
content. It must contain the producer tag: if a "Channel name" is given in the user prompt,
the artist says it plainly, ONCE, in a [Spoken] (or [Whispered]) line — just the name, maybe
one plain word around it. Do NOT dress the tag up — no metaphor, no poetry, no "one more for
the archive"; it is only the artist saying their name.
Every song needs a DIFFERENT intro. Build it from the creative direction's "intro_concept";
the shapes available include (and are not limited to):
  - cold open: one line IN the character's voice from a specific moment, then the tag;
  - sound first: an in-world sound as a short bracket tag — [Chains rattle], [Bell tolls] —
    then the producer tag and one concrete image;
  - chant: one word or name from the lore repeated as a chant, the tag cutting through it;
  - question/challenge to the listener about the character;
  - scene-setter: where and when we are (the battlefield, the night of the massacre);
  - the tag alone over the beat, then straight into the verse (a 1-line intro is fine).
The tag may come first, last or in the middle. NEVER use the frames "This one's for..." or
"<Name>, the boy/man who..." and never open with "<channel> again" — the channel has
overused them. If a CHANNEL CATALOG block is given, the intro must not resemble any intro in
it (frame, sentence shape, epithet style).
The channel name is the ARTIST / producer tag ONLY. It is NOT the character, NOT a person in
the story, and must appear ONLY inside this [Intro] tag — NEVER in a verse, chorus, pre-chorus,
bridge or [Outro], and NEVER spoken as if it were the character or part of their world.
[Outro] style — the quietest, most direct closing lines, in the CHARACTER's own voice: the
last thing they would say, ideally paying off the CALLBACK. Do NOT put the channel name /
producer tag here — that belongs only in the [Intro].
Adapt to the genre:
  - Cold / menacing pure-rap lane → the hook may be a rapped chant instead of sung; keep
    the [Break] → [Drop] contrast anyway.
  - Cypher / anthem → verse-per-character feel, [Gang Vocals] + [Chant] on the hook.
For a vs / matchup: alternate perspectives (tag each voice), make the chorus the clash.

TONE & CONTENT
- The song is FROM or ABOUT the character — their personality, powers and arc told
  through SCENES (see EMOTIONAL CRAFT), never a plot summary.
- Turn the mined lore, abilities and easter eggs into lines of your own invention — never
  a restatement of how the reference phrased them.
- Imagery comes from the character's own world (powers, techniques, weapons, the anime's
  objects) — no metaphors from unrelated real-world domains (legal, finance, corporate,
  sports, cooking, gambling) unless the character lives there.
- Canonical ability/technique/form names (jutsu, powers, transformations — e.g. "Mangekyou
  Sharingan", "Amaterasu", "Rasengan", "Bankai") MUST appear verbatim — fans expect to hear
  the exact name and it is a mark of authenticity. What you EVOKE instead: the reference's
  own hook phrasing and lyrical catchphrases — rewrite those; keep canonical names.
- The hook should be something viewers sing back in the comments.

LINE RULES (lyrics also become on-screen karaoke captions)
- Rap verses per RAP CRAFT; every other line per the FLOW TARGET in the user prompt (default
  3–6 words). One breath, one idea per line.
- No "yeah yeah" filler beyond intentional (parenthetical) ad-libs.

OUTPUT FORMAT — a single valid JSON object, no markdown fences, no commentary:
{{
  "title": "Song title — short and evocative (e.g. 'Shadow Sovereign')",
  "style": "The Suno STYLE string — audio only, following the formula and banned-terms rule",
  "exclude": "Short comma-separated Suno Exclude-Styles list",
  "lyrics": "Full structured lyrics as one string. \\n for line breaks; each [tag] on its own line.",
  "pronunciations": {{"hard word exactly as written in the lyrics": "English phonetic respelling"}},
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


INTRO_REWRITE_SYSTEM_PROMPT = """\
You fix ONE section of a finished anime-rap song for Suno: its [Intro]. The intro echoes
the channel's previous songs, and listeners hear them back to back. Write a NEW intro
body that opens the same song differently.

Rules:
- 2-6 short lines (3-7 words each), a quick entry moment — not verse content.
- Keep the producer tag: the channel name said plainly ONCE in a [Spoken] or [Whispered]
  line, no metaphor around it. It may come first, last or in the middle.
- Do NOT use any of the flagged phrases, nor the frames "This one's for...",
  "<Name>, the boy/man who..." or "<channel> again".
- Follow the intro concept if given; draw images and sounds from the character's world.
- Delivery tags and sound effects in short [brackets] of at most 4 words. (Parentheses) only
  for ad-libs that must be sung, because Suno sings whatever is inside them.

OUTPUT FORMAT — a single valid JSON object, no markdown fences:
{"intro": "the new [Intro] body only, without the [Intro] tag itself; \\n between lines"}
"""


REVIEW_SYSTEM_PROMPT = f"""\
You are the final editor of an anime-rap song about to be generated on Suno AI. You receive a
finished draft. Correct what breaks the rules below and leave everything else alone.

{_SUNO_LYRICS_RULES}

{_RAP_CRAFT_RULES}

{_COHERENCE_RULES}

{_PRONUNCIATION_RULES}

HOW TO EDIT
- Check every reference against the CHARACTER LORE, when given, and what you know of the source
  material. If you cannot confirm a fact or a term, rewrite the line so it no longer depends on it.
- Keep what works: the section order, the scenes, the hook and its double meaning, the callback,
  the plain words at the emotional peaks, and canonical names spelled correctly.
- Change only the lines that need it, in the song's voice, with your own words. Never add
  sections or lengthen the song. The channel name stays only in the [Intro].

OUTPUT FORMAT — a single valid JSON object, no markdown fences, no commentary:
{{
  "lyrics": "the full corrected lyrics as one string. \\n for line breaks; each [tag] on its own line.",
  "pronunciations": {{"hard word exactly as written in the lyrics": "English phonetic respelling"}},
  "changes": ["one short line per fix: what changed and which rule it broke", "..."]
}}
"""


def _deduplicate_intro(lyrics: str, catalog: list, direction: CreativeDirection,
                       channel_name: str, log) -> str:
    """Rewrite the [Intro] once if it reuses wording from the catalog's intros."""
    intro = catalog_service.section(lyrics, "intro")
    echoes = catalog_service.echoed_intro_phrases(intro, catalog)
    if not echoes:
        return lyrics
    log(f"Intro echoes previous songs ({'; '.join(echoes)}) — rewriting the intro.")
    previous = [f"- {s.intro.replace(chr(10), ' / ')}" for s in catalog if s.intro]
    user_prompt = "\n".join([
        f"Channel name: {channel_name}" if channel_name else "No channel name.",
        f"Intro concept: {direction.intro_concept}" if direction.intro_concept else "",
        "", "Flagged phrases (copied from previous intros): " + "; ".join(echoes),
        "", "Previous intros on the channel:", *previous,
        "", "The full song (rewrite ONLY its [Intro] body):", lyrics,
        "", "Return only valid JSON.",
    ])
    data = anthropic_client.complete_json(INTRO_REWRITE_SYSTEM_PROMPT, user_prompt)
    new_intro = (data.get("intro") or "").strip()
    if not new_intro:
        return lyrics
    still = catalog_service.echoed_intro_phrases(new_intro, catalog)
    if still:
        log(f"New intro still echoes: {'; '.join(still)} — kept it anyway, review by hand.")
    return catalog_service.replace_section(lyrics, "intro", new_intro)


def _read_pronunciations(data: dict) -> dict[str, str]:
    raw = data.get("pronunciations")
    return {str(k): str(v) for k, v in raw.items()} if isinstance(raw, dict) else {}


def _review_lyrics(lyrics: str, pronunciations: dict[str, str], *, topic: str,
                   direction: CreativeDirection, lore_profile: ReferenceLoreProfile | None,
                   log) -> tuple[str, dict[str, str]]:
    """Second pass over the draft. A failed review keeps the draft: it has already been paid for."""
    lint = lyrics_lint_service.lint_lyrics(lyrics)
    flagged = lint.warnings + lyrics_lint_service.check_length(lyrics)
    parts = [f"Character / matchup: {topic}", "", _format_direction_for_prompt(direction)]
    if lore_profile and not lore_profile.is_empty():
        parts += ["", _format_lore_for_prompt(lore_profile)]
    if flagged:
        parts += ["", "Flagged by the automatic checks (in Portuguese):", *flagged]
    parts += [
        "", "Current pronunciations: " + json.dumps(pronunciations, ensure_ascii=False),
        "", "The draft:", lyrics,
        "", "Return only valid JSON.",
    ]
    log("Reviewing the lyrics against the rules and the lore...")
    try:
        data = anthropic_client.complete_json(REVIEW_SYSTEM_PROMPT, "\n".join(parts))
    except Exception as exc:
        log(f"Review failed ({exc}) — keeping the draft.")
        return lyrics, pronunciations
    reviewed = (data.get("lyrics") or "").strip()
    if not reviewed:
        log("Review came back empty — keeping the draft.")
        return lyrics, pronunciations
    for change in data.get("changes") or []:
        log(f"Review: {change}")
    return reviewed, _read_pronunciations(data) or pronunciations


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
    genre: str = "",
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

    Answers "what is the NEW song?" — sound included — from the character and, when a
    reference exists, its music profile and mined lore. The planner picks the sonic
    direction itself (in the reference's lane, or by character archetype, steering away
    from the catalog's recent styles); a non-empty ``genre`` forces it instead. Persisted so a re-run of the
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
        # Same for the forced genre — and a brief from before the planner chose the sound
        # (no sonic_direction) is re-planned when no genre is forced. A brief from before
        # the energy map (no scenes/energy plan) is re-planned too.
        if (data.get("planned_mode", "original") == effective_mode
                and data.get("planned_genre", "") == genre
                and (genre or data.get("sonic_direction"))
                and data.get("energy_map")):
            return CreativeDirection(**data)

    topic = project.topic or project.character
    lines = []
    note = _format_instruction_note(user_instruction, domain="direction")
    if note:
        lines += [note, ""]
    lines += [
        f"Character / matchup: {topic}",
        f"User-forced genre (sonic_direction MUST be in it): {genre}" if genre else
        "Genre: not forced — choose the sonic_direction yourself (see SONIC DIRECTION).",
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
            "The sonic_direction stays in the reference's lane and uses its BPM; vary only "
            "within that lane (harder vs. more melodic, a different signature instrument).",
        ]
    elif not genre:
        lines += ["", _ARCHETYPE_RANGE_NO_REF]
    if lore_profile and not lore_profile.is_empty():
        lines += ["", _format_lore_for_prompt(lore_profile)]

    catalog_block = catalog_service.format_catalog_for_prompt(
        catalog_service.load_catalog(project))
    if catalog_block:
        lines += ["", catalog_block]

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
    # Stamp the mode/genre this brief was planned for so a later switch invalidates it.
    direction.planned_mode = effective_mode
    direction.planned_genre = genre

    project.run_dir.mkdir(parents=True, exist_ok=True)
    project.creative_direction_path.write_text(
        direction.model_dump_json(indent=2), encoding="utf-8"
    )
    return direction


def generate_package(project: VideoProject, genre: str = "", *, is_vs: bool = False,
                     reference_profile: dict | None = None,
                     ref_index: int = 0,
                     mode: str = "original",
                     follow_structure: bool = False,
                     new_hook: bool = True,
                     user_instruction: str = "",
                     review: bool = False,
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

    ``review`` adds a second model call that corrects the draft against the rules and the
    lore. The deterministic checks always run last, on whatever text comes out.
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
        f"Chosen genre / style direction: {genre or direction.sonic_direction}",
    ]
    if project.character:
        lines.append(f"Character name(s): {project.character}")
    if project.anime:
        lines.append(f"Anime: {project.anime}")
    channel_name = project.channel_slug.replace("-", " ").title() if project.channel_slug else ""
    if channel_name:
        lines.append(f"Channel name: {channel_name}")
    if is_vs:
        lines.append(
            "This is a VS / matchup song — alternate perspectives and make the chorus the clash."
        )

    lines += ["", _format_direction_for_prompt(direction)]

    target_wpl, target_words, flow_note = _flow_target(reference_profile)
    lines += ["", _format_flow_target(target_wpl, target_words, flow_note)]

    catalog = catalog_service.load_catalog(project)
    catalog_block = catalog_service.format_catalog_for_prompt(catalog)
    if catalog_block:
        lines += ["", catalog_block]

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
    lyrics = data.get("lyrics", "")
    pronunciations = _read_pronunciations(data)
    if review:
        lyrics, pronunciations = _review_lyrics(lyrics, pronunciations, topic=topic,
                                                direction=direction, lore_profile=lore_profile,
                                                log=log)
    lint = lyrics_lint_service.lint_lyrics(
        _deduplicate_intro(lyrics, catalog, direction, channel_name, log))
    lyrics = lint.lyrics
    style = data.get("style", "")
    warnings = (lint.warnings
                + lyrics_lint_service.check_length(suno_lyrics(lyrics, pronunciations))
                + lyrics_lint_service.check_style(style))
    for fix in lint.fixes:
        log(f"Lyrics fix: {fix}")
    if lint.adlibs:
        log("Kept as sung ad-libs: " + "; ".join(f"({a})" for a in lint.adlibs))
    for term, respelling in pronunciations.items():
        log(f"Suno pronunciation: {term} → {respelling}")
    for warning in warnings:
        log(f"ATENÇÃO: {warning}")
    wpl, line_count = _words_per_line(lyrics)
    log(f"Lyrics density: {wpl:.1f} words/line over {line_count} lines "
        f"(sung lines target ~{target_wpl:g}, rap 8–14 syllables).")
    if wpl > _DENSE_MEAN_WORDS_PER_LINE:
        log("Lines are very dense on average — Suno will rush them; consider regenerating.")
    package = SongPackage(
        title=data.get("title", ""),
        style=style,
        exclude=data.get("exclude", ""),
        lyrics=lyrics,
        pronunciations=pronunciations,
        warnings=warnings,
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
            "pronunciations": package.pronunciations,
            "warnings": package.warnings,
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
