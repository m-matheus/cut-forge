"""Reference lore-mining service — turn a reference transcript into character KNOWLEDGE.

This is the explicit "lore mining" step. It reads the auto-transcribed reference rap
(which was written about the SAME character) and extracts structured facts, abilities,
events, relationships, themes and — most importantly — anime/manga easter eggs that the
model might not otherwise know.

CRITICAL BOUNDARY: this step extracts KNOWLEDGE, never EXPRESSION. It must not
paraphrase the reference line by line, translate it, or reproduce its hooks/metaphors.
The song writer later uses the mined knowledge as raw material for a NEW composition.

The music DNA (BPM/flow) lives in a completely separate profile — see
``reference_service`` / ``reference_profile.json``.
"""
from __future__ import annotations

import json

from cutforge.integrations import anthropic_client
from cutforge.models.lore import ReferenceLoreProfile
from cutforge.models.project import VideoProject
from cutforge.services import reference_service

LORE_MINER_SYSTEM_PROMPT = """\
You are a LORE MINER for an anime-rap channel. You are given the auto-transcribed
lyrics of an existing rap that was written ABOUT a specific anime/manga character. Your
ONLY job is to extract KNOWLEDGE about the character from that transcript.

You are NOT rewriting, translating, paraphrasing or summarizing the song. You are
mining it for facts. Think of yourself as a wiki editor reading song lyrics to learn
about a character — not a translator.

WHY THIS MATTERS
The transcript may name very specific things the general model does not know: an obscure
technique, a particular fight, a secondary event from a saga, an object, a title, a
relationship, a symbol. These specifics ("easter eggs") are the most valuable output —
capture them even if you are unsure, and mark the confidence.

CLASSIFY EVERYTHING you find into the right bucket. Distinguish CANON KNOWLEDGE from the
reference composer's ARTISTIC EXPRESSION:
- FACT / EVENT / ABILITY / RELATIONSHIP / TITLE / SYMBOL / TRAIT / THEME
  → knowledge about the character (usable as raw material).
- EASTER_EGG → a specific anime/manga reference; record what it points to and what it
  likely means (interpreted_meaning) and any related lore.
- METAPHOR → an image the reference composer invented. Record it ONLY as context so the
  writer knows what NOT to reuse. Never present a composer's metaphor as canon.
- AUTHOR_INTERPRETATION → the reference composer's personal take/opinion, not canon.
- UNCERTAIN → anything you are unsure about, or that may be a transcription error.

RULES
- The transcript is auto-transcribed and may contain mis-heard words — flag doubtful
  items as UNCERTAIN with low confidence rather than inventing.
- Do NOT turn every line into a canonical fact. A line can be pure vibe with no lore.
- Do NOT reproduce distinctive phrases, hooks or metaphors as if they were facts.
- If the transcript is mostly filler (channel promos, "uh oh oh", ad-libs), return
  mostly-empty lists — that is a valid, correct answer.
- All output in English regardless of the transcript's language.

OUTPUT FORMAT — a single valid JSON object, no markdown fences, no commentary:
{
  "character": "the character the reference is about (best guess)",
  "facts": [ { "fact": "...", "category": "ability|event|relationship|trait|title|symbol|theme|role|other", "confidence": "high|medium|low" } ],
  "events": [ { "event": "...", "importance": "high|medium|low", "confidence": "high|medium|low" } ],
  "abilities": [ { "name": "...", "description": "...", "confidence": "high|medium|low" } ],
  "relationships": [ { "characters": ["...", "..."], "relationship": "...", "confidence": "high|medium|low" } ],
  "themes": ["..."],
  "personality_traits": ["..."],
  "easter_eggs": [ { "reference": "...", "interpreted_meaning": "...", "related_lore": "...", "confidence": "high|medium|low" } ],
  "author_interpretations": ["the reference composer's own takes — context only"],
  "uncertain_items": ["doubtful or possibly mis-transcribed items"]
}
Every list may be empty. Return ONLY the JSON object.
"""


def _build_user_prompt(project: VideoProject, transcript: str, source_title: str) -> str:
    who = project.character or project.topic or "the character"
    lines = [f"Character this reference is about: {who}"]
    if project.anime:
        lines.append(f"Anime / series: {project.anime}")
    if source_title:
        lines.append(f"Reference track title: {source_title}")
    lines += [
        "",
        "Reference transcript (auto-transcribed — mine it for character knowledge, "
        "do NOT paraphrase or translate it):",
        transcript,
        "",
        "Extract the lore. Return only valid JSON.",
    ]
    return "\n".join(lines)


def mine_reference_lore(project: VideoProject, index: int = 0, *, refresh: bool = False,
                        on_log=None) -> ReferenceLoreProfile | None:
    """Mine the reference transcript into a ReferenceLoreProfile (cached to disk).

    ``index`` selects which reference to mine (0 = primary).
    Returns ``None`` when there is no reference to mine. The result is persisted to
    ``project.ref_lore_profile_path(index)`` and reused on subsequent calls unless
    ``refresh=True``.
    """
    log = on_log or (lambda _m: None)

    music_profile = reference_service.load_reference_profile(project, index=index)
    if not music_profile:
        return None

    lore_path = project.ref_lore_profile_path(index)
    # Legacy flat path fallback for index 0.
    if index == 0 and not lore_path.parent.exists():
        lore_path = project.reference_lore_profile_path

    if lore_path.exists() and not refresh:
        data = json.loads(lore_path.read_text(encoding="utf-8"))
        return ReferenceLoreProfile(**data)

    transcript = (music_profile.get("transcript") or "").strip()
    if not transcript:
        log(f"No transcript for reference {index} — skipping lore mining.")
        return None

    source_title = music_profile.get("source_title", "")
    user_prompt = _build_user_prompt(project, transcript, source_title)

    log(f"Mining reference {index} transcript for character lore…")
    data = anthropic_client.complete_json(LORE_MINER_SYSTEM_PROMPT, user_prompt)
    data.setdefault("source_title", source_title)
    profile = ReferenceLoreProfile(**data)

    lore_path.parent.mkdir(parents=True, exist_ok=True)
    lore_path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    log(
        f"Lore {index} mined: {len(profile.facts)} facts, {len(profile.abilities)} abilities, "
        f"{len(profile.easter_eggs)} easter eggs."
    )
    return profile


def load_reference_lore_profile(project: VideoProject, index: int = 0) -> ReferenceLoreProfile | None:
    """Return the cached lore profile for one reference, or None if not yet mined."""
    lore_path = project.ref_lore_profile_path(index)
    if lore_path.exists():
        return ReferenceLoreProfile(**json.loads(lore_path.read_text(encoding="utf-8")))
    # Legacy flat layout fallback.
    if index == 0 and project.reference_lore_profile_path.exists():
        return ReferenceLoreProfile(
            **json.loads(project.reference_lore_profile_path.read_text(encoding="utf-8"))
        )
    return None


def load_all_lore_profiles(project: VideoProject) -> list[ReferenceLoreProfile]:
    """Return all mined lore profiles for this run, in index order."""
    profiles = []
    for i in range(max(len(project.reference_urls), 1)):
        p = load_reference_lore_profile(project, index=i)
        if p is not None:
            profiles.append(p)
    return profiles


def merge_lore_profiles(profiles: list[ReferenceLoreProfile]) -> ReferenceLoreProfile | None:
    """Merge multiple lore profiles into one, deduplicating by normalized text."""
    if not profiles:
        return None
    if len(profiles) == 1:
        return profiles[0]

    def _norm(s: str) -> str:
        return s.strip().lower()

    seen_facts: set[str] = set()
    seen_events: set[str] = set()
    seen_abilities: set[str] = set()
    seen_eggs: set[str] = set()

    merged_facts, merged_events, merged_abilities, merged_rels = [], [], [], []
    merged_themes: set[str] = set()
    merged_traits: set[str] = set()
    merged_eggs, merged_interps, merged_uncertain = [], [], []

    for p in profiles:
        for f in p.facts:
            key = _norm(f.fact)
            if key not in seen_facts:
                seen_facts.add(key)
                merged_facts.append(f)
        for e in p.events:
            key = _norm(e.event)
            if key not in seen_events:
                seen_events.add(key)
                merged_events.append(e)
        for a in p.abilities:
            key = _norm(a.name)
            if key not in seen_abilities:
                seen_abilities.add(key)
                merged_abilities.append(a)
        merged_rels.extend(p.relationships)
        merged_themes.update(t.strip() for t in p.themes)
        merged_traits.update(t.strip() for t in p.personality_traits)
        for eg in p.easter_eggs:
            key = _norm(eg.reference)
            if key not in seen_eggs:
                seen_eggs.add(key)
                merged_eggs.append(eg)
        merged_interps.extend(p.author_interpretations)
        merged_uncertain.extend(p.uncertain_items)

    return ReferenceLoreProfile(
        character=profiles[0].character,
        facts=merged_facts,
        events=merged_events,
        abilities=merged_abilities,
        relationships=merged_rels,
        themes=sorted(merged_themes),
        personality_traits=sorted(merged_traits),
        easter_eggs=merged_eggs,
        author_interpretations=merged_interps,
        uncertain_items=merged_uncertain,
        source_title=" + ".join(p.source_title for p in profiles if p.source_title),
    )
