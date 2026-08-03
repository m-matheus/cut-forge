"""Reference lore profile — character knowledge mined from a reference rap transcript.

This is deliberately kept SEPARATE from the reference *music* profile
(``reference_service`` / ``reference_profile.json``, which carries BPM, flow, onset
density — the sonic DNA). The lore profile carries only KNOWLEDGE about the character
(facts, abilities, events, easter eggs), never the reference's lyrical expression.

The song writer consumes this as factual/creative raw material — it never paraphrases
or reconstructs the reference's lines.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Confidence = Literal["high", "medium", "low"]
Importance = Literal["high", "medium", "low"]

# How a mined item was classified. FACT/EVENT/ABILITY/etc. are canon-ish knowledge;
# METAPHOR / AUTHOR_INTERPRETATION are the reference composer's own artistic choices
# (kept for context, NOT to be reused); UNCERTAIN is anything the miner is unsure about.
FactCategory = Literal[
    "ability", "event", "relationship", "trait", "title", "symbol",
    "theme", "easter_egg", "author_interpretation", "metaphor", "other",
]


class LoreFact(BaseModel):
    fact: str
    category: FactCategory = "other"
    confidence: Confidence = "medium"


class LoreEvent(BaseModel):
    event: str
    importance: Importance = "medium"
    confidence: Confidence = "medium"


class LoreAbility(BaseModel):
    name: str
    description: str = ""
    confidence: Confidence = "medium"


class LoreRelationship(BaseModel):
    characters: list[str] = Field(default_factory=list)
    relationship: str = ""
    confidence: Confidence = "medium"


class EasterEgg(BaseModel):
    """A specific anime/manga reference embedded in the transcript.

    The whole reason a reference transcript is worth mining: it may name an obscure
    technique, a specific fight or a secondary event the model would not otherwise know.
    """

    reference: str
    interpreted_meaning: str = ""
    related_lore: str = ""
    confidence: Confidence = "medium"


class ReferenceLoreProfile(BaseModel):
    """Structured character knowledge extracted from a reference transcript.

    Stored at ``project.reference_lore_profile_path`` next to the music profile so it
    is only mined once per run (LLM extraction is cached to disk).
    """

    character: str = ""
    facts: list[LoreFact] = Field(default_factory=list)
    events: list[LoreEvent] = Field(default_factory=list)
    abilities: list[LoreAbility] = Field(default_factory=list)
    relationships: list[LoreRelationship] = Field(default_factory=list)
    themes: list[str] = Field(default_factory=list)
    personality_traits: list[str] = Field(default_factory=list)
    easter_eggs: list[EasterEgg] = Field(default_factory=list)
    # The reference composer's OWN interpretations — context only, never to be reused.
    author_interpretations: list[str] = Field(default_factory=list)
    uncertain_items: list[str] = Field(default_factory=list)

    # Provenance — which transcript this was mined from (so a stale cache can be spotted).
    source_title: str = ""

    def is_empty(self) -> bool:
        """True when nothing usable was mined (used to skip the writer's lore block)."""
        return not any([
            self.facts, self.events, self.abilities, self.relationships,
            self.themes, self.personality_traits, self.easter_eggs,
        ])
