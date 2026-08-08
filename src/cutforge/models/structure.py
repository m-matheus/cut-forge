"""Narrative structure profile — the proven story SKELETON of reference rap(s).

This is deliberately kept SEPARATE from both the reference *music* profile
(``reference_service`` / ``reference_profile.json`` — BPM/flow, the sonic DNA) and the
reference *lore* profile (``lore.py`` — character KNOWLEDGE). It carries only the SHAPE
of the song: the ordered story beats, the section arrangement, where the hook lands, the
emotional arc and the flow/cadence pattern.

CRITICAL BOUNDARY: this profile stores STRUCTURE, never EXPRESSION. There is no field
anywhere that holds a lyric, a phrase, a rhyme or a hook wording — only abstract slot and
shape descriptors. When multiple references about the same character are given, the
profile describes the COMMON skeleton they share (the "proven formula"), not any single
song's lines.

The "follow structure" mode of ``song_service`` consumes this as the blueprint to FOLLOW
while inventing 100% original wording. The default "max originality" mode ignores it.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Confidence = Literal["high", "medium", "low"]
Intensity = Literal["high", "medium", "low"]

# The narrative role a beat plays in the skeleton. Purely structural — never phrasing.
BeatFunction = Literal[
    "setup", "escalation", "turn", "climax", "resolution", "hook_anchor", "callback",
]


class NarrativeBeat(BaseModel):
    """One abstract story/emotional beat in the skeleton — SHAPE only, never phrasing."""

    order: int = 0                       # 1-based position in the skeleton
    section: str = ""                    # section it lives in (e.g. "Verse 1", "Chorus")
    beat: str = ""                       # abstract beat, no lyric (e.g. "establish the
    #                                      character at their lowest, isolated and doubted")
    function: BeatFunction = "escalation"
    maps_to_lore: str = ""               # KIND of lore that fills this slot (a slot label,
    #                                      e.g. "signature ability" — never a phrase)
    intensity: Intensity = "medium"      # energy at this beat


class NarrativeStructureProfile(BaseModel):
    """The proven story skeleton synthesized from one or more reference transcripts.

    Stored ONCE per run at ``project.narrative_structure_path``. When several references
    exist, this is the SHARED skeleton across them (a cross-reference synthesis, not a
    per-reference union) — which is why, unlike the lore profile, it is not merged.
    """

    character: str = ""
    overall_shape: str = ""              # one-line summary of the skeleton
    section_arrangement: list[str] = Field(default_factory=list)  # ordered section tags
    beats: list[NarrativeBeat] = Field(default_factory=list)
    hook_placement: str = ""             # where the hook lands, how often, its function
    emotional_arc: str = ""              # the proven arc (feelings only, no phrasing)
    flow_cadence_notes: str = ""         # cadence pattern (where density rises/falls)
    # What is COMMON across the references (the "proven formula"), plus where they diverge.
    # For a single reference, describes that one skeleton.
    shared_pattern_notes: str = ""

    reference_count: int = 0
    source_titles: list[str] = Field(default_factory=list)  # provenance
    confidence: Confidence = "medium"

    def is_empty(self) -> bool:
        """True when no usable skeleton was extracted (skip the writer's structure block)."""
        return not any([self.beats, self.section_arrangement])
