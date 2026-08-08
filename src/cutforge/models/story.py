"""Story-content profile — the shared STORY of reference rap(s).

This is the counterpart to the narrative-structure profile
(``models/structure.py`` — the abstract SHAPE), and is deliberately kept separate from
both the reference *music* profile (``reference_profile.json`` — BPM/flow) and the
reference *lore* profile (``lore.py`` — the character's raw knowledge base).

CRITICAL BOUNDARY — the mirror image of the structure profile's boundary:
- The structure profile stores SHAPE, never CONTENT ("what is said" is forbidden).
- This profile stores CONTENT — the story, the points, the ideas, the imagery that MUST
  be PRESERVED — but never the EXPRESSION: no field holds a rhyme, a verse or the verbatim
  wording of a hook. It captures "WHAT is said" and its order; it throws away "HOW it is
  said" (the phrasing/rhymes). ``hook_concept`` is the IDEA of the reference's hook (so the
  per-song toggle can keep or replace it), never the hook's exact lyric.

The "rewrite the story" mode of ``song_service`` consumes this as the story to preserve
while re-writing every line with new rhymes, rhythm and phrasing. The "max originality"
and "follow structure" modes ignore it.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from cutforge.models.structure import BeatFunction, Confidence


class StoryPoint(BaseModel):
    """One story point to PRESERVE — the idea/claim made here, never its phrasing."""

    order: int = 0                       # 1-based position in the story
    section: str = ""                    # section it lives in (e.g. "Verse 1", "Chorus")
    point: str = ""                      # WHAT is said here — the idea/claim, no rhyme or
    #                                      phrasing (e.g. "he vows to surpass his rival
    #                                      after his humiliating first defeat")
    key_images: list[str] = Field(default_factory=list)  # concrete images/ideas to KEEP,
    #                                      described (not quoted) — e.g. "the broken blade",
    #                                      "the empty throne"
    function: BeatFunction = "escalation"


class StoryContentProfile(BaseModel):
    """The shared STORY synthesized from one or more reference transcripts.

    Stored ONCE per run at ``project.story_content_path``. When several references exist,
    this is the COMMON story they share (a cross-reference synthesis, not a per-reference
    union) — which is why, like the structure profile and unlike the lore profile, it is
    not merged.
    """

    character: str = ""
    logline: str = ""                    # the whole story in 1–2 lines
    story_points: list[StoryPoint] = Field(default_factory=list)  # the ORDERED story to keep
    hook_concept: str = ""               # the IDEA of the reference's hook/refrain, not its
    #                                      exact lyric
    themes: list[str] = Field(default_factory=list)
    key_ideas: list[str] = Field(default_factory=list)  # central messages/ideas shared
    emotional_arc: str = ""              # how the feeling evolves (feelings, no phrasing)
    # What is COMMON across the references (the shared story), plus where they diverge.
    # For a single reference, describes that one story.
    shared_pattern_notes: str = ""

    reference_count: int = 0
    source_titles: list[str] = Field(default_factory=list)  # provenance
    confidence: Confidence = "medium"

    def is_empty(self) -> bool:
        """True when no usable story was extracted (skip the writer's story block)."""
        return not self.story_points
