"""The Suno song package produced by the lyrics-generation step."""
from __future__ import annotations

from pydantic import BaseModel, Field


class GenreDirection(BaseModel):
    """One suggested genre/vibe direction for a character (from the suggest step)."""

    label: str
    style: str
    why: str = ""
    ref_index: int = 0  # which reference this direction was derived from


class GenreSuggestions(BaseModel):
    character_read: str = ""
    directions: list[GenreDirection] = Field(default_factory=list)


class CreativeDirection(BaseModel):
    """The brief for a NEW, original song — produced before any lyrics are written.

    Answers "what is the new song we are trying to make?", never "how do we rewrite the
    reference?". Fed the character, the anime, the (optional) reference music + lore
    profiles and the user's topic; its job is to commit to an original angle so the
    writer never falls back to paraphrasing the reference.
    """

    core_theme: str = ""
    narrative_angle: str = ""
    emotional_arc: str = ""
    hook_concept: str = ""
    key_lore_points: list[str] = Field(default_factory=list)
    original_metaphor_direction: str = ""
    delivery_personality: str = ""
    things_to_avoid: list[str] = Field(default_factory=list)
    # Which lyrics mode this brief was planned for ("original" | "structure" | "rewrite").
    # Used to invalidate the cached direction when the user switches modes. Set by the
    # planner, never by the LLM's JSON output.
    planned_mode: str = ""


class SongPackage(BaseModel):
    """A complete, Suno-ready song package. Written to suno_prompt.json + lyrics.txt."""

    title: str
    style: str
    lyrics: str
    # Suno "Exclude Styles" field — kills drift (e.g. "singing, melodic vocals, auto-tune"
    # keeps a rap track from turning into sung mush). Paste into Suno's Exclude box.
    exclude: str = ""
    suno_tips: str = ""
    # Context echoed back for later steps (thumbnail, metadata).
    topic: str = ""
    character: str = ""
    anime: str = ""
