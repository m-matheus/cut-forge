"""The Suno song package produced by the lyrics-generation step."""
from __future__ import annotations

from pydantic import BaseModel, Field


class GenreDirection(BaseModel):
    """One suggested genre/vibe direction for a character (from the suggest step)."""

    label: str
    style: str
    why: str = ""


class GenreSuggestions(BaseModel):
    character_read: str = ""
    directions: list[GenreDirection] = Field(default_factory=list)


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
