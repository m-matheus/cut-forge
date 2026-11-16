"""The Suno song package produced by the lyrics-generation step."""
from __future__ import annotations

from pydantic import BaseModel, Field


class CreativeDirection(BaseModel):
    """The brief for a NEW, original song — produced before any lyrics are written.

    Answers "what is the new song we are trying to make?", never "how do we rewrite the
    reference?". Fed the character, the anime, the (optional) reference music + lore
    profiles and the user's topic; its job is to commit to an original angle so the
    writer never falls back to paraphrasing the reference.
    """

    # The sonic world as a Suno style seed (subgenre, energy, vocal, signature
    # instruments, BPM) — chosen by the planner unless the user forced a genre.
    sonic_direction: str = ""
    core_theme: str = ""
    narrative_angle: str = ""
    emotional_arc: str = ""
    hook_concept: str = ""
    # 4–6 concrete SCENES to dramatize (kept under the old field name for cached briefs).
    key_lore_points: list[str] = Field(default_factory=list)
    original_metaphor_direction: str = ""
    delivery_personality: str = ""
    # Section-by-section energy/delivery plan (quiet → build → break → drop → ...) that the
    # writer turns into Suno energy tags.
    energy_map: str = ""
    # A concrete detail planted early and paid off at the end with new meaning.
    callback: str = ""
    # How THIS song opens (cold open, sound first, chant...) — chosen per song so intros
    # stop converging on one template.
    intro_concept: str = ""
    things_to_avoid: list[str] = Field(default_factory=list)
    # Which lyrics mode this brief was planned for ("original" | "structure" | "rewrite").
    # Used to invalidate the cached direction when the user switches modes. Set by the
    # planner, never by the LLM's JSON output.
    planned_mode: str = ""
    # The genre the user forced when this brief was planned ("" = planner's choice).
    # A different forced genre invalidates the cache, like a mode switch.
    planned_genre: str = ""


class SongPackage(BaseModel):
    """A complete, Suno-ready song package. Written to suno_prompt.json + lyrics.txt."""

    title: str
    style: str
    lyrics: str
    # Suno "Exclude Styles" field — kills drift (e.g. "female vocals, heavy autotune, lo-fi").
    # Paste into Suno's Exclude box.
    exclude: str = ""
    # Hard word as written in the lyrics → English phonetic respelling. Only the text pasted
    # into Suno uses the respelling; captions and the release lyrics keep the canonical word.
    pronunciations: dict[str, str] = Field(default_factory=dict)
    # What the deterministic checks left for a human to judge (lyrics_lint_service).
    warnings: list[str] = Field(default_factory=list)
    suno_tips: str = ""
    # Context echoed back for later steps (thumbnail, metadata).
    topic: str = ""
    character: str = ""
    anime: str = ""
