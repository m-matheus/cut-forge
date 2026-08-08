"""VideoProject — the central handle for one run.

A "run" is one music-video project living under ``output/{YYYYMMDD}-{slug}/``. This
object knows the channel, the creative context (character/anime/mood/language) and
resolves every file path in the run layout, so services never hard-code paths.

The lightweight metadata (character, anime, mood, language, title...) is persisted to
``project.json`` at the run root so the app can reopen a run across restarts.
"""
from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from cutforge.config.channels import Channel, load_channel
from cutforge.config.settings import get_settings

PROJECT_FILE = "project.json"


class VideoProject(BaseModel):
    """Persisted run context + path resolver."""

    run_id: str                    # folder name, e.g. "20260722-jinwoo-shadow-monarch"
    channel_slug: str = "enkai"
    language: str = "en"           # en | es | pt

    # Creative context
    topic: str = ""
    character: str = ""
    anime: str = ""
    mood: str = ""

    # Filled by later steps
    title: str = ""
    footage_url: str = ""
    reference_urls: list[str] = Field(default_factory=list)
    # DEPRECATED — kept so old project.json files still load. Migrated to
    # reference_urls on first load via model_post_init.
    reference_url: str = ""
    # DEPRECATED — kept only so old project.json files still load.
    content_blend: str = "rhythm"

    def model_post_init(self, __context) -> None:
        # Migrate legacy single reference_url → reference_urls list.
        if self.reference_url and not self.reference_urls:
            self.reference_urls = [self.reference_url]

    # --- Channel (not serialized; resolved on demand) ---
    @property
    def channel(self) -> Channel:
        return load_channel(self.channel_slug)

    # --- Paths ---
    @property
    def run_dir(self) -> Path:
        return get_settings().output_dir / self.run_id

    @property
    def audio_dir(self) -> Path:
        return self.run_dir / "audio"

    @property
    def lyrics_path(self) -> Path:
        return self.run_dir / "lyrics.txt"

    @property
    def suno_prompt_path(self) -> Path:
        return self.run_dir / "suno_prompt.json"

    @property
    def track_path(self) -> Path:
        return self.audio_dir / "track.mp3"

    @property
    def whisper_cache_path(self) -> Path:
        return self.audio_dir / "whisper_transcript.json"

    @property
    def alignment_path(self) -> Path:
        return self.audio_dir / "lyrics_alignment.json"

    @property
    def captions_srt_path(self) -> Path:
        return self.audio_dir / "captions.srt"

    @property
    def premiere_transcript_path(self) -> Path:
        # Adobe Premiere transcript JSON (schema v1.0.0) for Text panel > Import transcript.
        return self.audio_dir / "premiere_transcript.json"

    @property
    def footage_dir(self) -> Path:
        return self.run_dir / "footage"

    @property
    def footage_path(self) -> Path:
        return self.footage_dir / "source.mp4"

    @property
    def thumbnail_dir(self) -> Path:
        return self.run_dir / "thumbnail"

    @property
    def thumbnail_path(self) -> Path:
        return self.thumbnail_dir / "thumbnail.jpg"

    @property
    def thumbnail_refs_dir(self) -> Path:
        # Optional style/composition reference images uploaded for this run.
        return self.thumbnail_dir / "refs"

    def thumbnail_ref_paths(self) -> list[Path]:
        """Reference images for this run's thumbnail, sorted; empty if none."""
        refs_dir = self.thumbnail_refs_dir
        if not refs_dir.exists():
            return []
        exts = {".png", ".jpg", ".jpeg", ".webp"}
        return sorted(p for p in refs_dir.iterdir()
                      if p.is_file() and p.suffix.lower() in exts)

    @property
    def metadata_path(self) -> Path:
        return self.run_dir / "script" / "metadata.json"

    @property
    def premiere_dir(self) -> Path:
        return self.run_dir / "premiere"

    @property
    def premiere_project_path(self) -> Path:
        return self.premiere_dir / f"{self.run_id}.xml"

    @property
    def title_card_path(self) -> Path:
        return self.premiere_dir / "title_card.png"

    # --- Reference rap(s) (optional inspiration sources) ---
    @property
    def reference_dir(self) -> Path:
        return self.run_dir / "reference"

    def ref_dir(self, index: int) -> Path:
        return self.reference_dir / str(index)

    def ref_audio_path(self, index: int) -> Path:
        return self.ref_dir(index) / "reference.mp3"

    def ref_whisper_path(self, index: int) -> Path:
        return self.ref_dir(index) / "reference_whisper.json"

    def ref_lyrics_path(self, index: int) -> Path:
        return self.ref_dir(index) / "reference_lyrics.txt"

    def ref_rhythm_path(self, index: int) -> Path:
        return self.ref_dir(index) / "reference_rhythm.json"

    def ref_profile_path(self, index: int) -> Path:
        return self.ref_dir(index) / "reference_profile.json"

    def ref_lore_profile_path(self, index: int) -> Path:
        return self.ref_dir(index) / "reference_lore_profile.json"

    # --- Legacy single-reference paths (backward compat — old flat layout) ---
    @property
    def reference_audio_path(self) -> Path:
        return self.reference_dir / "reference.mp3"

    @property
    def reference_whisper_path(self) -> Path:
        return self.reference_dir / "reference_whisper.json"

    @property
    def reference_lyrics_path(self) -> Path:
        return self.reference_dir / "reference_lyrics.txt"

    @property
    def reference_rhythm_path(self) -> Path:
        return self.reference_dir / "reference_rhythm.json"

    @property
    def reference_profile_path(self) -> Path:
        return self.reference_dir / "reference_profile.json"

    @property
    def reference_lore_profile_path(self) -> Path:
        return self.reference_dir / "reference_lore_profile.json"

    @property
    def creative_direction_path(self) -> Path:
        # The original-song brief produced before lyrics are written.
        return self.run_dir / "creative_direction.json"

    @property
    def narrative_structure_path(self) -> Path:
        # The proven story skeleton synthesized from the reference(s) — "follow structure"
        # mode only. One blueprint per run (not per-reference).
        return self.run_dir / "narrative_structure.json"

    @property
    def story_content_path(self) -> Path:
        # The shared STORY synthesized from the reference(s) — "rewrite the story" mode
        # only. One profile per run (not per-reference), like narrative_structure.
        return self.run_dir / "story_content.json"

    # --- Persistence ---
    def save(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / PROJECT_FILE).write_text(
            self.model_dump_json(indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, run_id: str) -> "VideoProject":
        path = get_settings().output_dir / run_id / PROJECT_FILE
        if not path.exists():
            raise FileNotFoundError(f"No project.json for run '{run_id}' at {path}")
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def create(cls, run_id: str, **kwargs) -> "VideoProject":
        project = cls(run_id=run_id, **kwargs)
        project.save()
        return project


def list_runs() -> list[str]:
    """Return run_ids (folder names) that contain a project.json, newest first."""
    base = get_settings().output_dir
    if not base.exists():
        return []
    runs = [c.name for c in base.iterdir() if (c / PROJECT_FILE).exists()]
    return sorted(runs, reverse=True)


def list_runs_summary() -> list[dict]:
    """Return lightweight dicts with display info for each run, newest first."""
    result = []
    for run_id in list_runs():
        try:
            p = VideoProject.load(run_id)
            result.append({
                "run_id": run_id,
                "character": p.character,
                "anime": p.anime,
                "language": p.language,
                "title": p.title,
            })
        except Exception:
            result.append({"run_id": run_id, "character": "", "anime": "", "language": "", "title": ""})
    return result
