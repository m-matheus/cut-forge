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
    channel_slug: str = "zenkai-beats"
    language: str = "en"           # en | es | pt

    # Creative context
    topic: str = ""
    character: str = ""
    anime: str = ""
    mood: str = ""

    # Filled by later steps
    title: str = ""
    footage_url: str = ""
    reference_url: str = ""       # YouTube URL of a reference rap (optional inspiration)

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
    def captions_path(self) -> Path:
        return self.audio_dir / "captions.ass"

    @property
    def captions_srt_path(self) -> Path:
        return self.audio_dir / "captions.srt"

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
    def metadata_path(self) -> Path:
        return self.run_dir / "script" / "metadata.json"

    @property
    def premiere_dir(self) -> Path:
        return self.run_dir / "premiere"

    @property
    def premiere_project_path(self) -> Path:
        return self.premiere_dir / "project.xml"

    @property
    def title_card_path(self) -> Path:
        return self.premiere_dir / "title_card.png"

    # --- Reference rap (optional inspiration source) ---
    @property
    def reference_dir(self) -> Path:
        return self.run_dir / "reference"

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
