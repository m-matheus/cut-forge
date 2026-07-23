"""Metadata service — generate YouTube title, description and tags via Anthropic."""
from __future__ import annotations

import json

from cutforge.integrations import anthropic_client
from cutforge.models.project import VideoProject

_LANG_NAMES = {"en": "English", "es": "Spanish", "pt": "Brazilian Portuguese"}

METADATA_SYSTEM_PROMPT = """\
You write YouTube metadata for an anime music channel (7 Minutoz / Rustage / Sensei Beats style).
Given a song title, character and anime, produce a title, description and tags that maximize
click-through while staying honest about the content (original AI-assisted song over fan-made
AMV footage).

TITLE FORMAT (follow this shape):
  Character - "Song" | Anime Song (Music Video)
  For a matchup: Char A vs Char B - "Song" | Anime1 x Anime2 (Music Video)
Keep it under 100 characters.

DESCRIPTION:
- Open with a one-line hook about the character/song.
- Include the fair-use / fan-made disclaimer.
- End with a call to comment ("Who should we make a track for next?").
- A few relevant hashtags at the end.

TAGS: 8-12 relevant search tags (character, anime, "anime song", "anime rap", "amv", etc.).

OUTPUT FORMAT
Return a single valid JSON object. No markdown fences, no commentary.
{
  "title": "...",
  "description": "...",
  "tags": ["...", "..."]
}
"""


def generate_metadata(project: VideoProject, *, on_log=None) -> dict:
    """Generate and persist script/metadata.json (title, description, tags)."""
    song_title = project.title or project.character
    lang = _LANG_NAMES.get(project.language, "English")
    user_prompt = (
        f"Song title: {song_title}\n"
        f"Character(s): {project.character}\n"
        f"Anime: {project.anime}\n"
        f"Mood: {project.mood}\n\n"
        f"Write the title, description and tags in {lang}. Return only valid JSON."
    )
    data = anthropic_client.complete_json(METADATA_SYSTEM_PROMPT, user_prompt)
    metadata = {
        "title": data.get("title", ""),
        "description": data.get("description", ""),
        "tags": data.get("tags", []),
        "content_type": "music",
        "channel": project.channel_slug,
        "language": project.language,
        "character": project.character,
        "anime": project.anime,
        "mood": project.mood,
    }

    project.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    project.metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Adopt the generated video title onto the project.
    if metadata["title"]:
        project.title = metadata["title"]
        project.save()

    if on_log:
        on_log(f"Metadata saved: {metadata['title']}")
    return metadata
