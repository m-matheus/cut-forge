"""Metadata service — generate YouTube title, description and tags via Anthropic."""
from __future__ import annotations

import json

from cutforge.integrations import anthropic_client
from cutforge.models.project import VideoProject

_LANG_NAMES = {"en": "English", "es": "Spanish", "pt": "Brazilian Portuguese"}

METADATA_SYSTEM_PROMPT = """\
You write YouTube metadata for an anime rap / music channel (Rustage / None Like Joshua /
Daddyphatsnaps / 7 Minutoz style). Given a song title, character and anime, produce a title,
description and tags that maximize click-through while staying honest about the content
(original AI-assisted song over fan-made AMV footage).

TITLE FORMAT — character name FIRST, in CAPS, so it wins the search box.
People search "gojo rap", not the song title, so lead with the character.
  CHARACTER RAP | "Song" | Enkai [Anime]
  For a matchup: CHAR A VS CHAR B RAP | "Song" | Enkai [Anime1 x Anime2]
  For a group/cypher: ANIME RAP CYPHER | "Song" | Enkai [Anime]
Use RAP for a single-artist track, CYPHER for a multi-character group track.
Always keep the channel tag "Enkai" and put the anime name in [square brackets] at the end.
Keep the quoted song title short (2-4 words). Keep the whole title under 100 characters.
Examples of the shape (do not copy verbatim):
  GOJO RAP | "Six Eyes" | Enkai [Jujutsu Kaisen]
  NARUTO RAP | "Hokage" | Enkai [Naruto]
  SANJI VS ZORO RAP | "Rivals" | Enkai [One Piece]

DESCRIPTION — keep it tight (3-5 short lines):
- Open with a one-line hook about the character/song.
- Include the fair-use / fan-made disclaimer (footage used transformatively; characters
  belong to their studio; song is an original AI-assisted tribute).
- End with a call to comment ("Who should we make a track for next?").
- A handful of relevant hashtags on the last line.

TAGS: 12-15 SEO-heavy search tags. Mix bare and compound forms, e.g. for a character:
  <character>, <character> rap, <character> song, <character> amv,
  <anime>, <anime> rap, anime rap, anime song, anime music video, amv,
  rap, hip hop, anime cypher
Use the actual character/anime names supplied. Lowercase, no hashtags in tags.

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
