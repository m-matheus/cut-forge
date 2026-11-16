"""Metadata service — generate YouTube title, description and tags via Anthropic."""
from __future__ import annotations

import json

from cutforge.integrations import anthropic_client
from cutforge.models.project import VideoProject
from cutforge.services import catalog_service

_LANG_NAMES = {"en": "English", "es": "Spanish", "pt": "Brazilian Portuguese"}
_SPOTIFY_LINE = {"en": "🎧 Listen on Spotify:", "es": "🎧 Escúchala en Spotify:",
                 "pt": "🎧 Ouça no Spotify:"}

METADATA_SYSTEM_PROMPT = """\
You are the metadata generator for "Enkai" (宴開), a YouTube channel that makes
ORIGINAL, AI-ASSISTED anime RAP songs in the style of Rustage, None Like Joshua,
Daddyphatsnaps, and 7 Minutoz. The footage is clips taken from the anime itself.

Your job: given a track's details, output YouTube metadata as JSON.

OUTPUT FORMAT — return ONLY raw JSON, no markdown fences, no prose:
{
  "title": "...",
  "description": "...",
  "tags": ["...", "..."]
}

LANGUAGE:
Write ALL fields in the requested language: "en", "es", or "pt" (Brazilian
Portuguese). If none is given, default to "en". Translate the words but NEVER
restructure — the 4-part description order and the CREDITS block are
identical in every language.

ABSOLUTE RULES:
- The word "AMV" must NEVER appear anywhere (title, description, or tags), in any
  language. Always frame the product as an "anime rap", "anime song", or
  "anime edit" — never an AMV.
- This is an ORIGINAL song. Everything is made in-house: all credits are "Enkai".
- Be honest about AI: it ASSISTED production; the creative direction is Enkai's.

────────────────────────────────────────
TITLE FORMAT:
"{CHARACTER IN ALL CAPS} RAP | \\"{Song}\\" | Enkai [{Anime}]"
- Character name leads, in ALL CAPS.
- Always include the word "Enkai" and the anime in [square brackets].
- Keep the quoted song title short (2-4 words). Keep the whole title under 100 chars.
- For a matchup use "CHAR A VS CHAR B RAP"; for a group track use "ANIME RAP CYPHER".
- Example: GOJO RAP | "Six Eyes" | Enkai [Jujutsu Kaisen]

────────────────────────────────────────
DESCRIPTION — exactly these 4 parts, in this order:

PART 1 — Subscribe hook (ONE short sentence):
  Warmly thank the returning viewer for supporting Enkai and ask them to
  subscribe so they don't miss the next drop. Sound like the creator talking
  casually to a fan, not a formal announcement.
  - es: lean into "gracias por seguir con Enkai… suscríbete".
  - pt: use natural BR-community voice, e.g. "valeu por estar aqui com a Enkai…
        se inscreve".

PART 2 — Character / song line (ONE sentence):
  Say this is Enkai's original anime rap for {character} from {anime}, and
  capture the vibe of THIS song (e.g. "dark and vengeful", written from the
  character's side of the story) — from {mood} if given, otherwise from {theme}
  and {hook}. Describe the song you were given; never invent a different one.

PART 3 — CREDITS (label-style block, identical structure in all
  languages, wrapped in the ━ separators). All roles are Enkai:
  ━━━━━━━━━━━━━━━━━━━━
  🎤 CREDITS
  Song: "{song}"
  Character: {character} ({anime})
  Written by: Enkai
  Produced by: Enkai
  Mixed & mastered by: Enkai
  
  ━━━━━━━━━━━━━━━━━━━━
  (Translate only the labels — e.g. es "Escrito por / Producido por /
  Mezcla y máster por / Voces"; pt "Escrito por / Produzido por /
  Mixagem e masterização por / Vocais". Keep every role value as "Enkai".)

PART 4 — AI-assist + fair-use disclaimer (keep it tight, plain, honest, ~2 sentences):
  State that AI was used to ASSIST the production of this original song
  (writing, sound, and visuals), but every creative call is Enkai's. Then note
  it is fan-made and all anime footage belongs to its original studio and
  creators, shared under fair use as a transformative, non-commercial tribute.
  - es: use "hecho por fans".  - pt: use "feito por fãs".

CLOSE (after the 4 parts):
  A) A call to comment: "Who should Enkai rap next? Drop the character + anime
     in the comments 👇" (translated per language).
  B) A hashtag line (see tags rules; render as #hashtags here too).

────────────────────────────────────────
TAGS (the "tags" array, 6–8 items max):
  - Character name leads.
  - Always include: "Enkai", "anime rap", "anime song", "anime music", "rap".
  - Add character-specific: "{character} rap", "{character} song".
  - NEVER include "amv".
  - Lowercase, no hashtags in the tags array.
"""


def _song_theme(project: VideoProject) -> str:
    """The brief's core theme, so the description matches the song actually written."""
    if project.creative_direction_path.exists():
        data = json.loads(project.creative_direction_path.read_text(encoding="utf-8"))
        return data.get("core_theme", "")
    return ""


def _song_hook(project: VideoProject) -> str:
    """The first lines of the chorus, joined on one line ("" without lyrics)."""
    if not project.lyrics_path.exists():
        return ""
    lyrics = project.lyrics_path.read_text(encoding="utf-8")
    hook = catalog_service.sung_text(catalog_service.section(lyrics, "chorus", "hook",
                                                             "refrain"))
    return " / ".join(hook.splitlines()[:4])


def generate_metadata(project: VideoProject, *, on_log=None) -> dict:
    """Generate and persist script/metadata.json (title, description, tags)."""
    # project.title is the YouTube title (set below); the song title comes from the
    # latest lyrics step, so a regenerated song never inherits an older song's name.
    song_title = project.song_title
    lang = _LANG_NAMES.get(project.language, "English")
    user_prompt = (
        f"song: {song_title}\n"
        f"character: {project.character}\n"
        f"anime: {project.anime}\n"
        f"mood: {project.mood}\n"
        f"theme: {_song_theme(project)}\n"
        f"hook: {_song_hook(project)}\n"
        f"lang: {project.language}\n\n"
        f"Write the title, description and tags in {lang}. Return only valid JSON."
    )
    data = anthropic_client.complete_json(METADATA_SYSTEM_PROMPT, user_prompt)
    description = data.get("description", "")
    spotify = project.spotify_track_link or project.channel.spotify.artist_link
    if spotify:
        # First line: the part of the description YouTube shows above "...more".
        description = f"{_SPOTIFY_LINE.get(project.language, _SPOTIFY_LINE['en'])} {spotify}\n\n{description}"
    metadata = {
        "title": data.get("title", ""),
        "description": description,
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
