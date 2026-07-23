"""Thumbnail service — music-video thumbnail via the OpenAI Responses API.

MUSIC_BASELINE ported from the old ``generate_thumbnail.py``: the character is the hero,
album-cover energy, background carries the song's vibe. Fully AI-generated in one shot
(no PIL compositing) — resized/cropped to 1280x720 for YouTube.
"""
from __future__ import annotations

from cutforge.integrations import openai_images
from cutforge.models.project import VideoProject

MUSIC_BASELINE = """
Create a high-CTR anime music video thumbnail. Study how top channels (Sensei Beats,
Rustage, 7 Minutoz) design their thumbnails — this is the benchmark.

CHARACTER (primary focus):
The featured character fills 70-80% of the frame. Close shot: face, neck, and upper
chest. The face must be FRONT-FACING or at a slight dramatic angle toward the viewer —
the eyes must make direct, intense contact with the camera. Render the character in
their most powerful, iconic form. Maximum detail, sharp edges, cinematic anime key art.

BACKGROUND (secondary, supporting):
Simple but atmospheric — energy aura, particles, color burst, or an abstract
environmental effect tied to the character's power. The background MUST NOT compete
with the character. Dark edges / subtle vignette to push the character forward.

COLOR GRADING:
One dominant color tied to the character's signature palette. Highly saturated,
cinematic, contrasty. Instantly recognizable by color alone.

STYLE:
Album cover energy. Dramatic rim lighting and glow on the character, lit from behind
with their signature color. Face is detailed and expressive.

NO watermarks, NO channel logos, NO character names on the image.
Only include a genre badge if one is specified below.
""".strip()


def _build_request(project: VideoProject, genre_badge: str | None) -> str:
    sections = [
        ("Featured character(s)", project.character),
        ("Anime / Series", project.anime),
        ("Song vibe / mood", project.mood or "dark, powerful, cinematic"),
    ]
    context = "\n\n".join(f"{label}:\n{value}" for label, value in sections if value)

    badge_block = ""
    if genre_badge:
        badge_block = (
            f"GENRE BADGE (bottom-left corner), Sensei Beats style: a small 'ZENKAI BEATS' "
            f"credit line above a large bold '{genre_badge.upper()}' in a heavy condensed "
            f"impact font, both on a dark semi-transparent rectangle bleeding into the "
            f"bottom-left edges."
        )

    palette_hint = (
        f"Draw the background color palette from {project.anime}'s official art."
        if project.anime else ""
    )

    return "\n\n".join(filter(None, [
        "Create a thumbnail for an anime music video (character key art, album-cover energy).",
        context,
        MUSIC_BASELINE,
        palette_hint,
        badge_block,
    ])).strip()


def generate_thumbnail(project: VideoProject, *, genre_badge: str | None = None,
                       on_log=None) -> str:
    """Generate the 16:9 music thumbnail and save it as thumbnail/thumbnail.jpg."""
    if not project.character:
        raise ValueError("Thumbnail needs a character — set it on the project first.")

    request = _build_request(project, genre_badge)
    raw_path = project.thumbnail_dir / "raw.png"
    openai_images.generate_image(request, raw_path, portrait=False, on_log=on_log)

    from PIL import Image
    Image.open(raw_path).convert("RGB").resize((1280, 720), Image.LANCZOS).save(
        str(project.thumbnail_path), "JPEG", quality=93
    )
    if on_log:
        size_kb = project.thumbnail_path.stat().st_size // 1024
        on_log(f"Thumbnail saved: {project.thumbnail_path.name} ({size_kb} KB)")
    return str(project.thumbnail_path)
