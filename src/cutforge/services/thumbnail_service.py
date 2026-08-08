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
Rustage, 7 Minutoz, Ishida Music) design their thumbnails — this is the benchmark.

ART STYLE (critical):
Clean, ultra-detailed 2D anime illustration. Sharp outlines, cel-shading with dramatic
lighting. NOT 3D render, NOT realistic, NOT photorealistic. The quality must match
professional anime key art — like an official illustration from the anime studio itself,
but with maximum cinematic energy.

CHARACTER (primary focus):
The featured character fills 70-80% of the frame. Close shot: face, neck, and upper
chest visible. The face must be FRONT-FACING or at a slight dramatic angle toward the
viewer — eyes must make direct, intense contact with the camera. Render the character
in their most powerful, iconic form with their exact signature costume and accessories.
Expression: powerful and iconic, matching both the character's personality and the
song's mood — whether that's an arrogant smirk, stoic intensity, fierce determination,
sorrowful resolve, or a battle-ready glare. Let the character's nature guide the face.
Eyes are vivid and expressive; if the character has a signature power or energy color,
their eyes should reflect or glow with it.

LIGHTING (critical):
Extreme rim/edge lighting outlines the character in their signature power color, as if
their body emits energy. The character glows. Face lit dramatically from below or the
side. Strong contrast between the lit face and the character's outfit silhouette.

BACKGROUND (secondary, supporting):
Two-tone or split-color background preferred — two contrasting colors drawn from the
character's own palette, divided diagonally or with an energy explosion at the center.
Add energy particles, lightning bolts, speed lines, or abstract power bursts that match
the character's abilities or the song's mood. Background MUST NOT compete with the
character — use a strong vignette/dark blur toward the edges.

COLOR GRADING:
Hyper-saturated, electric, neon-level vibrancy. Maximum contrast. Colors must look
almost unrealistically vivid — like a phone wallpaper people stop to stare at. Pick the
character's two most iconic colors and use them as the dominant palette.

COMPOSITION:
Album cover energy. The character commands the frame with an aura that fully matches
the song's mood and their own personality. Every pixel should make the viewer stop
scrolling.

NO watermarks, NO channel logos, NO character names, NO text on the image.
Only include a genre badge if one is specified below.
""".strip()


def _build_request(project: VideoProject, genre_badge: str | None) -> str:
    sections = [
        ("Featured character(s) — render their exact canonical design (hair, eyes, "
         "outfit, accessories — do NOT change or omit any signature feature)",
         project.character),
        ("Anime / Series", project.anime),
        ("Song vibe / mood", project.mood or "dark, powerful, cinematic"),
    ]
    context = "\n\n".join(f"{label}:\n{value}" for label, value in sections if value)

    badge_block = ""
    if genre_badge:
        badge_block = (
            f"GENRE BADGE (bottom-left corner), Sensei Beats style: a small 'ENKAI' "
            f"credit line above a large bold '{genre_badge.upper()}' in a heavy condensed "
            f"impact font, both on a dark semi-transparent rectangle bleeding into the "
            f"bottom-left edges."
        )

    palette_hint = (
        f"The character's signature colors and {project.anime}'s official color palette "
        f"should dominate the composition — use neon/electric versions of those exact colors."
        if project.anime else
        "Use hyper-saturated, neon/electric versions of the character's signature colors."
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
