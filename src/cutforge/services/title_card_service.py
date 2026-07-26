"""Title-card service — render a transparent PNG intro overlay for the Premiere timeline.

Replaces the old ffmpeg ``build_title_intro`` (multi-agent-video-publisher's
compose_music_video.py): a dark scrim with the song title in white and a
``CHANNEL | Anime`` subtitle in the channel's brand color. Here we emit a 1920x1080 RGBA
PNG with alpha so premiere_service can drop it on a V2 track above the footage; Premiere
respects the transparency, so it reads as a title overlay on the opening seconds.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from cutforge.models.project import VideoProject

# Font fallbacks (PIL needs a file path, not a family name).
_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\Montserrat-ExtraBold.ttf",
    r"C:\Windows\Fonts\ariblk.ttf",          # Arial Black
    r"C:\Windows\Fonts\arialbd.ttf",         # Arial Bold
    r"C:\Windows\Fonts\arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def _load_font(size: int):
    from PIL import ImageFont
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    c = color.strip().lstrip("#")
    if len(c) == 6:
        return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    return (0, 229, 255)  # fallback: brand cyan


def _resolve_title(project: VideoProject) -> str:
    """Prefer the clean Suno title (e.g. 'Zero Mana, All Heart') over the formatted
    project.title ('Asta - "Zero Mana, All Heart" | ... (Music Video)')."""
    if project.suno_prompt_path.exists():
        try:
            data = json.loads(project.suno_prompt_path.read_text(encoding="utf-8"))
            if data.get("title"):
                return str(data["title"]).strip()
        except (ValueError, OSError):
            pass
    return (project.title or project.character or "").strip()


def _draw_centered(draw, text, font, cx, y, fill, shadow="black"):
    """Draw ``text`` horizontally centered at ``cx``, top at ``y``, with a drop shadow."""
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    x = cx - w // 2
    draw.text((x + 4, y + 4), text, font=font, fill=shadow)  # shadow
    draw.text((x, y), text, font=font, fill=fill)


def generate_title_card(project: VideoProject, *, on_log=None) -> Path:
    """Render premiere/title_card.png — a transparent 1080p title overlay. Returns its path."""
    from PIL import Image, ImageDraw

    channel = project.channel
    w, h = channel.video.width, channel.video.height
    cx = w // 2

    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Dark scrim (~45%) so the text reads over any footage frame.
    draw.rectangle([0, 0, w, h], fill=(0, 0, 0, 115))

    title = _resolve_title(project)
    subtitle_parts = [channel.name]
    if project.anime:
        subtitle_parts.append(project.anime)
    subtitle = "  |  ".join(subtitle_parts)

    title_font = _load_font(110)
    subtitle_font = _load_font(52)
    accent = _hex_to_rgb(channel.brand.primary_color) + (255,)

    _draw_centered(draw, title, title_font, cx, h // 2 - 90, fill=(255, 255, 255, 255))
    _draw_centered(draw, subtitle, subtitle_font, cx, h // 2 + 50, fill=accent)

    project.premiere_dir.mkdir(parents=True, exist_ok=True)
    out = project.title_card_path
    img.save(str(out), "PNG")
    if on_log:
        on_log(f"Title card saved: {out.name} (\"{title}\" / {subtitle})")
    return out
