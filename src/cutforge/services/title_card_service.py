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

# Font fallbacks (PIL needs a file path, not a family name). Prefer the repo's bundled
# Montserrat ExtraBold so the card looks identical on any machine, then system fonts.
_BUNDLED_FONT = Path(__file__).resolve().parents[1] / "assets" / "fonts" / "Montserrat-ExtraBold.ttf"
_FONT_CANDIDATES = [
    str(_BUNDLED_FONT),
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


def _fit_font(draw, text, start_size, max_width):
    """Return the largest font (<= start_size) whose rendered ``text`` fits ``max_width``."""
    from PIL import ImageFont
    size = start_size
    while size > 24:
        font = _load_font(size)
        bbox = draw.textbbox((0, 0), text, font=font)
        if (bbox[2] - bbox[0]) <= max_width:
            return font
        size -= 4
    return _load_font(size)


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

    # Shrink to fit ~85% of the frame width so a long song title never overflows.
    max_text_w = int(w * 0.85)
    title_font = _fit_font(draw, title, 110, max_text_w)
    subtitle_font = _fit_font(draw, subtitle, 52, max_text_w)
    accent = _hex_to_rgb(channel.brand.primary_color) + (255,)

    _draw_centered(draw, title, title_font, cx, h // 2 - 90, fill=(255, 255, 255, 255))
    _draw_centered(draw, subtitle, subtitle_font, cx, h // 2 + 50, fill=accent)

    project.premiere_dir.mkdir(parents=True, exist_ok=True)
    out = project.title_card_path
    img.save(str(out), "PNG")
    if on_log:
        on_log(f"Title card saved: {out.name} (\"{title}\" / {subtitle})")
    return out
