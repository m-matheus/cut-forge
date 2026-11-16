"""The "listen on Spotify" call over the intro: a QR code clip with alpha for Premiere.

Ported from LoreForge's brand_service: one full-frame transparent still (so the timeline
needs no motion parameters — the part of FCP7 import that always behaves), turned into a
PNG-in-MOV clip with alpha that fades in and out. It opens the song once the run has its
``spotify_track_url``, else the channel's artist profile.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import segno
from PIL import Image, ImageColor, ImageDraw

from cutforge.models.project import VideoProject

FADE = 0.6   # seconds

# PIL needs a font file, not a family name: the bundled Montserrat ExtraBold first, so the
# notice looks the same on any machine, then system fallbacks.
_BUNDLED_FONT = Path(__file__).resolve().parents[1] / "assets" / "fonts" / "Montserrat-ExtraBold.ttf"
_FONT_CANDIDATES = [
    str(_BUNDLED_FONT),
    r"C:\Windows\Fonts\ariblk.ttf",          # Arial Black
    r"C:\Windows\Fonts\arialbd.ttf",         # Arial Bold
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


def target_url(project: VideoProject) -> str:
    """The track link if set, else the artist profile, else ""."""
    return project.spotify_track_link or project.channel.spotify.artist_link


def notice_frame(project: VideoProject, url: str) -> Image.Image:
    """Full-frame RGBA still: dark plate, black-on-white QR code, call text beneath it."""
    ch = project.channel
    spec = ch.spotify
    w, h = ch.video.width, ch.video.height
    matrix = segno.make(url, error="m").matrix
    quiet = 2                                         # white modules around the code
    module = max(2, int(h * spec.qr_height_frac / (len(matrix) + 2 * quiet)))
    code = module * (len(matrix) + 2 * quiet)
    pad = module * 2
    call = spec.notice_text.get(project.language) or spec.notice_text.get("en", "")
    artist = (spec.artist_name or ch.name).upper()
    call_font = _load_font(max(12, int(code * 0.105)))
    artist_font = _load_font(max(10, int(code * 0.08)))
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    text_w = max(probe.textlength(call, font=call_font), probe.textlength(artist, font=artist_font))
    plate_w = int(max(code + 2 * pad, text_w + 4 * pad))
    plate_h = int(code + 2 * pad + call_font.size * 1.35 + artist_font.size * 1.4)

    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    x0 = int(spec.qr_center[0] * w - plate_w / 2)
    y0 = int(spec.qr_center[1] * h - plate_h / 2)
    x0 = min(max(0, x0), w - plate_w)
    y0 = min(max(0, y0), h - plate_h)
    night = ImageColor.getrgb(ch.brand.background_color)
    draw.rounded_rectangle([x0, y0, x0 + plate_w, y0 + plate_h], radius=pad,
                           fill=night + (235,))
    qx, qy = x0 + (plate_w - code) // 2, y0 + pad
    draw.rounded_rectangle([qx, qy, qx + code, qy + code], radius=module, fill=(255, 255, 255, 255))
    for r, row in enumerate(matrix):
        for c, on in enumerate(row):
            if on:
                x, y = qx + (c + quiet) * module, qy + (r + quiet) * module
                draw.rectangle([x, y, x + module - 1, y + module - 1], fill=(0, 0, 0, 255))
    ty = qy + code + module * 1.5
    cx = x0 + plate_w / 2
    draw.text((cx, ty), call, font=call_font, anchor="mt",
              fill=ImageColor.getrgb(ch.brand.primary_color) + (255,))
    draw.text((cx, ty + call_font.size * 1.35), artist, font=artist_font, anchor="mt",
              fill=ImageColor.getrgb(ch.brand.text_color) + (255,))
    return img


def build_notice(project: VideoProject, url: str, duration: float, out: Path) -> Path:
    """The still as a PNG-in-MOV clip with alpha, fading in and out over ``duration``."""
    out.parent.mkdir(parents=True, exist_ok=True)
    still = out.with_suffix(".png")
    notice_frame(project, url).save(still)
    fade = min(FADE, duration / 3)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-loop", "1", "-framerate", str(project.channel.video.fps), "-t", f"{duration:.3f}",
           "-i", str(still),
           "-vf", f"fade=t=in:st=0:d={fade}:alpha=1,"
                  f"fade=t=out:st={duration - fade:.3f}:d={fade}:alpha=1",
           "-c:v", "png", "-pix_fmt", "rgba", str(out)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed building the Spotify QR clip: {result.stderr[-500:]}")
    return out
