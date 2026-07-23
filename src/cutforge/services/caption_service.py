"""Caption service — build karaoke ASS captions from a lyric alignment.

Ports ``build_ass_karaoke`` from the old ``generate_captions.py``: one Dialogue event
per lyric line with per-word ``\\k`` highlight tags, clamped to a sane range so bad
timestamps can't freeze or pre-fill the highlight.
"""
from __future__ import annotations

import os

from cutforge.models.alignment import Alignment, LyricLine
from cutforge.models.project import VideoProject

# Named color presets (ASS format: &H00BBGGRR)
_COLOR_PRESETS = {
    "yellow": "&H0000FFFF",
    "white": "&H00FFFFFF",
    "cyan": "&H00FFFF00",
    "red": "&H000000FF",
    "orange": "&H000080FF",
}

MIN_K_CS = 6     # 0.06s floor — no instant flash / pre-fill
MAX_K_CS = 250   # 2.5s ceiling — no multi-second freeze on one word


def hex_to_ass(color_str: str) -> str:
    """Convert a color name or #RRGGBB hex to ASS &H00BBGGRR format."""
    c = color_str.strip().lower()
    if c in _COLOR_PRESETS:
        return _COLOR_PRESETS[c]
    c = c.lstrip("#")
    if len(c) == 6:
        r, g, b = c[0:2], c[2:4], c[4:6]
        return f"&H00{b.upper()}{g.upper()}{r.upper()}"
    raise ValueError(
        f"Unknown color: {color_str!r}. Use a name "
        f"(yellow/white/cyan/red/orange) or #RRGGBB hex."
    )


def seconds_to_ass_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    cs = round((s - int(s)) * 100)
    return f"{h}:{m:02d}:{int(s):02d}.{cs:02d}"


def resolve_caption_font() -> str:
    """Prefer Montserrat ExtraBold -> Arial Black -> Arial."""
    if os.path.exists(r"C:\Windows\Fonts\Montserrat-ExtraBold.ttf"):
        return "Montserrat ExtraBold"
    if os.path.exists(r"C:\Windows\Fonts\ariblk.ttf"):
        return "Arial Black"
    return "Arial"


def build_ass_karaoke(lines: list[LyricLine], *, color: str = "cyan",
                      unsung: str = "white") -> str:
    """Build a landscape karaoke ASS with per-word \\k highlight tags."""
    play_res_x, play_res_y = 1920, 1080
    font_size = 84
    font_name = resolve_caption_font()

    sung = hex_to_ass(color)
    not_sung = hex_to_ass(unsung)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {play_res_x}
PlayResY: {play_res_y}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Karaoke,{font_name},{font_size},{sung},{not_sung},&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,5,3,2,80,80,90,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"""

    out = [header]
    for line in lines:
        words = line.words
        if not words:
            continue
        line_start = line.start
        parts = []
        elapsed_cs = 0
        for i, w in enumerate(words):
            nxt = words[i + 1].start if i + 1 < len(words) else w.end
            dur_cs = round((nxt - w.start) * 100)
            dur_cs = max(MIN_K_CS, min(dur_cs, MAX_K_CS))
            elapsed_cs += dur_cs
            text = w.word.upper().replace("{", "").replace("}", "")
            parts.append(f"{{\\k{dur_cs}}}{text} ")
        text_line = "".join(parts).rstrip()
        line_end = line_start + elapsed_cs / 100.0
        out.append(
            f"Dialogue: 0,{seconds_to_ass_time(line_start)},{seconds_to_ass_time(line_end)},"
            f"Karaoke,,0,0,0,,{text_line}"
        )
    return "\n".join(out) + "\n"


def generate_captions(project: VideoProject, alignment: Alignment | None = None, *,
                      color: str | None = None, on_log=None) -> str:
    """Write karaoke captions.ass for the run. Returns the ASS text.

    ``color`` defaults to the channel's mood->color mapping for this project.
    """
    if alignment is None:
        import json
        if not project.alignment_path.exists():
            raise FileNotFoundError(
                f"lyrics_alignment.json not found — run alignment first."
            )
        data = json.loads(project.alignment_path.read_text(encoding="utf-8"))
        alignment = Alignment(**data)

    channel = project.channel
    if color is None:
        color = channel.captions.color_for_mood(project.mood)
    unsung = channel.captions.unsung_color

    ass = build_ass_karaoke(alignment.lines, color=color, unsung=unsung)
    project.audio_dir.mkdir(parents=True, exist_ok=True)
    project.captions_path.write_text(ass, encoding="utf-8")
    if on_log:
        on_log(f"Karaoke captions saved: {project.captions_path.name} "
               f"({alignment.line_count} lines, sung={color}, unsung={unsung})")
    return ass
