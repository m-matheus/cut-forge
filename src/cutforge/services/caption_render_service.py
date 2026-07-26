"""Caption-overlay renderer — burn the kinetic ASS onto a transparent video.

Premiere cannot import ``.ass`` natively, and SRT is static plain text (no positioning,
no per-word karaoke, no pop-in). To get the kinetic caption look into the timeline while
keeping it editable, we render the ASS onto a fully transparent canvas and export a
ProRes 4444 ``.mov`` with an alpha channel. premiere_service drops this on a V3 track
above the footage; Premiere composites the alpha, so only the animated text shows.

Requires ffmpeg with libass + prores_ks (same implicit ffmpeg dependency yt-dlp already
relies on). The ASS-path escaping for the libass filter mirrors the old
``compose_music_video.py`` (Windows drive letters need ``C\\:/``).
"""
from __future__ import annotations

import re
import shutil
import subprocess

from cutforge.models.project import VideoProject
from cutforge.services import caption_service


def _escape_ass_path(path) -> str:
    """Escape a path for the libass ``ass=`` filter (Windows ``C:/`` -> ``C\\:/``)."""
    ass_path = path.resolve().as_posix()
    return re.sub(r"^([A-Za-z]):/", r"\1\\:/", ass_path)


def render_caption_overlay(project: VideoProject, *, duration_s: float | None = None,
                           color: str | None = None, words_per_group: int = 3,
                           on_log=None) -> "object":
    """Render the kinetic captions to a transparent ProRes 4444 .mov. Returns its path.

    Rebuilds the ASS in the kinetic style (independent of the on-disk captions.ass, which
    may be karaoke), then burns it onto a transparent canvas sized to the channel video.
    """
    from pathlib import Path

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH — needed to render the kinetic caption overlay. "
            "Install ffmpeg (same dependency yt-dlp uses)."
        )

    if not project.alignment_path.exists():
        raise FileNotFoundError(
            "lyrics_alignment.json not found — run alignment before rendering captions."
        )

    channel = project.channel
    w, h = channel.video.width, channel.video.height
    fps = float(channel.video.fps)

    import json
    from cutforge.models.alignment import Alignment
    data = json.loads(project.alignment_path.read_text(encoding="utf-8"))
    alignment = Alignment(**data)

    # Duration: caller-provided (song length) or fall back to the alignment's last line end.
    if duration_s is None:
        duration_s = max((l.end for l in alignment.lines), default=0.0) + 1.0

    # Write a dedicated kinetic ASS next to the overlay so the render is reproducible and
    # never depends on whatever style captions.ass currently holds.
    if color is None:
        color = channel.captions.color_for_mood(project.mood)
    unsung = channel.captions.unsung_color
    ass_text = caption_service.build_ass_music_kinetic(
        alignment.lines, color=color, unsung=unsung, words_per_group=words_per_group)
    kinetic_ass_path = project.audio_dir / "captions_kinetic.ass"
    project.audio_dir.mkdir(parents=True, exist_ok=True)
    kinetic_ass_path.write_text(ass_text, encoding="utf-8")

    out_path = project.captions_overlay_path
    escaped = _escape_ass_path(kinetic_ass_path)

    # The libass `ass` filter composites onto an OPAQUE frame — feeding it color@0.0
    # still yields alpha=255 everywhere (the transparent background is lost), so the
    # overlay would cover the footage as solid black. Instead we render the ASS on solid
    # black, then rebuild the alpha channel from luminance: bright text -> opaque, black
    # background -> transparent (alphamerge). ProRes 4444 (yuva444p10le) carries the alpha
    # so Premiere composites only the text over the footage.
    filter_complex = (
        f"[0]ass='{escaped}'[t];"
        f"[t]split[t1][t2];"
        f"[t2]format=gray,geq=lum='clip(lum(X,Y)*3,0,255)'[a];"
        f"[t1][a]alphamerge[out]"
    )
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c=black:s={w}x{h}:r={fps:g}:d={duration_s:.3f}",
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le",
        str(out_path),
    ]
    if on_log:
        on_log(f"Rendering kinetic caption overlay ({duration_s:.1f}s @ {w}x{h})…")

    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    for line in process.stdout:
        line = line.rstrip()
        if line and on_log:
            on_log(line)
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg caption-overlay render failed (exit {process.returncode})")

    if on_log:
        size_mb = out_path.stat().st_size / (1024 * 1024)
        on_log(f"Caption overlay saved: {out_path.name} ({size_mb:.1f} MB)")
    return out_path
