"""yt-dlp footage downloader.

Ported from the old ``fetch_amv.py`` but WITHOUT scene splitting — CutForge downloads
the source clip whole and the user cuts it in Premiere. Uses the yt-dlp module via the
current interpreter (no external binary assumption beyond ffmpeg for stream merge, which
yt-dlp locates on PATH if present).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

YT_DLP = [sys.executable, "-m", "yt_dlp"]


def probe(url: str) -> dict:
    """Fetch video metadata without downloading."""
    result = subprocess.run(
        YT_DLP + ["--dump-json", "--no-playlist", url],
        capture_output=True, text=True, encoding="utf-8", check=True,
    )
    info = json.loads(result.stdout)
    return {
        "url": url,
        "title": info.get("title", ""),
        "channel": info.get("channel", info.get("uploader", "")),
        "duration": info.get("duration", 0),
        "upload_date": info.get("upload_date", ""),
    }


def download(url: str, dest: Path, *, on_log=None) -> dict:
    """Download the best <=1080p mp4 to ``dest`` and return metadata.

    ``on_log`` (optional callable) receives yt-dlp stdout lines for live progress.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    meta = probe(url)
    if on_log:
        on_log(f"Downloading: {meta['title']} ({meta['duration']}s)")

    cmd = YT_DLP + [
        "-f", "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4][height<=1080]/best",
        "--merge-output-format", "mp4",
        "-o", str(dest),
        "--no-playlist",
        "--newline",
        url,
    ]
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
        raise RuntimeError(f"yt-dlp failed (exit {process.returncode}) for {url}")

    meta["local_path"] = str(dest)
    return meta
