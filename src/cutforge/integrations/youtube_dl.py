"""yt-dlp footage downloader.

Ported from the old ``fetch_amv.py`` but WITHOUT scene splitting — CutForge downloads
the source clip whole and the user cuts it in Premiere. Uses the yt-dlp module via the
current interpreter (no external binary assumption beyond ffmpeg for stream merge, which
yt-dlp locates on PATH if present).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

YT_DLP = [sys.executable, "-m", "yt_dlp"]

# YouTube's DEFAULT player client (an Android variant) hides the channel's MANUAL
# subtitle tracks in environments without a JavaScript runtime — it only exposes the
# ``live_chat`` replay. The ``web_embedded`` client lists and serves the real manual
# subtitles regardless. ``--ignore-no-formats-error`` is required alongside it because
# yt-dlp still resolves (and here fails to find, without a JS runtime) a video format
# even under ``--skip-download``; without the flag it aborts before writing the subs.
_SUB_CLIENT_ARGS = ["--extractor-args", "youtube:player_client=web_embedded"]


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


def list_manual_subtitles(url: str) -> list[dict]:
    """Return the MANUAL (channel-authored) subtitle languages available for ``url``.

    Each entry is ``{"code": "<yt-dlp lang code>", "name": "<human label>"}``. These are
    the exact codes to pass to ``download_subtitles`` (e.g. ``pt``, ``pt-BR``, ``en``,
    ``ja``) — YouTube's real codes, not the human names shown in the player. Automatic
    (ASR) captions under ``automatic_captions`` are deliberately ignored.

    ``live_chat`` is filtered out: it is the live-stream chat replay (JSON, not lyric
    text) that YouTube exposes under ``subtitles``, never an actual caption track. Tracks
    that offer no text format (vtt/srt/ttml) are dropped for the same reason.
    """
    result = subprocess.run(
        YT_DLP + _SUB_CLIENT_ARGS
        + ["--dump-json", "--ignore-no-formats-error", "--no-playlist", url],
        capture_output=True, text=True, encoding="utf-8", check=True,
    )
    info = json.loads(result.stdout)
    subs = info.get("subtitles") or {}
    text_exts = {"vtt", "srt", "ttml", "srv1", "srv2", "srv3"}
    out = []
    for code, tracks in subs.items():
        if code == "live_chat":
            continue
        name = ""
        has_text = False
        if isinstance(tracks, list):
            for t in tracks:
                if t.get("ext") in text_exts:
                    has_text = True
                if not name:
                    name = t.get("name", "") or ""
        if not has_text:
            continue
        out.append({"code": code, "name": name})
    out.sort(key=lambda s: s["code"])
    return out


def download(url: str, dest: Path, *, on_log=None) -> dict:
    """Download the best <=1080p mp4 to ``dest`` and return metadata.

    ``on_log`` (optional callable) receives yt-dlp stdout lines for live progress.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    meta = probe(url)
    if on_log:
        on_log(f"Downloading: {meta['title']} ({meta['duration']}s)")

    # Force H.264 (avc1) video: Premiere Pro cannot decode AV1 (av01), which YouTube now
    # serves inside .mp4 containers — so an [ext=mp4] filter alone lets AV1 through and the
    # clip imports as "Media offline". avc1 is universally supported. Fall back to any mp4
    # (then anything) only if no H.264 rendition exists.
    cmd = YT_DLP + [
        "-f", "bestvideo[vcodec^=avc1][height<=1080]+bestaudio[ext=m4a]/"
              "best[vcodec^=avc1][height<=1080]/best[ext=mp4][height<=1080]/best",
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


# --- Manual (channel-authored) subtitle download -------------------------------------

_VTT_TIMING = re.compile(r"^\d{2}:\d{2}:\d{2}[.,]\d{3}\s*-->")
_VTT_CUE_INDEX = re.compile(r"^\d+$")
# Any inline/markup tag: <c>, </c>, <c.colorXXXX>, <b>, </b>, karaoke <00:00:01.234>, etc.
_VTT_ANY_TAG = re.compile(r"<[^>]*>")
# Zero-width & non-breaking spacing that animated captions inject between glyphs.
_VTT_ZERO_WIDTH = re.compile(r"[​‌‍﻿ ]")
_WS_RUN = re.compile(r"\s+")


def _vtt_to_text(vtt: str) -> str:
    """Strip a WebVTT subtitle file down to plain lyric text.

    Handles heavily-animated karaoke captions (the common case for lyric videos): a giant
    ``STYLE``/``::cue`` colour table in the header, every line wrapped in ``<c.colorXXXX>``
    and ``<b>`` tags with ``​`` zero-width spaces sprinkled between glyphs, and each
    line repeated across dozens of near-identical cues as the colour sweep animates.

    Removes the header block, ``NOTE``/``STYLE``/``::cue`` styling, numeric cue indices,
    ``HH:MM:SS.mmm --> …`` timing lines, ALL markup tags and zero-width spacing, then
    normalises whitespace. Consecutive identical lines are collapsed — which, once the
    tags/zero-width noise is gone, folds the karaoke sweep of a line down to one copy while
    still preserving a chorus line that genuinely recurs later in the song.
    """
    # Drop the leading header block (``WEBVTT`` line plus ``Kind:``/``Language:`` metadata
    # and any inline ``Style:``/``::cue`` colour table) up to the first blank line.
    raw_lines = vtt.splitlines()
    if raw_lines and raw_lines[0].lstrip().startswith("WEBVTT"):
        i = 0
        while i < len(raw_lines) and raw_lines[i].strip():
            i += 1
        raw_lines = raw_lines[i:]

    lines: list[str] = []
    for raw in raw_lines:
        line = raw.strip()
        if not line:
            continue
        # Skip styling / structural noise (defensive — most is dropped with the header).
        if (line.startswith("WEBVTT") or line.startswith("NOTE") or line.startswith("STYLE")
                or line.startswith("::cue") or line.startswith("Style:") or line == "}"
                or line == "##"):
            continue
        if "-->" in line and _VTT_TIMING.match(line):
            continue
        if _VTT_CUE_INDEX.match(line):
            continue
        line = _VTT_ANY_TAG.sub("", line)
        line = _VTT_ZERO_WIDTH.sub("", line)
        line = _WS_RUN.sub(" ", line).strip()
        if not line:
            continue
        # Collapse consecutive exact repeats (animated/karaoke-caption artifact).
        if lines and lines[-1] == line:
            continue
        lines.append(line)
    return "\n".join(lines)


def download_subtitles(url: str, out_dir: Path, *, lang: str = "en", on_log=None) -> str:
    """Download ONLY the channel's manual subtitles for ``url`` in ``lang``; return plain text.

    Deliberately passes ``--write-subs`` WITHOUT ``--write-auto-subs``: YouTube's automatic
    captions are ASR-quality (the same problem Whisper has), whereas manual subs are the
    channel's own — usually the correct lyrics. Downloads no media (``--skip-download``).

    Returns the cleaned, dedup'd transcript, or ``""`` when the video has no manual
    subtitles in the requested language.
    """
    log = on_log or (lambda _m: None)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(out_dir / "%(id)s.%(ext)s")
    cmd = YT_DLP + _SUB_CLIENT_ARGS + [
        "--write-subs",
        "--skip-download",
        "--ignore-no-formats-error",
        "--no-playlist",
        "--sub-langs", lang,
        "--sub-format", "vtt",
        "-o", out_template,
        "--newline",
        url,
    ]
    log(f"Fetching manual subtitles ({lang})…")
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    for line in process.stdout:
        line = line.rstrip()
        if line:
            log(line)
    process.wait()
    # A missing manual sub is not an error here — yt-dlp just writes nothing.
    vtt_files = sorted(out_dir.glob("*.vtt"))
    if not vtt_files:
        log(f"No manual subtitles found for language '{lang}'.")
        return ""
    text = _vtt_to_text(vtt_files[0].read_text(encoding="utf-8", errors="replace"))
    log(f"Manual subtitles parsed: {len(text.splitlines())} lines.")
    return text


def download_audio(url: str, dest: Path, *, audio_format: str = "mp3", on_log=None) -> dict:
    """Download audio-only from ``url``, transcode to ``audio_format``, save to ``dest``.

    Uses yt-dlp's extract-audio postprocessor (requires ffmpeg on PATH, same implicit
    dependency as the mp4 merge in ``download``). Returns metadata with ``local_path``.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    meta = probe(url)
    if on_log:
        on_log(f"Downloading audio: {meta['title']} ({meta['duration']}s)")

    # -x rewrites the output extension, so template with %(ext)s and point at the stem.
    out_template = str(dest.with_suffix(f".%(ext)s"))
    cmd = YT_DLP + [
        "-x",
        "--audio-format", audio_format,
        "--audio-quality", "0",
        "-f", "bestaudio/best",
        "-o", out_template,
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
        raise RuntimeError(f"yt-dlp audio download failed (exit {process.returncode}) for {url}")

    meta["local_path"] = str(dest)
    return meta
