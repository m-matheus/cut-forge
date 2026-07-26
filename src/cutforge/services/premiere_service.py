"""Premiere export service — build an OTIO timeline and write FCP7 XML.

Adobe Premiere Pro imports Final Cut Pro 7 XML (XMEML) natively. We build a timeline
with the footage on a V1 video track, the song on an A1 audio track, and a marker at
each lyric-line start (so the user can cut the footage on the beat), then serialize it
with OTIO's ``fcp_xml`` adapter (from the ``otio-fcp-adapter`` plugin).

Learned from round-trip testing on Python 3.13 / OTIO 0.18.1:
- ExternalReference MUST have ``available_range`` set (else the adapter raises
  ``AttributeError: 'NoneType' object has no attribute 'start_time'``).
- Frame counts must be integers — use ``round(seconds * fps)``.
- Markers attach to the Clip, not the Track.
- Media URLs are absolute ``file://`` paths so Premiere can relink.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from cutforge.models.alignment import Alignment
from cutforge.models.project import VideoProject


def _audio_duration_seconds(path: Path) -> float:
    """Duration of an audio/video file. Tries mutagen (mp3), then a generic fallback."""
    try:
        from mutagen.mp3 import MP3
        return MP3(str(path)).info.length
    except Exception:
        pass
    try:
        from mutagen import File as MutagenFile
        mf = MutagenFile(str(path))
        if mf is not None and mf.info is not None:
            return float(mf.info.length)
    except Exception:
        pass
    raise RuntimeError(f"Could not determine duration of {path}")


def _file_url(path: Path) -> str:
    # Premiere Pro's FCP7 XML importer expects an explicit ``localhost`` host on Windows:
    # ``file://localhost/C:/…``. Python's ``Path.as_uri()`` emits a host-less ``file:///C:/…``,
    # which Premiere mis-parses into ``\\\C:\…`` and shows as "Media offline". Insert the host.
    uri = path.resolve().as_uri()  # file:///C:/... (or file:///home/... on POSIX)
    return uri.replace("file:///", "file://localhost/", 1)


def build_project(project: VideoProject, *, on_log=None) -> Path:
    """Build the Premiere-importable FCP7 XML for the run. Returns the .xml path."""
    import opentimelineio as otio
    from opentimelineio.opentime import RationalTime, TimeRange

    if not project.track_path.exists():
        raise FileNotFoundError(f"track.mp3 not found at {project.track_path}")
    if not project.footage_path.exists():
        raise FileNotFoundError(f"footage source.mp4 not found at {project.footage_path}")

    fps = float(project.channel.video.fps)
    duration_s = _audio_duration_seconds(project.track_path)
    total_frames = max(1, round(duration_s * fps))
    full_range = TimeRange(RationalTime(0, fps), RationalTime(total_frames, fps))

    if on_log:
        on_log(f"Song duration {duration_s:.1f}s -> {total_frames} frames @ {fps:.0f}fps")

    timeline = otio.schema.Timeline(name=project.title or project.run_id)

    # V1 — footage (whole clip; user cuts in Premiere)
    video_track = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    footage_clip = otio.schema.Clip(
        name="footage",
        media_reference=otio.schema.ExternalReference(
            target_url=_file_url(project.footage_path),
            available_range=full_range,
        ),
        source_range=full_range,
    )
    video_track.append(footage_clip)

    # A1 — song
    audio_track = otio.schema.Track(name="A1", kind=otio.schema.TrackKind.Audio)
    audio_clip = otio.schema.Clip(
        name="track",
        media_reference=otio.schema.ExternalReference(
            target_url=_file_url(project.track_path),
            available_range=full_range,
        ),
        source_range=full_range,
    )
    audio_track.append(audio_clip)

    timeline.tracks.append(video_track)
    timeline.tracks.append(audio_track)

    # Markers at each lyric-line start (cut on the beat)
    marker_count = 0
    if project.alignment_path.exists():
        data = json.loads(project.alignment_path.read_text(encoding="utf-8"))
        alignment = Alignment(**data)
        for line in alignment.lines:
            frame = round(line.start * fps)
            if frame < 0 or frame >= total_frames:
                continue
            footage_clip.markers.append(otio.schema.Marker(
                name=line.text[:40] or f"line {marker_count + 1}",
                marked_range=TimeRange(RationalTime(frame, fps), RationalTime(0, fps)),
            ))
            marker_count += 1
    if on_log:
        on_log(f"Added {marker_count} lyric-line markers")

    # Write FCP7 XML
    project.premiere_dir.mkdir(parents=True, exist_ok=True)
    out_path = project.premiere_project_path
    otio.adapters.write_to_file(timeline, str(out_path), adapter_name="fcp_xml")

    # Copy the channel endcard alongside for convenience
    endcard = project.channel.asset_path("endcard")
    if endcard and endcard.exists():
        shutil.copy2(endcard, project.premiere_dir / "endcard.png")

    if on_log:
        on_log(f"Premiere project written: {out_path}")
    return out_path
