"""Tests for the Premiere FCP7 XML export — the core of CutForge."""
import json
import wave
from pathlib import Path

import pytest

import opentimelineio as otio

from cutforge.models.project import VideoProject


def _make_silent_wav(path: Path, seconds: float, rate: int = 8000) -> None:
    """Write a tiny silent WAV so mutagen/OTIO can read a real duration."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(seconds * rate)
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * frames)


@pytest.fixture
def run_project(tmp_path, monkeypatch):
    # Point the output dir at a temp location.
    from cutforge.config import settings as settings_mod
    monkeypatch.setattr(settings_mod.Settings, "output_dir",
                        property(lambda self: tmp_path / "output"))
    settings_mod.get_settings.cache_clear()

    project = VideoProject.create(
        run_id="20260722-test", character="Jinwoo", anime="Solo Leveling", mood="dark cold",
        title="Shadow Sovereign",
    )
    # Fake footage (any file) + a real silent WAV track so duration works.
    project.footage_path.parent.mkdir(parents=True, exist_ok=True)
    project.footage_path.write_bytes(b"\x00" * 1024)  # placeholder mp4
    _make_silent_wav(project.track_path.with_suffix(".wav"), seconds=12.0)
    # premiere_service reads track_path (.mp3); repoint it to the wav we made.
    return project, project.track_path.with_suffix(".wav")


def test_premiere_export_roundtrip(run_project, monkeypatch):
    project, wav_track = run_project

    # Make track_path point to our real wav (has a readable duration).
    monkeypatch.setattr(type(project), "track_path",
                        property(lambda self: wav_track))

    # Alignment with 4 lyric lines.
    alignment = {
        "audio": str(wav_track),
        "lines": [
            {"start": 1.0, "end": 2.5, "words": [
                {"word": "Shadow", "start": 1.0, "end": 1.7},
                {"word": "Monarch", "start": 1.7, "end": 2.5}]},
            {"start": 3.0, "end": 4.0, "words": [
                {"word": "Rise", "start": 3.0, "end": 4.0}]},
            {"start": 5.0, "end": 6.5, "words": [
                {"word": "Arise", "start": 5.0, "end": 6.5}]},
            {"start": 8.0, "end": 9.0, "words": [
                {"word": "Sovereign", "start": 8.0, "end": 9.0}]},
        ],
    }
    project.alignment_path.parent.mkdir(parents=True, exist_ok=True)
    project.alignment_path.write_text(json.dumps(alignment), encoding="utf-8")

    from cutforge.services import premiere_service
    out = premiere_service.build_project(project)
    assert out.exists()

    # Round-trip: re-read the XML with OTIO and validate structure.
    back = otio.adapters.read_from_file(str(out), adapter_name="fcp_xml")
    video_tracks = [t for t in back.tracks if t.kind == otio.schema.TrackKind.Video]
    audio_tracks = [t for t in back.tracks if t.kind == otio.schema.TrackKind.Audio]
    assert len(video_tracks) == 1
    assert len(audio_tracks) == 1

    markers = [m for c in back.find_clips() for m in c.markers]
    assert len(markers) == 4
    names = {m.name for m in markers}
    assert "Shadow Monarch" in names
