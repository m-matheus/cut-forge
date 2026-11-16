"""Tests for the Spotify release step (distributor form) and the QR call in Premiere."""
import json
import wave
from datetime import date
from pathlib import Path

import pytest

from cutforge.config.channels import load_channel
from cutforge.models.project import VideoProject
from cutforge.services import release_service


@pytest.fixture
def project(tmp_path, monkeypatch):
    from cutforge.config import settings as settings_mod
    monkeypatch.setattr(settings_mod.Settings, "output_dir",
                        property(lambda self: tmp_path / "output"))
    settings_mod.get_settings.cache_clear()
    p = VideoProject.create(run_id="20261020-gaara", character="Gaara", anime="Naruto",
                            title="GAARA RAP | \"Sand Coffin\" | Enkai [Naruto]")
    p.lyrics_path.write_text(
        "[Intro]\n[Spoken]\n(sand hisses)\nEnkai.\n\n[Verse 1]\n[Rap]\n"
        "The gourd stays full (yeah)\n[Chorus]\n[Gang Vocals]\nSand coffin, close",
        encoding="utf-8")
    p.suno_prompt_path.write_text(json.dumps({"title": "Sand Coffin",
                                              "style": "dark trap, cold, rapped male vocal"}),
                                  encoding="utf-8")
    return p


@pytest.fixture
def spotify(monkeypatch):
    """Patch the loaded channel's spotify block (channel.json stays untouched)."""
    spec = load_channel("enkai").spotify
    def set_(**kw):
        for k, v in kw.items():
            monkeypatch.setattr(spec, k, v)
    return set_


def _field(form, label):
    return next(f for f in form["fields"] if f["label"] == label)


def _silent_wav(path: Path, seconds: float = 2.0, rate: int = 44100) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(b"\x00\x00\x00\x00" * int(seconds * rate))


def test_clean_lyrics_drops_tags_and_stage_directions():
    out = release_service.clean_lyrics(
        "[Intro]\n[Spoken]\n(sand hisses)\nEnkai.\n\n[Verse 1]\n[Rap]\nline one (yeah)\n")
    assert out == "Enkai.\n\nline one (yeah)"


def test_explicit_words_across_languages():
    assert release_service.explicit_words("no curse here") == []
    assert release_service.explicit_words("Fuck it, porra, mierda") == ["fuck", "mierda", "porra"]


def test_release_title_uses_song_title_not_youtube_title(project):
    assert release_service.release_title(project, "Sand Coffin") == "Sand Coffin (Gaara Rap)"
    # A Suno title that already names the character is kept as is.
    assert release_service.release_title(project, "One Eye Left (Gaara)") == "One Eye Left (Gaara)"


def test_form_flags_missing_wav_and_songwriter(project, spotify):
    spotify(songwriter="", artist_url="")
    project.track_path.parent.mkdir(parents=True, exist_ok=True)
    project.track_path.write_bytes(b"\x00")

    form = release_service.build_form(project, today=date(2026, 10, 20))

    assert _field(form, "Título")["value"] == "Sand Coffin (Gaara Rap)"
    assert _field(form, "Áudio")["value"] == "audio/track.mp3"
    assert _field(form, "Data de lançamento")["value"] == "2026-11-03"
    assert _field(form, "Conteúdo explícito")["value"] == "Não"
    assert _field(form, "YouTube Content ID")["value"] == "Não"
    warnings = " ".join(form["warnings"])
    assert "track.wav" in warnings and "songwriter" in warnings and "artist_url" in warnings


def test_form_with_wav_and_config_has_no_warnings(project, spotify):
    spotify(songwriter="Jane Doe", artist_url="https://open.spotify.com/artist/abc?si=x")
    _silent_wav(project.track_wav_path)

    form = release_service.build_form(project)

    assert form["warnings"] == []
    assert _field(form, "Áudio")["value"] == "audio/track.wav"
    assert "44100 Hz" in _field(form, "Áudio")["hint"]
    assert _field(form, "Compositor (nome legal)")["value"] == "Jane Doe"


def test_run_writes_cover_lyrics_and_form(project, monkeypatch):
    from PIL import Image

    def fake_image(request, out, **kw):
        assert kw["size"] == "1024x1024"
        assert "NO people" in request
        Image.new("RGB", (1024, 1024), (200, 80, 0)).save(out)
        return out

    monkeypatch.setattr(release_service.openai_images, "generate_image", fake_image)
    monkeypatch.setattr(release_service.anthropic_client, "complete_json",
                        lambda *a, **k: {"concept": "a cracked gourd", "palette": "red", "mood": "cold"})

    release_service.run(project, {}, lambda _m: None)

    assert Image.open(project.cover_path).size == (3000, 3000)
    assert project.release_lyrics_path.read_text(encoding="utf-8").startswith("Enkai.")
    assert json.loads(project.release_path.read_text(encoding="utf-8"))["cover_art"] == "cover_01.png"


def test_spotify_track_link_validation(project):
    project.spotify_track_url = "https://open.spotify.com/track/123?si=abc"
    assert project.spotify_track_link == "https://open.spotify.com/track/123"
    project.spotify_track_url = "https://open.spotify.com/album/123"
    assert project.spotify_track_link == ""


def test_premiere_adds_qr_track_over_the_intro(project, spotify, monkeypatch):
    from cutforge.services import premiere_service, spotify_notice_service

    spotify(artist_url="https://open.spotify.com/artist/abc")
    project.spotify_track_url = "https://open.spotify.com/track/xyz"
    _silent_wav(project.track_wav_path, seconds=12.0, rate=8000)
    monkeypatch.setattr(type(project), "track_path", property(lambda self: self.track_wav_path))
    project.footage_path.parent.mkdir(parents=True, exist_ok=True)
    project.footage_path.write_bytes(b"\x00" * 1024)
    built = {}

    def fake_notice(p, url, duration, out):
        built.update(url=url, duration=duration)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\x00")
        return out

    monkeypatch.setattr(spotify_notice_service, "build_notice", fake_notice)

    xml = premiere_service.build_project(project).read_text(encoding="utf-8")

    assert built == {"url": "https://open.spotify.com/track/xyz", "duration": 8.0}
    assert "spotify_qr.mov" in xml


def test_notice_frame_renders_a_plate(project, spotify):
    from cutforge.services import spotify_notice_service

    img = spotify_notice_service.notice_frame(project, "https://open.spotify.com/artist/abc")
    assert img.size == (1920, 1080) and img.mode == "RGBA"
    assert img.getbbox() is not None          # something was drawn
    assert img.getpixel((0, 0))[3] == 0       # the rest stays transparent
