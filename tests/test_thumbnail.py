"""Tests for the thumbnail service: reference cleanup, refusal fallback, variants and crop."""
import json

import pytest
from PIL import Image

from cutforge.models.project import VideoProject
from cutforge.services import thumbnail_service as T


@pytest.fixture
def project(tmp_path, monkeypatch):
    from cutforge.config import settings as settings_mod
    monkeypatch.setattr(settings_mod.Settings, "output_dir",
                        property(lambda self: tmp_path / "output"))
    settings_mod.get_settings.cache_clear()
    p = VideoProject.create(run_id="20261006-test", character="Subaru")
    p.anime = "Re:Zero"
    return p


def _fake_images(monkeypatch, calls, refuse_refs=False):
    def generate(prompt, out, portrait=False, reference_images=None, on_log=None):
        if refuse_refs and reference_images:
            raise RuntimeError("No image returned by the Responses API: I can't help with that")
        calls.append({"prompt": prompt, "refs": reference_images, "out": out.name})
        Image.new("RGB", (1536, 1024), (40, 30, 20)).save(out)
    monkeypatch.setattr(T.openai_images, "generate_image", generate)


# A YouTube card screenshot: artwork inside a frame, a progress bar and a duration badge.
CLEANUP = {"content_box": {"x0": 0.1, "y0": 0.1, "x1": 0.9, "y1": 0.85},
           "overlays": [{"x0": 0.75, "y0": 0.7, "x1": 0.88, "y1": 0.8, "what": "duration badge"}]}
DIRECTION = {k: k + " spec" for k in T._DIRECTION_KEYS}
DIRECTION["subject"] = "Subaru kneeling in the rain, fists clenched"


def _fake_claude(monkeypatch, asked):
    def complete_json(system, user, **kw):
        asked.append(system)
        if system is T.CLEANUP_SYSTEM:
            return json.loads(json.dumps(CLEANUP))
        assert system is T.REFERENCE_DIRECTION_SYSTEM
        return dict(DIRECTION)
    monkeypatch.setattr(T.anthropic_client, "complete_json", complete_json)


def _add_ref(project):
    project.thumbnail_refs_dir.mkdir(parents=True)
    Image.new("RGB", (500, 300), (200, 200, 200)).save(project.thumbnail_refs_dir / "ref_01.png")


def test_refs_are_cleaned_and_sent_as_mood_board(project, monkeypatch):
    calls, asked = [], []
    _fake_images(monkeypatch, calls)
    _fake_claude(monkeypatch, asked)
    _add_ref(project)

    result = T.run(project, {}, lambda m: None)

    assert [c["out"] for c in calls] == ["refthumb_01.png", "refthumb_02.png"]
    sent = calls[0]["refs"]
    assert [p.parent.name for p in sent] == ["refs_clean"]          # never the raw screenshot
    assert Image.open(sent[0]).size == (400, 225)                    # cropped to the artwork
    prompt = calls[0]["prompt"]
    assert "Subaru (from Re:Zero)" in prompt and "mood-board references only" in prompt
    assert "no progress bar" in prompt and "timestamp badge" in prompt
    assert result == {"path": str(project.thumbnail_path), "art": "refthumb_01.png",
                      "mode": "reference"}
    assert Image.open(project.thumbnail_path).size == (1280, 720)
    assert T.selection(project) == {"art": "refthumb_01.png", "mode": "reference"}

    # Redo appends variants and reuses the cached cleanup; pick makes no new images.
    T.run(project, {"refresh": True, "variants": 1}, lambda m: None)
    assert asked.count(T.CLEANUP_SYSTEM) == 1
    assert T.run(project, {"refresh": True, "pick": "refthumb_03.png"}, lambda m: None)["art"] == "refthumb_03.png"
    assert len(calls) == 3
    assert [p.name for p in T.variant_paths(project)] == [
        "refthumb_01.png", "refthumb_02.png", "refthumb_03.png"]


def test_refused_refs_fall_back_to_a_described_direction(project, monkeypatch):
    calls, asked = [], []
    _fake_images(monkeypatch, calls, refuse_refs=True)
    _fake_claude(monkeypatch, asked)
    _add_ref(project)

    T.run(project, {"variants": 1}, lambda m: None)

    assert len(calls) == 1 and calls[0]["refs"] is None              # refs are not sent again
    prompt = calls[0]["prompt"]
    assert "SUBJECT: Subaru kneeling in the rain" in prompt
    assert "- Lighting: lighting spec" in prompt and "NO text" in prompt


def test_without_refs_uses_key_art_baseline(project, monkeypatch):
    calls = []
    _fake_images(monkeypatch, calls)
    monkeypatch.setattr(T.anthropic_client, "complete_json",
                        lambda *a, **k: {"expression": "desperate resolve"})

    result = T.run(project, {"variants": 1}, lambda m: None)

    assert result["mode"] == "keyart" and calls[0]["refs"] is None
    assert "Expression: desperate resolve." in calls[0]["prompt"]
    assert T.MUSIC_BASELINE in calls[0]["prompt"]


def test_finish_cover_crops_instead_of_stretching(tmp_path):
    src, out = tmp_path / "a.png", tmp_path / "a.jpg"
    img = Image.new("RGB", (1536, 1024), (0, 0, 0))
    img.paste((255, 255, 255), (0, 0, 1536, 50))                     # band cut by the crop
    img.save(src)
    T._finish(src, out)
    result = Image.open(out)
    assert result.size == (1280, 720)
    assert result.getpixel((640, 2))[0] < 30                         # top band cropped away


def test_pick_rejects_unknown_variant(project, monkeypatch):
    with pytest.raises(FileNotFoundError):
        T.run(project, {"pick": "../../etc.png"}, lambda m: None)
