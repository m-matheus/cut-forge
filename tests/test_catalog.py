"""Tests for the channel catalog memory (cross-song intro/hook repetition)."""
import json

import pytest

from cutforge.models.project import VideoProject
from cutforge.services import catalog_service, song_service


@pytest.fixture
def project(tmp_path, monkeypatch):
    from cutforge.config import settings as settings_mod
    monkeypatch.setattr(settings_mod.Settings, "output_dir",
                        property(lambda self: tmp_path / "output"))
    settings_mod.get_settings.cache_clear()
    return VideoProject.create(run_id="20261020-gaara", character="Gaara", anime="Naruto")


def _seed_run(run_id: str, intro: str, *, channel: str = "enkai") -> None:
    VideoProject.create(run_id=run_id, channel_slug=channel)
    lyrics = f"[Intro]\n[Spoken]\n{intro}\n\n[Verse 1]\n[Rap]\nverse line\n\n[Chorus]\nhook line one\nhook line two"
    (VideoProject.load(run_id).run_dir / "lyrics.txt").write_text(lyrics, encoding="utf-8")


_TEMPLATE_INTROS = {
    "20261011-obito": "Enkai, let's go...\nThis one's for Obito Uchiha.\nThe boy they buried under the rock.",
    "20260928-kakashi": "Enkai again...\nThis one's for Kakashi Hatake.\nThe Copy Ninja of the Leaf.",
    "20260911-ichigo": "Enkai again...\nThis one's for the boy who bled into a blade.",
}


def test_sections_split_on_structure_tags_only():
    lyrics = "[Intro]\n[Spoken]\nEnkai.\n(bell tolls)\n\n[Verse 1]\n[Rap]\nline"
    assert catalog_service.section(lyrics, "intro") == "[Spoken]\nEnkai.\n(bell tolls)"
    assert catalog_service.sung_text(catalog_service.section(lyrics, "intro")) == "Enkai."


def test_load_catalog_newest_first_skips_self_and_other_channels(project):
    for run_id, intro in _TEMPLATE_INTROS.items():
        _seed_run(run_id, intro)
    _seed_run("20261015-other", "Someone else's intro", channel="other")
    (project.run_dir / "lyrics.txt").write_text("[Intro]\nmine", encoding="utf-8")

    catalog = catalog_service.load_catalog(project)

    assert [s.run_id for s in catalog] == ["20261011-obito", "20260928-kakashi", "20260911-ichigo"]
    assert catalog[0].hook == "hook line one\nhook line two"


def test_echo_check_flags_the_template_and_passes_a_fresh_intro(project):
    for run_id, intro in _TEMPLATE_INTROS.items():
        _seed_run(run_id, intro)
    catalog = catalog_service.load_catalog(project)

    templated = "[Spoken]\nEnkai again...\nThis one's for Gaara of the Sand."
    assert "this one's for" in catalog_service.echoed_intro_phrases(templated, catalog)
    fresh = "[Spoken]\n(sand hisses)\nEnkai.\nThe gourd remembers every scream."
    assert catalog_service.echoed_intro_phrases(fresh, catalog) == []


def test_replace_section_keeps_tag_and_following_sections():
    lyrics = "[Intro]\n[Spoken]\nold\n\n[Verse 1]\n[Rap]\nline"
    out = catalog_service.replace_section(lyrics, "intro", "[Spoken]\nnew")
    assert out == "[Intro]\n[Spoken]\nnew\n\n[Verse 1]\n[Rap]\nline"


class _Stub:
    def __init__(self, payloads):
        self._payloads = list(payloads)
        self.calls = []

    def complete_json(self, system, user_prompt, **kwargs):
        self.calls.append((system, user_prompt))
        return self._payloads.pop(0)


_DIRECTION = {"core_theme": "x", "intro_concept": "sand hissing, then the tag"}


def test_generate_rewrites_an_intro_that_echoes_the_catalog(project, monkeypatch):
    for run_id, intro in _TEMPLATE_INTROS.items():
        _seed_run(run_id, intro)
    song = {"title": "Sand", "style": "dark trap, 140 BPM", "exclude": "",
            "lyrics": "[Intro]\n[Spoken]\nEnkai again...\nThis one's for Gaara.\n\n[Verse 1]\n[Rap]\nline",
            "suno_tips": ""}
    fixed = {"intro": "[Spoken]\n(sand hisses)\nEnkai.\nThe gourd remembers every scream."}
    stub = _Stub([_DIRECTION, song, fixed])
    monkeypatch.setattr(song_service, "anthropic_client", stub)

    pkg = song_service.generate_package(project, "dark trap")

    assert len(stub.calls) == 3
    # Planner and writer both see the catalog; the brief's intro concept reaches the writer.
    assert "CHANNEL CATALOG" in stub.calls[0][1]
    assert "CHANNEL CATALOG" in stub.calls[1][1]
    assert "Intro concept: sand hissing" in stub.calls[1][1]
    assert "this one's for" in stub.calls[2][1]
    assert pkg.lyrics.startswith("[Intro]\n[Spoken]\n[Sand hisses]\nEnkai.")
    assert "[Verse 1]\n[Rap]\nline" in pkg.lyrics
    assert project.lyrics_path.read_text(encoding="utf-8") == pkg.lyrics


def test_generate_keeps_a_fresh_intro_without_extra_call(project, monkeypatch):
    for run_id, intro in _TEMPLATE_INTROS.items():
        _seed_run(run_id, intro)
    song = {"title": "Sand", "style": "dark trap, 140 BPM", "exclude": "",
            "lyrics": "[Intro]\n[Whispered]\nEnkai.\n\n[Verse 1]\n[Rap]\nline", "suno_tips": ""}
    stub = _Stub([_DIRECTION, song])
    monkeypatch.setattr(song_service, "anthropic_client", stub)

    pkg = song_service.generate_package(project, "dark trap")

    assert len(stub.calls) == 2
    assert pkg.lyrics == song["lyrics"]


def test_final_chorus_and_breakdown_open_sections():
    from cutforge.services import catalog_service
    lyrics = ("[Chorus]\n[Sung]\nnobody's there\n[Breakdown]\n[Whispered]\nwoke up\n"
              "[Final Chorus]\n[Belted]\nnot nobody anymore")
    names = [name for name, _ in catalog_service.sections(lyrics)]
    assert names == ["chorus", "breakdown", "chorus"]
