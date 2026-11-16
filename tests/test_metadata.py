"""Tests for YouTube metadata generation — it must describe the CURRENT song."""
import json

import pytest

from cutforge.models.project import VideoProject
from cutforge.services import metadata_service


@pytest.fixture
def project(tmp_path, monkeypatch):
    from cutforge.config import settings as settings_mod
    monkeypatch.setattr(settings_mod.Settings, "output_dir",
                        property(lambda self: tmp_path / "output"))
    settings_mod.get_settings.cache_clear()
    # project.title still holds an OLDER song's YouTube title (stale since a regeneration).
    p = VideoProject.create(run_id="20261011-obito", character="Obito", anime="Naruto",
                            title="OBITO RAP | \"One Eye Left\" | Enkai [Naruto]")
    p.suno_prompt_path.write_text(json.dumps({"title": "Sorry I'm Late"}), encoding="utf-8")
    p.lyrics_path.write_text("[Intro]\nEnkai.\n[Chorus]\n[Sung, emotional]\n"
                             "So nobody has to bleed\n(nobody)\nNobody's there",
                             encoding="utf-8")
    p.creative_direction_path.write_text(json.dumps({"core_theme": "a late boy's vow"}),
                                         encoding="utf-8")
    return p


class _Stub:
    def __init__(self):
        self.user_prompt = ""

    def complete_json(self, system, user_prompt, **kwargs):
        self.user_prompt = user_prompt
        return {"title": "OBITO RAP | \"Sorry I'm Late\" | Enkai [Naruto]",
                "description": "d", "tags": ["obito"]}


def test_metadata_uses_current_song_title_hook_and_theme(project, monkeypatch):
    stub = _Stub()
    monkeypatch.setattr(metadata_service, "anthropic_client", stub)

    metadata_service.generate_metadata(project)

    assert "song: Sorry I'm Late" in stub.user_prompt
    assert "One Eye Left" not in stub.user_prompt
    assert "hook: So nobody has to bleed / Nobody's there" in stub.user_prompt
    assert "theme: a late boy's vow" in stub.user_prompt
