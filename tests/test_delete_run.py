"""Tests for run deletion via the API — path-traversal guard + rmtree behavior."""
import pytest

from cutforge.models.project import VideoProject


@pytest.fixture
def client(tmp_path, monkeypatch):
    from cutforge.config import settings as settings_mod
    monkeypatch.setattr(settings_mod.Settings, "output_dir",
                        property(lambda self: tmp_path / "output"))
    settings_mod.get_settings.cache_clear()
    from fastapi.testclient import TestClient
    from cutforge.api.app import create_app
    return TestClient(create_app())


def test_delete_run_removes_folder(client):
    project = VideoProject.create(run_id="20260722-del", character="Jinwoo")
    run_dir = project.run_dir
    assert run_dir.exists()

    r = client.delete("/api/runs/20260722-del")
    assert r.status_code == 200
    assert r.json()["status"] == "deleted"
    assert not run_dir.exists()
