"""Tests for pipeline step state detection and runner idempotency."""
import pytest

from cutforge.models.project import VideoProject
from cutforge.pipeline import steps as steps_mod
from cutforge.pipeline import runner


@pytest.fixture
def project(tmp_path, monkeypatch):
    from cutforge.config import settings as settings_mod
    monkeypatch.setattr(settings_mod.Settings, "output_dir",
                        property(lambda self: tmp_path / "output"))
    settings_mod.get_settings.cache_clear()
    return VideoProject.create(run_id="20260722-test", character="Jinwoo")


def test_wizard_state_initial(project):
    state = steps_mod.wizard_state(project)
    by_id = {s["id"]: s for s in state}
    # Nothing produced yet: lyrics/footage/metadata are runnable; align/captions/premiere locked.
    assert by_id["lyrics"]["status"] == "pending"
    assert by_id["footage"]["status"] == "pending"
    assert by_id["align"]["status"] == "locked"
    assert by_id["captions"]["status"] == "locked"
    assert by_id["premiere"]["status"] == "locked"
    # thumbnail can run because character is set
    assert by_id["thumbnail"]["status"] == "pending"


def test_step_done_detected(project):
    project.lyrics_path.parent.mkdir(parents=True, exist_ok=True)
    project.lyrics_path.write_text("Shadow rises", encoding="utf-8")
    project.suno_prompt_path.write_text("{}", encoding="utf-8")
    state = {s["id"]: s for s in steps_mod.wizard_state(project)}
    assert state["lyrics"]["status"] == "done"


def test_runner_locked_step_returns_locked(project):
    logs = []
    result = runner.run_step("20260722-test", "align", log=logs.append)
    assert result["status"] == "locked"
    assert any("bloqueado" in line for line in logs)


def test_runner_unknown_step_raises(project):
    with pytest.raises(ValueError):
        runner.run_step("20260722-test", "nope")
