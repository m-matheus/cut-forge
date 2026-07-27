"""Pipeline step registry — the ordered wizard steps, their state and dependencies.

Each step knows: its id/label, which output file proves it's done, and which upstream
condition must hold before it can run. The UI derives the wizard card states from here.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

from cutforge.models.project import VideoProject


class StepStatus(str, Enum):
    LOCKED = "locked"        # dependency not met yet
    PENDING = "pending"      # ready to run
    DONE = "done"            # output exists
    RUNNING = "running"      # currently executing (tracked by the runner)
    ERROR = "error"          # last run failed


@dataclass(frozen=True)
class Step:
    id: str
    label: str
    # Returns True when this step's output already exists on disk.
    is_done: Callable[[VideoProject], bool]
    # Returns True when the step is allowed to run (deps satisfied).
    can_run: Callable[[VideoProject], bool]
    # Human hint about what unblocks the step (shown when locked).
    requires_hint: str = ""


def _always(_p: VideoProject) -> bool:
    return True


# Ordered wizard steps.
STEPS: list[Step] = [
    Step(
        id="reference",
        label="Rap de referência (opcional)",
        is_done=lambda p: p.reference_profile_path.exists(),
        can_run=_always,
        requires_hint="Cole a URL do YouTube do rap de referência.",
    ),
    Step(
        id="lyrics",
        label="Letra & estilo",
        is_done=lambda p: p.suno_prompt_path.exists() and p.lyrics_path.exists(),
        can_run=_always,
    ),
    Step(
        id="footage",
        label="Footage",
        is_done=lambda p: p.footage_path.exists(),
        can_run=_always,
        requires_hint="Informe a URL do YouTube.",
    ),
    Step(
        id="align",
        label="Alinhar letra",
        is_done=lambda p: p.alignment_path.exists(),
        can_run=lambda p: p.track_path.exists() and p.lyrics_path.exists(),
        requires_hint="Precisa de audio/track.mp3 (gerado no Suno) e da letra.",
    ),
    Step(
        id="captions",
        label="Transcript Premiere",
        is_done=lambda p: p.premiere_transcript_path.exists(),
        can_run=lambda p: p.alignment_path.exists(),
        requires_hint="Rode o alinhamento primeiro.",
    ),
    Step(
        id="thumbnail",
        label="Thumbnail",
        is_done=lambda p: p.thumbnail_path.exists(),
        can_run=lambda p: bool(p.character),
        requires_hint="Defina o personagem no run.",
    ),
    Step(
        id="metadata",
        label="Metadata",
        is_done=lambda p: p.metadata_path.exists(),
        can_run=_always,
    ),
    Step(
        id="premiere",
        label="Exportar Premiere",
        is_done=lambda p: p.premiere_project_path.exists(),
        can_run=lambda p: p.footage_path.exists() and p.track_path.exists(),
        requires_hint="Precisa do footage e do track.mp3.",
    ),
]

STEP_BY_ID: dict[str, Step] = {s.id: s for s in STEPS}


def step_status(step: Step, project: VideoProject,
                running_ids: set[str] | None = None,
                error_ids: set[str] | None = None) -> StepStatus:
    running_ids = running_ids or set()
    error_ids = error_ids or set()
    if step.id in running_ids:
        return StepStatus.RUNNING
    if step.is_done(project):
        return StepStatus.DONE
    if step.id in error_ids:
        return StepStatus.ERROR
    if step.can_run(project):
        return StepStatus.PENDING
    return StepStatus.LOCKED


def wizard_state(project: VideoProject, running_ids: set[str] | None = None,
                 error_ids: set[str] | None = None) -> list[dict]:
    """Return a UI-friendly list of {id, label, status, requires_hint} for the wizard."""
    return [
        {
            "id": s.id,
            "label": s.label,
            "status": step_status(s, project, running_ids, error_ids).value,
            "requires_hint": s.requires_hint,
        }
        for s in STEPS
    ]
