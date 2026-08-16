"""Tests for the original-song generation flow.

The whole point of this module is to prove the reference is split into two independent
signals — MUSIC DNA (drives style) and mined LORE (feeds the writer as knowledge) — and
that the final song is always an ORIGINAL composition (no content_blend borrowing, no
adaptation mode). The Anthropic client is stubbed so no network is hit; we assert on the
prompts the services build and on the persisted artifacts.
"""
import json

import pytest

from cutforge.models.lore import ReferenceLoreProfile
from cutforge.models.project import VideoProject
from cutforge.services import lore_service, song_service


@pytest.fixture
def project(tmp_path, monkeypatch):
    from cutforge.config import settings as settings_mod
    monkeypatch.setattr(settings_mod.Settings, "output_dir",
                        property(lambda self: tmp_path / "output"))
    settings_mod.get_settings.cache_clear()
    return VideoProject.create(run_id="20260802-test", character="Midoriya",
                               anime="My Hero Academia", topic="Midoriya")


def _write_reference_profile(project, transcript="Midoriya inherited One For All. Blackwhip."):
    """Seed a reference MUSIC profile on disk (as reference_service would)."""
    project.reference_dir.mkdir(parents=True, exist_ok=True)
    profile = {
        "source_url": "https://youtu.be/x",
        "source_title": "Rap do Midoriya",
        "bpm": 112.3,
        "time_signature": 4,
        "onset_rate_per_sec": 3.04,
        "flow": {"word_count": 8, "words_per_sec": 0.87, "syllables_per_beat": 0.46},
        "transcript": transcript,
    }
    project.reference_profile_path.write_text(json.dumps(profile), encoding="utf-8")
    return profile


class _StubAnthropic:
    """Records every (system, user_prompt) and returns queued JSON payloads in order."""

    def __init__(self, payloads):
        self._payloads = list(payloads)
        self.calls = []  # list of (system, user_prompt)

    def complete_json(self, system, user_prompt, **kwargs):
        self.calls.append((system, user_prompt))
        return self._payloads.pop(0)


_LORE_PAYLOAD = {
    "character": "Midoriya",
    "facts": [{"fact": "Inherited One For All", "category": "event", "confidence": "high"}],
    "abilities": [{"name": "Blackwhip", "description": "tendril quirk", "confidence": "medium"}],
    "events": [{"event": "Fought Muscular", "importance": "high", "confidence": "medium"}],
    "relationships": [{"characters": ["Midoriya", "All Might"], "relationship": "mentor",
                        "confidence": "high"}],
    "themes": ["burden of saving others"],
    "personality_traits": ["selfless"],
    "easter_eggs": [{"reference": "Detroit Smash", "interpreted_meaning": "signature move",
                      "related_lore": "All Might's finisher", "confidence": "high"}],
    "author_interpretations": ["hero at heart"],
    "uncertain_items": ["possible mis-transcription of a name"],
}

_DIRECTION_PAYLOAD = {
    "core_theme": "carrying a borrowed power",
    "narrative_angle": "first-person vow",
    "emotional_arc": "doubt to resolve",
    "hook_concept": "an original chant about inherited fire",
    "key_lore_points": ["One For All", "Blackwhip"],
    "original_metaphor_direction": "lightning as inheritance",
    "delivery_personality": "hungry, determined",
    "things_to_avoid": ["reference hook", "generic hype"],
}

_SONG_PAYLOAD = {
    "title": "Inherited Fire",
    "style": "orchestral trap, heroic, aggressive rapped male vocal, booming 808s, 112 BPM",
    "exclude": "singing, melodic vocals",
    "lyrics": "[Intro]\nNew line here\n[Verse 1]\n[Rap]\nOriginal words",
    "suno_tips": "Push the drop on the last chorus.",
}


# ---------------------------------------------------------------------------
# Lore mining
# ---------------------------------------------------------------------------

def test_lore_mining_produces_valid_structure(project, monkeypatch):
    _write_reference_profile(project)
    stub = _StubAnthropic([_LORE_PAYLOAD])
    monkeypatch.setattr(lore_service, "anthropic_client", stub)

    lore = lore_service.mine_reference_lore(project)

    assert isinstance(lore, ReferenceLoreProfile)
    assert lore.character == "Midoriya"
    assert lore.abilities[0].name == "Blackwhip"
    assert lore.easter_eggs[0].reference == "Detroit Smash"
    assert not lore.is_empty()
    # Persisted for reuse.
    assert project.reference_lore_profile_path.exists()
    # The miner prompt forbids paraphrasing/translating the reference.
    system, _ = stub.calls[0]
    assert "KNOWLEDGE" in system
    assert "paraphras" in system.lower()  # "paraphrase"/"paraphrasing" forbidden


def test_lore_mining_is_cached(project, monkeypatch):
    _write_reference_profile(project)
    stub = _StubAnthropic([_LORE_PAYLOAD])
    monkeypatch.setattr(lore_service, "anthropic_client", stub)

    lore_service.mine_reference_lore(project)
    lore_service.mine_reference_lore(project)  # second call must hit the cache

    assert len(stub.calls) == 1  # LLM invoked only once


def test_lore_mining_returns_none_without_reference(project, monkeypatch):
    stub = _StubAnthropic([])
    monkeypatch.setattr(lore_service, "anthropic_client", stub)
    assert lore_service.mine_reference_lore(project) is None
    assert stub.calls == []  # no LLM call when there is nothing to mine


# ---------------------------------------------------------------------------
# Generation without a reference
# ---------------------------------------------------------------------------

def test_generate_without_reference(project, monkeypatch):
    stub = _StubAnthropic([_DIRECTION_PAYLOAD, _SONG_PAYLOAD])
    monkeypatch.setattr(song_service, "anthropic_client", stub)

    pkg = song_service.generate_package(project, "orchestral trap")

    assert pkg.title == "Inherited Fire"
    assert project.lyrics_path.exists()
    assert project.suno_prompt_path.exists()
    # Two LLM calls: creative direction + writer. No lore mining (no reference).
    assert len(stub.calls) == 2
    # The writer prompt commits to originality even with no reference.
    writer_system, _ = stub.calls[1]
    assert "ORIGINAL" in writer_system
    # No reference => no lore profile mined.
    assert not project.reference_lore_profile_path.exists()


# ---------------------------------------------------------------------------
# Generation with a reference — the core behaviour
# ---------------------------------------------------------------------------

def test_generate_with_reference_flow(project, monkeypatch):
    _write_reference_profile(project)
    # Order of LLM calls: lore mining → creative direction → writer.
    stub = _StubAnthropic([_LORE_PAYLOAD, _DIRECTION_PAYLOAD, _SONG_PAYLOAD])
    monkeypatch.setattr(song_service, "anthropic_client", stub)
    monkeypatch.setattr(lore_service, "anthropic_client", stub)

    pkg = song_service.generate_package(project, "orchestral trap")

    assert pkg.title == "Inherited Fire"
    assert len(stub.calls) == 3
    assert project.reference_lore_profile_path.exists()
    assert project.creative_direction_path.exists()

    # 1) Creative direction receives the mined lore.
    direction_system, direction_user = stub.calls[1]
    assert "What is the NEW song" in direction_system
    assert "Blackwhip" in direction_user  # lore fed in
    assert "One For All" in direction_user

    # 2) The writer receives BOTH the music profile (BPM) and the lore + direction.
    writer_system, writer_user = stub.calls[2]
    assert "112" in writer_user                    # music DNA (BPM)
    assert "SONIC DNA" in writer_user              # music profile block
    assert "CREATIVE DIRECTION" in writer_user     # original brief
    assert "CHARACTER LORE" in writer_user         # mined knowledge
    assert "Detroit Smash" in writer_user          # easter egg surfaced to writer

    # 3) The writer is explicitly told to create an original work and NOT copy expression.
    assert "COMPLETELY ORIGINAL" in writer_system
    ws = writer_system.lower()
    assert "paraphrase the reference" in ws
    assert "reproduce its distinctive phrases" in ws
    assert "reuse its metaphors" in ws


def test_reference_transcript_not_pasted_into_writer(project, monkeypatch):
    """The reference's raw lyrics must never reach the writer as text to reuse."""
    distinctive = "EU SEMPRE FUI UM HEROI totally distinctive punchline"
    _write_reference_profile(project, transcript=distinctive)
    stub = _StubAnthropic([_LORE_PAYLOAD, _DIRECTION_PAYLOAD, _SONG_PAYLOAD])
    monkeypatch.setattr(song_service, "anthropic_client", stub)
    monkeypatch.setattr(lore_service, "anthropic_client", stub)

    song_service.generate_package(project, "orchestral trap")

    _, writer_user = stub.calls[2]
    _, direction_user = stub.calls[1]
    # The transcript only appears in the lore-mining prompt (call 0), never downstream.
    assert distinctive not in writer_user
    assert distinctive not in direction_user
    _, miner_user = stub.calls[0]
    assert distinctive in miner_user


def test_content_blend_kwarg_is_ignored(project, monkeypatch):
    """Legacy content_blend must not change behaviour or reintroduce borrowing."""
    _write_reference_profile(project)
    stub = _StubAnthropic([_LORE_PAYLOAD, _DIRECTION_PAYLOAD, _SONG_PAYLOAD])
    monkeypatch.setattr(song_service, "anthropic_client", stub)
    monkeypatch.setattr(lore_service, "anthropic_client", stub)

    # Passing the deprecated kwarg must be accepted and produce the same original flow.
    pkg = song_service.generate_package(project, "orchestral trap", content_blend="strong")

    assert pkg.title == "Inherited Fire"
    # No prompt anywhere should carry blend/borrow instructions.
    for _system, _user in stub.calls:
        assert "blend" not in _system.lower()
        assert "borrow" not in _system.lower()


def test_creative_direction_is_cached(project, monkeypatch):
    _write_reference_profile(project)
    stub = _StubAnthropic([_LORE_PAYLOAD, _DIRECTION_PAYLOAD])
    monkeypatch.setattr(song_service, "anthropic_client", stub)
    monkeypatch.setattr(lore_service, "anthropic_client", stub)

    lore = lore_service.mine_reference_lore(project)
    d1 = song_service.plan_creative_direction(project, "orchestral trap", lore_profile=lore)
    d2 = song_service.plan_creative_direction(project, "orchestral trap", lore_profile=lore)

    assert d1.core_theme == d2.core_theme
    # lore(1) + direction(1); the second plan call reused the cached file.
    assert len(stub.calls) == 2


def test_creative_direction_replanned_on_mode_switch(project, monkeypatch):
    """Switching lyrics mode must invalidate the cached brief and use the new planner.

    Regression: a direction planned in 'original' mode (unrelated to the reference
    story) was reused when the user later switched to 'rewrite', so the brief kept
    pointing the writer away from the story to retell.
    """
    from cutforge.models.story import StoryContentProfile, StoryPoint

    _write_reference_profile(project)
    lore = ReferenceLoreProfile(character="Midoriya")
    story = StoryContentProfile(
        character="Midoriya",
        logline="A quirkless boy inherits power and vows to become the top hero.",
        story_points=[StoryPoint(order=1, section="Verse 1", point="he is chosen",
                                 function="setup")],
    )

    # First plan in the default (original) mode, then re-plan in rewrite mode.
    stub = _StubAnthropic([_DIRECTION_PAYLOAD, _DIRECTION_PAYLOAD])
    monkeypatch.setattr(song_service, "anthropic_client", stub)

    d_original = song_service.plan_creative_direction(
        project, "orchestral trap", lore_profile=lore)
    assert d_original.planned_mode == "original"

    d_rewrite = song_service.plan_creative_direction(
        project, "orchestral trap", lore_profile=lore, story_profile=story)

    # The mode switch forced a re-plan (two LLM calls, not one cached).
    assert len(stub.calls) == 2
    assert d_rewrite.planned_mode == "rewrite"
    # The second call used the rewrite planner, not the original one.
    rewrite_system, _ = stub.calls[1]
    assert "RE-EXPRESS this fixed story" in rewrite_system
    # A same-mode re-plan now hits the cache (no third call).
    song_service.plan_creative_direction(
        project, "orchestral trap", lore_profile=lore, story_profile=story)
    assert len(stub.calls) == 2


def test_suggest_genres_accepts_no_blend(project, monkeypatch):
    stub = _StubAnthropic([{"character_read": "hero", "directions": [
        {"label": "Epic Trap", "style": "orchestral trap, 112 BPM", "why": "fits"}]}])
    monkeypatch.setattr(song_service, "anthropic_client", stub)

    suggestions = song_service.suggest_genres(project)
    assert suggestions.directions[0].label == "Epic Trap"
