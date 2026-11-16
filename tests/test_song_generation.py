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
    "energy_map": "Intro whispered → V1 calm → Chorus sung → Break → Drop → Outro quiet",
    "callback": "the scribbled hero notebook, returned in the outro",
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


# ---------------------------------------------------------------------------
# Sound chosen by the brief (no genre-suggestion step)
# ---------------------------------------------------------------------------

def test_planner_picks_the_sound_when_no_genre_is_forced(project, monkeypatch):
    direction = {**_DIRECTION_PAYLOAD, "sonic_direction": "rap rock, 160 BPM"}
    stub = _StubAnthropic([direction, _SONG_PAYLOAD])
    monkeypatch.setattr(song_service, "anthropic_client", stub)

    song_service.generate_package(project)

    planner_system, planner_user = stub.calls[0]
    assert "SONIC DIRECTION" in planner_system
    assert "Genre: not forced" in planner_user
    assert "GENRE SELECTION" in planner_user          # archetypes, since no reference
    _, writer_user = stub.calls[1]
    assert "Chosen genre / style direction: rap rock, 160 BPM" in writer_user


def test_forced_genre_reaches_planner_and_invalidates_cache(project, monkeypatch):
    stub = _StubAnthropic([
        {**_DIRECTION_PAYLOAD, "sonic_direction": "dark trap, 140 BPM"},
        {**_DIRECTION_PAYLOAD, "sonic_direction": "uk drill, 142 BPM"},
    ])
    monkeypatch.setattr(song_service, "anthropic_client", stub)

    first = song_service.plan_creative_direction(project)
    assert song_service.plan_creative_direction(project) == first   # cached
    forced = song_service.plan_creative_direction(project, "uk drill")

    assert len(stub.calls) == 2                       # the forced genre re-planned
    assert "User-forced genre (sonic_direction MUST be in it): uk drill" in stub.calls[1][1]
    assert forced.planned_genre == "uk drill"


def test_catalog_styles_reach_the_planner(project, monkeypatch):
    other = VideoProject.create(run_id="20261001-gojo")
    other.lyrics_path.write_text("[Intro]\nEnkai.\n[Verse 1]\nline", encoding="utf-8")
    other.suno_prompt_path.write_text(json.dumps({"style": "gritty boom bap, 90 BPM"}),
                                      encoding="utf-8")
    stub = _StubAnthropic([{**_DIRECTION_PAYLOAD, "sonic_direction": "x"}])
    monkeypatch.setattr(song_service, "anthropic_client", stub)

    song_service.plan_creative_direction(project)

    assert "20261001-gojo: gritty boom bap, 90 BPM" in stub.calls[0][1]



# ---------------------------------------------------------------------------
# Emotional craft: flow target, energy map, sung hook
# ---------------------------------------------------------------------------

def test_flow_target_is_measured_from_real_lyric_lines():
    transcript = "\n".join(["Uchiha blood born from legends"] * 10 + ["Then there's me"] * 20)
    wpl, total, note = song_service._flow_target(
        {"lyrics_source": "subtitles", "transcript": transcript})
    assert wpl == 3                      # median line length of the reference
    assert total == 450                  # 0.8 × 110 words, clamped up to the minimum
    assert "measured from the reference" in note


def test_flow_target_ignores_whisper_segments_and_clamps():
    long_lines = "\n".join(["word " * 12] * 30)
    assert song_service._flow_target({"lyrics_source": "whisper", "transcript": long_lines})[2] \
        == "channel default"
    wpl, _, _ = song_service._flow_target({"lyrics_source": "manual", "transcript": long_lines})
    assert wpl == 7                      # clamped: a 12-word "line" is never a target


def test_writer_gets_flow_target_energy_map_and_craft_rules(project, monkeypatch):
    stub = _StubAnthropic([_DIRECTION_PAYLOAD, _SONG_PAYLOAD])
    monkeypatch.setattr(song_service, "anthropic_client", stub)

    song_service.generate_package(project, "orchestral trap")

    writer_system, writer_user = stub.calls[1]
    assert "FLOW TARGET (channel default)" in writer_user
    assert "ENERGY MAP" in writer_user and "Break → Drop" in writer_user
    assert "CALLBACK" in writer_user
    assert "EMOTIONAL CRAFT" in writer_system
    # Singing is no longer excluded by default — the sung hook is the default.
    assert '"singing, melodic vocals, auto-tune, sung chorus"' not in writer_system


# ---------------------------------------------------------------------------
# Suno formatting, rap craft and coherence rules
# ---------------------------------------------------------------------------

def test_writer_gets_the_new_rules_in_every_mode():
    for system in (song_service.GENERATE_SYSTEM_PROMPT_NO_REF,
                   song_service.GENERATE_SYSTEM_PROMPT_WITH_REF,
                   song_service.GENERATE_SYSTEM_PROMPT_STRUCTURE_WITH_REF,
                   song_service.GENERATE_SYSTEM_PROMPT_REWRITE_WITH_REF):
        assert "RAP CRAFT" in system and "8–14 syllables" in system
        assert "COHERENCE" in system and "PRONUNCIATION" in system
        assert '"pronunciations"' in system
        assert "any artist, band or producer name" in system


def test_prompts_no_longer_ask_for_sound_effects_in_parentheses():
    prompts = (song_service.GENERATE_SYSTEM_PROMPT_NO_REF, song_service.INTRO_REWRITE_SYSTEM_PROMPT,
               song_service.CREATIVE_DIRECTION_SYSTEM_PROMPT)
    for system in prompts:
        for old in ("(stone cracking)", "(wind)", "(chains rattle)", "(bell tolls)",
                    "sound ad-lib", "ad-libs in (parentheses)"):
            assert old not in system


def test_flow_target_leaves_rap_verses_to_rap_craft():
    block = song_service._format_flow_target(3.0, 600, "channel default")
    assert "Sung, quiet and peak lines: ~3 words" in block
    assert "Rap verses: 8–14 syllables" in block and "most 8–10" in block
    assert "Never merge" not in block


def test_generated_lyrics_go_through_the_checks(project, monkeypatch):
    long_style = "x" * 201
    song = {**_SONG_PAYLOAD, "style": long_style, "lyrics":
            "[Intro]\nEnkai.\n(organ swells)\n[Verse 1: aggressive rap, with a very long tag]\n"
            "Original words (yeah)"}
    stub = _StubAnthropic([_DIRECTION_PAYLOAD, song])
    monkeypatch.setattr(song_service, "anthropic_client", stub)
    logged = []

    pkg = song_service.generate_package(project, "orchestral trap", on_log=logged.append)

    assert "[Organ swells]" in pkg.lyrics and "(organ swells)" not in pkg.lyrics
    assert project.lyrics_path.read_text(encoding="utf-8") == pkg.lyrics
    saved = json.loads(project.suno_prompt_path.read_text(encoding="utf-8"))
    assert any("Tag longa" in w for w in saved["warnings"])
    assert any("201 caracteres" in w for w in saved["warnings"])
    assert any("(organ swells) virou [Organ swells]" in m for m in logged)
    assert any("(yeah)" in m for m in logged)


# ---------------------------------------------------------------------------
# Pronunciation: respelled for Suno only
# ---------------------------------------------------------------------------

def test_suno_lyrics_respells_words_but_not_tags():
    sheet = "[Drop: Auswählen]\nAuswählen.\nYhwach's army\nYhwach Prime rises\nNot Yhwachian"
    out = song_service.suno_lyrics(sheet, {"Auswählen": "Ows-vay-len", "Yhwach": "You-ha-bak",
                                           "Yhwach Prime": "You-ha-bak Prime!", "Empty": " "})
    assert out == ("[Drop: Auswählen]\nOws-vay-len.\nYou-ha-bak's army\n"
                   "You-ha-bak Prime! rises\nNot Yhwachian")
    assert song_service.suno_lyrics(sheet, {}) == sheet


def test_canonical_lyrics_undoes_the_respelling():
    pronunciations = {"Auswählen": "Ows-vay-len", "Yhwach": "You-ha-bak", "Empty": " "}
    sheet = "[Drop: Auswählen]\nAuswählen.\nYhwach's army"
    assert song_service.canonical_lyrics(
        song_service.suno_lyrics(sheet, pronunciations), pronunciations) == sheet
    assert song_service.canonical_lyrics(sheet, {}) == sheet


def test_canonical_names_stay_in_lyrics_and_the_api_serves_the_suno_text(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from cutforge.api.app import create_app
    from cutforge.config import settings as settings_mod

    monkeypatch.setattr(settings_mod.Settings, "output_dir",
                        property(lambda self: tmp_path / "output"))
    settings_mod.get_settings.cache_clear()
    project = VideoProject.create(run_id="20261103-yhwach", character="Yhwach")
    song = {**_SONG_PAYLOAD, "lyrics": "[Drop]\nAuswählen.",
            "pronunciations": {"Auswählen": "Ows-vay-len"}}
    monkeypatch.setattr(song_service, "anthropic_client",
                        _StubAnthropic([_DIRECTION_PAYLOAD, song]))

    song_service.generate_package(project, "orchestral trap")

    assert project.lyrics_path.read_text(encoding="utf-8") == "[Drop]\nAuswählen."
    output = TestClient(create_app()).get(f"/api/runs/{project.run_id}/output/lyrics").json()
    assert output["lyrics"] == "[Drop]\nAuswählen."
    assert output["suno_lyrics"] == "[Drop]\nOws-vay-len."
    assert output["pronunciations"] == {"Auswählen": "Ows-vay-len"}


def test_runs_from_before_pronunciations_still_copy_their_lyrics(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from cutforge.api.app import create_app
    from cutforge.config import settings as settings_mod

    monkeypatch.setattr(settings_mod.Settings, "output_dir",
                        property(lambda self: tmp_path / "output"))
    settings_mod.get_settings.cache_clear()
    project = VideoProject.create(run_id="20261011-obito", character="Obito")
    project.lyrics_path.write_text("[Verse 1]\nline", encoding="utf-8")
    project.suno_prompt_path.write_text(json.dumps({"style": "trap"}), encoding="utf-8")

    output = TestClient(create_app()).get(f"/api/runs/{project.run_id}/output/lyrics").json()

    assert output["suno_lyrics"] == "[Verse 1]\nline"
    assert output["warnings"] == []


# ---------------------------------------------------------------------------
# Optional review pass
# ---------------------------------------------------------------------------

_REVIEW_PAYLOAD = {
    "lyrics": "[Intro]\nReviewed line\n(wind)\n[Verse 1: calm rap]\nFixed words",
    "pronunciations": {"Blackwhip": "Black-whip"},
    "changes": ["Verse 1: moved the event to the right arc (FACTS)"],
}


def test_review_is_off_by_default(project, monkeypatch):
    stub = _StubAnthropic([_DIRECTION_PAYLOAD, _SONG_PAYLOAD])
    monkeypatch.setattr(song_service, "anthropic_client", stub)

    pkg = song_service.generate_package(project, "orchestral trap")

    assert len(stub.calls) == 2
    assert pkg.lyrics == _SONG_PAYLOAD["lyrics"]


def test_review_corrects_the_draft_and_the_checks_run_on_the_result(project, monkeypatch):
    _write_reference_profile(project)
    draft = {**_SONG_PAYLOAD, "lyrics": "[Verse 1: " + "very " * 7 + "long]\nDraft words"}
    stub = _StubAnthropic([_LORE_PAYLOAD, _DIRECTION_PAYLOAD, draft, _REVIEW_PAYLOAD])
    monkeypatch.setattr(song_service, "anthropic_client", stub)
    monkeypatch.setattr(lore_service, "anthropic_client", stub)
    logged = []

    pkg = song_service.generate_package(project, "orchestral trap", review=True,
                                        on_log=logged.append)

    review_system, review_user = stub.calls[3]
    assert "final editor" in review_system
    assert "COHERENCE" in review_system and "RAP CRAFT" in review_system
    assert "CHARACTER LORE" in review_user and "Detroit Smash" in review_user
    assert "CREATIVE DIRECTION" in review_user
    assert "Draft words" in review_user
    assert "Tag longa" in review_user            # the automatic checks steer the review
    assert "Reviewed line" in pkg.lyrics
    assert "[Wind]" in pkg.lyrics                # the checks have the last word
    assert pkg.pronunciations == {"Blackwhip": "Black-whip"}
    assert any("moved the event" in m for m in logged)


@pytest.mark.parametrize("outcome", [ValueError("not JSON"), {"lyrics": "  "}])
def test_failed_review_keeps_the_draft(project, monkeypatch, outcome):
    class _Flaky(_StubAnthropic):
        def complete_json(self, system, user_prompt, **kwargs):
            if system == song_service.REVIEW_SYSTEM_PROMPT and isinstance(outcome, Exception):
                raise outcome
            return outcome if system == song_service.REVIEW_SYSTEM_PROMPT \
                else super().complete_json(system, user_prompt, **kwargs)

    song = {**_SONG_PAYLOAD, "pronunciations": {"Blackwhip": "Black-whip"}}
    monkeypatch.setattr(song_service, "anthropic_client", _Flaky([_DIRECTION_PAYLOAD, song]))
    logged = []

    pkg = song_service.generate_package(project, "orchestral trap", review=True,
                                        on_log=logged.append)

    assert pkg.lyrics == _SONG_PAYLOAD["lyrics"]
    assert pkg.pronunciations == {"Blackwhip": "Black-whip"}
    assert any("keeping the draft" in m for m in logged)


def test_brief_without_energy_map_is_replanned(project, monkeypatch):
    old = {k: v for k, v in _DIRECTION_PAYLOAD.items() if k != "energy_map"}
    stub = _StubAnthropic([{**old, "sonic_direction": "dark trap, 140 BPM"},
                           {**_DIRECTION_PAYLOAD, "sonic_direction": "dark trap, 140 BPM"}])
    monkeypatch.setattr(song_service, "anthropic_client", stub)

    song_service.plan_creative_direction(project)
    song_service.plan_creative_direction(project)
    assert len(stub.calls) == 2          # the pre-energy-map brief was not reused
