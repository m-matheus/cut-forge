"""Narrative-structure service — extract the proven story SKELETON from reference rap(s).

This is the "follow structure" counterpart to the lore miner. It reads the
auto-transcribed reference rap(s) — written about the SAME character — and extracts the
STORY SKELETON: the ordered story beats, the section arrangement, where the hook lands,
the emotional arc and the flow/cadence pattern. When several references are given, it
synthesizes the COMMON skeleton they share — the "proven formula" the user wants to reuse.

CRITICAL BOUNDARY: this step extracts STRUCTURE, never EXPRESSION. It must never copy,
paraphrase or quote any phrase, rhyme, metaphor or hook wording. The output describes
SHAPE only; the writer later fills that skeleton with 100% original wording.

Unlike the lore profile (mined per-reference then merged), the structure is a
cross-reference SYNTHESIS produced in a single LLM pass over all transcripts, so it is
stored ONCE per run at ``project.narrative_structure_path`` — there is no merge step.

The music DNA (BPM/flow) lives in a separate profile — see ``reference_service``.
"""
from __future__ import annotations

import json

from cutforge.integrations import anthropic_client
from cutforge.models.project import VideoProject
from cutforge.models.structure import NarrativeStructureProfile
from cutforge.services import reference_service

STRUCTURE_MINER_SYSTEM_PROMPT = """\
You are a NARRATIVE STRUCTURE ANALYST for an anime-rap channel. You are given the
auto-transcribed lyrics of one or more existing raps written ABOUT the same anime/manga
character. Your ONLY job is to extract the SONG SKELETON — the proven STRUCTURE these
songs are built on.

You are NOT rewriting, translating, paraphrasing or summarizing the lyrics. You are
reverse-engineering the SHAPE: the order of story beats, how the sections are arranged,
where the hook lands, how the emotion rises and falls, and the flow/cadence pattern.

WHY THIS MATTERS
Many successful anime-rap channels reuse the SAME narrative formula for a character —
the same story beats, in the same order, with the hook in the same place and the same
emotional arc — changing only the words and the beat. Capturing that proven skeleton lets
a writer build a brand-new song on a structure that already works.

WHEN MULTIPLE TRANSCRIPTS ARE GIVEN (the important case)
Do NOT produce one skeleton per song. Find the COMMON skeleton they SHARE — the formula
all of them follow. Describe the shared spine in the beats/arrangement/arc/hook fields,
and record where the songs DIVERGE (and any beat only some of them use) in
"shared_pattern_notes". Weight what is common; do not overfit to a single song.
WHEN ONLY ONE TRANSCRIPT IS GIVEN: describe that one song's skeleton; put a note in
"shared_pattern_notes" that it comes from a single reference.

HARD BAN — STRUCTURE, NOT EXPRESSION
- NEVER copy, quote or paraphrase any phrase, line, rhyme, metaphor or hook wording.
- Every field must describe SHAPE at an abstract level. A "beat" is "establish the
  character at their lowest, doubted by everyone" — NOT the words used to say it.
- "maps_to_lore" is a SLOT LABEL naming the KIND of lore that fills a beat (e.g.
  "signature ability", "rival relationship", "tragic turning point") — never a phrase.
- If any field reads like it could be sung as a lyric, it is WRONG. Rewrite it as shape.

RULES
- The transcripts are auto-transcribed and may contain mis-heard words — describe the
  structure at a level robust to transcription errors; do not anchor beats to exact words.
- Sections: infer the arrangement (intro/verse/chorus/bridge/outro or drop-driven, etc.)
  from repetition and shifts in the transcript.
- Order the beats 1..N following the song's timeline.
- All output in English regardless of the transcripts' language.

OUTPUT FORMAT — a single valid JSON object, no markdown fences, no commentary:
{
  "character": "the character these references are about (best guess)",
  "overall_shape": "one-line summary of the skeleton (e.g. 'cold intro -> boast verse -> anthemic hook -> vulnerability bridge -> triumphant close')",
  "section_arrangement": ["Intro", "Verse 1", "Chorus", "Verse 2", "Chorus", "Bridge", "Verse 3", "Outro"],
  "beats": [
    { "order": 1, "section": "Intro", "beat": "abstract story/emotional beat, no phrasing", "function": "setup|escalation|turn|climax|resolution|hook_anchor|callback", "maps_to_lore": "kind of lore that fills this slot", "intensity": "high|medium|low" }
  ],
  "hook_placement": "where the hook lands, how often it recurs, and its narrative function",
  "emotional_arc": "how the feeling evolves across the song (feelings only, no phrasing)",
  "flow_cadence_notes": "the cadence pattern — where density/tempo rises and falls",
  "shared_pattern_notes": "what is COMMON across the references (the proven formula) and where they diverge; for a single reference, note that",
  "reference_count": 0,
  "source_titles": [],
  "confidence": "high|medium|low"
}
Every list may be empty. Return ONLY the JSON object.
"""


def _format_instruction_note(user_instruction: str) -> str:
    """Return a high-priority steering block for a structure re-extraction, or "".

    Prepended to the USER prompt so the model weights it heavily while the SYSTEM
    prompt's non-negotiable rules (structure-not-expression) still bound it.
    """
    instruction = (user_instruction or "").strip()
    if not instruction:
        return ""
    return (
        "!!! HIGH-PRIORITY USER STEERING (overrides default extraction focus) !!!\n"
        "The user is RE-EXTRACTING this structure and gave an explicit instruction. "
        "Prioritise it when deciding which beats to surface and how to shape the "
        "skeleton, while still obeying the system rules: extract STRUCTURE only (never "
        "reproduce the references' phrasing/hooks/metaphors), and describe SHAPE, not "
        "lyrics. If the instruction re-weights focus (e.g. 'emphasise the Shibuya turn "
        "in the bridge', 'give the hook more anchors'), follow it.\n"
        f"USER INSTRUCTION: {instruction}\n"
    )


def _build_user_prompt(project: VideoProject, transcripts: list[tuple[str, str]],
                       user_instruction: str = "") -> str:
    who = project.character or project.topic or "the character"
    lines: list[str] = []
    note = _format_instruction_note(user_instruction)
    if note:
        lines += [note, ""]

    lines.append(f"Character these references are about: {who}")
    if project.anime:
        lines.append(f"Anime / series: {project.anime}")

    n = len(transcripts)
    if n > 1:
        lines += [
            "",
            f"These are {n} different songs about the SAME character. Extract the COMMON, "
            "SHARED skeleton they all follow (the proven formula) — NOT one skeleton per "
            "song. Record divergences in shared_pattern_notes.",
        ]
    else:
        lines += [
            "",
            "This is a single reference. Extract its skeleton; note in "
            "shared_pattern_notes that it comes from one reference.",
        ]

    lines.append("")
    for i, (title, transcript) in enumerate(transcripts):
        header = f"=== REFERENCE {i + 1}" + (f": {title}" if title else "") + " ==="
        lines += [header, transcript, ""]

    lines += [
        "Extract the narrative structure (SHAPE only, never phrasing). Return only valid JSON.",
    ]
    return "\n".join(lines)


def extract_structure_profile(project: VideoProject, *, refresh: bool = False,
                              user_instruction: str = "",
                              on_log=None) -> NarrativeStructureProfile | None:
    """Synthesize the shared narrative skeleton from all references (cached per run).

    Reads every reference's transcript (from the music profile), feeds them together to
    the structure miner in ONE pass, and persists the result to
    ``project.narrative_structure_path``. Returns ``None`` when there is no reference
    transcript to analyze. Reused on subsequent calls unless ``refresh=True``.
    ``user_instruction`` optionally steers what the miner emphasises (used on refresh).
    """
    log = on_log or (lambda _m: None)

    profiles = reference_service.load_all_reference_profiles(project)
    transcripts: list[tuple[str, str]] = []
    for p in profiles:
        transcript = (p.get("transcript") or "").strip()
        if transcript:
            transcripts.append((p.get("source_title", ""), transcript))

    if not transcripts:
        log("No reference transcript to analyze — skipping structure extraction.")
        return None

    path = project.narrative_structure_path
    if path.exists() and not refresh:
        data = json.loads(path.read_text(encoding="utf-8"))
        return NarrativeStructureProfile(**data)

    user_prompt = _build_user_prompt(project, transcripts, user_instruction)

    log(f"Analyzing {len(transcripts)} reference transcript(s) for the shared skeleton…")
    data = anthropic_client.complete_json(STRUCTURE_MINER_SYSTEM_PROMPT, user_prompt)
    # Provenance is authoritative from our side, not the model's guess.
    data["reference_count"] = len(transcripts)
    data["source_titles"] = [t for t, _ in transcripts if t]
    profile = NarrativeStructureProfile(**data)

    project.run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    log(
        f"Structure extracted: {len(profile.beats)} beats, "
        f"{len(profile.section_arrangement)} sections (from {len(transcripts)} reference(s))."
    )
    return profile


def load_structure_profile(project: VideoProject) -> NarrativeStructureProfile | None:
    """Return the cached narrative-structure profile for this run, or None."""
    path = project.narrative_structure_path
    if path.exists():
        return NarrativeStructureProfile(**json.loads(path.read_text(encoding="utf-8")))
    return None
