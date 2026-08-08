"""Story-content service — extract the shared STORY from reference rap(s).

This is the "rewrite the story" counterpart to the narrative-structure miner. Where the
structure miner extracts abstract SHAPE (and is forbidden from touching content), this
miner extracts the STORY CONTENT — the logline, the ordered story points, the key ideas
and imagery, the hook concept and the emotional arc — the things the user wants to KEEP.
When several references about the same character are given, it synthesizes the COMMON
story they share.

CRITICAL BOUNDARY — the mirror of the structure miner's:
- The structure miner may describe shape but never content.
- This miner MUST preserve content (what is said, in order) but must NEVER copy,
  paraphrase or quote any verbatim rhyme, verse or hook wording. It records the IDEA, not
  the words. ``hook_concept`` is the concept of the reference's hook, never its lyric.

Like the structure profile (and unlike the per-reference-then-merged lore profile), the
story is a cross-reference SYNTHESIS produced in a single LLM pass over all transcripts,
so it is stored ONCE per run at ``project.story_content_path`` — there is no merge step.

The music DNA (BPM/flow) lives in a separate profile — see ``reference_service``.
"""
from __future__ import annotations

import json

from cutforge.integrations import anthropic_client
from cutforge.models.project import VideoProject
from cutforge.models.story import StoryContentProfile
from cutforge.services import reference_service

STORY_MINER_SYSTEM_PROMPT = """\
You are a STORY ANALYST for an anime-rap channel. You are given the transcribed lyrics of
one or more existing raps written ABOUT the same anime/manga character. Your ONLY job is
to extract the SHARED STORY these songs tell — WHAT they say about the character, in what
order, and the ideas and imagery they carry.

WHY THIS MATTERS
The channel wants to make a NEW rap that tells the SAME story with the SAME ideas as the
references, but re-written with DIFFERENT rhymes, a different rhythm and (optionally) a
different hook. To do that, a writer needs the story CONTENT to preserve — not the shape,
and not the words. You capture the content; a later step rewrites the expression.

WHEN MULTIPLE TRANSCRIPTS ARE GIVEN (the important case)
Do NOT produce one story per song. Find the COMMON story they SHARE — the narrative,
points and ideas all of them tell. Describe the shared story in the logline/story_points/
ideas/arc fields, and record where the songs DIVERGE (or a point only some of them make)
in "shared_pattern_notes". Weight what is common; do not overfit to a single song.
WHEN ONLY ONE TRANSCRIPT IS GIVEN: describe that one song's story; put a note in
"shared_pattern_notes" that it comes from a single reference.

HARD BAN — CONTENT, NOT EXPRESSION
- PRESERVE what is said and its order. This is the OPPOSITE of a shape-only analysis:
  the story points, ideas and imagery ARE the point.
- But NEVER copy, quote, translate or paraphrase any rhyme, verse, line or hook WORDING.
  Describe the IDEA in your own neutral words, not the words the reference used.
- A story point is "he vows to surpass the rival who humiliated him" — NOT the lyric that
  says it, and NOT a light reword of that lyric that keeps its rhyme or cadence.
- "key_images" are concrete images/ideas to keep (e.g. "the broken blade", "the empty
  throne") described plainly — never a quoted phrase.
- "hook_concept" is the IDEA of the hook/refrain (what it's about, its function) — never
  its actual sung words.
- If any field reads like it could be sung as a lyric, it is WRONG. Rewrite it as neutral
  description.

RULES
- The transcripts may contain mis-heard words — describe the story at a level robust to
  transcription errors; do not anchor points to exact words.
- Sections: infer the arrangement (intro/verse/chorus/bridge/outro) from repetition and
  shifts, and tag each story point with the section it lives in.
- Order the story points 1..N following the song's timeline.
- All output in English regardless of the transcripts' language.

OUTPUT FORMAT — a single valid JSON object, no markdown fences, no commentary:
{
  "character": "the character these references are about (best guess)",
  "logline": "the whole shared story in 1-2 lines (idea, not phrasing)",
  "story_points": [
    { "order": 1, "section": "Verse 1", "point": "what is said here, as a neutral idea", "key_images": ["concrete image/idea to keep", "..."], "function": "setup|escalation|turn|climax|resolution|hook_anchor|callback" }
  ],
  "hook_concept": "the IDEA of the hook/refrain — what it's about, never its words",
  "themes": ["central theme", "..."],
  "key_ideas": ["central message/idea shared across the references", "..."],
  "emotional_arc": "how the feeling evolves across the song (feelings only, no phrasing)",
  "shared_pattern_notes": "what is COMMON across the references (the shared story) and where they diverge; for a single reference, note that",
  "reference_count": 0,
  "source_titles": [],
  "confidence": "high|medium|low"
}
Every list may be empty. Return ONLY the JSON object.
"""


def _format_instruction_note(user_instruction: str) -> str:
    """Return a high-priority steering block for a story re-extraction, or "".

    Prepended to the USER prompt so the model weights it heavily while the SYSTEM
    prompt's non-negotiable rules (content-not-expression) still bound it.
    """
    instruction = (user_instruction or "").strip()
    if not instruction:
        return ""
    return (
        "!!! HIGH-PRIORITY USER STEERING (overrides default extraction focus) !!!\n"
        "The user is RE-EXTRACTING this story and gave an explicit instruction. "
        "Prioritise it when deciding which story points and ideas to surface, while still "
        "obeying the system rules: preserve CONTENT (the story/ideas), but NEVER reproduce "
        "the references' phrasing/rhymes/hook wording. If the instruction re-weights focus "
        "(e.g. 'emphasise the betrayal arc', 'keep the origin-story details'), follow it.\n"
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
            f"These are {n} different songs about the SAME character telling the SAME "
            "story. Extract the COMMON story they SHARE (the narrative, points and ideas "
            "all of them tell) — NOT one story per song. Record divergences in "
            "shared_pattern_notes.",
        ]
    else:
        lines += [
            "",
            "This is a single reference. Extract its story; note in shared_pattern_notes "
            "that it comes from one reference.",
        ]

    lines.append("")
    for i, (title, transcript) in enumerate(transcripts):
        header = f"=== REFERENCE {i + 1}" + (f": {title}" if title else "") + " ==="
        lines += [header, transcript, ""]

    lines += [
        "Extract the shared story (CONTENT to preserve, never phrasing). Return only valid JSON.",
    ]
    return "\n".join(lines)


def extract_story_profile(project: VideoProject, *, refresh: bool = False,
                          user_instruction: str = "",
                          on_log=None) -> StoryContentProfile | None:
    """Synthesize the shared story from all references (cached per run).

    Reads every reference's transcript (from the music profile), feeds them together to
    the story miner in ONE pass, and persists the result to ``project.story_content_path``.
    Returns ``None`` when there is no reference transcript to analyze. Reused on subsequent
    calls unless ``refresh=True``. ``user_instruction`` optionally steers what the miner
    emphasises (used on refresh).
    """
    log = on_log or (lambda _m: None)

    profiles = reference_service.load_all_reference_profiles(project)
    transcripts: list[tuple[str, str]] = []
    for p in profiles:
        transcript = (p.get("transcript") or "").strip()
        if transcript:
            transcripts.append((p.get("source_title", ""), transcript))

    if not transcripts:
        log("No reference transcript to analyze — skipping story extraction.")
        return None

    path = project.story_content_path
    if path.exists() and not refresh:
        data = json.loads(path.read_text(encoding="utf-8"))
        return StoryContentProfile(**data)

    user_prompt = _build_user_prompt(project, transcripts, user_instruction)

    log(f"Analyzing {len(transcripts)} reference transcript(s) for the shared story…")
    data = anthropic_client.complete_json(STORY_MINER_SYSTEM_PROMPT, user_prompt)
    # Provenance is authoritative from our side, not the model's guess.
    data["reference_count"] = len(transcripts)
    data["source_titles"] = [t for t, _ in transcripts if t]
    profile = StoryContentProfile(**data)

    project.run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    log(
        f"Story extracted: {len(profile.story_points)} points, "
        f"{len(profile.key_ideas)} key ideas (from {len(transcripts)} reference(s))."
    )
    return profile


def load_story_profile(project: VideoProject) -> StoryContentProfile | None:
    """Return the cached story-content profile for this run, or None."""
    path = project.story_content_path
    if path.exists():
        return StoryContentProfile(**json.loads(path.read_text(encoding="utf-8")))
    return None
