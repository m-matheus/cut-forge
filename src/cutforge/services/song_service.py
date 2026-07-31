"""Song generation service — suggests genre directions and generates the full Suno package.

The system prompts are tuned to how Suno AI (v4.5/v5) actually reads its two fields:
- STYLE = the sonic world only (audio descriptors), never visual/video terms.
- LYRICS = words + bracketed structure/delivery tags that Suno obeys.

The model is framed as an expert anime-rap producer (the lane of BASARA, M4RKIM,
ANIRAP, Rustage, 7 Minutoz) so the reference track and archetype drive a precise,
Suno-ready style string instead of vague "epic anime song" prompts.
"""
from __future__ import annotations

import json

from cutforge.integrations import anthropic_client
from cutforge.models.project import VideoProject
from cutforge.models.song import GenreDirection, GenreSuggestions, SongPackage


# --- Shared Suno craft knowledge, embedded in every generation prompt ---------
# This is the distilled "how Suno actually behaves" + "what the anime-rap scene
# sounds like" reference. Kept in one constant so suggest and generate stay in sync.

_SUNO_STYLE_RULES = """\
HOW TO WRITE THE SUNO "STYLE" STRING (this is where most bad songs are lost)
Suno generates AUDIO, not video. The style field describes ONLY what you HEAR.

Formula — order matters, Suno weights the leading tokens most:
  <dominant subgenre>, <mood/energy>, <vocal: gender + delivery>, <2-4 signature
  instruments>, <production word>, <one BPM number> BPM
- Lead with a SPECIFIC subgenre (e.g. "dark trap", "UK drill", "orchestral trap"),
  never bare "hip-hop" / "rap" — bare genres produce generic mush.
- 8-15 comma-separated descriptors, roughly 120-200 characters. Not prose, not a
  wall of tags. Every descriptor must control a real production layer.
- Exactly ONE BPM number (e.g. "140 BPM"), never a range.
- Specify vocals as character + delivery: "aggressive rapped male vocal",
  "anthemic gang-vocal hook", "vulnerable autotuned sung-rap".
- Name 2-4 concrete instruments with qualities: "distorted 808 bass",
  "soaring strings", "dark minor piano", "sliding drill hi-hats".

FORCE RAP (Suno's default drifts to melodic singing):
- Put a rap-delivery token in the style ("aggressive rap vocals", "spoken flow",
  "fast bars", "confident rap flow") AND use [Rap]/[Rapped] tags in the lyrics.
- Only use melodic vocal words ("autotune", "melodic", "sung") when you genuinely
  want a sung hook.

BANNED IN THE STYLE STRING — these are VISUAL/meta terms Suno cannot hear and they
poison the generation. NEVER put any of them in "style":
  AMV, anime edit, anime rap, "cinematic anime rap", anime AMV, music video,
  montage, edit, clip, scene, 4K, visuals, "epic anime", the character/anime name,
  "make it hard", "best song", or any words about the video or the topic.
Translate that intent into SOUND instead. "Cinematic anime rap / AMV" becomes
  "epic orchestral trap, big cinematic drums, epic choir, hard 808s".
The "anime" feel comes from orchestral/choir layering + the archetype's mood, NOT
from the word "anime". Do not mix contradictory genres (e.g. "trap, lofi chill").
"""

_ANIME_RAP_RANGE = """\
ANIME-RAP RANGE (map the character's archetype to a genre lane; the reference,
when present, OVERRIDES this — see below). All descriptors are SOUND, no visuals.

1. Dark villain        -> dark trap / drill / phonk      | 130-150 BPM
   distorted 808, sliding hi-hats, dark violin + choir stabs, minor piano
   | deep menacing rapped male vocal, reverb-drenched ad-libs       (lane: M4RKIM)

2. Unstoppable hero /   -> orchestral / epic hybrid trap  | 140-160 BPM
   power-up             booming 808s, crisp trap hats, soaring strings + brass,
   epic choir, timpani | anthemic rap + big sung/gang-vocal hook
                                        (lane: Rustage, GameboyJones, Sensei Beats)

3. Hot-blooded shonen  -> rap rock / trap metal           | 150-170 BPM
   distorted guitars, double-kick, heavy 808, breakdown | shouted/screamed rap
                                     (lane: 7 Minutoz roots, None Like Joshua)

4. Tragic / emotional  -> melodic trap / emo rap          | 130-150 BPM
   sad piano, clean guitar, warm pads, soft 808, laid-back hats
   | vulnerable autotuned sung-rap, sung hook       (lane: Divide, BASARA melodic)

5. Godlike / ancient   -> cinematic hybrid orchestral     | 90 BPM (or 150 half-time)
   war drums / taiko, full orchestra, epic choir, braams, deep sub hits
   | commanding reverbed rap over an orchestral bed, chant textures

The scene sweet spot is ~140-160 BPM. What makes these songs HIT: character-POV
writing with real lore; the biggest drop timed to the character's signature moment;
orchestral/choir layering for scale; and one anthemic, sing-along hook.
"""


SUGGEST_SYSTEM_PROMPT = f"""\
You are a producer for an anime-rap channel — the lane of BASARA, M4RKIM, ANIRAP,
Rustage and 7 Minutoz. Given a character or matchup, propose distinct GENRE/VIBE
directions for an original song that matches that character's personality and power
fantasy, and that a producer in this scene would actually make.

{_ANIME_RAP_RANGE}

For the given character(s), propose 3 directions. Each must genuinely fit THAT
character — tie the genre to their personality, not a generic "epic anime song".

OUTPUT FORMAT
Return a single valid JSON object. No markdown fences, no commentary.

{{
  "character_read": "1-2 sentences on the character's vibe that should drive the music",
  "directions": [
    {{
      "label": "Short genre name (e.g. 'Cold Drill')",
      "style": "A Suno-ready style string — see the STYLE rules below",
      "why": "One sentence: why this fits the character"
    }}
  ]
}}

{_SUNO_STYLE_RULES}

Rules:
- Exactly 3 directions, each clearly different from the others.
- Each style must follow the formula and the BANNED-terms rule above.
- All descriptors in English (Suno reads English style strings best).
"""


GENERATE_SYSTEM_PROMPT = f"""\
You are an expert songwriter and music producer for an anime-rap channel — the lane
of BASARA, M4RKIM, ANIRAP, Rustage and 7 Minutoz. You write an original song about a
specific anime character (or matchup) that will be generated on Suno AI and played
over AMV footage on YouTube. You know exactly how Suno reads its Style and Lyrics
fields and you exploit that to get a hard, clean, on-genre track — not generic mush.

You produce a COMPLETE Suno package: a style string, an exclude string, structured
lyrics, and a title.

TONE & CONTENT
- The song is FROM or ABOUT the character — capture their personality, powers and arc
  through vivid, specific imagery. Not a plot summary — a hype anthem / character piece.
- Reference the character's actual abilities, iconic moments and traits, but through
  metaphor and attitude, not exposition.
- Match the chosen genre's energy in word choice and rhythm.
- Keep it hype and quotable — the hook should be something viewers scream in the comments.
- Do NOT use trademarked catchphrases verbatim; evoke them instead.

{_ANIME_RAP_RANGE}

{_SUNO_STYLE_RULES}

THE "EXCLUDE" FIELD
Return a short comma-separated list of styles to EXCLUDE, to stop Suno from drifting.
For a hard rap track this is typically "singing, melodic vocals, auto-tune, sung chorus";
for an emotional/sung track, exclude the opposite ("aggressive, harsh, distorted"). Tailor
it to the chosen genre. Keep it short (3-6 terms).

LYRICS STRUCTURE
Suno obeys bracketed tags in the LYRICS field (structure tags most reliably). Write a
song of AT LEAST ~2.5 minutes of material. A reliable arrangement:
  [Intro]        short — sets the mood (an atmospheric line or 2)
  [Verse 1]      establish the character, their world, their attitude (4-8 lines)
  [Chorus]       the hook — punchy, repeatable, the emotional core (4-6 lines)
  [Verse 2]      escalate: power, a defining moment, a turn (4-8 lines)
  [Chorus]       repeat the hook
  [Bridge]       shift energy: a threat, a vow, or quiet-before-storm (2-4 lines)
  [Verse 3] or [Outro]   final push / hard closing line
This is a STARTING POINT, not a mold — vary it to fit the chosen genre and (when given)
the reference's own arrangement:
  - drop-driven / EDM-ish beat -> add [Build] then [Drop], time the drop to the
    character's signature moment.
  - emotional / melodic -> fewer, longer sung hooks, maybe an [Instrumental Break];
    drop the gang-vocal chant.
  - cypher / anthem -> verse-per-character feel, [Gang Vocals] + [Chant] on the hook.
  - rap rock / boom-bap -> trade-off verses, sing-along chant hook.
Section-length caps (over-long sections make Suno rush the delivery): verses 4-8 lines,
choruses 4-6, bridges 2-4. Put the strongest line FIRST in each section.
For a "vs" / matchup song, alternate perspectives and make the chorus the clash.

VOCAL & DELIVERY TAGS (put these in the LYRICS, on their own lines)
- [Rap] / [Rapped] on each verse to force rapping (not singing).
- [Male Vocal] / [Female Vocal] to pin the voice.
- [Gang Vocals] / [Chant] on the hook for anthemic power; [Call and Response] on a bridge.
- [Fast Rap] / [Double Time] / [Slow Flow] for cadence.
- Ad-libs go in (parentheses) at line ends: (yeah) (uh) (gang). Anything NOT bracketed
  or parenthesized WILL be sung — keep every non-lyric cue bracketed.

LINE RULES (these lyrics also become on-screen karaoke captions)
- Each line should be a natural caption length — roughly 4-9 words. Avoid very long lines.
- One idea per line. Clean, punchy phrasing reads better on screen.
- Real words only — no "yeah yeah" filler padding beyond intentional (parenthetical) ad-libs.

OUTPUT FORMAT
Return a single valid JSON object. No markdown fences, no commentary.

{{
  "title": "Song title — short and evocative (just the song name, e.g. 'Shadow Sovereign')",
  "style": "The Suno STYLE string — audio only, following the formula and banned-terms rule",
  "exclude": "Short comma-separated Suno Exclude-Styles list",
  "lyrics": "Full structured lyrics as one string. Use \\n for line breaks; each [tag] on its own line.",
  "suno_tips": "One short line of advice for the Suno generation"
}}

Rules:
- style must obey the STYLE rules: no visual/meta terms, one BPM number, specific subgenre first.
- lyrics MUST include section tags each on its own line, plus [Rap]/[Male|Female Vocal] tags.
- Target at least ~2.5 minutes of lyrics — enough sections, not padded.
"""


SUGGEST_MOOD_SYSTEM_PROMPT = """\
You are a creative director for an anime-rap channel (BASARA / M4RKIM / Rustage lane).
Given an anime character (or matchup), describe the MOOD / VIBE that an original song
about them should have — the emotional atmosphere that drives the music, caption colors
and thumbnail.

Return a SINGLE short line: 3-6 comma-separated descriptors capturing the character's
energy (e.g. "dark, cold, ominous power, shadow army" or "hot-blooded, explosive,
heroic, hype"). No commentary, no quotes, no markdown — just the descriptors line.
"""


# --- Reference-inspiration addenda, keyed by how much CONTENT may be borrowed ---
# Every level shares one hard, non-negotiable anti-plagiarism floor; the levels differ
# only in how much theme / imagery / hook-shape / phrasing may be reused. NOTE: the
# GENRE always follows the reference regardless of level — content_blend controls only
# the lyrical CONTENT, never the musical lane.

_REFERENCE_HARD_FLOOR = """\
HARD FLOOR (applies at EVERY level — never violate):
- NEVER reproduce any line, hook, or distinctive phrase from the reference transcript
  verbatim, and never merely swap a word or two — that is still copying.
- NEVER reuse the reference's proper nouns, character names, or brand catchphrases.
- The transcript is auto-transcribed and may contain mis-heard words.
ALWAYS match the reference's RHYTHM:
- The tempo/energy feel (target roughly the given BPM) and rhythmic density.
- The flow: line length and syllables-per-bar. Faster BPM / higher words-per-second =>
  shorter, denser lines; slower => more spacious lines.
Put the reference's BPM (one number) in the Suno style string.
"""

_REFERENCE_CONTENT_RULES = {
    "rhythm": """\
CONTENT: Use the reference ONLY as a rhythm/energy signal. Do NOT borrow its themes,
imagery, message, or hook structure — write the character piece entirely from your own
ideas. The transcript is vibe-only; treat its words as off-limits.""",
    "light": """\
CONTENT (light borrow): You MAY echo 1-2 of the reference's core themes and its overall
tone, but expressed in completely fresh words. Do NOT copy its specific images or
metaphors — invent your own that fit the CutForge character.""",
    "moderate": """\
CONTENT (moderate borrow): You MAY reuse the reference's core themes, its semantic field
of imagery/metaphors, and its hook shape (e.g. a short repeating chorus, call-and-response,
a chant) — all rewritten around the CutForge character with fresh wording. No verbatim
phrases; the images should clearly rhyme with the reference's world without quoting it.""",
    "strong": """\
CONTENT (strong borrow): You MAY take the reference's themes, imagery, message and hook
structure, follow its section arc, AND lift its most striking phrases as inspiration —
but you MUST heavily rewrite each one (swap the vocabulary, the referents, and the angle
so it reads as a new line about the CutForge character). Aim for "unmistakably inspired by,
never a copy." The hard floor above still holds: no literal text, no proper nouns reused.""",
}


def _reference_addendum(level: str) -> str:
    """Return the reference system-prompt addendum for a content-blend level."""
    rules = _REFERENCE_CONTENT_RULES.get(level, _REFERENCE_CONTENT_RULES["rhythm"])
    return (
        "\n\nREFERENCE INSPIRATION\n"
        "You are given a reference rap the user admires: its transcript, BPM, and flow "
        "metrics. Produce a STANDALONE, original composition about the CutForge character.\n"
        "GENRE FOLLOWS THE REFERENCE: infer the reference's subgenre/lane from its tempo, "
        "energy and transcript, and make the song sit in THAT lane (the chosen genre "
        "direction already reflects it). The archetype range above only fills gaps the "
        "reference doesn't specify.\n"
        f"{_REFERENCE_HARD_FLOOR}\n{rules}"
    )


def suggest_mood(character: str, anime: str = "") -> str:
    """Return a short mood/vibe descriptor line for a character (stateless)."""
    who = f"{character} from {anime}" if anime else character
    user_prompt = (
        f"Character / matchup: {who}\n\n"
        "Give the mood/vibe line for an original anime song about this. One line only."
    )
    text = anthropic_client.complete_text(SUGGEST_MOOD_SYSTEM_PROMPT, user_prompt)
    if not text:
        return ""
    return text.strip().strip('"').splitlines()[0]


def suggest_genres(project: VideoProject, *, content_blend: str | None = None) -> GenreSuggestions:
    """Return 3 genre directions for the project's character/topic.

    When the run has a reference rap, the reference's tempo and inferred musical lane
    ANCHOR all three directions — genre follows the reference. The three options are
    variations within that same lane (not three unrelated genres). ``content_blend`` is
    accepted for signature compatibility but no longer gates the anchoring: the genre
    always follows the reference when one exists; content_blend only governs how much
    lyrical CONTENT is borrowed at generation time.
    """
    from cutforge.services import reference_service

    topic = project.topic or project.character
    profile = reference_service.load_reference_profile(project)

    lines = [
        f"Character / matchup: {topic}",
        "",
        "Propose 3 genre/vibe directions for an original anime song about this.",
    ]
    if profile:
        lines += [
            "",
            "REFERENCE (the user picked this rap to emulate — the genre MUST follow it):",
            f"- BPM: {profile.get('bpm')} (use this tempo; put one BPM number in each style)",
            f"- Onset density: {profile.get('onset_rate_per_sec')} onsets/s",
            f"- Reference title: {profile.get('source_title')}",
            f"- Transcript (infer the subgenre/energy/vocal style from it — do NOT copy "
            f"words): {profile.get('transcript', '')[:1200]}",
            "",
            "Anchor ALL 3 directions to the reference's lane and tempo — they should sound "
            "like they belong on the same playlist as the reference. Make the 3 options "
            "variations WITHIN that lane (e.g. harder vs. more melodic, sparser vs. more "
            "orchestral), not three unrelated genres. Every direction must still genuinely "
            "fit the character.",
        ]
    lines.append("Return only valid JSON.")
    user_prompt = "\n".join(lines)

    data = anthropic_client.complete_json(SUGGEST_SYSTEM_PROMPT, user_prompt)
    directions = [GenreDirection(**d) for d in data.get("directions", [])]
    return GenreSuggestions(character_read=data.get("character_read", ""), directions=directions)


def generate_package(project: VideoProject, genre: str, *, is_vs: bool = False,
                     reference_profile: dict | None = None,
                     content_blend: str = "rhythm") -> SongPackage:
    """Generate the full Suno package and write lyrics.txt + suno_prompt.json.

    If ``reference_profile`` is None, auto-loads a saved profile for this run (produced
    by the optional ``reference`` step). When present, the song is inspired by the
    reference rap and its GENRE follows the reference; ``content_blend`` controls only
    HOW MUCH of the reference's CONTENT (themes, imagery, hook shape, phrasing) may be
    borrowed — from "rhythm" (rhythm only, the strict default) up to "strong" (heavy
    thematic borrow, still non-verbatim).
    """
    # Auto-load a saved reference profile (lazy import avoids a circular dependency).
    if reference_profile is None:
        from cutforge.services import reference_service
        reference_profile = reference_service.load_reference_profile(project)

    topic = project.topic or project.character
    lines = [
        f"Character / matchup: {topic}",
        f"Chosen genre / style direction: {genre}",
    ]
    if project.character:
        lines.append(f"Character name(s): {project.character}")
    if project.anime:
        lines.append(f"Anime: {project.anime}")
    if is_vs:
        lines.append("This is a VS / matchup song — alternate perspectives and make the "
                     "chorus their clash.")

    system_prompt = GENERATE_SYSTEM_PROMPT
    if reference_profile:
        system_prompt = GENERATE_SYSTEM_PROMPT + _reference_addendum(content_blend)
        flow = reference_profile.get("flow", {})
        rhythm_only = content_blend == "rhythm"
        header = ("REFERENCE (genre + tempo signal; do NOT copy any words):" if rhythm_only
                  else f"REFERENCE (genre + tempo + content inspiration, blend='{content_blend}' "
                       "— never copy verbatim):")
        transcript_label = ("- Reference transcript (reference only — do not copy any line):"
                            if rhythm_only
                            else "- Reference transcript (content inspiration — rewrite, never copy verbatim):")
        lines.append("")
        lines.append(header)
        lines.append(f"- BPM: {reference_profile.get('bpm')} (put this one number in the style)")
        lines.append(f"- Time signature: {reference_profile.get('time_signature')}/4")
        lines.append(f"- Onset density: {reference_profile.get('onset_rate_per_sec')} onsets/s")
        lines.append(f"- Flow: {flow.get('words_per_sec')} words/s, "
                     f"~{flow.get('syllables_per_beat')} syllables/beat")
        lines.append(transcript_label)
        lines.append(reference_profile.get("transcript", ""))

    lines.append("")
    lines.append("Write the complete Suno package. Return only valid JSON.")
    user_prompt = "\n".join(lines)

    data = anthropic_client.complete_json(system_prompt, user_prompt)
    package = SongPackage(
        title=data.get("title", ""),
        style=data.get("style", ""),
        exclude=data.get("exclude", ""),
        lyrics=data.get("lyrics", ""),
        suno_tips=data.get("suno_tips", ""),
        topic=topic,
        character=project.character,
        anime=project.anime,
    )

    # Persist: lyrics.txt (paste into Suno) + suno_prompt.json (metadata for later steps)
    project.run_dir.mkdir(parents=True, exist_ok=True)
    project.lyrics_path.write_text(package.lyrics, encoding="utf-8")
    project.suno_prompt_path.write_text(
        json.dumps({
            "title": package.title,
            "style": package.style,
            "exclude": package.exclude,
            "suno_tips": package.suno_tips,
            "topic": package.topic,
            "character": package.character,
            "anime": package.anime,
            "language": project.language,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Cache the title back onto the project for downstream steps.
    if package.title and not project.title:
        project.title = package.title
        project.save()

    return package
