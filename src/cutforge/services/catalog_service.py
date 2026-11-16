"""Channel catalog memory — keeps new songs from echoing the channel's previous ones.

Every song is written by a fresh LLM call that has never seen the channel's earlier
songs, so it falls into the same frames over and over ("Enkai again... This one's for
X. X, the boy who..."). This module reads the channel's previous runs and gives the
writer two things:

- a prompt block with the recent intros, hooks and Suno styles, plus the phrases that
  already recur across the catalog, so the planner and writer know what NOT to do;
- a deterministic check (the cross-song version of LoreForge's lint_service n-gram
  check) that flags a new intro reusing a previous intro's wording, so song_service can
  rewrite just that section.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field

from cutforge.models.project import PROJECT_FILE, VideoProject

CATALOG_SIZE = 8          # previous songs shown to the writer
RECURRING_MIN_SONGS = 3   # a phrase in this many catalog songs is a channel tic

# Tags that open a new section. Delivery tags ([Spoken], [Rap], [Male Vocal]...) don't.
_SECTION_TAG = re.compile(
    r"^\[(?:final\s+)?(intro|verse|chorus|hook|pre-chorus|post-chorus|bridge|outro|drop|"
    r"build|instrumental|breakdown|break|interlude|refrain|end)\b[^\]]*\]\s*$",
    re.IGNORECASE,
)
_TAG = re.compile(r"\[[^\]]*\]")
_ADLIB = re.compile(r"\([^)]*\)")
_WORD = re.compile(r"[^\W\d_](?:[^\W\d_]|['’])*")

_STOPWORDS = frozenset("""
a an and the of to in on at for from by with is are was were be been it its i me my you
your we our he his she her they them their this that these those as or but so if not no
who what when where how all just like i'm it's don't can't won't ain't i'll i've you're
de da do das dos e o a os as um uma em no na para por que se com el la los las y en un una
""".split())


@dataclass
class CatalogSong:
    run_id: str
    intro: str
    hook: str
    lyrics: str = field(repr=False)
    style: str = ""   # the Suno style string the song was generated with


def is_section_tag(line: str) -> bool:
    """True when ``line`` opens a new section, with or without an inline description."""
    return bool(_SECTION_TAG.match(line.strip()))


def sections(lyrics: str) -> list[tuple[str, str]]:
    """``[(section_name, body)]`` in order; body keeps delivery tags and ad-libs."""
    out: list[tuple[str, list[str]]] = []
    for line in lyrics.splitlines():
        m = _SECTION_TAG.match(line.strip())
        if m:
            out.append((m.group(1).lower(), []))
        elif out:
            out[-1][1].append(line)
    return [(name, "\n".join(body).strip()) for name, body in out]


def section(lyrics: str, *names: str) -> str:
    """Body of the first section whose name is in ``names`` ("" if absent)."""
    return next((body for name, body in sections(lyrics) if name in names), "")


def sung_text(text: str) -> str:
    """The words a listener hears: delivery tags and (ad-libs) removed, one line each."""
    lines = (_ADLIB.sub("", _TAG.sub("", ln)).strip() for ln in text.splitlines())
    return "\n".join(ln for ln in lines if ln)


def _tokens(text: str) -> list[str]:
    return [w.lower().replace("’", "'") for w in _WORD.findall(sung_text(text))]


def _ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def load_catalog(project: VideoProject, limit: int = CATALOG_SIZE) -> list[CatalogSong]:
    """The channel's most recent songs (newest first), excluding this run."""
    base = project.run_dir.parent
    if not base.exists():
        return []
    songs: list[CatalogSong] = []
    for run in sorted((p for p in base.iterdir() if p.is_dir()), reverse=True):
        if run.name == project.run_id or not (run / "lyrics.txt").exists():
            continue
        meta = run / PROJECT_FILE
        if meta.exists():
            slug = json.loads(meta.read_text(encoding="utf-8")).get("channel_slug", "enkai")
            if slug != project.channel_slug:
                continue
        lyrics = (run / "lyrics.txt").read_text(encoding="utf-8")
        suno = run / "suno_prompt.json"
        style = json.loads(suno.read_text(encoding="utf-8")).get("style", "")             if suno.exists() else ""
        songs.append(CatalogSong(
            run_id=run.name,
            intro=sung_text(section(lyrics, "intro")),
            hook=sung_text(section(lyrics, "chorus", "hook", "refrain")),
            lyrics=lyrics,
            style=style,
        ))
        if len(songs) >= limit:
            break
    return songs


def recurring_phrases(catalog: list[CatalogSong],
                      min_songs: int = RECURRING_MIN_SONGS) -> list[str]:
    """3-word phrases (with ≥2 content words) used in ``min_songs``+ catalog songs."""
    where: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for song in catalog:
        for gram in _ngrams(_tokens(song.lyrics), 3):
            if sum(t not in _STOPWORDS for t in gram) >= 2:
                where[gram].add(song.run_id)
    hits = [(len(runs), " ".join(g)) for g, runs in where.items() if len(runs) >= min_songs]
    return [phrase for _, phrase in sorted(hits, reverse=True)]


def echoed_intro_phrases(intro: str, catalog: list[CatalogSong]) -> list[str]:
    """Phrases of a new intro that copy previous intros' wording.

    Intros are short and the problem is the FRAME ("this one's for", "the boy who"), so
    stopwords are not filtered here: any 3-word sequence shared with 2+ previous intros,
    or any 4-word sequence shared with one, is an echo.
    """
    new = _tokens(intro)
    prev = [_tokens(s.intro) for s in catalog if s.intro]
    echoes: list[str] = []
    for gram in sorted(_ngrams(new, 3)):
        if sum(gram in _ngrams(p, 3) for p in prev) >= 2:
            echoes.append(" ".join(gram))
    for gram in sorted(_ngrams(new, 4)):
        phrase = " ".join(gram)
        if any(gram in _ngrams(p, 4) for p in prev) and not any(e in phrase for e in echoes):
            echoes.append(phrase)
    return echoes


def format_catalog_for_prompt(catalog: list[CatalogSong]) -> str:
    """Writer-facing block: what the channel already did, so this song does something else."""
    if not catalog:
        return ""
    parts = [
        "CHANNEL CATALOG — the channel's most recent songs. Listeners hear these back to "
        "back, so this song must NOT sound like a template of them. Your [Intro] must use "
        "a different opening frame, sentence shape and epithet style than every intro "
        "below; your hook must not reuse their hook wording or shape.",
        "",
        "Previous intros (newest first):",
    ]
    for song in catalog:
        if song.intro:
            parts.append(f"  • {song.run_id}: " + " / ".join(song.intro.splitlines()))
    hooks = [s for s in catalog if s.hook]
    if hooks:
        parts += ["", "Previous hooks (first lines):"]
        for song in hooks:
            parts.append(f"  • {song.run_id}: " + " / ".join(song.hook.splitlines()[:2]))
    styles = [s for s in catalog if s.style]
    if styles:
        parts += ["", "Previous styles (newest first) — the sound must not repeat the latest "
                  "ones either:"]
        for song in styles:
            parts.append(f"  • {song.run_id}: {song.style}")
    tics = recurring_phrases(catalog)
    if tics:
        parts += ["", "Phrases that already recur across the catalog — do NOT use them: "
                  + "; ".join(f'"{t}"' for t in tics[:20])]
    return "\n".join(parts)


def replace_section(lyrics: str, name: str, new_body: str) -> str:
    """Swap the body of the first ``[name...]`` section, keeping its tag line."""
    lines = lyrics.splitlines()
    start = next((i for i, ln in enumerate(lines)
                  if (m := _SECTION_TAG.match(ln.strip())) and m.group(1).lower() == name),
                 None)
    if start is None:
        return lyrics
    end = next((j for j in range(start + 1, len(lines))
                if _SECTION_TAG.match(lines[j].strip())), len(lines))
    tail = [""] if end < len(lines) and lines[end - 1].strip() == "" else []
    return "\n".join(lines[:start + 1] + new_body.strip().splitlines() + tail + lines[end:])
