"""Release service — everything a distributor's form asks for, to put the song on Spotify.

The channel releases as an ARTIST (DistroKid / ONErpm / Tunecore...), not as a podcast:
the song lands in the music catalog, counts streams and earns royalties. This step
prepares the release; the upload itself is done by hand in the distributor's site.

- **Cover** (``release/cover.jpg``, 3000x3000): stores reject artwork that shows
  someone else's copyrighted character, so unlike the YouTube thumbnail the cover never
  depicts the character. Claude picks one symbolic object/scene from their world and the
  image is generated from that.
- **Audio**: Suno's WAV download (``audio/track.wav``) when present; the MP3 is flagged,
  never converted (a WAV made from an MP3 is still lossy, and some distributors test it).
- **Lyrics** (``release/lyrics.txt``): Suno's tags and stage directions removed.
- **Form** (``release/release.json``): every field in form order, plus warnings. The
  UI shows them with Copy buttons, like the YouTube ones.

Once the release is live, the run's ``spotify_track_url`` makes the video's QR code
open the song instead of the artist profile (premiere step).
"""
from __future__ import annotations

import json
import re
from datetime import date, timedelta
from pathlib import Path

from PIL import Image

from cutforge.integrations import anthropic_client, openai_images
from cutforge.models.project import VideoProject
from cutforge.services import catalog_service

COVER_SIDE = 3000          # the size every major distributor asks for
LEAD_DAYS = 14             # Spotify for Artists pitching needs the release 7+ days out

_LANG_NAMES = {"en": "English", "es": "Spanish", "pt": "Portuguese"}

# Strong language that makes stores require the Explicit flag (en / es / pt).
_EXPLICIT = re.compile(
    r"\b(fuck\w*|motherfuck\w*|shit\w*|bitch\w*|niggas?|cunt|dick|pussy|"
    r"puta|puto|mierda|joder|cabr[oó]n\w*|pendej\w*|verga|"
    r"porra|caralho|merda|foda\w*|fod[ae]r|buceta|cacete|arrombad\w*)\b",
    re.IGNORECASE,
)
# A line that is only a stage direction / sound effect, e.g. "(engine sputters)".
_DIRECTION_LINE = re.compile(r"^\(([^)]*)\)$")

COVER_CONCEPT_SYSTEM_PROMPT = """\
You are the art director of an anime-rap ARTIST releasing a single on Spotify. Design the
square cover art concept for a song about an anime character.

HARD RULE — streaming stores reject artwork showing copyrighted characters. The cover
must NOT depict the character, their face, silhouette, outfit, or any official logo,
clan crest, emblem or symbol from the series. Instead pick ONE symbolic object or scene
from the character's world that a fan recognises by meaning, rendered generically (a
cracked gourd spilling sand under a red moon; a broken katana on a rain-soaked shrine
step; three blades crossed in fog) — something that also stands on its own as album art.

Return a single valid JSON object, no markdown fences:
{
  "concept": "the one image, 1-2 concrete sentences",
  "palette": "2-3 dominant colours and the lighting",
  "mood": "3-5 words"
}
"""

COVER_REQUEST = """
Create square album cover art for a hip-hop single. Painterly, cinematic digital art with
a dark, high-contrast look and one clear focal point, readable at thumbnail size (a phone's
music player shows it at ~60 px).

SUBJECT: {concept}
PALETTE / LIGHT: {palette}
MOOD: {mood}

NO people, NO faces, NO characters, NO logos, NO emblems or symbols from any franchise,
NO text, NO lettering, NO watermarks, NO borders.
""".strip()


# ---- Cover ----

def cover_variants(project: VideoProject) -> list[Path]:
    d = project.release_dir
    return sorted(d.glob("cover_*.png")) if d.exists() else []


def _cover_concept(project: VideoProject, song: str, log) -> dict:
    who = f"{project.character} from {project.anime}" if project.anime else project.character
    data = anthropic_client.complete_json(
        COVER_CONCEPT_SYSTEM_PROMPT,
        f"Character: {who}\nSong title: {song}\nMood: {project.mood or 'intense'}\n\n"
        "Return only valid JSON.")
    log(f"Cover concept: {data.get('concept', '')}")
    return data


def _finish_cover(src: Path, out: Path) -> None:
    """Centre-crop to a square and upscale to 3000x3000 RGB JPEG."""
    img = Image.open(src).convert("RGB")
    side = min(img.size)
    left, top = (img.width - side) // 2, (img.height - side) // 2
    img = img.crop((left, top, left + side, top + side))
    img = img.resize((COVER_SIDE, COVER_SIDE), Image.LANCZOS)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, "JPEG", quality=95)


def _cover(project: VideoProject, song: str, params: dict, log) -> Path:
    """Pick (``pick``), reuse, or generate the cover art; write cover.jpg from it."""
    project.release_dir.mkdir(parents=True, exist_ok=True)
    if params.get("pick"):
        chosen = project.release_dir / Path(str(params["pick"])).name
        if not chosen.is_file():
            raise FileNotFoundError(f"No such cover variant: {chosen.name}")
    else:
        existing = cover_variants(project)
        if existing and not params.get("refresh"):
            chosen = existing[-1]
        else:
            c = _cover_concept(project, song, log)
            chosen = project.release_dir / f"cover_{len(existing) + 1:02d}.png"
            openai_images.generate_image(
                COVER_REQUEST.format(concept=c.get("concept", ""),
                                     palette=c.get("palette", ""), mood=c.get("mood", "")),
                chosen, size=openai_images.SIZE_SQUARE, on_log=log)
    _finish_cover(chosen, project.cover_path)
    log(f"Cover: {project.cover_path.name} from {chosen.name} ({COVER_SIDE}x{COVER_SIDE}).")
    return chosen


# ---- Lyrics ----

def clean_lyrics(lyrics: str) -> str:
    """Lyrics as listeners read them: no [tags], no stage-direction lines, one blank line
    between sections. Inline ad-libs like "(yeah)" are sung, so they stay."""
    blocks = []
    for _name, body in catalog_service.sections(lyrics) or [("", lyrics)]:
        lines = []
        for ln in body.splitlines():
            ln = re.sub(r"\[[^\]]*\]", "", ln).strip()
            if ln and not _DIRECTION_LINE.match(ln):
                lines.append(ln)
        if lines:
            blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def explicit_words(lyrics: str) -> list[str]:
    return sorted({m.group(0).lower() for m in _EXPLICIT.finditer(lyrics)})


# ---- Audio ----

def _audio_field(project: VideoProject) -> tuple[str, str, list[str]]:
    """(value, hint, warnings) for the audio file."""
    if project.track_wav_path.exists():
        try:
            import soundfile
            info = soundfile.info(str(project.track_wav_path))
            spec = f"{info.samplerate} Hz, {info.subtype_info}, {info.channels} ch"
        except Exception:  # noqa: BLE001 — the spec line is informative only
            spec = "WAV"
        return "audio/track.wav", spec, []
    if project.track_path.exists():
        return ("audio/track.mp3", "MP3 — prefira o WAV",
                ["Sem audio/track.wav: baixe o WAV no Suno (⋯ → Download → WAV Audio) e "
                 "envie no card de lançamento. Distribuidoras pedem áudio sem perdas; não "
                 "converta o MP3 em WAV (continua com perdas)."])
    return "—", "", ["Nenhum áudio no run ainda (audio/track.wav ou track.mp3)."]


# ---- Form ----

def release_title(project: VideoProject, song: str) -> str:
    """The channel's title template, unless the song title already names the character
    (Suno titles like "One Eye Left (Obito)" would otherwise say it twice)."""
    first_name = (project.character or "").split()[0].lower() if project.character else ""
    if first_name and first_name in song.lower():
        return song
    template = project.channel.spotify.title_template
    return template.format(song=song, character=project.character or "",
                           anime=project.anime or "").replace(" ()", "").strip()


def build_form(project: VideoProject, *, today: date | None = None) -> dict:
    """Every field of a distributor's single-release form, in order, plus warnings."""
    ch = project.channel
    spec = ch.spotify
    artist = spec.artist_name or ch.name
    song = project.song_title
    lyrics = clean_lyrics(project.lyrics_path.read_text(encoding="utf-8")) \
        if project.lyrics_path.exists() else ""
    style = ""
    if project.suno_prompt_path.exists():
        style = json.loads(project.suno_prompt_path.read_text(encoding="utf-8")).get("style", "")
    explicit = explicit_words(lyrics)
    audio, audio_hint, warnings = _audio_field(project)
    release_day = (today or date.today()) + timedelta(days=LEAD_DAYS)

    if not spec.songwriter:
        warnings.append("Compositor vazio: defina spotify.songwriter (seu nome legal) no "
                        f"channels/{ch.slug}/channel.json. O Spotify não aceita só o nome "
                        "artístico como compositor.")
    if not spec.artist_link:
        warnings.append("Sem spotify.artist_url no channel.json: depois do 1º lançamento, "
                        "cole o link do perfil (open.spotify.com/artist/...) para o QR do "
                        "vídeo e a descrição do YouTube apontarem para ele.")

    fields = [
        {"label": "Artista principal", "value": artist},
        {"label": "Título", "value": release_title(project, song),
         "hint": "Nome do personagem no título ajuda na busca."},
        {"label": "Gênero principal", "value": spec.genre,
         "hint": f"Secundário: escolha pelo estilo — {', '.join(style.split(', ')[:3])}"
         if style else ""},
        {"label": "Idioma da letra", "value": _LANG_NAMES.get(project.language, "English")},
        {"label": "Conteúdo explícito", "value": "Sim" if explicit else "Não",
         "hint": f"Encontrado: {', '.join(explicit)}" if explicit
         else "Nenhum palavrão encontrado na letra; confira ouvindo."},
        {"label": "Compositor (nome legal)", "value": spec.songwriter or "—"},
        {"label": "Produtor", "value": artist},
        {"label": "Data de lançamento", "value": release_day.isoformat(),
         "hint": f"{LEAD_DAYS} dias à frente: dá tempo de mandar a faixa para as playlists "
                 "editoriais no Spotify for Artists (precisa de 7+ dias)."},
        {"label": "Uso de IA",
         "value": "Sim — instrumental e vocais gerados com Suno (plano pago); letra escrita "
                  "com assistência de IA.",
         "hint": "Declare quando a distribuidora perguntar."},
        {"label": "Capa", "value": "release/cover.jpg", "hint": f"{COVER_SIDE}x{COVER_SIDE} JPG"},
        {"label": "Áudio", "value": audio, "hint": audio_hint},
        {"label": "Letra", "value": lyrics, "multiline": True},
        {"label": "YouTube Content ID", "value": "Não",
         "hint": "Deixe DESLIGADO (ou libere o seu canal na distribuidora): com ele ligado, "
                 "os seus próprios vídeos no YouTube levam claim da sua música."},
    ]
    return {"song": song, "fields": fields, "warnings": warnings}


def run(project: VideoProject, params: dict, on_log) -> dict:
    """Prepare the release: cover, clean lyrics and the distributor form."""
    log = on_log
    if not project.lyrics_path.exists():
        raise FileNotFoundError("Release needs the lyrics — run the lyrics step first.")
    song = project.song_title
    art = _cover(project, song, params, log)

    form = build_form(project)
    form["cover_art"] = art.name
    project.release_lyrics_path.write_text(
        next(f["value"] for f in form["fields"] if f["label"] == "Letra"), encoding="utf-8")
    project.release_path.write_text(json.dumps(form, indent=2, ensure_ascii=False),
                                    encoding="utf-8")
    for w in form["warnings"]:
        log(f"ATENÇÃO: {w}")
    log(f"Release form: {project.release_path.relative_to(project.run_dir).as_posix()}")
    return {"title": release_title(project, song), "warnings": len(form["warnings"])}


def refresh_form(project: VideoProject) -> dict:
    """Rebuild the form (e.g. after a WAV upload) without touching the cover."""
    form = build_form(project)
    if project.release_path.exists():
        old = json.loads(project.release_path.read_text(encoding="utf-8"))
        form["cover_art"] = old.get("cover_art", "")
    project.release_dir.mkdir(parents=True, exist_ok=True)
    project.release_path.write_text(json.dumps(form, indent=2, ensure_ascii=False),
                                    encoding="utf-8")
    return form
