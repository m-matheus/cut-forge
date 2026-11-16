"""Thumbnail service — music-video thumbnail via the OpenAI Responses API.

Two modes (same split as lore-forge):

- **No references** (key art, ``keyart_NN.png``): a cheap Claude call (``_analyze_visual``)
  turns the free-text character/anime/mood into concrete visual attributes (expression,
  iconic prop, eye treatment, colors...), and the image is rendered from the shared
  MUSIC_BASELINE ("thumbnail DNA") + those attributes. This is what lets ONE prompt
  template serve ANY character without hard-coding a look.
- **References** in thumbnail/refs/ (``refthumb_NN.png``): they are usually screenshots of
  other channels' cards, so Claude first cleans each one (crops the screenshot frame and the
  progress bar, blurs the duration badge, logos and lettering) — otherwise the image model
  copies that UI into our art. The cleaned refs go to the image tool as a mood board; if it
  refuses them, Claude turns them into a visual-direction spec and the art is generated
  from that text alone.

MUSIC_BASELINE is the fixed visual language of high-CTR anime rap thumbnails (Sensei Beats,
Rustage, 7 Minutoz, Ishida Music): oversized front-facing face, glowing eyes, one iconic
gesture, explosive neon background, extreme contrast — readable at small thumbnail size.

Every run generates ``variants`` images; ``pick`` re-finishes another one without new
images. The chosen art is cover-cropped (never stretched) to 1280x720.
"""
from __future__ import annotations

import json
import math
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageFilter

from cutforge.integrations import anthropic_client, openai_images
from cutforge.models.project import VideoProject

WIDTH, HEIGHT = 1280, 720

MUSIC_BASELINE = """
Create a high-CTR YouTube anime rap thumbnail, designed to be instantly recognizable at a
small mobile thumbnail size (readable even at ~160px wide). This is NOT a wallpaper or a
generic key-art illustration — it is an aggressive, click-winning YouTube thumbnail. Study
how top channels (Sensei Beats, Rustage, 7 Minutoz, Ishida Music) design theirs.

ART STYLE:
Clean, ultra-detailed 2D anime illustration with sharp confident linework and cel-shading.
NOT 3D, NOT CGI, NOT photorealistic.

CHARACTER (the whole point of the image):
ONE character, extreme close-up portrait — the face and upper chest fill roughly 80% of the
frame. Front-facing or a slight dramatic three-quarter angle, eyes locked on the viewer. The
full face stays inside the frame; never crop the eyes. Avoid full-body or distant shots.
Render the character's exact canonical design — do not redesign them.

LIGHTING:
Extreme rim/edge lighting in the character's signature color, as if the body emits energy.
The character glows; deep shadows against very bright highlights; strong separation from the
background.

BACKGROUND (supporting, never competing):
Abstract high-energy field — a two-tone color split, energy explosion, lightning, particles
or aura. Keep the area right behind the face relatively clean and use a dark vignette toward
the edges so the face and eyes stay the clear focal point. No literal scenes or locations.

COLOR GRADING:
Hyper-saturated, electric, neon-level vibrancy with maximum contrast — colors almost
unrealistically vivid, the kind of image people stop scrolling for.

NO watermarks, NO channel logos, NO character names, NO text on the image.
""".strip()

VISUAL_ANALYSIS_SYSTEM_PROMPT = """\
You are an art director for an anime rap / music YouTube channel. Given a character, their
anime and the song's mood, decide the concrete visual choices for a high-CTR thumbnail so the
image is instantly recognizable AND matches the song's energy. Base every choice on the
character's real canonical design — never invent features.

Return a single valid JSON object. No markdown fences, no commentary.
{
  "expression": "one short phrase — the facial expression, chosen to fit BOTH the character's personality and the song's mood (e.g. arrogant smirk, stoic glare, sorrowful resolve, manic grin)",
  "iconic_gesture": "one simple, instantly recognizable gesture or prop tied to this character (e.g. 'adjusting round sunglasses', 'hand wreathed in cursed energy near the face'), or 'none' if the character has no signature gesture — do NOT force one",
  "eye_treatment": "eye color and any signature glow/effect (e.g. 'glowing electric-cyan eyes', 'red Sharingan')",
  "signature_features": "the must-keep canonical identity cues: hairstyle & color, outfit, accessories",
  "dominant_colors": ["2-3 colors, most dominant first — the character's signature palette rendered as electric/neon versions"],
  "energy_motif": "the background energy tied to this character's power or the song mood (e.g. 'violet cursed-energy explosion', 'blue lightning')"
}
"""

_LANG_NOTE = "Keep all values in English (they are image-prompt fragments, not shown to viewers)."


def _analyze_visual(project: VideoProject, *, on_log=None) -> dict:
    """Ask Claude for concrete per-character visual attributes. Returns {} on any failure."""
    user_prompt = (
        f"Character(s): {project.character}\n"
        f"Anime / Series: {project.anime or 'unknown'}\n"
        f"Song mood: {project.mood or 'dark, powerful, cinematic'}\n\n"
        f"{_LANG_NOTE} Return only valid JSON."
    )
    try:
        data = anthropic_client.complete_json(VISUAL_ANALYSIS_SYSTEM_PROMPT, user_prompt)
    except Exception as exc:  # analysis is an enhancement — never block the thumbnail on it
        if on_log:
            on_log(f"Visual analysis skipped ({exc}); using baseline prompt.")
        return {}
    if on_log and data.get("expression"):
        on_log(f"Visual direction: {data.get('expression')} · {', '.join(data.get('dominant_colors', []))}")
    return data


def _analysis_block(data: dict) -> str:
    """Render the analysis dict into an image-prompt section (empty string if no data)."""
    lines = []
    if data.get("expression"):
        lines.append(f"Expression: {data['expression']}.")
    gesture = data.get("iconic_gesture", "").strip()
    if gesture and gesture.lower() != "none":
        lines.append(f"Iconic gesture / prop: {gesture} — keep it simple and large.")
    if data.get("eye_treatment"):
        lines.append(f"Eyes: {data['eye_treatment']}.")
    if data.get("signature_features"):
        lines.append(f"Keep canonical: {data['signature_features']}.")
    if data.get("dominant_colors"):
        lines.append(f"Dominant palette (in order): {', '.join(data['dominant_colors'])}.")
    if data.get("energy_motif"):
        lines.append(f"Background energy: {data['energy_motif']}.")
    if not lines:
        return ""
    return "CHARACTER VISUAL DIRECTION (follow exactly):\n" + "\n".join(lines)


def _build_request(project: VideoProject, analysis: dict | None = None) -> str:
    """Key-art prompt (no reference images): the full baseline drives the look."""
    sections = [
        ("Featured character(s)", project.character),
        ("Anime / Series", project.anime),
        ("Song vibe / mood", project.mood or "dark, powerful, cinematic"),
    ]
    context = "\n\n".join(f"{label}:\n{value}" for label, value in sections if value)

    return "\n\n".join(filter(None, [
        "Create a thumbnail for an anime music video (character key art, album-cover energy).",
        context,
        _analysis_block(analysis or {}),
        MUSIC_BASELINE,
    ])).strip()


# ---- Reference mode ----

# Refs are usually screenshots of a YouTube card: the art sits inside a frame (page
# background, border, rounded corners) with player UI on top. Sent as-is, the image model
# reproduces all of it (it drew a "4:31" badge and a red progress bar on the Subaru art).
CLEANUP_SYSTEM = """\
You inspect screenshots of YouTube thumbnails before they are used as style references.
Find, as fractions of the image width and height (0 = left/top, 1 = right/bottom):

- "content_box": the thumbnail artwork itself. Exclude everything that is not artwork: page
  background, borders, rounded-corner padding, and any watched-progress bar running along an
  edge (cut it off). When the image is already just the artwork, return 0, 0, 1, 1.
- "overlays": every element drawn ON TOP of the artwork that is not part of the
  illustration: duration / timestamp badges, music or play icons, progress bars, channel
  logos, watermarks, titles, captions and any other lettering. Empty list when there is none.

Return a single valid JSON object. No markdown fences, no commentary.
{"content_box": {"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0},
 "overlays": [{"x0": 0.0, "y0": 0.0, "x1": 0.0, "y1": 0.0, "what": "short label"}]}
"""

REFERENCE_DIRECTION_SYSTEM = """\
You are the art director of an anime rap / music YouTube channel. The attached thumbnails are
a mood board: they show the visual language the channel wants, not images to copy. Describe
that language as a general design spec: composition and framing, camera distance, how much
of the frame the face fills, expression intensity, lighting, colour relationships,
background energy, art style and mood. Describe it in general terms; never mention the
references' logos, lettering or any video-player UI.

Then write the subject of a NEW image of the given character: their canonical design, an
expression that fits the song's mood, and a pose different from the references. One or two
concrete sentences, enough to paint from.

Return a single valid JSON object. No markdown fences, no commentary. Keys (all strings):
"composition", "camera", "lighting", "palette", "background", "expression", "style", "mood",
"subject".
"""
_DIRECTION_KEYS = ("composition", "camera", "lighting", "palette", "background", "expression",
                   "style", "mood")

_NO_UI = (
    "The image IS the artwork itself, full-bleed edge to edge: no screenshot frame, no "
    "borders, no rounded corners, no video-player UI, no progress bar, no duration or "
    "timestamp badge, no play or music icons.\n\n"
    "NO text, NO letters, NO numbers, NO logos, NO watermarks, NO channel names."
)

# First attempt: the cleaned refs go to the image tool as a mood board.
REF_REQUEST = (
    "Create a completely new high-CTR YouTube thumbnail artwork for an anime music video "
    "about {who}. Song mood: {mood}.\n\n"
    "The attached images are visual mood-board references only. Use them to understand the "
    "visual language: framing and camera distance, how much of the frame the face fills, "
    "expression intensity, lighting, colour relationships, background energy and overall "
    "readability at thumbnail size.\n\n"
    "Create a new image, not a copy: a different pose, hand position and background "
    "arrangement. Do not recreate, trace or closely imitate any reference image.\n\n"
    "{who} appears in their exact canonical design (correct hair, eyes, outfit and "
    "accessories). Clean 2D anime illustration, sharp linework, cel-shading. Keep the hands "
    "and anatomy clean and correct.\n\n" + _NO_UI
)

# Fallback when the image tool refuses the refs: they are described, not sent.
MOOD_REQUEST = (
    "Create a completely new high-CTR YouTube thumbnail artwork for an anime music video "
    "about {who}.\n\n"
    "VISUAL DIRECTION:\n{direction}\n\n"
    "SUBJECT: {subject}\n\n"
    "{who} appears in their exact canonical design (correct hair, eyes, outfit and "
    "accessories). Clean 2D anime illustration, readable at YouTube thumbnail size. Do not "
    "reproduce any existing promotional artwork, screenshot or thumbnail.\n\n" + _NO_UI
)


def _who(project: VideoProject) -> str:
    return f"{project.character} (from {project.anime})" if project.anime else project.character


def _image_block(path: Path) -> dict:
    """An image as a PNG content block for Claude, capped at 1568 px (its useful maximum)."""
    img = Image.open(path).convert("RGB")
    img.thumbnail((1568, 1568))
    buf = BytesIO()
    img.save(buf, "PNG")
    return anthropic_client.image_block(buf.getvalue())


def _box(b: dict, w: int, h: int) -> tuple[int, int, int, int]:
    """A fractional box -> clamped pixel box."""
    x0, x1 = sorted((float(b.get("x0", 0)), float(b.get("x1", 1))))
    y0, y1 = sorted((float(b.get("y0", 0)), float(b.get("y1", 1))))
    return (max(0, int(x0 * w)), max(0, int(y0 * h)),
            min(w, math.ceil(x1 * w)), min(h, math.ceil(y1 * h)))


def _clean_ref(project: VideoProject, ref: Path, log) -> Path:
    """Crop one reference to its artwork and blur the overlays on it (cached)."""
    out = project.thumbnail_dir / "refs_clean" / f"{ref.stem}.png"
    if out.exists():
        return out
    try:
        data = anthropic_client.complete_json(
            CLEANUP_SYSTEM,
            [_image_block(ref), anthropic_client.plain("Inspect this screenshot.")],
            max_tokens=3000, effort="low")
    except Exception as exc:  # cleanup is an enhancement — never block the thumbnail on it
        log(f"Could not inspect {ref.name} ({exc}); sending it as-is.")
        return ref

    img = Image.open(ref).convert("RGB")
    full_w, full_h = img.size
    box = _box(data.get("content_box") or {}, full_w, full_h)
    if box[2] - box[0] < full_w * 0.3 or box[3] - box[1] < full_h * 0.3:
        box = (0, 0, full_w, full_h)  # implausible box: keep the whole image
    img = img.crop(box)

    # Overlays come in full-image fractions: shift them into the cropped frame, then cover
    # each one (plus a small margin) with a heavy blur of the art around it.
    blur = img.filter(ImageFilter.GaussianBlur(max(12, img.width // 40)))
    pad = max(4, img.width // 100)
    for o in data.get("overlays") or []:
        x0, y0, x1, y1 = _box(o, full_w, full_h)
        region = (max(0, x0 - box[0] - pad), max(0, y0 - box[1] - pad),
                  min(img.width, x1 - box[0] + pad), min(img.height, y1 - box[1] + pad))
        if region[2] > region[0] and region[3] > region[1]:
            img.paste(blur.crop(region), region)

    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    labels = ", ".join(o.get("what", "?") for o in data.get("overlays") or []) or "none"
    log(f"Cleaned {ref.name}: artwork {img.width}x{img.height}, blurred overlays: {labels}")
    return out


def _reference_direction(project: VideoProject, refs: list[Path], refresh: bool, log) -> dict:
    """Claude's visual-direction spec of the references + a new subject (cached; ``refresh``
    asks for a new one, so Redo also gets a new image)."""
    path = project.thumbnail_dir / "ref_direction.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))
    user = [*(_image_block(r) for r in refs), anthropic_client.plain(
        f"Character: {_who(project)}\n"
        f"Song mood: {project.mood or 'dark, powerful, cinematic'}\n"
        "Return the design spec of these references and the subject of the new image.")]
    data = anthropic_client.complete_json(REFERENCE_DIRECTION_SYSTEM, user,
                                          max_tokens=3000, effort="low")
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    log(f"Reference direction: {data.get('subject', '')}")
    return data


def _mood_prompt(project: VideoProject, d: dict) -> str:
    lines = [f"- {k.capitalize()}: {d[k]}" for k in _DIRECTION_KEYS if d.get(k)]
    return MOOD_REQUEST.format(who=_who(project), direction="\n".join(lines),
                               subject=d.get("subject", ""))


# ---- Shared ----

def _generate(project: VideoProject, prefix: str, prompt: str, refs: list[Path],
              variants: int, log) -> Path:
    """Append ``variants`` new images as {prefix}_NN.png; return the first new one."""
    existing = sorted(project.thumbnail_dir.glob(f"{prefix}_*.png"))
    first = None
    for i in range(len(existing), len(existing) + variants):
        out = project.thumbnail_dir / f"{prefix}_{i + 1:02d}.png"
        openai_images.generate_image(prompt, out, portrait=False,
                                     reference_images=refs or None, on_log=log)
        first = first or out
    return first


def _finish(src: Path, out: Path) -> None:
    """Cover-crop to 1280x720. The model returns 1536x1024 (3:2): resizing that straight
    to 16:9 squashed the art vertically."""
    img = Image.open(src).convert("RGB")
    scale = max(WIDTH / img.width, HEIGHT / img.height)
    img = img.resize((math.ceil(img.width * scale), math.ceil(img.height * scale)), Image.LANCZOS)
    left, top = (img.width - WIDTH) // 2, (img.height - HEIGHT) // 2
    out.parent.mkdir(parents=True, exist_ok=True)
    img.crop((left, top, left + WIDTH, top + HEIGHT)).save(out, "JPEG", quality=93)


def variant_paths(project: VideoProject) -> list[Path]:
    """Every generated variant, reference ones first."""
    d = project.thumbnail_dir
    if not d.exists():
        return []
    return [p for pat in ("refthumb_*.png", "keyart_*.png") for p in sorted(d.glob(pat))]


def selection(project: VideoProject) -> dict:
    """Which variant thumbnail.jpg was made from ({} before the first run)."""
    path = project.thumbnail_dir / "thumbnail.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def run(project: VideoProject, params: dict, on_log) -> dict:
    """Generate the 16:9 thumbnail as thumbnail/thumbnail.jpg.

    ``params``: ``variants`` (default 2) new images per run; ``pick`` (a variant file name)
    re-finishes that variant without new images; ``reference="mood"`` skips sending the refs
    and goes straight to the described direction; ``refresh`` (set by the runner on Redo)
    generates new variants instead of reusing the existing ones.
    """
    log = on_log
    if not project.character:
        raise ValueError("Thumbnail needs a character — set it on the project first.")
    variants = max(1, int(params.get("variants", 2)))
    refresh = bool(params.get("refresh"))
    project.thumbnail_dir.mkdir(parents=True, exist_ok=True)

    chosen = None
    if params.get("pick"):
        chosen = project.thumbnail_dir / Path(str(params["pick"])).name
        if not chosen.is_file():
            raise FileNotFoundError(f"No such variant: {chosen.name}")

    refs = project.thumbnail_ref_paths()
    ref_mode = chosen.name.startswith("refthumb_") if chosen else bool(refs)
    prefix = "refthumb" if ref_mode else "keyart"

    if chosen is None:
        existing = sorted(project.thumbnail_dir.glob(f"{prefix}_*.png"))
        if existing and not refresh:
            chosen = existing[0]
        elif ref_mode:
            log(f"Following {len(refs)} reference(s): {[r.name for r in refs]}")
            clean = [_clean_ref(project, r, log) for r in refs]
            if params.get("reference") != "mood":
                prompt = REF_REQUEST.format(who=_who(project),
                                            mood=project.mood or "dark, powerful, cinematic")
                try:
                    chosen = _generate(project, prefix, prompt, clean, variants, log)
                except RuntimeError as exc:
                    if "No image returned" not in str(exc):
                        raise
                    # Retrying with the same references gets the same answer: describe
                    # them instead and leave them out of the request.
                    log(f"Image tool refused the references ({exc}). Switching to a "
                        "description of them; the references are not sent.")
            if chosen is None:
                direction = _reference_direction(project, clean, refresh, log)
                chosen = _generate(project, prefix, _mood_prompt(project, direction),
                                   [], variants, log)
        else:
            analysis = _analyze_visual(project, on_log=log)
            chosen = _generate(project, prefix, _build_request(project, analysis),
                               [], variants, log)

    _finish(chosen, project.thumbnail_path)
    mode = "reference" if ref_mode else "keyart"
    (project.thumbnail_dir / "thumbnail.json").write_text(
        json.dumps({"art": chosen.name, "mode": mode}, indent=2), encoding="utf-8")
    size_kb = project.thumbnail_path.stat().st_size // 1024
    log(f"Thumbnail: {project.thumbnail_path.name} from {chosen.name} ({size_kb} KB, {mode} mode).")
    return {"path": str(project.thumbnail_path), "art": chosen.name, "mode": mode}
