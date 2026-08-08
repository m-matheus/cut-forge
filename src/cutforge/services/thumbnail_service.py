"""Thumbnail service — music-video thumbnail via the OpenAI Responses API.

Two stages:
  1. A cheap Claude text call (``_analyze_visual``) turns the free-text character/anime/mood
     into concrete visual attributes (expression, iconic prop, eye treatment, colors...).
     This is what lets ONE prompt template serve ANY character without hard-coding a look.
  2. ``openai_images.generate_image`` renders the 16:9 thumbnail from a prompt built out of
     the shared MUSIC_BASELINE ("thumbnail DNA") + those per-character attributes.

MUSIC_BASELINE is the fixed visual language of high-CTR anime rap thumbnails (Sensei Beats,
Rustage, 7 Minutoz, Ishida Music): oversized front-facing face, glowing eyes, one iconic
gesture, explosive neon background, extreme contrast — readable at small thumbnail size.
Fully AI-generated in one shot (no PIL compositing) — resized/cropped to 1280x720.
"""
from __future__ import annotations

from cutforge.integrations import anthropic_client, openai_images
from cutforge.models.project import VideoProject

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


_STYLE_REF_NOTE = (
    "STYLE REFERENCE IMAGES are attached. Use them ONLY as a guide for composition, "
    "framing/crop, lighting, color energy and how accessories and hands are posed — match "
    "that visual language. Do NOT copy their exact character art; render the character "
    "described above in their own canonical design."
)


def _build_request(project: VideoProject, analysis: dict | None = None,
                   *, has_refs: bool = False) -> str:
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
        _STYLE_REF_NOTE if has_refs else "",
    ])).strip()


def generate_thumbnail(project: VideoProject, *, on_log=None) -> str:
    """Generate the 16:9 music thumbnail and save it as thumbnail/thumbnail.jpg."""
    if not project.character:
        raise ValueError("Thumbnail needs a character — set it on the project first.")

    refs = project.thumbnail_ref_paths()
    analysis = _analyze_visual(project, on_log=on_log)
    request = _build_request(project, analysis, has_refs=bool(refs))
    raw_path = project.thumbnail_dir / "raw.png"
    openai_images.generate_image(request, raw_path, portrait=False,
                                 reference_images=refs, on_log=on_log)

    from PIL import Image
    Image.open(raw_path).convert("RGB").resize((1280, 720), Image.LANCZOS).save(
        str(project.thumbnail_path), "JPEG", quality=93
    )
    if on_log:
        size_kb = project.thumbnail_path.stat().st_size // 1024
        on_log(f"Thumbnail saved: {project.thumbnail_path.name} ({size_kb} KB)")
    return str(project.thumbnail_path)
