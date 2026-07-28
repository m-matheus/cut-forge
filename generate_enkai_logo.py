"""One-shot script: generate a professional logo for the Enkai channel using GPT-4o image gen."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from cutforge.integrations.openai_images import generate_image

PROMPT = """
Design a professional, high-quality logo for a Brazilian YouTube music channel called "ENKAI".
The channel covers anime rap and hip-hop beats — similar to channels like Basara Music, Sensei Beats,
7 Minutoz, and AniRap. The logo will be used as a YouTube channel icon (square format).

STYLE DIRECTION:
Create a bold, iconic emblem logo. Dark background (near black). The design should feel
powerful, premium, and instantly recognizable at small sizes (like a YouTube avatar).

DESIGN CONCEPT — choose the strongest of these two directions:

Option A — Lettermark:
Bold stylized letters "EK" treated as a single interlocked monogram.
The letters should be custom-designed (not just a font), with sharp angular cuts and
geometric construction — inspired by fighting game logos and Japanese streetwear.
Apply a crimson-to-gold gradient on the letterforms with subtle inner glow.
A thin geometric frame or badge shape (hexagon, diamond, or circle) borders the mark.

Option B — Symbol:
A flame or forge-shaped abstract mark incorporating the letter "E" or "K" subtly within it.
Think of it as a weapon being forged — fire, heat, and precision.
The mark is monochromatic with one accent color (deep red or electric gold).

TECHNICAL REQUIREMENTS:
- Square 1:1 format, works at 800x800px minimum
- Dark background (#0A0A0A or deep navy)
- NO text other than the "EK" mark or a small "ENKAI" wordmark beneath it
- Clean edges, high contrast — must be legible even at 88x88px (YouTube avatar size)
- Professional logo quality — as if designed by a senior brand designer at a creative agency
- No gradients that look cheap; use purposeful, controlled color transitions
- The overall impression: powerful, dark, premium Japanese streetwear / anime brand

Output: the logo centered on a dark square background. No mockups, no phone screens,
no merchandise. Just the clean logo on its dark background.
""".strip()

out = Path("c:/tmp/enkai-logo-ai.png")

print("Generating Enkai logo via GPT-4o...")
generate_image(
    PROMPT,
    out,
    portrait=False,   # will use 1536x1024; we'll keep it as-is (can crop to square in PS)
    quality="high",
    on_log=print,
)
print(f"\nDone. File saved to: {out}")
