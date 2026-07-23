"""OpenAI image generation via the Responses API (ChatGPT's internal image pipeline).

Ported from ``generate_thumbnail.py``: gpt-4o receives a natural-language request and
internally calls the image_generation tool — the same flow as typing into ChatGPT.
"""
from __future__ import annotations

import base64
from pathlib import Path

from cutforge.config.settings import get_settings

# 1536x1024 = landscape 16:9 (YouTube); 1024x1536 = portrait 9:16 (Shorts/TikTok)
SIZE_LANDSCAPE = "1536x1024"
SIZE_PORTRAIT = "1024x1536"


def generate_image(
    request: str,
    out_path: Path,
    *,
    portrait: bool = False,
    quality: str = "high",
    on_log=None,
) -> Path:
    """Generate an image from a natural-language request and write it to ``out_path``."""
    from openai import OpenAI

    settings = get_settings()
    client = OpenAI(api_key=settings.require("openai_api_key"))
    size = SIZE_PORTRAIT if portrait else SIZE_LANDSCAPE

    if on_log:
        on_log(f"Requesting image ({size}): {request[:100]}...")

    response = client.responses.create(
        model="gpt-4o",
        input=request,
        tools=[{"type": "image_generation", "quality": quality, "size": size}],
    )
    image_data = next(
        (item.result for item in response.output
         if item.type == "image_generation_call"),
        None,
    )
    if not image_data:
        raise RuntimeError("No image returned by the Responses API")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(base64.b64decode(image_data))
    if on_log:
        on_log(f"Image saved: {out_path.name}")
    return out_path
