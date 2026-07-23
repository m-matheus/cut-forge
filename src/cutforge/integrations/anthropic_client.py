"""Thin Anthropic Messages API wrapper.

Ports the call pattern from the old ``generate_song_prompt.py`` — extended thinking
plus tolerant JSON parsing (the model sometimes wraps output in ```json fences).
"""
from __future__ import annotations

import json

from cutforge.config.settings import get_settings

# Pin the official API. The Anthropic SDK otherwise auto-reads ANTHROPIC_BASE_URL /
# ANTHROPIC_AUTH_TOKEN from the environment — if CutForge is launched from a shell
# that has those set (e.g. a Claude Code / local-proxy terminal), calls would be
# routed to that proxy and rejected with a 401, instead of using the .env key.
_API_BASE_URL = "https://api.anthropic.com"


def _make_client():
    import anthropic

    settings = get_settings()
    return anthropic.Anthropic(
        api_key=settings.require("anthropic_api_key"),
        base_url=_API_BASE_URL,
    )


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        # drop opening fence line
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
    return raw.strip()


def complete_json(
    system: str,
    user_prompt: str,
    *,
    max_tokens: int = 8000,
    effort: str = "high",
    model: str | None = None,
) -> dict:
    """Call Claude and parse the response as a single JSON object.

    Adaptive thinking is enabled (Claude decides how much to think; ``effort``
    tunes the depth). Raises ValueError with the raw text if the model returns
    non-JSON.
    """
    settings = get_settings()
    client = _make_client()
    model = model or settings.anthropic_model

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_prompt}],
        thinking={"type": "adaptive"},
        output_config={"effort": effort},
    )
    text = next((b.text for b in response.content if b.type == "text"), "")
    cleaned = _strip_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Claude did not return valid JSON:\n{text[:1000]}") from exc


def complete_text(
    system: str,
    user_prompt: str,
    *,
    max_tokens: int = 4000,
    model: str | None = None,
) -> str:
    """Call Claude and return the plain text response (no JSON parsing)."""
    settings = get_settings()
    client = _make_client()
    response = client.messages.create(
        model=model or settings.anthropic_model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return next((b.text for b in response.content if b.type == "text"), "").strip()
