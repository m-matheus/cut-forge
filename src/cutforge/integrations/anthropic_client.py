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


def _fix_control_chars(s: str) -> str:
    """Escape raw control characters that appear literally inside JSON string values.

    Claude occasionally emits a bare newline mid-word inside a string (e.g. the
    word wraps at a column boundary), producing invalid JSON. This walks the text
    character-by-character to replace control characters only inside string literals.
    """
    result = []
    in_string = False
    escaped = False
    replacements = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}
    for ch in s:
        if escaped:
            result.append(ch)
            escaped = False
        elif in_string and ch == "\\":
            result.append(ch)
            escaped = True
        elif ch == '"':
            result.append(ch)
            in_string = not in_string
        elif in_string and ch in replacements:
            result.append(replacements[ch])
        elif in_string and ord(ch) < 0x20:
            result.append(f"\\u{ord(ch):04x}")
        else:
            result.append(ch)
    return "".join(result)


def _strip_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        # drop opening fence line
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
        if raw.endswith("```"):
            raw = raw.rsplit("```", 1)[0]
    return raw.strip()


def image_block(data: bytes, media_type: str = "image/png") -> dict:
    """A user content block carrying an image (base64)."""
    import base64

    return {"type": "image", "source": {"type": "base64", "media_type": media_type,
                                        "data": base64.b64encode(data).decode("ascii")}}


def plain(text: str) -> dict:
    return {"type": "text", "text": text}


def complete_json(
    system: str,
    user_prompt: str | list[dict],
    *,
    max_tokens: int = 32000,
    effort: str = "high",
    model: str | None = None,
) -> dict:
    """Call Claude and parse the response as a single JSON object.

    Adaptive thinking is enabled (Claude decides how much to think; ``effort``
    tunes the depth). ``user_prompt`` is a string or a list of content blocks (e.g.
    ``image_block`` + ``plain``). Raises ValueError with the raw text if the model returns
    non-JSON.

    ``max_tokens`` covers thinking *and* the visible answer, so it must leave room for
    both — at high effort the thinking alone can eat most of a small budget and the JSON
    gets cut mid-string. Streams the request because the SDK refuses large ``max_tokens``
    on a plain (non-streaming) call.
    """
    settings = get_settings()
    client = _make_client()
    model = model or settings.anthropic_model

    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_prompt}],
        thinking={"type": "adaptive"},
        output_config={"effort": effort},
    ) as stream:
        response = stream.get_final_message()
    text = next((b.text for b in response.content if b.type == "text"), "")
    if response.stop_reason == "max_tokens":
        raise ValueError(
            f"Claude hit max_tokens={max_tokens} before finishing the JSON "
            f"(output_tokens={response.usage.output_tokens}). Raise max_tokens or lower "
            f"effort.\n{text[:1000]}"
        )
    cleaned = _fix_control_chars(_strip_fences(text))
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
