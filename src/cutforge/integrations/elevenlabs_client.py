"""ElevenLabs text-to-speech wrapper (optional for music videos).

Ported from ``generate_voice.py``. Music videos usually skip narration, but this is
kept for the pluggable Hakase-style narration path and any spoken intro/outro.
"""
from __future__ import annotations

import base64
import time

from cutforge.config.settings import get_settings


def _client():
    from elevenlabs.client import ElevenLabs

    settings = get_settings()
    return ElevenLabs(api_key=settings.require("elevenlabs_api_key"))


def text_to_speech(text: str, *, voice_id: str | None = None, model: str | None = None,
                   max_retries: int = 3) -> bytes:
    """Convert text to speech, returning MP3 bytes (with simple backoff retry)."""
    settings = get_settings()
    client = _client()
    voice_id = voice_id or settings.voice_id
    model = model or settings.elevenlabs_model

    for attempt in range(max_retries):
        try:
            audio = client.text_to_speech.convert(
                text=text, voice_id=voice_id, model_id=model,
                output_format="mp3_44100_128",
            )
            if isinstance(audio, bytes):
                return audio
            return b"".join(chunk for chunk in audio)
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise


def _chars_to_words(alignment) -> list[dict]:
    """Collapse ElevenLabs per-character timings into per-word timings."""
    chars = alignment.characters
    starts = alignment.character_start_times_seconds
    ends = alignment.character_end_times_seconds
    words: list[dict] = []
    current: list[str] = []
    word_start = word_end = None
    for char, start, end in zip(chars, starts, ends):
        if char in (" ", "\n", "\t"):
            if current:
                words.append({"word": "".join(current), "start": word_start, "end": word_end})
                current, word_start = [], None
        else:
            if not current:
                word_start = start
            current.append(char)
            word_end = end
    if current:
        words.append({"word": "".join(current), "start": word_start, "end": word_end})
    return words


def text_to_speech_with_timestamps(text: str, *, voice_id: str | None = None,
                                   model: str | None = None,
                                   max_retries: int = 3) -> tuple[bytes, list[dict]]:
    """Convert text to speech and return (mp3 bytes, word-level timings)."""
    settings = get_settings()
    client = _client()
    voice_id = voice_id or settings.voice_id
    model = model or settings.elevenlabs_model

    for attempt in range(max_retries):
        try:
            response = client.text_to_speech.convert_with_timestamps(
                text=text, voice_id=voice_id, model_id=model,
                output_format="mp3_44100_128",
            )
            audio_bytes = base64.b64decode(response.audio_base_64)
            return audio_bytes, _chars_to_words(response.alignment)
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
