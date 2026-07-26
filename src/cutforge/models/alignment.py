"""Lyric alignment models — the timed lyric structure produced by Whisper alignment.

Mirrors the JSON written to ``audio/lyrics_alignment.json`` and consumed by the
caption and Premiere-export steps.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class LyricWord(BaseModel):
    word: str
    start: float
    end: float


class LyricLine(BaseModel):
    start: float
    end: float
    words: list[LyricWord] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(w.word for w in self.words)


class Alignment(BaseModel):
    audio: str = ""
    lines: list[LyricLine] = Field(default_factory=list)

    @property
    def line_count(self) -> int:
        return len(self.lines)

    @property
    def word_count(self) -> int:
        return sum(len(l.words) for l in self.lines)

    def to_json_dict(self) -> dict:
        """Serialize in the on-disk shape used by lyrics_alignment.json."""
        return {
            "audio": self.audio,
            "line_count": self.line_count,
            "word_count": self.word_count,
            "lines": [
                {
                    "start": round(l.start, 3),
                    "end": round(l.end, 3),
                    "words": [
                        {"word": w.word, "start": round(w.start, 3), "end": round(w.end, 3)}
                        for w in l.words
                    ],
                }
                for l in self.lines
            ],
        }
