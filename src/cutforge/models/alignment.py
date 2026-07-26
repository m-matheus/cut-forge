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

    def shifted(self, seconds: float) -> "Alignment":
        """Return a copy with all timestamps shifted by ``seconds`` (positive = later)."""
        if seconds == 0.0:
            return self
        new_lines = []
        for line in self.lines:
            new_words = [
                LyricWord(word=w.word,
                          start=max(0.0, round(w.start + seconds, 3)),
                          end=max(0.0, round(w.end + seconds, 3)))
                for w in line.words
            ]
            new_lines.append(LyricLine(
                start=max(0.0, round(line.start + seconds, 3)),
                end=max(0.0, round(line.end + seconds, 3)),
                words=new_words,
            ))
        return Alignment(audio=self.audio, lines=new_lines)

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
