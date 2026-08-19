"""Typed application settings, loaded once from .env.

Replaces the ad-hoc ``load_config()`` dict of the old project with a validated,
importable singleton. Only the keys CutForge v1 actually needs are declared.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

# Project root = two levels up from this file's package (src/cutforge/config/settings.py).
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Environment-backed configuration.

    Values come from the process environment or a ``.env`` file at the project root.
    API keys are optional at import time so the app can boot (and show the UI) even
    before keys are set — services raise a clear error when a key is actually needed.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ):
        # .env wins over the process environment. CutForge is often launched from a
        # shell that already exports ANTHROPIC_MODEL / ANTHROPIC_BASE_URL etc. for a
        # local proxy (e.g. Claude Code); those must not override the app's own .env.
        return (
            init_settings,
            dotenv_settings,
            env_settings,
            file_secret_settings,
        )

    # --- API keys ---
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    elevenlabs_api_key: str | None = None

    # --- Model overrides ---
    anthropic_model: str = "claude-sonnet-5"
    elevenlabs_model: str = "eleven_v3"
    voice_id: str = "JBFqnCBsd6RMkjVDRZzb"
    whisper_model: str = "whisper-1"

    # --- Paths ---
    output_base_dir: str = "output"

    # --- YouTube ---
    # Optional path to a Netscape-format cookies file exported from a logged-in browser
    # (e.g. via the "Get cookies.txt LOCALLY" extension). Required for age-restricted or
    # members-only videos. Set YOUTUBE_COOKIES_FILE=/path/to/cookies.txt in .env.
    youtube_cookies_file: str | None = None

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def output_dir(self) -> Path:
        base = Path(self.output_base_dir)
        return base if base.is_absolute() else PROJECT_ROOT / base

    @property
    def channels_dir(self) -> Path:
        return PROJECT_ROOT / "channels"

    def require(self, key: str) -> str:
        """Return an API key or raise a clear, actionable error if it's missing."""
        value = getattr(self, key, None)
        if not value:
            raise RuntimeError(
                f"{key.upper()} is not set. Add it to your .env file "
                f"(copy .env.example) before running this step."
            )
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide Settings singleton."""
    return Settings()
