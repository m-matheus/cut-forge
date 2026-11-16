"""Channel configuration model + loader.

Each channel lives in ``channels/{slug}/channel.json`` with its brand, video specs,
voice, caption colors and asset paths. This keeps the app scalable: adding a channel
is dropping a new folder, not touching code.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

from cutforge.config.settings import get_settings


class Brand(BaseModel):
    primary_color: str = "#00E5FF"
    accent_color: str = "#FF1744"
    background_color: str = "#050505"
    text_color: str = "#FFFFFF"


class VideoSpec(BaseModel):
    resolution: str = "1920x1080"
    width: int = 1920
    height: int = 1080
    fps: int = 30


class VoiceSpec(BaseModel):
    voice_id: str = "JBFqnCBsd6RMkjVDRZzb"
    model: str = "eleven_v3"


class CaptionSpec(BaseModel):
    default_color: str = "cyan"
    unsung_color: str = "white"
    mood_colors: dict[str, str] = Field(default_factory=dict)

    def color_for_mood(self, mood: str | None) -> str:
        """Map a free-text mood to a caption color, falling back to the default."""
        if not mood:
            return self.default_color
        m = mood.lower()
        for keyword, color in self.mood_colors.items():
            if keyword in m:
                return color
        return self.default_color


class SpotifySpec(BaseModel):
    """Release as an ARTIST through a distributor, and the video's "on Spotify" call.

    The release step fills the distributor's form; the premiere step lays a QR code over
    the intro that opens the song (``spotify_track_url`` on the run) or, until that link
    exists, the artist profile.
    """
    artist_url: str = ""        # https://open.spotify.com/artist/... once the profile exists
    artist_name: str = ""       # "" = the channel name
    songwriter: str = ""        # legal name(s): distributors and Spotify require a real name
    title_template: str = "{song} ({character} Rap)"
    genre: str = "Hip-Hop/Rap"
    notice_start: float = 1.0   # seconds into the song
    notice_duration: float = 8.0
    qr_center: tuple[float, float] = (0.87, 0.30)   # frame fractions; clear of the captions
    qr_height_frac: float = 0.2                    # ~216 px at 1080p
    notice_text: dict[str, str] = Field(default_factory=lambda: {
        "en": "LISTEN ON SPOTIFY", "es": "ESCÚCHALA EN SPOTIFY", "pt": "OUÇA NO SPOTIFY",
    })

    @property
    def artist_link(self) -> str:
        """The public artist link, or "" (artists.spotify.com only opens for the owner)."""
        url = self.artist_url.strip().split("?")[0]
        return url if url.startswith("https://open.spotify.com/") else ""


class Channel(BaseModel):
    name: str
    slug: str
    content_type: str = "music"
    languages: list[str] = Field(default_factory=lambda: ["en"])
    brand: Brand = Field(default_factory=Brand)
    video: VideoSpec = Field(default_factory=VideoSpec)
    voice: VoiceSpec = Field(default_factory=VoiceSpec)
    captions: CaptionSpec = Field(default_factory=CaptionSpec)
    assets: dict[str, str] = Field(default_factory=dict)
    description_template: str = ""
    spotify: SpotifySpec = Field(default_factory=SpotifySpec)

    # Populated by the loader — absolute path to this channel's directory.
    root_dir: Path | None = None

    def asset_path(self, key: str) -> Path | None:
        """Resolve an asset (e.g. 'endcard', 'icon') to an absolute path, or None."""
        rel = self.assets.get(key)
        if not rel or self.root_dir is None:
            return None
        return (self.root_dir / rel).resolve()


def _channels_dir() -> Path:
    return get_settings().channels_dir


@lru_cache(maxsize=None)
def load_channel(slug: str) -> Channel:
    """Load and validate ``channels/{slug}/channel.json``."""
    path = _channels_dir() / slug / "channel.json"
    if not path.exists():
        raise FileNotFoundError(f"Channel config not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    channel = Channel(**data)
    channel.root_dir = path.parent
    return channel


def list_channels() -> list[Channel]:
    """Return every channel that has a valid channel.json."""
    base = _channels_dir()
    if not base.exists():
        return []
    out: list[Channel] = []
    for child in sorted(base.iterdir()):
        if (child / "channel.json").exists():
            try:
                out.append(load_channel(child.name))
            except Exception:
                continue
    return out
