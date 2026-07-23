"""Footage service — download the background video (no scene splitting)."""
from __future__ import annotations

from cutforge.integrations import youtube_dl
from cutforge.models.project import VideoProject


def download_footage(project: VideoProject, url: str, *, on_log=None) -> dict:
    """Download the YouTube footage to footage/source.mp4 and record the URL."""
    meta = youtube_dl.download(url, project.footage_path, on_log=on_log)
    project.footage_url = url
    project.save()
    return meta
