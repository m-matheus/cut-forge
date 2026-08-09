"""Footage service — download background video clips (supports multiple per run)."""
from __future__ import annotations

from cutforge.integrations import youtube_dl
from cutforge.models.project import VideoProject


def download_footage(project: VideoProject, url: str, index: int = 0, *, on_log=None) -> dict:
    """Download YouTube footage to footage/source_NN.mp4 and record the URL."""
    dest = project.footage_path_at(index)
    dest.parent.mkdir(parents=True, exist_ok=True)
    meta = youtube_dl.download(url, dest, on_log=on_log)

    # Keep footage_urls in sync.
    urls = list(project.footage_urls)
    if index < len(urls):
        urls[index] = url
    else:
        while len(urls) < index:
            urls.append("")
        urls.append(url)
    project.footage_urls = urls
    project.save()
    return meta


def remove_footage(project: VideoProject, index: int) -> None:
    """Delete a footage file and remove its URL from the project."""
    path = project.footage_path_at(index)
    path.unlink(missing_ok=True)

    urls = list(project.footage_urls)
    if index < len(urls):
        urls.pop(index)
    project.footage_urls = urls
    project.save()

    # Rename remaining files to close the gap (source_03 → source_02, etc.)
    for i in range(index, len(urls)):
        old = project.footage_path_at(i + 1)
        new = project.footage_path_at(i)
        if old.exists() and not new.exists():
            old.rename(new)
