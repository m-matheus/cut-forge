"""Desktop entry point — run the FastAPI app and open the UI.

Two ways to run:
  python -m cutforge.ui.desktop              # native window (pywebview)
  python -m cutforge.ui.desktop --browser    # serve + open your default browser
  python -m cutforge.ui.desktop --no-window  # serve only (open the URL yourself)

Use --port N to pin a port (default: try 8760, else pick a free one). Running as a
local web app in the browser avoids the fragile PyInstaller/.exe packaging path and
gives instant reloads — it's the recommended day-to-day workflow.
"""
from __future__ import annotations

import socket
import sys
import threading
import time
import webbrowser

import uvicorn

from cutforge.api.app import app

DEFAULT_PORT = 8760


def _arg_value(flag: str, default: str | None = None) -> str | None:
    """Return the value following ``flag`` in argv (e.g. --port 9000)."""
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


def _port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) != 0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _resolve_port(host: str) -> int:
    """Pin the requested port (or the default) if free, else grab any free port."""
    requested = _arg_value("--port")
    if requested:
        return int(requested)
    if _port_is_free(host, DEFAULT_PORT):
        return DEFAULT_PORT
    return _free_port()


def _serve(host: str, port: int) -> None:
    uvicorn.run(app, host=host, port=port, log_level="warning")


def main() -> None:
    host = "127.0.0.1"
    port = _resolve_port(host)
    url = f"http://{host}:{port}"

    server = threading.Thread(target=_serve, args=(host, port), daemon=True)
    server.start()

    # Give uvicorn a moment to bind before pointing the window/browser at it.
    time.sleep(1.0)

    # Browser mode (recommended): serve locally and open the default browser.
    if "--browser" in sys.argv:
        print(f"CutForge running at {url} (Ctrl+C to stop)")
        webbrowser.open(url)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            return

    # Server-only mode: print the URL and wait (open it yourself).
    if "--no-window" in sys.argv:
        print(f"CutForge running at {url} (Ctrl+C to stop)")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            return

    # Default: native window via pywebview.
    import webview
    webview.create_window("CutForge", url, width=1280, height=860, min_size=(900, 640))
    webview.start()


if __name__ == "__main__":
    main()
