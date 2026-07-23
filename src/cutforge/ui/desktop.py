"""Desktop entry point — run the FastAPI app in a thread and open a pywebview window.

Dev:   python -m cutforge.ui.desktop
Server-only (browser):  python -m cutforge.ui.desktop --no-window
"""
from __future__ import annotations

import socket
import sys
import threading
import time

import uvicorn

from cutforge.api.app import app


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve(host: str, port: int) -> None:
    uvicorn.run(app, host=host, port=port, log_level="warning")


def main() -> None:
    host = "127.0.0.1"
    port = _free_port()
    url = f"http://{host}:{port}"

    server = threading.Thread(target=_serve, args=(host, port), daemon=True)
    server.start()

    # Give uvicorn a moment to bind before pointing the window at it.
    time.sleep(1.0)

    if "--no-window" in sys.argv:
        print(f"CutForge running at {url} (Ctrl+C to stop)")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            return

    import webview
    webview.create_window("CutForge", url, width=1280, height=860, min_size=(900, 640))
    webview.start()


if __name__ == "__main__":
    main()
