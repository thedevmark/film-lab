"""Film Lab — minimal Flask host for the photo style processor.

Private personal tool. Browser mode by default; run from source with:

    pip install -r requirements.txt
    python app.py

Then open http://localhost:3100 in your browser.
"""

from __future__ import annotations

import os
import platform
import socket
import sys
import threading
from pathlib import Path

from flask import Flask, send_from_directory

from film import register_film_routes


# ── Paths ─────────────────────────────────────────────────────────────────────

if getattr(sys, "frozen", False):
    INTERNAL_DIR = Path(sys._MEIPASS)
    BASE_DIR = Path(sys.executable).parent.resolve()
else:
    INTERNAL_DIR = Path(__file__).parent.resolve()
    BASE_DIR = INTERNAL_DIR

if platform.system() == "Windows":
    local_app_data = os.environ.get("LOCALAPPDATA")
    APP_STATE_DIR = (Path(local_app_data) if local_app_data else BASE_DIR) / "film-lab"
else:
    APP_STATE_DIR = BASE_DIR / ".appstate"

APP_STATE_DIR.mkdir(parents=True, exist_ok=True)
PRESETS_FILE = APP_STATE_DIR / "film_presets.json"


# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder=str(INTERNAL_DIR / "static"))

# Uploads are written to disk and then decoded and graded synchronously in the
# request thread, holding several full-resolution float32 copies. Without a cap
# the body size — and so the peak memory — is whatever the caller sends.
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024 * 1024


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


register_film_routes(app, PRESETS_FILE)


# ── Server start ──────────────────────────────────────────────────────────────

DEFAULT_HOST = os.environ.get("FILM_LAB_HOST", "localhost").strip() or "localhost"
try:
    DEFAULT_PORT = int(os.environ.get("FILM_LAB_PORT", "3100"))
except ValueError:
    DEFAULT_PORT = 3100


def find_available_port(host: str, preferred: int) -> int:
    for candidate in [preferred] + list(range(preferred + 1, preferred + 50)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, candidate))
                return candidate
            except OSError:
                continue
    raise RuntimeError(f"No available port found near {preferred}.")


def start_server(host: str = DEFAULT_HOST, port: int | None = None) -> int:
    port = port or find_available_port(host, DEFAULT_PORT)

    def _run():
        app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return port


if __name__ == "__main__":
    port = find_available_port(DEFAULT_HOST, DEFAULT_PORT)
    print(f"Film Lab running on http://{DEFAULT_HOST}:{port}")
    app.run(host=DEFAULT_HOST, port=port, debug=False, use_reloader=False, threaded=True)
