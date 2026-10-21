# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tiny static-asset web server + status JSON for the camera companion.

Serves three things on ``http://127.0.0.1:8766``::

    GET /              → index.html  (live preview + transcript + reply)
    GET /static/*      → JS / CSS assets shipped under ``web/``
    GET /frame.jpg     → most-recent camera frame (JPEG, polled by the UI)
    GET /status        → JSON status from the SkillRunner

The WebSocket transcript channel (``ws://127.0.0.1:8765``) is set up
by ``server.py`` and uses the ``websockets`` library; it's not handled
in this module.

Stdlib-only at import. Heavy deps (cv2 for JPEG encoding) are imported
lazily on the first ``/frame.jpg`` request.
"""

from __future__ import annotations

import json
import logging
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8766


class WebUI:
    """Thread-backed http.server serving the static web/ + status JSON.

    The latest camera frame is pushed via :meth:`update_frame`; the
    HTTP layer returns the last frame for any ``/frame.jpg`` GET.
    """

    def __init__(
        self,
        runner: Any,
        web_root: Path,
        *,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
    ) -> None:
        self._runner = runner
        self._web_root = Path(web_root).resolve()
        self._host = host
        self._port = port
        self._latest_frame_jpeg: Optional[bytes] = None
        self._frame_lock = threading.Lock()
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> str:
        handler_factory = _make_handler(
            runner=self._runner,
            web_root=self._web_root,
            get_frame=self._get_frame,
        )
        self._server = ThreadingHTTPServer(
            (self._host, self._port), handler_factory
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="camera-companion-web",
            daemon=True,
        )
        self._thread.start()
        url = f"http://{self._host}:{self._port}"
        logger.info("web_ui listening on %s (root=%s)", url, self._web_root)
        return url

    def stop(self) -> None:
        if self._server is not None:
            try:
                self._server.shutdown()
            except Exception:  # noqa: BLE001
                logger.exception("web_ui: server.shutdown raised")
            try:
                self._server.server_close()
            except Exception:  # noqa: BLE001
                pass
        self._server = None
        self._thread = None

    # ------------------------------------------------------------------
    # Frame producer hooks
    # ------------------------------------------------------------------

    def update_frame(self, jpeg_bytes: bytes) -> None:
        """Called by server.py whenever a new camera frame is captured.
        The data is pre-encoded JPEG so the HTTP handler is hot-path-free."""
        with self._frame_lock:
            self._latest_frame_jpeg = bytes(jpeg_bytes)

    def _get_frame(self) -> Optional[bytes]:
        with self._frame_lock:
            return self._latest_frame_jpeg


def _make_handler(
    *,
    runner: Any,
    web_root: Path,
    get_frame: Any,
) -> type:
    """Build a request handler bound to the runner + web_root + frame
    accessor. Returns a class suitable for ThreadingHTTPServer."""

    class _Handler(BaseHTTPRequestHandler):
        # Quieter than the default stderr-spam logger.
        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            logger.debug("web %s - " + fmt, self.address_string(), *args)

        def do_GET(self) -> None:  # noqa: N802 - http.server convention
            path = self.path.split("?", 1)[0]
            if path == "/" or path == "/index.html":
                self._serve_static(web_root / "index.html", "text/html")
                return
            if path == "/frame.jpg":
                self._serve_frame()
                return
            if path == "/status":
                self._serve_status()
                return
            if path == "/scenarios":
                self._serve_json(runner.scenarios())
                return
            if path.startswith("/static/"):
                rel = path[len("/static/"):].lstrip("/")
                target = (web_root / rel).resolve()
                # Defence against path traversal.
                try:
                    target.relative_to(web_root)
                except ValueError:
                    self.send_error(HTTPStatus.FORBIDDEN)
                    return
                if target.is_file():
                    self._serve_static(target, _guess_mime(target))
                    return
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802 - http.server convention
            path = self.path.split("?", 1)[0]
            if path != "/control":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            raw = self.rfile.read(length) if length > 0 else b""
            try:
                body = json.loads(raw.decode("utf-8")) if raw else {}
                if not isinstance(body, dict):
                    raise ValueError("body must be a JSON object")
                action = str(body.get("action") or "").strip().lower()
                if action == "start":
                    result = runner.start_detection()
                elif action == "stop":
                    result = runner.stop_detection()
                elif action == "set_scenario":
                    result = runner.set_scenario(
                        str(body.get("scenario") or "auto"),
                        str(body.get("custom_prompt") or ""),
                        bool(body.get("reasoning") or False),
                    )
                else:
                    self._serve_json(
                        {"error": {"message": f"unknown action: {action!r}"}},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
            except (ValueError, json.JSONDecodeError) as exc:
                self._serve_json(
                    {"error": {"type": type(exc).__name__, "message": str(exc)}},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            except Exception as exc:  # noqa: BLE001
                self._serve_json(
                    {"error": {"type": type(exc).__name__, "message": str(exc)}},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                return
            self._serve_json(result)

        def _serve_static(self, path: Path, mime: str) -> None:
            if not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            data = path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _serve_frame(self) -> None:
            data = get_frame()
            if data is None:
                # No frame yet — return a 1x1 transparent PNG so the
                # client renders without a broken-image icon.
                data = (
                    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
                    b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00"
                    b"\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cb\x00"
                    b"\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00"
                    b"IEND\xaeB`\x82"
                )
                mime = "image/png"
            else:
                mime = "image/jpeg"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _serve_status(self) -> None:
            try:
                obj = runner.status()
            except Exception as exc:  # noqa: BLE001
                self._serve_json(
                    {"error": {"type": type(exc).__name__, "message": str(exc)}},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
                return
            self._serve_json(obj)

        def _serve_json(
            self, obj: Any, *, status: HTTPStatus = HTTPStatus.OK
        ) -> None:
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

    return _Handler


def _guess_mime(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".html": "text/html",
        ".htm": "text/html",
        ".js": "application/javascript",
        ".css": "text/css",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".svg": "image/svg+xml",
        ".json": "application/json",
    }.get(suffix, "application/octet-stream")
