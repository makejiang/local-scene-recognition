# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Named-pipe RPC server for the local-scene-recognition Marvis skill.

Mirrors the protocol the local-realtime-translator reference skill uses
(see ``D:\\Workspace\\Intel_Project\\os.linux.ubuntu.cloud.baseos.ai-skills\\src\\skills\\local-realtime-translator\\src\\scripts\\server.py``):
a Windows named pipe at ``\\\\.\\pipe\\local-scene-recognition`` carrying
JSON messages over the stdlib ``multiprocessing.connection`` framing.

Each message is a dict with an ``op`` key. Supported ops::

    {"op": "status"}                                    → state dict
    {"op": "start"}                                     → start session
    {"op": "ask_text", "question": "..."}               → reply dict
    {"op": "ask_audio", "audio_path": "C:/..."}         → reply dict
    {"op": "ask_audio", "record_seconds": 4.0}          → reply dict
    {"op": "stop"}                                      → {"state": "stopped"}
    {"op": "start_detection"}                           → status dict
    {"op": "stop_detection"}                            → status dict
    {"op": "set_scenario", "scenario": "flashcard_fr"}  → status dict
    {"op": "set_scenario", "scenario": "custom",
     "custom_prompt": "..."}                            → status dict
    {"op": "scenarios"}                                 → scenario catalog
    {"op": "shutdown"}                                  → graceful exit

Each request gets a response dict containing either a ``result`` field
(success) or an ``error`` field (failure, with ``type`` + ``message``).

This module is stdlib-only at import; the heavy
:class:`SkillRunner` is imported lazily on first ``start``.
"""

from __future__ import annotations

import json
import logging
import threading
from multiprocessing.connection import Listener
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


PIPE_ADDRESS = r"\\.\pipe\local-scene-recognition"
AUTHKEY = b"local-scene-recognition"


class PipeServer:
    """Single-threaded request dispatcher over the named pipe.

    The server owns ONE :class:`SkillRunner` and serializes all calls
    to it via an internal lock (the pipe is fundamentally request /
    response but a client might race a ``stop`` against an in-flight
    ``ask_text``).
    """

    def __init__(
        self,
        runner: Any,  # SkillRunner; typed Any to avoid import cycle
        *,
        on_shutdown: Optional[Any] = None,
    ) -> None:
        self._runner = runner
        self._lock = threading.Lock()
        self._on_shutdown = on_shutdown
        self._listener: Optional[Listener] = None
        self._shutdown_requested = threading.Event()

    def dispatch(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Public entry point for embedding this dispatcher without owning
        the named pipe (e.g. ``scripts/server.py`` binds its own
        ``Listener`` earlier so it can answer boot-state ``status`` calls
        while models are still downloading/loading, and only delegates
        runtime ops here once :class:`SkillRunner` is ready)."""
        return self._dispatch(msg)

    def serve_forever(self) -> None:
        """Open the named pipe and dispatch until shutdown is requested.

        Returns when an ``op: shutdown`` message arrives.
        """
        try:
            self._listener = Listener(PIPE_ADDRESS, authkey=AUTHKEY)
        except OSError as exc:
            # The pipe is already owned — almost always another live
            # server won the race (the PID single-instance lock normally
            # prevents getting this far). Fail with a clear, typed error
            # instead of a raw traceback; the caller's finally still
            # releases the camera + GPU it acquired.
            logger.error(
                "pipe_server: could not bind %s (already in use by another "
                "server?): %s", PIPE_ADDRESS, exc,
            )
            raise RuntimeError(
                f"pipe address {PIPE_ADDRESS} already in use — another "
                f"camera-companion server is likely running."
            ) from exc
        logger.info("pipe_server listening on %s", PIPE_ADDRESS)
        try:
            while not self._shutdown_requested.is_set():
                try:
                    conn = self._listener.accept()
                except Exception as exc:  # noqa: BLE001 - log + continue
                    logger.warning("listener.accept raised: %s", exc)
                    continue
                self._handle_connection(conn)
        finally:
            self._listener.close()
            self._listener = None
            logger.info("pipe_server stopped")

    def _handle_connection(self, conn: Any) -> None:
        try:
            raw = conn.recv()
            if isinstance(raw, (bytes, bytearray)):
                msg = json.loads(raw.decode("utf-8"))
            elif isinstance(raw, str):
                msg = json.loads(raw)
            elif isinstance(raw, dict):
                msg = raw
            else:
                raise ValueError(
                    f"unsupported pipe payload type: {type(raw).__name__}"
                )
            reply = self._dispatch(msg)
        except Exception as exc:  # noqa: BLE001 - protocol-level error
            logger.exception("pipe_server: dispatch failed")
            reply = {
                "error": {"type": type(exc).__name__, "message": str(exc)}
            }
        try:
            conn.send(reply)
        except Exception:  # noqa: BLE001 - client may have gone away
            logger.exception("pipe_server: send failed")
        finally:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

    def _dispatch(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        op = str(msg.get("op", "")).strip().lower()
        if not op:
            raise ValueError("missing 'op' key in message")
        with self._lock:
            if op == "status":
                return {"result": self._runner.status()}
            if op == "start":
                cfg = msg.get("config") or {}
                from .skill_runner import SkillRunnerConfig

                runner_cfg = _build_config(cfg)
                return {"result": self._runner.start(runner_cfg)}
            if op == "ask_text":
                q = str(msg.get("question") or "").strip()
                if not q:
                    raise ValueError("ask_text requires a non-empty 'question'")
                return {"result": self._runner.ask_text(q)}
            if op == "ask_audio":
                audio_path = msg.get("audio_path")
                record_seconds = msg.get("record_seconds")
                return {
                    "result": self._runner.ask_audio(
                        audio_path=audio_path,
                        record_seconds=record_seconds,
                    )
                }
            if op == "stop":
                return {"result": self._runner.stop()}
            if op == "start_detection":
                return {"result": self._runner.start_detection()}
            if op == "stop_detection":
                return {"result": self._runner.stop_detection()}
            if op == "set_scenario":
                return {
                    "result": self._runner.set_scenario(
                        str(msg.get("scenario") or "auto"),
                        str(msg.get("custom_prompt") or ""),
                        bool(msg.get("reasoning") or False),
                    )
                }
            if op == "scenarios":
                return {"result": self._runner.scenarios()}
            if op == "shutdown":
                self._shutdown_requested.set()
                stop_result = self._runner.stop()
                if callable(self._on_shutdown):
                    try:
                        self._on_shutdown()
                    except Exception:  # noqa: BLE001
                        logger.exception("on_shutdown raised")
                return {"result": {"state": "shutdown", "stop": stop_result}}
            raise ValueError(f"unknown op: {op!r}")


def _build_config(raw: Dict[str, Any]) -> Any:
    """Translate a JSON ``config`` block from the client into a
    :class:`SkillRunnerConfig` instance. Field names match the dataclass.
    Unknown fields are ignored (forward-compat)."""
    from pathlib import Path

    from .skill_runner import SkillRunnerConfig

    kwargs: Dict[str, Any] = {}
    for key in (
        "device",
        "fallback_device",
        "response_language",
        "asr_language",
        "fast_mode",
        "asr_on_gpu",
        "no_tts",
        "no_vision_text",
        "continuous_voice",
        "show_preview",
        "save_diagnostics",
    ):
        if key in raw:
            kwargs[key] = raw[key]
    if "model_dir" in raw:
        kwargs["model_dir"] = Path(str(raw["model_dir"])).expanduser()
    if "vad_model_path" in raw:
        kwargs["vad_model_path"] = Path(str(raw["vad_model_path"])).expanduser()
    if "extra_env" in raw and isinstance(raw["extra_env"], dict):
        kwargs["extra_env"] = {
            str(k): str(v) for k, v in raw["extra_env"].items()
        }
    return SkillRunnerConfig(**kwargs)
