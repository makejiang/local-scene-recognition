# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Long-running pipeline server for the local-scene-recognition skill.

Started directly by ``client.py`` as a detached process, so the OpenVINO
compile cost is paid ONCE and survives across multiple ``run.ps1``
invocations until an explicit shutdown.

Unlike a one-shot request/response skill, this server hosts a
long-running interactive pipeline (camera + ASR + vision-LM + TTS) and
a web UI. It therefore binds its Windows named pipe
(``\\\\.\\pipe\\local-scene-recognition``) IMMEDIATELY on startup —
before the (potentially multi-minute) model download / GPU compile.
``client.py`` uses the pipe to confirm startup, then polls
``{"op": "status"}`` over that same pipe to watch the state machine
progress:

    starting -> downloading -> loading -> running
                                        |-> error

Once ``running``, every other op (``ask_text``, ``ask_audio``,
``start_detection``, ``stop_detection``, ``set_scenario``,
``scenarios``, ``start``, ``stop``, ``shutdown``) is delegated verbatim
to ``camera_companion_marvis.pipe_server.PipeServer.dispatch``, so the
wire protocol and reply/error shapes are unchanged from before this
refactor (see ``tests/tui_test.py``, which talks to the pipe directly).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
import traceback
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Any, Dict, Optional

_HERE = Path(__file__).resolve().parent
_SKILL_ROOT = _HERE.parent

# Make ``camera_companion_marvis`` (and ``ensure_models`` when imported by
# absolute name) importable when this file is run from the skill's scripts/ dir.
for _p in (str(_SKILL_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PIPE_ADDRESS = r"\\.\pipe\local-scene-recognition"
AUTHKEY = b"local-scene-recognition"
DEFAULT_SHUTDOWN_TIMEOUT = 10.0

STATE_STARTING = "starting"
STATE_DOWNLOADING = "downloading"
STATE_LOADING = "loading"
STATE_RUNNING = "running"
STATE_ERROR = "error"

OPENVINO_ROOT = Path(os.environ.get("USERPROFILE", str(Path.home()))) / ".openvino"
LOG_DIR = OPENVINO_ROOT / "log"

# Bootstrap log — written by run.ps1 BEFORE Python is even spawned.
# When we land here we know run.ps1 ran. Forward our first lines into
# this file so a single tail covers run.ps1 + server.py.
import tempfile as _tempfile  # noqa: E402

_BOOTSTRAP_LOG = (
    Path(os.environ["CAMERA_COMPANION_BOOTSTRAP_LOG"])
    if "CAMERA_COMPANION_BOOTSTRAP_LOG" in os.environ
    else Path(_tempfile.gettempdir()) / "camera-companion-bootstrap.log"
)


def _bootstrap(msg: str) -> None:
    """Append a timestamped line to the bootstrap log.

    Best-effort: never raises. Useful for the small window between
    process start and ``_setup_logging`` succeeding.
    """
    try:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        _BOOTSTRAP_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _BOOTSTRAP_LOG.open("a", encoding="utf-8") as fh:
            fh.write(f"[{stamp}] server.py: {msg}\n")
    except Exception:
        pass


_bootstrap(f"module imported; sys.executable={sys.executable}; cwd={os.getcwd()}")

LOG_LOCK = threading.Lock()


def _normalize_log_path(log_path: str | None) -> Path | None:
    if not log_path:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_path = f"~/.openvino/log/camera-companion-server-py-{timestamp}.log"
    path = Path(log_path).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def _setup_logging(log_path: Optional[Path]) -> None:
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        if log_path is None:
            ts = time.strftime("%Y%m%d-%H%M%S")
            log_path = LOG_DIR / f"camera-companion-server-py-{ts}.log"
        handlers.insert(0, logging.FileHandler(log_path, encoding="utf-8"))
    except Exception as exc:
        # Fall back to stderr-only logging so server.py never silently
        # loses its trace. Bootstrap log captures this too.
        _bootstrap(
            f"_setup_logging: FileHandler({log_path}) failed: "
            f"{type(exc).__name__}: {exc}; using stderr only"
        )
        log_path = None
    logging.basicConfig(
        level=logging.INFO, format=fmt, handlers=handlers, force=True,
    )
    if log_path is not None:
        logging.info("server.py started; log file = %s", log_path)
    else:
        logging.info("server.py started; logging to stderr only")


def _log_message(log_path: Path | None, message: str) -> None:
    if log_path is None:
        return
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with LOG_LOCK:
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(f"[{timestamp}] [server pid={os.getpid()}] {message}\n")
    except OSError:
        pass


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------
def _downloads_needed(skill_root: Path) -> bool:
    """True if any model still needs downloading. Reused by main() to
    avoid advertising the 'downloading' state (which makes client.py
    bail with exit 3) when the models are already cached. On any error
    reading info.json, return True so the download phase still runs."""
    from ensure_models import all_present  # type: ignore

    info_path = skill_root / "info.json"
    if not info_path.is_file():
        return True
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
        return not all_present(info)
    except Exception:  # noqa: BLE001 - be safe: attempt the download
        return True


def _run_downloads_loop(skill_root: Path, log_fn) -> int:
    """Run ``ensure_models.download_all`` until it reports done or a hard
    failure. Unlike the previous exit-code-3-and-relaunch-the-process
    design, the server is now a long-lived detached process, so a
    partial per-invocation download budget just means
    "call download_all again" instead of exiting — client.py's
    ``--continue`` simply reconnects to the same still-downloading
    server via the pipe.

    Returns 0 on success, 1 on hard failure.
    """
    from ensure_models import download_all  # type: ignore

    info_path = skill_root / "info.json"
    if not info_path.is_file():
        log_fn(f"info.json missing at {info_path}")
        return 1
    info = json.loads(info_path.read_text(encoding="utf-8"))
    while True:
        try:
            rc = download_all(info)
        except Exception:
            log_fn(f"download_all raised:\n{traceback.format_exc()}")
            return 1
        if rc == 0:
            return 0
        if rc == 1:
            return 1
        log_fn("download budget elapsed for this pass; continuing download loop")


def _build_runner_config(args: argparse.Namespace) -> Any:
    from camera_companion_marvis.skill_runner import SkillRunnerConfig

    model_dir = OPENVINO_ROOT / "models" / "MiniCPM-o-4_5-OV"
    vad_path = OPENVINO_ROOT / "models" / "silero_vad" / "silero_vad.onnx"
    return SkillRunnerConfig(
        model_dir=Path(model_dir),
        vad_model_path=Path(vad_path),
        response_language=args.language,
        asr_language=args.language,
        no_tts=bool(args.no_tts),
        fast_mode=not bool(args.no_fast_mode),
    )


def _ensure_asr_runtime_deps() -> Optional[str]:
    """Verify every runtime dep the voice loop + ASR actually need is
    importable in this venv. Returns ``None`` on success, or a
    human-readable description of the missing-dep state on failure.

    The MiniCPM-o audio encoder needs ``librosa`` for the Whisper-style
    log-mel front-end; the continuous-voice loop needs ``sounddevice``
    (microphone capture) and ``soundfile`` (WAV writes); OpenVINO,
    transformers, optimum and numpy < 2.0 are the runtime spine. When
    any of these is missing, the skill silently breaks in a different
    place — for example, missing ``sounddevice`` lets ``SkillRunner.start``
    succeed and the WebUI come up "running", but the voice-loop thread
    crashes on first frame so the pill never leaves ``idle`` and the
    user thinks ASR is broken (real symptom we saw on the test
    machine, where install-env.ps1 had healed librosa but somehow not
    sounddevice).

    This preflight only DETECTS — it never tries to ``pip install``
    from inside the server process. An earlier attempt did and broke
    the venv (librosa pulls scipy / numba / numpy and the upgrade-on-
    top-of-live-process left OpenVINO plugin DLLs unable to load with
    WinError 1455). The correct heal path is out-of-process: re-run
    ``scripts\\install-env.ps1`` (or recreate the venv), which
    serialises pip against the full requirements set.

    Best-effort: never raises. The caller only treats this as fatal
    when the user is sure to be blocked; a missing optional dep just
    surfaces a warning through the status reply / WebUI.
    """
    # Probe every dep the .bat live demo exercises. Mapping name →
    # importable module name → pip spec used in requirements.txt.
    probes = (
        ("librosa", "librosa", "librosa>=0.10.0"),
        ("scipy", "scipy", "scipy>=1.10.0"),
        ("sounddevice", "sounddevice", "sounddevice>=0.4.6"),
        ("soundfile", "soundfile", "soundfile>=0.12"),
        ("opencv-python", "cv2", "opencv-python>=4.8"),
        ("onnxruntime", "onnxruntime", "onnxruntime>=1.18"),
    )
    missing: list[tuple[str, str, str]] = []
    for pkg_name, module_name, pip_spec in probes:
        try:
            __import__(module_name)
        except ImportError as exc:
            missing.append((pkg_name, str(exc), pip_spec))
    if not missing:
        _bootstrap("preflight: all runtime deps importable")
        return None

    names = ", ".join(p[0] for p in missing)
    pip_args = " ".join(p[2] for p in missing)
    msg = (
        f"Missing Python deps in venv: {names}. The voice loop and "
        f"ASR will not work until these are installed. To heal, "
        f"either re-run scripts\\install-env.ps1, OR run from "
        f"PowerShell:  & \"{sys.executable}\" -m pip install {pip_args}"
    )
    _bootstrap(f"preflight: {msg}")
    logging.error("preflight: %s", msg)
    for pkg_name, exc_msg, _ in missing:
        logging.error("preflight: %s -> ImportError(%s)", pkg_name, exc_msg)
    return msg


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
class Server:
    def __init__(self, args: argparse.Namespace, log_path: Path | None = None) -> None:
        self.args = args
        self.start_time = time.time()
        self.shutdown_event = threading.Event()
        self.shutdown_timeout = DEFAULT_SHUTDOWN_TIMEOUT
        self.listener: Listener | None = None
        self.runtime_lock = threading.Lock()
        self.runner: Any = None
        self.web_ui: Any = None
        self._pipe_dispatcher: Any = None
        self.web_url: Optional[str] = None
        self.session_id: Optional[str] = None
        self.warmup: Any = None
        self.preflight_warning: Optional[str] = None
        self.log_path = log_path
        self.state: str = STATE_STARTING
        self.init_error: str = ""
        self.init_thread: threading.Thread | None = None

    def log(self, message: str) -> None:
        _log_message(self.log_path, message)

    # ── init worker: download -> load -> run ──────────────────────────
    def _init_runtime_worker(self) -> None:
        try:
            if not self.args.skip_models and _downloads_needed(_SKILL_ROOT):
                with self.runtime_lock:
                    self.state = STATE_DOWNLOADING
                self.log("init: downloading required models")

                def _dl_log(msg: str) -> None:
                    print(f"[server] {msg}", flush=True)
                    self.log(msg)

                rc = _run_downloads_loop(_SKILL_ROOT, _dl_log)
                if rc != 0:
                    raise RuntimeError("model download failed")
            else:
                self.log("models already present; skipping download phase")

            # Preflight: detect missing runtime deps in the venv. Surfaced
            # as a warning on the running status reply rather than a hard
            # failure — see _ensure_asr_runtime_deps docstring.
            missing_deps_msg = _ensure_asr_runtime_deps()

            with self.runtime_lock:
                self.state = STATE_LOADING
            self.log("init: loading SkillRunner (OpenVINO pipeline + camera session)")

            from camera_companion_marvis.pipe_server import PipeServer
            from camera_companion_marvis.skill_runner import SkillRunner
            from camera_companion_marvis.web_ui import WebUI

            runner = SkillRunner()
            cfg = _build_runner_config(self.args)
            self.log(f"init: SkillRunner.start with cfg={cfg!r}")
            boot = runner.start(cfg)
            self.log(f"init: SkillRunner.start succeeded: session={boot.get('session_id')}")

            web_root = _SKILL_ROOT / "web"
            web_ui = WebUI(runner, web_root)
            web_url = web_ui.start()
            self.log(f"init: WebUI started at {web_url}")

            try:
                runner.set_frame_callback(web_ui.update_frame)
            except Exception:  # noqa: BLE001 - best effort
                self.log(f"set_frame_callback raised:\n{traceback.format_exc()}")

            pipe_dispatcher = PipeServer(runner, on_shutdown=web_ui.stop)

            with self.runtime_lock:
                self.runner = runner
                self.web_ui = web_ui
                self._pipe_dispatcher = pipe_dispatcher
                self.web_url = web_url
                self.session_id = boot.get("session_id")
                self.warmup = boot.get("warmup")
                self.preflight_warning = missing_deps_msg
                self.state = STATE_RUNNING
            self.log(f"init: running ({web_url})")

            banner = [
                "",
                "=== 本地启蒙学习助手已就绪 / Early Learning Assistant ready ===",
                f"  回答语言 / language: {self.args.language}",
                f"  请在浏览器打开 / Open in browser:  {web_url}",
                f"  会话 ID / session: {boot.get('session_id')}",
                "  对着摄像头说话或按空格键提问。",
                "  停止服务 / stop:  scripts\\run.ps1 --stop",
                "",
            ]
            for line in banner:
                print(line, flush=True)
                logging.info("%s", line)
        except Exception:
            error_text = traceback.format_exc()
            with self.runtime_lock:
                self.init_error = error_text
                self.state = STATE_ERROR
            self.log(f"init failed:\n{error_text}")

    def _start_init_thread(self) -> None:
        if self.init_thread is not None and self.init_thread.is_alive():
            return
        self.init_thread = threading.Thread(
            target=self._init_runtime_worker, name="runtime-init", daemon=True
        )
        self.init_thread.start()
        self.log("init thread started")

    def _do_shutdown(self, timeout: float | None = None) -> Dict[str, Any]:
        """Stop the runner + WebUI and signal the accept loop to exit.

        Idempotent: safe to call more than once (e.g. once from the pipe
        `shutdown` op and once from a SIGINT handler in --dev mode).
        """
        if timeout is not None:
            try:
                self.shutdown_timeout = float(timeout)
            except (TypeError, ValueError):
                self.shutdown_timeout = DEFAULT_SHUTDOWN_TIMEOUT
        with self.runtime_lock:
            pipe_dispatcher = self._pipe_dispatcher
        inner: Dict[str, Any] = {}
        if pipe_dispatcher is not None:
            try:
                inner = pipe_dispatcher.dispatch({"op": "shutdown"})
            except Exception:
                self.log(f"pipe_dispatcher shutdown raised:\n{traceback.format_exc()}")
        self.shutdown_event.set()
        return inner if isinstance(inner, dict) else {}

    # ── control-op dispatch ────────────────────────────────────────────
    def dispatch(self, msg: dict) -> dict:
        op = str(msg.get("op", "")).strip().lower()
        self.log(f"dispatching op={op!r}")

        with self.runtime_lock:
            state = self.state
            init_error = self.init_error
            pipe_dispatcher = self._pipe_dispatcher
            web_url = self.web_url
            session_id = self.session_id
            warmup = self.warmup
            preflight_warning = self.preflight_warning

        if op == "status":
            reply: Dict[str, Any] = {
                "ok": True,
                "state": state,
                "pid": os.getpid(),
                "uptime_s": time.time() - self.start_time,
                "web_url": web_url,
                "session_id": session_id,
                "warmup": warmup,
            }
            if preflight_warning:
                reply["preflight_warning"] = preflight_warning
            if state == STATE_ERROR and init_error:
                reply["error"] = init_error
            try:
                from ensure_models import get_download_progress  # type: ignore
                progress = get_download_progress()
            except Exception:
                progress = None
            if progress is not None:
                reply["progress"] = progress
            if pipe_dispatcher is not None:
                try:
                    reply.update(pipe_dispatcher.dispatch({"op": "status"}))
                except Exception:
                    self.log(f"runner.status() raised:\n{traceback.format_exc()}")
            return reply

        if op == "shutdown":
            timeout = msg.get("timeout", DEFAULT_SHUTDOWN_TIMEOUT)
            self.log(f"shutdown requested (timeout={timeout})")
            inner = self._do_shutdown(timeout)
            reply = {"ok": True, "state": "shutting_down"}
            reply.update(inner)
            return reply

        # Every other op requires the runner to be up.
        if pipe_dispatcher is None:
            self.log(f"op {op!r} rejected: state={state}")
            reply = {"ok": False, "state": state, "error": f"runtime not ready: {state}"}
            if state == STATE_ERROR and init_error:
                reply["error"] = init_error
            return reply

        # Delegate verbatim to the existing pipe protocol (ask_text,
        # ask_audio, start_detection, stop_detection, set_scenario,
        # scenarios, start, stop) — same reply/error shapes as before
        # this refactor.
        try:
            return pipe_dispatcher.dispatch(msg)
        except Exception as exc:  # noqa: BLE001 - protocol-level error
            self.log(f"pipe dispatch failed for op={op!r}:\n{traceback.format_exc()}")
            return {"error": {"type": type(exc).__name__, "message": str(exc)}}

    def serve_forever(self) -> None:
        assert self.listener is not None
        while not self.shutdown_event.is_set():
            try:
                conn = self.listener.accept()
                self.log("accepted client connection")
            except OSError:
                if self.shutdown_event.is_set():
                    self.log("listener stopped after shutdown was requested")
                    return
                raise

            try:
                msg = conn.recv()
                op_name = msg.get("op") if isinstance(msg, dict) else None
                self.log(f"client request: op={op_name!r}")
                reply = self.dispatch(msg if isinstance(msg, dict) else {})
                conn.send(reply)
                self.log(f"server reply ok={reply.get('ok')}")
            except EOFError:
                self.log("client connection closed before a full request was received")
            except Exception:
                error_text = traceback.format_exc()
                try:
                    conn.send({"ok": False, "error": error_text})
                except Exception:
                    pass
                self.log(f"accept-loop error:\n{error_text}")
                print(f"[server] accept-loop error:\n{error_text}", flush=True)
            finally:
                try:
                    conn.close()
                except Exception:
                    pass


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog="server.py")
    parser.add_argument("--log", default=None, help="Log file path.")
    parser.add_argument(
        "--language",
        default=os.environ.get("CAMERA_COMPANION_LANGUAGE", "auto"),
        choices=("auto", "zh", "en"),
        help="Default reply + ASR language.",
    )
    parser.add_argument(
        "--no-tts", action="store_true",
        default=_env_flag("CAMERA_COMPANION_NO_TTS"),
    )
    parser.add_argument(
        "--no-fast-mode", action="store_true",
        default=_env_flag("CAMERA_COMPANION_NO_FAST_MODE"),
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip the download phase (used by hermetic tests).",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help=(
            "Dev mode: install a SIGINT handler for graceful Ctrl-C when "
            "running this file directly in a console."
        ),
    )
    args = parser.parse_args(argv)
    _bootstrap(
        f"main() parsed args: language={args.language} no_tts={args.no_tts} "
        f"no_fast_mode={args.no_fast_mode} skip_models={args.skip_models} dev={args.dev}"
    )

    try:
        _setup_logging(Path(args.log) if args.log else None)
    except Exception as exc:
        _bootstrap(f"_setup_logging failed: {type(exc).__name__}: {exc}")
        print(
            f"[camera-companion] ERROR: _setup_logging failed: {exc}",
            file=sys.stderr, flush=True,
        )
        # Don't abort — continue without file logging; stderr still works.
    logging.info(
        "server.py CWD=%s SKILL_ROOT=%s sys.executable=%s",
        os.getcwd(), _SKILL_ROOT, sys.executable,
    )

    server = Server(args, log_path=_normalize_log_path(args.log))
    server.log(f"server starting with argv={sys.argv[1:]}")
    try:
        server.listener = Listener(PIPE_ADDRESS, family="AF_PIPE", authkey=AUTHKEY)
    except OSError as e:
        server.log(f"bind failed: {e}")
        print(f"[server] bind failed: {e}", flush=True)
        _bootstrap(f"listener bind failed: {e}")
        return 2

    server.log(f"listener ready on {PIPE_ADDRESS}")
    _bootstrap(f"listener bound on {PIPE_ADDRESS}; starting init thread")
    server._start_init_thread()
    print(
        f"local-scene-recognition server listening on {PIPE_ADDRESS} (pid={os.getpid()})",
        flush=True,
    )

    if args.dev:
        def _on_signal(*_: Any) -> None:
            logging.info("SIGINT received; shutting down")
            threading.Thread(target=server._do_shutdown, daemon=True).start()

        try:
            signal.signal(signal.SIGINT, _on_signal)
        except Exception:  # noqa: BLE001
            pass

    worker = threading.Thread(target=server.serve_forever, name="accept-loop", daemon=True)
    worker.start()

    try:
        while not server.shutdown_event.wait(timeout=0.5):
            if not worker.is_alive():
                server.log("accept-loop thread exited unexpectedly")
                return 1
    finally:
        try:
            if server.listener is not None:
                server.listener.close()
        except Exception:
            pass

    worker.join(timeout=server.shutdown_timeout)
    if worker.is_alive():
        server.log(f"worker did not finish within {server.shutdown_timeout:.1f}s; forcing exit")
        print(f"[server] worker did not finish within {server.shutdown_timeout:.1f}s; forcing exit", flush=True)
        os._exit(1)
    server.log("server exited cleanly")
    return 0


if __name__ == "__main__":
    # Top-level crash trap. Without this, an exception inside main() before
    # _setup_logging() succeeds (e.g. permission denied on the log dir,
    # broken venv) propagates as an opaque "Python process exited" with
    # nothing in the bootstrap log — invisible to client.py.
    import traceback as _traceback

    try:
        _bootstrap("main() entry")
        rc = main()
        _bootstrap(f"main() returned {rc}")
        raise SystemExit(rc)
    except SystemExit:
        raise
    except BaseException as _exc:  # noqa: BLE001
        _err = "".join(_traceback.format_exception(_exc))
        _bootstrap(f"main() raised:\n{_err}")
        print(_err, file=sys.stderr, flush=True)
        raise SystemExit(99)
