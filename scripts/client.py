# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Short-lived CLI controller for the local-scene-recognition server.

This is **not** where the camera companion's work happens. It starts the
detached ``server.py`` directly when needed, then controls it over a Windows
named pipe.

Subcommands (selected by flag):

    (default)            start the pipeline server, print the web URL, exit
    --continue           resume a first-run model download
    --server-status      report server state (+ web URL when running)
    --server-shutdown    cleanly stop the server (releases camera + GPU)
    --restart            stop-then-start with the new config

The web UI on http://127.0.0.1:8766 is the user-facing interface; this
client just controls when it goes up + down. Successful start exits 0,
download-in-progress exits 3, hard failures exit 1.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from multiprocessing.connection import Client
from pathlib import Path
from typing import Any, Dict, Optional

import server_task

_HERE = Path(__file__).resolve().parent
_SKILL_ROOT = _HERE.parent

PIPE_ADDRESS = r"\\.\pipe\local-scene-recognition"
AUTHKEY = b"local-scene-recognition"
SKILL_NAME = "local-scene-recognition"
# The server binds its pipe IMMEDIATELY on startup (before the model
# download / GPU compile), so 60s is plenty here — unlike the old
# synchronous subprocess design, this wait is NOT gated on the ~2min GPU
# compile. That is watched via `_wait_for_running`'s RPC status poll,
# bounded by the download_wait_timeout_minutes setting below.
SERVER_BOOT_TIMEOUT = 60.0
SERVER_BOOT_POLL_INTERVAL = 0.3
STATUS_POLL_INTERVAL = 2.0
# While a model download is running the status is polled more often so the
# progress line refreshes several times per second of wall clock time.
PROGRESS_POLL_INTERVAL = 1.0
DEFAULT_SHUTDOWN_TIMEOUT = 10.0

OPENVINO_ROOT = Path(os.environ.get("USERPROFILE", str(Path.home()))) / ".openvino"
LOG_DIR = OPENVINO_ROOT / "log"
PENDING_REQUEST_PATH = OPENVINO_ROOT / "local-scene-recognition-pending-request.json"
# The server runs in place, from the skill install dir.
SERVER_PATH = _HERE / "server.py"
LAUNCHER_PATH = _HERE / "task_launcher.py"
TASK_RUN_DIR = _SKILL_ROOT / "run"
# Written by the client, read by task_launcher.py: a Scheduled Task cannot
# carry a custom environment block.
TASK_ENV_FILE = TASK_RUN_DIR / "server-env.json"
# Written by task_launcher.py before it imports the server.
TASK_PID_FILE = TASK_RUN_DIR / "task-server.pid"
TASK_PID_GRACE = 10.0

# Bootstrap log shared with run.ps1 and server.py.
import tempfile as _tempfile

_BOOTSTRAP_LOG = (
    Path(os.environ["CAMERA_COMPANION_BOOTSTRAP_LOG"])
    if "CAMERA_COMPANION_BOOTSTRAP_LOG" in os.environ
    else Path(_tempfile.gettempdir()) / "camera-companion-bootstrap.log"
)


def _bootstrap(msg: str) -> None:
    try:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        _BOOTSTRAP_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _BOOTSTRAP_LOG.open("a", encoding="utf-8") as fh:
            fh.write(f"[{stamp}] client.py: {msg}\n")
    except Exception:
        pass


_bootstrap(f"module imported; sys.executable={sys.executable}; cwd={os.getcwd()}")


def _setup_logging(log_path: Optional[Path]) -> None:
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    handlers = [logging.StreamHandler(sys.stderr)]
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        if log_path is None:
            ts = time.strftime("%Y%m%d-%H%M%S")
            log_path = LOG_DIR / f"camera-companion-client-py-{ts}.log"
        handlers.insert(0, logging.FileHandler(log_path, encoding="utf-8"))
    except Exception as exc:
        _bootstrap(
            f"_setup_logging: FileHandler failed: "
            f"{type(exc).__name__}: {exc}; using stderr only"
        )
        log_path = None
    logging.basicConfig(
        level=logging.INFO, format=fmt, handlers=handlers, force=True,
    )
    logging.info(
        "client.py started; log file = %s; bootstrap = %s",
        log_path, _BOOTSTRAP_LOG,
    )


# ── pending-request (for --continue resume) ────────────────────────────
def _save_pending_request(language: str, no_tts: bool, no_fast_mode: bool,
                          log_path: Path | None) -> None:
    try:
        PENDING_REQUEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "language": language,
            "no_tts": no_tts,
            "no_fast_mode": no_fast_mode,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "log_path": str(log_path) if log_path is not None else None,
        }
        PENDING_REQUEST_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as exc:
        logging.warning("failed to save pending request: %s", exc)


def _load_pending_request() -> dict | None:
    try:
        data = json.loads(PENDING_REQUEST_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _delete_pending_request() -> None:
    try:
        PENDING_REQUEST_PATH.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        logging.warning("failed to delete pending request: %s", exc)


# ── pipe helpers ────────────────────────────────────────────────────────
def _try_connect():
    try:
        return Client(PIPE_ADDRESS, authkey=AUTHKEY)
    except (FileNotFoundError, OSError, EOFError):
        return None


def _hidden_startupinfo():
    if os.name != "nt" or not hasattr(subprocess, "STARTUPINFO"):
        return None
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= getattr(subprocess, "STARTF_USESHOWWINDOW", 0)
    startupinfo.wShowWindow = getattr(subprocess, "SW_HIDE", 0)
    return startupinfo


def _pythonw_executable() -> str:
    python_exe = Path(sys.executable)
    pythonw_exe = python_exe.with_name("pythonw.exe")
    return str(pythonw_exe) if pythonw_exe.exists() else str(python_exe)


def _server_overrides(language: str, no_tts: bool, no_fast_mode: bool) -> dict:
    return {
        "CAMERA_COMPANION_LANGUAGE": language,
        "CAMERA_COMPANION_NO_TTS": "1" if no_tts else "0",
        "CAMERA_COMPANION_NO_FAST_MODE": "1" if no_fast_mode else "0",
        "OPENVINO_TELEMETRY_OPT_OUT": "1",
    }


def _read_task_pid() -> int | None:
    try:
        return int(TASK_PID_FILE.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _clear_task_files() -> None:
    for path in (TASK_PID_FILE, TASK_ENV_FILE):
        try:
            path.unlink()
        except OSError:
            pass


class _TaskServerHandle:
    """A Popen-shaped view of a server running under a Scheduled Task.

    The boot loop only needs ``poll()`` / ``returncode``.  Liveness comes from
    the pid file the launcher writes rather than the task state: Task
    Scheduler state can lag the actual process, while the pid check looks at
    the launched process directly.  The task itself is queried only once a
    failure is established, to put the action's exit code in the error message.
    """

    def __init__(self, started_at: float):
        self._started_at = started_at
        self.returncode: int | None = None

    def _fail(self, reason: str) -> int:
        code = server_task.last_result(SKILL_NAME, logging.info)
        if code in server_task.NON_EXIT_RESULTS:
            # "still running" / "never ran" are task states, not exit codes;
            # reporting one as the server's exit status would be a lie.
            code = None
        logging.info("task-backed server not running (%s); last_result=%s", reason, code)
        self.returncode = -1 if code is None else code
        return self.returncode

    def poll(self) -> int | None:
        if self.returncode is not None:
            return self.returncode
        pid = _read_task_pid()
        if pid is None:
            # The launcher writes the pid before the (slow) server import, so a
            # pid file that never appears means the task did not run at all --
            # blocked by policy, wrong interpreter, bad action.  Without this
            # the caller would wait out the whole boot timeout for a server
            # that was never launched.
            if time.time() - self._started_at < TASK_PID_GRACE:
                return None
            return self._fail(f"no pid file after {TASK_PID_GRACE:.0f}s")
        if server_task.pid_alive(pid):
            return None
        return self._fail(f"pid {pid} exited")


def _spawn_server_direct(language: str, no_tts: bool, no_fast_mode: bool) -> subprocess.Popen:
    """Start the server as a child process (hosts that let it outlive us)."""
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(server_task.HOST_ENV_PREFIXES)}
    env.update(_server_overrides(language, no_tts, no_fast_mode))
    OPENVINO_ROOT.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [_pythonw_executable(), str(SERVER_PATH)],
        creationflags=creationflags,
        startupinfo=_hidden_startupinfo(),
        close_fds=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(OPENVINO_ROOT),
        env=env,
    )
    logging.info("spawned server directly: pid=%s", proc.pid)
    return proc


def _spawn_server_task(language: str, no_tts: bool, no_fast_mode: bool) -> _TaskServerHandle:
    """Start the server under a Scheduled Task so it outlives this command."""
    OPENVINO_ROOT.mkdir(parents=True, exist_ok=True)
    _clear_task_files()
    server_task.write_env_file(
        TASK_ENV_FILE,
        server_task.select_env(os.environ, _server_overrides(language, no_tts, no_fast_mode)),
    )
    started_at = time.time()
    server_task.start(
        skill=SKILL_NAME,
        launcher=LAUNCHER_PATH,
        env_file=TASK_ENV_FILE,
        pid_file=TASK_PID_FILE,
        server_py=SERVER_PATH,
        server_args=[],
        working_dir=OPENVINO_ROOT,
        log=logging.info,
    )
    logging.info("started server via scheduled task %s", server_task.task_name(SKILL_NAME))
    return _TaskServerHandle(started_at)


def _spawn_server(language: str, no_tts: bool, no_fast_mode: bool):
    """Start the server the way this host requires.

    Returns a handle exposing ``poll()``/``returncode`` either way, so the boot
    loop reads the same.
    """
    if server_task.should_use_task():
        try:
            return _spawn_server_task(language, no_tts, no_fast_mode)
        except (RuntimeError, OSError) as exc:
            # Falling back leaves the server inside the host's job object, so
            # it will still die with the turn -- but that is exactly today's
            # behaviour, whereas failing here would be a new way to break.
            logging.warning("scheduled task path unavailable (%s); falling back to direct spawn", exc)
            print(f"[WARN] 计划任务不可用, 回退为直接启动 (服务可能无法在命令结束后存活): {exc}",
                  file=sys.stderr)
    return _spawn_server_direct(language, no_tts, no_fast_mode)


def _adopt_booting_server() -> _TaskServerHandle | None:
    """Return a handle for a task-backed server that is up but not yet listening.

    The pipe is the only liveness signal the rest of the client has, and it
    only answers once the server has bound it.  A re-invocation during that
    window -- which is exactly what the `--continue` resume path does -- would
    otherwise conclude there is no server and start a second one contending
    for the same camera.
    """
    pid = _read_task_pid()
    if pid is None or not server_task.pid_alive(pid):
        return None
    logging.info("task-backed server already booting (pid=%s); waiting instead of starting another", pid)
    return _TaskServerHandle(time.time())


def _read_info_json() -> dict:
    return json.loads((_SKILL_ROOT / "info.json").read_text(encoding="utf-8"))


def _download_wait_deadline() -> float | None:
    raw_timeout = _read_info_json().get("download_wait_timeout_minutes", 9)
    try:
        timeout_minutes = float(raw_timeout)
    except (TypeError, ValueError):
        logging.warning(
            "invalid download_wait_timeout_minutes=%r; defaulting to 9 minutes",
            raw_timeout,
        )
        timeout_minutes = 9.0
    if timeout_minutes < 0:
        logging.info("download wait timeout disabled")
        return None
    return time.time() + timeout_minutes * 60.0


def _send(conn, msg: dict) -> dict:
    try:
        conn.send(dict(msg))
        reply = conn.recv()
    finally:
        conn.close()
    return reply if isinstance(reply, dict) else {}


def _ensure_server(language: str, no_tts: bool, no_fast_mode: bool) -> None:
    conn = _try_connect()
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass
        return
    proc = _adopt_booting_server()
    if proc is None:
        logging.info("server not running; starting it")
        proc = _spawn_server(language, no_tts, no_fast_mode)
    deadline = time.time() + SERVER_BOOT_TIMEOUT
    while time.time() < deadline:
        conn = _try_connect()
        if conn is not None:
            conn.close()
            return
        if proc.poll() is not None:
            raise RuntimeError(f"server exited during startup (exit={proc.returncode})")
        time.sleep(SERVER_BOOT_POLL_INTERVAL)
    raise RuntimeError(f"server did not come up within {SERVER_BOOT_TIMEOUT:.0f}s")


def _wait_for_running(deadline: float | None) -> tuple[str, dict]:
    last_reply: dict = {}
    while True:
        if deadline is not None and time.time() >= deadline:
            return ("timeout", last_reply)
        conn = _try_connect()
        if conn is None:
            time.sleep(STATUS_POLL_INTERVAL)
            continue
        reply = _send(conn, {"op": "status"})
        last_reply = reply
        state = reply.get("state")
        logging.info("wait_for_running: state=%s", state)
        _report_download_progress(reply)
        if state == "running":
            return ("running", reply)
        if state == "error":
            return ("error", reply)
        remaining = None if deadline is None else deadline - time.time()
        if remaining is not None and remaining <= 0:
            return ("timeout", last_reply)
        interval = (
            PROGRESS_POLL_INTERVAL
            if _PROGRESS_STATE.get("progress") is not None
            else STATUS_POLL_INTERVAL
        )
        time.sleep(interval if remaining is None else min(interval, remaining))


def _print_ready(reply: Dict[str, Any]) -> None:
    url = reply.get("web_url") or "http://127.0.0.1:8766"
    # open the web UI in the default browser, but don't wait for it to exit (it may be a long-running process like Chrome)
    try:
        import webbrowser
        webbrowser.open(url, new=2)
    except Exception as exc:
        logging.warning("failed to open web browser: %s", exc)
        
    banner = [
        "",
        "=== 本地启蒙学习助手已就绪 / Early Learning Assistant ready ===",
        f"  请在浏览器打开 / Open in browser:  {url}",
        f"  会话 ID / session: {reply.get('session_id')}",
        "  对着摄像头说话或按空格键提问。",
        "  停止服务 / 请直接告诉我\"停止服务\"",
        "",
    ]
    for line in banner:
        print(line, flush=True)


_PROGRESS_STATE: dict = {}


def _format_download_progress(progress) -> str:
    try:
        from model_download import format_download_progress
    except Exception:  # noqa: BLE001 — progress display must never break the client
        return "模型下载中..."
    return format_download_progress(progress)


def _should_display_progress(state) -> bool:
    try:
        from model_download import should_display_progress
    except Exception:  # noqa: BLE001 — throttling must never break the client
        return True
    return should_display_progress(state)


def _report_download_progress(reply, force: bool = False) -> None:
    progress = reply.get("progress") if isinstance(reply, dict) else None
    if progress is not None:
        _PROGRESS_STATE["progress"] = progress
    elif not force:
        return
    line = _format_download_progress(progress or _PROGRESS_STATE.get("progress"))
    if not force and line == _PROGRESS_STATE.get("line"):
        return
    # First line shows immediately, then at most one line per 5 minutes, so a
    # multi-GB download reports in periodically instead of flooding the caller.
    if not force and not _should_display_progress(_PROGRESS_STATE):
        return
    _PROGRESS_STATE["line"] = line
    print(line, flush=True)
    logging.info("%s", line)


# ── commands ────────────────────────────────────────────────────────────
def _cmd_start(language: str, no_tts: bool, no_fast_mode: bool) -> int:
    logging.info("_cmd_start: language=%s no_tts=%s no_fast_mode=%s", language, no_tts, no_fast_mode)
    deadline = _download_wait_deadline()
    try:
        _ensure_server(language, no_tts, no_fast_mode)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    outcome, reply = _wait_for_running(deadline)

    if outcome == "timeout":
        _save_pending_request(language, no_tts, no_fast_mode, None)
        if _PROGRESS_STATE.get("progress") is not None:
            _report_download_progress({}, force=True)
        print(
            "模型正在下载, 请用命令 `scripts\\run.ps1 --continue` 继续运行\n"
            "Model download in progress; re-run with --continue.",
            flush=True,
        )
        return 3
    if outcome == "error":
        logging.error("_cmd_start: server reported error: %s", reply.get("error"))
        print("[ERROR] 服务初始化失败 / Server initialization failed:", file=sys.stderr)
        print(reply.get("error", "<unknown init error>"), file=sys.stderr)
        _delete_pending_request()
        return 1

    _delete_pending_request()
    _print_ready(reply)
    return 0


def _cmd_continue(_language: str, _no_tts: bool, _no_fast_mode: bool) -> int:
    pending = _load_pending_request()
    if pending is None:
        print("无待处理请求, 请先使用 `scripts\\run.ps1` 启动服务", file=sys.stderr)
        return 1
    language = pending.get("language", "auto")
    no_tts = bool(pending.get("no_tts", False))
    no_fast_mode = bool(pending.get("no_fast_mode", False))
    return _cmd_start(language, no_tts, no_fast_mode)


def _cmd_status(_args: argparse.Namespace) -> int:
    conn = _try_connect()
    if conn is None:
        print(json.dumps({"state": "stopped"}, ensure_ascii=False, indent=2), flush=True)
        return 2
    reply = _send(conn, {"op": "status"})
    state = reply.get("state")
    if state == "running":
        _print_ready(reply)
        return 0
    print(json.dumps(reply, ensure_ascii=False, indent=2), flush=True)
    return 0 if state == "running" else 2


def _cmd_shutdown_sync(timeout: float) -> int:
    conn = _try_connect()
    if conn is None:
        print("服务未运行 / server not running.", flush=True)
        return 0
    reply = _send(conn, {"op": "shutdown", "timeout": timeout})
    if not reply.get("ok", False):
        print(f"shutdown failed: {reply.get('error', '<unknown>')}", file=sys.stderr)
        return 1
    print(f"正在停止服务 / server shutting down (grace period: {timeout:.1f}s)", flush=True)
    return 0


def _wait_for_server_gone(deadline: float) -> bool:
    """Poll until the old server has really exited.

    A restart must wait for this: the new server binds the same WebUI
    port + pipe, so starting it before the old one releases them races
    the ``ThreadingHTTPServer`` bind in ``WebUI.start``.

    The server closes its pipe listener BEFORE the process exits (model and
    camera teardown take several seconds), so the pipe going quiet is not
    enough.  On the scheduled-task path the process must be gone too,
    otherwise `_adopt_booting_server` would latch onto the dying server and
    wait out the boot timeout for a pipe that is never coming back.
    """
    while time.time() < deadline:
        conn = _try_connect()
        if conn is None:
            if not server_task.pid_alive(_read_task_pid()):
                return True
        else:
            try:
                conn.close()
            except Exception:
                pass
        time.sleep(SERVER_BOOT_POLL_INTERVAL)
    return False


def _discard_task() -> None:
    """Drop the scheduled task once no server is using it.

    Deliberately best-effort and never reflected in the exit code: the server
    is already down, which is what the caller asked for, and failing the
    command over a bookkeeping step would be backwards.  Only touches the task
    scheduler when there is evidence a task was used, so hosts on the direct
    spawn path do not pay for a COM call on every shutdown.
    """
    if os.name != "nt":
        return
    if not (TASK_ENV_FILE.exists() or TASK_PID_FILE.exists() or server_task.should_use_task()):
        return
    server_task.delete(SKILL_NAME, logging.info)
    _clear_task_files()


def _cmd_shutdown(_args: argparse.Namespace) -> int:
    rc = _cmd_shutdown_sync(DEFAULT_SHUTDOWN_TIMEOUT)
    if rc != 0:
        return rc
    gone = _wait_for_server_gone(time.time() + DEFAULT_SHUTDOWN_TIMEOUT + 5.0)
    if gone:
        _discard_task()
        print("已停止, 资源已释放 / stopped; camera + mic + GPU released.", flush=True)
        return 0
    print("shutdown timed out", flush=True)
    return 2


def _cmd_restart(language: str, no_tts: bool, no_fast_mode: bool) -> int:
    """Force a clean restart: shut any running server down, wait for it to
    fully exit (freeing the pipe + WebUI port), then start fresh.

    Unlike the default start — which reuses a live server and so keeps
    running stale code after an upgrade — this guarantees the
    freshly-installed scripts are the ones loaded. Costs a full model
    reload (cold start + camera re-open).
    """
    conn = _try_connect()
    if conn is None:
        print("server not running; starting fresh")
        return _cmd_start(language, no_tts, no_fast_mode)

    reply = _send(conn, {"op": "shutdown", "timeout": DEFAULT_SHUTDOWN_TIMEOUT})
    if not reply.get("ok", False):
        print(f"shutdown failed: {reply.get('error', '<unknown>')}", file=sys.stderr)
        return 1
    print(f"restarting: old server shutting down (grace period: {DEFAULT_SHUTDOWN_TIMEOUT:.1f}s)...")

    gone = _wait_for_server_gone(time.time() + DEFAULT_SHUTDOWN_TIMEOUT + 5.0)
    if not gone:
        print("[ERROR] old server did not exit in time; aborting restart "
              "(try `scripts\\run.ps1 --stop` then start again)", file=sys.stderr)
        return 1
    logging.info("old server gone; starting fresh")
    return _cmd_start(language, no_tts, no_fast_mode)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog="client.py")
    parser.add_argument("--log", default=None)
    parser.add_argument(
        "--language", default="auto", choices=("auto", "zh", "en")
    )
    parser.add_argument("--no-tts", action="store_true")
    parser.add_argument("--no-fast-mode", action="store_true")
    parser.add_argument("--continue", dest="cont", action="store_true")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--server-status", action="store_true")
    parser.add_argument("--server-shutdown", action="store_true")
    args = parser.parse_args(argv)
    _bootstrap(
        f"main() parsed args: language={args.language} cont={args.cont} "
        f"restart={args.restart} server_status={args.server_status} "
        f"server_shutdown={args.server_shutdown}"
    )

    _setup_logging(Path(args.log) if args.log else None)

    if args.server_shutdown:
        _bootstrap("dispatching to _cmd_shutdown")
        return _cmd_shutdown(args)
    if args.server_status:
        _bootstrap("dispatching to _cmd_status")
        return _cmd_status(args)
    if args.restart:
        _bootstrap("dispatching to _cmd_restart")
        return _cmd_restart(args.language, args.no_tts, args.no_fast_mode)
    if args.cont:
        _bootstrap("dispatching to _cmd_continue")
        return _cmd_continue(args.language, args.no_tts, args.no_fast_mode)
    _bootstrap("dispatching to _cmd_start")
    return _cmd_start(args.language, args.no_tts, args.no_fast_mode)


if __name__ == "__main__":
    import traceback as _traceback

    try:
        _bootstrap("client.py: main() entry")
        rc = main()
        _bootstrap(f"client.py: main() returned {rc}")
        raise SystemExit(rc)
    except SystemExit:
        raise
    except BaseException as _exc:  # noqa: BLE001
        _err = "".join(_traceback.format_exception(_exc))
        _bootstrap(f"client.py: main() raised:\n{_err}")
        print(_err, file=sys.stderr, flush=True)
        raise SystemExit(99)
