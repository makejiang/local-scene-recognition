# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Start the resident server through a Windows Scheduled Task.

Kill-on-close hosts (WorkBuddy / CodeBuddy) run every command inside a job
object and tear down the whole process tree when the turn ends, so a server
started with ``subprocess.Popen`` does not survive the call.  Registering a
triggerless Scheduled Task and starting it on demand puts the server under the
Task Scheduler service instead, outside both the job object and the host
sandbox.

Hosts that do not do this keep the plain ``Popen`` path -- see
``should_use_task``.  Everything in this module is best-effort: when a task
cannot be registered the caller falls back to ``Popen``, which is no worse than
the behaviour without this module.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# Variables these hosts inject into every command.  Their presence is the
# host-detection signal, and they are stripped from what the server inherits.
HOST_ENV_PREFIXES = ("WORKBUDDY_", "CODEBUDDY_")

# Explicit override, mainly so the task path can be exercised on a normal
# Windows box where no kill-on-close host is present.
SPAWN_MODE_ENV = "LOCAL_SKILL_SPAWN_MODE"

# Only these are carried into the task. A Scheduled Task inherits the user
# environment from the registry rather than the caller's, so "pass everything
# through" is not achievable anyway -- and a full snapshot would write whatever
# credentials the host keeps in its environment to disk.
CONFIG_ENV_PREFIXES = (
    "YOLO_",
    "CAMERA_COMPANION_",
    "RT_",
    "AGENT_TRANSLATOR_",
    "OPENVINO_",
)

TASK_FOLDER = "\\OpenVINO"

# LastTaskResult values that report a task STATE rather than the action's exit
# code: 0x41301 "task is currently running", 0x41303 "task has never run".
# Verified against a live task: a running one reports 267009, and only after
# the action exits does the real code (e.g. 17) appear.
RESULT_RUNNING = 267009
RESULT_NEVER_RAN = 267011
NON_EXIT_RESULTS = (RESULT_RUNNING, RESULT_NEVER_RAN)
TASK_CREATE_OR_UPDATE = 6
TASK_LOGON_INTERACTIVE_TOKEN = 3
TASK_ACTION_EXEC = 0
TASK_INSTANCES_IGNORE_NEW = 2


def task_name(skill: str) -> str:
    return f"{TASK_FOLDER}\\{skill}"


def should_use_task(env: dict | None = None) -> bool:
    """True when the server must outlive the command that starts it."""
    env = os.environ if env is None else env
    mode = (env.get(SPAWN_MODE_ENV) or "").strip().lower()
    if mode == "task":
        return True
    if mode == "direct":
        return False
    if os.name != "nt":
        return False
    return any(key.startswith(HOST_ENV_PREFIXES) for key in env)


def select_env(base_env: dict, overrides: dict) -> dict:
    """Whitelist the variables to carry into the task.

    ``overrides`` (what the caller explicitly sets for this run) is always
    carried; anything inherited is carried only if it matches a config prefix.
    """
    selected = {
        key: value
        for key, value in base_env.items()
        if key.startswith(CONFIG_ENV_PREFIXES) and not key.startswith(HOST_ENV_PREFIXES)
    }
    selected.update(overrides)
    return {str(k): str(v) for k, v in selected.items()}


def pid_alive(pid: int | None) -> bool:
    """Best-effort liveness check for the pid the launcher published.

    Deliberately stdlib-only (ctypes on Windows, signal 0 elsewhere): this runs
    in every skill that uses the task path, and not all of them depend on
    psutil.  A psutil-based check that silently answers "dead" when the package
    is missing would make a perfectly healthy server look like a crash.
    """
    if not pid or pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
            if not handle:
                return False
            try:
                code = ctypes.c_ulong()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return False
                return code.value == STILL_ACTIVE
            finally:
                kernel32.CloseHandle(handle)
        except Exception:  # noqa: BLE001 - conservative: assume gone
            return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by another user
    except OSError:
        return False
    return True


def _service():
    import win32com.client

    service = win32com.client.Dispatch("Schedule.Service")
    service.Connect()
    return service


def _folder(service, create: bool = False):
    try:
        return service.GetFolder(TASK_FOLDER)
    except Exception as exc:
        if not create or getattr(exc, "hresult", None) != -2147024894:
            raise
    return service.GetFolder("\\").CreateFolder(TASK_FOLDER)


def _current_user() -> str:
    domain = os.environ.get("USERDOMAIN", "")
    user = os.environ.get("USERNAME", "")
    if domain and user:
        return f"{domain}\\{user}"
    return user or os.environ.get("USER", "")


def _definition(service, command: str, arguments: str, working_dir: str, description: str,
                userid: str | None = None):
    definition = service.NewTask(0)
    definition.RegistrationInfo.Description = description
    # An empty UserId is meaningful: the service then runs the task as the
    # calling user's interactive token.  See ``_register_task`` for why that is
    # sometimes the only form of principal this machine accepts.
    definition.Principal.UserId = _current_user() if userid is None else userid
    definition.Principal.LogonType = TASK_LOGON_INTERACTIVE_TOKEN
    definition.Principal.RunLevel = 0
    settings = definition.Settings
    settings.MultipleInstances = TASK_INSTANCES_IGNORE_NEW
    settings.DisallowStartIfOnBatteries = False
    settings.StopIfGoingOnBatteries = False
    settings.AllowHardTerminate = True
    settings.StartWhenAvailable = False
    settings.RunOnlyIfNetworkAvailable = False
    settings.IdleSettings.StopOnIdleEnd = False
    settings.IdleSettings.RestartOnIdle = False
    settings.AllowDemandStart = True
    settings.Enabled = True
    settings.Hidden = False
    settings.RunOnlyIfIdle = False
    settings.WakeToRun = False
    settings.ExecutionTimeLimit = "PT0S"
    settings.Priority = 7
    action = definition.Actions.Create(TASK_ACTION_EXEC)
    action.Path = command
    action.Arguments = arguments
    action.WorkingDirectory = working_dir
    return definition


def write_env_file(path: Path, values: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")


def _register_task(service, folder, skill: str, command: str, arguments: str,
                   working_dir: str, description: str, log):
    """Register the task, retrying with an implicit principal when needed.

    Some accounts -- notably a local account with a blank password -- make Task
    Scheduler reject an explicit ``DOMAIN\\user`` interactive-token principal
    with ERROR_LOGON_FAILURE (0x8007052E) or E_ACCESSDENIED (0x80070005),
    because the service tries to validate a credential it was never given.
    Leaving the principal's UserId empty lets the service fall back to the
    calling user's own interactive token: same session, same desktop, still
    outside the host's job object, which is everything this module needs.
    """
    for userid in (_current_user(), ""):
        definition = _definition(service, command, arguments, working_dir, description, userid)
        try:
            task = folder.RegisterTaskDefinition(
                skill, definition, TASK_CREATE_OR_UPDATE, userid, "",
                TASK_LOGON_INTERACTIVE_TOKEN,
            )
        except Exception as exc:
            if not userid:
                raise
            log(f"registration as '{userid}' rejected ({exc}); retrying with an implicit principal")
            continue
        if not userid:
            log("registered scheduled task with an implicit principal (current interactive user)")
        return task
    raise RuntimeError("no principal variant was accepted")


def start(
    skill: str,
    launcher: Path,
    env_file: Path,
    pid_file: Path,
    server_py: Path,
    server_args: list[str],
    working_dir: Path,
    log,
) -> None:
    """Register the task (overwriting any stale definition) and run it.

    Raises RuntimeError on any failure so the caller can fall back to Popen.
    """
    name = task_name(skill)
    # list2cmdline applies the Windows quoting rules the task's Arguments
    # string is parsed with, so paths containing spaces survive.
    arguments = subprocess.list2cmdline(
        [str(launcher), str(env_file), str(pid_file), str(server_py), *server_args]
    )
    try:
        service = _service()
        folder = _folder(service, create=True)
        task = _register_task(
            service, folder, skill, _pythonw_executable(), arguments,
            str(working_dir),
            f"{skill} resident server (started on demand, no trigger)",
            log,
        )
        log(f"registered scheduled task {name}")
        task.Run("")
        log(f"started scheduled task {name}")
    except Exception as exc:
        raise RuntimeError(f"scheduled task start failed: {exc}") from exc


def delete(skill: str, log) -> bool:
    """Remove the task. Best-effort: a failure here must not fail the caller."""
    name = task_name(skill)
    try:
        _folder(_service()).DeleteTask(skill, 0)
    except Exception as exc:
        log(f"scheduled task delete failed ({name}): {exc}")
        return False
    log(f"deleted scheduled task {name}")
    return True


def last_result(skill: str, log) -> int | None:
    """The task action's last exit code, or None when it cannot be determined.

    This runs only on the failure path to decorate an error message.
    """
    name = task_name(skill)
    try:
        return int(_folder(_service()).GetTask(skill).LastTaskResult)
    except Exception as exc:
        log(f"last_result lookup failed ({name}): {exc}")
        return None


def _pythonw_executable() -> str:
    python_exe = Path(sys.executable)
    pythonw_exe = python_exe.with_name("pythonw.exe")
    return str(pythonw_exe) if pythonw_exe.exists() else str(python_exe)
