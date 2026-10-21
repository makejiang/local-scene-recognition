# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Run server.py as a Windows Scheduled Task's own process.

Why this exists
---------------
Kill-on-close hosts (WorkBuddy / CodeBuddy) run every command inside a job
object that tears down the whole process tree when the turn ends, so a server
started with ``subprocess.Popen`` dies with the turn -- taking the camera
window and any in-flight model download with it.  A Scheduled Task is started
by the Task Scheduler service, so its process belongs to neither the job object
nor the host sandbox and keeps running.

A Scheduled Task cannot carry a custom environment block: the task XML has no
environment element, and the task process inherits the user environment from
the registry rather than from whoever registered it.  ``client.py`` therefore
writes the variables the server needs to a JSON file, and this launcher replays
them into ``os.environ`` before handing control to the server.

The server is run IN-PROCESS via ``runpy``, never through ``os.exec*``: on
Windows ``os.execv`` does not replace the process image, it spawns a new
process and exits the current one.  The Scheduled Task would go straight back
to ``Ready`` while the real server kept running as an orphan, destroying the
"is it still booting or is it gone?" signal the client depends on.

Usage (the task action, assembled by ``client.py``)::

    pythonw.exe task_launcher.py <env.json> <pid file> <server.py> [args...]
"""

from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path


def _replay_env(env_json: Path) -> None:
    """Merge the caller's whitelisted variables into this process."""
    try:
        payload = json.loads(env_json.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # A missing or corrupt file must not stop the server from starting:
        # the variables it carries are configuration, and every one of them has
        # a default on the server side.
        return
    if not isinstance(payload, dict):
        return
    for key, value in payload.items():
        if isinstance(key, str) and isinstance(value, str):
            os.environ[key] = value


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        return 2

    env_json = Path(argv[0])
    pid_file = Path(argv[1])
    server_py = Path(argv[2])
    server_args = list(argv[3:])

    _replay_env(env_json)

    # Publish the pid BEFORE importing the server: the import pulls in
    # OpenVINO/torch and can take tens of seconds, and the client uses the
    # appearance of this file to tell "the task really started" from "the task
    # never ran at all" (blocked by policy, wrong interpreter, ...).
    try:
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(str(os.getpid()), encoding="utf-8")
    except OSError:
        pass

    sys.argv = [str(server_py)] + server_args
    runpy.run_path(str(server_py), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
