# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Marvis glue for the local-scene-recognition skill.

This package contains the bridge between the Marvis client/server
protocol (named-pipe IPC + run.ps1 PowerShell entry) and the existing
:class:`local_camera_companion_skill.environment_skill.CameraCompanionSkill`
class. It deliberately holds NO model code — everything heavy lives in
the upstream ``local_camera_companion_skill`` and ``local_minicpm_runtime``
packages so the Marvis layer stays a thin, hot-swappable veneer.

Submodules
----------

skill_runner
    Wraps ``CameraCompanionSkill`` with a config that mirrors the
    ``scripts/run_camera_companion_live_demo.bat`` defaults
    (fast_mode=True, asr_on_gpu=True, voice-prime on, narrative
    carve-out, short-transcript retry, sub-timings logging).

pipe_server
    Named-pipe RPC protocol (``\\\\.\\pipe\\local-scene-recognition``).
    Mirrors the request/response shape used by the local-realtime-translator
    reference skill.

web_ui
    Tiny stdlib http.server thread serving the static web/ assets +
    JPEG frames + a status JSON endpoint. WebSocket transcript channel
    runs in the same process via ``websockets``.

The actual long-running server entry-point is ``scripts/server.py``;
this package is its dependency.
"""

__all__ = ["skill_runner", "pipe_server", "web_ui"]
