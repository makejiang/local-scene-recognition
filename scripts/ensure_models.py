# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Download the model bundles declared in info.json.

Wrapper around either the shared Marvis ``model_download.py`` (copied
by build.ps1 to ``scripts/model_download.py``) or, when running
standalone, the ``modelscope`` snapshot_download API.

Resumable: each ModelScope dir lands at ``<dir>.partial`` and is
atomically renamed to ``<dir>`` only when every ``required_files``
entry is present. Re-running picks up where it left off.

Return codes used by server.py:
    0 — all models present
    3 — partial; re-run with --continue
    1 — hard error
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Keep ModelScope's transient download cache in the OS temp directory. The
# completed model remains under %USERPROFILE%\.openvino\models below.
_temp_root = os.environ.get("TEMP") or os.environ.get("TMP")
if _temp_root:
    os.environ.setdefault("MODELSCOPE_CACHE", str(Path(_temp_root) / "openvino-modelscope-cache"))


OPENVINO_ROOT = Path(os.environ.get("USERPROFILE", str(Path.home()))) / ".openvino"
MODELS_ROOT = OPENVINO_ROOT / "models"

# Per-invocation download budget. ``run.ps1`` caps each ``client.py``
# invocation at ~10 min; leave headroom for the 'loading' phase that
# follows.
DOWNLOAD_BUDGET_SECONDS = 8 * 60.0


def _is_complete(target_dir: Path, required: list) -> bool:
    if not target_dir.is_dir():
        return False
    for rel in required:
        if not (target_dir / rel).is_file():
            return False
    return True


def all_present(info: Dict[str, Any]) -> bool:
    """True iff every model in ``info['models']`` is already complete on
    disk (no download needed). Lets server.py skip the misleading
    'downloading' state when the models are already cached."""
    models = info.get("models") or []
    if not isinstance(models, list) or not models:
        return False
    for spec in models:
        dir_name = str(spec.get("dir_name", "")).strip()
        required = list(spec.get("required_files") or [])
        if not dir_name:
            return False
        if not _is_complete(MODELS_ROOT / dir_name, required):
            return False
    return True


def get_download_progress() -> dict | None:
    from model_download import get_download_progress as _get_download_progress
    return _get_download_progress()


def _download_one(spec: Dict[str, Any]) -> str:
    """Download a single model spec. Returns one of:
        "already_present", "downloaded", "partial", "error".
    Does NOT raise — server.py wants to keep going on transient
    errors so the next ``--continue`` can retry."""
    model_id = str(spec.get("model_id", "")).strip()
    dir_name = str(spec.get("dir_name", "")).strip()
    required = list(spec.get("required_files") or [])
    if not model_id or not dir_name:
        logger.error("malformed model spec: %r", spec)
        return "error"

    target_dir = MODELS_ROOT / dir_name
    if _is_complete(target_dir, required):
        return "already_present"

    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    partial = MODELS_ROOT / f"{dir_name}.partial"
    partial.mkdir(parents=True, exist_ok=True)

    try:
        from modelscope.hub.snapshot_download import (  # type: ignore
            snapshot_download,
        )
    except ImportError:
        logger.error(
            "modelscope is not installed; cannot download %s. "
            "Run install-env.ps1 first.",
            model_id,
        )
        return "error"

    allow_patterns = spec.get("allow_patterns") or None
    logger.info(
        "downloading %s → %s (allow_patterns=%s)",
        model_id, partial, allow_patterns,
    )
    try:
        # snapshot_download honors HTTPS_PROXY / HTTP_PROXY env vars.
        snapshot_download(
            model_id=model_id,
            local_dir=str(partial),
            allow_patterns=allow_patterns,
        )
    except Exception:  # noqa: BLE001 - log + retry next pass
        logger.exception("snapshot_download failed for %s", model_id)
        return "partial"

    if not _is_complete(partial, required):
        logger.warning(
            "download for %s completed but required files missing: %s",
            model_id,
            [r for r in required if not (partial / r).is_file()],
        )
        return "partial"

    # Atomic rename: replace existing target_dir if a stale partial sits there.
    if target_dir.exists():
        try:
            shutil.rmtree(target_dir)
        except Exception:
            logger.exception("failed to remove stale %s", target_dir)
            return "error"
    partial.rename(target_dir)
    logger.info("downloaded %s → %s", model_id, target_dir)
    return "downloaded"


def download_all(info: Dict[str, Any]) -> int:
    """Download every model in ``info['models']``. Returns the exit
    code server.py should propagate to client.py:
        0 — all done
        3 — some still partial; re-run with --continue
        1 — hard error
    """
    models = info.get("models") or []
    if not isinstance(models, list) or not models:
        logger.error("info.json has no models list")
        return 1

    try:
        from model_download import ModelInfo, ensure_models
        infos = [
            ModelInfo(
                model_id=str(spec.get("model_id", "")).strip(),
                dir_name=str(spec.get("dir_name", "")).strip(),
                required_files=tuple(spec.get("required_files") or []),
                allow_patterns=(
                    tuple(spec.get("allow_patterns"))
                    if spec.get("allow_patterns")
                    else None
                ),
            )
            for spec in models
        ]
    except Exception:
        logger.exception("info.json models list is unusable")
        return 1

    try:
        ensure_models(infos, MODELS_ROOT, logger=logger.info)
    except Exception:
        # ``ensure_models`` already retried each model a few times, so this is
        # most likely a flaky network. Report "partial" so the caller keeps
        # looping (server.py) or the user re-runs with --continue, instead of
        # turning a transient blip into a hard failure.
        logger.exception("model download incomplete; retry with --continue")
        return 3
    return 0


if __name__ == "__main__":
    import json
    import sys

    info_path = (
        Path(sys.argv[1]) if len(sys.argv) > 1
        else Path(__file__).resolve().parent.parent / "info.json"
    )
    info_data = json.loads(info_path.read_text(encoding="utf-8"))
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(download_all(info_data))
