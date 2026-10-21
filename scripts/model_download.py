# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for atomically downloading and validating local models.

Provides:
- ModelDownloadTemplate / download_required_model (backward-compatible)
- ModelInfo / load_skill_info / load_model_infos / ensure_models (new API)
- AssetInfo / load_asset_infos / ensure_assets (non-model static assets)
- download progress tracking (get_download_progress / format_download_progress)
"""

from __future__ import annotations

import fnmatch
import json
import os
import shutil
import threading
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, NamedTuple, Sequence


def _configure_modelscope_cache() -> None:
    """Keep ModelScope's transient cache inside the OS temporary directory.

    The final, validated models still live under ``~/.openvino/models``. This
    only moves ModelScope's own download cache, whose cleanup is blocked by
    some restricted hosts when it is under the user profile.
    """
    temp_root = os.environ.get("TEMP") or os.environ.get("TMP")
    if temp_root:
        os.environ.setdefault(
            "MODELSCOPE_CACHE", str(Path(temp_root) / "openvino-modelscope-cache")
        )


_configure_modelscope_cache()


class ModelValidation(NamedTuple):
    ok: bool
    reason: str = ""


@dataclass(frozen=True)
class ModelDownloadTemplate:
    models_root: Path
    required_files: tuple[str, ...]
    snapshot_target_kwarg: str = "local_dir"
    snapshot_kwargs: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "models_root", Path(self.models_root))
        object.__setattr__(self, "required_files", tuple(self.required_files))


@dataclass(frozen=True)
class ModelInfo:
    model_id: str
    dir_name: str
    required_files: tuple[str, ...]
    # Optional allowlist of glob patterns passed to snapshot_download so only the
    # needed files are fetched. Repos that ship the same weights in several
    # frameworks (PyTorch + TF + Flax + Rust + ONNX + CoreML, e.g. BERT /
    # Opus-MT) would otherwise download gigabytes of unused duplicates. ``None``
    # downloads the full snapshot (the default for most models).
    allow_patterns: tuple[str, ...] | None = None


def validate_model_dir(local_dir: Path, required_files: Sequence[str]) -> ModelValidation:
    if not local_dir.is_dir():
        return ModelValidation(ok=False, reason="directory missing")

    missing_files = [
        relative_path
        for relative_path in required_files
        if not (local_dir / relative_path).is_file()
    ]
    if missing_files:
        return ModelValidation(
            ok=False,
            reason=f"missing files: {', '.join(missing_files)}",
        )
    return ModelValidation(ok=True)


def _assert_under_models_root(path: Path, models_root: Path) -> Path:
    resolved = path.resolve()
    root_resolved = models_root.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise RuntimeError(
            f"refusing to touch path outside models root: {resolved}"
        ) from exc
    return resolved


def _remove_tree_safely(path: Path, models_root: Path) -> None:
    resolved = _assert_under_models_root(path, models_root)
    if resolved.exists():
        shutil.rmtree(resolved)


def _backup_invalid_model_dir(local_dir: Path, models_root: Path) -> Path:
    _assert_under_models_root(local_dir, models_root)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    suffix = f"{stamp}-{os.getpid()}"
    backup_dir = local_dir.with_name(f"{local_dir.name}.invalid-{suffix}")
    while backup_dir.exists():
        suffix = f"{suffix}-1"
        backup_dir = local_dir.with_name(f"{local_dir.name}.invalid-{suffix}")
    _assert_under_models_root(backup_dir, models_root)
    os.replace(local_dir, backup_dir)
    return backup_dir


def _emit(logger: Callable[[str], None] | None, message: str) -> None:
    if logger is not None:
        logger(message)


# ---------------------------------------------------------------------------
# Download progress tracking
#
# ``snapshot_download`` offers no progress callback, so progress is sampled by
# a monitor thread that periodically measures the on-disk size of the
# ``.partial`` directory and compares it against the total size reported by the
# ModelScope hub. The snapshot lives in module-level state so a long-lived
# server can return it from its ``status`` RPC while the init thread downloads.
# ---------------------------------------------------------------------------

# Sampling is a cheap directory walk, so it runs twice a second to keep the
# percentage/speed readout fresh for the client polling the server. Sampling is
# deliberately decoupled from *display*: a multi-GB download must not flood the
# user with progress lines, but the lines we do show have to carry a current
# speed/ETA rather than a 5-minute average — so fast samples feed a slow
# display (see PROGRESS_DISPLAY_INTERVAL / should_display_progress).
PROGRESS_SAMPLE_INTERVAL = 0.5

# How often a *user-visible* download progress line may be emitted. The first
# line is shown as soon as a download starts, then at most one line per
# interval, so a long download reports in every 5 minutes instead of streaming
# hundreds of lines into the conversation.
PROGRESS_DISPLAY_INTERVAL = 300.0

_PROGRESS_LOCK = threading.Lock()
_PROGRESS: dict | None = None


def should_display_progress(
    state: dict,
    now: float | None = None,
    interval: float = PROGRESS_DISPLAY_INTERVAL,
) -> bool:
    """Rate-limit user-visible progress lines to one per ``interval`` seconds.

    ``state`` is any caller-owned dict used to remember when a line was last
    shown (kept under the ``"displayed_at"`` key). The first call always returns
    ``True`` so the start of a download is never a silent wait; later calls
    return ``True`` only once ``interval`` seconds have elapsed.
    """
    now = time.time() if now is None else now
    last = state.get("displayed_at")
    if last is None or (now - last) >= interval:
        state["displayed_at"] = now
        return True
    return False


def get_download_progress() -> dict | None:
    """Return a snapshot of the in-flight download progress (or None)."""
    with _PROGRESS_LOCK:
        return dict(_PROGRESS) if _PROGRESS is not None else None


def clear_download_progress() -> None:
    with _PROGRESS_LOCK:
        globals()["_PROGRESS"] = None


def set_download_progress(progress: dict | None) -> None:
    """Publish a progress snapshot (used by skills with a bespoke downloader)."""
    with _PROGRESS_LOCK:
        globals()["_PROGRESS"] = dict(progress) if progress is not None else None


_set_download_progress = set_download_progress


def format_bytes(num_bytes: float | None) -> str:
    if num_bytes is None or num_bytes < 0:
        return "未知"
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{value:.0f} B"
        value /= 1024.0
    return f"{value:.1f} TB"


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "未知"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}秒"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}分{secs:02d}秒"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}小时{minutes:02d}分"


def format_download_progress(progress: Mapping | None, bar_width: int = 20) -> str:
    """Render a progress snapshot as a single human-readable Chinese line."""
    if not progress:
        return "模型下载准备中..."

    model_id = progress.get("model_id") or "模型"
    index = progress.get("model_index")
    total_models = progress.get("model_total")
    percent = progress.get("percent")
    downloaded = progress.get("downloaded_bytes")
    total = progress.get("total_bytes")
    speed = progress.get("speed_bps")
    eta = progress.get("eta_s")

    if isinstance(percent, (int, float)):
        filled = int(round(bar_width * max(0.0, min(100.0, float(percent))) / 100.0))
        bar = "█" * filled + "░" * (bar_width - filled)
        head = f"[{bar}] {float(percent):5.1f}%"
    else:
        head = "[" + "░" * bar_width + "]  --.-%"

    parts = [f"模型下载中 {head}"]
    if total:
        parts.append(f"{format_bytes(downloaded)}/{format_bytes(total)}")
    elif downloaded:
        parts.append(f"已下载 {format_bytes(downloaded)}")
    if speed:
        parts.append(f"{format_bytes(speed)}/s")
    if eta is not None:
        parts.append(f"剩余约 {format_duration(eta)}")
    if index and total_models and total_models > 1:
        parts.append(f"第 {index}/{total_models} 个模型")
    parts.append(str(model_id))
    return " | ".join(parts)


def _dir_size(path: Path) -> int:
    total = 0
    try:
        for root, _dirs, files in os.walk(path):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    # File vanished mid-walk (temp file renamed) — ignore.
                    continue
    except OSError:
        return total
    return total


def remote_model_size(model_id: str, allow_patterns: Sequence[str] | None = None) -> int | None:
    """Best-effort total download size (bytes) for a ModelScope model.

    Returns ``None`` when the hub cannot be queried; callers must treat the
    total as unknown rather than failing the download.
    """
    try:
        from modelscope.hub.api import HubApi

        files = HubApi().get_model_files(model_id, recursive=True)
    except Exception:  # noqa: BLE001 — progress must never break a download
        return None

    total = 0
    for entry in files or []:
        if not isinstance(entry, Mapping):
            continue
        if str(entry.get("Type", "")).lower() == "tree":
            continue
        rel_path = entry.get("Path") or entry.get("path") or ""
        if allow_patterns and not any(
            fnmatch.fnmatch(rel_path, pattern) for pattern in allow_patterns
        ):
            continue
        try:
            total += int(entry.get("Size") or entry.get("size") or 0)
        except (TypeError, ValueError):
            continue
    return total or None


class _ProgressMonitor:
    """Samples the size of a ``.partial`` directory on a background thread."""

    def __init__(
        self,
        model_id: str,
        partial_dir: Path,
        total_bytes: int | None,
        model_index: int = 1,
        model_total: int = 1,
        callback: Callable[[dict], None] | None = None,
        interval: float = PROGRESS_SAMPLE_INTERVAL,
    ) -> None:
        self._model_id = model_id
        self._partial_dir = Path(partial_dir)
        self._total_bytes = total_bytes
        self._model_index = model_index
        self._model_total = model_total
        self._callback = callback
        self._interval = max(0.1, interval)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = time.time()
        self._last_sample: tuple[float, int] | None = None
        self._speed_bps: float | None = None

    def __enter__(self) -> "_ProgressMonitor":
        self.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.stop()

    def start(self) -> None:
        self._publish(self._sample_bytes())
        self._thread = threading.Thread(
            target=self._run, name="model-download-progress", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval * 2)
            self._thread = None

    def _sample_bytes(self) -> int:
        return _dir_size(self._partial_dir)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self._publish(self._sample_bytes())

    def _publish(self, downloaded: int) -> None:
        now = time.time()
        if self._last_sample is not None:
            elapsed = now - self._last_sample[0]
            delta = downloaded - self._last_sample[1]
            if elapsed > 0 and delta >= 0:
                instant = delta / elapsed
                # Exponential moving average keeps the readout steady while
                # still reacting to a stalled or accelerating transfer.
                self._speed_bps = (
                    instant if self._speed_bps is None else 0.7 * self._speed_bps + 0.3 * instant
                )
        self._last_sample = (now, downloaded)

        percent: float | None = None
        eta: float | None = None
        if self._total_bytes:
            percent = min(99.9, 100.0 * downloaded / float(self._total_bytes))
            if self._speed_bps and self._speed_bps > 0:
                eta = max(0.0, (self._total_bytes - downloaded) / self._speed_bps)

        progress = {
            "state": "downloading",
            "model_id": self._model_id,
            "model_index": self._model_index,
            "model_total": self._model_total,
            "downloaded_bytes": downloaded,
            "total_bytes": self._total_bytes,
            "percent": percent,
            "speed_bps": self._speed_bps,
            "eta_s": eta,
            "elapsed_s": now - self._started_at,
            "updated_at": now,
        }
        _set_download_progress(progress)
        if self._callback is not None:
            try:
                self._callback(dict(progress))
            except Exception:  # noqa: BLE001 — a bad reporter must not stop a download
                pass


def download_required_model(
    model_id: str,
    local_dir: Path,
    snapshot_download,
    template: ModelDownloadTemplate,
    logger: Callable[[str], None] | None = None,
    max_attempts: int = 3,
    sleep: Callable[[float], None] = time.sleep,
    progress_callback: Callable[[dict], None] | None = None,
    model_index: int = 1,
    model_total: int = 1,
) -> None:
    partial_dir = local_dir.with_name(f"{local_dir.name}.partial")
    #_remove_tree_safely(partial_dir, template.models_root)

    _emit(logger, f"downloading model {model_id} -> {partial_dir}")
    snapshot_kwargs = dict(template.snapshot_kwargs or {})
    snapshot_kwargs[template.snapshot_target_kwarg] = str(partial_dir)

    total_bytes = remote_model_size(model_id, snapshot_kwargs.get("allow_patterns"))
    _emit(logger, f"expected download size for {model_id}: {format_bytes(total_bytes)}")
    monitor = _ProgressMonitor(
        model_id=model_id,
        partial_dir=partial_dir,
        total_bytes=total_bytes,
        model_index=model_index,
        model_total=model_total,
        callback=progress_callback,
    )

    # A single snapshot_download can fail mid-transfer on a flaky network — most
    # painfully on a large weight file (e.g. the 440MB bert-base-uncased
    # safetensors), where modelscope raises FileDownloadError after exhausting
    # its own per-file retries. Retry the whole call a few times with backoff: a
    # fresh attempt reuses modelscope's on-disk cache (already-fetched files are
    # skipped / resumed), so this completes the download rather than restarting
    # 440MB each time. The partial dir is deliberately NOT wiped between attempts
    # so that resume can take effect. The last exception propagates if every
    # attempt fails, so a genuinely unreachable model still fails init loudly.
    attempts = max(1, max_attempts)
    with monitor:
        for attempt in range(1, attempts + 1):
            try:
                snapshot_download(model_id, **snapshot_kwargs)
                break
            except Exception as exc:  # noqa: BLE001 — surface AND retry any download error
                if attempt >= attempts:
                    _emit(logger, f"download of {model_id} failed after {attempt} attempt(s): {exc}")
                    raise
                backoff = 2.0 * attempt  # 2s, 4s, ... — modest, generous enough for a blip
                _emit(
                    logger,
                    f"download of {model_id} failed (attempt {attempt}/{attempts}): {exc}; "
                    f"retrying in {backoff:.0f}s",
                )
                sleep(backoff)

    partial_validation = validate_model_dir(partial_dir, template.required_files)
    if not partial_validation.ok:
        raise RuntimeError(
            f"downloaded model {model_id} failed validation: {partial_validation.reason}"
        )

    current_validation = validate_model_dir(local_dir, template.required_files)
    if local_dir.exists() and not current_validation.ok:
        backup_dir = _backup_invalid_model_dir(local_dir, template.models_root)
        _emit(logger, f"backed up invalid model dir {local_dir} -> {backup_dir}")
    elif current_validation.ok:
        _remove_tree_safely(partial_dir, template.models_root)
        return

    try:
        os.replace(partial_dir, local_dir)
    except OSError as exc:
        raise RuntimeError(f"failed to install downloaded model {model_id}: {exc}") from exc

    _emit(logger, f"installed downloaded model {model_id} -> {local_dir}")


def load_skill_info(info_json_path: Path) -> dict:
    """Load the full info.json as a dict."""
    return json.loads(info_json_path.read_text(encoding="utf-8"))


def load_model_infos(info_json_path: Path) -> list[ModelInfo]:
    """Parse the 'models' array from info.json into ModelInfo objects."""
    data = load_skill_info(info_json_path)
    models_raw = data.get("models", [])
    return [
        ModelInfo(
            model_id=m["model_id"],
            dir_name=m["dir_name"],
            required_files=tuple(m["required_files"]),
            allow_patterns=tuple(m["allow_patterns"]) if m.get("allow_patterns") else None,
        )
        for m in models_raw
    ]


def ensure_models(
    models: list[ModelInfo],
    models_root: Path,
    logger: Callable[[str], None] | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> None:
    """Validate and download all models that aren't already present.

    While a download is running, progress is published to the module-level
    snapshot (see :func:`get_download_progress`) and, when supplied, pushed to
    ``progress_callback``. The snapshot is cleared once everything is on disk.
    """
    missing = [
        m for m in models
        if not validate_model_dir(models_root / m.dir_name, m.required_files).ok
    ]
    if not missing:
        return

    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "modelscope is required to download models. "
            "Please run 'scripts\\install-env.ps1' first."
        ) from exc

    models_root.mkdir(parents=True, exist_ok=True)

    total_models = len(missing)
    for index, m in enumerate(missing, start=1):
        local_dir = models_root / m.dir_name
        validation = validate_model_dir(local_dir, m.required_files)
        if validation.reason and validation.reason != "directory missing":
            _emit(logger, f"model {m.model_id} is incomplete: {validation.reason}; re-downloading")

        template = ModelDownloadTemplate(
            models_root=models_root,
            required_files=m.required_files,
            snapshot_kwargs=(
                {"allow_patterns": list(m.allow_patterns)} if m.allow_patterns else None
            ),
        )
        try:
            download_required_model(
                model_id=m.model_id,
                local_dir=local_dir,
                snapshot_download=snapshot_download,
                template=template,
                logger=logger,
                progress_callback=progress_callback,
                model_index=index,
                model_total=total_models,
            )
        except Exception:
            clear_download_progress()
            raise

    clear_download_progress()

    failed = [
        f"{m.model_id} ({validate_model_dir(models_root / m.dir_name, m.required_files).reason})"
        for m in missing
        if not validate_model_dir(models_root / m.dir_name, m.required_files).ok
    ]
    if failed:
        raise RuntimeError(f"model download did not complete: {', '.join(failed)}")


# ---------------------------------------------------------------------------
# Static (non-model) assets
#
# Some skills ship reference data that is far too big for the release zip but
# far too small to deserve a ModelScope *model* repo — local-tts's reference
# voices, for example, were 2.7MB of the skill's 2.8MB package. Those are
# declared in ``info.json`` under ``assets`` and fetched from a plain URL on
# first run, landing in ``~/.openvino/assets/<skill>/<dir_name>``.
# ---------------------------------------------------------------------------

# A single small file over HTTPS; unlike a multi-GB model there is nothing to
# resume, so a stalled connection is cheaper to abandon than to wait out.
ASSET_DOWNLOAD_TIMEOUT = 60.0


@dataclass(frozen=True)
class AssetInfo:
    """One downloadable zip archive declared in ``info.json``.

    ``dir_name`` is both the install directory under the assets root and the
    top-level directory the archive is expected to contain: ``voice-ref.zip``
    holds ``ref/...`` and installs as ``<assets_root>/ref``. An archive whose
    files sit at the root (no wrapper directory) is also accepted. **Any other
    top-level name is not** — repacking the archive as ``voice-ref/...`` while
    ``dir_name`` stays ``ref`` fails validation loudly rather than silently
    installing ``ref/voice-ref/...``.
    """

    url: str
    dir_name: str
    required_files: tuple[str, ...]


def load_asset_infos(info_json_path: Path) -> list[AssetInfo]:
    """Parse the optional 'assets' array from info.json into AssetInfo objects."""
    data = load_skill_info(info_json_path)
    return [
        AssetInfo(
            url=a["url"],
            dir_name=a["dir_name"],
            required_files=tuple(a.get("required_files", ())),
        )
        for a in data.get("assets", [])
    ]


def _download_to_file(
    url: str,
    dest: Path,
    timeout: float = ASSET_DOWNLOAD_TIMEOUT,
    urlopen: Callable[..., object] | None = None,
) -> None:
    opener = urlopen or urllib.request.urlopen
    with opener(url, timeout=timeout) as response:  # type: ignore[operator]
        with dest.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def _extract_zip_safely(archive: Path, target_dir: Path) -> None:
    """Extract ``archive`` into ``target_dir``, rejecting paths that escape it."""
    root = target_dir.resolve()
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            try:
                (root / member).resolve().relative_to(root)
            except ValueError as exc:
                raise RuntimeError(
                    f"refusing to extract unsafe path from {archive.name}: {member}"
                ) from exc
        zf.extractall(root)


def _strip_wrapper_dir(staging_dir: Path, dir_name: str) -> Path:
    """Return the directory to install: the wrapper dir if present, else staging."""
    entries = list(staging_dir.iterdir())
    if len(entries) == 1 and entries[0].is_dir() and entries[0].name == dir_name:
        return entries[0]
    return staging_dir


def download_required_asset(
    asset: AssetInfo,
    assets_root: Path,
    logger: Callable[[str], None] | None = None,
    max_attempts: int = 3,
    sleep: Callable[[float], None] = time.sleep,
    urlopen: Callable[..., object] | None = None,
    timeout: float = ASSET_DOWNLOAD_TIMEOUT,
) -> None:
    """Download, extract and atomically install one asset archive.

    The archive is staged beside the install directory and only renamed into
    place once ``required_files`` are all present, so an interrupted run can
    never leave a half-extracted directory that later runs would mistake for a
    complete install.
    """
    assets_root = Path(assets_root)
    assets_root.mkdir(parents=True, exist_ok=True)

    target_dir = assets_root / asset.dir_name
    # The suffix keeps two concurrent invocations of the same skill from
    # stomping on each other's staging area.
    suffix = f"partial-{os.getpid()}"
    archive = assets_root / f"{asset.dir_name}.{suffix}.zip"
    staging = assets_root / f"{asset.dir_name}.{suffix}"

    try:
        attempts = max(1, max_attempts)
        for attempt in range(1, attempts + 1):
            try:
                _emit(logger, f"downloading asset {asset.url} -> {archive}")
                _download_to_file(asset.url, archive, timeout=timeout, urlopen=urlopen)
                break
            except Exception as exc:  # noqa: BLE001 — surface AND retry any download error
                if attempt >= attempts:
                    _emit(
                        logger,
                        f"download of {asset.url} failed after {attempt} attempt(s): {exc}",
                    )
                    raise
                backoff = 2.0 * attempt
                _emit(
                    logger,
                    f"download of {asset.url} failed (attempt {attempt}/{attempts}): {exc}; "
                    f"retrying in {backoff:.0f}s",
                )
                sleep(backoff)

        _remove_tree_safely(staging, assets_root)
        staging.mkdir(parents=True)
        _extract_zip_safely(archive, staging)

        source = _strip_wrapper_dir(staging, asset.dir_name)
        validation = validate_model_dir(source, asset.required_files)
        if not validation.ok:
            raise RuntimeError(
                f"downloaded asset {asset.url} failed validation: {validation.reason}"
            )

        # Anything already at the target is incomplete (ensure_assets only calls
        # us for assets that failed validation), and os.replace cannot overwrite
        # a non-empty directory on Windows.
        _remove_tree_safely(target_dir, assets_root)
        try:
            os.replace(source, target_dir)
        except OSError as exc:
            raise RuntimeError(f"failed to install asset {asset.url}: {exc}") from exc
        _emit(logger, f"installed asset {asset.url} -> {target_dir}")
    finally:
        try:
            archive.unlink()
        except OSError:
            pass
        shutil.rmtree(staging, ignore_errors=True)


def ensure_assets(
    assets: Sequence[AssetInfo],
    assets_root: Path,
    logger: Callable[[str], None] | None = None,
    **kwargs,
) -> None:
    """Download every asset that isn't already installed under ``assets_root``."""
    assets_root = Path(assets_root)
    missing = [
        a for a in assets
        if not validate_model_dir(assets_root / a.dir_name, a.required_files).ok
    ]
    if not missing:
        return

    for asset in missing:
        download_required_asset(asset, assets_root, logger=logger, **kwargs)

    failed = [
        f"{a.url} ({validate_model_dir(assets_root / a.dir_name, a.required_files).reason})"
        for a in missing
        if not validate_model_dir(assets_root / a.dir_name, a.required_files).ok
    ]
    if failed:
        raise RuntimeError(f"asset download did not complete: {', '.join(failed)}")


__all__ = [
    "ASSET_DOWNLOAD_TIMEOUT",
    "AssetInfo",
    "ModelDownloadTemplate",
    "clear_download_progress",
    "format_bytes",
    "format_download_progress",
    "format_duration",
    "get_download_progress",
    "remote_model_size",
    "set_download_progress",
    "should_display_progress",
    "PROGRESS_DISPLAY_INTERVAL",
    "PROGRESS_SAMPLE_INTERVAL",
    "ModelInfo",
    "ModelValidation",
    "download_required_asset",
    "download_required_model",
    "ensure_assets",
    "ensure_models",
    "load_asset_infos",
    "load_model_infos",
    "load_skill_info",
    "validate_model_dir",
]
