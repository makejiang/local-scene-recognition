# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Thin wrapper around :class:`CameraCompanionSkill` for the Marvis layer.

The wrapper enforces the **validated** configuration from
``scripts/run_camera_companion_live_demo.bat`` so Marvis users get the
same behaviour PR #8 / PR #9 / PR #10 stabilised:

- ``fast_mode=True``           — drops the COT JSON envelope for faster TTS streaming.
- ``asr_on_gpu`` (audio_offload=False) — LM forward on iGPU for ASR.
- ``--no-vision-text`` opt-in  — Marvis defaults to the optimum vision-text path
                                   (auto-resolved from the bundled subdir).
- ``voice_prime=True`` for TTS — speaker consistency across turns.
- ``response_language``         — passed from the run.ps1 ``-Language`` arg.

The wrapper is intentionally synchronous; the named-pipe server.py
serialises requests so we don't need internal locking.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

# All imports below this line are lazy because importing the
# local_camera_companion_skill package triggers heavy deps. We defer
# them until ``SkillRunner.start`` is called.

logger = logging.getLogger(__name__)

# Camera frames and voice recordings, kept in the skill install dir.
_MEDIA_ROOT = Path(__file__).resolve().parent.parent / "run" / "media"

# Web-UI scenario catalog. ``value`` maps to a ``SceneQuestionType`` member,
# except for three pseudo-scenarios resolved in ``_resolve_scenario``:
# ``auto`` (regex auto-classify, the legacy behaviour), ``custom`` (free-text
# prompt authored in the UI), and ``figurine`` (model/手办 identification,
# resolved to the ``objects`` type). Labels are bilingual for the dropdown.
# Kept as plain strings so this module stays import-light — the enum lives in
# the heavy skill package and is imported lazily only inside the voice loop.
_SCENARIO_CATALOG = (
    {"value": "auto", "label": "自动识别 / Auto"},
    {"value": "general", "label": "通用问答 / General"},
    {"value": "objects", "label": "物品识别 / Objects"},
    {"value": "figurine", "label": "模型/手办 / Figurines"},
    {"value": "emotion", "label": "情绪 / Emotion"},
    {"value": "flashcard", "label": "英语单词卡 / English flashcard"},
    {"value": "custom", "label": "自定义 / Custom"},
)
_SCENARIO_VALUES = frozenset(s["value"] for s in _SCENARIO_CATALOG)


# Ready-made prompts surfaced as one-click chips in the web UI's Custom
# panel. Selecting a chip switches the scenario to ``custom`` and fills the
# free-text box with ``prompt`` (still editable). Each prompt is authored to
# be self-contained and pins Chinese answers, so it works regardless of the
# global response-language toggle. Plain strings — no enum / skill import.
#
# Prompts pin Chinese numerals in their examples (not "15"/"25") and ask for a
# full carrier sentence: the TTS engine garbles Arabic digits and clips very
# short fragments, so spelling numbers out and padding the answer to a whole
# sentence keeps the audio clean.
_CUSTOM_PRESETS = (
    {
        "label": "算术 / Arithmetic",
        "prompt": (
            "用户举着一张算术卡片（加法、减法或乘法）。请看清卡片上的算式并计算"
            "结果，然后用一句完整的中文话说出算式和答案，数字一律用中文，例如"
            "「七加八等于十五」或「六乘四等于二十四」。只说这一句，不要解释。"
            "如果算式看不清，就说「这张卡片看不清」。"
        ),
    },
    # NOTE: the 时间 / Telling time preset was removed — these worksheet clocks
    # have near-equal-length hands the model can't read reliably, even in the
    # slow reasoning path. The per-preset ``reasoning`` plumbing (PR #5) stays in
    # place but is dormant (no preset sets it) for possible future reuse.
    {
        "label": "美国硬币 / U.S. coins",
        "prompt": (
            "用户举着一张或多张美国硬币卡片，请仔细观察后判断面值，不要凭猜测"
            "或默认答案。主要看正面的人物头像（卡片通常显示头像面）：林肯是 "
            "penny 一美分、杰斐逊是 nickel 五美分、罗斯福是 dime 十美分、华盛顿"
            "是 quarter 二十五美分、肯尼迪是 half dollar 五十美分。也可结合背面"
            "图案（林肯纪念堂或盾牌是 penny、蒙蒂塞洛庄园是 nickel、火炬是 dime、"
            "老鹰或各州图案是 quarter）、硬币上的英文字样（ONE CENT、FIVE CENTS、"
            "ONE DIME、QUARTER DOLLAR、HALF DOLLAR）、以及大小和颜色（penny 是"
            "铜红色，dime 最小最薄，quarter 最大）来判断。不要读出硬币上的年份。"
            "请用一句完整的中文话说出每一枚硬币的名称和面值，数字用中文；如果"
            "画面中有多枚硬币，请同时算出它们的总和。只说这一句，不要解释。如果"
            "看不清或无法确定面值，就说「这张卡片看不清」，不要随便猜一个面值。"
        ),
    },
)


DEFAULT_OV_MODEL_DIR = Path.home() / ".openvino" / "models" / "MiniCPM-o-4_5-OV"
DEFAULT_VAD_MODEL_PATH = (
    Path.home() / ".openvino" / "models" / "silero_vad" / "silero_vad.onnx"
)


@dataclass
class SkillRunnerConfig:
    """All knobs the Marvis layer exposes. Mirrors the live-demo .bat
    defaults; PowerShell args in run.ps1 can override a subset."""

    model_dir: Path = DEFAULT_OV_MODEL_DIR
    vad_model_path: Path = DEFAULT_VAD_MODEL_PATH
    device: str = "GPU"
    fallback_device: str = "CPU"
    response_language: str = "auto"   # auto / zh / en
    asr_language: str = "auto"        # auto / zh / en
    fast_mode: bool = True
    asr_on_gpu: bool = True           # → audio_offload=False
    no_tts: bool = False
    no_vision_text: bool = False      # default keeps optimum path
    continuous_voice: bool = True
    show_preview: bool = False        # web UI handles preview, not cv2
    save_diagnostics: bool = True
    extra_env: Dict[str, str] = field(default_factory=dict)


@dataclass
class _SkillState:
    """Mutable handles the runner owns once ``start()`` succeeds."""

    skill: Any = None
    pipeline: Any = None
    preview: Any = None          # CameraPreviewController
    recorder: Any = None         # AudioRecorder
    tts_engine: Any = None       # MiniCPMTTSPlaybackEngine (or None when --no-tts)
    turn_taker: Any = None       # VADTurnTaker
    session_id: Optional[str] = None
    warmup_info: Dict[str, Any] = field(default_factory=dict)
    # Threads
    frame_pump_thread: Optional[threading.Thread] = None
    voice_loop_thread: Optional[threading.Thread] = None
    # Cooperative shutdown signal; threads observe this and exit.
    stop_event: threading.Event = field(default_factory=threading.Event)


@dataclass
class _LiveTurn:
    """Snapshot of the most recent turn for the /status endpoint."""

    transcript: str = ""
    reply_text: str = ""
    scene_description: str = ""
    asr_ms: int = 0
    llm_first_sentence_ms: int = 0
    tts_first_audio_ms: int = 0
    state: str = "idle"  # idle / listening / processing / answering


class SkillRunner:
    """Marvis-side runner: load the skill, hold a session, dispatch calls.

    Lifecycle:

      1. ``start(config)``        - load OV pipeline, build skill,
                                    call ``warmup_asr()``, open a camera
                                    session, return the camera URL.
      2. ``ask_text(question)``   - thin pass-through.
      3. ``ask_audio(audio_path)``- thin pass-through.
      4. ``status()``             - state + last-turn timings.
      5. ``stop()``               - release camera + GPU; idempotent.

    All public methods are thread-safe via the server's request queue
    (the server pipe handler is single-threaded).
    """

    def __init__(self) -> None:
        self._state = _SkillState()
        self._config: Optional[SkillRunnerConfig] = None
        self._last_response_debug: Dict[str, Any] = {}
        # Live UI mirror: the WebUI's /status endpoint reads this; the
        # frame-pump and voice-loop threads write to it.
        self._live = _LiveTurn()
        self._live_lock = threading.Lock()
        # Web-UI controls. Detection starts DISABLED (cleared) so the
        # server boots idle — the VAD loop runs but drops turns until the
        # user presses Start. Scenario selection defaults to "auto"
        # (legacy regex auto-classify).
        self._detection_enabled = threading.Event()
        self._scenario_lock = threading.Lock()
        self._selected_scenario = "auto"
        self._custom_prompt = ""
        # When True, the current custom scenario runs the slow (reasoning)
        # path instead of fast mode (used by the 时间 preset for clocks).
        self._custom_reasoning = False
        # The WebUI hands us a callback once it's running. Without it
        # we still work — the runner just stores frames in memory and
        # the /frame.jpg fallback PNG keeps serving.
        self._on_frame: Optional[Callable[[bytes], None]] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, config: SkillRunnerConfig) -> Dict[str, Any]:
        """Build the skill + open a session. Returns a status dict
        suitable for the pipe protocol::

            {"state": "running", "session_id": ..., "warmup": {...}}

        Raises on any unrecoverable error; caller maps to error state.
        """
        if self._state.skill is not None:
            # Idempotent: already running.
            return self._build_status()

        self._config = config
        self._apply_env(config)

        # Lazy imports — these pull in OpenVINO, transformers, etc.
        from local_minicpm_runtime.minicpm_o45_pipeline import (
            MiniCPMO45Pipeline,
        )
        from local_camera_companion_skill.environment_skill import (
            CameraCompanionSkill,
        )
        from local_camera_companion_skill.config import (
            CameraCompanionConfig,
        )

        if not config.model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {config.model_dir}. "
                f"Run 'scripts\\\\run.ps1 --continue' to download "
                f"the model bundle from ModelScope."
            )

        # Auto-resolve the optimum vision-text subdir (matches
        # scripts/run_camera_companion_live_demo.py:1028-1062 default).
        vt_dir: Optional[Path] = None
        if not config.no_vision_text:
            candidate = config.model_dir / "optimum_vision_text"
            if (candidate / "openvino_language_model.xml").is_file():
                vt_dir = candidate

        logger.info(
            "skill_runner.start: model_dir=%s device=%s asr_on_gpu=%s "
            "fast_mode=%s vt_dir=%s",
            config.model_dir,
            config.device,
            config.asr_on_gpu,
            config.fast_mode,
            vt_dir,
        )

        pipeline = MiniCPMO45Pipeline(
            model_dir=str(config.model_dir),
            device=config.device,
            fallback_device=config.fallback_device,
            enable_tts=not config.no_tts,
            vision_text_model_dir=str(vt_dir) if vt_dir else None,
            vision_text_device="GPU",
        )
        pipeline.load()
        self._state.pipeline = pipeline

        # Camera preview, audio recorder, TTS engine — matches the
        # live-demo factories at run_camera_companion_live_demo.py
        # ``_default_preview_factory`` / ``_default_recorder_factory``
        # / ``_default_tts_engine_factory``. Without these the skill
        # has no camera, no mic, no speaker. The web UI was black
        # because preview was never constructed.
        from local_camera_companion_skill.camera_preview import (
            CameraPreviewController,
        )
        from local_camera_companion_skill.audio_capture import (
            create_audio_recorder,
        )

        media_root = _MEDIA_ROOT
        media_root.mkdir(parents=True, exist_ok=True)
        frames_dir = media_root / "frames"
        audio_dir = media_root / "live_audio"
        frames_dir.mkdir(parents=True, exist_ok=True)
        audio_dir.mkdir(parents=True, exist_ok=True)

        preview = CameraPreviewController(
            output_dir=str(frames_dir), mode="continuous"
        )
        try:
            preview.start(camera_index=0)
        except Exception as exc:
            logger.exception("CameraPreviewController.start raised")
            try:
                preview.stop()
            except Exception:  # noqa: BLE001
                pass
            raise RuntimeError(
                f"Failed to open camera (index 0): {exc}"
            ) from exc
        self._state.preview = preview

        recorder = create_audio_recorder(
            "sounddevice", output_dir=str(audio_dir),
        )
        self._state.recorder = recorder

        tts_engine: Any = None
        if not config.no_tts:
            from local_camera_companion_skill.tts_playback import (
                MiniCPMTTSPlaybackEngine,
            )

            tts_dir = media_root / "live_tts"
            tts_dir.mkdir(parents=True, exist_ok=True)
            # Retain TTS WAVs after playback so per-turn audio can be
            # inspected by the operator. The voice-loop ``on_event``
            # handler moves each turn's WAVs into ``live_tts/turn_NNN/``
            # right after playback finishes, mirroring the existing
            # ``vad_utts/vad_utt_N.wav`` numbering for inputs.
            tts_engine = MiniCPMTTSPlaybackEngine(
                pipeline=pipeline,
                output_dir=tts_dir,
                retain_for_debug=True,
            )
        self._state.tts_engine = tts_engine
        # Per-turn TTS WAV bookkeeping. The voice-loop increments
        # ``_tts_turn_counter`` once per spoken reply and moves WAVs
        # under ``live_tts/turn_<counter>/``.
        self._tts_dir: Optional[Path] = (
            (media_root / "live_tts") if not config.no_tts else None
        )
        self._tts_turn_counter: int = 0

        # Build the skill's CameraCompanionConfig. The interactive
        # section captures the .bat defaults.
        skill_config = CameraCompanionConfig.from_dict(
            {
                "runtime": {
                    "model_dir": str(config.model_dir),
                    "device": config.device,
                    "fallback_device": config.fallback_device,
                },
                "interaction": {
                    "response_language": config.response_language,
                    "asr_language": config.asr_language,
                    "fast_mode": config.fast_mode,
                    "continuous_voice_enabled": config.continuous_voice,
                    "streaming_tts_enabled": True,
                    "llm_token_streaming_enabled": True,
                },
                "tts": {"enabled": not config.no_tts},
                "vad": {
                    "enabled": True,
                    "model_path": str(config.vad_model_path),
                },
            }
        )

        # ASR adapter (Fast-mode + audio_offload knobs flow through).
        from local_camera_companion_skill.asr_adapter import (
            create_asr_adapter,
        )

        asr_adapter = create_asr_adapter(
            "minicpm_audio",
            pipeline=pipeline,
            language=_lang_label(config.asr_language),
            fast_mode=config.fast_mode,
            audio_offload=not config.asr_on_gpu,
        )

        self._state.skill = CameraCompanionSkill(
            config=skill_config,
            pipeline=pipeline,
            preview_controller=preview,
            audio_recorder=recorder,
            asr_adapter=asr_adapter,
            tts_engine=tts_engine,
        )

        # Pre-compile audio encoder + LM CPU twin so turn 1 doesn't
        # eat 60-90 s of cold compile (matches the live-demo banner).
        self._state.warmup_info = self._state.skill.warmup_asr()
        logger.info("warmup: %s", self._state.warmup_info)

        # Open a single long-lived camera session for the web UI.
        start_resp = self._state.skill.start_camera_session()
        self._state.session_id = start_resp.session_id
        logger.info("session started: %s", self._state.session_id)
        self._set_live_state("idle")

        # Start background threads: frame-pump and (when continuous
        # voice is enabled) the VAD-driven turn-taker.
        self._state.stop_event.clear()
        self._start_frame_pump()
        if config.continuous_voice:
            self._start_voice_loop()
        else:
            logger.info(
                "continuous_voice disabled; ASR turns must be driven "
                "via the pipe protocol (ask_text / ask_audio)."
            )

        return self._build_status()

    def stop(self) -> Dict[str, Any]:
        if self._state.skill is None:
            return {"state": "stopped"}
        self._state.stop_event.set()
        # Tear down threads in the reverse order of construction.
        # Each step is best-effort so one failure doesn't block the
        # others.
        tt = self._state.turn_taker
        if tt is not None:
            try:
                tt.stop()
            except Exception:  # noqa: BLE001
                logger.exception("turn_taker.stop raised")
        for label, thread in (
            ("voice_loop", self._state.voice_loop_thread),
            ("frame_pump", self._state.frame_pump_thread),
        ):
            if thread is not None and thread.is_alive():
                thread.join(timeout=2.0)
                if thread.is_alive():
                    logger.warning("%s did not exit in 2s", label)
        try:
            if self._state.session_id is not None:
                self._state.skill.stop_camera_session(self._state.session_id)
        except Exception as exc:  # noqa: BLE001 - best-effort teardown
            logger.warning("stop_camera_session raised: %s", exc)
        if self._state.preview is not None:
            try:
                self._state.preview.stop()
            except Exception:  # noqa: BLE001
                logger.exception("preview.stop raised")
        if self._state.recorder is not None:
            try:
                stop_recording = getattr(self._state.recorder, "stop", None)
                if callable(stop_recording):
                    stop_recording()
            except Exception:  # noqa: BLE001
                logger.exception("recorder.stop raised")
        if self._state.tts_engine is not None:
            try:
                stop_tts = getattr(self._state.tts_engine, "stop", None)
                if callable(stop_tts):
                    stop_tts()
            except Exception:  # noqa: BLE001
                logger.exception("tts_engine.stop raised")
        self._state = _SkillState()
        self._set_live_state("stopped")
        return {"state": "stopped"}

    def set_frame_callback(self, cb: Optional[Callable[[bytes], None]]) -> None:
        """Hand the WebUI's ``update_frame`` to us so the frame-pump
        thread can push JPEG bytes to the live preview pane."""
        self._on_frame = cb

    # ------------------------------------------------------------------
    # Request handlers
    # ------------------------------------------------------------------

    def ask_text(self, question: str) -> Dict[str, Any]:
        """Forward a text question. Returns a serialisable dict shaped
        like the pipe protocol response."""
        self._require_running()
        resp = self._state.skill.ask_about_current_view(
            self._state.session_id,
            question,
        )
        return self._response_to_dict(resp)

    def ask_audio(
        self,
        audio_path: Optional[str] = None,
        record_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Forward an audio question. ``audio_path`` is a pre-recorded
        WAV; ``record_seconds`` triggers a fresh capture via the skill's
        recorder."""
        self._require_running()
        kwargs: Dict[str, Any] = {}
        if audio_path is not None:
            kwargs["audio_path"] = Path(audio_path)
        if record_seconds is not None:
            kwargs["record_seconds"] = float(record_seconds)
        resp = self._state.skill.ask_about_current_view_audio(
            self._state.session_id,
            **kwargs,
        )
        return self._response_to_dict(resp)

    def status(self) -> Dict[str, Any]:
        return self._build_status()

    # ------------------------------------------------------------------
    # Web-UI controls (scenario selection + Start/Stop detection)
    # ------------------------------------------------------------------

    def scenarios(self) -> Dict[str, Any]:
        """List selectable scenarios + the current selection, for the
        web UI dropdown."""
        with self._scenario_lock:
            active = self._selected_scenario
            custom_prompt = self._custom_prompt
        return {
            "scenarios": [dict(s) for s in _SCENARIO_CATALOG],
            "presets": [dict(p) for p in _CUSTOM_PRESETS],
            "active": active,
            "custom_prompt": custom_prompt,
            "detection_active": self._detection_enabled.is_set(),
        }

    def set_scenario(
        self, scenario: str, custom_prompt: str = "", reasoning: bool = False
    ) -> Dict[str, Any]:
        """Select the active scenario (and optional custom prompt) used
        for subsequent voice turns. ``reasoning`` runs custom turns on the
        slow path (used by the time preset). Raises ``ValueError`` on an
        unknown scenario value."""
        value = str(scenario or "auto").strip().lower()
        if value not in _SCENARIO_VALUES:
            raise ValueError(f"unknown scenario: {scenario!r}")
        with self._scenario_lock:
            self._selected_scenario = value
            self._custom_prompt = str(custom_prompt or "")
            self._custom_reasoning = bool(reasoning)
        return self._build_status()

    def start_detection(self) -> Dict[str, Any]:
        """Begin acting on voice turns. The VAD loop keeps running
        either way; this flag gates whether detected turns are
        processed, so Start/Stop take effect immediately."""
        self._detection_enabled.set()
        return self._build_status()

    def stop_detection(self) -> Dict[str, Any]:
        """Pause processing of voice turns (VAD stays armed)."""
        self._detection_enabled.clear()
        return self._build_status()

    def _resolve_scenario(self) -> Tuple[Any, Optional[str], bool]:
        """Map the web-UI scenario selection to ``(question_type,
        custom_rules, reasoning)`` for the skill call. Imports the enum
        lazily so this module stays import-light. ``reasoning`` only ever
        applies to the custom scenario."""
        from local_camera_companion_skill.schemas import SceneQuestionType

        with self._scenario_lock:
            value = self._selected_scenario
            custom_prompt = self._custom_prompt
            reasoning = self._custom_reasoning
        if value == "custom":
            return SceneQuestionType.general, (custom_prompt or None), reasoning
        if value == "auto":
            return SceneQuestionType.general, None, False
        if value == "figurine":
            # No dedicated enum member; object-scanning is the best fit for
            # identifying a model / 手办 held up to the camera.
            return SceneQuestionType.objects, None, False
        try:
            return SceneQuestionType(value), None, False
        except ValueError:
            return SceneQuestionType.general, None, False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_status(self) -> Dict[str, Any]:
        with self._live_lock:
            live = {
                "turn_state": self._live.state,
                "transcript": self._live.transcript,
                "reply_text": self._live.reply_text,
                "scene_description": self._live.scene_description,
                "asr_ms": self._live.asr_ms,
                "llm_first_sentence_ms": self._live.llm_first_sentence_ms,
                "tts_first_audio_ms": self._live.tts_first_audio_ms,
            }
        with self._scenario_lock:
            scenario = self._selected_scenario
            custom_prompt = self._custom_prompt
        return {
            "state": "running" if self._state.skill is not None else "stopped",
            "session_id": self._state.session_id,
            "warmup": dict(self._state.warmup_info),
            "live": live,
            "last_response": dict(self._last_response_debug),
            "detection_active": self._detection_enabled.is_set(),
            "scenario": scenario,
            "custom_prompt": custom_prompt,
        }

    def _set_live_state(self, state: str, **fields: Any) -> None:
        """Mutate the /status mirror under the lock."""
        with self._live_lock:
            self._live.state = state
            for key, value in fields.items():
                if hasattr(self._live, key):
                    setattr(self._live, key, value)

    def _require_running(self) -> None:
        if self._state.skill is None or self._state.session_id is None:
            raise RuntimeError(
                "SkillRunner has not been started yet; call start() first."
            )

    def _apply_env(self, config: SkillRunnerConfig) -> None:
        """Set the env vars that the existing skill reads at import / init.

        Matches the .bat:
            set LOCAL_CAMERA_COMPANION_OV_CONFIG_PROFILE=low_memory
            set LOCAL_CAMERA_COMPANION_MAX_NEW_TOKENS=256
        """
        env_defaults = {
            "LOCAL_CAMERA_COMPANION_OV_CONFIG_PROFILE": "low_memory",
            "LOCAL_CAMERA_COMPANION_MAX_NEW_TOKENS": "256",
            # Always-on TTS synthesis fingerprint (native_tts prints text_ids,
            # condition_std/absmax, and audio-token unique/first16 to stderr).
            # Used to root-cause wrong-phoneme garble; overridable via extra_env.
            "LOCAL_CAMERA_COMPANION_DEBUG_TTS": "1",
        }
        env_defaults.update(config.extra_env)
        for key, val in env_defaults.items():
            os.environ.setdefault(key, val)

    def _response_to_dict(self, resp: Any) -> Dict[str, Any]:
        """Convert a CameraCompanionResponse into a JSON-serialisable
        dict for the pipe protocol. Captures last_response_debug so
        /status surfaces the most recent ASR/LLM/TTS timings."""
        scene = resp.scene
        scene_dict: Optional[Dict[str, Any]] = None
        if scene is not None:
            scene_dict = {
                "description": scene.description,
                "objects": list(scene.objects or []),
                "people_count": scene.people_count,
                "apparent_emotion": scene.apparent_emotion,
                "emotion_confidence": (
                    scene.emotion_confidence.value
                    if scene.emotion_confidence is not None
                    else None
                ),
                "safety_notes": list(scene.safety_notes or []),
            }
        out = {
            "session_id": resp.session_id,
            "question_type": (
                resp.question_type.value
                if resp.question_type is not None
                else None
            ),
            "reply_text": resp.reply_text,
            "scene": scene_dict,
            "should_play_tts": resp.should_play_tts,
            "suggested_interaction": resp.suggested_interaction,
            "debug": dict(resp.debug or {}),
        }
        self._last_response_debug = dict(resp.debug or {})
        # Mirror the response onto the live status so the web UI's
        # /status polling shows the latest turn even when the request
        # came in via the pipe protocol (ask_text / ask_audio).
        debug = resp.debug or {}
        raw_transcript = str(debug.get("transcript") or "").strip()
        if raw_transcript:
            transcript_for_ui = raw_transcript
        elif debug.get("reason"):
            # Surface the safe_response reason so the user sees WHY
            # there is no transcript (asr_failed, empty_transcript,
            # no_frame_available, no_audio_recorder, ...) instead of a
            # generic placeholder.
            transcript_for_ui = f"(no transcript — {debug['reason']})"
        else:
            transcript_for_ui = "(text)"
        self._set_live_state(
            "idle",
            transcript=transcript_for_ui,
            reply_text=resp.reply_text or "",
            scene_description=(
                resp.scene.description if resp.scene else ""
            ) or "",
            asr_ms=int(debug.get("asr_ms", 0) or 0),
            llm_first_sentence_ms=int(
                debug.get("llm_first_sentence_ms", 0) or 0
            ),
            tts_first_audio_ms=int(debug.get("tts_first_audio_ms", 0) or 0),
        )
        return out

    # ------------------------------------------------------------------
    # Background threads: frame pump + voice loop
    # ------------------------------------------------------------------

    def _start_frame_pump(self) -> None:
        """Spawn a thread that reads frames from the camera preview
        and pushes JPEG-encoded bytes to the WebUI's update_frame
        callback. Runs at ~5 FPS to bound loopback bandwidth.

        Without this the web preview pane is permanently black: the
        page polls /frame.jpg but WebUI never receives any bytes."""
        if self._state.preview is None:
            logger.warning("frame pump skipped: no preview controller")
            return
        thread = threading.Thread(
            target=self._frame_pump_loop,
            name="camera-companion-frame-pump",
            daemon=True,
        )
        self._state.frame_pump_thread = thread
        thread.start()
        logger.info("frame pump thread started")

    def _frame_pump_loop(self) -> None:
        preview = self._state.preview
        stop = self._state.stop_event
        # Lazy: only import cv2 here so the module-level import stays
        # heavy-dep-free for the hermetic test suite.
        try:
            import cv2  # type: ignore
        except ImportError:
            logger.exception(
                "frame pump: cv2 unavailable; preview will stay blank"
            )
            return
        last_push = 0.0
        target_interval_s = 0.2  # ~5 FPS
        while not stop.is_set():
            try:
                now = time.monotonic()
                if now - last_push < target_interval_s:
                    time.sleep(0.05)
                    continue
                frame = preview.read_latest_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue
                # CameraPreviewController.read_latest_frame returns RGB
                # (the camera_preview module normalises cv2's native BGR
                # → RGB at the source so PIL.save / model inputs see the
                # correct ordering — see camera_preview.py:_bgr_to_rgb).
                # cv2.imencode expects BGR for JPEGs, so flip back here.
                # Without the flip the web preview shows blue / purple
                # skin tones (R and B channels swapped).
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                ok, buf = cv2.imencode(
                    ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 70]
                )
                if not ok:
                    time.sleep(0.05)
                    continue
                last_push = now
                if self._on_frame is not None:
                    try:
                        self._on_frame(bytes(buf))
                    except Exception:  # noqa: BLE001
                        logger.exception("on_frame callback raised")
            except Exception:  # noqa: BLE001
                logger.exception("frame pump iteration failed")
                time.sleep(0.3)
        logger.info("frame pump thread exited")

    def _start_voice_loop(self) -> None:
        """Spawn the VAD-driven continuous-voice loop. Mirrors the
        live demo's run-loop at
        ``scripts/run_camera_companion_live_demo.py`` (the
        ``_build_continuous_voice_turn_taker`` + ``on_event`` block).
        """
        thread = threading.Thread(
            target=self._voice_loop_run,
            name="camera-companion-voice-loop",
            daemon=True,
        )
        self._state.voice_loop_thread = thread
        thread.start()
        logger.info("voice loop thread started")

    def _voice_loop_run(self) -> None:
        """Body of the voice loop. Builds a VAD turn-taker, runs it
        until ``stop_event`` is set, and wires VAD events to skill
        ASR + scene-QA calls (mirrors live demo on_event)."""
        try:
            from local_camera_companion_skill.silero_vad import (
                SileroVADAdapter,
            )
            from local_camera_companion_skill.continuous_audio_capture import (
                ContinuousAudioCapture,
            )
            from local_camera_companion_skill.vad_turn_taker import (
                VADTurnTaker,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "voice loop: VAD components unavailable; "
                "continuous voice disabled"
            )
            # Surface the error into the web UI's transcript pane so the
            # user sees WHY the pill never moves off "idle". Without
            # this, the voice loop silently dies and the user thinks
            # ASR is broken.
            self._set_live_state(
                "idle",
                transcript=(
                    f"(voice loop disabled — import error: "
                    f"{type(exc).__name__}: {exc})"
                ),
            )
            return

        cfg = self._config
        assert cfg is not None  # set in start()

        media_root = _MEDIA_ROOT
        vad_out_dir = media_root / "vad_utts"
        vad_out_dir.mkdir(parents=True, exist_ok=True)

        try:
            vad = SileroVADAdapter(model_path=str(cfg.vad_model_path))
            capture = ContinuousAudioCapture(sample_rate=16000)
            capture.start()
        except Exception as exc:
            logger.exception("voice loop: VAD/capture init failed")
            self._set_live_state(
                "idle",
                transcript=(
                    f"(voice loop disabled — VAD/capture init failed: "
                    f"{type(exc).__name__}: {exc})"
                ),
            )
            return

        def writer(samples: Any, *, path_hint: str = "vad_utt") -> str:
            import wave
            import numpy as np  # type: ignore

            arr = np.asarray(samples, dtype=np.float32)
            arr_i16 = (arr * 32767.0).clip(-32768, 32767).astype(np.int16)
            wav_path = vad_out_dir / f"{path_hint}.wav"
            with wave.open(str(wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(arr_i16.tobytes())
            return str(wav_path)

        tt = VADTurnTaker(
            vad=vad,
            capture=capture,
            wav_writer=writer,
            threshold=0.5,
            min_speech_duration_ms=250,
            silence_hangover_ms=400,
            post_tts_suppression_ms=200,
            frame_duration_ms=32,
            chunk_samples=512,
        )
        self._state.turn_taker = tt

        def on_event(kind: str, payload: Dict[str, Any]) -> None:
            # User-level pause: while detection is disabled the VAD loop
            # keeps running (mic stays warm) but turns are dropped so the
            # UI Start/Stop buttons take effect immediately.
            if not self._detection_enabled.is_set():
                return
            if kind == "start_of_speech":
                self._set_live_state("listening")
                return
            if kind != "end_of_speech":
                return
            self._set_live_state("processing")
            audio_path = payload.get("audio_path")
            if not audio_path:
                self._set_live_state("idle")
                return
            # Gate VAD off while ASR + LLM + TTS runs.
            try:
                tt.set_vad_enabled(False)
            except Exception:  # noqa: BLE001
                pass
            try:
                logger.info(
                    "voice turn: audio=%s size=%d bytes",
                    audio_path,
                    _safe_filesize(audio_path),
                )
                q_type, custom_rules, reasoning = self._resolve_scenario()
                try:
                    resp = self._state.skill.ask_about_current_view_audio(
                        self._state.session_id,
                        audio_path=str(audio_path),
                        question_type=q_type,
                        custom_rules=custom_rules,
                        reasoning=reasoning,
                    )
                except Exception:
                    logger.exception("ask_about_current_view_audio raised")
                    self._set_live_state("idle")
                    return
                # Log the safe-response reason when the skill couldn't
                # produce a real answer. Without this we can't tell from
                # the web UI whether ASR raised, the transcript was
                # empty, or the camera view wasn't yet available.
                dbg = getattr(resp, "debug", None) or {}
                if dbg.get("reason"):
                    logger.warning(
                        "voice turn safe_response: reason=%s transcript=%r "
                        "asr_ms=%s reply=%r",
                        dbg.get("reason"),
                        dbg.get("transcript"),
                        dbg.get("asr_ms"),
                        getattr(resp, "reply_text", ""),
                    )
                # Convert + mirror onto /live status (also updates
                # self._last_response_debug for /status).
                self._response_to_dict(resp)
                # If the skill produced a real reply with TTS, flag the
                # "answering" state so the pill mirrors playback. Safe
                # responses (reason=="..." in debug, no TTS) stay at
                # "idle" — _response_to_dict already wrote the
                # diagnostic transcript / reply into the live mirror.
                if getattr(resp, "should_play_tts", False):
                    self._set_live_state("answering")
                # Archive this turn's TTS WAVs under
                # live_tts/turn_NNN/ so the operator can replay each
                # reply locally. The engine was constructed with
                # retain_for_debug=True so the WAVs still exist on
                # disk at this point.
                tts_engine = self._state.tts_engine
                tts_dir = self._tts_dir
                if (
                    tts_engine is not None
                    and tts_dir is not None
                    and hasattr(tts_engine, "pop_session_basenames")
                ):
                    try:
                        basenames = tts_engine.pop_session_basenames()
                    except Exception:  # noqa: BLE001
                        basenames = []
                    if basenames:
                        self._tts_turn_counter += 1
                        try:
                            _move_tts_wavs_to_turn_dir(
                                tts_dir,
                                basenames,
                                self._tts_turn_counter,
                            )
                        except Exception:  # noqa: BLE001 - best-effort
                            logger.exception(
                                "voice turn: archiving TTS WAVs failed"
                            )
                # Diagnostic: log the actual transcript + reply text
                # for every successful turn so on-device field reports
                # contain the full input→output trace, not just the
                # safe_response cases. Flashcard turns also surface
                # the model's raw pre-wrap output and any
                # filler-rejection reason so the next field report
                # shows WHY the user heard the UNKNOWN fallback.
                if not dbg.get("reason"):
                    logger.info(
                        "voice turn ok: transcript=%r reply=%r "
                        "question_type=%s "
                        "flashcard_pre_wrap=%r "
                        "flashcard_rejected_filler=%r "
                        "tts_sentences=%r",
                        dbg.get("transcript"),
                        getattr(resp, "reply_text", "")[:200],
                        getattr(resp, "question_type", None),
                        dbg.get("flashcard_pre_wrap"),
                        dbg.get("flashcard_rejected_filler"),
                        (dbg.get("tts", {}) or {}).get(
                            "tts_sentences_enqueued"
                        ),
                    )
            finally:
                # Drain mic buffer that piled up during TTS playback
                # so we don't reprocess speaker leakage as a new
                # utterance.
                cap = getattr(tt, "_capture_handle", None) or capture
                if cap is not None:
                    drain = getattr(cap, "drain", None)
                    if callable(drain):
                        try:
                            drain()
                        except Exception:  # noqa: BLE001
                            pass
                try:
                    tt.note_tts_finished()
                except Exception:  # noqa: BLE001
                    pass
                try:
                    tt.set_vad_enabled(True)
                except Exception:  # noqa: BLE001
                    pass
                self._set_live_state("idle")

        tt.set_event_callback(on_event)
        logger.info("voice loop: turn-taker armed, running until stop")
        try:
            tt.run_until_stopped()
        except Exception:  # noqa: BLE001
            logger.exception("voice loop: turn-taker run_until_stopped raised")
        finally:
            try:
                capture.stop()
            except Exception:  # noqa: BLE001
                pass
            logger.info("voice loop thread exited")


def _safe_filesize(path: Any) -> int:
    """Return ``Path(path).stat().st_size`` or 0 — never raises.
    Used by the voice-turn log so a missing/zero-byte WAV is visible
    in the diagnostic line."""
    try:
        return Path(str(path)).stat().st_size
    except Exception:  # noqa: BLE001 - best-effort logging helper
        return 0


def _move_tts_wavs_to_turn_dir(
    tts_dir: Path, basenames: list, turn_number: int
) -> Path:
    """Move ``basenames`` (and their ``.txt`` sidecars, if any) from
    ``tts_dir`` into ``tts_dir / turn_<turn_number:03d>/``.

    Returns the per-turn directory path. Missing source files are
    skipped silently — best-effort archival; the engine has already
    finished playback, so a missing file means the WAV was cleaned
    up by some other code path. Never raises so a filesystem hiccup
    can't kill the voice loop.
    """
    turn_dir = tts_dir / f"turn_{turn_number:03d}"
    try:
        turn_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return turn_dir
    for basename in basenames or []:
        # WAV first.
        src = tts_dir / basename
        if src.exists() and src.is_file():
            try:
                src.replace(turn_dir / basename)
            except OSError:
                # Best-effort — a locked file (e.g. WMP still reading)
                # leaves it under the flat live_tts/ dir for the next
                # run to sweep up.
                pass
        # Sidecar TXT carries the synthesised text. Pair with the WAV
        # in the same per-turn dir so operators always see both.
        sidecar_name = Path(basename).with_suffix(".txt").name
        sidecar_src = tts_dir / sidecar_name
        if sidecar_src.exists() and sidecar_src.is_file():
            try:
                sidecar_src.replace(turn_dir / sidecar_name)
            except OSError:
                pass
    return turn_dir


def _lang_label(asr_language: str) -> Optional[str]:
    """Map ``auto``/``zh``/``en`` to the MiniCPM hint label used by the ASR
    adapter (mirrors ``run_camera_companion_live_demo.py:_asr_language_label``).

    The skill is Chinese-primary (the flashcard trigger catalog is
    Chinese-only; the SKILL.md ZH trigger list dominates the Marvis
    intent surface). Without a hint MiniCPM-o's audio LM frequently
    auto-translates Chinese speech into English ("Okay, I will
    translate this English."), which both breaks the flashcard
    classifier and corrupts the LLM's downstream prompt. Defaulting
    ``auto`` to ``Chinese`` keeps Chinese speech intact; English
    questions still work because the response-language flow stays
    ``auto``."""
    value = (asr_language or "auto").strip().lower()
    if value == "en":
        return "English"
    # ``zh`` and ``auto`` both produce the Chinese hint.
    return "Chinese"
