# Early Learning Assistant (本地启蒙学习助手) — Marvis skill packaging

This directory packages the early-learning assistant as a [Tencent
Marvis](https://docs.marvis.qq.com) skill for Intel AIPC platforms.
The actual model + skill logic lives in the **parent repo** at
`../../src/local_camera_companion_skill/`; this tree is purely
glue: manifests, a PowerShell entry point, and a tiny named-pipe
server/client pair.

## Layout

| Path | Role |
|---|---|
| `SKILL.md` | Marvis routing manifest (frontmatter `name` + `description` + `license`; body = user-facing guide) |
| `info.json` | Runtime config: venv name, Python version, download timeout, ModelScope model list |
| `meta.json` | App-store metadata: display name, icons, use cases |
| `requirements.txt` | Python deps (pinned to upstream real-runtime extras) |
| `scripts/run.ps1` | **Fixed Marvis entry point** — parses args, AIPC check, dispatches to `client.py` |
| `scripts/client.py` | Short-lived controller — spawns the server, polls state, prints the web URL |
| `scripts/server.py` | Long-running pipeline server — downloads → loads → serves named pipe + web UI |
| `scripts/ensure_models.py` | ModelScope downloader (atomic `.partial` → final rename) |
| `camera_companion_marvis/skill_runner.py` | Thin wrapper around `CameraCompanionSkill` with the validated `.bat` defaults baked in |
| `camera_companion_marvis/pipe_server.py` | Windows named-pipe RPC (`\\.\pipe\local-scene-recognition`) |
| `camera_companion_marvis/web_ui.py` | Stdlib `http.server` for the static web UI + `/frame.jpg` + `/status` |
| `web/index.html`, `web/status.js` | Live preview + transcript + reply page |

## Preserved behaviour from the upstream skill

All of the following ship ON by default (mirrors `scripts/run_camera_companion_live_demo.bat`):

- `--fast-mode` (Fast-mode no-COT envelope; sub-3s LLM first sentence).
- `--asr-on-gpu` (audio LM forward on iGPU; ~5-10x faster than the CPU twin).
- Optimum vision-text path (auto-resolved from `MiniCPM-o-4_5-OV/optimum_vision_text/`).
- Voice-prime TTS (consistent speaker across turns).
- Narrative carve-out for depicted-character story questions.
- Layer-A short-clip duration gate + Layer-C assistant-template filter.
- Short-transcript retry on suspiciously short ASR output.
- `<think>` tag stripping + filler-prefix stripping.
- `<|SOA:|>` / `<|UTT:|>` special-token stripping.
- ASR sub-timings (`asr_encoder_ms`, `asr_lm_prefill_ms`, `asr_lm_decode_ms`).

PowerShell args that override defaults:

| Arg | Effect |
|---|---|
| `-Language zh` / `-Language en` / `-Language auto` | Pin response + ASR language (default `auto`) |
| `-NoTts` | Disable spoken replies |
| `-NoFastMode` | Use the slow-path COT JSON envelope (~3x slower, slightly higher quality on edge cases) |

## Lifecycle commands

```
scripts\run.ps1                  # start (auto language)
scripts\run.ps1 --continue       # resume first-run download
scripts\run.ps1 --status         # show state + web URL
scripts\run.ps1 --stop           # stop the server (free camera + GPU)
scripts\run.ps1 --restart        # force-restart
```

## Models

Auto-downloaded from ModelScope on first run:

- `yinquan529/MiniCPM-o-4_5-OV-CameraCompanion` (~22 GB; INT4 LM + INT8
  vision + FP16 audio/TTS).
- `yinquan529/silero-vad-camera-companion` (~800 KB).

Both land at `~/.openvino/models/<dir_name>/`. Each `run.ps1` invocation
is capped at ~10 minutes; if the first run can't fit the whole download
in one window, `client.py` exits 3 and Marvis (or the user) re-runs with
`--continue` until "ready" prints.

## Building a release zip

This skill folder is a **standalone sibling** of the camera-companion
source repo (`local-camera-companion-aipc`), so `build.ps1` pulls the
Python packages it vendors from next door rather than from a parent.

```
cd local-scene-recognition
.\build.ps1
# Produces .\release\skill-local-scene-recognition-<version>-<ts>.zip
```

The build vendors two things:

- The camera-companion Python packages (`local_camera_companion_skill`
  + `local_minicpm_runtime`) from the sibling repo at
  `..\local-camera-companion-aipc`. Override with the
  `CAMERA_COMPANION_REPO` env var.
- Common Marvis scripts (`install-env.ps1`, `git-bash-env.sh`,
  `model_download.py`) from a sibling ai-skills repo.
  Override with the `MARVIS_COMMON_DIR` env var.

## Dev mode (no Marvis required)

```
cd local-scene-recognition\src
.\scripts\run.ps1
```

`run.ps1` will warn that `bin\platform.exe` is missing and skip the
AIPC hardware check. Everything else (download, load, serve) runs.
Open `http://127.0.0.1:8766` in a browser.

## Testing

The skill ships its own test harness under `src/tests/`, modeled on
`local-asr/src/tests/` (structured multi-case runner, prerequisite
checks, clean-up flags, pass/fail summary).

Lightweight control-surface check (no venv, models, or AIPC needed —
runs `--help` and `--doctor` only):

```
cd src
.\tests\test.ps1 -SkipHeavy
```

Full end-to-end harness (requires Windows + models + a usable mic +
camera; exercises start → status → stop, tolerating a first-run
download-in-progress):

```
cd src
.\tests\test.ps1
```

Housekeeping flags:

```
.\tests\test.ps1 -CleanEnv       # remove the venv + runtime state, then exit
.\tests\test.ps1 -CleanModels    # remove downloaded models, then exit
```

Interactive TUI client (talks to a running server over the named pipe):

```
pip install -r tests\requirements-dev.txt
python tests\tui_test.py
```

## Caveats

- **First-run download is large** (~22 GB). Operators should expect the
  `--continue` loop to take 4-8 invocations on a typical connection.
- **GPU memory pressure**: on a 32 GB AIPC the resident process sits at
  ~20 GB steady state. Use `scripts\run.ps1 --stop` to release camera,
  microphone, and GPU resources when finished.
- **TTS voice drift** is reduced by the voice-prime but not eliminated.
  Per-utterance reference audio is a future enhancement.
- **Real-person privacy** rule overrides the narrative carve-out: the
  skill will never invent biographical detail about real people in
  frame.
