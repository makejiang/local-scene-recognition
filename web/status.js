// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Early Learning Assistant live web UI client.
//
// Three light responsibilities:
//   1. Poll /frame.jpg every 300 ms and refresh the preview image.
//   2. Poll /status every 1 s for the live transcript / reply / scene.
//   3. (Optional) Open a WebSocket to ws://127.0.0.1:8765 for
//      streaming sentence-level updates if the server provides them.
//
// We deliberately AVOID requiring the WebSocket — the HTTP-polled
// path always works. The WS is just lower-latency when present.

(function () {
  'use strict';

  const IMG = document.getElementById('preview-img');
  const STATUS_PILL = document.getElementById('status-pill');
  const TRANSCRIPT = document.getElementById('transcript');
  const REPLY = document.getElementById('reply');
  const REPLY_META = document.getElementById('reply-meta');
  const WS_STATUS = document.getElementById('ws-status');
  const SCENARIO = document.getElementById('scenario');
  const CUSTOM_PROMPT = document.getElementById('custom-prompt');
  const PRESET_ROW = document.getElementById('preset-row');
  const START_BTN = document.getElementById('start-btn');
  const STOP_BTN = document.getElementById('stop-btn');

  const FRAME_POLL_MS = 300;
  const STATUS_POLL_MS = 1000;
  const WS_URL = 'ws://127.0.0.1:8765';

  let lastTurnId = null;
  let detectionActive = false;
  // Whether the active custom scenario should run the slow reasoning path.
  // Set by a preset chip; cleared when the user changes the dropdown or
  // hand-edits the prompt (those are plain fast-mode custom turns).
  let currentReasoning = false;

  // --- controls: scenario picker + Start/Stop ------------------------
  async function postControl(payload) {
    try {
      const resp = await fetch('/control', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (resp.ok) {
        applyStatus(await resp.json());
      }
    } catch (err) {
      /* transient; the /status poll will re-sync the UI */
    }
  }

  function toggleCustomVisibility() {
    const isCustom = SCENARIO.value === 'custom';
    CUSTOM_PROMPT.classList.toggle('hidden', !isCustom);
  }

  function sendScenario() {
    postControl({
      action: 'set_scenario',
      scenario: SCENARIO.value,
      custom_prompt: CUSTOM_PROMPT.value,
      reasoning: currentReasoning,
    });
  }

  // Clicking a preset chip switches to the Custom scenario and drops its
  // (still-editable) prompt into the textarea, then pushes it to the runner.
  // The user presses Start as usual.
  function applyPreset(prompt, reasoning) {
    SCENARIO.value = 'custom';
    toggleCustomVisibility();
    CUSTOM_PROMPT.value = prompt;
    currentReasoning = !!reasoning;
    sendScenario();
  }

  function renderPresets(presets) {
    PRESET_ROW.innerHTML = '';
    (presets || []).forEach((p) => {
      const chip = document.createElement('button');
      chip.type = 'button';
      chip.className = 'chip';
      chip.textContent = p.label;
      chip.addEventListener('click', () => applyPreset(p.prompt || '', p.reasoning));
      PRESET_ROW.appendChild(chip);
    });
  }

  async function loadScenarios() {
    try {
      const resp = await fetch('/scenarios', { cache: 'no-store' });
      if (!resp.ok) return;
      const data = await resp.json();
      SCENARIO.innerHTML = '';
      (data.scenarios || []).forEach((s) => {
        const opt = document.createElement('option');
        opt.value = s.value;
        opt.textContent = s.label;
        SCENARIO.appendChild(opt);
      });
      renderPresets(data.presets);
      if (data.active) SCENARIO.value = data.active;
      if (typeof data.custom_prompt === 'string') {
        CUSTOM_PROMPT.value = data.custom_prompt;
      }
      toggleCustomVisibility();
    } catch (err) {
      /* /scenarios unavailable — controls stay empty, polling continues */
    }
  }

  let customDebounce = null;
  SCENARIO.addEventListener('change', () => {
    currentReasoning = false;
    toggleCustomVisibility();
    sendScenario();
  });
  CUSTOM_PROMPT.addEventListener('input', () => {
    currentReasoning = false;
    if (customDebounce) clearTimeout(customDebounce);
    customDebounce = setTimeout(sendScenario, 500);
  });
  START_BTN.addEventListener('click', () => postControl({ action: 'start' }));
  STOP_BTN.addEventListener('click', () => postControl({ action: 'stop' }));

  loadScenarios();

  // --- frame polling --------------------------------------------------
  function pollFrame() {
    // Append a cache-buster so the browser doesn't reuse a stale image.
    IMG.src = '/frame.jpg?ts=' + Date.now();
  }
  setInterval(pollFrame, FRAME_POLL_MS);

  // --- status rendering ----------------------------------------------
  function applyStatus(data) {
      const state = (data && data.state) || 'unknown';
      // The runner's voice loop transitions through:
      //   idle → listening → processing → answering → idle
      // Surface the most informative label in the top-right pill.
      const live = (data && data.live) || {};
      const turnState = live.turn_state || '';
      detectionActive = !!(data && data.detection_active);
      // When running but detection is paused, the pill shows "paused"
      // instead of a turn state so Start/Stop status is obvious.
      let pillText = turnState && state === 'running' ? turnState : state;
      if (state === 'running' && !detectionActive) {
        pillText = '已暂停 / paused';
      }
      STATUS_PILL.textContent = pillText;
      STATUS_PILL.className = 'status ' + (
        state === 'running' ? 'running' : (state === 'error' ? 'error' : '')
      );
      // Reflect detection state on the Start/Stop buttons.
      START_BTN.disabled = detectionActive;
      STOP_BTN.disabled = !detectionActive;
      const transcript = live.transcript || '';
      const replyText = live.reply_text || '';
      if (transcript) {
        TRANSCRIPT.textContent = transcript;
        TRANSCRIPT.classList.remove('placeholder');
      }
      if (replyText) {
        REPLY.textContent = replyText;
        REPLY.classList.remove('placeholder');
      }
      const metaParts = [];
      const asrMs = live.asr_ms;
      const llmMs = live.llm_first_sentence_ms;
      const ttsMs = live.tts_first_audio_ms;
      if (typeof asrMs === 'number' && asrMs > 0) metaParts.push('ASR ' + asrMs + ' ms');
      if (typeof llmMs === 'number' && llmMs > 0) metaParts.push('LLM ' + llmMs + ' ms');
      if (typeof ttsMs === 'number' && ttsMs > 0) metaParts.push('TTS ' + ttsMs + ' ms');
      REPLY_META.textContent = metaParts.join(' · ');
  }

  // --- status polling -------------------------------------------------
  async function pollStatus() {
    try {
      const resp = await fetch('/status', { cache: 'no-store' });
      if (!resp.ok) {
        STATUS_PILL.textContent = '错误 / error';
        STATUS_PILL.className = 'status error';
        return;
      }
      applyStatus(await resp.json());
    } catch (err) {
      STATUS_PILL.textContent = '断开 / offline';
      STATUS_PILL.className = 'status error';
    }
  }
  pollStatus();
  setInterval(pollStatus, STATUS_POLL_MS);

  // NOTE: there is no WebSocket server. The previous design had a
  // ws://127.0.0.1:8765 channel for sentence-level streaming, but the
  // server-side WS was never implemented and the JS retry loop
  // surfaced a misleading offline-retry footer to the user. HTTP
  // polling above already covers transcript / reply / scene from the
  // server's /status endpoint (which the runner's continuous-voice
  // thread updates per turn). When low-latency sentence streaming is
  // needed, add a real WS endpoint and re-enable the JS path.
  WS_STATUS.textContent = '';

  // --- keyboard shortcuts ---------------------------------------------
  // The skill's continuous-voice loop is always-on: VAD detects when
  // the user starts speaking and ends a turn on silence. Space is a
  // no-op for now — kept as a placeholder for future push-to-talk.
  window.addEventListener('keydown', (ev) => {
    if (ev.code === 'Space' && !ev.repeat) {
      ev.preventDefault();
    }
  });
})();
