# Voice queue controls, dictation quality, agent-card layout — design

Date: 2026-06-10
Status: approved (user approved design in session; "build it")

## Problem

1. Voice dictation accuracy is poor. The pipeline decodes with whisper
   `base.en` (142 MB) using `Greedy { best_of: 1 }` — the cheapest possible
   configuration. VAD/segmentation are fine; the model is the bottleneck.
2. Queued voice tasks cannot be paused, cleared, edited, or removed. The
   dispatcher is strict FIFO with only the kill-word path (`kill_drain`),
   which also aborts the *running* task.
3. Agent-card layout: the input pill, the model/autonomy dropdown buttons,
   and the voice chips share one loosely structured column with negative
   margin/transform hacks. The user wants: a taller composer whose background
   encompasses the dropdown trigger buttons (everything staying in place),
   uniform margins on all four sides, and the queued tasks below the dropdown
   buttons in a separate container with a blurred, tinted background.

## 1. Dictation quality (on-device, no new deps)

- Add `WhisperModel::LargeV3Turbo` — wire name `large-v3-turbo`, file
  `ggml-large-v3-turbo-q5_0.bin` from the existing HuggingFace base URL.
  Exact upstream size 574,041,195 bytes; static bounds 540 MB..610 MB.
  whisper-rs already builds with the `metal` feature on macOS, so decode is
  GPU-accelerated; q5_0 turbo decodes short utterances in well under a second
  on Apple Silicon.
- New default model for fresh configs (`VoiceConfig::default`). Existing
  `voice.json` files keep their saved model. Settings dropdown gains
  "Best · 574 MB". `language("en")` stays (valid on the multilingual model).
- `stt.rs`: `SamplingStrategy::Greedy { best_of: 1 }` →
  `BeamSearch { beam_size: 5, patience: -1.0 }`.
- The download card stops hardcoding "~142 MB" and shows the configured
  model's size (hook keeps `config.model` from `voice_get_status`).
- VAD config, segmenter, hallucination blocklist: unchanged.

## 2. Queue controls: pause / clear / edit / remove

### Dispatcher (`src-tauri/src/voice/dispatcher.rs`)

The queue mutex becomes `Mutex<QueueInner { tasks: VecDeque<VoiceTask>,
paused: bool }>` so pause is checked under the same lock the condvar waits
on (no lost wakeups).

- `set_paused(bool)`: while paused, `pop_blocking` does not pop; the running
  task (if any) finishes normally. Mic keeps listening; new utterances still
  enqueue. Unpause notifies the condvar.
- `clear_queued() -> Vec<VoiceTask>`: drains queued tasks only. Unlike
  `kill_drain`, does NOT bump the kill generation and does NOT abort the
  running task.
- `remove_queued(id) -> Option<VoiceTask>`, `edit_queued(id, text) -> bool`:
  operate on a queued task by id; a task that already popped returns
  None/false and the UI shows "task already started".
- Kill words are unaffected: `kill_drain` still drains everything (paused or
  not) and aborts the running task.

### Commands / events (`voice/mod.rs`, registered in `lib.rs`)

- `voice_queue_pause(paused: bool)`, `voice_queue_clear()`,
  `voice_queue_remove(id)`, `voice_queue_edit(id, text)` — all gated to the
  `quick_tooltip` window; they `ensure_dispatcher` so pause-before-first-use
  sticks.
- New event `voice:queue { paused }`; `voice_get_status` gains
  `queuePaused`. Cleared/removed tasks emit `voice:task { id, state:
  "removed" }` (new chip state, rendered as "removed").

### Frontend

`VoiceTranscriptFeed` becomes a task-queue panel:

- Header row: "Tasks" label, Paused badge when paused, pause/resume button,
  clear-queued button.
- Queued chips: click text → inline edit input (Enter saves via
  `voice_queue_edit`, Esc cancels) and a ✕ remove button. Editing claims the
  tooltip keyboard mode on focus and restores it on blur (same pattern as
  the main input). Running/terminal chips render as today.
- `useVoiceSession` exposes `queuePaused`, `setQueuePaused`, `clearQueue`,
  `removeTask`, `editTask`, listens for `voice:queue`, seeds from
  `voice_get_status`.

## 3. Agent-card layout

- New `.quick-tooltip-agent-composer` container (border-radius 18) wraps the
  input row, the voice level meter, and the model/autonomy dropdown row. The
  frosted background (`rgba(42,45,44,0.46)` tint + native frost pane on
  macOS, backdrop-filter on Windows) and the `SvgInsetBorder` move from the
  input pill to this container. The input row becomes transparent. Nothing
  changes position; the background grows to encompass the dropdown buttons.
- Uniform padding inside the composer (8px on all four sides) and uniform
  form padding; the `translateY(-6px)` / negative-margin hacks on the model
  row are removed.
- The queue panel renders BELOW the composer (below the dropdown buttons) as
  a separate container with its own blurred tinted background: add its class
  to `QUICK_TOOLTIP_FROST_REGION_SELECTOR` (macOS native frost) and to the
  Windows backdrop-filter rule.
- Height plumbing kept in sync (three places): `--quick-tooltip-agent-card-h`
  (css), `QUICK_TOOLTIP_AGENT_CARD_MIN_HEIGHT` (QuickTooltip.tsx),
  `QUICK_TOOLTIP_AGENT_CARD_H` (lib.rs) bump 88 → ~106 to fit the taller
  composer; max height 300 → 340 (css max is driven by measurement; tsx and
  lib.rs constants) so four chips + composer fit.

## Testing

- `cargo test`: new dispatcher tests (pause blocks pop, unpause wakes, clear
  leaves running id, remove/edit hit only queued ids, edit-after-pop fails),
  model spec round-trip for `large-v3-turbo`, default-config test updates.
- `cargo check`, `npm run build` (tsc strict).
- Manual QA in `npm run tauri dev` (overlay behavior is not unit-testable):
  dictation accuracy with the new model, pause/resume/clear/edit flows,
  frost panes behind composer + queue panel, Esc behavior in inline edit.
