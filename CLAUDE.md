# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Product

Screenie AI is a macOS-first (Windows-second) Tauri 2 menu-bar app: hotkey → drag-select region → send the cropped PNG to a vision LLM → render the streamed answer in a floating overlay. The overlay is a non-activating HUD; it must not steal focus from the underlying app or block its keystrokes (other than `Esc` while visible).

Stack: Tauri 2 + Rust backend, React 19 + TypeScript + Vite frontend, plain CSS (no Tailwind despite what older notes may say), `keyring` for OS-native secret storage, providers for OpenAI / Anthropic / Gemini / Ollama (local).

## Commands

Lockfile is `package-lock.json` → use **npm**.

Frontend:
- `npm install` — install JS deps.
- `npm run dev` — Vite only, port `1420` (strict; fails if taken). Tauri's `beforeDevCommand` invokes this automatically.
- `npm run build` — `tsc` (typecheck; `strict`, `noUnusedLocals`, `noUnusedParameters` are on) then `vite build` to `dist/`.

Tauri / Rust (run from repo root unless noted):
- `npm run tauri dev` — full desktop app with HMR.
- `npm run tauri build` — production bundle (DMG / MSI / NSIS).
- `cd src-tauri && cargo check` — fast Rust validation; required after any backend edit.
- `cd src-tauri && cargo test` — Rust unit tests (run a single test with `cargo test <name>`).

There is no JS lint script and no JS test runner configured — frontend correctness comes from `tsc` plus manual QA in `npm run tauri dev`. Overlay behavior in particular cannot be unit-tested; verify in the running app.

Releases: pushing a tag `v*` triggers `.github/workflows/release.yml`, which builds macOS Universal + Windows x64 in parallel and uploads to a draft GitHub Release. Code-signing env vars are only exported when the corresponding repo secrets exist, so unsigned local builds work without changes.

## Architecture

### Three windows, one bundle, dispatched in `src/main.tsx`

The frontend is a single Vite SPA that branches on the URL:

- `?mode=overlay` → `<Overlay />` — the capture/result HUD (window label `overlay`).
- `?mode=chat`    → `<Chat />`    — detached chat panel (label `chat`).
- otherwise        → `<App />`     — settings + onboarding (label `main`).

Overlay and Chat render **without** `React.StrictMode` on purpose — they consume one-shot Rust state on mount (`take_pending_capture`, `take_chat_seed`), and StrictMode's dev double-invoke would eat the payload. Don't wrap them.

Each window is gated by its own Tauri capability file: `src-tauri/capabilities/default.json` (windows: `main`, `chat`) and `overlay.json` (windows: `overlay`). Many commands also enforce `window.label()` via `require_window` at the top of the handler, so the overlay can't invoke settings-only commands.

### Rust entry and command surface

`src-tauri/src/main.rs` is a 4-line shim into `screenieai_lib::run()`. The real entry — `src-tauri/src/lib.rs` — is a single ~2.7k-line file housing the tray, window setup, every `#[tauri::command]` (~45 of them), `AppState`, the capture-flow orchestrator (`trigger_capture_flow`), and the global-shortcut dispatcher.

To add a new command: write the `#[tauri::command]` fn and **also add it to the `tauri::generate_handler!` list** in `run()` near the bottom — the registration is not automatic.

`AppState` is the single source of truth for cross-command coordination (pending capture awaiting overlay pickup, current `CancelFlag` for the in-flight AI stream, hotkey config, `overlay_alive`, `tutorial_mode`, `capture_in_progress` reentry guard, chat-seed handoff). Read its struct comments before touching any flow that crosses commands — most subtle bugs in this codebase are state-coordination bugs.

### AI provider abstraction (`src-tauri/src/ai/`)

Each provider exposes one streaming `stream(req, cancel, on_event)` fn (`openai.rs`, `anthropic.rs`, `gemini.rs`, `ollama.rs`). Shared in `mod.rs`:

- `AskEvent::Chunk { text }` and `AskEvent::Usage { input_tokens, output_tokens }` flow to the frontend through a `tauri::ipc::Channel` opened by `ask_ai`. The renderer accumulates `Chunk`s and uses the final `Usage` for cost display + monthly totals.
- `CancelFlag = Arc<AtomicBool>` is replaced on every new `ask_ai` (which trips its predecessor) and also tripped by `close_overlay`. Stream loops **must** poll it between chunks or closing the overlay won't actually stop the request.
- `cloud_client()` / `local_client()` build `reqwest::Client`s with the right timeouts; cloud uses `rustls-tls`.
- `drain_sse_event` + `sse_data` parse SSE. Always buffer **raw bytes** — multi-byte UTF-8 codepoints (CJK, emoji from OCR/translation answers) routinely straddle chunk boundaries; decoding mid-chunk aborts the stream.
- Cloud providers should call `crate::capture::downscale_for_cloud(image_b64, 1024)` before sending; full-Retina screenshots are 1500–2500 image-tokens otherwise.
- The system prompt is `response_format_instructions(profile)`. It enforces strict KaTeX rules and gates math syntax behind "actually mathematical content" — read the v1/v2/v3 commentary in `ai/mod.rs` before changing it; earlier versions injected math into prose answers.

`AiError` serializes to a plain string, so frontend `invoke().catch(...)` gets a human-readable message instead of a tag union.

### Capture pipeline (`src-tauri/src/capture/`)

Platform-split: `macos.rs` (ScreenCaptureKit, linked in `build.rs` via `framework=ScreenCaptureKit`), `win.rs` (GDI BitBlt). `mod.rs` is shared and exposes `capture_rect`, `crop_png_b64`, `downscale_for_cloud`. Sizes cross the API boundary as **logical** points; platform code converts to device pixels using the monitor scale factor. An effectively all-black macOS capture sets `blank: true`, which the React layer treats as missing Screen Recording permission.

### macOS overlay native bridge — the load-bearing part

`src-tauri/src/macos_window.m` (~2.5k lines of Objective-C, compiled by `cc` in `build.rs`) implements behavior Tauri's portable APIs cannot: `NSWindowStyleMaskNonactivatingPanel`, `acceptsFirstMouse`, `NSWindow.ignoresMouseEvents` toggled per-frame against React-supplied interactive rects, a `CGEventTap` for global Esc consumption (with NSEvent-monitor fallback when Accessibility permission is missing), `NSVisualEffectView` "frost panes" pooled beneath the WKWebView, ScreenCaptureKit "exclude self" full-display capture, and Apple Vision OCR.

**Read `docs/overlay-window-behavior.md` before changing overlay focus, click-through, Esc handling, or Spaces/fullscreen behavior.** It documents which APIs are load-bearing, which fallbacks exist, and why the obvious "simpler" approaches break first-click, fullscreen Spaces, or Esc suppression.

The Rust ↔ ObjC interface is the `extern "C"` block near the top of `lib.rs`. Callbacks back into Rust (`handle_overlay_escape_pressed`, `handle_overlay_capture_drag`, …) emit Tauri events to the `overlay` window for JS to react to. Keep ObjC stateless or pool-managed; let `AppState` own all logical state.

`windows_window.rs` is the equivalent Win32 bridge (layered transparent topmost window, `WM_NCHITTEST → HTTRANSPARENT` outside UI rects, low-level keyboard hook for Esc, `SetWindowDisplayAffinity` for exclude-self capture, DWM Mica for Settings/Chat). The WinRT features in `Cargo.toml` also power on-device OCR via `Windows.Media.Ocr.OcrEngine`.

### Capture flow

`trigger_capture_flow` is the single entry exercised by the capture hotkey, tray click, and `repeat_last_capture`. It: (1) takes the `capture_in_progress` reentry guard, (2) hides the settings window if visible (so the overlay can join the active fullscreen Space), (3) snapshots the previous-frontmost app via `screenie_remember_previous_app` so unhandled keystrokes can later be forwarded back, (4) picks the monitor under the cursor with `pick_cursor_monitor`, (5) builds (or reuses) the overlay window, (6) stores the full-screen PNG in `AppState.pending`, and (7) orders the overlay front (without `NSApp.activate`).

The overlay's React side then `take_pending_capture`s the payload, the user drag-selects, clicks Send → `crop_capture` → `ask_ai` (with a `Channel<AskEvent>`); responses stream back. Pinning a chat opens the detached `chat` window via `open_chat_window`, which hydrates from `take_chat_seed`.

### Secrets (`src-tauri/src/secrets.rs`)

Thin `keyring` wrapper with an **explicit allowlist** (`anthropic_api_key`, `openai_api_key`, `gemini_api_key`) under service `com.screenieai.app`. Anything outside the allowlist returns `SecretError::InvalidName`. Don't bypass the allowlist; add new entries to `ALLOWED_NAMES` if a new provider needs storage.

### History (`src-tauri/src/history.rs`)

Persists to `<app_data>/history/{index.json, <id>.png, <id>.thumb.png}`, capped at 200 entries. Full PNG and 240px thumb are split files so the list view doesn't decode the full bitmap.

### Hotkeys

User config persists to `<app_data>/hotkeys.json`. Defaults: capture `CommandOrControl+Shift+KeyA`, repeat-last-capture `CommandOrControl+Alt+KeyA`, settings `CommandOrControl+Shift+Comma`. The shortcut handler in `run()` reads the live config on every press and dispatches by string-matching the firing shortcut — keep `set_hotkey_config`, the dispatcher branches, and the registration loop in setup all in sync when adding a slot.

## Conventions

- Read-only investigation first; small reviewable diffs; no unrelated refactors. The 2.7k-line `lib.rs` and 5k-line `Overlay.tsx` make sweeping changes unsafe.
- `tauri.conf.json` enables `macOSPrivateApi: true` and the activation policy is set to `Accessory` (no Dock icon unless Settings is shown). Don't `app.activate()` from anywhere outside the settings/onboarding flow.
- UI: minimal black/white, frosted/glassy only where useful (NSVisualEffectView panes under overlay regions, sidebar vibrancy on the chat window). No heavy gradients, no glow, no noisy animation.
- Use Context7 for Tauri/React/Rust/plugin API uncertainty rather than guessing — the surface area is large and version-sensitive.
- Don't add a new dependency without justifying it; the Rust dep list in `Cargo.toml` is intentionally small.

## Specialized review agents

`.claude/agents/` defines per-surface reviewers: `tauri-rust-native-reviewer`, `react-tauri-frontend-reviewer`, `overlay-window-specialist`, `ai-provider-security-reviewer`, `performance-reviewer`, `build-release-reviewer`, `product-ux-polish-reviewer`, `qa-test-planner`, plus a `tauri-lead-coordinator` to orchestrate them. Use the matching reviewer when a change touches its area; for cross-surface changes, route through the coordinator.
