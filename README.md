# Screenie AI

Screenie AI is a macOS-first menu-bar app for capturing a screen region and asking a vision-capable AI provider (OpenAI, Anthropic, Gemini, or local Ollama) about it. Built with Tauri 2 + Rust + React.

## Quick start

```bash
git clone <repo>
cd screenieAI
npm install
npm run tauri dev
```

The app installs into your menu bar — no Dock icon. Left-click the tray glyph (or hit `⌘⇧A`) to capture; right-click for Settings / Quit.

## Development

| Task | Command |
|---|---|
| Frontend dev server | `npm run dev` |
| Full desktop app with HMR | `npm run tauri dev` |
| Type-check + bundle frontend | `npm run build` |
| Rust validation | `cd src-tauri && cargo check` |
| Rust tests | `cd src-tauri && cargo test` |
| Production bundle | `npm run tauri build` |

After every dev rebuild, run `./scripts/macos-resign-dev.sh` so macOS TCC keeps your Screen Recording grant across binaries.

## Permissions

| OS | Permission | Why | When prompted |
|---|---|---|---|
| macOS | **Screen Recording** | Capture the selected region | First capture |
| macOS | **Accessibility** | Consume Esc while the overlay is up (keeps fullscreen Safari from also exiting) and return focus to the previously-frontmost app after capture | First overlay session (the CGEventTap install fails without it) |
| Windows | — | None; capture works out of the box via GDI / WinRT OCR | n/a |

To re-prompt on macOS:

```bash
./scripts/macos-resign-dev.sh --reset
```

## Provider API keys

Set keys in **Settings → Providers**. Keys are stored in the OS keychain (`com.screenieai.app` service); they never touch disk in plain text. Local Ollama runs at `http://localhost:11434` and needs no key.

## Release

Releases are cut by pushing a tag matching `v*`:

```bash
git tag v0.2.0
git push --tags
```

This triggers `.github/workflows/release.yml`, which builds:

- **macOS Universal** — one DMG, both Apple Silicon and Intel.
- **Windows x64** — NSIS `.exe` (recommended; auto-fetches WebView2) and WiX `.msi` (for `winget` / enterprise).

Both artifacts land in a **draft** GitHub Release. Review the binaries, then click "Publish release" in the GitHub UI to ship.

### Code signing

The workflow exports cert / notarization env vars only when their corresponding repo secrets exist, so unsigned builds work without changes. To enable signing, add these secrets (see `.github/workflows/release.yml` for the exact names):

| Secret | Notes |
|---|---|
| `APPLE_CERTIFICATE` + `APPLE_CERTIFICATE_PASSWORD` | Developer ID Application certificate (base64) and its password |
| `APPLE_SIGNING_IDENTITY` | Common name on the cert (e.g. `Developer ID Application: …`) |
| `APPLE_ID` + `APPLE_PASSWORD` + `APPLE_TEAM_ID` | Apple ID + app-specific password + Team ID, for notarization |
| `WINDOWS_CERTIFICATE` + `WINDOWS_CERTIFICATE_PASSWORD` | Authenticode cert (base64) and password |
| `TAURI_SIGNING_PRIVATE_KEY` + `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` | Tauri updater signing key (only if the updater plugin is enabled — currently it isn't) |

After signing is configured, Gatekeeper / SmartScreen prompts disappear.

### Icons

`src-tauri/icons/` is the bundle icon set. The master `icon.png` is 512×512. To regenerate the full set from a higher-resolution source:

```bash
npx tauri icon path/to/source-1024.png
```

A 1024×1024 master produces sharper Dock + Retina rendering than the current 512×512.

## CI

`.github/workflows/ci.yml` runs `npm ci`, `npm run build`, and `cargo check` on every PR and every push to `main`, on macOS and Windows runners. It does **not** run the full bundler — that's release-only.

## Project structure

```
src/                React frontend (Overlay, Chat, Settings, Onboarding)
src-tauri/          Rust + Tauri 2 backend
  src/lib.rs        Command surface, AppState, capture flow, tray
  src/macos_window.m  Non-activating panel + Esc CGEventTap + vibrancy
  src/windows_window.rs  Win32 equivalent
  src/ai/           Streaming provider clients (OpenAI/Anthropic/Gemini/Ollama)
  src/capture/      Platform-split screen capture (macOS ScreenCaptureKit, Win GDI)
docs/               Design docs (overlay-window-behavior.md is load-bearing)
scripts/            Dev helpers
CLAUDE.md           AI assistant guidance for the codebase
```

## Feedback

Open an issue on GitHub.
