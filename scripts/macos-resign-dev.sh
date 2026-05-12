#!/usr/bin/env bash
# Re-sign the dev binary with a stable identifier so macOS TCC keeps the
# Screen Recording grant across rebuilds. Run this AFTER cargo/tauri rebuilds
# the binary if you find the system permission prompt re-appears.
#
# Why this is needed:
#   - `tauri dev` runs the raw binary at src-tauri/target/debug/screenieai
#     (NOT a .app bundle).
#   - cargo's default ad-hoc signature gives the binary a random identifier
#     (e.g. `screenieai-5512a881b314db15`) that changes on every rebuild.
#   - macOS TCC keys grants by identifier — random identifier = "new app"
#     every rebuild = re-prompt + re-grant cycle forever.
#
# After running this once:
#   1. Run `tccutil reset ScreenCapture com.screenieai.app` (one-time)
#   2. `npm run tauri dev`
#   3. When macOS prompts for Screen Recording, click Allow
#   4. Grant persists across rebuilds AS LONG AS this script re-runs after
#      each `cargo build`.
#
# The Info.plist is also embedded into the binary by build.rs (see
# src-tauri/build.rs `__TEXT,__info_plist` linker flag), so macOS reads
# NSScreenCaptureUsageDescription / CFBundleIdentifier directly from the
# Mach-O section even when the binary isn't inside a .app bundle.

set -euo pipefail

BIN="src-tauri/target/debug/screenieai"

if [ ! -f "$BIN" ]; then
  echo "error: $BIN not found — build first with 'cargo build --manifest-path src-tauri/Cargo.toml' or 'npm run tauri dev'" >&2
  exit 1
fi

codesign --force --identifier com.screenieai.app --sign - "$BIN"
echo "Re-signed $BIN with identifier com.screenieai.app"
echo "If macOS still re-prompts: run 'tccutil reset ScreenCapture com.screenieai.app', restart dev, grant once."
