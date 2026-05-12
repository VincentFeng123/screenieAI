#!/usr/bin/env bash
# Re-sign the Screenie AI binaries with a stable identifier so macOS TCC keeps
# Screen Recording / Accessibility / Audio grants across rebuilds.
#
# Why this is needed:
#   Without a real Apple Developer ID certificate, every cargo/tauri build
#   gets an ad-hoc signature with a random identifier (e.g.
#   `screenieai-5512a881b314db15`). macOS TCC keys grants by identifier, so
#   each rebuild looks like a "new app" and your grants evaporate. Using
#   `--identifier com.screenieai.app` forces a stable identifier that
#   matches `bundle.identifier` in tauri.conf.json AND the embedded
#   Info.plist's CFBundleIdentifier, so TCC sees one consistent app across
#   rebuilds.
#
# Usage:
#   ./scripts/macos-resign-dev.sh           # signs whatever exists
#   ./scripts/macos-resign-dev.sh --reset   # also clears stale TCC entries
#                                           # (you'll be re-prompted on next launch)
#
# When to run:
#   - Once after every `cargo build` / `npm run tauri dev` first launch.
#   - With --reset if macOS keeps re-prompting despite the script.

set -euo pipefail

DEV_BIN="src-tauri/target/debug/screenieai"
INSTALLED_APP="/Applications/Screenie AI.app"
BUNDLE_ID="com.screenieai.app"

# P-F: flag parsing. Previously `--reset` was only honored as $1; later
# flags were silently ignored. Loop through all args, fail-loud on unknown
# flags so typos surface instead of silently no-op-ing.
reset_tcc=false
print_usage() {
  cat <<EOF
Usage: $0 [--reset] [--help]
  --reset   Also clear stale TCC entries (you'll be re-prompted on next launch).
  --help    Show this help.
EOF
}
while [ $# -gt 0 ]; do
  case "$1" in
    --reset) reset_tcc=true ;;
    --help|-h) print_usage; exit 0 ;;
    *)
      echo "error: unknown flag '$1'" >&2
      print_usage >&2
      exit 2
      ;;
  esac
  shift
done

resigned_anything=false

if [ -f "$DEV_BIN" ]; then
  codesign --force --identifier "$BUNDLE_ID" --sign - "$DEV_BIN"
  echo "Re-signed dev binary: $DEV_BIN"
  resigned_anything=true
fi

if [ -d "$INSTALLED_APP" ]; then
  codesign --force --deep --identifier "$BUNDLE_ID" --sign - "$INSTALLED_APP"
  echo "Re-signed installed app: $INSTALLED_APP"
  resigned_anything=true
fi

if [ "$resigned_anything" = false ]; then
  echo "error: neither dev binary ($DEV_BIN) nor installed app ($INSTALLED_APP) found." >&2
  echo "Build first: 'npm run tauri dev' (creates dev binary) or 'npm run tauri build' (creates installed app)." >&2
  exit 1
fi

if [ "$reset_tcc" = true ]; then
  echo
  echo "Clearing stale TCC entries (you'll be re-prompted on next launch — grant once)..."
  # `All` covers ScreenCapture + Accessibility + PostEvent + Microphone +
  # everything else TCC tracks for this bundle id. Safer than enumerating
  # categories — Apple adds new ones (e.g. Camera, Reminders, Calendar) and
  # forgetting one means another stuck-prompt loop.
  tccutil reset All "$BUNDLE_ID" 2>&1 | tail -3
fi

echo
echo "Done. If macOS still re-prompts:"
echo "  1. Quit Screenie AI completely (Cmd+Q from menu bar)"
echo "  2. Re-run this script with --reset"
echo "  3. Relaunch the app and grant when prompted"
