# Agent dev-mode permissions (macOS TCC)

macOS keys privacy grants (Screen Recording, Accessibility, Automation) to the
**binary path + code signature**, not just the bundle id. `build.rs` embeds
`Info.plist` into the dev binary's `__TEXT,__info_plist` section so
`npm run tauri dev` exposes the stable bundle id `com.screenieai.app`, but the
ad-hoc signature changes whenever the binary is rebuilt at a **different
path** — e.g. after copying or moving the project folder.

Symptom: System Settings shows Screen Recording enabled, yet every capture
comes back all-black (the TCC "denied" placeholder). The agent's capture
health check detects this at task start and reports it in the Quick Tooltip.

Fix:

1. `tccutil reset ScreenCapture com.screenieai.app`
2. Restart `npm run tauri dev` and approve the Screen Recording prompt.
3. Fully quit and reopen the app (the grant is cached per-process at launch).

If the prompt never appears, add the dev binary
(`src-tauri/target/debug/screenieai`) manually under System Settings >
Privacy & Security > Screen Recording. The same path-keying applies to
Accessibility (`tccutil reset Accessibility com.screenieai.app`) and Safari
Automation (`tccutil reset AppleEvents`).
