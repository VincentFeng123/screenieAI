//! Tauri commands for the capture/recording engine.
//!
//! Registered by module path in `generate_handler!` (the voice-module
//! pattern). Window gating is in-command via label checks — custom commands
//! are not capability-file-gated in this repo.
//!
//! No-silent-capture invariant: every entry point here runs because either
//! the user invoked it from Screenie's UI or a user-initiated agent task
//! called the engine. Nothing in this module is reachable from a timer,
//! startup hook, or background event.

use tauri::WebviewWindow;

use super::engine::{self, CapturePermissionMap};
use super::CaptureError;

fn require_capture_window(window: &WebviewWindow, allowed: &[&str]) -> Result<(), CaptureError> {
    if allowed.contains(&window.label()) {
        Ok(())
    } else {
        Err(CaptureError::Other(
            "command not allowed from this window".into(),
        ))
    }
}

/// The perceive/act permission map. `probe: true` additionally runs a real
/// capture to detect the macOS stale-grant case (CGPreflight cached
/// granted-at-launch but captures come back blank → relaunch required).
#[tauri::command]
pub async fn capture_permission(
    window: WebviewWindow,
    probe: Option<bool>,
) -> Result<CapturePermissionMap, CaptureError> {
    require_capture_window(&window, &["quick_tooltip", "main", "overlay"])?;
    let probe = probe.unwrap_or(false);
    tauri::async_runtime::spawn_blocking(move || engine::permission_map_blocking(probe))
        .await
        .map_err(|e| CaptureError::Other(format!("permission task join: {e}")))
}
