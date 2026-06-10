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

use super::engine::{
    self, CaptureTarget, CapturePermissionMap, CapturedFrame, FrameOpts, RecordOpts,
    RecordingFile, RecordingHandle,
};
use super::CaptureError;
use crate::AppState;

fn require_capture_window(window: &WebviewWindow, allowed: &[&str]) -> Result<(), CaptureError> {
    if allowed.contains(&window.label()) {
        Ok(())
    } else {
        Err(CaptureError::Other(
            "command not allowed from this window".into(),
        ))
    }
}

/// One still of a display / window / region for perception. Base64 PNG
/// in-memory by default; `opts.persist` also writes it to the captures
/// directory and returns the path.
#[tauri::command]
pub async fn capture_frame(
    app: tauri::AppHandle,
    window: WebviewWindow,
    target: CaptureTarget,
    opts: Option<FrameOpts>,
) -> Result<CapturedFrame, CaptureError> {
    require_capture_window(&window, &["quick_tooltip", "main"])?;
    let opts = opts.unwrap_or_default();
    let persist_dir = if opts.persist {
        Some(crate::app_data_dir(&app).map_err(CaptureError::Other)?)
    } else {
        None
    };
    engine::capture_frame(target, opts, persist_dir).await
}

/// Fixed-duration recording → saved clip. Clamped to the per-format caps
/// (mp4 ≤ maxDurationS, gif ≤ 30s). The macOS screen-sharing indicator is
/// visible for the whole recording.
#[tauri::command]
pub async fn record_clip(
    app: tauri::AppHandle,
    window: WebviewWindow,
    state: tauri::State<'_, AppState>,
    target: CaptureTarget,
    duration_s: f64,
    opts: Option<RecordOpts>,
) -> Result<RecordingFile, CaptureError> {
    require_capture_window(&window, &["quick_tooltip", "main"])?;
    let app_data = crate::app_data_dir(&app).map_err(CaptureError::Other)?;
    engine::record_clip(
        state.recording.clone(),
        app_data,
        target,
        duration_s,
        opts.unwrap_or_default(),
    )
    .await
}

/// Open-ended session recording. One active session at a time; auto-stops
/// at maxDurationS / maxClipBytes.
#[tauri::command]
pub async fn start_recording(
    app: tauri::AppHandle,
    window: WebviewWindow,
    state: tauri::State<'_, AppState>,
    target: CaptureTarget,
    opts: Option<RecordOpts>,
) -> Result<RecordingHandle, CaptureError> {
    require_capture_window(&window, &["quick_tooltip", "main"])?;
    let app_data = crate::app_data_dir(&app).map_err(CaptureError::Other)?;
    engine::start_recording(
        state.recording.clone(),
        app_data,
        target,
        opts.unwrap_or_default(),
    )
    .await
}

/// Stop a session by id and return the finished clip. Returns the stored
/// outcome when the session already auto-stopped at a cap.
#[tauri::command]
pub async fn stop_recording(
    window: WebviewWindow,
    state: tauri::State<'_, AppState>,
    session_id: String,
) -> Result<RecordingFile, CaptureError> {
    require_capture_window(&window, &["quick_tooltip", "main"])?;
    engine::stop_recording(state.recording.clone(), Some(session_id)).await
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
