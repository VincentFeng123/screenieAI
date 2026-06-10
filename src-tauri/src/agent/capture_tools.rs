//! Agent-facing capture/recording seam.
//!
//! The agent loop perceives and records through this trait — injected like
//! the grounder — so the loop never touches Tauri state and tests swap in a
//! mock. The concrete implementation (`TauriCaptureEngine` in lib.rs) maps
//! `CaptureScope` to real targets and drives `capture::engine`.
//!
//! Errors are Strings: they become planner feedback in the step result,
//! never run failures (the webLookup convention).

use super::executor::AgentAbortState;
use super::types::CaptureScope;
use crate::capture::engine::{CapturePermissionMap, CapturedFrame, RecordingFile};
#[cfg(test)]
use crate::capture::engine::PermissionState;

#[async_trait::async_trait(?Send)]
pub trait CaptureEngine {
    /// The perceive/act permission map. `probe` additionally runs a real
    /// capture to expose the macOS stale-grant case — use it for the
    /// explicit capturePermission action, not per-frame prechecks (a
    /// captured frame's own `blank` flag is the per-capture ground truth).
    async fn permissions(&self, probe: bool) -> CapturePermissionMap;

    /// One downscaled still of the scope; also persisted to the captures
    /// directory so the user can inspect what the agent saw.
    async fn capture_frame(&self, scope: CaptureScope) -> Result<CapturedFrame, String>;

    /// Fixed-duration mp4 clip. Implementations MUST observe `abort` and
    /// stop/finalize the clip early instead of recording past an abort.
    async fn record_clip(
        &self,
        scope: CaptureScope,
        seconds: u32,
        abort: &AgentAbortState,
    ) -> Result<RecordingFile, String>;

    /// Open-ended session; one at a time (a second start is an error).
    async fn start_recording(&self, scope: CaptureScope) -> Result<(), String>;

    /// End the session this engine started and return the saved clip.
    async fn stop_recording(&self) -> Result<RecordingFile, String>;

    /// Whether a session started through THIS engine is still running.
    fn recording_active(&self) -> bool;
}

/// Default for the test-only loop wrapper: reports unknown permissions and
/// declines every operation with planner-readable feedback.
#[cfg(test)]
pub struct NoopCaptureEngine;

#[cfg(test)]
#[async_trait::async_trait(?Send)]
impl CaptureEngine for NoopCaptureEngine {
    async fn permissions(&self, _probe: bool) -> CapturePermissionMap {
        CapturePermissionMap {
            screen_recording: PermissionState::Undetermined,
            accessibility: PermissionState::Undetermined,
            screen_capture_verified: None,
        }
    }

    async fn capture_frame(&self, _scope: CaptureScope) -> Result<CapturedFrame, String> {
        Err("screen capture is not available in this configuration".into())
    }

    async fn record_clip(
        &self,
        _scope: CaptureScope,
        _seconds: u32,
        _abort: &AgentAbortState,
    ) -> Result<RecordingFile, String> {
        Err("screen recording is not available in this configuration".into())
    }

    async fn start_recording(&self, _scope: CaptureScope) -> Result<(), String> {
        Err("screen recording is not available in this configuration".into())
    }

    async fn stop_recording(&self) -> Result<RecordingFile, String> {
        Err("no recording session is active".into())
    }

    fn recording_active(&self) -> bool {
        false
    }
}
