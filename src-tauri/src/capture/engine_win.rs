//! Windows capture-engine seam. Parity scaffold only: compiles and reports
//! capabilities honestly so the cross-platform build stays green. The real
//! implementation (Windows Graphics Capture or ffmpeg gdigrab) drops in here
//! behind the same `engine` dispatch functions later.

use super::engine::{CapturePermissionMap, PermissionState};
use super::CaptureError;

pub(crate) fn permission_map() -> CapturePermissionMap {
    // Windows has no TCC equivalent for GDI capture: if the process runs, it
    // can capture. Accessibility likewise needs no per-app grant for the
    // UIA-style reads the agent performs.
    CapturePermissionMap {
        screen_recording: PermissionState::Granted,
        accessibility: PermissionState::Granted,
        screen_capture_verified: None,
    }
}

pub(crate) fn unsupported<T>(what: &str) -> Result<T, CaptureError> {
    Err(CaptureError::Unsupported(format!(
        "{what} is not yet available on Windows"
    )))
}
