//! Cross-platform capture/recording engine façade.
//!
//! Deliberately NOT a trait: platform selection is compile-time, and the
//! repo's established shape is cfg-dispatched free functions (see
//! [`super::capture_rect`]). The agent layer gets its own injectable trait
//! (`agent::capture_tools::CaptureEngine`) for testability; this module is
//! the concrete machinery underneath.
//!
//! v1 scope guard: no audio (`with_audio: true` is rejected — the seam for
//! v2 is `SCStreamConfiguration.capturesAudio` plus a second
//! `AVAssetWriterInput`), no cloud, no background capture. Capture only runs
//! from an explicit user action or a user-initiated agent task.

#![allow(dead_code)] // consumed incrementally across the capture phases

use serde::{Deserialize, Serialize};

use super::CaptureError;

/// What to capture. Coordinates are GLOBAL logical points — the same space
/// as [`super::capture_rect`]. Window ids are CGWindowIDs (identical to
/// `SCWindow.windowID`).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum CaptureTarget {
    /// A whole display; `None` = the main/first display.
    Display { id: Option<u32> },
    Window { id: u32 },
    Region { x: f64, y: f64, w: f64, h: f64 },
}

fn default_max_dimension() -> u32 {
    // Long-edge cap sized for vision-model token budgets.
    1568
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FrameOpts {
    /// Downscale so the long edge is at most this many pixels.
    #[serde(default = "default_max_dimension")]
    pub max_dimension: u32,
    /// Also write the PNG into the captures directory and return its path.
    #[serde(default)]
    pub persist: bool,
    /// Exclude Screenie's own windows (overlay, tooltip) from the capture.
    /// Display targets only; ignored for window targets.
    #[serde(default = "default_true")]
    pub exclude_self: bool,
    #[serde(default)]
    pub show_cursor: bool,
}

impl Default for FrameOpts {
    fn default() -> Self {
        Self {
            max_dimension: default_max_dimension(),
            persist: false,
            exclude_self: true,
            show_cursor: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ClipFormat {
    #[default]
    Mp4,
    Gif,
}

impl ClipFormat {
    pub fn extension(self) -> &'static str {
        match self {
            ClipFormat::Mp4 => "mp4",
            ClipFormat::Gif => "gif",
        }
    }
}

fn default_max_duration_s() -> u32 {
    300
}

fn default_max_clip_bytes() -> u64 {
    256 * 1024 * 1024
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecordOpts {
    #[serde(default)]
    pub format: ClipFormat,
    /// Frames per second; defaults depend on format (10 mp4 / 8 gif).
    #[serde(default)]
    pub fps: Option<u32>,
    /// v1: must be false; rejected with a clear error otherwise.
    #[serde(default)]
    pub with_audio: bool,
    /// Hard cap; open-ended sessions auto-stop when they reach it.
    #[serde(default = "default_max_duration_s")]
    pub max_duration_s: u32,
    /// Disk budget; the session auto-stops when the file reaches it.
    #[serde(default = "default_max_clip_bytes")]
    pub max_clip_bytes: u64,
    /// Long-edge output cap in pixels; defaults depend on format
    /// (1920 mp4 / 640 gif).
    #[serde(default)]
    pub max_dimension: Option<u32>,
    #[serde(default = "default_true")]
    pub show_cursor: bool,
    /// Exclude Screenie's own windows. Display targets only.
    #[serde(default = "default_true")]
    pub exclude_self: bool,
}

impl Default for RecordOpts {
    fn default() -> Self {
        Self {
            format: ClipFormat::Mp4,
            fps: None,
            with_audio: false,
            max_duration_s: default_max_duration_s(),
            max_clip_bytes: default_max_clip_bytes(),
            max_dimension: None,
            show_cursor: true,
            exclude_self: true,
        }
    }
}

/// GIF encoding is memory- and CPU-bound (per-frame NeuQuant quantization),
/// so it gets much tighter caps than mp4.
pub const GIF_MAX_DURATION_S: u32 = 30;
pub const GIF_MAX_FPS: u32 = 10;
pub const GIF_DEFAULT_FPS: u32 = 8;
pub const GIF_MAX_DIMENSION: u32 = 960;
pub const GIF_DEFAULT_DIMENSION: u32 = 640;
pub const MP4_MAX_FPS: u32 = 30;
pub const MP4_DEFAULT_FPS: u32 = 10;
pub const MP4_DEFAULT_DIMENSION: u32 = 1920;

/// Effective (clamped) recording parameters derived from user opts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EffectiveRecordParams {
    pub fps: u32,
    pub max_dimension: u32,
    pub max_duration_s: u32,
}

/// Clamp requested options to the per-format caps. Pure logic, unit-tested.
pub fn effective_record_params(opts: &RecordOpts) -> EffectiveRecordParams {
    match opts.format {
        ClipFormat::Mp4 => EffectiveRecordParams {
            fps: opts.fps.unwrap_or(MP4_DEFAULT_FPS).clamp(1, MP4_MAX_FPS),
            max_dimension: opts.max_dimension.unwrap_or(MP4_DEFAULT_DIMENSION).max(64),
            max_duration_s: opts.max_duration_s.max(1),
        },
        ClipFormat::Gif => EffectiveRecordParams {
            fps: opts.fps.unwrap_or(GIF_DEFAULT_FPS).clamp(1, GIF_MAX_FPS),
            max_dimension: opts
                .max_dimension
                .unwrap_or(GIF_DEFAULT_DIMENSION)
                .clamp(64, GIF_MAX_DIMENSION),
            max_duration_s: opts.max_duration_s.clamp(1, GIF_MAX_DURATION_S),
        },
    }
}

/// Reject audio up front: v1 records video only.
pub fn validate_record_opts(opts: &RecordOpts) -> Result<(), CaptureError> {
    if opts.with_audio {
        return Err(CaptureError::Unsupported(
            "audio capture is not available in v1 — call again with withAudio: false".into(),
        ));
    }
    Ok(())
}

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct CapturedFrame {
    pub png_base64: String,
    pub width: u32,
    pub height: u32,
    /// Set when `persist` was requested.
    pub path: Option<String>,
    /// True when the frame is the macOS TCC all-black placeholder — the
    /// ground truth for a broken Screen Recording grant (CGPreflight caches
    /// per-process at launch and can stay stale-true).
    pub blank: bool,
}

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct RecordingHandle {
    pub session_id: String,
    pub started_at_ms: u64,
    pub format: ClipFormat,
    /// Final destination path the clip will have after a successful stop.
    pub path: String,
}

#[derive(Debug, Serialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct RecordingFile {
    pub path: String,
    pub bytes: u64,
    pub duration_s: f64,
    pub width: u32,
    pub height: u32,
    pub fps: u32,
    pub format: ClipFormat,
    pub frame_count: u64,
    /// None = explicit stop; otherwise "maxDuration" | "maxBytes" |
    /// "sourceLost" | an error description from the native layer.
    pub stopped_reason: Option<String>,
}

#[derive(Debug, Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum PermissionState {
    Granted,
    Denied,
    /// Reserved: macOS exposes no public non-prompting tri-state for Screen
    /// Recording/Accessibility, so the macOS implementation only ever
    /// reports granted/denied. Stubs and future platforms may use this.
    Undetermined,
}

/// The perceive/act permission map the agent reads before trying to capture.
/// Accessibility is READ here (the action layer owns requesting it); this
/// module never duplicates that flow.
#[derive(Debug, Serialize, Clone, Copy)]
#[serde(rename_all = "camelCase")]
pub struct CapturePermissionMap {
    pub screen_recording: PermissionState,
    pub accessibility: PermissionState,
    /// Result of a live capture probe (`Some(false)` = preflight says
    /// granted but captures come back blank: the stale-grant case that needs
    /// an app relaunch). `None` when not probed.
    pub screen_capture_verified: Option<bool>,
}

/// One still of a display / window / region, downscaled to
/// `opts.max_dimension` on the long edge. Returns base64 PNG in-memory (the
/// agent usually wants pixels, not a file); `persist_dir` additionally
/// writes it under `<dir>/captures/` and returns the path.
pub async fn capture_frame(
    target: CaptureTarget,
    opts: FrameOpts,
    persist_dir: Option<std::path::PathBuf>,
) -> Result<CapturedFrame, CaptureError> {
    #[cfg(target_os = "macos")]
    {
        super::engine_macos::capture_frame(target, opts, persist_dir).await
    }
    #[cfg(target_os = "windows")]
    {
        let _ = (target, opts, persist_dir);
        super::engine_win::unsupported("screen frame capture")
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        let _ = (target, opts, persist_dir);
        Err(CaptureError::Unsupported(
            "screen frame capture is not available on this platform".into(),
        ))
    }
}

/// Build the permission map. Blocking (FFI + optional real capture probe) —
/// call from `spawn_blocking`.
pub fn permission_map_blocking(probe: bool) -> CapturePermissionMap {
    #[cfg(target_os = "macos")]
    {
        let screen_recording = if super::engine_macos::screen_capture_access_granted() {
            PermissionState::Granted
        } else {
            PermissionState::Denied
        };
        let accessibility =
            if unsafe { super::engine_macos::screenie_has_accessibility_access() } {
                PermissionState::Granted
            } else {
                PermissionState::Denied
            };
        // Only a real capture reveals a stale grant, and probing while denied
        // would just confirm the denial at the cost of a full-display grab.
        let screen_capture_verified = if probe && screen_recording == PermissionState::Granted {
            Some(crate::agent_capture_health().is_ok())
        } else {
            None
        };
        CapturePermissionMap {
            screen_recording,
            accessibility,
            screen_capture_verified,
        }
    }
    #[cfg(target_os = "windows")]
    {
        let _ = probe;
        super::engine_win::permission_map()
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        let _ = probe;
        CapturePermissionMap {
            screen_recording: PermissionState::Undetermined,
            accessibility: PermissionState::Undetermined,
            screen_capture_verified: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_target_deserializes_all_kinds() {
        let display: CaptureTarget = serde_json::from_str(r#"{"kind":"display"}"#).unwrap();
        assert_eq!(display, CaptureTarget::Display { id: None });
        let display: CaptureTarget =
            serde_json::from_str(r#"{"kind":"display","id":7}"#).unwrap();
        assert_eq!(display, CaptureTarget::Display { id: Some(7) });
        let window: CaptureTarget =
            serde_json::from_str(r#"{"kind":"window","id":123}"#).unwrap();
        assert_eq!(window, CaptureTarget::Window { id: 123 });
        let region: CaptureTarget =
            serde_json::from_str(r#"{"kind":"region","x":10.0,"y":20.0,"w":800.0,"h":600.0}"#)
                .unwrap();
        assert_eq!(
            region,
            CaptureTarget::Region {
                x: 10.0,
                y: 20.0,
                w: 800.0,
                h: 600.0
            }
        );
    }

    #[test]
    fn frame_opts_defaults_apply() {
        let opts: FrameOpts = serde_json::from_str("{}").unwrap();
        assert_eq!(opts.max_dimension, 1568);
        assert!(!opts.persist);
        assert!(opts.exclude_self);
        assert!(!opts.show_cursor);
    }

    #[test]
    fn record_opts_defaults_and_camel_case() {
        let opts: RecordOpts = serde_json::from_str("{}").unwrap();
        assert_eq!(opts.format, ClipFormat::Mp4);
        assert_eq!(opts.max_duration_s, 300);
        assert_eq!(opts.max_clip_bytes, 256 * 1024 * 1024);
        assert!(!opts.with_audio);

        let opts: RecordOpts = serde_json::from_str(
            r#"{"format":"gif","fps":15,"withAudio":false,"maxDurationS":60,"maxDimension":720}"#,
        )
        .unwrap();
        assert_eq!(opts.format, ClipFormat::Gif);
        assert_eq!(opts.fps, Some(15));
        assert_eq!(opts.max_duration_s, 60);
        assert_eq!(opts.max_dimension, Some(720));
    }

    #[test]
    fn with_audio_is_rejected() {
        let opts: RecordOpts = serde_json::from_str(r#"{"withAudio":true}"#).unwrap();
        let err = validate_record_opts(&opts).unwrap_err();
        assert!(err.to_string().contains("audio"), "got: {err}");
    }

    #[test]
    fn gif_params_are_clamped_mp4_params_are_loose() {
        let gif = RecordOpts {
            format: ClipFormat::Gif,
            fps: Some(30),
            max_duration_s: 300,
            max_dimension: Some(4000),
            ..RecordOpts::default()
        };
        let params = effective_record_params(&gif);
        assert_eq!(params.fps, GIF_MAX_FPS);
        assert_eq!(params.max_duration_s, GIF_MAX_DURATION_S);
        assert_eq!(params.max_dimension, GIF_MAX_DIMENSION);

        let gif_defaults = RecordOpts {
            format: ClipFormat::Gif,
            ..RecordOpts::default()
        };
        let params = effective_record_params(&gif_defaults);
        assert_eq!(params.fps, GIF_DEFAULT_FPS);
        assert_eq!(params.max_dimension, GIF_DEFAULT_DIMENSION);

        let mp4 = RecordOpts::default();
        let params = effective_record_params(&mp4);
        assert_eq!(params.fps, MP4_DEFAULT_FPS);
        assert_eq!(params.max_dimension, MP4_DEFAULT_DIMENSION);
        assert_eq!(params.max_duration_s, 300);
    }

    /// Live evidence for Phase 1: print the real permission map with probe.
    /// Needs a Screen Recording grant for the test process:
    /// cargo test --lib spike_permission_map -- --ignored --nocapture
    #[test]
    #[ignore = "manual: queries real TCC state and runs a capture probe"]
    fn spike_permission_map() {
        let map = permission_map_blocking(true);
        println!(
            "permission map: {}",
            serde_json::to_string_pretty(&map).unwrap()
        );
    }

    #[test]
    fn permission_map_serializes_camel_case() {
        let map = CapturePermissionMap {
            screen_recording: PermissionState::Granted,
            accessibility: PermissionState::Denied,
            screen_capture_verified: Some(false),
        };
        let json = serde_json::to_value(&map).unwrap();
        assert_eq!(json["screenRecording"], "granted");
        assert_eq!(json["accessibility"], "denied");
        assert_eq!(json["screenCaptureVerified"], false);
    }
}
