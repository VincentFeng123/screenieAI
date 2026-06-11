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
    /// All connected displays composed in global logical coordinate space.
    /// Still-frame target only; recording uses single display/window streams.
    VirtualDesktop,
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

// ---------------------------------------------------------------------------
// Recording lifecycle (platform-neutral session management)
// ---------------------------------------------------------------------------

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Outcome of a consuming native stop.
pub(crate) struct StopOutcome {
    /// The output file is usable (possibly a partial clip).
    pub finalized: bool,
    /// Failure cause, or partial-clip reason (e.g. the user hit the system
    /// screen-sharing stop button).
    pub error: Option<String>,
    pub frame_count: u64,
}

#[cfg(target_os = "macos")]
pub(crate) type PlatformRecording = super::engine_macos::MacRecording;

/// Never constructed off-macOS — `start_recording` dispatches to the
/// platform stub before a session can exist. Methods exist so the
/// platform-neutral lifecycle below typechecks everywhere.
#[cfg(not(target_os = "macos"))]
pub(crate) struct PlatformRecording;

#[cfg(not(target_os = "macos"))]
impl PlatformRecording {
    pub(crate) fn state(&self) -> i32 {
        2
    }
    pub(crate) fn failure(&self) -> Option<String> {
        None
    }
    pub(crate) fn stop(self) -> StopOutcome {
        StopOutcome {
            finalized: false,
            error: Some("recording is not supported on this platform".into()),
            frame_count: 0,
        }
    }
}

/// One live recording session. Owned by the `RecordingSlots::active` slot;
/// always consumed through `stop_session`.
pub(crate) struct ActiveRecording {
    pub session_id: String,
    pub started: Instant,
    pub started_at_ms: u64,
    pub format: ClipFormat,
    pub params: EffectiveRecordParams,
    pub max_clip_bytes: u64,
    pub inprogress_path: PathBuf,
    pub final_path: PathBuf,
    pub width: u32,
    pub height: u32,
    pub native: PlatformRecording,
    /// Tripped by any stop path; the watchdog exits when it sees it.
    pub stop_latch: Arc<AtomicBool>,
}

/// Session-coordination state, embedded in `AppState` behind an `Arc` so the
/// watchdog (and tests) can hold it without an `AppHandle`. Engine constraint:
/// ONE active session at a time.
#[derive(Default)]
pub struct RecordingSlots {
    active: Mutex<Option<ActiveRecording>>,
    /// Most recent finished session, so a `stop_recording` that races the
    /// watchdog's auto-stop (or a `record_clip` poller) still gets the
    /// outcome instead of "no active recording".
    finished: Mutex<Option<(String, Result<RecordingFile, String>)>>,
    /// Latch closing the check-then-start gap (mirrors `voice_starting`).
    starting: AtomicBool,
    /// True while a stop is finalizing — losers of the take-race poll
    /// `finished` instead of erroring.
    stopping: AtomicBool,
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

pub fn recording_active(slots: &RecordingSlots) -> bool {
    slots
        .active
        .lock()
        .map(|g| g.is_some())
        .unwrap_or(false)
}

/// Start an open-ended recording session. One at a time; a second start is
/// rejected with `CaptureError::Busy`.
pub async fn start_recording(
    slots: Arc<RecordingSlots>,
    app_data: PathBuf,
    target: CaptureTarget,
    opts: RecordOpts,
) -> Result<RecordingHandle, CaptureError> {
    #[cfg(target_os = "macos")]
    {
        start_recording_impl(slots, app_data, target, opts).await
    }
    #[cfg(target_os = "windows")]
    {
        let _ = (slots, app_data, target, opts);
        super::engine_win::unsupported("screen recording")
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        let _ = (slots, app_data, target, opts);
        Err(CaptureError::Unsupported(
            "screen recording is not available on this platform".into(),
        ))
    }
}

#[cfg(target_os = "macos")]
async fn start_recording_impl(
    slots: Arc<RecordingSlots>,
    app_data: PathBuf,
    target: CaptureTarget,
    opts: RecordOpts,
) -> Result<RecordingHandle, CaptureError> {
    validate_record_opts(&opts)?;
    let params = effective_record_params(&opts);

    if slots
        .starting
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return Err(CaptureError::Busy("another recording is starting".into()));
    }
    // Clear the latch on every exit path.
    struct StartLatch(Arc<RecordingSlots>);
    impl Drop for StartLatch {
        fn drop(&mut self) {
            self.0.starting.store(false, Ordering::SeqCst);
        }
    }
    let _latch = StartLatch(slots.clone());

    if let Ok(guard) = slots.active.lock() {
        if let Some(active) = guard.as_ref() {
            return Err(CaptureError::Busy(format!(
                "a recording session is already active ({}); stop it first",
                active.session_id
            )));
        }
    }

    let (inprogress_path, final_path) =
        super::storage::new_clip_paths(&app_data, opts.format.extension())?;

    let spec_target = target.clone();
    let spec_opts = opts.clone();
    let spec_inprogress = inprogress_path.clone();
    let (spec, native) = tauri::async_runtime::spawn_blocking(move || {
        let spec = super::engine_macos::resolve_start_spec_blocking(
            &spec_target,
            &spec_opts,
            &params,
            spec_inprogress,
        )?;
        let native = super::engine_macos::MacRecording::start(&spec)?;
        Ok::<_, CaptureError>((spec, native))
    })
    .await
    .map_err(|e| CaptureError::Other(format!("recording start task join: {e}")))??;

    let session_id = uuid::Uuid::new_v4().to_string();
    let started_at_ms = now_ms();
    let stop_latch = Arc::new(AtomicBool::new(false));
    let handle = RecordingHandle {
        session_id: session_id.clone(),
        started_at_ms,
        format: opts.format,
        path: final_path.to_string_lossy().into_owned(),
    };

    let active = ActiveRecording {
        session_id: session_id.clone(),
        started: Instant::now(),
        started_at_ms,
        format: opts.format,
        params,
        max_clip_bytes: opts.max_clip_bytes,
        inprogress_path,
        final_path,
        width: spec.out_width,
        height: spec.out_height,
        native,
        stop_latch: stop_latch.clone(),
    };
    if let Ok(mut guard) = slots.active.lock() {
        *guard = Some(active);
    } else {
        return Err(CaptureError::Other("recording slot poisoned".into()));
    }

    spawn_watchdog(slots, session_id, stop_latch);
    Ok(handle)
}

/// Caps enforcement and native-failure detection. Owns auto-stop entirely —
/// the ObjC layer keeps zero timers. A deliberate deviation from early
/// drafts: there is NO "no frames for N seconds" stall heuristic, because
/// SCK legitimately delivers nothing while the screen is static; real source
/// loss (display unplugged, window closed, TCC revoked) surfaces through
/// `didStopWithError` → native state == failed, which this loop does catch.
#[cfg(target_os = "macos")]
fn spawn_watchdog(slots: Arc<RecordingSlots>, session_id: String, stop_latch: Arc<AtomicBool>) {
    tauri::async_runtime::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_millis(500)).await;
            if stop_latch.load(Ordering::SeqCst) {
                return;
            }
            let mut reason: Option<String> = None;
            {
                let Ok(guard) = slots.active.lock() else {
                    return;
                };
                let Some(active) = guard.as_ref().filter(|a| a.session_id == session_id)
                else {
                    return; // session already stopped
                };
                if active.started.elapsed().as_secs() >= active.params.max_duration_s as u64 {
                    reason = Some("maxDuration".into());
                } else if std::fs::metadata(&active.inprogress_path)
                    .map(|m| m.len() >= active.max_clip_bytes)
                    .unwrap_or(false)
                {
                    reason = Some("maxBytes".into());
                } else if active.native.state() == 2 {
                    reason = Some(
                        active
                            .native
                            .failure()
                            .unwrap_or_else(|| "recording failed".into()),
                    );
                }
            }
            if let Some(reason) = reason {
                let _ = stop_session(&slots, Some(&session_id), Some(reason)).await;
                return;
            }
        }
    });
}

/// Stop a session and return the finished clip. `session_id: None` stops
/// whatever is active (run-end cleanup). Tolerates racing the watchdog's
/// auto-stop: if someone else is mid-stop, polls the finished slot for the
/// real outcome instead of erroring.
pub async fn stop_recording(
    slots: Arc<RecordingSlots>,
    session_id: Option<String>,
) -> Result<RecordingFile, CaptureError> {
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        match stop_session(&slots, session_id.as_deref(), None).await {
            Err(CaptureError::Recording(msg)) if msg == STOP_IN_FLIGHT => {
                if Instant::now() >= deadline {
                    return Err(CaptureError::Recording(
                        "recording stop timed out waiting for the in-flight stop".into(),
                    ));
                }
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
            other => return other,
        }
    }
}

const STOP_IN_FLIGHT: &str = "__stop_in_flight__";

/// Single funnel for every stop path (explicit, auto-stop, run-end, exit).
async fn stop_session(
    slots: &Arc<RecordingSlots>,
    session_id: Option<&str>,
    auto_reason: Option<String>,
) -> Result<RecordingFile, CaptureError> {
    // Take the session out of the slot if it matches.
    let mut mismatch: Option<String> = None;
    let taken = {
        let Ok(mut guard) = slots.active.lock() else {
            return Err(CaptureError::Other("recording slot poisoned".into()));
        };
        match (guard.as_ref(), session_id) {
            (Some(active), Some(id)) if active.session_id != id => {
                mismatch = Some(active.session_id.clone());
                None
            }
            (Some(_), _) => guard.take(),
            (None, _) => None,
        }
    };

    let Some(active) = taken else {
        if let Some(current) = mismatch {
            return Err(CaptureError::Recording(format!(
                "session id mismatch: the active recording is {current}"
            )));
        }
        // A racing stop (usually the watchdog) holds the session — wait for
        // its outcome instead of erroring.
        if slots.stopping.load(Ordering::SeqCst) {
            return Err(CaptureError::Recording(STOP_IN_FLIGHT.into()));
        }
        // Already finished: hand back the stored outcome for that session.
        if let Some(id) = session_id {
            if let Ok(finished) = slots.finished.lock() {
                if let Some((fid, result)) = finished.as_ref() {
                    if fid == id {
                        return result.clone().map_err(CaptureError::Recording);
                    }
                }
            }
        }
        return Err(CaptureError::Recording(
            "no active recording session".into(),
        ));
    };

    active.stop_latch.store(true, Ordering::SeqCst);
    slots.stopping.store(true, Ordering::SeqCst);
    struct StopGuard<'a>(&'a AtomicBool);
    impl Drop for StopGuard<'_> {
        fn drop(&mut self) {
            self.0.store(false, Ordering::SeqCst);
        }
    }
    let _stop_guard = StopGuard(&slots.stopping);

    let ActiveRecording {
        session_id: id,
        started,
        format,
        params,
        inprogress_path,
        final_path,
        width,
        height,
        native,
        ..
    } = active;

    let outcome = tauri::async_runtime::spawn_blocking(move || native.stop())
        .await
        .map_err(|e| CaptureError::Other(format!("recording stop task join: {e}")))?;
    let duration_s = started.elapsed().as_secs_f64();

    let result: Result<RecordingFile, String> = if outcome.finalized {
        match super::storage::promote_clip(&inprogress_path, &final_path) {
            Ok(()) => {
                let bytes = std::fs::metadata(&final_path).map(|m| m.len()).unwrap_or(0);
                Ok(RecordingFile {
                    path: final_path.to_string_lossy().into_owned(),
                    bytes,
                    duration_s,
                    width,
                    height,
                    fps: params.fps,
                    format,
                    frame_count: outcome.frame_count,
                    stopped_reason: auto_reason.or(outcome.error),
                })
            }
            Err(e) => {
                let _ = std::fs::remove_file(&inprogress_path);
                Err(format!("failed to finalize clip: {e}"))
            }
        }
    } else {
        let _ = std::fs::remove_file(&inprogress_path);
        Err(outcome
            .error
            .unwrap_or_else(|| "recording failed".into()))
    };

    if let Ok(mut finished) = slots.finished.lock() {
        *finished = Some((id, result.clone()));
    }
    result.map_err(CaptureError::Recording)
}

/// Fixed-duration clip: start, wait out the (clamped) duration — yielding to
/// the watchdog if a cap or failure stops the session first — then stop.
pub async fn record_clip(
    slots: Arc<RecordingSlots>,
    app_data: PathBuf,
    target: CaptureTarget,
    duration_s: f64,
    opts: RecordOpts,
) -> Result<RecordingFile, CaptureError> {
    let params = effective_record_params(&opts);
    let duration = duration_s.clamp(0.5, params.max_duration_s as f64);
    let handle = start_recording(slots.clone(), app_data, target, opts).await?;

    let deadline = Instant::now() + Duration::from_secs_f64(duration);
    loop {
        tokio::time::sleep(Duration::from_millis(200)).await;
        let still_active = slots
            .active
            .lock()
            .map(|g| {
                g.as_ref()
                    .is_some_and(|a| a.session_id == handle.session_id)
            })
            .unwrap_or(false);
        if !still_active {
            // Watchdog beat us to it (cap or failure); fetch its outcome.
            return stop_recording(slots, Some(handle.session_id)).await;
        }
        if Instant::now() >= deadline {
            break;
        }
    }
    stop_recording(slots, Some(handle.session_id)).await
}

/// Best-effort bounded stop for app exit: never leave the macOS recording
/// indicator on or an orphan growing in `.inprogress/`. Waits at most ~2s;
/// a clip that can't finalize in time is collected by the startup sweep.
pub fn stop_active_recording_on_exit(slots: Arc<RecordingSlots>) {
    if !recording_active(&slots) {
        return;
    }
    let (tx, rx) = std::sync::mpsc::channel::<()>();
    std::thread::spawn(move || {
        let _ = tauri::async_runtime::block_on(stop_recording(slots, None));
        let _ = tx.send(());
    });
    let _ = rx.recv_timeout(Duration::from_secs(2));
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
        let desktop: CaptureTarget = serde_json::from_str(r#"{"kind":"virtualDesktop"}"#).unwrap();
        assert_eq!(desktop, CaptureTarget::VirtualDesktop);
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

    /// Phase-3 evidence: full lifecycle against live ScreenCaptureKit.
    /// cargo test --lib spike_recording_lifecycle -- --ignored --nocapture
    #[cfg(target_os = "macos")]
    #[tokio::test]
    #[ignore = "manual spike: records the real screen; needs Screen Recording TCC"]
    async fn spike_recording_lifecycle() {
        let slots = Arc::new(RecordingSlots::default());
        let tmp = std::env::temp_dir().join("screenie-record-spike");
        let _ = std::fs::remove_dir_all(&tmp);

        // 1. Fixed-duration mp4 clip.
        let clip = record_clip(
            slots.clone(),
            tmp.clone(),
            CaptureTarget::Display { id: None },
            3.0,
            RecordOpts::default(),
        )
        .await
        .expect("mp4 record_clip");
        println!(
            "mp4: {} ({} bytes, {} frames, {:.2}s, reason={:?})",
            clip.path, clip.bytes, clip.frame_count, clip.duration_s, clip.stopped_reason
        );
        assert!(clip.path.ends_with(".mp4"));
        assert!(std::path::Path::new(&clip.path).exists());
        assert!(clip.bytes > 4096, "tiny mp4: {} bytes", clip.bytes);
        assert!((2.5..=5.0).contains(&clip.duration_s), "{}", clip.duration_s);
        let mp4_bytes = std::fs::read(&clip.path).unwrap();
        assert!(mp4_bytes.windows(4).any(|w| w == b"moov"), "mp4 not finalized");

        // 2. GIF clip via the frame-tap pipeline.
        let gif_opts = RecordOpts {
            format: ClipFormat::Gif,
            ..RecordOpts::default()
        };
        let gif = record_clip(
            slots.clone(),
            tmp.clone(),
            CaptureTarget::Display { id: None },
            3.0,
            gif_opts,
        )
        .await
        .expect("gif record_clip");
        println!(
            "gif: {} ({} bytes, {} frames, reason={:?})",
            gif.path, gif.bytes, gif.frame_count, gif.stopped_reason
        );
        let gif_bytes = std::fs::read(&gif.path).unwrap();
        assert!(gif_bytes.starts_with(b"GIF89a"), "not a GIF89a file");
        assert!(gif.frame_count >= 1);
        assert!(gif.width <= GIF_MAX_DIMENSION && gif.height <= GIF_MAX_DIMENSION);

        // 3. One-session guard: a second start while active is rejected.
        let handle = start_recording(
            slots.clone(),
            tmp.clone(),
            CaptureTarget::Display { id: None },
            RecordOpts::default(),
        )
        .await
        .expect("session start");
        let second = start_recording(
            slots.clone(),
            tmp.clone(),
            CaptureTarget::Display { id: None },
            RecordOpts::default(),
        )
        .await;
        assert!(
            matches!(second, Err(CaptureError::Busy(_))),
            "second start should be Busy, got {second:?}"
        );
        // Give SCK time to deliver at least one complete frame (it only
        // sends frames when pixels change) before stopping.
        tokio::time::sleep(Duration::from_millis(2000)).await;
        let stopped = stop_recording(slots.clone(), Some(handle.session_id.clone()))
            .await
            .expect("explicit stop");
        println!("session: {} ({} bytes)", stopped.path, stopped.bytes);
        assert!(stopped.stopped_reason.is_none(), "{:?}", stopped.stopped_reason);

        // 4. Auto-stop at the duration cap; a late stop returns the stash.
        let capped = RecordOpts {
            max_duration_s: 2,
            ..RecordOpts::default()
        };
        let handle = start_recording(
            slots.clone(),
            tmp.clone(),
            CaptureTarget::Display { id: None },
            capped,
        )
        .await
        .expect("capped start");
        tokio::time::sleep(Duration::from_secs(4)).await;
        assert!(!recording_active(&slots), "watchdog should have auto-stopped");
        let auto = stop_recording(slots.clone(), Some(handle.session_id.clone()))
            .await
            .expect("late stop should return stash");
        println!("auto-stopped: reason={:?} ({:.2}s)", auto.stopped_reason, auto.duration_s);
        assert_eq!(auto.stopped_reason.as_deref(), Some("maxDuration"));

        // No orphans left behind.
        let staging = super::super::storage::inprogress_dir(&tmp);
        let leftovers = std::fs::read_dir(&staging)
            .map(|d| d.count())
            .unwrap_or(0);
        assert_eq!(leftovers, 0, "orphans left in .inprogress");

        let _ = std::fs::remove_dir_all(&tmp);
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
