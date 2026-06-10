//! macOS capture/recording engine: FFI into capture_record_macos.m.
//!
//! Mechanism lives in ObjC (SCStream → AVAssetWriter, SCScreenshotManager
//! stills); all logical session state (which session is active, caps, timers)
//! lives in Rust. Every blocking FFI call here waits on internal semaphores —
//! call only from `spawn_blocking`, never from an async context directly.

#![allow(dead_code)] // wired up incrementally across the capture phases

use std::ffi::{c_char, c_void, CStr};

/// Mirrors `ScreenieRecordConfig` in capture_record_macos.m.
#[repr(C)]
pub(crate) struct ScreenieRecordConfig {
    pub display_id: u32,
    pub window_id: u32,
    /// SCStreamConfiguration.sourceRect in POINTS relative to the display
    /// origin; src_w <= 0 means full display.
    pub src_x: f64,
    pub src_y: f64,
    pub src_w: f64,
    pub src_h: f64,
    pub fps: u32,
    /// Output PIXELS — pre-computed even values (H.264 requirement).
    pub out_width: u32,
    pub out_height: u32,
    pub show_cursor: bool,
    pub exclude_self: bool,
}

/// GIF-mode frame tap. Fires on the SCStream sample queue: copy and return
/// immediately; never block.
pub(crate) type ScreenieFrameCallback = extern "C" fn(
    bgra: *const u8,
    len: usize,
    width: u32,
    height: u32,
    bytes_per_row: usize,
    pts_seconds: f64,
    ctx: *mut c_void,
);

extern "C" {
    // capture_record_macos.m
    pub(crate) fn screenie_screenshot_api_available() -> bool;
    pub(crate) fn screenie_has_accessibility_access() -> bool;
    pub(crate) fn screenie_capture_list_targets() -> *const c_char;
    pub(crate) fn screenie_recording_start(
        cfg: *const ScreenieRecordConfig,
        out_path_utf8: *const c_char,
        frame_cb: Option<ScreenieFrameCallback>,
        frame_ctx: *mut c_void,
        error_out: *mut *mut c_char,
    ) -> *mut c_void;
    pub(crate) fn screenie_recording_state(handle: *mut c_void) -> i32;
    pub(crate) fn screenie_recording_error(handle: *mut c_void) -> *const c_char;
    pub(crate) fn screenie_recording_frame_count(handle: *mut c_void) -> u64;
    pub(crate) fn screenie_recording_stop(
        handle: *mut c_void,
        error_out: *mut *mut c_char,
    ) -> bool;
    pub(crate) fn screenie_recording_release(handle: *mut c_void);

    // macos_window.m
    fn screenie_free_string(ptr: *const c_char);
    fn screenie_has_screen_capture_access() -> bool;
}

/// Copy a malloc'd C string returned by the bridge and free the original.
/// Returns None for NULL.
pub(crate) fn take_bridge_string(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    let owned = unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned();
    unsafe { screenie_free_string(ptr) };
    Some(owned)
}

/// Take a `char **error_out` slot filled by the bridge, if any.
pub(crate) fn take_bridge_error(slot: *mut c_char) -> Option<String> {
    take_bridge_string(slot as *const c_char)
}

pub(crate) fn screen_capture_access_granted() -> bool {
    unsafe { screenie_has_screen_capture_access() }
}

#[cfg(test)]
mod spike_tests {
    use super::*;
    use std::ptr;

    /// Phase-0 engine viability spike: record ~3s of the main display to an
    /// mp4 and verify the file finalizes. Needs a Screen Recording TCC grant
    /// for the test binary's responsible process, hence ignored by default:
    ///
    /// cargo test --lib spike_record_3s_mp4 -- --ignored --nocapture
    #[test]
    #[ignore = "manual spike: records the main display; needs Screen Recording TCC"]
    fn spike_record_3s_mp4() {
        let preflight = screen_capture_access_granted();
        println!("CGPreflightScreenCaptureAccess: {preflight}");

        let targets = take_bridge_string(unsafe { screenie_capture_list_targets() });
        println!(
            "list_targets: {}",
            targets.as_deref().map(|t| &t[..t.len().min(400)]).unwrap_or("<null>")
        );
        assert!(
            targets.is_some(),
            "shareable-content enumeration failed (missing Screen Recording grant?)"
        );

        let out_path = std::env::temp_dir().join("screenie-spike.mp4");
        let _ = std::fs::remove_file(&out_path);
        let c_path = std::ffi::CString::new(out_path.to_string_lossy().as_bytes()).unwrap();

        let cfg = ScreenieRecordConfig {
            display_id: 0,
            window_id: 0,
            src_x: 0.0,
            src_y: 0.0,
            src_w: 0.0,
            src_h: 0.0,
            fps: 10,
            out_width: 1280,
            out_height: 720,
            show_cursor: true,
            exclude_self: false,
        };

        let mut err_slot: *mut std::ffi::c_char = ptr::null_mut();
        let handle =
            unsafe { screenie_recording_start(&cfg, c_path.as_ptr(), None, ptr::null_mut(), &mut err_slot) };
        if handle.is_null() {
            let err = take_bridge_error(err_slot).unwrap_or_else(|| "<no error text>".into());
            panic!("recording_start failed: {err}");
        }

        std::thread::sleep(std::time::Duration::from_secs(3));
        let frames_mid = unsafe { screenie_recording_frame_count(handle) };
        let state_mid = unsafe { screenie_recording_state(handle) };
        println!("mid-recording: state={state_mid} frames={frames_mid}");

        let mut stop_err: *mut std::ffi::c_char = ptr::null_mut();
        let ok = unsafe { screenie_recording_stop(handle, &mut stop_err) };
        let stop_err_text = take_bridge_error(stop_err);
        let frames = unsafe { screenie_recording_frame_count(handle) };
        unsafe { screenie_recording_release(handle) };
        println!("stop ok={ok} err={stop_err_text:?} frames={frames}");
        assert!(ok, "recording_stop failed: {stop_err_text:?}");

        let meta = std::fs::metadata(&out_path).expect("output mp4 missing");
        println!("clip: {} ({} bytes, {frames} frames)", out_path.display(), meta.len());
        assert!(meta.len() > 4096, "suspiciously small mp4: {} bytes", meta.len());

        // A finalized (non-fragmented) mp4 must contain a moov atom.
        let bytes = std::fs::read(&out_path).expect("read mp4");
        assert!(
            bytes.windows(4).any(|w| w == b"moov"),
            "mp4 missing moov atom — writer did not finalize"
        );
    }
}
