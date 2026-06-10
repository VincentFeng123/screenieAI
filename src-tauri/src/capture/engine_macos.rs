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
    pub(crate) fn screenie_frontmost_window_id() -> u32;
    pub(crate) fn screenie_capture_list_targets() -> *const c_char;
    pub(crate) fn screenie_capture_target_png(
        display_id: u32,
        window_id: u32,
        max_dimension: u32,
        exclude_self: bool,
        show_cursor: bool,
    ) -> *const c_char;
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

/// CGWindowID of the frontmost standard window not owned by this process.
pub(crate) fn frontmost_window_id() -> Option<u32> {
    let id = unsafe { screenie_frontmost_window_id() };
    (id != 0).then_some(id)
}

// ---------------------------------------------------------------------------
// Shareable-target enumeration
// ---------------------------------------------------------------------------

/// Parsed form of the JSON from `screenie_capture_list_targets`. Coordinates
/// are global logical points (same space as `capture::capture_rect`).
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct TargetsPayload {
    pub displays: Vec<DisplayTarget>,
    pub windows: Vec<WindowTarget>,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DisplayTarget {
    pub id: u32,
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    pub pixel_width: f64,
    pub pixel_height: f64,
}

impl DisplayTarget {
    pub(crate) fn scale(&self) -> f64 {
        if self.width > 0.0 && self.pixel_width > 0.0 {
            self.pixel_width / self.width
        } else {
            1.0
        }
    }

    fn contains(&self, px: f64, py: f64) -> bool {
        px >= self.x && px < self.x + self.width && py >= self.y && py < self.y + self.height
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct WindowTarget {
    pub id: u32,
    pub title: String,
    pub app: String,
    pub pid: i32,
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
    pub on_screen: bool,
    pub layer: i32,
}

/// Blocking: enumerates shareable content over FFI (internal 5s semaphore).
pub(crate) fn list_targets_blocking() -> Result<TargetsPayload, CaptureError> {
    let json = take_bridge_string(unsafe { screenie_capture_list_targets() }).ok_or_else(|| {
        CaptureError::Other(
            "shareable-content enumeration failed (is Screen Recording allowed?)".into(),
        )
    })?;
    serde_json::from_str(&json)
        .map_err(|e| CaptureError::Other(format!("targets payload parse: {e}")))
}

// ---------------------------------------------------------------------------
// Single-frame capture
// ---------------------------------------------------------------------------

use std::path::PathBuf;

use base64::{engine::general_purpose::STANDARD, Engine as _};

use super::engine::{CaptureTarget, CapturedFrame, FrameOpts};
use super::CaptureError;

/// QA escape hatch: force the pre-macOS-14 screencapture-CLI path on modern
/// machines so the fallback stays exercised (real 12.3–13.x hardware is
/// rarely available).
fn force_legacy_capture() -> bool {
    std::env::var("SCREENIE_FORCE_LEGACY_CAPTURE").is_ok_and(|v| v == "1")
}

fn sck_still_path_available() -> bool {
    (unsafe { screenie_screenshot_api_available() }) && !force_legacy_capture()
}

fn sck_target_png_blocking(
    display_id: u32,
    window_id: u32,
    max_dimension: u32,
    exclude_self: bool,
    show_cursor: bool,
) -> Result<String, CaptureError> {
    take_bridge_string(unsafe {
        screenie_capture_target_png(display_id, window_id, max_dimension, exclude_self, show_cursor)
    })
    .ok_or_else(|| {
        CaptureError::Other(
            "ScreenCaptureKit still capture failed (target gone, capture denied, or timed out)"
                .into(),
        )
    })
}

async fn sck_target_png(
    display_id: u32,
    window_id: u32,
    max_dimension: u32,
    exclude_self: bool,
    show_cursor: bool,
) -> Result<String, CaptureError> {
    tauri::async_runtime::spawn_blocking(move || {
        sck_target_png_blocking(display_id, window_id, max_dimension, exclude_self, show_cursor)
    })
    .await
    .map_err(|e| CaptureError::Other(format!("capture task join: {e}")))?
}

async fn list_targets() -> Result<TargetsPayload, CaptureError> {
    tauri::async_runtime::spawn_blocking(list_targets_blocking)
        .await
        .map_err(|e| CaptureError::Other(format!("targets task join: {e}")))?
}

fn display_for_region(
    targets: &TargetsPayload,
    x: f64,
    y: f64,
    w: f64,
    h: f64,
) -> Result<&DisplayTarget, CaptureError> {
    let (cx, cy) = (x + w / 2.0, y + h / 2.0);
    targets
        .displays
        .iter()
        .find(|d| d.contains(cx, cy))
        .or_else(|| targets.displays.first())
        .ok_or_else(|| CaptureError::Other("no display available for region capture".into()))
}

/// One still of a display / window / region, downscaled so the long edge is
/// at most `opts.max_dimension`. The perception fast path: on macOS 14+ a
/// single SCScreenshotManager call with GPU-side downscale; regions grab the
/// containing display at native pixels, then reuse the proven device-pixel
/// crop. Pre-14 (or SCREENIE_FORCE_LEGACY_CAPTURE=1) everything routes
/// through the screencapture CLI.
pub(crate) async fn capture_frame(
    target: CaptureTarget,
    opts: FrameOpts,
    persist_dir: Option<PathBuf>,
) -> Result<CapturedFrame, CaptureError> {
    let sck = sck_still_path_available();

    let png_base64 = match (&target, sck) {
        (CaptureTarget::Window { id }, true) => {
            sck_target_png(0, *id, opts.max_dimension, false, opts.show_cursor).await?
        }
        (CaptureTarget::Window { id }, false) => {
            let capture = super::macos::capture_window_cli(*id).await?;
            super::downscale_for_cloud(&capture.png_base64, opts.max_dimension).await?
        }
        (CaptureTarget::Display { id }, true) => {
            sck_target_png(
                id.unwrap_or(0),
                0,
                opts.max_dimension,
                opts.exclude_self,
                opts.show_cursor,
            )
            .await?
        }
        (CaptureTarget::Display { id }, false) => {
            let targets = list_targets().await?;
            let display = match id {
                Some(id) => targets.displays.iter().find(|d| d.id == *id),
                None => targets.displays.first(),
            }
            .ok_or_else(|| CaptureError::Other("requested display not found".into()))?;
            let capture = super::capture_rect(
                display.x as i32,
                display.y as i32,
                display.width as i32,
                display.height as i32,
            )
            .await?;
            super::downscale_for_cloud(&capture.png_base64, opts.max_dimension).await?
        }
        (CaptureTarget::Region { x, y, w, h }, true) => {
            let (x, y, w, h) = (*x, *y, *w, *h);
            if w < 1.0 || h < 1.0 {
                return Err(CaptureError::Other("region is empty".into()));
            }
            let targets = list_targets().await?;
            let display = display_for_region(&targets, x, y, w, h)?;
            // Native-pixel grab of the containing display, then the proven
            // device-pixel crop (cropping a pre-downscaled image would lose
            // region precision).
            let full =
                sck_target_png(display.id, 0, 0, opts.exclude_self, opts.show_cursor).await?;
            let scale = display.scale();
            let crop_x = ((x - display.x) * scale).max(0.0) as u32;
            let crop_y = ((y - display.y) * scale).max(0.0) as u32;
            let crop_w = (w * scale).round().max(1.0) as u32;
            let crop_h = (h * scale).round().max(1.0) as u32;
            let cropped = super::crop_png_b64(full, crop_x, crop_y, crop_w, crop_h).await?;
            super::downscale_for_cloud(&cropped.png_base64, opts.max_dimension).await?
        }
        (CaptureTarget::Region { x, y, w, h }, false) => {
            let capture =
                super::capture_rect(*x as i32, *y as i32, *w as i32, *h as i32).await?;
            super::downscale_for_cloud(&capture.png_base64, opts.max_dimension).await?
        }
    };

    finish_frame(png_base64, persist_dir).await
}

/// Shared tail: dimensions from the PNG header, the strict blank probe
/// (TCC ground truth), optional persistence.
async fn finish_frame(
    png_base64: String,
    persist_dir: Option<PathBuf>,
) -> Result<CapturedFrame, CaptureError> {
    tauri::async_runtime::spawn_blocking(move || {
        let bytes = STANDARD
            .decode(&png_base64)
            .map_err(|e| CaptureError::Other(format!("base64 decode: {e}")))?;
        let (width, height) = super::png_dimensions(&bytes)
            .ok_or_else(|| CaptureError::Other("could not parse PNG dimensions".into()))?;
        let blank = super::png_is_blank(&bytes);
        let path = match persist_dir {
            Some(dir) => Some(
                super::storage::persist_frame(&dir, &bytes)?
                    .to_string_lossy()
                    .into_owned(),
            ),
            None => None,
        };
        Ok(CapturedFrame {
            png_base64,
            width,
            height,
            path,
            blank,
        })
    })
    .await
    .map_err(|e| CaptureError::Other(format!("frame finish task join: {e}")))?
}

// ---------------------------------------------------------------------------
// Recording (mp4 via native AVAssetWriter, gif via frame-tap + Rust encoder)
// ---------------------------------------------------------------------------

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::Arc;

use super::engine::{ClipFormat, EffectiveRecordParams, RecordOpts, StopOutcome};

/// One BGRA frame copied out of the SCStream tap (tight stride).
struct GifFrame {
    bgra: Vec<u8>,
    width: u32,
    height: u32,
    pts_seconds: f64,
}

/// Shared with the ObjC trampoline as an `Arc` raw pointer. The pointer is
/// reclaimed only after `screenie_recording_stop` + `_release` have returned
/// (the sample queue is drained by then, so no further callbacks can fire).
struct GifTapCtx {
    tx: SyncSender<GifFrame>,
    dropped: AtomicU64,
}

/// Fires on the SCStream sample queue: copy and bail, never block. A full
/// channel means the encoder is behind — drop the frame and count it.
extern "C" fn gif_frame_trampoline(
    bgra: *const u8,
    _len: usize,
    width: u32,
    height: u32,
    bytes_per_row: usize,
    pts_seconds: f64,
    ctx: *mut c_void,
) {
    if bgra.is_null() || ctx.is_null() || width == 0 || height == 0 {
        return;
    }
    let ctx = unsafe { &*(ctx as *const GifTapCtx) };
    let row_bytes = width as usize * 4;
    let mut buf = Vec::with_capacity(row_bytes * height as usize);
    for row in 0..height as usize {
        let src = unsafe { std::slice::from_raw_parts(bgra.add(row * bytes_per_row), row_bytes) };
        buf.extend_from_slice(src);
    }
    let frame = GifFrame {
        bgra: buf,
        width,
        height,
        pts_seconds,
    };
    if ctx.tx.try_send(frame).is_err() {
        ctx.dropped.fetch_add(1, Ordering::Relaxed);
    }
}

struct GifEncodeStats {
    frames_written: u64,
}

/// Streaming GIF encoder: O(1 frame) memory, frames go straight to disk.
/// Per-frame delays come from real PTS deltas — SCK only delivers frames
/// when content changes, so fixed 1/fps delays would make static-screen
/// gifs play absurdly fast.
fn run_gif_encoder(
    path: std::path::PathBuf,
    width: u16,
    height: u16,
    fps: u32,
    rx: Receiver<GifFrame>,
) -> Result<GifEncodeStats, String> {
    use gif::{Encoder, Frame, Repeat};

    let file = std::fs::File::create(&path).map_err(|e| format!("gif create: {e}"))?;
    let writer = std::io::BufWriter::new(file);
    let mut encoder =
        Encoder::new(writer, width, height, &[]).map_err(|e| format!("gif encoder: {e}"))?;
    encoder
        .set_repeat(Repeat::Infinite)
        .map_err(|e| format!("gif repeat: {e}"))?;

    let default_delay_cs = ((100.0 / fps as f64).round() as u16).max(2);
    let mut frames_written: u64 = 0;
    // The delay stored on frame N is "how long to show N", i.e. the gap to
    // frame N+1 — so hold each frame back until its successor arrives.
    let mut pending: Option<GifFrame> = None;

    let mut write_frame = |frame: GifFrame, delay_cs: u16| -> Result<(), String> {
        let mut rgba = frame.bgra;
        for px in rgba.chunks_exact_mut(4) {
            px.swap(0, 2); // BGRA -> RGBA
        }
        let mut out = Frame::from_rgba_speed(frame.width as u16, frame.height as u16, &mut rgba, 10);
        out.delay = delay_cs.max(2);
        encoder
            .write_frame(&out)
            .map_err(|e| format!("gif frame write: {e}"))?;
        frames_written += 1;
        Ok(())
    };

    for frame in rx {
        if let Some(prev) = pending.take() {
            let delta_cs = ((frame.pts_seconds - prev.pts_seconds) * 100.0).round();
            let delay = if delta_cs.is_finite() && delta_cs > 0.0 {
                delta_cs.min(u16::MAX as f64) as u16
            } else {
                default_delay_cs
            };
            write_frame(prev, delay)?;
        }
        pending = Some(frame);
    }
    if let Some(last) = pending.take() {
        write_frame(last, default_delay_cs)?;
    }
    // Encoder drop writes the GIF trailer; BufWriter drop flushes.
    Ok(GifEncodeStats { frames_written })
}

struct GifPipeline {
    /// `Arc::into_raw` handed to the ObjC tap; reclaimed in `stop`.
    ctx: *mut GifTapCtx,
    encoder: Option<std::thread::JoinHandle<Result<GifEncodeStats, String>>>,
}

/// A live native recording session. All methods are blocking — call from
/// `spawn_blocking`. Send is sound: the opaque handle is only ever passed to
/// thread-safe extern fns (atomic properties; the recorder's real work runs
/// on its own dispatch queues).
pub(crate) struct MacRecording {
    handle: *mut c_void,
    gif: Option<GifPipeline>,
}

unsafe impl Send for MacRecording {}

pub(crate) struct StartSpec {
    pub display_id: u32,
    pub window_id: u32,
    /// Display-relative source rect in points (region targets only).
    pub src_rect: Option<(f64, f64, f64, f64)>,
    pub fps: u32,
    pub out_width: u32,
    pub out_height: u32,
    pub show_cursor: bool,
    pub exclude_self: bool,
    pub format: ClipFormat,
    /// Where the artifact is written while recording (`.inprogress/`).
    pub inprogress_path: std::path::PathBuf,
}

impl MacRecording {
    pub(crate) fn start(spec: &StartSpec) -> Result<Self, CaptureError> {
        let (src_x, src_y, src_w, src_h) = spec.src_rect.unwrap_or((0.0, 0.0, 0.0, 0.0));
        let cfg = ScreenieRecordConfig {
            display_id: spec.display_id,
            window_id: spec.window_id,
            src_x,
            src_y,
            src_w,
            src_h,
            fps: spec.fps,
            out_width: spec.out_width,
            out_height: spec.out_height,
            show_cursor: spec.show_cursor,
            exclude_self: spec.exclude_self,
        };

        let mut gif: Option<GifPipeline> = None;
        let (c_path, frame_cb, frame_ctx): (
            Option<std::ffi::CString>,
            Option<ScreenieFrameCallback>,
            *mut c_void,
        ) = match spec.format {
            ClipFormat::Mp4 => {
                let c_path = std::ffi::CString::new(
                    spec.inprogress_path.to_string_lossy().as_bytes(),
                )
                .map_err(|_| CaptureError::Other("invalid clip path".into()))?;
                (Some(c_path), None, std::ptr::null_mut())
            }
            ClipFormat::Gif => {
                // Bounded channel: ~4 in-flight frames (≈4 MiB at the gif
                // dimension caps). Overflow drops frames instead of stalling
                // the SCK sample queue.
                let (tx, rx) = sync_channel::<GifFrame>(4);
                let ctx = Arc::into_raw(Arc::new(GifTapCtx {
                    tx,
                    dropped: AtomicU64::new(0),
                })) as *mut GifTapCtx;
                let path = spec.inprogress_path.clone();
                let (w, h, fps) = (spec.out_width as u16, spec.out_height as u16, spec.fps);
                let encoder = std::thread::Builder::new()
                    .name("screenie-gif-encoder".into())
                    .spawn(move || run_gif_encoder(path, w, h, fps, rx))
                    .map_err(|e| CaptureError::Other(format!("gif encoder spawn: {e}")))?;
                gif = Some(GifPipeline {
                    ctx,
                    encoder: Some(encoder),
                });
                (None, Some(gif_frame_trampoline as ScreenieFrameCallback), ctx as *mut c_void)
            }
        };

        let mut err_slot: *mut c_char = std::ptr::null_mut();
        let handle = unsafe {
            screenie_recording_start(
                &cfg,
                c_path.as_ref().map_or(std::ptr::null(), |p| p.as_ptr()),
                frame_cb,
                frame_ctx,
                &mut err_slot,
            )
        };
        if handle.is_null() {
            // Tear the gif pipeline down before surfacing the error: drop the
            // tap context (no stream ever existed, so no callbacks) and let
            // the encoder thread run dry.
            if let Some(mut pipeline) = gif {
                drop(unsafe { Arc::from_raw(pipeline.ctx as *const GifTapCtx) });
                if let Some(encoder) = pipeline.encoder.take() {
                    let _ = encoder.join();
                }
            }
            let err = take_bridge_error(err_slot)
                .unwrap_or_else(|| "recording failed to start".into());
            return Err(CaptureError::Recording(err));
        }
        Ok(Self { handle, gif })
    }

    /// 0 = recording, 1 = stopped, 2 = failed.
    pub(crate) fn state(&self) -> i32 {
        unsafe { screenie_recording_state(self.handle) }
    }

    pub(crate) fn failure(&self) -> Option<String> {
        take_bridge_string(unsafe { screenie_recording_error(self.handle) })
    }

    pub(crate) fn frame_count(&self) -> u64 {
        unsafe { screenie_recording_frame_count(self.handle) }
    }

    /// Stop, finalize, release the native handle, and (gif) drain + join the
    /// encoder. Consumes self: the session is gone afterwards either way.
    pub(crate) fn stop(mut self) -> StopOutcome {
        let mut err_slot: *mut c_char = std::ptr::null_mut();
        let ok = unsafe { screenie_recording_stop(self.handle, &mut err_slot) };
        let mut error = take_bridge_error(err_slot);
        let mut frame_count = unsafe { screenie_recording_frame_count(self.handle) };
        // After stop the sample queue is drained: no further callbacks.
        unsafe { screenie_recording_release(self.handle) };
        self.handle = std::ptr::null_mut();

        let mut finalized = ok;
        if let Some(mut pipeline) = self.gif.take() {
            // Reclaim the tap context; dropping it hangs up the channel so
            // the encoder thread drains and exits.
            let ctx = unsafe { Arc::from_raw(pipeline.ctx as *const GifTapCtx) };
            let dropped = ctx.dropped.load(Ordering::Relaxed);
            drop(ctx);
            match pipeline.encoder.take().map(|t| t.join()) {
                Some(Ok(Ok(stats))) => {
                    frame_count = stats.frames_written;
                    if stats.frames_written == 0 {
                        finalized = false;
                        error.get_or_insert_with(|| {
                            "no frames were captured (zero complete frames delivered)".into()
                        });
                    } else if dropped > 0 {
                        eprintln!(
                            "[screenie] gif encoder dropped {dropped} frames (encoder slower than capture)"
                        );
                    }
                }
                Some(Ok(Err(e))) => {
                    finalized = false;
                    error = Some(e);
                }
                Some(Err(_)) => {
                    finalized = false;
                    error = Some("gif encoder thread panicked".into());
                }
                None => {}
            }
        }

        StopOutcome {
            finalized,
            error,
            frame_count,
        }
    }
}

/// Resolve a capture target into a native start spec: ids, output pixel
/// dimensions (long-edge clamped, floored to even for H.264), and the
/// display-relative source rect for regions. Blocking (target enumeration).
pub(crate) fn resolve_start_spec_blocking(
    target: &CaptureTarget,
    opts: &RecordOpts,
    params: &EffectiveRecordParams,
    inprogress_path: std::path::PathBuf,
) -> Result<StartSpec, CaptureError> {
    let targets = list_targets_blocking()?;

    let (display_id, window_id, src_rect, native_w, native_h) = match target {
        CaptureTarget::Display { id } => {
            let display = match id {
                Some(id) => targets.displays.iter().find(|d| d.id == *id),
                None => targets.displays.first(),
            }
            .ok_or_else(|| CaptureError::Other("requested display not found".into()))?;
            (display.id, 0, None, display.pixel_width, display.pixel_height)
        }
        CaptureTarget::Window { id } => {
            let window = targets
                .windows
                .iter()
                .find(|w| w.id == *id)
                .ok_or_else(|| {
                    CaptureError::Other("target window not found (it may have closed)".into())
                })?;
            let (cx, cy) = (window.x + window.width / 2.0, window.y + window.height / 2.0);
            let scale = targets
                .displays
                .iter()
                .find(|d| d.contains(cx, cy))
                .or_else(|| targets.displays.first())
                .map(|d| d.scale())
                .unwrap_or(1.0);
            (0, window.id, None, window.width * scale, window.height * scale)
        }
        CaptureTarget::Region { x, y, w, h } => {
            if *w < 1.0 || *h < 1.0 {
                return Err(CaptureError::Other("region is empty".into()));
            }
            let display = display_for_region(&targets, *x, *y, *w, *h)?;
            let scale = display.scale();
            (
                display.id,
                0,
                Some((x - display.x, y - display.y, *w, *h)),
                w * scale,
                h * scale,
            )
        }
    };

    if native_w < 1.0 || native_h < 1.0 {
        return Err(CaptureError::Other("capture target has no size".into()));
    }
    let long_edge = native_w.max(native_h);
    let scale_down = if long_edge > params.max_dimension as f64 {
        params.max_dimension as f64 / long_edge
    } else {
        1.0
    };
    // Floor to even: H.264 encoders reject odd dimensions.
    let out_width = (((native_w * scale_down) as u32).max(2) / 2) * 2;
    let out_height = (((native_h * scale_down) as u32).max(2) / 2) * 2;

    Ok(StartSpec {
        display_id,
        window_id,
        src_rect,
        fps: params.fps,
        out_width,
        out_height,
        show_cursor: opts.show_cursor,
        exclude_self: opts.exclude_self,
        format: opts.format,
        inprogress_path,
    })
}

#[cfg(test)]
mod spike_tests {
    use super::*;
    use std::ptr;

    /// Phase-2 evidence: one frame per target kind with timing, plus the
    /// legacy-CLI path. Needs Screen Recording TCC:
    /// cargo test --lib spike_capture_frame_targets -- --ignored --nocapture
    #[tokio::test]
    #[ignore = "manual spike: captures the real screen; needs Screen Recording TCC"]
    async fn spike_capture_frame_targets() {
        let time = |label: &str, frame: &CapturedFrame, t0: std::time::Instant| {
            println!(
                "{label}: {}x{} blank={} b64={}KiB path={:?} in {:?}",
                frame.width,
                frame.height,
                frame.blank,
                frame.png_base64.len() / 1024,
                frame.path,
                t0.elapsed()
            );
        };

        // Display target (default opts: downscale to 1568, exclude self).
        let t0 = std::time::Instant::now();
        let frame = capture_frame(
            CaptureTarget::Display { id: None },
            FrameOpts::default(),
            None,
        )
        .await
        .expect("display capture");
        time("display", &frame, t0);
        assert!(!frame.blank, "display frame came back blank");
        assert!(frame.width.max(frame.height) <= 1568);

        // Window target: first plausible on-screen app window.
        let targets = list_targets().await.expect("list targets");
        let window = targets
            .windows
            .iter()
            .find(|w| w.on_screen && w.layer == 0 && w.width >= 200.0 && w.height >= 150.0)
            .expect("no plausible window on screen");
        println!("window target: {} — {} ({})", window.app, window.title, window.id);
        let t0 = std::time::Instant::now();
        let frame = capture_frame(
            CaptureTarget::Window { id: window.id },
            FrameOpts::default(),
            None,
        )
        .await
        .expect("window capture");
        time("window", &frame, t0);
        assert!(!frame.blank);

        // Region target with persist (Retina crop correctness: 200x150pt).
        let tmp = std::env::temp_dir().join("screenie-frame-spike");
        let t0 = std::time::Instant::now();
        let frame = capture_frame(
            CaptureTarget::Region {
                x: 100.0,
                y: 100.0,
                w: 200.0,
                h: 150.0,
            },
            FrameOpts::default(),
            Some(tmp.clone()),
        )
        .await
        .expect("region capture");
        time("region", &frame, t0);
        let path = frame.path.clone().expect("persisted path");
        assert!(std::path::Path::new(&path).exists(), "persisted file missing");
        // 200pt at 1x..3x scale → 200..600 px wide; must respect the crop.
        assert!(
            (200..=600).contains(&frame.width),
            "region width {} outside expected device-pixel range",
            frame.width
        );

        // Legacy-CLI path (the pre-14 fallback) must also produce a frame.
        std::env::set_var("SCREENIE_FORCE_LEGACY_CAPTURE", "1");
        let t0 = std::time::Instant::now();
        let result = capture_frame(
            CaptureTarget::Display { id: None },
            FrameOpts::default(),
            None,
        )
        .await;
        std::env::remove_var("SCREENIE_FORCE_LEGACY_CAPTURE");
        let frame = result.expect("legacy display capture");
        time("display(legacy-cli)", &frame, t0);
        assert!(!frame.blank);

        let _ = std::fs::remove_dir_all(&tmp);
    }

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
