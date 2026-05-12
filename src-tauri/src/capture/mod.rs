use serde::{Serialize, Serializer};

#[cfg(target_os = "macos")]
mod macos;

#[cfg(target_os = "windows")]
mod win;

const MAX_PNG_B64_CHARS: usize = 96 * 1024 * 1024;
const MAX_IMAGE_PIXELS: u64 = 60_000_000;

#[derive(Debug, thiserror::Error)]
pub enum CaptureError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("image: {0}")]
    Image(#[from] image::ImageError),
    #[error("capture failed: {0}")]
    Other(String),
}

impl Serialize for CaptureError {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_string())
    }
}

/// Full-screen capture of the main display. The PNG bytes are at native (device) resolution.
#[derive(Debug, Serialize, Clone)]
pub struct ScreenCapture {
    /// PNG bytes (whole screen) base64-encoded.
    pub png_base64: String,
    /// Image width in *device* pixels (Retina counts physical pixels).
    pub width: u32,
    /// Image height in *device* pixels.
    pub height: u32,
    /// Current cursor x-coordinate in the overlay window's logical coordinate
    /// space. Filled by the Tauri window layer after capture.
    pub cursor_x: Option<f64>,
    /// Current cursor y-coordinate in the overlay window's logical coordinate
    /// space. Filled by the Tauri window layer after capture.
    pub cursor_y: Option<f64>,
    /// True when the capture appears effectively all-black, which on macOS
    /// almost always means Screen Recording permission is missing.
    pub blank: bool,
}

/// Cropped region returned to the caller after the user clicks Send.
#[derive(Debug, Serialize, Clone)]
pub struct CroppedCapture {
    pub png_base64: String,
    pub width: u32,
    pub height: u32,
}

pub async fn capture_rect(
    x_logical: i32,
    y_logical: i32,
    w_logical: i32,
    h_logical: i32,
) -> Result<ScreenCapture, CaptureError> {
    #[cfg(target_os = "macos")]
    {
        macos::capture_rect(x_logical, y_logical, w_logical, h_logical).await
    }
    #[cfg(target_os = "windows")]
    {
        win::capture_rect(x_logical, y_logical, w_logical, h_logical).await
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        let _ = (x_logical, y_logical, w_logical, h_logical);
        Err(CaptureError::Other(
            "screen capture not yet implemented for this platform".into(),
        ))
    }
}

/// Crop a base64-encoded PNG to the given bounds (in device pixels).
///
/// Async wrapper around [`crop_png_b64_blocking`] that runs the PNG decode
/// + crop + re-encode on the blocking pool. Calling commands are async and
/// often share the runtime worker driving an SSE stream — running this
/// inline pegs that worker for 50-200 ms on a Retina screenshot.
pub async fn crop_png_b64(
    src_b64: String,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
) -> Result<CroppedCapture, CaptureError> {
    tokio::task::spawn_blocking(move || crop_png_b64_blocking(&src_b64, x, y, w, h))
        .await
        .map_err(|e| CaptureError::Other(format!("crop task join: {e}")))?
}

fn crop_png_b64_blocking(
    src_b64: &str,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
) -> Result<CroppedCapture, CaptureError> {
    use base64::{engine::general_purpose::STANDARD, Engine as _};

    if src_b64.len() > MAX_PNG_B64_CHARS {
        return Err(CaptureError::Other("PNG payload too large".into()));
    }
    let src_bytes = STANDARD
        .decode(src_b64)
        .map_err(|e| CaptureError::Other(format!("base64 decode: {e}")))?;
    let img = image::load_from_memory_with_format(&src_bytes, image::ImageFormat::Png)?;

    let (img_w, img_h) = (img.width(), img.height());
    if (img_w as u64).saturating_mul(img_h as u64) > MAX_IMAGE_PIXELS {
        return Err(CaptureError::Other("image dimensions too large".into()));
    }
    let x = x.min(img_w.saturating_sub(1));
    let y = y.min(img_h.saturating_sub(1));
    let w = w.min(img_w - x).max(1);
    let h = h.min(img_h - y).max(1);

    let cropped = img.crop_imm(x, y, w, h);
    let mut out = Vec::with_capacity((w * h * 4) as usize);
    cropped.write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)?;

    Ok(CroppedCapture {
        png_base64: STANDARD.encode(&out),
        width: w,
        height: h,
    })
}

/// Downscale a base64-encoded PNG so its long edge is at most `max_long_edge`
/// pixels. Returns the input unchanged when it's already small enough. We use
/// this before sending images to cloud vision APIs — full-resolution Retina
/// captures are 1500–2500 image-tokens each, but quality holds up fine at
/// ~1024px on the long edge for screenshot-style content.
/// Async wrapper. Heavy PNG decode + resize + re-encode runs on the
/// blocking pool — provider stream() fns call this from their async path
/// and would otherwise block a tokio worker for 100-300 ms.
pub async fn downscale_for_cloud(b64: &str, max_long_edge: u32) -> Result<String, CaptureError> {
    let owned = b64.to_owned();
    tokio::task::spawn_blocking(move || downscale_for_cloud_blocking(&owned, max_long_edge))
        .await
        .map_err(|e| CaptureError::Other(format!("downscale task join: {e}")))?
}

fn downscale_for_cloud_blocking(b64: &str, max_long_edge: u32) -> Result<String, CaptureError> {
    use base64::{engine::general_purpose::STANDARD, Engine as _};

    if b64.len() > MAX_PNG_B64_CHARS {
        return Err(CaptureError::Other("PNG payload too large".into()));
    }
    let bytes = STANDARD
        .decode(b64)
        .map_err(|e| CaptureError::Other(format!("base64 decode: {e}")))?;
    let img = image::load_from_memory_with_format(&bytes, image::ImageFormat::Png)?;

    let (w, h) = (img.width(), img.height());
    if (w as u64).saturating_mul(h as u64) > MAX_IMAGE_PIXELS {
        return Err(CaptureError::Other("image dimensions too large".into()));
    }
    let long = w.max(h);
    if long <= max_long_edge {
        return Ok(b64.to_string());
    }

    let scale = max_long_edge as f64 / long as f64;
    let nw = ((w as f64) * scale).round().max(1.0) as u32;
    let nh = ((h as f64) * scale).round().max(1.0) as u32;
    let resized = img.resize_exact(nw, nh, image::imageops::FilterType::Triangle);

    let mut out = Vec::with_capacity((nw * nh * 4) as usize);
    resized.write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)?;
    Ok(STANDARD.encode(&out))
}
