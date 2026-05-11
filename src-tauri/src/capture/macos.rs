use super::{CaptureError, ScreenCapture};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use std::path::PathBuf;
use tokio::process::Command;

extern "C" {
    fn CGPreflightScreenCaptureAccess() -> bool;
}

/// True when the running process is currently authorized by macOS for
/// Screen Recording. Reads CoreGraphics's TCC-backed authorization
/// state — the same state `screencapture` will see when we shell out.
/// macOS caches this per-process at launch, so a toggle flipped in
/// System Settings is reflected here only after a process restart.
fn has_screen_recording_permission() -> bool {
    unsafe { CGPreflightScreenCaptureAccess() }
}

/// Capture an arbitrary rectangle of the desktop in *logical* pixels via the
/// built-in `screencapture` CLI. `-x` silences the shutter, `-R x,y,w,h`
/// constrains the capture to a region in global screen coordinates.
pub async fn capture_rect(
    x_logical: i32,
    y_logical: i32,
    w_logical: i32,
    h_logical: i32,
) -> Result<ScreenCapture, CaptureError> {
    let id = uuid::Uuid::new_v4();
    let mut path: PathBuf = std::env::temp_dir();
    path.push(format!("screenie-screen-{}.png", id));

    let rect = format!("{},{},{},{}", x_logical, y_logical, w_logical, h_logical);
    let status = Command::new("/usr/sbin/screencapture")
        .args(["-x", "-t", "png", "-R", &rect])
        .arg(&path)
        .status()
        .await?;

    if !status.success() {
        return Err(CaptureError::Other(format!(
            "screencapture exited with {:?}",
            status.code()
        )));
    }

    let bytes = tokio::fs::read(&path).await;
    let _ = tokio::fs::remove_file(&path).await;
    let bytes = bytes?;

    let (width, height) = png_dimensions(&bytes)
        .ok_or_else(|| CaptureError::Other("could not parse PNG dimensions".into()))?;

    // Only run the all-black heuristic when the OS itself reports we do
    // NOT have Screen Recording permission. When we do have it, the
    // capture pixels are whatever the user pointed at — a dark
    // terminal, a black wallpaper, a fullscreen dark-mode app — and
    // the heuristic produces false positives that latch the recovery
    // banner on every capture even though nothing is wrong.
    let blank = !has_screen_recording_permission() && is_blank(&bytes);

    Ok(ScreenCapture {
        png_base64: STANDARD.encode(&bytes),
        width,
        height,
        cursor_x: None,
        cursor_y: None,
        blank,
    })
}

fn png_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    if bytes.len() < 24 || &bytes[0..8] != b"\x89PNG\r\n\x1a\n" || &bytes[12..16] != b"IHDR" {
        return None;
    }
    let w = u32::from_be_bytes(bytes[16..20].try_into().ok()?);
    let h = u32::from_be_bytes(bytes[20..24].try_into().ok()?);
    Some((w, h))
}

/// Sparse-sample the decoded PNG and return true when no channel exceeds a
/// near-black threshold — a strong signal that Screen Recording permission
/// was denied (macOS hands back an all-black image in that case).
fn is_blank(bytes: &[u8]) -> bool {
    let img = match image::load_from_memory_with_format(bytes, image::ImageFormat::Png) {
        Ok(i) => i,
        Err(_) => return false,
    };
    let rgba = img.to_rgba8();
    let pixels = rgba.as_raw();
    let total = pixels.len() / 4;
    if total == 0 {
        return false;
    }
    let stride = (total / 200).max(1);
    for i in (0..total).step_by(stride) {
        let r = pixels[i * 4];
        let g = pixels[i * 4 + 1];
        let b = pixels[i * 4 + 2];
        if r >= 16 || g >= 16 || b >= 16 {
            return false;
        }
    }
    true
}
