use super::{CaptureError, ScreenCapture};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use std::path::PathBuf;
use tokio::process::Command;

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

    // Always probe the captured pixels for the macOS TCC "Screen Recording
    // denied" placeholder (uniform 0x00). The previous gate used
    // `CGPreflightScreenCaptureAccess()` which is per-process cached at
    // launch — after the user toggled the permission in System Settings,
    // our cached value stayed `false` until restart, so the heuristic ran
    // on every capture; the old `>= 16` threshold then false-positived dark
    // real captures (terminals, dark wallpapers, dim windows) and locked
    // users into the recovery banner. The strict `is_blank` below avoids
    // both pitfalls. Run on the blocking pool — full PNG decode of a Retina
    // screenshot is 30-80 ms.
    let probe = bytes.clone();
    let blank = tokio::task::spawn_blocking(move || is_blank(&probe))
        .await
        .unwrap_or(false);

    let png_base64 = tokio::task::spawn_blocking(move || STANDARD.encode(&bytes))
        .await
        .map_err(|e| CaptureError::Other(format!("base64 encode task join: {e}")))?;

    Ok(ScreenCapture {
        png_base64,
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

/// Sparse-sample the decoded PNG and return true ONLY when every sampled
/// channel is exactly 0. This matches macOS's TCC "Screen Recording
/// denied" placeholder (uniform 0x00) without false-positives on real
/// dark-mode / dim-content captures — even a dark terminal contains some
/// non-zero pixels (anti-aliased text, scrollbar tracks, focus rings).
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
        if r != 0 || g != 0 || b != 0 {
            return false;
        }
    }
    true
}
