use super::{png_dimensions, CaptureError, ScreenCapture};
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
    let rect = format!("{},{},{},{}", x_logical, y_logical, w_logical, h_logical);
    capture_cli(&["-x", "-t", "png", "-R", &rect]).await
}

/// Capture a single window by CGWindowID via the `screencapture` CLI
/// (`-l <id>`; `-o` omits the drop shadow). Pre-macOS-14 fallback for the
/// engine's window-target captures — `SCScreenshotManager` needs 14+.
pub async fn capture_window_cli(window_id: u32) -> Result<ScreenCapture, CaptureError> {
    let id_arg = window_id.to_string();
    capture_cli(&["-x", "-t", "png", "-o", "-l", &id_arg]).await
}

async fn capture_cli(args: &[&str]) -> Result<ScreenCapture, CaptureError> {
    let id = uuid::Uuid::new_v4();
    let mut path: PathBuf = std::env::temp_dir();
    path.push(format!("screenie-screen-{}.png", id));

    let status = Command::new("/usr/sbin/screencapture")
        .args(args)
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

/// Sparse-sample the decoded PNG and return true ONLY when every sampled
/// channel is exactly 0. This matches macOS's TCC "Screen Recording
/// denied" placeholder (uniform 0x00) without false-positives on real
/// dark-mode / dim-content captures — even a dark terminal contains some
/// non-zero pixels (anti-aliased text, scrollbar tracks, focus rings).
fn is_blank(bytes: &[u8]) -> bool {
    super::png_is_blank(bytes)
}
