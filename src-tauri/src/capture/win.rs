//! Windows-side screen region capture. Uses GDI `BitBlt` of the desktop DC.
//!
//! The overlay window is configured with `WDA_EXCLUDEFROMCAPTURE` (see
//! `crate::windows_window::configure_overlay_window`), so even live captures
//! that run while the overlay is visible — the refresh path used by the chat
//! panel backdrop — produce a clean image of what's underneath without baking
//! our own UI into the bitmap. That's the Windows equivalent of macOS
//! ScreenCaptureKit's "exclude self" content filter.
//!
//! Coordinates: callers pass logical (point) pixels matching Tauri's
//! `Monitor::position()` / `Monitor::size()` divided by `scale_factor()`.
//! GDI works in physical pixels in the virtual-screen coordinate space, so
//! we look up the per-monitor DPI for the rect's origin and convert.

use super::{CaptureError, ScreenCapture};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use image::{ImageBuffer, Rgba};
use windows::Win32::Foundation::{HWND, POINT};
use windows::Win32::Graphics::Gdi::{
    BitBlt, CreateCompatibleBitmap, CreateCompatibleDC, DeleteDC, DeleteObject, GetDC, GetDIBits,
    MonitorFromPoint, ReleaseDC, SelectObject, BITMAPINFO, BITMAPINFOHEADER, CAPTUREBLT,
    DIB_RGB_COLORS, HBITMAP, HDC, MONITOR_DEFAULTTOPRIMARY, RGBQUAD, SRCCOPY,
};
use windows::Win32::UI::HiDpi::{GetDpiForMonitor, MDT_EFFECTIVE_DPI};

pub async fn capture_rect(
    x_logical: i32,
    y_logical: i32,
    w_logical: i32,
    h_logical: i32,
) -> Result<ScreenCapture, CaptureError> {
    tokio::task::spawn_blocking(move || {
        capture_rect_blocking(x_logical, y_logical, w_logical, h_logical)
    })
    .await
    .map_err(|e| CaptureError::Other(format!("blocking task join: {e}")))?
}

fn capture_rect_blocking(
    x_logical: i32,
    y_logical: i32,
    w_logical: i32,
    h_logical: i32,
) -> Result<ScreenCapture, CaptureError> {
    let scale = dpi_scale_for_point(x_logical, y_logical);
    let x = (x_logical as f64 * scale).round() as i32;
    let y = (y_logical as f64 * scale).round() as i32;
    let w = (w_logical as f64 * scale).round() as i32;
    let h = (h_logical as f64 * scale).round() as i32;

    if w <= 0 || h <= 0 {
        return Err(CaptureError::Other(format!(
            "invalid capture rect: {w}x{h}"
        )));
    }
    if (w as u64) * (h as u64) > super::MAX_IMAGE_PIXELS {
        return Err(CaptureError::Other("image dimensions too large".into()));
    }

    let bgra = unsafe { bitblt_to_bgra(x, y, w, h) }?;

    // BGRA → RGBA in place. GDI may leave the alpha channel undefined, so
    // pin it to 0xff (opaque) — the PNG encoder otherwise produces a fully
    // transparent image even though the colour channels are correct.
    let mut rgba = bgra.clone();
    for px in rgba.chunks_exact_mut(4) {
        px.swap(0, 2);
        px[3] = 0xff;
    }
    let buf = ImageBuffer::<Rgba<u8>, _>::from_raw(w as u32, h as u32, rgba)
        .ok_or_else(|| CaptureError::Other("RgbaImage::from_raw failed".into()))?;
    let mut png: Vec<u8> = Vec::with_capacity((w * h) as usize);
    image::DynamicImage::ImageRgba8(buf)
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .map_err(|e| CaptureError::Other(format!("PNG encode: {e}")))?;

    let blank = is_blank_bgra(&bgra);

    Ok(ScreenCapture {
        png_base64: STANDARD.encode(&png),
        width: w as u32,
        height: h as u32,
        cursor_x: None,
        cursor_y: None,
        blank,
    })
}

unsafe fn bitblt_to_bgra(x: i32, y: i32, w: i32, h: i32) -> Result<Vec<u8>, CaptureError> {
    // `HWND(null)` = the desktop / screen device context.
    let null_hwnd = HWND(core::ptr::null_mut());
    let screen_dc = GetDC(null_hwnd);
    if screen_dc.is_invalid() {
        return Err(CaptureError::Other("GetDC failed".into()));
    }
    // Manual scope guards so any early error path still releases GDI handles.
    struct DcGuard(HDC);
    impl Drop for DcGuard {
        fn drop(&mut self) {
            unsafe {
                ReleaseDC(HWND(core::ptr::null_mut()), self.0);
            }
        }
    }
    let _screen = DcGuard(screen_dc);

    let mem_dc = CreateCompatibleDC(screen_dc);
    if mem_dc.is_invalid() {
        return Err(CaptureError::Other("CreateCompatibleDC failed".into()));
    }
    struct MemDcGuard(HDC);
    impl Drop for MemDcGuard {
        fn drop(&mut self) {
            unsafe {
                let _ = DeleteDC(self.0);
            }
        }
    }
    let _mem = MemDcGuard(mem_dc);

    let bitmap = CreateCompatibleBitmap(screen_dc, w, h);
    if bitmap.is_invalid() {
        return Err(CaptureError::Other("CreateCompatibleBitmap failed".into()));
    }
    struct BmpGuard(HBITMAP);
    impl Drop for BmpGuard {
        fn drop(&mut self) {
            unsafe {
                let _ = DeleteObject(self.0);
            }
        }
    }
    let _bmp = BmpGuard(bitmap);

    let old = SelectObject(mem_dc, bitmap);

    // `CAPTUREBLT` is required to include layered windows in the copy — most
    // of the user's actual UI on modern Windows is composited that way, so
    // omitting it produces a black frame on any window with a DWM thumbnail.
    let blt = BitBlt(mem_dc, 0, 0, w, h, screen_dc, x, y, SRCCOPY | CAPTUREBLT);

    // Read pixels via DIB.
    let mut info = BITMAPINFO {
        bmiHeader: BITMAPINFOHEADER {
            biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
            biWidth: w,
            // Negative height = top-down DIB so byte order is row 0 first.
            biHeight: -h,
            biPlanes: 1,
            biBitCount: 32,
            biCompression: 0, // BI_RGB
            biSizeImage: 0,
            biXPelsPerMeter: 0,
            biYPelsPerMeter: 0,
            biClrUsed: 0,
            biClrImportant: 0,
        },
        bmiColors: [RGBQUAD::default(); 1],
    };

    let stride = (w * 4) as usize;
    let mut buf = vec![0u8; stride * h as usize];

    let rows_read = GetDIBits(
        mem_dc,
        bitmap,
        0,
        h as u32,
        Some(buf.as_mut_ptr() as _),
        &mut info,
        DIB_RGB_COLORS,
    );

    SelectObject(mem_dc, old);

    if blt.is_err() {
        return Err(CaptureError::Other(format!(
            "BitBlt failed: {}",
            blt.unwrap_err()
        )));
    }
    if rows_read == 0 {
        return Err(CaptureError::Other("GetDIBits returned 0 rows".into()));
    }
    Ok(buf)
}

/// Look up the effective DPI of the monitor under the given logical point and
/// return its scale relative to 96 DPI. Tauri's app manifest pins per-monitor
/// DPI v2 awareness, so this matches what the JS-side `devicePixelRatio` sees.
fn dpi_scale_for_point(x_logical: i32, y_logical: i32) -> f64 {
    unsafe {
        let pt = POINT {
            x: x_logical,
            y: y_logical,
        };
        let monitor = MonitorFromPoint(pt, MONITOR_DEFAULTTOPRIMARY);
        let mut dpi_x: u32 = 96;
        let mut dpi_y: u32 = 96;
        if GetDpiForMonitor(monitor, MDT_EFFECTIVE_DPI, &mut dpi_x, &mut dpi_y).is_err() {
            return 1.0;
        }
        if dpi_x == 0 {
            return 1.0;
        }
        dpi_x as f64 / 96.0
    }
}

/// Mirror of the macOS `is_blank` heuristic — if sparse-sampled pixels are
/// all near-black, treat the capture as blank. BitBlt on Windows rarely
/// returns all-black, but DRM-protected surfaces (Netflix, certain banking
/// apps) and capture-blocked windows do produce black rectangles, and we
/// want the same recovery banner that Mac shows on Screen Recording denial.
fn is_blank_bgra(buf: &[u8]) -> bool {
    let total = buf.len() / 4;
    if total == 0 {
        return false;
    }
    let stride = (total / 200).max(1);
    for i in (0..total).step_by(stride) {
        let b = buf[i * 4];
        let g = buf[i * 4 + 1];
        let r = buf[i * 4 + 2];
        if r >= 16 || g >= 16 || b >= 16 {
            return false;
        }
    }
    true
}
