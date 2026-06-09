//! Three-tier page text extraction for the agent's readPage action.
//!
//! Tier 1: Safari page text via `do JavaScript` (most faithful for web pages).
//! Tier 2: AX tree text walk of the frontmost app (native apps).
//! Tier 3: Apple Vision OCR of the focused window (canvas/Electron-style UIs
//!         whose AX tree is empty).

use super::macos_observer::{is_safari_app, MacObserver};
use super::safari_dom;
use super::types::FocusedApp;
use super::vision::XcapWindowCapturer;
use std::io::Cursor;

const MIN_USEFUL_PAGE_TEXT_CHARS: usize = 200;

unsafe extern "C" {
    fn screenie_ocr_png(png_bytes: *const u8, png_len: usize) -> *const std::os::raw::c_char;
    fn screenie_free_string(ptr: *const std::os::raw::c_char);
}

pub(crate) fn read_focused_page_text(
    observer: &MacObserver,
    focused: &FocusedApp,
) -> Result<String, String> {
    let mut errors: Vec<String> = Vec::new();
    let mut best: Option<String> = None;

    if is_safari_app(focused) {
        match safari_dom::read_safari_page_text() {
            Ok(text) if text.chars().count() >= MIN_USEFUL_PAGE_TEXT_CHARS => return Ok(text),
            Ok(text) if !text.trim().is_empty() => best = Some(text),
            Ok(_) => {}
            Err(err) => errors.push(err.to_string()),
        }
    }

    match observer.read_static_text() {
        Ok(text) if text.chars().count() >= MIN_USEFUL_PAGE_TEXT_CHARS => return Ok(text),
        Ok(text) if !text.trim().is_empty() => {
            if best
                .as_ref()
                .map(|current| text.len() > current.len())
                .unwrap_or(true)
            {
                best = Some(text);
            }
        }
        Ok(_) => {}
        Err(err) => errors.push(err.to_string()),
    }

    match ocr_focused_window(focused) {
        Ok(text) if text.chars().count() >= MIN_USEFUL_PAGE_TEXT_CHARS => return Ok(text),
        Ok(text) if !text.trim().is_empty() => {
            if best
                .as_ref()
                .map(|current| text.len() > current.len())
                .unwrap_or(true)
            {
                best = Some(text);
            }
        }
        Ok(_) => {}
        Err(err) => errors.push(err),
    }

    if let Some(text) = best {
        return Ok(text);
    }
    if errors.is_empty() {
        Err("no readable text found in the focused window".into())
    } else {
        Err(errors.join("; "))
    }
}

fn ocr_focused_window(focused: &FocusedApp) -> Result<String, String> {
    use super::vision::VisionWindowCapturer;

    let capturer = XcapWindowCapturer;
    let window = capturer
        .focused_window(focused.pid)
        .map_err(|err| err.to_string())?
        .ok_or_else(|| "no focused window available for OCR".to_string())?;
    let image = capturer
        .capture_window(&window)
        .map_err(|err| err.to_string())?;

    let mut png_bytes = Vec::new();
    image::DynamicImage::ImageRgba8(image)
        .write_to(&mut Cursor::new(&mut png_bytes), image::ImageFormat::Png)
        .map_err(|err| format!("encode window capture: {err}"))?;

    let ptr = unsafe { screenie_ocr_png(png_bytes.as_ptr(), png_bytes.len()) };
    if ptr.is_null() {
        return Err("Vision OCR failed on the focused window capture".into());
    }
    // SAFETY: the C side guarantees a NUL-terminated UTF-8 string when the
    // pointer is non-null; copy it out and free the original.
    let text = unsafe { std::ffi::CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned();
    unsafe { screenie_free_string(ptr) };
    Ok(text)
}
