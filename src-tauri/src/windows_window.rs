//! Windows-side bridge — counterpart of `macos_window.m`.
//!
//! The Mac bridge is an Objective-C file compiled by the `cc` crate; on
//! Windows we call Win32 directly through the `windows` crate, so this is
//! pure Rust. Features implemented here:
//!
//! - Overlay window styles: `WS_EX_LAYERED | WS_EX_NOACTIVATE |
//!   WS_EX_TOOLWINDOW`. The overlay never steals focus from the underlying
//!   app, never appears in Alt+Tab, and never adds a taskbar entry — the
//!   closest practical analogue of macOS's `NSWindowStyleMaskNonactivatingPanel`.
//!
//! - Topmost ordering via `SetWindowPos(HWND_TOPMOST, …)` — equivalent of
//!   macOS's `NSStatusWindowLevel`.
//!
//! - Partial click-through: a JS-supplied list of interactive rects, plus a
//!   low-level mouse hook (`WH_MOUSE_LL`) that toggles `WS_EX_TRANSPARENT`
//!   based on cursor hit-test. Empty overlay regions pass clicks through to
//!   apps underneath, interactive panels receive them — the same UX the Mac
//!   side gets from `ignoresMouseEvents`.
//!
//! - Esc consumption: a low-level keyboard hook (`WH_KEYBOARD_LL`) returns
//!   `LRESULT(1)` for Esc while the overlay is visible, so fullscreen
//!   Chrome / Safari etc. never sees the dismiss key. The hook emits
//!   `overlay-escape-pressed` to JS first. This is the direct Win32 mirror
//!   of macOS's `CGEventTap` at `kCGSessionEventTap`.
//!
//! - Exclude-self capture via `SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE)`
//!   so live captures of the desktop don't include our overlay's pixels.
//!   Windows-equivalent of ScreenCaptureKit's exclude-self content filter.
//!
//! - Mica backdrop + immersive dark mode on the Settings window via DWM,
//!   on Windows 11. Older Windows ignores the attributes harmlessly.
//!
//! - Frontmost-app snapshot via `GetForegroundWindow()` — Windows counterpart
//!   of `NSWorkspace.frontmostApplication`. Used to track what the user was
//!   doing before the overlay appeared.

#![allow(non_snake_case)]

use std::sync::atomic::{AtomicBool, AtomicIsize, Ordering};
use std::sync::{Mutex, OnceLock};

use tauri::{AppHandle, Emitter, WebviewWindow};

use core::ffi::c_void;

use windows::core::{w, PCWSTR};
use windows::Win32::Foundation::{BOOL, HINSTANCE, HWND, LPARAM, LRESULT, POINT, TRUE, WPARAM};
use windows::Win32::Graphics::Dwm::{
    DwmExtendFrameIntoClientArea, DwmSetWindowAttribute, DWMSBT_TRANSIENTWINDOW,
    DWMWA_SYSTEMBACKDROP_TYPE, DWMWA_USE_IMMERSIVE_DARK_MODE,
};
use windows::Win32::Graphics::Gdi::ScreenToClient;
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::Controls::MARGINS;
use windows::Win32::UI::Input::KeyboardAndMouse::VK_ESCAPE;
use windows::Win32::UI::Shell::ShellExecuteW;
use windows::Win32::UI::WindowsAndMessaging::{
    CallNextHookEx, GetCursorPos, GetForegroundWindow, GetWindowLongPtrW, SetWindowDisplayAffinity,
    SetWindowLongPtrW, SetWindowPos, SetWindowsHookExW, ShowWindow, UnhookWindowsHookEx,
    GWL_EXSTYLE, HHOOK, HWND_TOPMOST, KBDLLHOOKSTRUCT, MSLLHOOKSTRUCT, SWP_NOACTIVATE, SWP_NOMOVE,
    SWP_NOSIZE, SW_SHOWNOACTIVATE, SW_SHOWNORMAL, WDA_EXCLUDEFROMCAPTURE, WH_KEYBOARD_LL,
    WH_MOUSE_LL, WM_KEYDOWN, WM_MOUSEMOVE, WM_SYSKEYDOWN, WS_EX_LAYERED, WS_EX_NOACTIVATE,
    WS_EX_TOOLWINDOW, WS_EX_TRANSPARENT,
};

// `windows-rs 0.58` represents every HANDLE-style type as
// `struct Foo(pub *mut c_void)`. Pointers aren't `Send + Sync`, so we can't
// store them directly in a `static`/`Mutex` — instead we keep the raw bits
// as an `isize` (via `as isize`/`as *mut _`) and rebuild the strongly-typed
// wrapper at the call site.
fn hwnd_from_raw(raw: isize) -> HWND {
    HWND(raw as *mut c_void)
}
fn hwnd_to_raw(h: HWND) -> isize {
    h.0 as isize
}
fn hhook_from_raw(raw: isize) -> HHOOK {
    HHOOK(raw as *mut c_void)
}
fn hhook_to_raw(h: HHOOK) -> isize {
    h.0 as isize
}

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct InteractionRegion {
    x: f64,
    y: f64,
    w: f64,
    h: f64,
}

struct OverlayState {
    /// Raw pointer bits for the overlay HWND. Stored as isize so the
    /// containing `Mutex<Option<OverlayState>>` is `Send` (raw pointers
    /// aren't `Send` by default, but isize is). Reconstruct via
    /// `hwnd_from_raw` at the call site.
    hwnd_raw: isize,
    regions: Vec<InteractionRegion>,
    passthrough_enabled: bool,
    transparent_now: bool,
}

static OVERLAY: OnceLock<Mutex<Option<OverlayState>>> = OnceLock::new();
static OVERLAY_APP: OnceLock<Mutex<Option<AppHandle>>> = OnceLock::new();
static KEYBOARD_HOOK: AtomicIsize = AtomicIsize::new(0);
static MOUSE_HOOK: AtomicIsize = AtomicIsize::new(0);
static OVERLAY_VISIBLE: AtomicBool = AtomicBool::new(false);
static TEXT_INPUT_FOCUSED: AtomicBool = AtomicBool::new(false);
static PREVIOUS_FOREGROUND_HWND: AtomicIsize = AtomicIsize::new(0);

fn overlay_state() -> &'static Mutex<Option<OverlayState>> {
    OVERLAY.get_or_init(|| Mutex::new(None))
}

fn overlay_app() -> &'static Mutex<Option<AppHandle>> {
    OVERLAY_APP.get_or_init(|| Mutex::new(None))
}

// ---------------------------------------------------------------------------
// Public API — names mirror the macOS bridge helpers for symmetry.
// ---------------------------------------------------------------------------

/// Apply Mica system backdrop + immersive dark mode to the main settings
/// window. No-op on Windows 10 (the DWM attributes are silently ignored).
pub fn configure_main_window(window: &WebviewWindow) {
    let Some(hwnd) = get_hwnd(window) else {
        return;
    };
    enable_dark_mode(hwnd);
    apply_mica_backdrop(hwnd);
}

/// Apply the overlay's extended window styles + the screen-capture exclusion
/// flag. Stashes the HWND so the mouse hook can find it for per-region
/// passthrough toggling. Idempotent — safe to call on every show.
pub fn configure_overlay_window(window: &WebviewWindow, app: &AppHandle) {
    let Some(hwnd) = get_hwnd(window) else {
        return;
    };

    unsafe {
        let ex = GetWindowLongPtrW(hwnd, GWL_EXSTYLE) as u32;
        let new_ex = ex | WS_EX_LAYERED.0 | WS_EX_NOACTIVATE.0 | WS_EX_TOOLWINDOW.0;
        // We intentionally do NOT set WS_EX_TRANSPARENT here — the mouse
        // hook toggles it dynamically based on whether the cursor is over
        // an interactive React-rendered region.
        SetWindowLongPtrW(hwnd, GWL_EXSTYLE, new_ex as isize);

        // Make the overlay invisible to every capture API (BitBlt, PrintWindow,
        // DXGI Output Duplication, Graphics Capture). Available since
        // Windows 10 build 19041 — older Windows treats this as a no-op.
        let _ = SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE);
    }

    if let Ok(mut g) = overlay_state().lock() {
        match g.as_mut() {
            // Preserve existing passthrough/region config across re-shows so
            // we don't lose the regions React already pushed for this window.
            Some(s) => {
                s.hwnd_raw = hwnd_to_raw(hwnd);
            }
            None => {
                *g = Some(OverlayState {
                    hwnd_raw: hwnd_to_raw(hwnd),
                    regions: Vec::new(),
                    passthrough_enabled: false,
                    transparent_now: false,
                });
            }
        }
    }

    if let Ok(mut g) = overlay_app().lock() {
        *g = Some(app.clone());
    }
}

/// Bring the overlay to the topmost level without activating it. Counterpart
/// of `[window orderFrontRegardless]` after setting `NSStatusWindowLevel` on
/// macOS.
pub fn order_overlay_window(window: &WebviewWindow) {
    let Some(hwnd) = get_hwnd(window) else {
        return;
    };
    unsafe {
        let _ = ShowWindow(hwnd, SW_SHOWNOACTIVATE);
        let _ = SetWindowPos(
            hwnd,
            HWND_TOPMOST,
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE,
        );
    }
    OVERLAY_VISIBLE.store(true, Ordering::Release);
}

/// Install the low-level keyboard + mouse hooks. Idempotent. Returns true
/// when at least one hook is active.
pub fn install_overlay_escape_monitor() -> bool {
    OVERLAY_VISIBLE.store(true, Ordering::Release);
    let kb = install_keyboard_hook();
    let m = install_mouse_hook();
    kb || m
}

/// Tear down both hooks and mark the overlay invisible. The OverlayState
/// (hwnd + region snapshot) is intentionally kept so the next show can reuse
/// it without re-pushing regions from React.
pub fn uninstall_overlay_escape_monitor() {
    OVERLAY_VISIBLE.store(false, Ordering::Release);
    uninstall_keyboard_hook();
    uninstall_mouse_hook();
    // Clear any leftover transparency state so the next show starts clean.
    if let Ok(mut g) = overlay_state().lock() {
        if let Some(s) = g.as_mut() {
            if s.transparent_now {
                let hwnd = hwnd_from_raw(s.hwnd_raw);
                unsafe {
                    let ex = GetWindowLongPtrW(hwnd, GWL_EXSTYLE) as u32;
                    SetWindowLongPtrW(
                        hwnd,
                        GWL_EXSTYLE,
                        (ex & !WS_EX_TRANSPARENT.0) as isize,
                    );
                }
                s.transparent_now = false;
            }
        }
    }
}

pub fn set_overlay_interaction_regions(
    regions: Vec<(f64, f64, f64, f64)>,
    passthrough_enabled: bool,
) {
    if let Ok(mut g) = overlay_state().lock() {
        if let Some(s) = g.as_mut() {
            s.regions = regions
                .into_iter()
                .filter(|(_, _, w, h)| w.is_finite() && h.is_finite() && *w > 0.5 && *h > 0.5)
                .map(|(x, y, w, h)| InteractionRegion { x, y, w, h })
                .collect();
            s.passthrough_enabled = passthrough_enabled;
        }
    }
    update_passthrough_from_cursor();
}

pub fn clear_overlay_interaction_regions() {
    set_overlay_interaction_regions(Vec::new(), false);
}

pub fn set_overlay_text_input_focused(focused: bool) {
    TEXT_INPUT_FOCUSED.store(focused, Ordering::Release);
}

pub fn remember_previous_app() {
    unsafe {
        let h = GetForegroundWindow();
        PREVIOUS_FOREGROUND_HWND.store(hwnd_to_raw(h), Ordering::Release);
    }
}

pub fn forget_previous_app() {
    PREVIOUS_FOREGROUND_HWND.store(0, Ordering::Release);
    TEXT_INPUT_FOCUSED.store(false, Ordering::Release);
}

/// Closest Windows analogue of macOS's `x-apple.systempreferences:` deep
/// link. Opens the Graphics Capture privacy page on Windows 11 22H2+; older
/// builds silently no-op. The blank-capture banner button uses this.
pub fn open_screen_settings() {
    unsafe {
        let _ = ShellExecuteW(
            HWND(core::ptr::null_mut()),
            w!("open"),
            w!("ms-settings:privacy-graphicscapture"),
            PCWSTR::null(),
            PCWSTR::null(),
            SW_SHOWNORMAL,
        );
    }
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

fn get_hwnd(window: &WebviewWindow) -> Option<HWND> {
    match window.hwnd() {
        // Reconstruct via the inner raw pointer. Tauri uses a different
        // `windows` crate version (0.61) than we do (0.58); both versions
        // wrap the same `*mut c_void`, so we round-trip through the inner
        // pointer to avoid a direct type-level dependency.
        Ok(h) => Some(HWND(h.0 as *mut c_void)),
        Err(e) => {
            eprintln!("[screenie] hwnd lookup failed: {e}");
            None
        }
    }
}

fn install_keyboard_hook() -> bool {
    if KEYBOARD_HOOK.load(Ordering::Acquire) != 0 {
        return true;
    }
    let result = unsafe {
        let h_module = match GetModuleHandleW(PCWSTR::null()) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("[screenie] GetModuleHandleW failed: {e}");
                return false;
            }
        };
        // HMODULE and HINSTANCE are layout-identical (both `struct(isize)`),
        // but explicit construction avoids relying on whichever `impl From`
        // the windows-rs version we resolve to happens to ship.
        let hinst = HINSTANCE(h_module.0);
        SetWindowsHookExW(WH_KEYBOARD_LL, Some(keyboard_hook_proc), hinst, 0)
    };
    match result {
        Ok(h) => {
            KEYBOARD_HOOK.store(hhook_to_raw(h), Ordering::Release);
            true
        }
        Err(e) => {
            eprintln!("[screenie] keyboard hook install failed: {e}");
            false
        }
    }
}

fn uninstall_keyboard_hook() {
    let raw = KEYBOARD_HOOK.swap(0, Ordering::AcqRel);
    if raw != 0 {
        unsafe {
            let _ = UnhookWindowsHookEx(hhook_from_raw(raw));
        }
    }
}

fn install_mouse_hook() -> bool {
    if MOUSE_HOOK.load(Ordering::Acquire) != 0 {
        return true;
    }
    let result = unsafe {
        let h_module = match GetModuleHandleW(PCWSTR::null()) {
            Ok(h) => h,
            Err(_) => return false,
        };
        let hinst = HINSTANCE(h_module.0);
        SetWindowsHookExW(WH_MOUSE_LL, Some(mouse_hook_proc), hinst, 0)
    };
    match result {
        Ok(h) => {
            MOUSE_HOOK.store(hhook_to_raw(h), Ordering::Release);
            true
        }
        Err(e) => {
            eprintln!("[screenie] mouse hook install failed: {e}");
            false
        }
    }
}

fn uninstall_mouse_hook() {
    let raw = MOUSE_HOOK.swap(0, Ordering::AcqRel);
    if raw != 0 {
        unsafe {
            let _ = UnhookWindowsHookEx(hhook_from_raw(raw));
        }
    }
}

unsafe extern "system" fn keyboard_hook_proc(
    code: i32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    // `code < 0` means "pass through immediately" per Microsoft docs.
    if code >= 0 && OVERLAY_VISIBLE.load(Ordering::Relaxed) {
        let msg = wparam.0 as u32;
        if msg == WM_KEYDOWN || msg == WM_SYSKEYDOWN {
            let info = &*(lparam.0 as *const KBDLLHOOKSTRUCT);
            if info.vkCode == VK_ESCAPE.0 as u32 {
                if let Ok(g) = overlay_app().lock() {
                    if let Some(app) = g.as_ref() {
                        let _ = app.emit_to("overlay", "overlay-escape-pressed", ());
                    }
                }
                // Consume the event — equivalent of returning NULL from the
                // macOS CGEventTap so fullscreen Chrome/Safari etc. never
                // sees the Esc keystroke.
                return LRESULT(1);
            }
        }
    }
    // CallNextHookEx's first argument has been ignored since Win 95 — docs
    // explicitly say to pass NULL even when you hold the hook handle.
    CallNextHookEx(HHOOK(core::ptr::null_mut()), code, wparam, lparam)
}

unsafe extern "system" fn mouse_hook_proc(
    code: i32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    if code >= 0 && OVERLAY_VISIBLE.load(Ordering::Relaxed) {
        let msg = wparam.0 as u32;
        if msg == WM_MOUSEMOVE {
            let info = &*(lparam.0 as *const MSLLHOOKSTRUCT);
            handle_mouse_move_screen(info.pt);
        }
    }
    CallNextHookEx(HHOOK(core::ptr::null_mut()), code, wparam, lparam)
}

fn handle_mouse_move_screen(screen_pt: POINT) {
    // Take a snapshot under the lock, then release before doing the
    // SetWindowLongPtrW call. Low-level hooks have a strict timeout
    // (`LowLevelHooksTimeout`, default 300 ms) — if the proc takes longer,
    // Windows silently disables the hook. So keep the locked section tiny.
    let snapshot = {
        let g = match overlay_state().lock() {
            Ok(g) => g,
            Err(_) => return,
        };
        match g.as_ref() {
            Some(s) => (
                s.hwnd_raw,
                s.passthrough_enabled,
                s.regions.clone(),
                s.transparent_now,
            ),
            None => return,
        }
    };
    let (hwnd_raw, passthrough_enabled, regions, currently_transparent) = snapshot;
    let hwnd = hwnd_from_raw(hwnd_raw);

    if !passthrough_enabled {
        if currently_transparent {
            set_transparent_now(hwnd, false);
        }
        return;
    }

    let mut pt = screen_pt;
    unsafe {
        let _ = ScreenToClient(hwnd, &mut pt);
    }
    let cx = pt.x as f64;
    let cy = pt.y as f64;
    let in_region = regions
        .iter()
        .any(|r| cx >= r.x && cx < r.x + r.w && cy >= r.y && cy < r.y + r.h);
    let want_transparent = !in_region;
    if want_transparent != currently_transparent {
        set_transparent_now(hwnd, want_transparent);
    }
}

fn update_passthrough_from_cursor() {
    let mut pt = POINT { x: 0, y: 0 };
    unsafe {
        if GetCursorPos(&mut pt).is_err() {
            return;
        }
    }
    handle_mouse_move_screen(pt);
}

fn set_transparent_now(hwnd: HWND, transparent: bool) {
    if let Ok(mut g) = overlay_state().lock() {
        if let Some(s) = g.as_mut() {
            s.transparent_now = transparent;
        }
    }
    unsafe {
        let ex = GetWindowLongPtrW(hwnd, GWL_EXSTYLE) as u32;
        let new_ex = if transparent {
            ex | WS_EX_TRANSPARENT.0
        } else {
            ex & !WS_EX_TRANSPARENT.0
        };
        SetWindowLongPtrW(hwnd, GWL_EXSTYLE, new_ex as isize);
    }
}

fn apply_mica_backdrop(hwnd: HWND) {
    unsafe {
        // DWMSBT_TRANSIENTWINDOW = "Mica Alt" — the deeper-tinted Mica
        // variant introduced in Windows 11 22H2. Older Win 11 builds get
        // the regular Mica fallback; Windows 10 ignores the attribute.
        let backdrop = DWMSBT_TRANSIENTWINDOW.0;
        let _ = DwmSetWindowAttribute(
            hwnd,
            DWMWA_SYSTEMBACKDROP_TYPE,
            &backdrop as *const _ as *const _,
            std::mem::size_of_val(&backdrop) as u32,
        );
        // Extend the DWM frame into the entire client area so the Mica
        // backdrop shows through the WebView. The WebView is rendered with
        // a transparent background (Tauri's `transparent: true`), so
        // wherever the React CSS doesn't paint, Mica becomes visible.
        let margins = MARGINS {
            cxLeftWidth: -1,
            cxRightWidth: -1,
            cyTopHeight: -1,
            cyBottomHeight: -1,
        };
        let _ = DwmExtendFrameIntoClientArea(hwnd, &margins);
    }
}

fn enable_dark_mode(hwnd: HWND) {
    unsafe {
        // Forces Win 11's dark non-client area (titlebar, etc.) so the
        // light-themed default doesn't peek out around the Mica backdrop.
        let dark: BOOL = TRUE;
        let _ = DwmSetWindowAttribute(
            hwnd,
            DWMWA_USE_IMMERSIVE_DARK_MODE,
            &dark as *const _ as *const _,
            std::mem::size_of_val(&dark) as u32,
        );
    }
}
