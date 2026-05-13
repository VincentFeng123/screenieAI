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

use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use tauri::{AppHandle, Emitter, WebviewWindow};

use core::ffi::c_void;

use windows::core::{w, PCWSTR};
use windows::Win32::Foundation::{BOOL, HINSTANCE, HWND, LPARAM, LRESULT, POINT, RECT, TRUE, WPARAM};
use windows::Win32::Graphics::Dwm::{
    DwmExtendFrameIntoClientArea, DwmSetWindowAttribute, DWMSBT_TRANSIENTWINDOW,
    DWMWA_SYSTEMBACKDROP_TYPE, DWMWA_USE_IMMERSIVE_DARK_MODE,
};
use windows::Win32::Graphics::Gdi::{
    CreateCompatibleBitmap, CreateCompatibleDC, DeleteDC, DeleteObject, GetDC, GetDIBits,
    GetMonitorInfoW, MonitorFromWindow, ReleaseDC, ScreenToClient, SelectObject, SetBrushOrgEx,
    SetStretchBltMode, StretchBlt, BITMAPINFO, BITMAPINFOHEADER, DIB_RGB_COLORS, HALFTONE,
    HBITMAP, HDC, MONITORINFO, MONITOR_DEFAULTTONEAREST, RGBQUAD, SRCCOPY,
};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::Controls::MARGINS;
use windows::Win32::UI::Input::KeyboardAndMouse::{
    GetAsyncKeyState, SendInput, INPUT, INPUT_0, INPUT_KEYBOARD, INPUT_MOUSE, KEYBDINPUT,
    KEYBD_EVENT_FLAGS, KEYEVENTF_KEYUP, MOUSEEVENTF_HWHEEL, MOUSEEVENTF_LEFTDOWN,
    MOUSEEVENTF_LEFTUP, MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP, MOUSEEVENTF_RIGHTDOWN,
    MOUSEEVENTF_RIGHTUP, MOUSEEVENTF_WHEEL, MOUSEINPUT, VIRTUAL_KEY, VK_CONTROL, VK_ESCAPE,
    VK_LWIN, VK_MENU, VK_RWIN, VK_SHIFT,
};
use windows::Win32::UI::Shell::ShellExecuteW;
use windows::Win32::UI::WindowsAndMessaging::{
    AllowSetForegroundWindow, CallNextHookEx, CallWindowProcW, DefWindowProcW, EnumChildWindows,
    EnumWindows, GetCursorPos, GetForegroundWindow, GetWindowLongPtrW, GetWindowRect,
    GetWindowThreadProcessId, IsWindow, IsWindowVisible, SetForegroundWindow,
    SetWindowDisplayAffinity, SetWindowLongPtrW, SetWindowPos, SetWindowsHookExW, ShowWindow,
    UnhookWindowsHookEx, GWLP_WNDPROC, GWL_EXSTYLE, HHOOK, HTTRANSPARENT, HWND_TOPMOST,
    KBDLLHOOKSTRUCT, LLKHF_INJECTED, MSLLHOOKSTRUCT, SWP_FRAMECHANGED, SWP_NOACTIVATE,
    SWP_NOMOVE, SWP_NOSIZE, SWP_NOZORDER, SW_SHOWNOACTIVATE, SW_SHOWNORMAL, WDA_EXCLUDEFROMCAPTURE,
    WH_KEYBOARD_LL, WH_MOUSE_LL, WM_KEYDOWN, WM_LBUTTONUP, WM_MBUTTONUP, WM_MOUSEMOVE,
    WM_NCHITTEST, WM_RBUTTONUP, WM_SYSKEYDOWN, WNDPROC, WS_EX_LAYERED, WS_EX_NOACTIVATE,
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
/// Raw pointer to the original `WNDPROC` for the overlay HWND, stored after
/// we subclass with `overlay_subclass_wndproc`. The subclass handler forwards
/// every non-`WM_NCHITTEST` message back to this proc via `CallWindowProcW`.
/// Non-zero only while the subclass is installed.
static OLD_WNDPROC: AtomicIsize = AtomicIsize::new(0);
static OVERLAY_VISIBLE: AtomicBool = AtomicBool::new(false);
static TEXT_INPUT_FOCUSED: AtomicBool = AtomicBool::new(false);
static PREVIOUS_FOREGROUND_HWND: AtomicIsize = AtomicIsize::new(0);
/// Set true while a JS-driven drag is in progress. Mirrors macOS's
/// `screenieOverlayMouseCaptureActive`. While set, the WH_MOUSE_LL hook
/// stops toggling `WS_EX_TRANSPARENT` based on cursor position so the
/// drag's mousemove/mouseup events keep flowing to our WebView even when
/// the cursor leaves an interaction region.
static MOUSE_CAPTURE_ACTIVE: AtomicBool = AtomicBool::new(false);
/// Set true while a synthetic click/scroll is being relayed to the
/// underlying app. Mirrors macOS's `screenieOverlayClickRelayActive`.
/// While set, the mouse hook stops re-evaluating passthrough so the
/// SendInput-posted event doesn't race the cursor-driven toggle.
static CLICK_RELAY_ACTIVE: AtomicBool = AtomicBool::new(false);
/// Generation counter so a stale relay's late-cleanup task can't reset
/// state that a newer relay just installed.
static CLICK_RELAY_GENERATION: AtomicU64 = AtomicU64::new(0);
/// Background-change observer state. Mirrors macOS's space/window-list
/// poll timer + `overlay-background-changed` emitter — drives the
/// frosted-backdrop refresh on the React side when other windows move.
static BG_OBSERVER_RUNNING: AtomicBool = AtomicBool::new(false);
static BG_LAST_WINLIST_HASH: AtomicU64 = AtomicU64::new(0);
static BG_LAST_SIGNAL_MS: AtomicU64 = AtomicU64::new(0);

/// 60fps continuous backdrop capture loop. Replaces the window-list-only
/// polling with a downsampled BitBlt + hash-based change detection, so the
/// frosted panels show LIVE desktop content (videos, scrolling, animations
/// in the apps underneath) instead of a stale bitmap that only refreshes
/// when windows move. The loop tick is cheap (~1ms BitBlt + hash); the
/// expensive PNG encode + IPC emit runs only on hash change.
static CONTINUOUS_CAPTURE_RUNNING: AtomicBool = AtomicBool::new(false);

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
///
/// Also marks the window as excluded from screen capture
/// (`WDA_EXCLUDEFROMCAPTURE`). This is what lets `trigger_capture_flow`
/// fire BitBlt immediately on hotkey without first waiting for `Settings`
/// or `Chat` to finish hiding — the desktop snapshot will skip our own
/// pixels regardless of whether the window is still visible. Macs get the
/// same property "for free" via ScreenCaptureKit's exclude-self filter.
pub fn configure_main_window(window: &WebviewWindow) {
    let Some(hwnd) = get_hwnd(window) else {
        return;
    };
    enable_dark_mode(hwnd);
    apply_mica_backdrop(hwnd);
    unsafe {
        let _ = SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE);
    }
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

    // Install the WndProc subclass so WM_NCHITTEST drives click-through
    // synchronously. Subclassing the brand-new HWND (which hasn't started
    // dispatching messages yet — `window.show()` hasn't been called by
    // `show_overlay_window` yet) avoids any race with the message pump.
    // Idempotent: subsequent re-shows of the same HWND just return true.
    install_overlay_wndproc_subclass(hwnd);
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
    let hwnd_to_clear = if let Ok(mut g) = overlay_state().lock() {
        if let Some(s) = g.as_mut() {
            if s.transparent_now {
                s.transparent_now = false;
                Some(s.hwnd_raw)
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };
    if let Some(hwnd_raw) = hwnd_to_clear {
        let hwnd = hwnd_from_raw(hwnd_raw);
        unsafe {
            set_hwnd_transparent_style(hwnd, false);
            set_child_windows_transparent_style(hwnd, false);
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

/// Toggle the JS-driven mouse-capture flag. Mirrors macOS's
/// `screenie_set_overlay_mouse_capture`. `useRectDrag` calls this with
/// `true` on mousedown and `false` on mouseup. While active, the WH_MOUSE_LL
/// hook stops toggling `WS_EX_TRANSPARENT`, so the in-flight drag's
/// mousemove/mouseup events keep flowing to our WebView even after the
/// cursor leaves the original interaction region.
pub fn set_overlay_mouse_capture(active: bool) {
    if active {
        CLICK_RELAY_ACTIVE.store(false, Ordering::Release);
        // Bump generation so any pending click-relay cleanup task
        // doesn't later flip CLICK_RELAY_ACTIVE back to false a second
        // time and steal the drag's receivable state.
        CLICK_RELAY_GENERATION.fetch_add(1, Ordering::AcqRel);
    }
    MOUSE_CAPTURE_ACTIVE.store(active, Ordering::Release);
    if active {
        // Force the overlay receivable now — without this, a drag started
        // while the cursor is inside a region but then drifts into empty
        // overlay space would lose mousemove the moment the hook's
        // cursor-driven toggle kicks in.
        let info = if let Ok(g) = overlay_state().lock() {
            g.as_ref().map(|s| (s.hwnd_raw, s.transparent_now))
        } else {
            None
        };
        if let Some((hwnd_raw, was_transparent)) = info {
            if was_transparent {
                set_transparent_now(hwnd_from_raw(hwnd_raw), false);
            }
        }
    } else {
        // Drag ended — re-evaluate transparency based on cursor position.
        update_passthrough_from_cursor();
    }
}

/// Synthesize a click at the current cursor position so the underlying app
/// receives it. Mirrors macOS's
/// `screenie_relay_overlay_click_at_current_mouse`. JS calls this when a
/// mousedown on the captured rect turned out to be a click (no drag) — the
/// rect is one of our interaction regions so the click landed on our
/// WebView, but the user expected it to pass through to whatever app is
/// behind the rect.
///
/// `button_number` matches `MouseEvent.button`: 0 = left, 1 = middle (mac
/// quirk: 1 = right), 2 = right. The mac side maps 1 → right; we keep that
/// quirk for parity with the JS callers.
pub fn relay_overlay_pointer_click(button_number: i32) -> bool {
    let hwnd_raw = match overlay_state().lock() {
        Ok(g) => match g.as_ref() {
            Some(s) => s.hwnd_raw,
            None => return false,
        },
        Err(_) => return false,
    };
    let hwnd = hwnd_from_raw(hwnd_raw);

    let generation = CLICK_RELAY_GENERATION.fetch_add(1, Ordering::AcqRel) + 1;
    CLICK_RELAY_ACTIVE.store(true, Ordering::Release);
    MOUSE_CAPTURE_ACTIVE.store(false, Ordering::Release);
    set_transparent_now(hwnd, true);

    tauri::async_runtime::spawn(async move {
        // Give the OS one beat to apply the WS_EX_TRANSPARENT change before
        // the synthetic click is hit-tested. Without this, the click can
        // race the style change and land back on our overlay instead of
        // the underlying window. The injected event would still be
        // ignored by our own region toggle because CLICK_RELAY_ACTIVE
        // short-circuits `handle_mouse_move_screen`, but it would also
        // confuse Win32 hit testing.
        tokio::time::sleep(std::time::Duration::from_millis(8)).await;
        unsafe {
            post_synthetic_click(button_number);
        }
        // Hold the relay flag long enough for the synthetic click to be
        // dispatched and for the underlying app to process it (it'll
        // typically activate via WM_MOUSEACTIVATE → MA_ACTIVATE). Then
        // re-evaluate transparency from the cursor's current position.
        tokio::time::sleep(std::time::Duration::from_millis(80)).await;
        if CLICK_RELAY_GENERATION.load(Ordering::Acquire) == generation {
            CLICK_RELAY_ACTIVE.store(false, Ordering::Release);
            update_passthrough_from_cursor();
        }
    });
    true
}

/// Synthesize a wheel event at the current cursor position. Mirrors macOS's
/// `screenie_relay_overlay_scroll`. JS calls this from the rect's onWheel
/// handler so scrolling inside the captured region scrolls the underlying
/// app instead of being eaten by our WebView.
pub fn relay_overlay_wheel(delta_x: f64, delta_y: f64) -> bool {
    let hwnd_raw = match overlay_state().lock() {
        Ok(g) => match g.as_ref() {
            Some(s) => s.hwnd_raw,
            None => return false,
        },
        Err(_) => return false,
    };
    let hwnd = hwnd_from_raw(hwnd_raw);

    let generation = CLICK_RELAY_GENERATION.fetch_add(1, Ordering::AcqRel) + 1;
    CLICK_RELAY_ACTIVE.store(true, Ordering::Release);
    MOUSE_CAPTURE_ACTIVE.store(false, Ordering::Release);
    set_transparent_now(hwnd, true);

    tauri::async_runtime::spawn(async move {
        // Wheel events are less time-sensitive than clicks; a 1ms beat is
        // enough for the style change to settle.
        tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        unsafe {
            post_synthetic_wheel(delta_x, delta_y);
        }
        // Keep the overlay transparent for a short scroll burst so repeated
        // mouse-wheel ticks can reach the underlying app directly instead of
        // bouncing through JS/Rust for every notch.
        tokio::time::sleep(std::time::Duration::from_millis(140)).await;
        if CLICK_RELAY_GENERATION.load(Ordering::Acquire) == generation {
            CLICK_RELAY_ACTIVE.store(false, Ordering::Release);
            update_passthrough_from_cursor();
        }
    });
    true
}

/// Start the periodic background-change observer. Mirrors macOS's
/// `screenie_install_overlay_deactivate_hider` + the `screenie_*_background_*`
/// poll machinery. Hashes the visible top-level window list every 200 ms
/// and emits `overlay-background-changed` to React when it changes — the
/// frontend then refreshes the cached screenshot driving the frosted
/// backdrops behind the panels. Idempotent.
pub fn start_overlay_background_observer() {
    if BG_OBSERVER_RUNNING.swap(true, Ordering::AcqRel) {
        return;
    }
    BG_LAST_WINLIST_HASH.store(0, Ordering::Relaxed);
    BG_LAST_SIGNAL_MS.store(0, Ordering::Relaxed);

    tauri::async_runtime::spawn(async move {
        // Delay the first poll past WebView2 cold-start. The overlay's
        // initial mount + React layout completes well within 500ms; without
        // this gap the first window-list hash competes with the cold-start
        // render for main-thread time and amplifies hotkey lag. macOS uses
        // NSWorkspace observers which fire only on real events (Space
        // change), not a timer, so it doesn't need this delay.
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        if !BG_OBSERVER_RUNNING.load(Ordering::Acquire) {
            return;
        }
        let mut interval =
            tokio::time::interval(std::time::Duration::from_millis(200));
        // First tick fires immediately (tokio default). Skip it so the
        // observer's actual cadence is the configured interval.
        interval.tick().await;
        let mut tick_count: u64 = 0;
        let mut emit_count: u64 = 0;
        while BG_OBSERVER_RUNNING.load(Ordering::Acquire) {
            interval.tick().await;
            if !OVERLAY_VISIBLE.load(Ordering::Relaxed) {
                continue;
            }
            tick_count = tick_count.saturating_add(1);

            let next = unsafe { compute_winlist_hash() };
            if next == 0 {
                continue;
            }
            let last = BG_LAST_WINLIST_HASH.load(Ordering::Relaxed);
            let changed = if last == 0 {
                BG_LAST_WINLIST_HASH.store(next, Ordering::Relaxed);
                false
            } else if next != last {
                BG_LAST_WINLIST_HASH.store(next, Ordering::Relaxed);
                true
            } else {
                false
            };

            if changed {
                let now = current_ms();
                let last_signal = BG_LAST_SIGNAL_MS.load(Ordering::Relaxed);
                // Match the macOS debounce so React's
                // `suppressRefreshEventsUntilRef` window absorbs bursts.
                if now.saturating_sub(last_signal) >= 50 {
                    BG_LAST_SIGNAL_MS.store(now, Ordering::Relaxed);
                    if let Ok(g) = overlay_app().lock() {
                        if let Some(app) = g.as_ref() {
                            let emit_result = app.emit_to(
                                "overlay",
                                "overlay-background-changed",
                                (),
                            );
                            emit_count = emit_count.saturating_add(1);
                            // Log only the first few emits so the dev
                            // console shows the observer is alive without
                            // becoming a firehose. Subsequent emits are
                            // silent.
                            if emit_count <= 5 {
                                match emit_result {
                                    Ok(_) => eprintln!(
                                        "[screenie] bg observer emit #{emit_count} after {tick_count} ticks"
                                    ),
                                    Err(e) => eprintln!(
                                        "[screenie] bg observer emit FAILED: {e}"
                                    ),
                                }
                            }
                        }
                    }
                }
            }
        }
        BG_LAST_WINLIST_HASH.store(0, Ordering::Relaxed);
        BG_LAST_SIGNAL_MS.store(0, Ordering::Relaxed);
    });
}

pub fn stop_overlay_background_observer() {
    BG_OBSERVER_RUNNING.store(false, Ordering::Release);
}

/// Long edge of the downsampled bitmap pushed to React at 60fps. Tuned so
/// the BitBlt + GDI downsample + PNG encode + IPC payload all fit inside
/// the ~16ms frame budget. The bitmap is stretched back to monitor size
/// by the browser, providing implicit blur from bilinear interpolation
/// on top of the CSS `filter: blur()` the React-side panel applies.
const CONTINUOUS_CAPTURE_LONG_EDGE: i32 = 480;
const CONTINUOUS_CAPTURE_INTERVAL_MS: u64 = 16;

/// Start a 60fps tokio loop that captures a downsampled snapshot of the
/// monitor the overlay sits on, hashes it for change detection, and emits
/// `overlay-backdrop-update` with a fresh PNG base64 whenever the desktop
/// content changes. The bitmap div in `Frosted.tsx::WindowsBlurredBackdrop`
/// listens for this event and writes the new src directly to its
/// `style.backgroundImage` — bypassing React reconciliation so the
/// per-frame cost stays in the GDI capture + PNG encode, not in the JS
/// framework.
///
/// Idempotent — a second call is a no-op.
///
/// Why a continuous loop (vs the legacy window-list-hash observer): the
/// window list doesn't change when a video plays inside a static window,
/// when a page scrolls, or when an animation runs. The user reported "the
/// background of the overlay are not blurry (and it should also update
/// constantly and smoothly with the background changes)" — the
/// constantly/smoothly part is what this loop addresses. The old observer
/// also caught only the move-this-window kind of changes; this one catches
/// everything because it hashes actual pixel content.
pub fn start_overlay_continuous_capture(app: &AppHandle) {
    if CONTINUOUS_CAPTURE_RUNNING.swap(true, Ordering::AcqRel) {
        return;
    }
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        // Wait one tick before the first capture so the WebView2 has
        // mounted its initial bitmap from `screen.png_base64`; otherwise
        // a slow first frame could race the live event and show empty
        // panels for a moment.
        tokio::time::sleep(std::time::Duration::from_millis(120)).await;
        if !CONTINUOUS_CAPTURE_RUNNING.load(Ordering::Acquire) {
            return;
        }

        let mut interval = tokio::time::interval(
            std::time::Duration::from_millis(CONTINUOUS_CAPTURE_INTERVAL_MS),
        );
        // tokio's first tick fires immediately; skip it so our cadence is
        // truly CONTINUOUS_CAPTURE_INTERVAL_MS.
        interval.tick().await;
        let mut last_hash: u64 = 0;

        while CONTINUOUS_CAPTURE_RUNNING.load(Ordering::Acquire) {
            interval.tick().await;
            if !OVERLAY_VISIBLE.load(Ordering::Relaxed) {
                continue;
            }

            // GDI work blocks; run it on a blocking task so we never
            // freeze the tokio runtime thread. The captured bitmap is
            // moved into the next blocking task (PNG encode), so no extra
            // copy happens.
            let bitmap_opt = tokio::task::spawn_blocking(|| unsafe {
                capture_downsampled_overlay_backdrop()
            })
            .await
            .ok()
            .flatten();
            let Some(bitmap) = bitmap_opt else {
                continue;
            };

            // Cheap content-change check. FNV-1a on the raw RGBA buffer
            // is ~1ms for a 480-long-edge bitmap and avoids the much more
            // expensive PNG encode when the desktop hasn't changed.
            let h = fnv1a_hash(&bitmap.rgba);
            if h == last_hash {
                continue;
            }
            last_hash = h;

            // Move the bitmap into the encode task — `encode_bitmap_png_b64`
            // takes by value so the ~500KB RGBA buffer isn't cloned just to
            // satisfy the borrow checker.
            let png_b64_opt = tokio::task::spawn_blocking(move || encode_bitmap_png_b64(bitmap))
                .await
                .ok()
                .flatten();
            let Some(png_b64) = png_b64_opt else {
                continue;
            };

            let _ = app.emit_to("overlay", "overlay-backdrop-update", png_b64);
        }
        CONTINUOUS_CAPTURE_RUNNING.store(false, Ordering::Release);
    });
}

/// Stop the continuous backdrop capture loop. Called from
/// `lib.rs::close_overlay_now`. The loop's next iteration sees the flag
/// flip and returns; no awaiting needed.
pub fn stop_overlay_continuous_capture() {
    CONTINUOUS_CAPTURE_RUNNING.store(false, Ordering::Release);
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

// ---------------------------------------------------------------------------
// WndProc subclass — primary click-through mechanism (replaces the older
// `WS_EX_TRANSPARENT` toggle from the WH_MOUSE_LL hook). Windows asks our
// subclassed proc on every `WM_NCHITTEST` whether a given point belongs to
// us; we return `HTTRANSPARENT` for any point outside the React-pushed
// interaction regions, so clicks, hover, and scroll fall through to the
// app underneath. Synchronous, race-free; no cursor poll required.
//
// The mouse hook still exists for two side jobs that don't fit into
// `WM_NCHITTEST`: clearing a stuck `MOUSE_CAPTURE_ACTIVE` on mouse-up, and
// the cursor-poll fallback `set_transparent_now` toggle that runs while
// the subclass takes its first effect after install.
// ---------------------------------------------------------------------------

/// Subclass WndProc for the overlay HWND. Forwards everything except
/// `WM_NCHITTEST` to the original proc via `CallWindowProcW`. The
/// `unsafe extern "system"` signature is what Windows expects.
unsafe extern "system" fn overlay_subclass_wndproc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    if msg == WM_NCHITTEST {
        if let Some(result) = overlay_hit_test(hwnd, lparam) {
            return result;
        }
    }
    let old_proc_raw = OLD_WNDPROC.load(Ordering::Acquire);
    if old_proc_raw == 0 {
        // Subclass uninstalled mid-message (window destroying?) — fall back
        // to the system default so we never hang the message loop.
        return DefWindowProcW(hwnd, msg, wparam, lparam);
    }
    let old_proc: WNDPROC = std::mem::transmute(old_proc_raw);
    CallWindowProcW(old_proc, hwnd, msg, wparam, lparam)
}

/// Decide whether `WM_NCHITTEST` for the given lparam point should pass
/// through to the app underneath. Returns `Some(HTTRANSPARENT)` to declare
/// the overlay non-interactive at that point, `None` to let the original
/// WndProc make the call (which yields `HTCLIENT` for the WebView2 child
/// when the point is over a button / textarea / handle).
///
/// Three short-circuits return `None`:
/// - Overlay hidden (hooks still installed but no real overlay to interact with).
/// - JS is driving a drag (`MOUSE_CAPTURE_ACTIVE`) or relaying a synthetic
///   click (`CLICK_RELAY_ACTIVE`) — those flows want the overlay to keep
///   ownership of the cursor for their gesture.
/// - `passthrough_enabled == false` (e.g., selecting mode, where the user
///   needs to drag a rect anywhere on the screen).
fn overlay_hit_test(hwnd: HWND, lparam: LPARAM) -> Option<LRESULT> {
    if !OVERLAY_VISIBLE.load(Ordering::Relaxed) {
        return None;
    }
    if MOUSE_CAPTURE_ACTIVE.load(Ordering::Relaxed)
        || CLICK_RELAY_ACTIVE.load(Ordering::Relaxed)
    {
        return None;
    }

    let (passthrough_enabled, regions) = match overlay_state().lock() {
        Ok(g) => match g.as_ref() {
            Some(s) => (s.passthrough_enabled, s.regions.clone()),
            None => return None,
        },
        Err(_) => return None,
    };
    if !passthrough_enabled {
        return None;
    }

    // lparam packs the screen-coordinate cursor: low 16 bits = x, high 16 bits
    // = y, BOTH SIGNED 16-bit. Multi-monitor setups place secondary displays
    // at negative screen coords; sign-extending via `as i16 as i32` is the
    // standard Win32 idiom to recover them correctly.
    let raw = lparam.0 as i64;
    let x = (raw & 0xFFFF) as i16 as i32;
    let y = ((raw >> 16) & 0xFFFF) as i16 as i32;
    let mut pt = POINT { x, y };
    unsafe {
        let _ = ScreenToClient(hwnd, &mut pt);
    }
    let cx = pt.x as f64;
    let cy = pt.y as f64;
    let in_region = regions
        .iter()
        .any(|r| cx >= r.x && cx < r.x + r.w && cy >= r.y && cy < r.y + r.h);

    if !in_region {
        // `HTTRANSPARENT` is the standard Win32 sentinel meaning "this
        // window is not at the hit-test point; pass to the next window in
        // z-order." Identical effect to `WS_EX_TRANSPARENT`, but evaluated
        // synchronously per-event so there's no toggle race between the
        // cursor poll and the actual click.
        Some(LRESULT(HTTRANSPARENT as isize))
    } else {
        None
    }
}

/// Subclass the overlay HWND's WndProc. Idempotent — second calls are
/// no-ops. Logs to stderr so dev-mode (`npm run tauri dev`) surfaces
/// whether the subclass took effect.
fn install_overlay_wndproc_subclass(hwnd: HWND) -> bool {
    if OLD_WNDPROC.load(Ordering::Acquire) != 0 {
        return true;
    }
    // Cast through `*const ()` first — Rust's `function_casts_as_integer`
    // lint flags the direct `fn as isize` cast (the lint exists because
    // function pointers don't have a stable numerical relationship to the
    // address space on every target). Going through a pointer makes the
    // intent explicit and silences the warning without changing behavior.
    let new_proc_addr = overlay_subclass_wndproc as *const () as isize;
    let old = unsafe { SetWindowLongPtrW(hwnd, GWLP_WNDPROC, new_proc_addr) };
    if old == 0 {
        eprintln!(
            "[screenie] subclass install: SetWindowLongPtrW returned 0 (no prior proc?)"
        );
        return false;
    }
    OLD_WNDPROC.store(old, Ordering::Release);
    // Force Windows to recompute the window frame so the new proc takes
    // effect immediately. Without `SWP_FRAMECHANGED`, USER32 may briefly
    // dispatch a few hit tests to the cached old proc before catching up.
    unsafe {
        let _ = SetWindowPos(
            hwnd,
            HWND(core::ptr::null_mut()),
            0,
            0,
            0,
            0,
            SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE,
        );
    }
    eprintln!(
        "[screenie] subclass install: WndProc subclass active (click-through via WM_NCHITTEST)"
    );
    true
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
            // Skip events we ourselves injected via SendInput in
            // `forward_ctrl_digit_to_previous_app` — without this guard
            // the synthesized Ctrl+digit press would re-enter the hook,
            // be detected as a forwardable keystroke again, and the
            // second forward would fire SetForegroundWindow on top of
            // the one already in flight.
            let injected = info.flags.0 & LLKHF_INJECTED.0 != 0;
            if !injected {
                if info.vkCode == VK_ESCAPE.0 as u32 {
                    if let Ok(g) = overlay_app().lock() {
                        if let Some(app) = g.as_ref() {
                            let _ = app.emit_to("overlay", "overlay-escape-pressed", ());
                        }
                    }
                    // Consume the event — equivalent of returning NULL from
                    // the macOS CGEventTap so fullscreen Chrome/Safari etc.
                    // never sees the Esc keystroke.
                    return LRESULT(1);
                }
                // Mirror of the macOS local NSEvent monitor's Cmd+digit
                // forwarding: when the user is looking at the result panel
                // (no text input focused) and presses Ctrl+1..0, hand the
                // keystroke to whichever app was frontmost before the
                // overlay opened. Browsers map Ctrl+digit to tab switching;
                // forwarding lets the user flip tabs in Chrome/Edge/Firefox
                // without first dismissing the overlay. Every other shortcut
                // continues to flow normally — no broad keystroke-stealing.
                if try_forward_ctrl_digit(info.vkCode) {
                    return LRESULT(1);
                }
            }
        }
    }
    // CallNextHookEx's first argument has been ignored since Win 95 — docs
    // explicitly say to pass NULL even when you hold the hook handle.
    CallNextHookEx(HHOOK(core::ptr::null_mut()), code, wparam, lparam)
}

/// Returns true when the keystroke matched a forwardable Ctrl+digit and a
/// background task has been spawned to deliver it to the previously-frontmost
/// app — caller should consume the original event by returning `LRESULT(1)`
/// from the hook. Returns false in every other case (wrong key, modifiers
/// wrong, text input focused, no remembered foreground app), and the caller
/// should fall through to `CallNextHookEx` so the keystroke reaches the
/// normal focus target.
unsafe fn try_forward_ctrl_digit(vk_code: u32) -> bool {
    // Top-row digits VK_0..VK_9 are 0x30..0x39 — matches the Mac side which
    // limits to "Cmd alone + 1..9, 0" and deliberately ignores numpad digits
    // since those are usually for entry, not navigation.
    if !(0x30..=0x39).contains(&vk_code) {
        return false;
    }
    if TEXT_INPUT_FOCUSED.load(Ordering::Acquire) {
        return false;
    }
    let target_raw = PREVIOUS_FOREGROUND_HWND.load(Ordering::Acquire);
    if target_raw == 0 {
        return false;
    }
    if !ctrl_alone_pressed() {
        return false;
    }
    let vk = vk_code as u16;
    tauri::async_runtime::spawn(async move {
        // The user's Ctrl-down was processed by the OS before we got here;
        // a 2ms beat lets the keyboard input queue settle so the target's
        // GetKeyState/GetAsyncKeyState read the held Ctrl when the
        // synthesized digit lands.
        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        let target = hwnd_from_raw(target_raw);
        unsafe {
            if !IsWindow(target).as_bool() {
                return;
            }
            let mut pid: u32 = 0;
            GetWindowThreadProcessId(target, Some(&mut pid));
            if pid == 0 {
                return;
            }
            // Required when the foreground transition crosses processes:
            // without it, SetForegroundWindow can be silently demoted to a
            // taskbar flash (Microsoft's anti-focus-steal mitigation). The
            // grant is cheap and only valid for one transition.
            let _ = AllowSetForegroundWindow(pid);
            let _ = SetForegroundWindow(target);
        }
        // Give the activation a beat to actually take effect before
        // injecting the keystroke. Empirically 8ms is enough on a warm
        // system; matches the click-relay cushion in `relay_overlay_pointer_click`.
        tokio::time::sleep(std::time::Duration::from_millis(8)).await;
        unsafe {
            inject_ctrl_digit(vk);
        }
    });
    true
}

/// Returns true iff the Ctrl modifier is currently held with no Shift, Alt,
/// or Win modifier alongside it. Mirrors the Mac side's
/// `mods == NSEventModifierFlagCommand` check — combos like Ctrl+Shift+digit
/// are bound to other actions in browsers (move tab, select tab range) and
/// are deliberately not forwarded.
unsafe fn ctrl_alone_pressed() -> bool {
    let pressed = |vk: VIRTUAL_KEY| (GetAsyncKeyState(vk.0 as i32) as u16) & 0x8000 != 0;
    let ctrl = pressed(VK_CONTROL);
    let shift = pressed(VK_SHIFT);
    let alt = pressed(VK_MENU);
    let win = pressed(VK_LWIN) || pressed(VK_RWIN);
    ctrl && !shift && !alt && !win
}

/// Synthesize a Ctrl+digit press at the system level via `SendInput`.
/// Sequence is Ctrl-down, digit-down, digit-up, Ctrl-up so the target
/// window's WM_KEYDOWN sees Ctrl held when it processes the digit — that's
/// what TranslateAccelerator (the path Chrome/Edge/Firefox use for Ctrl+1..9
/// tab switching) reads to match the accelerator entry.
///
/// We send Ctrl explicitly even though the user is physically holding it:
/// the user's original Ctrl-down was delivered to whichever window had focus
/// at that moment (often our overlay's WebView, which doesn't propagate it
/// back out of WebKit), and the target app's per-message Ctrl-state isn't
/// guaranteed to reflect physical state across the foreground transition.
unsafe fn inject_ctrl_digit(digit_vk: u16) {
    let make_input = |vk: u16, key_up: bool| INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: INPUT_0 {
            ki: KEYBDINPUT {
                wVk: VIRTUAL_KEY(vk),
                wScan: 0,
                dwFlags: if key_up {
                    KEYEVENTF_KEYUP
                } else {
                    KEYBD_EVENT_FLAGS(0)
                },
                time: 0,
                dwExtraInfo: 0,
            },
        },
    };
    let inputs = [
        make_input(VK_CONTROL.0, false),
        make_input(digit_vk, false),
        make_input(digit_vk, true),
        make_input(VK_CONTROL.0, true),
    ];
    SendInput(&inputs, std::mem::size_of::<INPUT>() as i32);
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
        } else if msg == WM_LBUTTONUP || msg == WM_RBUTTONUP || msg == WM_MBUTTONUP {
            // Safety net for stuck drags. If a JS-driven drag was started
            // (set_overlay_mouse_capture(true)) but the JS mouseup
            // never fired — e.g., the overlay lost focus mid-drag, or
            // the WebView was re-mounted — the MOUSE_CAPTURE_ACTIVE flag
            // would otherwise be stuck true, freezing `WS_EX_TRANSPARENT`
            // at its drag-start value and breaking every subsequent click.
            // Auto-clear on any mouse-up so the next mousemove re-evaluates
            // transparency from cursor position. Cheap: one atomic compare,
            // no lock acquisition inside the hot path.
            if MOUSE_CAPTURE_ACTIVE.load(Ordering::Relaxed) {
                MOUSE_CAPTURE_ACTIVE.store(false, Ordering::Release);
            }
        }
    }
    CallNextHookEx(HHOOK(core::ptr::null_mut()), code, wparam, lparam)
}

fn handle_mouse_move_screen(screen_pt: POINT) {
    // Defer to the JS-driven drag / relay state machines when one of them
    // owns the cursor. Mirrors macOS's `screenieOverlayMouseCaptureActive`
    // / `screenieOverlayClickRelayActive` short-circuits in
    // `screenie_update_overlay_mouse_passthrough`.
    if MOUSE_CAPTURE_ACTIVE.load(Ordering::Relaxed)
        || CLICK_RELAY_ACTIVE.load(Ordering::Relaxed)
    {
        return;
    }
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
        // WebView2 is hosted in child HWNDs beneath Tauri's top-level
        // overlay HWND. Toggling WS_EX_TRANSPARENT only on the parent leaves
        // those child windows eligible for hit-testing, so the overlay still
        // eats clicks even though the parent says it is transparent. Apply
        // the same style to the full child tree whenever passthrough flips.
        set_hwnd_transparent_style(hwnd, transparent);
        set_child_windows_transparent_style(hwnd, transparent);
    }
}

unsafe fn set_child_windows_transparent_style(root: HWND, transparent: bool) {
    struct ChildStyleState {
        transparent: bool,
    }

    unsafe extern "system" fn enum_child(hwnd: HWND, lparam: LPARAM) -> BOOL {
        let state = &*(lparam.0 as *const ChildStyleState);
        set_hwnd_transparent_style(hwnd, state.transparent);
        TRUE
    }

    let state = ChildStyleState { transparent };
    let _ = EnumChildWindows(root, Some(enum_child), LPARAM(&state as *const _ as isize));
}

unsafe fn set_hwnd_transparent_style(hwnd: HWND, transparent: bool) {
    if !IsWindow(hwnd).as_bool() {
        return;
    }
    let ex = GetWindowLongPtrW(hwnd, GWL_EXSTYLE) as u32;
    let new_ex = if transparent {
        ex | WS_EX_TRANSPARENT.0
    } else {
        ex & !WS_EX_TRANSPARENT.0
    };
    if new_ex == ex {
        return;
    }

    SetWindowLongPtrW(hwnd, GWL_EXSTYLE, new_ex as isize);
    // Extended-style changes via SetWindowLongPtrW don't take effect on hit
    // testing until the frame is re-evaluated. This matters for both the
    // top-level overlay and WebView2's child HWNDs.
    let _ = SetWindowPos(
        hwnd,
        HWND(core::ptr::null_mut()),
        0,
        0,
        0,
        0,
        SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE,
    );
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

unsafe fn post_synthetic_click(button_number: i32) {
    // Match the mac mapping: 0 = left, 1 = right, 2 = middle. Anything
    // else falls back to left so unexpected JS button values don't post
    // an unintended right-click.
    let (down_flag, up_flag) = match button_number {
        1 => (MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP),
        2 => (MOUSEEVENTF_MIDDLEDOWN, MOUSEEVENTF_MIDDLEUP),
        _ => (MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP),
    };
    let down = INPUT {
        r#type: INPUT_MOUSE,
        Anonymous: INPUT_0 {
            mi: MOUSEINPUT {
                dx: 0,
                dy: 0,
                mouseData: 0,
                dwFlags: down_flag,
                time: 0,
                dwExtraInfo: 0,
            },
        },
    };
    let up = INPUT {
        r#type: INPUT_MOUSE,
        Anonymous: INPUT_0 {
            mi: MOUSEINPUT {
                dx: 0,
                dy: 0,
                mouseData: 0,
                dwFlags: up_flag,
                time: 0,
                dwExtraInfo: 0,
            },
        },
    };
    let inputs = [down, up];
    SendInput(&inputs, std::mem::size_of::<INPUT>() as i32);
}

unsafe fn post_synthetic_wheel(delta_x: f64, delta_y: f64) {
    // CSS deltaY > 0 = scroll down (content moves up). Windows wheel > 0 =
    // forward = away from user = content scrolls down. Same sign convention,
    // so DON'T negate — opposite of macOS's `CGScrollEventUnitPixel` path
    // which uses an inverted Y sign.
    //
    // Windows wheel units are 1/WHEEL_DELTA notches (WHEEL_DELTA = 120). Pass
    // CSS-pixel delta directly so 120 px of CSS delta produces one notch.
    let mut wheel_y = (-delta_y).round() as i32;
    let mut wheel_x = (-delta_x).round() as i32;
    if wheel_y == 0 && delta_y.abs() > 0.01 {
        wheel_y = if delta_y > 0.0 { -1 } else { 1 };
    }
    if wheel_x == 0 && delta_x.abs() > 0.01 {
        wheel_x = if delta_x > 0.0 { -1 } else { 1 };
    }

    if wheel_y != 0 {
        let input = INPUT {
            r#type: INPUT_MOUSE,
            Anonymous: INPUT_0 {
                mi: MOUSEINPUT {
                    dx: 0,
                    dy: 0,
                    mouseData: wheel_y as u32,
                    dwFlags: MOUSEEVENTF_WHEEL,
                    time: 0,
                    dwExtraInfo: 0,
                },
            },
        };
        SendInput(&[input], std::mem::size_of::<INPUT>() as i32);
    }
    if wheel_x != 0 {
        let input = INPUT {
            r#type: INPUT_MOUSE,
            Anonymous: INPUT_0 {
                mi: MOUSEINPUT {
                    dx: 0,
                    dy: 0,
                    mouseData: wheel_x as u32,
                    dwFlags: MOUSEEVENTF_HWHEEL,
                    time: 0,
                    dwExtraInfo: 0,
                },
            },
        };
        SendInput(&[input], std::mem::size_of::<INPUT>() as i32);
    }
}

/// FNV-1a–style 64-bit hash mix. Same shape as macOS's `screenie_hash_mix`
/// so the conceptual "fingerprint comparison" path matches across platforms.
fn hash_mix(hash: u64, value: u64) -> u64 {
    hash ^ value
        .wrapping_add(0x9e3779b97f4a7c15)
        .wrapping_add(hash << 6)
        .wrapping_add(hash >> 2)
}

fn current_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Enumerate all visible top-level windows owned by other processes and
/// hash their (hwnd, owner pid, rect) tuples. Mirrors macOS's
/// `screenie_overlay_background_window_fingerprint` which uses
/// `CGWindowListCopyWindowInfo`. The hash changes whenever windows move,
/// resize, appear, or disappear behind our overlay; a change triggers a
/// frosted-backdrop refresh on the JS side.
unsafe fn compute_winlist_hash() -> u64 {
    struct WinlistState {
        hash: u64,
        own_pid: u32,
        included: u32,
    }

    unsafe extern "system" fn enum_cb(hwnd: HWND, lparam: LPARAM) -> BOOL {
        let state = &mut *(lparam.0 as *mut WinlistState);
        if !IsWindowVisible(hwnd).as_bool() {
            return TRUE;
        }
        let mut pid: u32 = 0;
        GetWindowThreadProcessId(hwnd, Some(&mut pid));
        if pid == 0 || pid == state.own_pid {
            return TRUE;
        }
        let mut rect = RECT::default();
        if GetWindowRect(hwnd, &mut rect).is_err() {
            return TRUE;
        }
        let w = rect.right - rect.left;
        let h = rect.bottom - rect.top;
        if w <= 1 || h <= 1 {
            return TRUE;
        }
        state.included += 1;
        state.hash = hash_mix(state.hash, hwnd.0 as u64);
        state.hash = hash_mix(state.hash, pid as u64);
        state.hash = hash_mix(state.hash, rect.left as i64 as u64);
        state.hash = hash_mix(state.hash, rect.top as i64 as u64);
        state.hash = hash_mix(state.hash, w as u64);
        state.hash = hash_mix(state.hash, h as u64);
        TRUE
    }

    let mut state = WinlistState {
        hash: 1469598103934665603u64,
        own_pid: std::process::id(),
        included: 0,
    };
    let _ = EnumWindows(
        Some(enum_cb),
        LPARAM(&mut state as *mut _ as isize),
    );
    hash_mix(state.hash, state.included as u64)
}

// ---------------------------------------------------------------------------
// Continuous backdrop capture (60fps live blur)
// ---------------------------------------------------------------------------

struct DownsampledBitmap {
    width: u32,
    height: u32,
    /// Top-down RGBA8. GDI returns BGRA; we swap channels in
    /// `capture_downsampled_overlay_backdrop` so this buffer is ready for
    /// `image::ImageBuffer::<Rgba<u8>, _>::from_raw` without further work.
    rgba: Vec<u8>,
}

/// FNV-1a 64-bit hash of a byte slice. Used to skip the expensive
/// PNG-encode + IPC-emit when the desktop content hasn't changed between
/// 60fps capture ticks. Same hash as macOS's `screenie_hash_downsampled_image`
/// in `macos_window.m` conceptually — different implementation since here
/// we're hashing raw RGBA, not a downsampled CG image.
fn fnv1a_hash(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Capture the monitor under the overlay HWND, downsampled to ~480 long
/// edge so the BitBlt + DIB read fits comfortably in a 16ms frame budget.
/// Uses `HALFTONE` stretch mode for good downsample quality (GDI does the
/// box-filter on our behalf, much cheaper than running it ourselves in
/// software).
///
/// Returns `None` on any GDI failure — the continuous-capture loop just
/// skips that tick. WDA_EXCLUDEFROMCAPTURE on the overlay ensures the
/// captured bitmap doesn't include our own pixels (no feedback frosting).
unsafe fn capture_downsampled_overlay_backdrop() -> Option<DownsampledBitmap> {
    // Find the monitor under the overlay window. Snapshot the hwnd_raw
    // from state with the lock held briefly so we don't keep it during
    // GDI calls.
    let hwnd_raw = match overlay_state().lock().ok().and_then(|g| g.as_ref().map(|s| s.hwnd_raw)) {
        Some(h) if h != 0 => h,
        _ => return None,
    };
    let hwnd = hwnd_from_raw(hwnd_raw);
    if !IsWindow(hwnd).as_bool() {
        return None;
    }

    let monitor = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);
    if monitor.0.is_null() {
        return None;
    }
    // windows-rs 0.58 doesn't derive `Default` for `MONITORINFO`. Initialize
    // explicitly; `RECT::default()` IS derived and zeroes all four edges.
    let mut mon_info = MONITORINFO {
        cbSize: std::mem::size_of::<MONITORINFO>() as u32,
        rcMonitor: RECT::default(),
        rcWork: RECT::default(),
        dwFlags: 0,
    };
    if !GetMonitorInfoW(monitor, &mut mon_info).as_bool() {
        return None;
    }
    let mon_left = mon_info.rcMonitor.left;
    let mon_top = mon_info.rcMonitor.top;
    let mon_w = mon_info.rcMonitor.right - mon_left;
    let mon_h = mon_info.rcMonitor.bottom - mon_top;
    if mon_w <= 0 || mon_h <= 0 {
        return None;
    }

    // Preserve aspect ratio while clamping the long edge to keep the
    // payload small. A 16:9 monitor → 480x270; 16:10 → 480x300; etc.
    let scale = CONTINUOUS_CAPTURE_LONG_EDGE as f64 / mon_w.max(mon_h) as f64;
    let dst_w = ((mon_w as f64 * scale).round() as i32).max(1);
    let dst_h = ((mon_h as f64 * scale).round() as i32).max(1);

    // RAII guards so a mid-function early-return still releases GDI
    // handles — leaking DCs is a fast path to running out of GDI objects
    // on Windows.
    let null_hwnd = HWND(core::ptr::null_mut());
    let screen_dc = GetDC(null_hwnd);
    if screen_dc.is_invalid() {
        return None;
    }
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
        return None;
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

    let bitmap = CreateCompatibleBitmap(screen_dc, dst_w, dst_h);
    if bitmap.is_invalid() {
        return None;
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

    // HALFTONE gives GDI permission to do a box-filter when scaling down.
    // Without it, the default `BLACKONWHITE` mode produces a much rougher
    // nearest-neighbor downsample that visibly aliases. Required pair:
    // `SetBrushOrgEx(0, 0, None)` per Microsoft docs.
    let _ = SetStretchBltMode(mem_dc, HALFTONE);
    let _ = SetBrushOrgEx(mem_dc, 0, 0, None);

    // `StretchBlt` returns `BOOL` in windows-rs 0.58 (NOT a `Result<()>`
    // like `BitBlt` does — inconsistent windows-rs API surface). Treat
    // FALSE as failure.
    let stretched = StretchBlt(
        mem_dc, 0, 0, dst_w, dst_h, screen_dc, mon_left, mon_top, mon_w, mon_h, SRCCOPY,
    );
    if !stretched.as_bool() {
        SelectObject(mem_dc, old);
        return None;
    }

    let mut info = BITMAPINFO {
        bmiHeader: BITMAPINFOHEADER {
            biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
            biWidth: dst_w,
            // Negative height = top-down DIB (row 0 first).
            biHeight: -dst_h,
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

    let stride = (dst_w * 4) as usize;
    let mut buf = vec![0u8; stride * dst_h as usize];

    let rows_read = GetDIBits(
        mem_dc,
        bitmap,
        0,
        dst_h as u32,
        Some(buf.as_mut_ptr() as _),
        &mut info,
        DIB_RGB_COLORS,
    );

    SelectObject(mem_dc, old);

    if rows_read == 0 {
        return None;
    }

    // BGRA → RGBA in place. GDI leaves alpha undefined for opaque desktop
    // pixels; pin it to 0xff so the PNG encoder doesn't produce a fully
    // transparent image (a classic GDI gotcha — `capture/win.rs` has the
    // same fix-up, kept here so the two paths stay symmetric).
    for px in buf.chunks_exact_mut(4) {
        px.swap(0, 2);
        px[3] = 0xff;
    }

    Some(DownsampledBitmap {
        width: dst_w as u32,
        height: dst_h as u32,
        rgba: buf,
    })
}

/// PNG-encode a `DownsampledBitmap` and return its base64. Heavy-ish
/// (5–10ms for a 480-long-edge image on commodity hardware) — that's why
/// the continuous-capture loop hash-checks before calling this.
///
/// Takes the bitmap by value so the RGBA buffer is moved into
/// `ImageBuffer::from_raw` instead of being cloned. At 60fps that single
/// avoidance saves ~30MB/s of allocator churn on a 1080p monitor.
fn encode_bitmap_png_b64(bitmap: DownsampledBitmap) -> Option<String> {
    use base64::{engine::general_purpose::STANDARD, Engine as _};
    let img_buf = image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(
        bitmap.width,
        bitmap.height,
        bitmap.rgba,
    )?;
    let mut out: Vec<u8> = Vec::with_capacity((bitmap.width * bitmap.height) as usize);
    image::DynamicImage::ImageRgba8(img_buf)
        .write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
        .ok()?;
    Some(STANDARD.encode(&out))
}
