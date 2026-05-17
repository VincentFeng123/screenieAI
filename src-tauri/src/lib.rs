mod ai;
mod capture;
mod history;
mod ollama_install;
mod secrets;

#[cfg(target_os = "windows")]
mod windows_window;

use ai::{AiError, AskEvent, AskRequest, CancelFlag, UiMessage};
use capture::{CaptureError, CroppedCapture, ScreenCapture};
use history::{HistoryEntry, HistoryError};
use secrets::SecretError;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use tauri::{
    ipc::Channel,
    menu::{Menu, MenuItem, PredefinedMenuItem},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    utils::config::WindowEffectsConfig,
    window::{Effect as WindowEffect, EffectState as WindowEffectState},
    AppHandle, Emitter, LogicalPosition, LogicalSize, Manager, WebviewUrl, WebviewWindow,
    WebviewWindowBuilder, WindowEvent,
};

// `ActivationPolicy` only exists on macOS in Tauri 2 (an unconditional
// `use` would break the Windows build). Every call site is already inside
// `#[cfg(target_os = "macos")]`, so this import lives behind the same cfg.
#[cfg(target_os = "macos")]
use tauri::ActivationPolicy;

/// Hard cap on the base64 image-payload length accepted by `ask_ai` and
/// `open_chat_window`. A full-screen 5K Retina capture is ~7-20 MiB base64
/// (~25-30 MiB worst-case for photographic content); 32 MiB gives ~60%
/// margin while bounding the worst-case allocation a malicious frontend
/// can force the backend to perform before validation fires.
const MAX_AI_IMAGE_B64_CHARS: usize = 32 * 1024 * 1024;
const MAX_AI_MESSAGES: usize = 32;
const MAX_AI_MESSAGE_CHARS: usize = 24_000;
const MAX_AI_TOTAL_MESSAGE_CHARS: usize = 120_000;
pub(crate) const ALLOWED_OLLAMA_PULL_MODELS: &[&str] = &["llama3.2-vision"];
const HOTKEY_CONFIG_FILE: &str = "hotkeys.json";

/// Acquire a `Mutex` guard while ignoring lock poisoning.
///
/// Every `Mutex` in `AppState` protects a self-consistent leaf value
/// (`HotkeyConfig`, `Option<ChatSeed>`, etc.) — there are no
/// invariant-bound multi-field structures. If a previous lock holder
/// panicked, the data underneath is still valid; only the poison flag
/// was set. Recovering with `into_inner()` lets subsequent commands
/// (and, critically, the global-shortcut dispatcher in `run()`)
/// continue to operate instead of cascading-panicking and silently
/// killing the user's hotkeys for the lifetime of the process.
fn lock_poison_safe<T>(m: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(target_os = "macos")]
extern "C" {
    fn screenie_configure_main_window(window: *mut std::ffi::c_void) -> bool;
    fn screenie_configure_overlay_window(window: *mut std::ffi::c_void) -> bool;
    fn screenie_order_overlay_window(window: *mut std::ffi::c_void) -> bool;
    fn screenie_install_overlay_escape_monitor(callback: extern "C" fn() -> bool) -> bool;
    fn screenie_uninstall_overlay_escape_monitor();
    fn screenie_install_overlay_deactivate_hider(callback: extern "C" fn()) -> bool;
    fn screenie_uninstall_overlay_deactivate_hider();
    fn screenie_set_overlay_interaction_regions(
        window: *mut std::ffi::c_void,
        regions: *const NativeOverlayInteractionRegion,
        count: usize,
        passthrough_enabled: bool,
    ) -> bool;
    fn screenie_set_overlay_mouse_capture(
        window: *mut std::ffi::c_void,
        active: bool,
    ) -> bool;
    fn screenie_relay_overlay_click(
        window: *mut std::ffi::c_void,
        button_number: std::os::raw::c_int,
    ) -> bool;
    fn screenie_relay_overlay_wheel(
        window: *mut std::ffi::c_void,
        delta_x: f64,
        delta_y: f64,
        phase: std::os::raw::c_int,
    ) -> bool;
    fn screenie_window_display_id(window: *mut std::ffi::c_void) -> u32;
    fn screenie_capture_display_png_excluding_self(
        display_id: u32,
        width: usize,
        height: usize,
    ) -> *const std::os::raw::c_char;
    /// Run Apple Vision text recognition on the supplied PNG bytes.
    /// Returns a malloc'd UTF-8 C string the caller must free via
    /// `screenie_free_string`, or NULL on failure.
    fn screenie_ocr_png(png_bytes: *const u8, png_len: usize) -> *const std::os::raw::c_char;
    fn screenie_free_string(ptr: *const std::os::raw::c_char);
    /// Mount/move/remove NSVisualEffectView "frost panes" beneath the
    /// overlay's WKWebView at the supplied viewport rects. The native side
    /// pools the views so repeated calls don't churn the view hierarchy.
    /// See macos_window.m's vibrancy section for the full rationale.
    fn screenie_set_overlay_vibrancy_regions(
        window: *mut std::ffi::c_void,
        regions: *const NativeOverlayVibrancyRegion,
        count: usize,
    ) -> bool;
    fn screenie_clear_overlay_vibrancy_regions();
    /// Snapshot the user's currently-frontmost app so the overlay can
    /// later forward unhandled keystrokes back to it. Called once at the
    /// start of `trigger_capture_flow`, before our own app gains focus.
    fn screenie_remember_previous_app();
    /// Drop the saved previous-app reference and clear the
    /// text-input-focused flag. Called from `close_overlay_now`.
    fn screenie_forget_previous_app();
    /// JS-driven flag: while a text input has focus inside the overlay,
    /// the local NSEvent monitor passes every keystroke through (typing).
    /// When false, unhandled keystrokes are forwarded to the previous
    /// app instead of dying inside our WKWebView.
    fn screenie_set_overlay_text_input_focused(focused: bool);
}

/// Tray-bar icon: just the wink glyph on a transparent background. The
/// bundled PNG is a 512×512 master rendered from `tray-icon.svg` so both
/// platforms have plenty of source pixels for high-DPI taskbars.
///
/// - macOS: AppKit's template-image mode (`icon_as_template(true)` on the
///   tray builder) auto-tints the alpha mask black/white to match the
///   menu bar and downsamples cleanly via Core Graphics for Retina.
/// - Windows: there's no template-image equivalent, so `tray_icon_for_theme`
///   downsamples the master with Lanczos3 to match the system tray icon
///   size and inverts RGB on light theme so the glyph stays visible
///   against a light taskbar.
///
/// The Dock / Settings-window app icon is read straight from the bundled
/// `.icns` / `.ico` by the OS — we deliberately do not override it at
/// runtime so Settings, Onboarding, and the Dock all show the same
/// artwork the bundle ships.
const TRAY_ICON_PNG: &[u8] = include_bytes!("../icons/tray-icon.png");

/// Resolve the target tray icon size for the current system DPI on Windows.
/// We feed the shell an icon at the exact small-icon size so its built-in
/// scaler doesn't have to do any work — the Lanczos3 downsample we apply
/// to the 512×512 master in `tray_icon_for_theme` is always sharper than
/// the shell's bilinear fallback.
///
/// Returns 2× the system small-icon size, clamped to a reasonable range.
/// Doubling gives the shell headroom on multi-monitor setups where the
/// taskbar may be displayed at a different DPI than the one we sampled
/// (the shell's bicubic from a Lanczos3 source still beats bilinear from
/// a raw 512×512).
#[cfg(target_os = "windows")]
fn windows_tray_icon_target_size() -> u32 {
    use windows::Win32::UI::HiDpi::{GetDpiForSystem, GetSystemMetricsForDpi};
    use windows::Win32::UI::WindowsAndMessaging::{GetSystemMetrics, SM_CXSMICON};

    let system_size = unsafe {
        let dpi = GetDpiForSystem();
        let s = GetSystemMetricsForDpi(SM_CXSMICON, dpi);
        if s > 0 {
            s as u32
        } else {
            let fallback = GetSystemMetrics(SM_CXSMICON);
            if fallback > 0 {
                fallback as u32
            } else {
                16
            }
        }
    };
    (system_size.saturating_mul(2)).clamp(32, 128)
}

/// Pick the right tray icon bytes for the current system theme.
///
/// - macOS: hands the bundled 512×512 master to AppKit unchanged. The
///   `icon_as_template(true)` flag set on the tray builder tells AppKit
///   to use only the alpha channel and tint based on the menu bar color,
///   then downsample for the actual menu bar height (typically ~22pt
///   logical, 44px on Retina).
/// - Windows: downsamples the 512×512 master with Lanczos3 to the system
///   tray icon size, then inverts RGB on light theme so the glyph stays
///   visible against a light taskbar. Without the explicit downsample
///   the Win32 tray code path scales the full 512×512 with a bilinear
///   filter that visibly softens the wink's strokes — Lanczos3 from the
///   master keeps them crisp at every DPI level.
///
/// Returns the constructed `tauri::image::Image` owned (lifetime
/// `'static`) so the caller can hand it to `set_icon` without keeping the
/// backing buffer alive separately.
fn tray_icon_for_theme(theme: tauri::Theme) -> Result<tauri::image::Image<'static>, String> {
    #[cfg(target_os = "windows")]
    {
        let master = image::load_from_memory(TRAY_ICON_PNG)
            .map_err(|e| format!("decode tray icon: {e}"))?
            .to_rgba8();
        let target = windows_tray_icon_target_size();
        let mut rgba = if master.width() == target && master.height() == target {
            master
        } else {
            // Lanczos3 is the highest-quality downsampler the `image` crate
            // ships. For an 8× shrink (512 → 64) it preserves the wink's
            // 1-2px-equivalent strokes far better than the OS shell's
            // built-in bilinear, which is what makes the tray glyph look
            // muddy on default Tauri tray icons at low DPI.
            image::imageops::resize(
                &master,
                target,
                target,
                image::imageops::FilterType::Lanczos3,
            )
        };
        if matches!(theme, tauri::Theme::Light) {
            // Light taskbar → need a DARK glyph. Pure white inverts to
            // pure black; anti-aliased edges (partial whites) invert to
            // the corresponding partial darks, preserving the silhouette.
            // Alpha channel untouched so the transparent background stays
            // transparent.
            for pixel in rgba.pixels_mut() {
                pixel[0] = 255 - pixel[0];
                pixel[1] = 255 - pixel[1];
                pixel[2] = 255 - pixel[2];
            }
        }
        let (w, h) = rgba.dimensions();
        Ok(tauri::image::Image::new_owned(rgba.into_raw(), w, h))
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = theme;
        tauri::image::Image::from_bytes(TRAY_ICON_PNG)
            .map_err(|e| format!("decode tray icon: {e}"))
    }
}

#[cfg(target_os = "macos")]
static OVERLAY_ESCAPE_APP: OnceLock<Mutex<Option<AppHandle>>> = OnceLock::new();

/// Set to true while a capture/refresh is mid-flight. The native
/// background-changed callback fires from the visible-window-list poll and
/// the Space-changed observer; hiding the overlay (or the settings window)
/// during a capture would make those signals fire spuriously and re-trigger
/// `refresh_overlay_capture` recursively. Previously we suppressed that by
/// uninstalling the hider entirely, but
/// `screenie_uninstall_overlay_deactivate_hider` also drops all interaction-
/// region / mouse-capture state (it calls
/// `screenie_clear_overlay_mouse_passthrough` internally), so the overlay
/// briefly went fully click-through with no defined regions, and live frost
/// stopped updating any time `show_overlay_after_refresh` short-circuited
/// (e.g. when `overlay_alive` was already false). A small static gate keeps
/// the hider installed but mutes its callback for the refresh window.
#[cfg(target_os = "macos")]
static OVERLAY_REFRESH_IN_FLIGHT: AtomicBool = AtomicBool::new(false);

/// P-C-B3: exported so the ObjC Esc CGEventTap callback in macos_window.m
/// can decide whether to keep consuming Esc while the overlay is briefly
/// hidden during the recapture-refresh dance. Without this gate, the tap
/// short-circuits on `![overlay isVisible]` and Esc leaks through to the
/// app beneath us during the 300-500 ms hide window — fullscreen Safari
/// unfullscreens, slide presentations exit, etc.
#[cfg(target_os = "macos")]
#[no_mangle]
pub extern "C" fn screenie_is_overlay_refresh_in_flight() -> bool {
    OVERLAY_REFRESH_IN_FLIGHT.load(Ordering::Relaxed)
}

#[cfg(target_os = "macos")]
extern "C" fn handle_overlay_escape_pressed() -> bool {
    eprintln!("[screenie] esc native fired");
    let app = OVERLAY_ESCAPE_APP
        .get()
        .and_then(|store| store.lock().ok().and_then(|guard| guard.clone()));
    if let Some(app) = app {
        // Hand the Esc to JS via an event. The JS keydown listener can't
        // reliably fire when the overlay's nonactivating panel isn't the
        // key window (which is the common case after the user has
        // Cmd+Tabbed away), so the native NSEvent monitor is our always-on
        // fallback. We delegate the actual decision — close, drop
        // adjusting → selecting, or collapse the edit toolbar — to the
        // JS handler so we don't have to mirror that state in Rust.
        let _ = app.emit_to("overlay", "overlay-escape-pressed", ());
        return true;
    }
    false
}

#[cfg(target_os = "macos")]
extern "C" fn handle_overlay_background_changed() {
    if OVERLAY_REFRESH_IN_FLIGHT.load(Ordering::Relaxed) {
        return;
    }
    let app = OVERLAY_ESCAPE_APP
        .get()
        .and_then(|store| store.lock().ok().and_then(|guard| guard.clone()));
    if let Some(app) = app {
        let _ = app.emit_to("overlay", "overlay-background-changed", ());
    }
}

#[cfg(target_os = "macos")]
fn remember_overlay_escape_app(app: &AppHandle) {
    let store = OVERLAY_ESCAPE_APP.get_or_init(|| Mutex::new(None));
    if let Ok(mut guard) = store.lock() {
        *guard = Some(app.clone());
    }
}

/// Force the main settings window's NSWindow to extend its WebView behind the
/// title-bar zone, and suppress macOS's default title-bar separator.
///
/// Without `NSWindowStyleMaskFullSizeContentView`, the area above the WebView
/// is filled by `NSColor.windowBackgroundColor` (a medium dark gray), which
/// renders as a faint lighter strip at the top of the window — visibly
/// clipping the top of the scroll track. Tauri's `titleBarStyle: "Overlay"`
/// is supposed to set this bit, but on some macOS versions / Tauri builds the
/// flag isn't applied, so we re-assert it ourselves. `setTitlebarSeparatorStyle:
/// 1` (`.none`) removes any default 1px line / shadow at the bottom of the
/// title-bar zone for good measure.
#[cfg(target_os = "macos")]
fn configure_main_window(window: &tauri::WebviewWindow) {
    let window_clone = window.clone();
    if let Err(e) = window.run_on_main_thread(move || {
        let raw = match window_clone.ns_window() {
            Ok(p) => p,
            Err(err) => {
                eprintln!("[screenie] configure_main_window: ns_window err: {}", err);
                return;
            }
        };
        if raw.is_null() {
            eprintln!("[screenie] configure_main_window: null pointer");
            return;
        }
        let configured = unsafe { screenie_configure_main_window(raw.cast()) };
        if !configured {
            eprintln!("[screenie] configure_main_window: native helper failed");
        }
    }) {
        eprintln!("[screenie] configure_main_window: dispatch failed: {}", e);
    }
}

#[cfg(target_os = "windows")]
fn configure_main_window(window: &tauri::WebviewWindow) {
    // Apply Mica + immersive dark mode on Windows 11. On Win 10 the DWM
    // attributes are silently ignored. This mirrors what
    // `configure_main_window` does on macOS via `screenie_configure_main_window`
    // (which applies NSWindowStyleMaskFullSizeContentView + the corner mask).
    windows_window::configure_main_window(window);
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
fn configure_main_window(_window: &tauri::WebviewWindow) {}

#[derive(Default)]
struct AppState {
    /// The most recently captured full-screen PNG, awaiting pickup by the overlay window.
    pending: Mutex<Option<ScreenCapture>>,
    /// When true, `trigger_capture_flow` hides the onboarding window before
    /// capturing and the overlay's close handler restores it + emits
    /// `tutorial-capture-complete` so the tutorial step can advance. Set by
    /// the `set_tutorial_mode` command from the onboarding tutorial step.
    tutorial_mode: AtomicBool,
    /// Reentry guard for `trigger_capture_flow`. Without this, two concurrent
    /// tasks can race to build an overlay window with the same label, which
    /// raises an Objective-C exception that aborts the process.
    capture_in_progress: AtomicBool,
    /// Monotonic counter incremented at the start of every capture session
    /// AND every overlay close/Settings-open path. The capture flow snapshots
    /// the value at entry and re-checks it after each await; on mismatch it
    /// bails before staging `pending` or re-showing the overlay. Without this,
    /// a fast Esc + hotkey cycle could let the in-flight capture finish AFTER
    /// the close path ran, resurrecting the overlay (and re-staging stale
    /// `pending` for the next capture).
    capture_generation: std::sync::atomic::AtomicU64,
    /// Set when the settings window was visible and had to be hidden
    /// before capture so the overlay can join the active fullscreen Space.
    restore_main_after_overlay: AtomicBool,
    /// Cancellation flag for the OVERLAY's active AI stream. Replaced on each
    /// new overlay `ask_ai` (which cancels any in-flight overlay predecessor)
    /// and tripped by overlay teardown paths (`close_overlay`,
    /// `show_settings_window`, recapture). Per-window slots keep closing the
    /// overlay from cancelling the chat window's in-flight stream — they're
    /// independent product surfaces with independent lifecycles.
    overlay_ai_cancel: Mutex<Option<CancelFlag>>,
    /// Cancellation flag for the CHAT window's active AI stream. Replaced on
    /// each new chat `ask_ai`. Currently never cancelled by anything other
    /// than the next chat ask_ai or an explicit `cancel_ai` invoke from the
    /// chat window — opening or closing the overlay does not affect it.
    chat_ai_cancel: Mutex<Option<CancelFlag>>,
    /// Last shortcut-registration error, persisted so React can query it even
    /// if the startup event fired before the settings/onboarding listener.
    hotkey_error: Mutex<Option<String>>,
    /// Tripped by the "repeat last capture" hotkey just before firing the
    /// capture flow. The overlay frontend reads this on mount and, if set,
    /// pulls the previous rect from `localStorage` and skips selecting.
    repeat_pending: AtomicBool,
    /// True while the overlay is meant to be visible to the user. Cleared the
    /// instant a close path fires (Esc, tray, settings open) so the
    /// blur-triggered `refresh_overlay_capture → show_overlay_after_refresh`
    /// chain doesn't bring a closed overlay back.
    overlay_alive: AtomicBool,
    /// User-configured hotkey accelerators (Tauri shortcut format, e.g.
    /// "CommandOrControl+Shift+KeyA"). The shortcut handler dispatches by
    /// comparing the firing shortcut against the current config.
    hotkeys: Mutex<HotkeyConfig>,
    /// Seed payload handed to a freshly-opened detached chat window so it
    /// can hydrate the same chat thread the user pinned.
    chat_seed: Mutex<Option<ChatSeed>>,
}

#[derive(Clone, serde::Deserialize)]
struct OverlayInteractionRegion {
    x: f64,
    y: f64,
    w: f64,
    h: f64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NativeOverlayInteractionRegion {
    x: f64,
    y: f64,
    w: f64,
    h: f64,
}

/// Frost-pane geometry sent from the React layer once per layout pass.
/// `radius` is the panel's CSS corner radius in CSS pixels (a giant value
/// like 9999 means "fully rounded pill" — the native side clamps to half
/// the shorter side).
#[derive(Clone, serde::Deserialize)]
struct OverlayVibrancyRegion {
    x: f64,
    y: f64,
    w: f64,
    h: f64,
    radius: f64,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct NativeOverlayVibrancyRegion {
    x: f64,
    y: f64,
    w: f64,
    h: f64,
    radius: f64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct HotkeyConfig {
    capture: String,
    repeat: String,
    settings: String,
}

impl Default for HotkeyConfig {
    fn default() -> Self {
        // `CommandOrControl` resolves to Cmd on macOS and Ctrl elsewhere.
        // Repeat uses Alt as the third modifier so it stays a 3-key combo on
        // both platforms (Ctrl+Control on non-Mac would collide).
        Self {
            capture: "CommandOrControl+Shift+KeyA".to_string(),
            repeat: "CommandOrControl+Alt+KeyA".to_string(),
            settings: "CommandOrControl+Shift+Comma".to_string(),
        }
    }
}

fn replace_ai_cancel(state: &AppState, label: &str) -> CancelFlag {
    let next = Arc::new(AtomicBool::new(false));
    let slot = match label {
        "chat" => &state.chat_ai_cancel,
        // Default to the overlay slot for any unknown label so a misrouted
        // call still cancels SOMETHING rather than silently no-op'ing.
        _ => &state.overlay_ai_cancel,
    };
    if let Ok(mut g) = slot.lock() {
        if let Some(prev) = g.replace(next.clone()) {
            prev.store(true, Ordering::SeqCst);
        }
    }
    next
}

/// Cancel the OVERLAY's in-flight stream only. Used by overlay teardown
/// paths (`close_overlay_now`, `show_settings_window`, recapture). The
/// chat-window stream (if any) is intentionally left running because the
/// chat window has its own lifecycle independent of the overlay.
fn cancel_active_ai(state: &AppState) {
    if let Ok(g) = state.overlay_ai_cancel.lock() {
        if let Some(flag) = g.as_ref() {
            flag.store(true, Ordering::SeqCst);
        }
    }
}

/// Cancel the CHAT window's in-flight stream. Currently invoked via the
/// `cancel_ai` IPC command when called from the chat window.
fn cancel_active_chat_ai(state: &AppState) {
    if let Ok(g) = state.chat_ai_cancel.lock() {
        if let Some(flag) = g.as_ref() {
            flag.store(true, Ordering::SeqCst);
        }
    }
}

fn require_window(window: &WebviewWindow, expected: &str) -> Result<(), String> {
    if window.label() == expected {
        Ok(())
    } else {
        Err("command not allowed from this window".into())
    }
}

fn secret_name_for_provider(provider: &str) -> Result<Option<&'static str>, AiError> {
    match provider {
        "anthropic" => Ok(Some("anthropic_api_key")),
        "openai" => Ok(Some("openai_api_key")),
        "gemini" => Ok(Some("gemini_api_key")),
        "ollama" => Ok(None),
        other => Err(AiError::InvalidProvider(other.to_string())),
    }
}

fn provider_default_model(provider: &str) -> Result<&'static str, AiError> {
    match provider {
        "anthropic" => Ok("claude-sonnet-4-6"),
        "openai" => Ok("gpt-4o"),
        "gemini" => Ok("gemini-2.5-flash"),
        "ollama" => Ok("llama3.2-vision"),
        other => Err(AiError::InvalidProvider(other.to_string())),
    }
}

fn load_hotkey_config(app: &AppHandle) -> HotkeyConfig {
    let defaults = HotkeyConfig::default();
    let Ok(dir) = app_data_dir(app) else {
        return defaults;
    };
    let path = dir.join(HOTKEY_CONFIG_FILE);
    let Ok(bytes) = std::fs::read(path) else {
        return defaults;
    };
    let Ok(mut cfg) = serde_json::from_slice::<HotkeyConfig>(&bytes) else {
        return defaults;
    };
    if cfg.capture.trim().is_empty() {
        cfg.capture = defaults.capture;
    }
    if cfg.repeat.trim().is_empty() {
        cfg.repeat = defaults.repeat;
    }
    if cfg.settings.trim().is_empty() {
        cfg.settings = defaults.settings;
    }
    cfg
}

fn save_hotkey_config(app: &AppHandle, cfg: &HotkeyConfig) -> Result<(), String> {
    let dir = app_data_dir(app)?;
    std::fs::create_dir_all(&dir).map_err(|e| format!("create app data dir: {e}"))?;
    let path = dir.join(HOTKEY_CONFIG_FILE);
    let tmp = path.with_extension("json.tmp");
    let json = serde_json::to_vec_pretty(cfg).map_err(|e| format!("serialize hotkeys: {e}"))?;
    std::fs::write(&tmp, json).map_err(|e| format!("write hotkeys: {e}"))?;
    std::fs::rename(&tmp, &path).map_err(|e| format!("save hotkeys: {e}"))?;
    Ok(())
}

fn validate_ai_payload(messages: &[UiMessage], image_b64: &str) -> Result<(), AiError> {
    if image_b64.len() > MAX_AI_IMAGE_B64_CHARS {
        return Err(AiError::RequestTooLarge("image payload too large".into()));
    }
    if messages.len() > MAX_AI_MESSAGES {
        return Err(AiError::RequestTooLarge("too many chat messages".into()));
    }
    let mut total = 0usize;
    for message in messages {
        let len = message.content.chars().count();
        if len > MAX_AI_MESSAGE_CHARS {
            return Err(AiError::RequestTooLarge("message too long".into()));
        }
        total = total.saturating_add(len);
    }
    if total > MAX_AI_TOTAL_MESSAGE_CHARS {
        return Err(AiError::RequestTooLarge("chat history too long".into()));
    }
    Ok(())
}

/// Validate a chat-seed payload synthesized by the overlay before opening
/// the detached chat window. Mirrors `validate_ai_payload` — same image
/// cap, same per-message + total-history caps. Without this,
/// `open_chat_window` stashed unbounded blobs in `AppState.chat_seed`
/// indefinitely (until the chat window pulled them or the process exited).
fn validate_chat_seed_payload(
    png_b64: &str,
    messages_json: &str,
) -> Result<(), String> {
    if png_b64.len() > MAX_AI_IMAGE_B64_CHARS {
        return Err("image payload too large".into());
    }
    let max_json = MAX_AI_TOTAL_MESSAGE_CHARS.saturating_mul(2);
    if messages_json.len() > max_json {
        return Err("chat history too long".into());
    }
    let parsed: Vec<UiMessage> = serde_json::from_str(messages_json)
        .map_err(|e| format!("messages_json invalid: {e}"))?;
    if parsed.len() > MAX_AI_MESSAGES {
        return Err("too many chat messages".into());
    }
    let mut total = 0usize;
    for m in &parsed {
        let len = m.content.chars().count();
        if len > MAX_AI_MESSAGE_CHARS {
            return Err("message too long".into());
        }
        total = total.saturating_add(len);
    }
    if total > MAX_AI_TOTAL_MESSAGE_CHARS {
        return Err("chat history too long".into());
    }
    Ok(())
}

#[tauri::command]
fn take_pending_capture(state: tauri::State<'_, AppState>) -> Option<ScreenCapture> {
    state.pending.lock().ok().and_then(|mut g| g.take())
}

#[tauri::command]
async fn crop_capture(
    src_b64: String,
    x: u32,
    y: u32,
    w: u32,
    h: u32,
) -> Result<CroppedCapture, CaptureError> {
    // `capture::crop_png_b64` is itself an async wrapper that ships the
    // PNG decode + crop + re-encode work to spawn_blocking internally.
    capture::crop_png_b64(src_b64, x, y, w, h).await
}

#[tauri::command]
async fn refresh_overlay_capture(
    app: AppHandle,
    window: WebviewWindow,
) -> Result<ScreenCapture, CaptureError> {
    require_window(&window, "overlay").map_err(CaptureError::Other)?;

    // While the hide-and-recapture dance below is in flight, mute the native
    // background-changed callback so the Space/visible-window-list poll
    // doesn't re-trigger this command recursively. The earlier code achieved
    // this by calling `screenie_uninstall_overlay_deactivate_hider`, but that
    // helper also tears down the entire mouse-passthrough / interaction-region
    // state — leaving live frost frozen and the overlay fully click-through
    // with no regions defined. The static guard below is reset in every exit
    // path via the RAII helper.
    #[cfg(target_os = "macos")]
    let _refresh_guard = {
        struct RefreshInFlightGuard;
        impl Drop for RefreshInFlightGuard {
            fn drop(&mut self) {
                OVERLAY_REFRESH_IN_FLIGHT.store(false, Ordering::Relaxed);
            }
        }
        OVERLAY_REFRESH_IN_FLIGHT.store(true, Ordering::Relaxed);
        RefreshInFlightGuard
    };

    let monitor = window
        .current_monitor()
        .map_err(|e| CaptureError::Other(format!("current monitor: {e}")))?
        .or_else(|| app.primary_monitor().ok().flatten())
        .ok_or_else(|| CaptureError::Other("no monitor found".into()))?;
    let scale = monitor.scale_factor();
    let size = monitor.size();
    let pos = monitor.position();
    let logical_x = (pos.x as f64 / scale).round() as i32;
    let logical_y = (pos.y as f64 / scale).round() as i32;
    let logical_w = (size.width as f64 / scale).round() as i32;
    let logical_h = (size.height as f64 / scale).round() as i32;

    // Hide the overlay window before capturing so screencapture sees the
    // pure underlying screen with no contribution from our own UI. Pairs
    // with `show_overlay_after_refresh`, which the JS calls once it has
    // applied the new bitmap. The frontend orchestrates this whole dance
    // during the user's tab-away window so the hide/show is invisible.
    let was_visible = window.is_visible().unwrap_or(false);
    if was_visible {
        let _ = window.hide();
    }
    tokio::time::sleep(std::time::Duration::from_millis(240)).await;

    let mut result = capture::capture_rect(logical_x, logical_y, logical_w, logical_h).await;

    if let Ok(cap) = result.as_mut() {
        if let Ok(cursor) = app.cursor_position() {
            cap.cursor_x = Some(((cursor.x - pos.x as f64) / scale).clamp(0.0, logical_w as f64));
            cap.cursor_y = Some(((cursor.y - pos.y as f64) / scale).clamp(0.0, logical_h as f64));
        }
    }

    result
}

#[cfg(target_os = "macos")]
fn overlay_display_id_on_main(window: &tauri::WebviewWindow) -> u32 {
    let window_clone = window.clone();
    let (tx, rx) = std::sync::mpsc::channel();
    if window
        .run_on_main_thread(move || {
            let id = match window_clone.ns_window() {
                Ok(raw) if !raw.is_null() => unsafe { screenie_window_display_id(raw.cast()) },
                _ => 0,
            };
            let _ = tx.send(id);
        })
        .is_err()
    {
        return 0;
    }
    rx.recv_timeout(std::time::Duration::from_millis(500))
        .unwrap_or(0)
}

#[tauri::command]
async fn refresh_overlay_backdrop_capture(
    app: AppHandle,
    window: WebviewWindow,
) -> Result<ScreenCapture, CaptureError> {
    require_window(&window, "overlay").map_err(CaptureError::Other)?;

    #[cfg(target_os = "windows")]
    {
        // The overlay window already has `WDA_EXCLUDEFROMCAPTURE` set by
        // `windows_window::configure_overlay_window`, so a vanilla BitBlt
        // of the desktop naturally excludes our own pixels — no need for
        // a separate "exclude self" code path. The whole-screen capture
        // below is the same shape as macOS's SCK path: pick the monitor
        // under the overlay, snapshot it, attach cursor metadata.
        let monitor = window
            .current_monitor()
            .map_err(|e| CaptureError::Other(format!("current monitor: {e}")))?
            .or_else(|| app.primary_monitor().ok().flatten())
            .ok_or_else(|| CaptureError::Other("no monitor found".into()))?;
        let scale = monitor.scale_factor();
        let size = monitor.size();
        let pos = monitor.position();
        let logical_x = (pos.x as f64 / scale).round() as i32;
        let logical_y = (pos.y as f64 / scale).round() as i32;
        let logical_w = (size.width as f64 / scale).round() as i32;
        let logical_h = (size.height as f64 / scale).round() as i32;

        let mut cap = capture::capture_rect(logical_x, logical_y, logical_w, logical_h).await?;
        if let Ok(cursor) = app.cursor_position() {
            cap.cursor_x = Some(((cursor.x - pos.x as f64) / scale).clamp(0.0, logical_w as f64));
            cap.cursor_y = Some(((cursor.y - pos.y as f64) / scale).clamp(0.0, logical_h as f64));
        }
        return Ok(cap);
    }

    #[cfg(target_os = "macos")]
    {
        let monitor = window
            .current_monitor()
            .map_err(|e| CaptureError::Other(format!("current monitor: {e}")))?
            .or_else(|| app.primary_monitor().ok().flatten())
            .ok_or_else(|| CaptureError::Other("no monitor found".into()))?;
        let scale = monitor.scale_factor();
        let size = monitor.size();
        let pos = monitor.position();
        let logical_w = (size.width as f64 / scale).round() as i32;
        let logical_h = (size.height as f64 / scale).round() as i32;
        let display_id = overlay_display_id_on_main(&window);
        let capture_w = size.width as usize;
        let capture_h = size.height as usize;

        let png_base64 = tauri::async_runtime::spawn_blocking(move || {
            let ptr = unsafe {
                screenie_capture_display_png_excluding_self(display_id, capture_w, capture_h)
            };
            if ptr.is_null() {
                return Err(CaptureError::Other(
                    "ScreenCaptureKit backdrop capture failed".into(),
                ));
            }
            let s = unsafe { std::ffi::CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned();
            unsafe { screenie_free_string(ptr) };
            Ok(s)
        })
        .await
        .map_err(|e| CaptureError::Other(format!("capture task join: {e}")))??;

        let mut cap = ScreenCapture {
            png_base64,
            width: size.width,
            height: size.height,
            cursor_x: None,
            cursor_y: None,
            blank: false,
        };
        if let Ok(cursor) = app.cursor_position() {
            cap.cursor_x = Some(((cursor.x - pos.x as f64) / scale).clamp(0.0, logical_w as f64));
            cap.cursor_y = Some(((cursor.y - pos.y as f64) / scale).clamp(0.0, logical_h as f64));
        }
        Ok(cap)
    }

    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        let _ = (app, window);
        Err(CaptureError::Other(
            "ScreenCaptureKit backdrop capture is only available on macOS".into(),
        ))
    }
}

#[tauri::command]
fn show_overlay_after_refresh(app: AppHandle, window: WebviewWindow) -> Result<(), String> {
    require_window(&window, "overlay")?;
    // The frontend can request a show after legacy hide-and-recapture refresh.
    // Hiding the window via `close_overlay` trips
    // both signals, so without this guard the close path would race against
    // an in-flight refresh and bring the overlay right back.
    if let Some(state) = app.try_state::<AppState>() {
        if !state.overlay_alive.load(Ordering::Relaxed) {
            return Ok(());
        }
    }
    show_overlay_window(&app, &window);
    Ok(())
}

#[tauri::command]
async fn ask_ai(
    state: tauri::State<'_, AppState>,
    window: WebviewWindow,
    provider: Option<String>,
    model: Option<String>,
    response_profile: Option<String>,
    messages: Vec<UiMessage>,
    image_b64: String,
    on_chunk: Channel<AskEvent>,
) -> Result<(), AiError> {
    if window.label() != "overlay" && window.label() != "chat" {
        return Err(AiError::Http("command not allowed from this window".into()));
    }
    let provider = provider.unwrap_or_else(|| "anthropic".to_string());
    validate_ai_payload(&messages, &image_b64)?;
    let default_model = provider_default_model(&provider)?;
    let api_key = match secret_name_for_provider(&provider)? {
        Some(name) => secrets::get(name)
            .map_err(|e| AiError::Keyring(e.to_string()))?
            // No key in the keychain — fail FAST with a structured `NoKey`
            // error so the frontend can surface a "no key configured" CTA
            // (Open Settings) rather than letting the cloud provider answer
            // with a raw 401, which is what happens when the user skips the
            // onboarding ApiKeys step (provider defaults to "anthropic" but
            // no key is present).
            .ok_or(AiError::NoKey)?,
        None => String::new(),
    };
    let response_profile = normalize_response_profile(response_profile);
    let chunk_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let chunk_count_inner = chunk_count.clone();
    // Cancel any prior in-flight stream and install a fresh flag for this one.
    // The provider stream functions check this between chunks, so closing the
    // overlay or starting a new ask_ai stops the previous network task instead
    // of letting it drain to the end in the background.
    // Per-window cancel slots: closing the overlay no longer cancels an
    // in-flight chat stream and vice versa.
    let cancel = replace_ai_cancel(&state, window.label());
    let cancel_for_send = cancel.clone();
    let send = |event: AskEvent| {
        if cancel_for_send.load(std::sync::atomic::Ordering::SeqCst) {
            return;
        }
        if matches!(event, AskEvent::Chunk { .. }) {
            chunk_count_inner.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
        if let Err(e) = on_chunk.send(event) {
            eprintln!("[screenie] ask_ai on_chunk.send failed: {:?}", e);
        }
    };
    let mut result = match provider.as_str() {
        "ollama" => {
            let req = AskRequest {
                api_key: String::new(),
                model: model.unwrap_or_else(|| default_model.to_string()),
                response_profile,
                messages,
                image_b64,
            };
            ai::ollama::stream(req, cancel.clone(), send).await
        }
        "openai" => {
            let req = AskRequest {
                api_key,
                model: model.unwrap_or_else(|| default_model.to_string()),
                response_profile,
                messages,
                image_b64,
            };
            ai::openai::stream(req, cancel.clone(), send).await
        }
        "gemini" => {
            let req = AskRequest {
                api_key,
                model: model.unwrap_or_else(|| default_model.to_string()),
                response_profile,
                messages,
                image_b64,
            };
            ai::gemini::stream(req, cancel.clone(), send).await
        }
        "anthropic" => {
            let req = AskRequest {
                api_key,
                model: model.unwrap_or_else(|| default_model.to_string()),
                response_profile,
                messages,
                image_b64,
            };
            ai::anthropic::stream(req, cancel.clone(), send).await
        }
        other => Err(AiError::InvalidProvider(other.to_string())),
    };
    if cancel.load(std::sync::atomic::Ordering::SeqCst) {
        // Cancellation isn't a user-visible error — the overlay either closed
        // or kicked off a new request. Swallow the result silently.
        return Ok(());
    }
    if result.is_ok() && chunk_count.load(std::sync::atomic::Ordering::SeqCst) == 0 {
        result = Err(AiError::EmptyResponse {
            provider: provider.clone(),
        });
    }
    result
}

fn normalize_response_profile(profile: Option<String>) -> String {
    match profile.as_deref() {
        Some("balanced") => "balanced".to_string(),
        Some("detailed") => "detailed".to_string(),
        _ => "concise".to_string(),
    }
}

#[tauri::command]
async fn check_ollama() -> ai::ollama::OllamaStatus {
    ai::ollama::check_status().await
}

#[tauri::command]
async fn install_ollama(
    window: WebviewWindow,
    on_progress: Channel<ollama_install::InstallStatus>,
) -> Result<(), ollama_install::InstallError> {
    require_window(&window, "main").map_err(ollama_install::InstallError::Other)?;
    ollama_install::install(on_progress).await
}

#[tauri::command]
async fn pull_ollama_model(
    window: WebviewWindow,
    model: String,
    on_progress: Channel<ollama_install::PullStatus>,
) -> Result<(), ollama_install::InstallError> {
    require_window(&window, "main").map_err(ollama_install::InstallError::Other)?;
    if !ALLOWED_OLLAMA_PULL_MODELS.contains(&model.as_str()) {
        return Err(ollama_install::InstallError::Other(
            "model pull is not allowed".into(),
        ));
    }
    ollama_install::pull_model(model, on_progress).await
}

#[tauri::command]
fn set_secret(window: WebviewWindow, name: String, value: String) -> Result<(), SecretError> {
    require_window(&window, "main").map_err(|_| SecretError::InvalidName)?;
    secrets::set(&name, &value)
}

#[tauri::command]
fn get_secret(window: WebviewWindow, name: String) -> Result<Option<String>, SecretError> {
    require_window(&window, "main").map_err(|_| SecretError::InvalidName)?;
    secrets::get(&name)
}

#[tauri::command]
fn delete_secret(window: WebviewWindow, name: String) -> Result<(), SecretError> {
    require_window(&window, "main").map_err(|_| SecretError::InvalidName)?;
    secrets::delete(&name)
}

#[tauri::command]
async fn close_overlay(app: AppHandle) {
    close_overlay_now(&app);
}

fn close_overlay_now(app: &AppHandle) {
    if let Some(state) = app.try_state::<AppState>() {
        state.overlay_alive.store(false, Ordering::Relaxed);
        // Bump the capture generation so any in-flight capture task sees
        // its snapshot is now stale and bails before re-showing the overlay.
        state
            .capture_generation
            .fetch_add(1, Ordering::SeqCst);
        cancel_active_ai(&state);
    }
    #[cfg(target_os = "macos")]
    {
        // Clear native hooks/regions before hiding so stale passthrough or
        // Esc interception cannot survive a dismissed overlay.
        if let Some(w) = app.get_webview_window("overlay") {
            clear_overlay_interaction_regions_on_main(&w);
        }
        let _ = app.run_on_main_thread(|| {
            unsafe { screenie_clear_overlay_vibrancy_regions() };
            unsafe { screenie_forget_previous_app() };
            unsafe { screenie_uninstall_overlay_escape_monitor() };
            unsafe { screenie_uninstall_overlay_deactivate_hider() };
        });
    }
    #[cfg(target_os = "windows")]
    {
        // Mirror of the macOS cleanup: pull down the keyboard + mouse hooks,
        // drop any passthrough region state, forget the snapshotted
        // foreground app, and stop the background-change observer. Without
        // these, the low-level hooks would keep intercepting Esc system-wide
        // while the overlay is hidden and the poll task would keep emitting
        // refresh events into the void. Also halt the 60fps backdrop
        // capture loop so it stops doing GDI work for an invisible overlay.
        windows_window::stop_overlay_continuous_capture();
        windows_window::stop_overlay_background_observer();
        windows_window::clear_overlay_interaction_regions();
        windows_window::forget_previous_app();
        windows_window::uninstall_overlay_escape_monitor();
    }
    hide_overlay_window(app);
    finish_overlay_session(app);
}

#[tauri::command]
fn set_overlay_interaction_regions(
    window: WebviewWindow,
    regions: Vec<OverlayInteractionRegion>,
    passthrough_enabled: bool,
) -> Result<(), String> {
    require_window(&window, "overlay")?;
    #[cfg(target_os = "macos")]
    {
        set_overlay_interaction_regions_on_main(&window, regions, passthrough_enabled);
    }
    #[cfg(target_os = "windows")]
    {
        // React drives this from a useLayoutEffect on every render; the
        // Windows side stores the rects and answers WM_NCHITTEST with
        // HTTRANSPARENT outside them — same UX as the macOS
        // `ignoresMouseEvents` toggle, without hover-time style flips.
        let _ = &window;
        let tuples = regions.into_iter().map(|r| (r.x, r.y, r.w, r.h)).collect();
        windows_window::set_overlay_interaction_regions(tuples, passthrough_enabled);
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        let _ = (regions, passthrough_enabled);
    }
    Ok(())
}

#[tauri::command]
fn set_overlay_vibrancy_regions(
    window: WebviewWindow,
    regions: Vec<OverlayVibrancyRegion>,
) -> Result<(), String> {
    require_window(&window, "overlay")?;
    #[cfg(target_os = "macos")]
    {
        set_overlay_vibrancy_regions_on_main(&window, regions);
    }
    #[cfg(not(target_os = "macos"))]
    {
        // Per-panel native vibrancy is a macOS-specific feature (mounting
        // NSVisualEffectView siblings of the WKWebView). On Windows the
        // settings window gets system-level Mica via DWM and the overlay
        // panels use CSS `backdrop-filter` from overlay.css — both render
        // live frosted glass through the transparent WebView, so React
        // doesn't need to push per-region rects here.
        let _ = regions;
    }
    Ok(())
}

#[tauri::command]
fn set_overlay_text_input_focused(
    window: WebviewWindow,
    focused: bool,
) -> Result<(), String> {
    require_window(&window, "overlay")?;
    #[cfg(target_os = "macos")]
    {
        unsafe { screenie_set_overlay_text_input_focused(focused) };
    }
    #[cfg(target_os = "windows")]
    {
        windows_window::set_overlay_text_input_focused(focused);
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        let _ = focused;
    }
    Ok(())
}

#[tauri::command]
fn set_overlay_mouse_capture(window: WebviewWindow, active: bool) -> Result<(), String> {
    require_window(&window, "overlay")?;
    #[cfg(target_os = "macos")]
    {
        set_overlay_mouse_capture_on_main(&window, active);
    }
    #[cfg(target_os = "windows")]
    {
        // Match the macOS path: forward the JS-driven drag state to the
        // native side so WM_NCHITTEST keeps the overlay mouse-active until
        // the drag ends.
        let _ = &window;
        windows_window::set_overlay_mouse_capture(active);
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        let _ = active;
    }
    Ok(())
}

#[tauri::command]
fn relay_overlay_pointer_click(window: WebviewWindow, button: i32) -> Result<(), String> {
    require_window(&window, "overlay")?;
    #[cfg(target_os = "macos")]
    {
        relay_overlay_pointer_click_on_main(&window, button);
    }
    #[cfg(target_os = "windows")]
    {
        // Mirror of the macOS click-relay: temporarily force hit tests to
        // HTTRANSPARENT and SendInput a synthetic click at the cursor so the
        // app underneath receives it (and activates normally via
        // WM_MOUSEACTIVATE).
        let _ = &window;
        windows_window::relay_overlay_pointer_click(button);
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        let _ = button;
    }
    Ok(())
}

#[tauri::command]
fn relay_overlay_wheel(
    window: WebviewWindow,
    delta_x: f64,
    delta_y: f64,
    phase: i32,
) -> Result<(), String> {
    require_window(&window, "overlay")?;
    #[cfg(target_os = "macos")]
    {
        relay_overlay_wheel_on_main(&window, delta_x, delta_y, phase);
    }
    #[cfg(target_os = "windows")]
    {
        let _ = &window;
        // Win32 SendInput has no gesture-phase concept; phase is silently
        // ignored on Windows but accepted at the IPC boundary so the JS
        // caller can use one signature across platforms.
        let _ = phase;
        windows_window::relay_overlay_wheel(delta_x, delta_y);
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        let _ = (delta_x, delta_y, phase);
    }
    Ok(())
}

/// Stop the in-flight AI stream without closing the overlay. Trips the same
/// cancel flag `close_overlay` does — the streaming task in `ask_ai` returns
/// `Ok(())` between chunks, so the JS-side success path commits whatever
/// text accumulated so far as the final assistant message.
#[tauri::command]
fn cancel_ai(app: AppHandle, window: WebviewWindow) -> Result<(), String> {
    let label = window.label();
    if label != "overlay" && label != "chat" {
        return Err("command not allowed from this window".into());
    }
    if let Some(state) = app.try_state::<AppState>() {
        match label {
            "chat" => cancel_active_chat_ai(&state),
            _ => cancel_active_ai(&state),
        }
    }
    Ok(())
}

/// Hide the overlay window. Keeping it alive makes the next capture fast and
/// avoids rebuilding native window state every time.
#[cfg(target_os = "macos")]
fn hide_overlay_window(app: &AppHandle) {
    let app_clone = app.clone();
    if let Err(e) = app.run_on_main_thread(move || {
        if let Some(w) = app_clone.get_webview_window("overlay") {
            let _ = w.hide();
        }
    }) {
        eprintln!("[screenie] hide_overlay_window: dispatch failed: {}", e);
    }
}

#[cfg(not(target_os = "macos"))]
fn hide_overlay_window(app: &AppHandle) {
    // `hide()`, NOT `close()`: closing destroys the WebviewWindow, forcing a
    // full rebuild on every capture (which would also reset the WS_EX_*
    // styles + WDA_EXCLUDEFROMCAPTURE flag that windows_window::configure_overlay_window
    // set up). Hiding keeps the window alive for fast re-show, matching the
    // macOS lifecycle.
    if let Some(w) = app.get_webview_window("overlay") {
        let _ = w.hide();
    }
}

/// Deep-link into the platform's screen-capture privacy pane. macOS opens
/// System Settings → Privacy → Screen Recording; Windows 11 22H2+ opens
/// Settings → Privacy → Graphics capture (`ms-settings:privacy-graphicscapture`).
/// Older platforms / Linux silently no-op. Surfaced from the overlay's
/// permission banner.
#[tauri::command]
async fn open_screen_settings() {
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("/usr/bin/open")
            .arg("x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture")
            .spawn();
    }
    #[cfg(target_os = "windows")]
    {
        windows_window::open_screen_settings();
    }
}

/// Fully quit the app from Settings or the tray menu.
fn exit_app(app: AppHandle) {
    app.exit(0);
}

#[tauri::command]
fn quit_app(app: AppHandle, window: WebviewWindow) -> Result<(), String> {
    require_window(&window, "main")?;
    exit_app(app);
    Ok(())
}

/// Hard-quit and re-launch the process. Used by the Screen Recording
/// permission banner on macOS: TCC caches the screen-recording grant
/// per-process at launch, so a user who flips the toggle while
/// Screenie is running can't see the new state without a fresh
/// process. `AppHandle::restart()` runs the platform-specific cleanup
/// and re-spawns Screenie in one call (returns `!`).
#[tauri::command]
fn restart_app(app: AppHandle, window: WebviewWindow) -> Result<(), String> {
    if window.label() != "overlay" && window.label() != "main" {
        return Err("command not allowed from this window".into());
    }
    app.restart()
}

fn restore_main_window(app: &AppHandle, emit_tutorial_complete: bool) {
    // Patch 21 spawns `finish_overlay_session` (our caller) onto the async
    // runtime, so by the time we get here we are NOT on the main thread.
    // Wrap the AppKit interaction in `run_on_main_thread` so the
    // activation-policy flip + show/focus sequence happen on a single
    // main-loop iteration with no risk of partial reorder under load.
    let app_clone = app.clone();
    let _ = app.run_on_main_thread(move || {
        #[cfg(target_os = "macos")]
        let _ = app_clone.set_activation_policy(ActivationPolicy::Regular);

        if let Some(main) = app_clone.get_webview_window("main") {
            let _ = main.show();
            let _ = main.unminimize();
            let _ = main.set_focus();
            if emit_tutorial_complete {
                let _ = main.emit("tutorial-capture-complete", ());
            }
        }
    });
}

fn finish_overlay_session(app: &AppHandle) {
    if let Some(state) = app.try_state::<AppState>() {
        let tutorial = state.tutorial_mode.swap(false, Ordering::Relaxed);
        let restore_main = state
            .restore_main_after_overlay
            .swap(false, Ordering::Relaxed);
        if tutorial || restore_main {
            restore_main_window(app, tutorial);
        }
    }
}

/// Show + focus the settings window, promoting the app to a regular dock
/// presence so the window appears in Cmd-Tab and gets a Dock icon while it's
/// visible. Used by both the tray "Settings…" menu and by the JS router on
/// first launch (when onboarding is incomplete).
#[tauri::command]
async fn show_settings_window(app: AppHandle) -> Result<(), String> {
    // The overlay is always-on-top at NSStatusWindowLevel — opening Settings
    // while it's visible would put the new window behind it, so the user
    // sees nothing happen. Dismiss the overlay first, mirroring what
    // close_overlay does (cancel any in-flight AI stream + hide).
    if let Some(overlay) = app.get_webview_window("overlay") {
        if overlay.is_visible().unwrap_or(false) {
            if let Some(state) = app.try_state::<AppState>() {
                state.overlay_alive.store(false, Ordering::Relaxed);
                // Same race fix as `close_overlay_now`: invalidate any
                // in-flight capture so it doesn't re-show the overlay
                // after Settings has taken focus.
                state
                    .capture_generation
                    .fetch_add(1, Ordering::SeqCst);
                cancel_active_ai(&state);
            }
            #[cfg(target_os = "macos")]
            {
                let _ = app.run_on_main_thread(|| {
                    unsafe { screenie_uninstall_overlay_deactivate_hider() };
                });
            }
            hide_overlay_window(&app);
            finish_overlay_session(&app);
        }
    }

    #[cfg(target_os = "macos")]
    let _ = app.set_activation_policy(ActivationPolicy::Regular);
    if let Some(w) = app.get_webview_window("main") {
        let _ = w.show();
        let _ = w.unminimize();
        let _ = w.set_focus();
        Ok(())
    } else {
        Err("main window not found".into())
    }
}

/// Hide the settings window and demote back to Accessory (no Dock icon).
/// Triggered when the user closes the window with the red traffic-light
/// button — the close-event handler intercepts and routes here.
#[tauri::command]
async fn hide_settings_window(app: AppHandle) -> Result<(), String> {
    if let Some(w) = app.get_webview_window("main") {
        let _ = w.hide();
    }
    #[cfg(target_os = "macos")]
    let _ = app.set_activation_policy(ActivationPolicy::Accessory);
    Ok(())
}

/// Same effect as `hide_settings_window`, but a distinct entry point keeps
/// the JS call sites readable: "I'm done with onboarding" reads differently
/// than "close settings".
#[tauri::command]
async fn complete_onboarding(app: AppHandle) -> Result<(), String> {
    hide_settings_window(app).await
}

/// Disk-only check for Ollama.app. Distinct from `check_ollama` (which only
/// answers "is the daemon running?") so the onboarding Ollama step can
/// distinguish "not installed" from "installed but not launched".
#[tauri::command]
async fn check_ollama_installed() -> bool {
    ollama_install::is_installed_on_disk()
}

#[tauri::command]
fn get_hotkey_registration_error(state: tauri::State<'_, AppState>) -> Option<String> {
    state.hotkey_error.lock().ok().and_then(|g| g.clone())
}

/// Launch the installed Ollama app/daemon. Used by the onboarding step's
/// "Launch Ollama" CTA when the daemon isn't running yet.
#[tauri::command]
async fn launch_ollama(window: WebviewWindow) -> Result<(), String> {
    require_window(&window, "main")?;
    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("/usr/bin/open")
            .args(["-a", "Ollama"])
            .spawn()
            .map(|_| ())
            .map_err(|e| e.to_string())
    }
    #[cfg(target_os = "windows")]
    {
        // Launch the GUI helper first (which manages the daemon + tray
        // icon); fall back to `ollama.exe serve` if only the CLI is
        // present. Looks in %LOCALAPPDATA%\Programs\Ollama\ then in
        // Program Files\Ollama\ — the two locations the official
        // installer writes to.
        for env_key in ["LOCALAPPDATA", "ProgramFiles", "ProgramW6432"] {
            let Ok(base) = std::env::var(env_key) else {
                continue;
            };
            let dir = if env_key == "LOCALAPPDATA" {
                std::path::PathBuf::from(&base).join("Programs").join("Ollama")
            } else {
                std::path::PathBuf::from(&base).join("Ollama")
            };
            let app = dir.join("ollama app.exe");
            let cli = dir.join("ollama.exe");
            if app.exists() {
                return std::process::Command::new(&app)
                    .spawn()
                    .map(|_| ())
                    .map_err(|e| e.to_string());
            }
            if cli.exists() {
                return std::process::Command::new(&cli)
                    .arg("serve")
                    .spawn()
                    .map(|_| ())
                    .map_err(|e| e.to_string());
            }
        }
        Err("Ollama isn't installed at the expected location.".into())
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        Err("unsupported on this platform".into())
    }
}

/// Toggle interactive-tutorial mode. While active, `trigger_capture_flow`
/// hides the onboarding window before capturing, and the overlay's close
/// handler restores the window + emits `tutorial-capture-complete` to it.
#[tauri::command]
fn set_tutorial_mode(state: tauri::State<'_, AppState>, active: bool) {
    state.tutorial_mode.store(active, Ordering::Relaxed);
}

/// Apply the extra macOS overlay flags through an Objective-C helper that
/// catches `NSException` before control returns to Rust. The helper only
/// adjusts the existing NSWindow's level / Space behavior; it does not convert
/// the window into an NSPanel or install AppKit delegates.
#[cfg(target_os = "macos")]
fn configure_overlay_window_on_main(window: &tauri::WebviewWindow) {
    let raw = match window.ns_window() {
        Ok(p) => p,
        Err(err) => {
            eprintln!("[screenie] configure_overlay_window: ns_window err: {}", err);
            return;
        }
    };
    let configured = unsafe { screenie_configure_overlay_window(raw.cast()) };
    if !configured {
        eprintln!("[screenie] configure_overlay_window: native helper failed");
    }
}

#[cfg(target_os = "macos")]
fn order_overlay_window_on_main(window: &tauri::WebviewWindow) {
    let raw = match window.ns_window() {
        Ok(p) => p,
        Err(err) => {
            eprintln!("[screenie] order_overlay_window: ns_window err: {}", err);
            return;
        }
    };
    let ordered = unsafe { screenie_order_overlay_window(raw.cast()) };
    if !ordered {
        eprintln!("[screenie] order_overlay_window: native helper failed");
    }
}

#[cfg(target_os = "macos")]
fn set_overlay_interaction_regions_on_main(
    window: &tauri::WebviewWindow,
    regions: Vec<OverlayInteractionRegion>,
    passthrough_enabled: bool,
) {
    let window_clone = window.clone();
    let native_regions: Vec<NativeOverlayInteractionRegion> = regions
        .into_iter()
        .filter(|r| {
            r.x.is_finite()
                && r.y.is_finite()
                && r.w.is_finite()
                && r.h.is_finite()
                && r.w > 0.5
                && r.h > 0.5
        })
        .map(|r| NativeOverlayInteractionRegion {
            x: r.x,
            y: r.y,
            w: r.w,
            h: r.h,
        })
        .collect();

    if let Err(e) = window.run_on_main_thread(move || {
        let raw = match window_clone.ns_window() {
            Ok(p) => p,
            Err(err) => {
                eprintln!("[screenie] set_overlay_interaction_regions: ns_window err: {}", err);
                return;
            }
        };
        let ok = unsafe {
            screenie_set_overlay_interaction_regions(
                raw.cast(),
                native_regions.as_ptr(),
                native_regions.len(),
                passthrough_enabled,
            )
        };
        if !ok {
            eprintln!("[screenie] set_overlay_interaction_regions: native helper failed");
        }
    }) {
        eprintln!(
            "[screenie] set_overlay_interaction_regions: dispatch failed: {}",
            e
        );
    }
}

#[cfg(target_os = "macos")]
fn clear_overlay_interaction_regions_on_main(window: &tauri::WebviewWindow) {
    set_overlay_interaction_regions_on_main(window, Vec::new(), false);
}

#[cfg(target_os = "macos")]
fn set_overlay_vibrancy_regions_on_main(
    window: &tauri::WebviewWindow,
    regions: Vec<OverlayVibrancyRegion>,
) {
    let window_clone = window.clone();
    let native_regions: Vec<NativeOverlayVibrancyRegion> = regions
        .into_iter()
        .filter(|r| {
            r.x.is_finite()
                && r.y.is_finite()
                && r.w.is_finite()
                && r.h.is_finite()
                && r.w > 0.5
                && r.h > 0.5
        })
        .map(|r| NativeOverlayVibrancyRegion {
            x: r.x,
            y: r.y,
            w: r.w,
            h: r.h,
            radius: r.radius,
        })
        .collect();

    if let Err(e) = window.run_on_main_thread(move || {
        let raw = match window_clone.ns_window() {
            Ok(p) => p,
            Err(err) => {
                eprintln!(
                    "[screenie] set_overlay_vibrancy_regions: ns_window err: {}",
                    err
                );
                return;
            }
        };
        let ok = unsafe {
            screenie_set_overlay_vibrancy_regions(
                raw.cast(),
                native_regions.as_ptr(),
                native_regions.len(),
            )
        };
        if !ok {
            eprintln!("[screenie] set_overlay_vibrancy_regions: native helper failed");
        }
    }) {
        eprintln!(
            "[screenie] set_overlay_vibrancy_regions: dispatch failed: {}",
            e
        );
    }
}

#[cfg(target_os = "macos")]
fn set_overlay_mouse_capture_on_main(window: &tauri::WebviewWindow, active: bool) {
    let window_clone = window.clone();
    if let Err(e) = window.run_on_main_thread(move || {
        let raw = match window_clone.ns_window() {
            Ok(p) => p,
            Err(err) => {
                eprintln!("[screenie] set_overlay_mouse_capture: ns_window err: {}", err);
                return;
            }
        };
        let ok = unsafe { screenie_set_overlay_mouse_capture(raw.cast(), active) };
        if !ok {
            eprintln!("[screenie] set_overlay_mouse_capture: native helper failed");
        }
    }) {
        eprintln!("[screenie] set_overlay_mouse_capture: dispatch failed: {}", e);
    }
}

#[cfg(target_os = "macos")]
fn relay_overlay_pointer_click_on_main(window: &tauri::WebviewWindow, button: i32) {
    let window_clone = window.clone();
    if let Err(e) = window.run_on_main_thread(move || {
        let raw = match window_clone.ns_window() {
            Ok(p) => p,
            Err(err) => {
                eprintln!("[screenie] relay_overlay_pointer_click: ns_window err: {}", err);
                return;
            }
        };
        let ok = unsafe { screenie_relay_overlay_click(raw.cast(), button) };
        if !ok {
            eprintln!("[screenie] relay_overlay_pointer_click: native helper failed");
        }
    }) {
        eprintln!(
            "[screenie] relay_overlay_pointer_click: dispatch failed: {}",
            e
        );
    }
}

#[cfg(target_os = "macos")]
fn relay_overlay_wheel_on_main(
    window: &tauri::WebviewWindow,
    delta_x: f64,
    delta_y: f64,
    phase: i32,
) {
    let window_clone = window.clone();
    if let Err(e) = window.run_on_main_thread(move || {
        let raw = match window_clone.ns_window() {
            Ok(p) => p,
            Err(err) => {
                eprintln!("[screenie] relay_overlay_wheel: ns_window err: {}", err);
                return;
            }
        };
        let ok = unsafe {
            screenie_relay_overlay_wheel(
                raw.cast(),
                delta_x,
                delta_y,
                phase as std::os::raw::c_int,
            )
        };
        if !ok {
            eprintln!("[screenie] relay_overlay_wheel: native helper failed");
        }
    }) {
        eprintln!("[screenie] relay_overlay_wheel: dispatch failed: {}", e);
    }
}

/// Initial/manual capture path: show/order the overlay without focusing it.
#[cfg(target_os = "macos")]
fn show_overlay_window(app: &AppHandle, window: &tauri::WebviewWindow) {
    remember_overlay_escape_app(app);
    let window_clone = window.clone();
    if let Err(e) = app.run_on_main_thread(move || {
        let monitor_installed =
            unsafe { screenie_install_overlay_escape_monitor(handle_overlay_escape_pressed) };
        if !monitor_installed {
            eprintln!("[screenie] overlay escape monitor install failed");
        }
        configure_overlay_window_on_main(&window_clone);
        if let Ok(raw) = window_clone.ns_window() {
            let _ = unsafe {
                screenie_set_overlay_interaction_regions(
                    raw.cast(),
                    std::ptr::null(),
                    0,
                    false,
                )
            };
        }
        let _ = window_clone.show();
        order_overlay_window_on_main(&window_clone);
        // Keep ordinary app deactivation alone. The native monitor below
        // only refreshes when the background context actually changes, such
        // as a Space switch or horizontal navigation gesture.
        let hider_installed =
            unsafe { screenie_install_overlay_deactivate_hider(handle_overlay_background_changed) };
        if !hider_installed {
            eprintln!("[screenie] overlay deactivate hider install failed");
        }
    }) {
        eprintln!("[screenie] show_overlay_window: dispatch failed: {}", e);
    }
}

#[cfg(target_os = "windows")]
fn show_overlay_window(app: &AppHandle, window: &tauri::WebviewWindow) {
    // Mirror of the macOS `show_overlay_window` setup: configure the window
    // styles + exclude-self capture flag, show it without activating, push it
    // to topmost, start the continuous backdrop stream, and install the
    // low-level keyboard/mouse hooks (Esc consumption + stuck-drag cleanup).
    // Per-region click-through is handled by WM_NCHITTEST subclasses. The
    // old window-list background observer is deliberately
    // not started on Windows: when no-hide capture fails it pushes the
    // frontend into the legacy hide-and-recapture fallback, which reads as
    // repeated overlay flashing. The continuous stream keeps the frosted
    // panels fresh without hiding the window.
    //
    // CRITICAL: low-level hooks (`WH_KEYBOARD_LL` / `WH_MOUSE_LL`) only fire
    // on a thread with a Win32 message loop — Microsoft's docs say the OS
    // posts hook events as messages to the installer thread, which then
    // dispatches them via its message pump. Tokio worker threads have no
    // message loop, so installing the hooks here directly (this function
    // is called from `tauri::async_runtime::spawn`'d tasks) leaves both
    // hooks dormant: Esc never gets consumed and stuck drags never clear.
    // Dispatching to the
    // main UI thread (which Tauri runs the wry/winit message pump on)
    // puts the hooks on a thread that actually pumps. The macOS bridge
    // already follows this same pattern via `app.run_on_main_thread`.
    windows_window::configure_overlay_window(window, app);
    let _ = window.show();
    windows_window::order_overlay_window(window);
    // Continuous-capture loop is a tokio task and doesn't need to run on
    // the message-pump thread, so kick it off here regardless of the
    // main-thread dispatch result below.
    windows_window::start_overlay_continuous_capture(app);
    if let Err(e) = app.run_on_main_thread(|| {
        let _ = windows_window::install_overlay_escape_monitor();
    }) {
        eprintln!(
            "[screenie] show_overlay_window: main-thread dispatch failed: {e}"
        );
        // Fall back to in-place install — hooks won't fire, but the
        // overlay at least appears so the user can dismiss it.
        let _ = windows_window::install_overlay_escape_monitor();
    }
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
fn show_overlay_window(_app: &AppHandle, window: &tauri::WebviewWindow) {
    let _ = window.show();
}

/// Drop-on-exit guard that clears `AppState::capture_in_progress` on every
/// path out of `trigger_capture_flow`, including panics. Without this, a
/// panic anywhere inside `capture_and_show_overlay` (e.g. an Objective-C
/// exception that escapes a native helper, or a tokio panic in the SCK
/// blocking task) would leave the flag stuck true and silently kill every
/// future hotkey press until the user restarted the app. We hold the
/// AppHandle and re-fetch the managed state on drop so the guard owns no
/// borrows tied to the function-scope `tauri::State` wrapper.
struct CaptureInProgressGuard {
    app: AppHandle,
}

impl Drop for CaptureInProgressGuard {
    fn drop(&mut self) {
        if let Some(state) = self.app.try_state::<AppState>() {
            state.capture_in_progress.store(false, Ordering::SeqCst);
        }
    }
}

/// Public entry point. Acquires a reentry guard so a fast double-trigger
/// can't race two async tasks into building two overlay windows with the
/// same label — that race was the source of "Rust cannot catch foreign
/// exceptions" aborts when the second WebviewWindowBuilder hit Cocoa.
async fn trigger_capture_flow(app: AppHandle) {
    // Acquire-and-arm: only build the guard once we win the
    // compare-exchange. If another capture is already in flight, return
    // without arming so we don't accidentally clear someone else's flag.
    let _capture_guard = match app.try_state::<AppState>() {
        Some(state) => {
            if state.capture_in_progress.swap(true, Ordering::SeqCst) {
                eprintln!("[screenie] trigger_capture_flow: already in flight, skipping");
                return;
            }
            Some(CaptureInProgressGuard { app: app.clone() })
        }
        None => None,
    };

    // Snapshot whichever app is frontmost RIGHT NOW (before we hide our
    // settings window or build the overlay panel and steal any focus).
    // The overlay's local NSEvent monitor uses this to forward unhandled
    // keystrokes back to the user's previous app — Cmd+1 → Safari etc.
    // NSWorkspace.frontmostApplication is documented thread-safe, so no
    // dispatch-to-main is needed.
    #[cfg(target_os = "macos")]
    unsafe {
        screenie_remember_previous_app();
    }
    #[cfg(target_os = "windows")]
    {
        windows_window::remember_previous_app();
    }

    capture_and_show_overlay(app.clone()).await;
    // _capture_guard's Drop clears capture_in_progress here.
}

async fn capture_and_show_overlay(app: AppHandle) {
    eprintln!("[screenie] trigger_capture_flow: start");

    // Snapshot the capture generation now and bail out at every await
    // checkpoint if the value moves underneath us — that means the user
    // pressed Esc (or opened Settings) while the capture was in flight,
    // and we must not resurrect the overlay afterward.
    let session_generation = app
        .try_state::<AppState>()
        .map(|s| s.capture_generation.fetch_add(1, Ordering::SeqCst).wrapping_add(1))
        .unwrap_or(0);
    let session_cancelled = |app: &AppHandle| -> bool {
        app.try_state::<AppState>()
            .map(|s| s.capture_generation.load(Ordering::SeqCst) != session_generation)
            .unwrap_or(false)
    };

    // A new capture invalidates whatever the user was looking at — cancel
    // the in-flight stream (if any) so it stops eating bandwidth + tokens.
    // The result-mode chat unmounts when mode resets to "selecting" anyway.
    if let Some(state) = app.try_state::<AppState>() {
        cancel_active_ai(&state);
    }

    // Tutorial mode: briefly hide the onboarding window so the screenshot
    // doesn't include it. The overlay's close handler restores the window
    // and emits `tutorial-capture-complete` so the tutorial step advances.
    let tutorial = app
        .try_state::<AppState>()
        .map(|s| s.tutorial_mode.load(Ordering::Relaxed))
        .unwrap_or(false);
    if tutorial {
        if let Some(w) = app.get_webview_window("main") {
            let _ = w.hide();
        }
        // Brief pause so macOS finishes the hide before screencapture fires.
        // Windows hides synchronously; no settle time needed.
        #[cfg(target_os = "macos")]
        tokio::time::sleep(std::time::Duration::from_millis(150)).await;
        if session_cancelled(&app) {
            eprintln!("[screenie] capture cancelled during tutorial-hide settle");
            finish_overlay_session(&app);
            return;
        }
    } else if let Some(w) = app.get_webview_window("main") {
        if w.is_visible().unwrap_or(false) {
            if let Some(state) = app.try_state::<AppState>() {
                state
                    .restore_main_after_overlay
                    .store(true, Ordering::Relaxed);
            }
            let _ = w.hide();
            #[cfg(target_os = "macos")]
            let _ = app.set_activation_policy(ActivationPolicy::Accessory);
            // Native fullscreen Spaces are especially sensitive to an app's
            // normal window being visible in another Space. Let AppKit finish
            // hiding Settings before we build the overlay. Windows has no
            // Spaces concept and hide is synchronous, so we skip the wait.
            #[cfg(target_os = "macos")]
            tokio::time::sleep(std::time::Duration::from_millis(150)).await;
            if session_cancelled(&app) {
                eprintln!("[screenie] capture cancelled during settings-hide settle");
                finish_overlay_session(&app);
                return;
            }
        }
    }

    // Re-pressing the capture hotkey while the overlay is already on screen
    // would otherwise screenshot it — the dim layer + selection rect would
    // bake into the new background, leaving the user staring at a doubly-
    // dimmed frame. Hide the overlay first, give AppKit a beat to finish the
    // hide, then proceed with the fresh capture. The reentry guard in
    // `trigger_capture_flow` keeps a third press from racing this hide.
    //
    // While we're hidden mid-recapture, clear `overlay_alive` so the
    // blur-triggered `show_overlay_after_refresh` from the frontend doesn't
    // race the new capture and reveal the overlay before the fresh shot is
    // ready. We set it back to true below once `pending` is staged.
    // We deliberately do NOT clear `overlay_alive` here even though the
    // overlay is briefly hidden during a re-capture — that flag is the
    // JS-visible "is the overlay open" signal, and we want it to stay true
    // across the hide+show cycle so `show_overlay_after_refresh` can still
    // fire. The generation gate above handles the cancellation case (the
    // close path bumps the generation; our own hide here does not).
    if let Some(overlay) = app.get_webview_window("overlay") {
        if overlay.is_visible().unwrap_or(false) {
            if let Some(state) = app.try_state::<AppState>() {
                cancel_active_ai(&state);
            }
            let _ = overlay.hide();
            // macOS needs a beat for AppKit to finish the hide before the
            // next screencapture, otherwise the previous overlay's pixels
            // bake into the new background. On Windows the overlay carries
            // `WDA_EXCLUDEFROMCAPTURE`, so BitBlt skips it even while it's
            // still on-screen — no settle needed, and the wait was the
            // single biggest contributor to perceived hotkey lag on
            // re-trigger.
            #[cfg(target_os = "macos")]
            tokio::time::sleep(std::time::Duration::from_millis(180)).await;
            if session_cancelled(&app) {
                eprintln!("[screenie] capture cancelled during recapture-hide settle");
                finish_overlay_session(&app);
                return;
            }
        }
    }

    // P-C-R9: snapshot the cursor position ONCE at trigger time. The same
    // reading is used for both the monitor pick (so we capture the screen
    // the user pointed to) and the cursor metadata embedded in the captured
    // frame (so the overlay's cursor callout lands at the same logical
    // pixel). Re-reading after capture introduced a ~50-150 ms gap during
    // which the cursor could move to a different monitor, and the callout
    // would drift / land off-screen.
    let cursor_snapshot = app.cursor_position().ok();

    // 1. Pick the monitor where the cursor was at trigger time.
    let monitor = match pick_cursor_monitor(&app, cursor_snapshot) {
        Some(m) => m,
        None => {
            eprintln!("[screenie] no monitor found");
            let _ = app.emit("capture-error", "no monitor".to_string());
            finish_overlay_session(&app);
            return;
        }
    };
    let scale = monitor.scale_factor();
    let size = monitor.size();
    let pos = monitor.position();
    let logical_x = (pos.x as f64 / scale).round() as i32;
    let logical_y = (pos.y as f64 / scale).round() as i32;
    let logical_w = (size.width as f64 / scale).round() as i32;
    let logical_h = (size.height as f64 / scale).round() as i32;
    eprintln!(
        "[screenie] target monitor: pos=({},{}) size=({}x{}) scale={} → logical rect {},{},{},{}",
        pos.x, pos.y, size.width, size.height, scale, logical_x, logical_y, logical_w, logical_h
    );

    // 2. Capture just that monitor.
    let mut cap = match capture::capture_rect(logical_x, logical_y, logical_w, logical_h).await {
        Ok(c) => {
            eprintln!(
                "[screenie] capture ok: {}x{} ({} b64 chars)",
                c.width,
                c.height,
                c.png_base64.len()
            );
            c
        }
        Err(e) => {
            eprintln!("[screenie] capture failed: {}", e);
            let _ = app.emit("capture-error", e.to_string());
            finish_overlay_session(&app);
            return;
        }
    };
    if let Some(cursor) = cursor_snapshot {
        cap.cursor_x = Some(((cursor.x - pos.x as f64) / scale).clamp(0.0, logical_w as f64));
        cap.cursor_y = Some(((cursor.y - pos.y as f64) / scale).clamp(0.0, logical_h as f64));
    }

    // Final cancellation gate before we commit `pending` and re-show the
    // overlay. If the user dismissed the session while we were capturing,
    // do nothing — the close path already cleared `overlay_alive`.
    if session_cancelled(&app) {
        eprintln!("[screenie] capture cancelled before stage+show; discarding shot");
        finish_overlay_session(&app);
        return;
    }

    // 3. Stash for the overlay frontend to fetch.
    if let Some(state) = app.try_state::<AppState>() {
        if let Ok(mut g) = state.pending.lock() {
            *g = Some(cap);
        }
        state.overlay_alive.store(true, Ordering::Relaxed);
    }

    // 4. Open (or re-show) the overlay on that monitor.
    //
    // Lifecycle: the overlay is built ONCE on first capture. Subsequent
    // captures reposition the existing window and refresh the frontend.
    // Closing hides the window rather than destroying it, which keeps re-show
    // fast and avoids rebuilding native state.
    if let Some(existing) = app.get_webview_window("overlay") {
        eprintln!("[screenie] overlay exists -> reposition + refresh");
        let _ = existing.set_position(LogicalPosition::<f64>::new(
            logical_x as f64,
            logical_y as f64,
        ));
        let _ = existing.set_size(LogicalSize::<f64>::new(
            logical_w as f64,
            logical_h as f64,
        ));
        show_overlay_window(&app, &existing);
        let _ = existing.emit("overlay-refresh", ());
        let existing_for_retry = existing.clone();
        tauri::async_runtime::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(80)).await;
            let _ = existing_for_retry.emit("overlay-refresh", ());
        });
        return;
    }

    let win = WebviewWindowBuilder::new(
        &app,
        "overlay",
        WebviewUrl::App("index.html?mode=overlay".into()),
    )
    .title("")
    .decorations(false)
    .transparent(true)
    .always_on_top(true)
    .skip_taskbar(true)
    .resizable(false)
    .shadow(false)
    .visible(false)
    .build();

    match win {
        Ok(w) => {
            let _ = w.set_position(LogicalPosition::<f64>::new(
                logical_x as f64,
                logical_y as f64,
            ));
            let _ = w.set_size(LogicalSize::<f64>::new(
                logical_w as f64,
                logical_h as f64,
            ));
            let app_for_event = app.clone();
            w.on_window_event(move |event| match event {
                WindowEvent::Destroyed => {
                    // Window-event callbacks fire on the main loop on macOS,
                    // but `finish_overlay_session` flips the activation
                    // policy and re-shows the main settings window — non-
                    // trivial AppKit work that we don't want running
                    // synchronously inside the event-loop tick. Defer to the
                    // async runtime so this callback returns immediately and
                    // the queued AppKit messages are serviced on the next
                    // loop iteration (`restore_main_window` below then re-
                    // enters the main thread via `run_on_main_thread`).
                    let app_for_task = app_for_event.clone();
                    tauri::async_runtime::spawn(async move {
                        finish_overlay_session(&app_for_task);
                    });
                }
                _ => {}
            });
            show_overlay_window(&app, &w);
            eprintln!("[screenie] overlay window created and shown");
        }
        Err(e) => {
            eprintln!("[screenie] overlay window creation failed: {}", e);
            let _ = app.emit("capture-error", format!("overlay window: {e}"));
        }
    }
}

/// Build the menu-bar tray icon. Left-click triggers the capture flow;
/// right-click pops the Settings/Quit menu via Tauri's default handling.
fn setup_tray(app: &AppHandle) -> tauri::Result<()> {
    let settings_item = MenuItem::with_id(app, "settings", "Settings…", true, None::<&str>)?;
    let separator = PredefinedMenuItem::separator(app)?;
    let quit_item = MenuItem::with_id(app, "quit", "Quit Screenie AI", true, Some("Cmd+Q"))?;
    let menu = Menu::with_items(app, &[&settings_item, &separator, &quit_item])?;

    let mut tray_builder = TrayIconBuilder::with_id("main");
    // Snapshot the system theme so we ship the right glyph from the very
    // first paint. Falls back to Dark when the main window isn't available
    // yet (rare — Tauri builds declared windows before .setup runs — but
    // safer than panicking). Mac path ignores the theme arg internally.
    let theme = app
        .get_webview_window("main")
        .and_then(|w| w.theme().ok())
        .unwrap_or(tauri::Theme::Dark);
    match tray_icon_for_theme(theme) {
        Ok(icon) => {
            tray_builder = tray_builder.icon(icon);
        }
        Err(e) => {
            eprintln!("[screenie] tray icon decode failed ({e}); falling back to bundled app icon");
            if let Some(icon) = app.default_window_icon().cloned() {
                tray_builder = tray_builder.icon(icon);
            } else {
                eprintln!("[screenie] WARNING: no fallback icon available — tray will use the system fallback");
            }
        }
    }
    tray_builder
        // Treat the wink as a template image so macOS uses only its alpha
        // mask and tints to match the menu bar (black on light mode, white
        // on dark). The tray PNG is the wink glyph on a transparent
        // background — distinct from the Dock icon, whose black rounded
        // square would collapse to a solid tinted block under template
        // tinting. On Windows/Linux this flag is a no-op and the white
        // glyph renders as-is.
        .icon_as_template(true)
        .menu(&menu)
        // Critical: left-click triggers capture, NOT the menu. Right-click
        // (or click-and-hold) pops the menu via Tauri's default behaviour.
        .show_menu_on_left_click(false)
        .on_menu_event(|app, event| match event.id.as_ref() {
            "settings" => {
                let h = app.clone();
                tauri::async_runtime::spawn(async move {
                    let _ = show_settings_window(h).await;
                });
            }
            "quit" => {
                exit_app(app.clone());
            }
            _ => {}
        })
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                let app = tray.app_handle().clone();
                tauri::async_runtime::spawn(async move {
                    trigger_capture_flow(app).await;
                });
            }
        })
        .build(app)?;
    Ok(())
}

/// Resolve the OS app-data directory for the current app, e.g.
/// `~/Library/Application Support/com.screenieai.app/` on macOS. Used as
/// the root for capture history storage.
fn app_data_dir(app: &AppHandle) -> Result<std::path::PathBuf, String> {
    app.path()
        .app_data_dir()
        .map_err(|e| format!("app_data_dir: {e}"))
}

/// Frontend has finished a capture cycle and wants to remember the rect for
/// the "repeat last" hotkey. Stored on the frontend in localStorage; this
/// function only handles the Rust-side flag that says "the next capture
/// came from a repeat trigger, so reuse the stored rect".
#[tauri::command]
fn consume_repeat_pending(state: tauri::State<'_, AppState>) -> bool {
    state.repeat_pending.swap(false, Ordering::AcqRel)
}

/// Trigger the capture flow with the repeat-pending flag set. Called both
/// by the global "repeat" hotkey and by an in-app button.
async fn trigger_repeat_capture(app: AppHandle) {
    if let Some(state) = app.try_state::<AppState>() {
        state.repeat_pending.store(true, Ordering::Release);
    }
    trigger_capture_flow(app).await;
}

#[tauri::command]
async fn repeat_last_capture(app: AppHandle) {
    trigger_repeat_capture(app).await;
}

/// Append a finished capture+chat to history. The frontend calls this once
/// per turn (or once per capture session, depending on UX preference).
/// Async + spawn_blocking so the PNG decode + thumb encode + sync I/O don't
/// wedge the IPC dispatcher (one sync command at a time on a single thread).
#[tauri::command]
async fn add_history_entry(
    app: AppHandle,
    window: WebviewWindow,
    png_b64: String,
    width: u32,
    height: u32,
    provider: String,
    model: String,
    prompt: String,
    response: String,
) -> Result<HistoryEntry, HistoryError> {
    if window.label() != "overlay" && window.label() != "main" && window.label() != "chat" {
        return Err(HistoryError::Io(
            "command not allowed from this window".into(),
        ));
    }
    let dir = app_data_dir(&app).map_err(HistoryError::Io)?;
    tokio::task::spawn_blocking(move || {
        history::add_entry(
            &dir,
            history::AddArgs {
                png_b64,
                width,
                height,
                provider,
                model,
                prompt,
                response,
            },
        )
    })
    .await
    .map_err(|e| HistoryError::Io(format!("add_history task join: {e}")))?
}

#[tauri::command]
async fn list_history(
    app: AppHandle,
    window: WebviewWindow,
) -> Result<Vec<HistoryEntry>, HistoryError> {
    require_window(&window, "main").map_err(HistoryError::Io)?;
    let dir = app_data_dir(&app).map_err(HistoryError::Io)?;
    tokio::task::spawn_blocking(move || history::load_index(&dir))
        .await
        .map_err(|e| HistoryError::Io(format!("list_history task join: {e}")))?
}

#[tauri::command]
async fn delete_history_entry(
    app: AppHandle,
    window: WebviewWindow,
    id: String,
) -> Result<(), HistoryError> {
    require_window(&window, "main").map_err(HistoryError::Io)?;
    let dir = app_data_dir(&app).map_err(HistoryError::Io)?;
    tokio::task::spawn_blocking(move || history::delete_entry(&dir, &id))
        .await
        .map_err(|e| HistoryError::Io(format!("delete_history task join: {e}")))?
}

#[tauri::command]
async fn clear_history(app: AppHandle, window: WebviewWindow) -> Result<(), HistoryError> {
    require_window(&window, "main").map_err(HistoryError::Io)?;
    let dir = app_data_dir(&app).map_err(HistoryError::Io)?;
    tokio::task::spawn_blocking(move || history::clear_all(&dir))
        .await
        .map_err(|e| HistoryError::Io(format!("clear_history task join: {e}")))?
}

#[tauri::command]
async fn load_history_image(
    app: AppHandle,
    window: WebviewWindow,
    id: String,
) -> Result<String, HistoryError> {
    require_window(&window, "main").map_err(HistoryError::Io)?;
    let dir = app_data_dir(&app).map_err(HistoryError::Io)?;
    tokio::task::spawn_blocking(move || history::load_image_b64(&dir, &id))
        .await
        .map_err(|e| HistoryError::Io(format!("load_history_image task join: {e}")))?
}

#[tauri::command]
async fn load_history_thumb(
    app: AppHandle,
    window: WebviewWindow,
    id: String,
) -> Result<String, HistoryError> {
    require_window(&window, "main").map_err(HistoryError::Io)?;
    let dir = app_data_dir(&app).map_err(HistoryError::Io)?;
    tokio::task::spawn_blocking(move || history::load_thumb_b64(&dir, &id))
        .await
        .map_err(|e| HistoryError::Io(format!("load_history_thumb task join: {e}")))?
}

/// Run on-device OCR on the supplied PNG. Returns the recognized text or
/// an error string. Both desktop platforms ship with a built-in
/// recognition engine that we call into directly:
///
///   - macOS: Apple's Vision framework (`VNRecognizeTextRequest`).
///   - Windows: Microsoft's `Windows.Media.Ocr.OcrEngine` (the same one
///     PowerToys / Snip & Sketch / OneNote use under the hood).
///
/// In both cases the work is fully offline — no AI provider tokens, no
/// network round-trip, no extra binary or model file to bundle.
#[tauri::command]
async fn ocr_image_local(window: WebviewWindow, png_b64: String) -> Result<String, String> {
    if window.label() != "overlay" {
        return Err("command not allowed from this window".into());
    }
    // Vision (macOS) and OcrEngine (Windows) both perform a full PNG decode
    // followed by neural-net text recognition. On a Retina full-screen shot
    // this runs ~200 ms - 2 s, which is long enough to wedge the IPC
    // dispatcher and starve tray clicks, hotkey handlers, and other
    // in-flight commands. Move to the dedicated blocking pool.
    tokio::task::spawn_blocking(move || -> Result<String, String> {
        use base64::{engine::general_purpose::STANDARD, Engine as _};
        let bytes = STANDARD
            .decode(&png_b64)
            .map_err(|e| format!("base64 decode: {e}"))?;

        #[cfg(target_os = "macos")]
        {
            // Vision's request handler does not require the main thread.
            let ptr = unsafe { screenie_ocr_png(bytes.as_ptr(), bytes.len()) };
            if ptr.is_null() {
                return Err(
                    "Vision OCR failed (image could not be decoded or recognition errored)".into(),
                );
            }
            // SAFETY: the C side guarantees a NUL-terminated UTF-8 string
            // when the pointer is non-null. We copy into a Rust String, then
            // free.
            let s = unsafe { std::ffi::CStr::from_ptr(ptr) }
                .to_string_lossy()
                .into_owned();
            unsafe { screenie_free_string(ptr) };
            Ok(s)
        }
        #[cfg(target_os = "windows")]
        {
            windows_ocr_png(&bytes)
        }
        #[cfg(not(any(target_os = "macos", target_os = "windows")))]
        {
            let _ = bytes;
            Err("Local OCR isn't available on this platform yet.".into())
        }
    })
    .await
    .map_err(|e| format!("ocr task join: {e}"))?
}

/// Windows on-device OCR via WinRT. Loads the PNG into an
/// `InMemoryRandomAccessStream`, decodes to a `SoftwareBitmap`, then runs
/// `OcrEngine` against the user's preferred OCR language (falling back to
/// `en-US` if no profile-language engine is available — this is what
/// happens on a fresh install where the user has only US English).
#[cfg(target_os = "windows")]
fn windows_ocr_png(bytes: &[u8]) -> Result<String, String> {
    use windows::core::{Interface, HSTRING};
    use windows::Globalization::Language;
    use windows::Graphics::Imaging::BitmapDecoder;
    use windows::Media::Ocr::OcrEngine;
    use windows::Storage::Streams::{
        DataWriter, IRandomAccessStream, InMemoryRandomAccessStream,
    };

    fn err<E: std::fmt::Display>(stage: &str) -> impl Fn(E) -> String + '_ {
        move |e| format!("Windows OCR ({stage}): {e}")
    }

    let stream =
        InMemoryRandomAccessStream::new().map_err(err("stream init"))?;
    {
        let writer = DataWriter::CreateDataWriter(&stream)
            .map_err(err("writer init"))?;
        writer.WriteBytes(bytes).map_err(err("write bytes"))?;
        writer
            .StoreAsync()
            .map_err(err("store async"))?
            .get()
            .map_err(err("store await"))?;
        writer
            .FlushAsync()
            .map_err(err("flush async"))?
            .get()
            .map_err(err("flush await"))?;
        // Detach so the writer doesn't take ownership of the stream when it
        // drops out of scope.
        let _ = writer.DetachStream();
    }
    stream.Seek(0u64).map_err(err("seek"))?;

    // BitmapDecoder needs the stream as `IRandomAccessStream`.
    let stream_iface: IRandomAccessStream = stream.cast().map_err(err("stream cast"))?;
    let decoder = BitmapDecoder::CreateAsync(&stream_iface)
        .map_err(err("decoder create"))?
        .get()
        .map_err(err("decoder await"))?;
    let bitmap = decoder
        .GetSoftwareBitmapAsync()
        .map_err(err("bitmap"))?
        .get()
        .map_err(err("bitmap await"))?;

    // Try the user's first-preference OCR language; fall back to en-US.
    // `TryCreateFromUserProfileLanguages` returns null if the user has no
    // OCR-capable language installed at all, which the windows crate
    // surfaces as Err.
    let engine = OcrEngine::TryCreateFromUserProfileLanguages()
        .or_else(|_| {
            let lang = Language::CreateLanguage(&HSTRING::from("en-US"))?;
            OcrEngine::TryCreateFromLanguage(&lang)
        })
        .map_err(err("ocr engine"))?;

    let result = engine
        .RecognizeAsync(&bitmap)
        .map_err(err("recognize"))?
        .get()
        .map_err(err("recognize await"))?;
    let text = result.Text().map_err(err("text"))?;
    Ok(text.to_string())
}

/// Save the (optionally annotated) cropped PNG to disk. Writes to
/// `~/Pictures/Screenie/` (creating it if needed) with a millisecond-
/// suffixed filename, returning the absolute path so the frontend can
/// surface a "Saved to …" toast.
#[tauri::command]
async fn save_annotated_image(
    app: AppHandle,
    window: WebviewWindow,
    png_b64: String,
) -> Result<String, String> {
    if window.label() != "overlay" {
        return Err("command not allowed from this window".into());
    }
    let pictures = app
        .path()
        .picture_dir()
        .map_err(|e| format!("picture_dir: {e}"))?
        .join("Screenie");
    // Base64 decode of a many-megabyte annotated PNG plus a sync filesystem
    // write are both candidates for blocking the IPC dispatcher. Run on the
    // blocking pool so other commands keep flowing.
    tokio::task::spawn_blocking(move || -> Result<String, String> {
        use base64::{engine::general_purpose::STANDARD, Engine as _};
        let bytes = STANDARD
            .decode(&png_b64)
            .map_err(|e| format!("base64 decode: {e}"))?;
        std::fs::create_dir_all(&pictures).map_err(|e| format!("mkdir: {e}"))?;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let path = pictures.join(format!("Screenie-{}.png", now));
        std::fs::write(&path, &bytes).map_err(|e| format!("write: {e}"))?;
        Ok(path.to_string_lossy().into_owned())
    })
    .await
    .map_err(|e| format!("save task join: {e}"))?
}

#[derive(serde::Serialize)]
struct HotkeyConfigDto {
    capture: String,
    repeat: String,
    settings: String,
}

impl From<&HotkeyConfig> for HotkeyConfigDto {
    fn from(c: &HotkeyConfig) -> Self {
        Self {
            capture: c.capture.clone(),
            repeat: c.repeat.clone(),
            settings: c.settings.clone(),
        }
    }
}

#[tauri::command]
fn get_hotkey_config(state: tauri::State<'_, AppState>) -> HotkeyConfigDto {
    let cfg = lock_poison_safe(&state.hotkeys);
    HotkeyConfigDto::from(&*cfg)
}

#[tauri::command]
fn set_hotkey_config(
    app: AppHandle,
    window: WebviewWindow,
    capture: String,
    repeat: String,
    settings: String,
) -> Result<(), String> {
    if window.label() != "main" {
        return Err("command not allowed from this window".into());
    }
    apply_hotkey_config(&app, capture, repeat, settings)
}

#[cfg(desktop)]
fn apply_hotkey_config(
    app: &AppHandle,
    capture: String,
    repeat: String,
    settings: String,
) -> Result<(), String> {
    use std::str::FromStr;
    use tauri_plugin_global_shortcut::{GlobalShortcutExt, Shortcut};

    let new_capture =
        Shortcut::from_str(&capture).map_err(|e| format!("capture shortcut invalid: {e}"))?;
    let new_repeat =
        Shortcut::from_str(&repeat).map_err(|e| format!("repeat shortcut invalid: {e}"))?;
    let new_settings =
        Shortcut::from_str(&settings).map_err(|e| format!("settings shortcut invalid: {e}"))?;

    // Unregister the previous accelerators (read from state).
    let state = app.state::<AppState>();
    let prev: HotkeyConfig = lock_poison_safe(&state.hotkeys).clone();
    for s in [&prev.capture, &prev.repeat, &prev.settings] {
        if let Ok(sc) = Shortcut::from_str(s) {
            let _ = app.global_shortcut().unregister(sc);
        }
    }

    let mut errors = Vec::new();
    let mut registered = Vec::new();
    for (sc, label) in [
        (&new_capture, "capture"),
        (&new_repeat, "repeat"),
        (&new_settings, "settings"),
    ] {
        match app.global_shortcut().register(*sc) {
            Ok(()) => registered.push(*sc),
            Err(e) => errors.push(format!("{}: {}", label, e)),
        }
    }

    if errors.is_empty() {
        let saved = HotkeyConfig {
            capture,
            repeat,
            settings,
        };
        {
            let mut cfg = lock_poison_safe(&state.hotkeys);
            *cfg = saved.clone();
        }
        save_hotkey_config(app, &saved)?;
        // P-A-B12: clear any stale startup-failure error so the tutorial
        // banner and Settings error chip clear once the user has successfully
        // re-bound the shortcut.
        if let Ok(mut g) = state.hotkey_error.lock() {
            *g = None;
        }
        Ok(())
    } else {
        // Roll back to the previous config so the app stays in a working state.
        for sc in registered {
            let _ = app.global_shortcut().unregister(sc);
        }
        let _ = re_register_hotkeys(app, &prev);
        Err(errors.join("; "))
    }
}

#[cfg(not(desktop))]
fn apply_hotkey_config(_: &AppHandle, _: String, _: String, _: String) -> Result<(), String> {
    Err("global shortcuts not supported on this platform".into())
}

#[cfg(desktop)]
fn re_register_hotkeys(app: &AppHandle, cfg: &HotkeyConfig) -> Result<(), String> {
    use std::str::FromStr;
    use tauri_plugin_global_shortcut::{GlobalShortcutExt, Shortcut};
    let mut errors = Vec::new();
    for s in [&cfg.capture, &cfg.repeat, &cfg.settings] {
        if let Ok(sc) = Shortcut::from_str(s) {
            if let Err(e) = app.global_shortcut().register(sc) {
                errors.push(format!("{s}: {e}"));
            }
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("; "))
    }
}

/// Open (or focus) a detached chat window seeded with a base64 PNG and a
/// chat thread. The window has normal chrome, can be resized + moved, and
/// is closed independently of the overlay. The frontend hands the seed
/// data over via the URL query (length-bounded since base64 PNGs can be
/// big — we keep a copy in app state instead and the new window pulls it
/// via `take_chat_seed`).
#[tauri::command]
async fn open_chat_window(
    app: AppHandle,
    window: WebviewWindow,
    png_b64: String,
    width: u32,
    height: u32,
    provider: String,
    model: String,
    messages_json: String,
) -> Result<(), String> {
    if window.label() != "overlay" {
        return Err("command not allowed from this window".into());
    }
    // Mirror `ask_ai`'s payload validation. Without this, a buggy or
    // malicious overlay could stash gigabytes in `AppState.chat_seed`
    // until the chat window pulled them.
    validate_chat_seed_payload(&png_b64, &messages_json)?;
    {
        let state = app.state::<AppState>();
        let mut g = lock_poison_safe(&state.chat_seed);
        *g = Some(ChatSeed {
            png_b64,
            width,
            height,
            provider,
            model,
            messages_json,
        });
    }
    if let Some(existing) = app.get_webview_window("chat") {
        let _ = existing.show();
        let _ = existing.set_focus();
        let _ = existing.emit("chat-seed-changed", ());
        return Ok(());
    }
    let chat_builder = WebviewWindowBuilder::new(
        &app,
        "chat",
        WebviewUrl::App("index.html?mode=chat".into()),
    )
    .title("Screenie AI · Chat")
    .inner_size(420.0, 560.0)
    .min_inner_size(300.0, 320.0)
    .resizable(true)
    // Borderless + transparent so the in-page chat panel is the
    // entire visible window. `.shadow(false)` matches the overlay's
    // configuration so macOS doesn't paint its system shadow/border
    // around the rounded panel — the panel's own frost + rounded
    // corners are the only chrome.
    .transparent(true)
    .decorations(false)
    .shadow(false)
    // macOS NSVisualEffectView "hudWindow" vibrancy — heaviest practical
    // material, matches the overlay's frost panes + the floating-panel
    // look (ChatGPT / Linear / Notion style: heavy gaussian blur with
    // colors bleeding through). 24px corner radius matches the chat
    // panel's CSS border-radius.
    .effects(WindowEffectsConfig {
        effects: vec![WindowEffect::HudWindow],
        state: Some(WindowEffectState::Active),
        radius: Some(24.0),
        color: None,
    })
    // Match the original overlay's "lives above other apps" feel.
    .always_on_top(true)
    .skip_taskbar(true)
    .visible(true);
    let chat_win = chat_builder
        .build()
        .map_err(|e| format!("chat window: {e}"))?;
    // P-A-B7: cancel any in-flight chat AI stream when the user closes the
    // chat window. Without this, closing mid-stream leaves the request
    // running on a destroyed webview (Channel::send errors silently, the
    // stream loop only exits between chunks via the cancel flag).
    let app_for_event = app.clone();
    chat_win.on_window_event(move |event| {
        if let WindowEvent::Destroyed = event {
            if let Some(state) = app_for_event.try_state::<AppState>() {
                cancel_active_chat_ai(&state);
            }
        }
    });
    #[cfg(target_os = "windows")]
    {
        // Tauri's `WindowEffect::Sidebar` above is macOS-only; on Windows it's
        // silently ignored, so the borderless transparent chat window would
        // render with no backdrop at all. Apply Mica via DWM here so the
        // detached chat panel gets a live frosted-glass background that
        // matches the macOS sidebar vibrancy.
        windows_window::configure_main_window(&chat_win);
    }
    let _ = chat_win;
    Ok(())
}

#[tauri::command]
fn take_chat_seed(
    app: AppHandle,
    window: WebviewWindow,
) -> Option<ChatSeed> {
    if window.label() != "chat" {
        return None;
    }
    let state = app.state::<AppState>();
    let seed = lock_poison_safe(&state.chat_seed).take();
    seed
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct ChatSeed {
    png_b64: String,
    width: u32,
    height: u32,
    provider: String,
    model: String,
    messages_json: String,
}

/// Find the monitor whose physical bounds contain the cursor. Falls back to
/// the primary monitor if the cursor lookup fails.
/// Find the monitor whose physical bounds contain the cursor. Falls back to
/// the primary monitor if the cursor lookup fails.
///
/// P-C-R9: takes the cursor as a caller-supplied snapshot so the *same*
/// reading is used for both monitor selection and the post-capture cursor
/// metadata. Re-reading cursor_position 50-150 ms later could otherwise
/// return a position on a different monitor (multi-display setup, user
/// moved during capture) and the overlay's cursor callout would land on
/// the wrong screen.
fn pick_cursor_monitor(
    app: &AppHandle,
    cursor: Option<tauri::PhysicalPosition<f64>>,
) -> Option<tauri::Monitor> {
    let monitors = app.available_monitors().ok()?;
    if let Some(c) = cursor {
        for m in &monitors {
            let p = m.position();
            let s = m.size();
            let x0 = p.x as f64;
            let y0 = p.y as f64;
            let x1 = x0 + s.width as f64;
            let y1 = y0 + s.height as f64;
            if c.x >= x0 && c.x < x1 && c.y >= y0 && c.y < y1 {
                return Some(m.clone());
            }
        }
    }
    app.primary_monitor().ok().flatten().or_else(|| monitors.into_iter().next())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let mut builder = tauri::Builder::default()
        .manage(AppState::default())
        .plugin(tauri_plugin_opener::init());

    builder = builder
        .invoke_handler(tauri::generate_handler![
            take_pending_capture,
            crop_capture,
            refresh_overlay_backdrop_capture,
            refresh_overlay_capture,
            show_overlay_after_refresh,
            close_overlay,
            set_overlay_interaction_regions,
            set_overlay_vibrancy_regions,
            set_overlay_text_input_focused,
            set_overlay_mouse_capture,
            relay_overlay_pointer_click,
            relay_overlay_wheel,
            cancel_ai,
            open_screen_settings,
            quit_app,
            restart_app,
            ask_ai,
            check_ollama,
            check_ollama_installed,
            get_hotkey_registration_error,
            launch_ollama,
            install_ollama,
            pull_ollama_model,
            set_secret,
            get_secret,
            delete_secret,
            show_settings_window,
            hide_settings_window,
            complete_onboarding,
            set_tutorial_mode,
            consume_repeat_pending,
            repeat_last_capture,
            add_history_entry,
            list_history,
            delete_history_entry,
            clear_history,
            load_history_image,
            load_history_thumb,
            save_annotated_image,
            ocr_image_local,
            get_hotkey_config,
            set_hotkey_config,
            open_chat_window,
            take_chat_seed
        ]);

    #[cfg(desktop)]
    {
        use std::str::FromStr;
        use tauri_plugin_global_shortcut::{GlobalShortcutExt, Shortcut, ShortcutState};

        builder = builder.plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(move |app, shortcut, event| {
                    if event.state() != ShortcutState::Pressed {
                        return;
                    }
                    // Look up the current configured shortcuts; dispatch by
                    // matching the firing shortcut against each one. Reading
                    // state on every press is cheap (string compares) and
                    // keeps the handler in sync with `set_hotkey_config`.
                    let cfg = match app.try_state::<AppState>() {
                        Some(s) => lock_poison_safe(&s.hotkeys).clone(),
                        None => return,
                    };
                    let cap = Shortcut::from_str(&cfg.capture).ok();
                    let rep = Shortcut::from_str(&cfg.repeat).ok();
                    let settings = Shortcut::from_str(&cfg.settings).ok();
                    if cap.as_ref() == Some(shortcut) {
                        let app = app.clone();
                        tauri::async_runtime::spawn(async move {
                            trigger_capture_flow(app).await;
                        });
                    } else if rep.as_ref() == Some(shortcut) {
                        let app = app.clone();
                        tauri::async_runtime::spawn(async move {
                            trigger_repeat_capture(app).await;
                        });
                    } else if settings.as_ref() == Some(shortcut) {
                        let app = app.clone();
                        tauri::async_runtime::spawn(async move {
                            let _ = show_settings_window(app).await;
                        });
                    }
                })
                .build(),
        );

        builder = builder.setup(move |app| {
            let handle = app.handle();

            if let Err(e) = setup_tray(handle) {
                eprintln!("[screenie] tray setup FAILED: {}", e);
                return Err(Box::new(e));
            }

            #[cfg(target_os = "macos")]
            let _ = handle.set_activation_policy(ActivationPolicy::Accessory);

            if let Some(main) = handle.get_webview_window("main") {
                configure_main_window(&main);
                let app_handle = handle.clone();
                main.on_window_event(move |event| {
                    if let WindowEvent::CloseRequested { api, .. } = event {
                        api.prevent_close();
                        let h = app_handle.clone();
                        tauri::async_runtime::spawn(async move {
                            let _ = hide_settings_window(h).await;
                        });
                    }
                    // Windows tray glyph needs an explicit RGB swap on
                    // light/dark theme changes — `icon_as_template` is a
                    // no-op there, so a white glyph would otherwise stay
                    // invisible after the user flips to a light taskbar.
                    // We also re-render on DPI changes so the Lanczos3
                    // downsample in `tray_icon_for_theme` retargets to the
                    // new system small-icon size and the wink stays crisp
                    // when the user moves the laptop between docks at
                    // different scales. macOS's template-image path handles
                    // both transitions automatically.
                    #[cfg(target_os = "windows")]
                    {
                        let needs_refresh = matches!(
                            event,
                            WindowEvent::ThemeChanged(_)
                                | WindowEvent::ScaleFactorChanged { .. }
                        );
                        if needs_refresh {
                            let theme = match event {
                                WindowEvent::ThemeChanged(t) => *t,
                                _ => app_handle
                                    .get_webview_window("main")
                                    .and_then(|w| w.theme().ok())
                                    .unwrap_or(tauri::Theme::Dark),
                            };
                            if let Some(tray) = app_handle.tray_by_id("main") {
                                match tray_icon_for_theme(theme) {
                                    Ok(icon) => {
                                        let _ = tray.set_icon(Some(icon));
                                    }
                                    Err(e) => {
                                        eprintln!(
                                            "[screenie] tray icon refresh failed: {e}"
                                        );
                                    }
                                }
                            }
                        }
                    }
                });
            }

            // Register the configured hotkeys. A saved user configuration is
            // loaded before registration so custom shortcuts survive relaunch.
            {
                let loaded = load_hotkey_config(handle);
                let state = handle.state::<AppState>();
                let mut cfg = lock_poison_safe(&state.hotkeys);
                *cfg = loaded;
            }
            let state = handle.state::<AppState>();
            let cfg: HotkeyConfig = lock_poison_safe(&state.hotkeys).clone();
            let mut hotkey_failures: Vec<String> = Vec::new();
            for (acc, label) in [
                (&cfg.capture, "capture"),
                (&cfg.repeat, "repeat"),
                (&cfg.settings, "settings"),
            ] {
                match Shortcut::from_str(acc) {
                    Ok(sc) => match app.global_shortcut().register(sc) {
                        Ok(()) => eprintln!("[screenie] shortcut {} ({}) registered OK", label, acc),
                        Err(e) => {
                            eprintln!("[screenie] shortcut {} register FAILED: {}", label, e);
                            hotkey_failures.push(format!("{}: {}", label, e));
                        }
                    },
                    Err(e) => {
                        eprintln!("[screenie] shortcut {} parse failed: {}", label, e);
                        hotkey_failures.push(format!("{}: parse error: {}", label, e));
                    }
                }
            }
            if !hotkey_failures.is_empty() {
                let msg = hotkey_failures.join("; ");
                if let Some(state) = handle.try_state::<AppState>() {
                    if let Ok(mut g) = state.hotkey_error.lock() {
                        *g = Some(msg.clone());
                    }
                }
                if let Some(main) = handle.get_webview_window("main") {
                    let _ = main.emit("hotkey-registration-failed", msg);
                }
            }
            Ok(())
        });
    }

    builder
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> UiMessage {
        UiMessage {
            role: role.into(),
            content: content.into(),
        }
    }

    // ---------- validate_ai_payload ----------

    #[test]
    fn validate_ai_payload_happy_path() {
        let msgs = vec![msg("user", "hello")];
        assert!(validate_ai_payload(&msgs, "base64data").is_ok());
    }

    #[test]
    fn validate_ai_payload_rejects_oversized_image() {
        let huge = "x".repeat(MAX_AI_IMAGE_B64_CHARS + 1);
        let result = validate_ai_payload(&[], &huge);
        assert!(matches!(result, Err(AiError::RequestTooLarge(_))));
    }

    #[test]
    fn validate_ai_payload_rejects_too_many_messages() {
        let msgs: Vec<UiMessage> =
            (0..=MAX_AI_MESSAGES).map(|_| msg("user", "x")).collect();
        let result = validate_ai_payload(&msgs, "");
        assert!(matches!(result, Err(AiError::RequestTooLarge(_))));
    }

    #[test]
    fn validate_ai_payload_rejects_long_single_message() {
        let long = "x".repeat(MAX_AI_MESSAGE_CHARS + 1);
        let msgs = vec![msg("user", &long)];
        let result = validate_ai_payload(&msgs, "");
        assert!(matches!(result, Err(AiError::RequestTooLarge(_))));
    }

    #[test]
    fn validate_ai_payload_rejects_total_chars() {
        // Each message is at the per-message cap; enough of them push the
        // total past MAX_AI_TOTAL_MESSAGE_CHARS.
        let big = "x".repeat(MAX_AI_MESSAGE_CHARS);
        let n = (MAX_AI_TOTAL_MESSAGE_CHARS / MAX_AI_MESSAGE_CHARS) + 1;
        let n = n.min(MAX_AI_MESSAGES); // keep below the count cap
        let msgs: Vec<UiMessage> = (0..n).map(|_| msg("user", &big)).collect();
        let result = validate_ai_payload(&msgs, "");
        assert!(matches!(result, Err(AiError::RequestTooLarge(_))));
    }

    // ---------- secret_name_for_provider ----------

    #[test]
    fn secret_name_for_provider_known() {
        assert_eq!(
            secret_name_for_provider("anthropic").unwrap(),
            Some("anthropic_api_key")
        );
        assert_eq!(
            secret_name_for_provider("openai").unwrap(),
            Some("openai_api_key")
        );
        assert_eq!(
            secret_name_for_provider("gemini").unwrap(),
            Some("gemini_api_key")
        );
        // Ollama is local and has no secret slot.
        assert_eq!(secret_name_for_provider("ollama").unwrap(), None);
    }

    #[test]
    fn secret_name_for_provider_rejects_unknown() {
        assert!(matches!(
            secret_name_for_provider("evil"),
            Err(AiError::InvalidProvider(_))
        ));
    }

    // ---------- provider_default_model ----------

    #[test]
    fn provider_default_model_all_known() {
        assert!(provider_default_model("anthropic").is_ok());
        assert!(provider_default_model("openai").is_ok());
        assert!(provider_default_model("gemini").is_ok());
        assert!(provider_default_model("ollama").is_ok());
    }

    #[test]
    fn provider_default_model_unknown_rejected() {
        assert!(matches!(
            provider_default_model("evil"),
            Err(AiError::InvalidProvider(_))
        ));
    }

    // ---------- normalize_response_profile ----------

    #[test]
    fn normalize_profile_balanced_and_detailed_pass_through() {
        assert_eq!(
            normalize_response_profile(Some("balanced".into())),
            "balanced"
        );
        assert_eq!(
            normalize_response_profile(Some("detailed".into())),
            "detailed"
        );
    }

    #[test]
    fn normalize_profile_unknown_collapses_to_concise() {
        assert_eq!(normalize_response_profile(None), "concise");
        assert_eq!(
            normalize_response_profile(Some("nonsense".into())),
            "concise"
        );
        // Pin the case-sensitivity: an explicit "CONCISE" does NOT match.
        assert_eq!(
            normalize_response_profile(Some("CONCISE".into())),
            "concise"
        );
    }
}
