use super::executor::{CalibrationProbe, ClickPoint, TargetSummary};
use super::observer::{
    ensure_accessibility_permission_with, filter_candidates, is_structural_container_role,
    ObservedCandidate, MAX_OBSERVED_ELEMENTS,
};
use super::safari_dom;
use super::search::score_match;
use super::types::{
    intersect_rects, is_secure_text_role, normalize_signature_name, ChangeCounter,
    CoordinateSpace, Element, ElementSource, FocusedApp, FocusedAppProvider, MenuMatch,
    MenuPressOutcome, MenuScanResult, ObservationError, PlatformElementHandle, Rect,
    ScreenObserver, ScrollContainerKind, ScrollContext, UiChangeSignal,
};
use super::vision::{ObservationMetadata, ObservationMetadataProvider};
use core_foundation::base::TCFType;
use core_foundation::boolean::CFBoolean;
use core_foundation::dictionary::CFDictionary;
use core_foundation::string::CFString;
use core_foundation_sys::array::{
    CFArrayGetCount, CFArrayGetTypeID, CFArrayGetValueAtIndex, CFArrayRef,
};
use core_foundation_sys::base::{
    Boolean, CFCopyDescription, CFGetTypeID, CFRelease, CFRetain, CFTypeID, CFTypeRef,
};
use core_foundation_sys::dictionary::CFDictionaryRef;
use core_foundation_sys::number::{
    kCFNumberFloat64Type, CFBooleanGetTypeID, CFBooleanGetValue, CFBooleanRef, CFNumberGetTypeID,
    CFNumberGetValue, CFNumberRef,
};
use core_foundation_sys::runloop::{
    kCFRunLoopDefaultMode, CFRunLoopAddSource, CFRunLoopGetCurrent, CFRunLoopRef,
    CFRunLoopRemoveSource, CFRunLoopRun, CFRunLoopSourceContext, CFRunLoopSourceCreate,
    CFRunLoopSourceRef, CFRunLoopWakeUp,
};
use core_foundation_sys::string::{CFStringGetTypeID, CFStringRef};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::ffi::{c_char, c_void, CStr};
use std::fmt;
use std::mem;
use std::ptr;
use std::rc::Rc;
use std::sync::{mpsc, Arc, Mutex, OnceLock, PoisonError};
use std::thread;
use std::time::{Duration, Instant};

const MAX_AX_DEPTH: usize = 8;
const MAX_AX_WEB_DEPTH: usize = 24;
const MAX_AX_CHILDREN_PER_NODE: usize = 200;

const AX_ERROR_SUCCESS: i32 = 0;
const AX_ERROR_INVALID_UI_ELEMENT: i32 = -25200;
const AX_ERROR_CANNOT_COMPLETE: i32 = -25204;
const AX_ERROR_ATTRIBUTE_UNSUPPORTED: i32 = -25205;
const AX_ERROR_NO_VALUE: i32 = -25212;

const K_AX_VALUE_CGPOINT_TYPE: i32 = 1;
const K_AX_VALUE_CGSIZE_TYPE: i32 = 2;

type AXUIElementRef = CFTypeRef;
type AXValueRef = CFTypeRef;
type ObjcId = *mut c_void;
type Sel = *mut c_void;
type Pid = i32;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct CGPoint {
    x: f64,
    y: f64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct CGSize {
    width: f64,
    height: f64,
}

#[link(name = "ApplicationServices", kind = "framework")]
extern "C" {
    fn AXIsProcessTrusted() -> Boolean;
    fn AXIsProcessTrustedWithOptions(options: CFDictionaryRef) -> Boolean;
    fn AXUIElementCreateApplication(pid: Pid) -> AXUIElementRef;
    fn AXUIElementCreateSystemWide() -> AXUIElementRef;
    fn AXUIElementCopyAttributeValue(
        element: AXUIElementRef,
        attribute: CFStringRef,
        value: *mut CFTypeRef,
    ) -> i32;
    fn AXUIElementSetAttributeValue(
        element: AXUIElementRef,
        attribute: CFStringRef,
        value: CFTypeRef,
    ) -> i32;
    fn AXUIElementPerformAction(element: AXUIElementRef, action: CFStringRef) -> i32;
    fn AXUIElementCopyElementAtPosition(
        application: AXUIElementRef,
        x: f32,
        y: f32,
        element: *mut AXUIElementRef,
    ) -> i32;
    fn AXUIElementGetPid(element: AXUIElementRef, pid: *mut Pid) -> i32;
    fn AXValueGetType(value: AXValueRef) -> i32;
    fn AXValueGetTypeID() -> CFTypeID;
    fn AXValueGetValue(value: AXValueRef, value_type: i32, value_ptr: *mut c_void) -> Boolean;
}

#[link(name = "objc", kind = "dylib")]
extern "C" {
    fn objc_getClass(name: *const c_char) -> ObjcId;
    fn sel_registerName(name: *const c_char) -> Sel;
    fn objc_msgSend();
}

type AXObserverRef = CFTypeRef;
type AXObserverCallback = unsafe extern "C" fn(
    observer: AXObserverRef,
    element: AXUIElementRef,
    notification: CFStringRef,
    refcon: *mut c_void,
);

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct CGRectRaw {
    origin: CGPoint,
    size: CGSize,
}

#[link(name = "ApplicationServices", kind = "framework")]
extern "C" {
    fn CGMainDisplayID() -> u32;
    fn CGDisplayBounds(display: u32) -> CGRectRaw;
    fn AXObserverCreate(
        application: Pid,
        callback: AXObserverCallback,
        out_observer: *mut AXObserverRef,
    ) -> i32;
    fn AXObserverAddNotification(
        observer: AXObserverRef,
        element: AXUIElementRef,
        notification: CFStringRef,
        refcon: *mut c_void,
    ) -> i32;
    fn AXObserverRemoveNotification(
        observer: AXObserverRef,
        element: AXUIElementRef,
        notification: CFStringRef,
    ) -> i32;
    fn AXObserverGetRunLoopSource(observer: AXObserverRef) -> CFRunLoopSourceRef;
}

/// Notifications whose arrival means "the frontmost app's UI changed" for
/// settle purposes. Names are the values of the corresponding kAX…
/// Notification constants in AXNotificationConstants.h.
const AX_CHANGE_NOTIFICATIONS: &[&str] = &[
    "AXFocusedUIElementChanged",
    "AXWindowCreated",
    "AXValueChanged",
    "AXUIElementDestroyed",
];

/// Process-wide bridge from AXObserver callbacks (delivered on the dedicated
/// signal run-loop thread) to settle-path waiters on the agent thread. One
/// registration follows the frontmost app and is replaced when focus moves
/// to a different pid.
struct AxChangeSignalHub {
    counter: Arc<ChangeCounter>,
    slot: Mutex<AxRegistrationSlot>,
}

#[derive(Default)]
struct AxRegistrationSlot {
    active: Option<AxChangeRegistration>,
    /// Registration is retried only when the frontmost pid changes, so an
    /// app without AX notification support doesn't pay the failed AX
    /// round-trips on every settle call.
    failed_pid: Option<Pid>,
}

static AX_CHANGE_HUB: OnceLock<AxChangeSignalHub> = OnceLock::new();

fn ax_change_hub() -> &'static AxChangeSignalHub {
    AX_CHANGE_HUB.get_or_init(|| AxChangeSignalHub {
        counter: Arc::new(ChangeCounter::default()),
        slot: Mutex::new(AxRegistrationSlot::default()),
    })
}

/// Runs on the signal run-loop thread. Kept trivial on purpose: waiters
/// re-observe and work out what changed, so the callback carries no payload.
unsafe extern "C" fn ax_change_callback(
    _observer: AXObserverRef,
    _element: AXUIElementRef,
    _notification: CFStringRef,
    _refcon: *mut c_void,
) {
    if let Some(hub) = AX_CHANGE_HUB.get() {
        hub.counter.bump();
    }
}

struct AxChangeRegistration {
    pid: Pid,
    observer: AXObserverRef,
    app: AXUIElementRef,
    registered: Vec<&'static str>,
    runloop: CFRunLoopRef,
}

// The raw refs are only touched from whichever thread holds the hub mutex;
// the run-loop thread only drives the scheduled source and never sees the
// registration itself.
unsafe impl Send for AxChangeRegistration {}

impl Drop for AxChangeRegistration {
    fn drop(&mut self) {
        unsafe {
            // Stop the AX server sending first, then unschedule local
            // delivery, then release. Errors are ignored — the observed app
            // may already be gone.
            for name in &self.registered {
                let notification = CFString::new(name);
                let _ = AXObserverRemoveNotification(
                    self.observer,
                    self.app,
                    notification.as_concrete_TypeRef(),
                );
            }
            let source = AXObserverGetRunLoopSource(self.observer);
            if !source.is_null() {
                // No-op when the source was never added (failed registration
                // paths drop the struct before scheduling).
                CFRunLoopRemoveSource(self.runloop, source, kCFRunLoopDefaultMode);
            }
            CFRelease(self.observer);
            CFRelease(self.app);
        }
    }
}

extern "C" fn ax_signal_keep_alive_noop(_info: *const c_void) {}

/// The dedicated CFRunLoop thread AXObserver sources are scheduled on. A
/// permanent no-op source keeps `CFRunLoopRun` from returning while no
/// AXObserver is registered. `None` when the thread failed to start; settle
/// paths then stay on polling.
fn ax_signal_runloop() -> Option<CFRunLoopRef> {
    struct SendCFRunLoop(CFRunLoopRef);
    // CFRunLoop is one of the explicitly thread-safe CF types; the ref is
    // only used with CFRunLoopAddSource/RemoveSource/WakeUp.
    unsafe impl Send for SendCFRunLoop {}
    unsafe impl Sync for SendCFRunLoop {}

    static RUNLOOP: OnceLock<Option<SendCFRunLoop>> = OnceLock::new();
    RUNLOOP
        .get_or_init(|| {
            let (tx, rx) = mpsc::channel();
            let spawned = thread::Builder::new()
                .name("ax-change-signal".into())
                .spawn(move || unsafe {
                    let runloop = CFRunLoopGetCurrent();
                    let mut context = CFRunLoopSourceContext {
                        version: 0,
                        info: ptr::null_mut(),
                        retain: None,
                        release: None,
                        copyDescription: None,
                        equal: None,
                        hash: None,
                        schedule: None,
                        cancel: None,
                        perform: ax_signal_keep_alive_noop,
                    };
                    let keep_alive = CFRunLoopSourceCreate(ptr::null(), 0, &mut context);
                    if !keep_alive.is_null() {
                        CFRunLoopAddSource(runloop, keep_alive, kCFRunLoopDefaultMode);
                    }
                    if tx.send(SendCFRunLoop(runloop)).is_err() {
                        return;
                    }
                    loop {
                        CFRunLoopRun();
                        // Only reachable without the keep-alive source; pace
                        // the retry instead of spinning.
                        thread::sleep(Duration::from_millis(50));
                    }
                });
            if spawned.is_err() {
                return None;
            }
            rx.recv_timeout(Duration::from_secs(2)).ok()
        })
        .as_ref()
        .map(|handle| handle.0)
}

fn register_ax_change_observer(pid: Pid) -> Option<AxChangeRegistration> {
    let runloop = ax_signal_runloop()?;
    let mut observer: AXObserverRef = ptr::null();
    let err = unsafe { AXObserverCreate(pid, ax_change_callback, &mut observer) };
    if err != AX_ERROR_SUCCESS || observer.is_null() {
        return None;
    }
    let app = unsafe { AXUIElementCreateApplication(pid) };
    if app.is_null() {
        unsafe { CFRelease(observer) };
        return None;
    }

    // From here the registration owns both refs; early returns clean up
    // through its Drop.
    let mut registration = AxChangeRegistration {
        pid,
        observer,
        app,
        registered: Vec::new(),
        runloop,
    };
    for name in AX_CHANGE_NOTIFICATIONS {
        let notification = CFString::new(name);
        let err = unsafe {
            AXObserverAddNotification(
                observer,
                app,
                notification.as_concrete_TypeRef(),
                ptr::null_mut(),
            )
        };
        if err == AX_ERROR_SUCCESS {
            registration.registered.push(*name);
        }
    }
    if registration.registered.is_empty() {
        return None;
    }
    let source = unsafe { AXObserverGetRunLoopSource(observer) };
    if source.is_null() {
        return None;
    }
    unsafe {
        CFRunLoopAddSource(runloop, source, kCFRunLoopDefaultMode);
        CFRunLoopWakeUp(runloop);
    }
    Some(registration)
}

/// macOS Accessibility observer.
///
/// Bounds returned by this observer are macOS Accessibility screen points
/// with a top-left origin. For Milestone 3, those AX points and enigo's
/// CGEvent global positioning are assumed to be 1:1. Screenshot pixel
/// conversion belongs to a later milestone.
/// Electron exposes its AX tree only after a client writes
/// `AXManualAccessibility`; Chromium wants `AXEnhancedUserInterface` (the
/// VoiceOver flag — drop it from this list if apps misbehave under it).
/// Native apps reject both writes; that is expected and silent.
const ELECTRON_UNLOCK_ATTRIBUTES: &[&str] = &["AXManualAccessibility", "AXEnhancedUserInterface"];

#[derive(Clone, Debug, Default)]
pub struct MacObserver {
    handles: AxHandleRegistry,
    unlocked_pids: Rc<RefCell<HashSet<Pid>>>,
}

impl MacObserver {
    pub fn new() -> Self {
        Self::default()
    }

    fn ensure_app_ax_unlocked(&self, pid: Pid, app: AXUIElementRef) {
        if !should_attempt_ax_unlock(&mut self.unlocked_pids.borrow_mut(), pid) {
            return;
        }
        for attribute in ELECTRON_UNLOCK_ATTRIBUTES {
            let name = CFString::new(attribute);
            let value = CFBoolean::true_value();
            unsafe {
                let _ = AXUIElementSetAttributeValue(
                    app,
                    name.as_concrete_TypeRef(),
                    value.as_CFTypeRef(),
                );
            }
        }
    }

    fn observe_frontmost_app(&self) -> Result<Vec<Element>, ObservationError> {
        ensure_accessibility_permission_with(system_accessibility_trusted, prompt_accessibility)?;

        let focused_app = frontmost_application_info()?.ok_or(ObservationError::NoFrontmostApp)?;
        let pid = focused_app.pid.ok_or(ObservationError::NoFrontmostApp)?;
        let safari_mode = is_safari_app(&focused_app);
        let app = unsafe { AXUIElementCreateApplication(pid) };
        if app.is_null() {
            return Err(ObservationError::NoFrontmostApp);
        }
        let _app_ref = OwnedCf::new(app);
        self.ensure_app_ax_unlocked(pid, app);

        self.handles.clear();
        let mut walker = AxTreeWalker::new(self.handles.clone(), safari_mode);
        let walked_windows = if let Some(windows) = copy_array_attribute(app, "AXWindows")? {
            if cf_array_len(&windows)? > 0 {
                walker.walk_children_array(&windows, 0, false)?;
                true
            } else {
                false
            }
        } else {
            false
        };

        if !walked_windows {
            walker.walk_element(app, 0, false)?;
        }

        let mut elements = filter_candidates(walker.candidates);
        if safari_mode {
            if let Some(web_area) = largest_visible_web_area(&walker.web_areas) {
                let start_id = elements.len().saturating_add(1) as u32;
                let remaining = MAX_OBSERVED_ELEMENTS.saturating_sub(elements.len());
                if remaining > 0 {
                    let mut web_elements = safari_dom::observe_safari_dom(web_area, start_id)?;
                    web_elements.truncate(remaining);
                    log_suspect_safari_dom_geometry(web_area, &web_elements);
                    elements.extend(web_elements);
                }
            }
        }

        Ok(elements)
    }

    pub(crate) fn hit_test(
        &self,
        point: ClickPoint,
    ) -> Result<Option<TargetSummary>, ObservationError> {
        ensure_accessibility_permission_with(system_accessibility_trusted, prompt_accessibility)?;

        let system_wide = unsafe { AXUIElementCreateSystemWide() };
        if system_wide.is_null() {
            return Err(ObservationError::AxReadFailed(
                "AX system-wide element unavailable".into(),
            ));
        }
        let _system_wide_ref = OwnedCf::new(system_wide);

        let Some(hit) = copy_element_at_position(system_wide, point)? else {
            return Ok(None);
        };

        let own_pid = std::process::id() as Pid;
        if element_pid(hit.as_type_ref()) != Some(own_pid) {
            return Ok(Some(copy_target_summary(hit.as_type_ref())?));
        }

        // The topmost element here belongs to this process (HUD/confirmation
        // panel), which is click-through for real input. Re-run the hit-test
        // scoped to the frontmost app so preflight validates against what a
        // click would actually reach.
        eprintln!(
            "[screenie] agent hit-test skipped own window at ({}, {})",
            point.x, point.y
        );
        let Some(app) = frontmost_application_info()? else {
            return Ok(None);
        };
        let Some(app_pid) = app.pid.filter(|pid| *pid != own_pid) else {
            return Ok(None);
        };
        let app_element = unsafe { AXUIElementCreateApplication(app_pid) };
        if app_element.is_null() {
            return Ok(None);
        }
        let app_element = OwnedCf::new(app_element);
        match copy_element_at_position(app_element.as_type_ref(), point) {
            Ok(Some(app_hit)) => Ok(Some(copy_target_summary(app_hit.as_type_ref())?)),
            Ok(None) | Err(_) => Ok(None),
        }
    }

    fn refresh_by_signature(&self, el: &Element) -> Option<Element> {
        let matches = self
            .observe()
            .ok()?
            .into_iter()
            .filter(|candidate| candidate.signature == el.signature)
            .collect::<Vec<_>>();
        if matches.len() == 1 {
            let mut refreshed = matches.into_iter().next()?;
            refreshed.id = el.id;
            Some(refreshed)
        } else {
            None
        }
    }

    /// Collect readable text (names + values of every non-structural AX node,
    /// including static text) from the frontmost app, in tree order. Used by
    /// the agent's readPage action for non-Safari apps.
    pub(crate) fn read_static_text(&self) -> Result<String, ObservationError> {
        const MAX_STATIC_TEXT_BYTES: usize = 32_768;

        ensure_accessibility_permission_with(system_accessibility_trusted, prompt_accessibility)?;

        let focused_app = frontmost_application_info()?.ok_or(ObservationError::NoFrontmostApp)?;
        let pid = focused_app.pid.ok_or(ObservationError::NoFrontmostApp)?;
        let safari_mode = is_safari_app(&focused_app);
        let app = unsafe { AXUIElementCreateApplication(pid) };
        if app.is_null() {
            return Err(ObservationError::NoFrontmostApp);
        }
        let _app_ref = OwnedCf::new(app);
        self.ensure_app_ax_unlocked(pid, app);

        let mut walker = AxTreeWalker::new(AxHandleRegistry::default(), safari_mode);
        let walked_windows = if let Some(windows) = copy_array_attribute(app, "AXWindows")? {
            if cf_array_len(&windows)? > 0 {
                walker.walk_children_array(&windows, 0, false)?;
                true
            } else {
                false
            }
        } else {
            false
        };
        if !walked_windows {
            walker.walk_element(app, 0, false)?;
        }

        let mut text = String::new();
        for candidate in walker.candidates {
            for part in [
                candidate.name.as_str(),
                candidate.value.as_deref().unwrap_or(""),
            ] {
                let part = part.trim();
                if part.is_empty() {
                    continue;
                }
                if text.len() + part.len() + 1 > MAX_STATIC_TEXT_BYTES {
                    return Ok(text);
                }
                if !text.is_empty() {
                    text.push(' ');
                }
                text.push_str(part);
            }
        }
        Ok(text)
    }

    fn current_safari_web_area(&self) -> Result<Option<Rect>, ObservationError> {
        ensure_accessibility_permission_with(system_accessibility_trusted, prompt_accessibility)?;

        let focused_app = frontmost_application_info()?.ok_or(ObservationError::NoFrontmostApp)?;
        if !is_safari_app(&focused_app) {
            return Ok(None);
        }
        let pid = focused_app.pid.ok_or(ObservationError::NoFrontmostApp)?;
        let app = unsafe { AXUIElementCreateApplication(pid) };
        if app.is_null() {
            return Err(ObservationError::NoFrontmostApp);
        }
        let _app_ref = OwnedCf::new(app);

        let mut walker = AxTreeWalker::new(AxHandleRegistry::default(), true);
        let walked_windows = if let Some(windows) = copy_array_attribute(app, "AXWindows")? {
            if cf_array_len(&windows)? > 0 {
                walker.walk_children_array(&windows, 0, false)?;
                true
            } else {
                false
            }
        } else {
            false
        };
        if !walked_windows {
            walker.walk_element(app, 0, false)?;
        }

        Ok(largest_visible_web_area(&walker.web_areas))
    }

    /// Live scroll geometry for the agent's scroll anchor + postcondition
    /// (WI-3): focused window frame, largest visible scroll container under
    /// it, that container's vertical scrollbar position, and the display
    /// bounds for clamping.
    fn read_scroll_context(&self) -> Result<Option<ScrollContext>, ObservationError> {
        ensure_accessibility_permission_with(system_accessibility_trusted, prompt_accessibility)?;
        let focused_app = frontmost_application_info()?.ok_or(ObservationError::NoFrontmostApp)?;
        let pid = focused_app.pid.ok_or(ObservationError::NoFrontmostApp)?;
        let app = unsafe { AXUIElementCreateApplication(pid) };
        if app.is_null() {
            return Ok(None);
        }
        let _app_ref = OwnedCf::new(app);

        let Some(window) = copy_attribute(app, "AXFocusedWindow")? else {
            return Ok(None);
        };
        let window_bounds = copy_bounds(window.as_type_ref())?;
        let window_rect = rect_has_visible_bounds(window_bounds).then_some(window_bounds);
        let screen = main_display_bounds();

        let mut best: Option<ScrollContainerCandidate> = None;
        let mut visited = 0_usize;
        find_largest_scroll_container(
            window.as_type_ref(),
            0,
            window_rect,
            screen,
            &mut best,
            &mut visited,
        )?;
        let (container, container_kind, vertical_position) = match best.as_ref() {
            Some(candidate) => (
                Some(candidate.bounds),
                Some(candidate.kind),
                container_vertical_position(candidate.handle.as_type_ref()).unwrap_or(None),
            ),
            None => (None, None, None),
        };

        Ok(Some(ScrollContext {
            container,
            container_kind,
            window: window_rect,
            screen,
            vertical_position,
        }))
    }
}

impl ScreenObserver for MacObserver {
    fn observe(&self) -> Result<Vec<Element>, ObservationError> {
        self.observe_frontmost_app()
    }

    fn scroll_context(&self) -> Option<ScrollContext> {
        self.read_scroll_context().ok().flatten()
    }

    /// Event-driven settle support: ensure an AXObserver is registered on
    /// the frontmost app and hand out a waiter on the shared change counter.
    /// Never prompts for Accessibility; any unavailability returns `None`
    /// and the caller polls instead.
    fn change_signal(&self) -> Option<Box<dyn UiChangeSignal>> {
        if !system_accessibility_trusted() {
            return None;
        }
        let pid = frontmost_application_info().ok().flatten()?.pid?;
        let hub = ax_change_hub();
        let mut slot = hub.slot.lock().unwrap_or_else(PoisonError::into_inner);
        if slot.active.as_ref().map(|reg| reg.pid) != Some(pid) {
            if slot.failed_pid == Some(pid) {
                return None;
            }
            // Drop the previous app's registration before creating the new
            // one so at most one AXObserver source is ever scheduled.
            slot.active = None;
            match register_ax_change_observer(pid) {
                Some(registration) => {
                    slot.active = Some(registration);
                    slot.failed_pid = None;
                }
                None => {
                    eprintln!(
                        "[screenie] agent ax-signal registration failed pid={pid}; settle falls back to polling"
                    );
                    slot.failed_pid = Some(pid);
                    return None;
                }
            }
        }
        Some(Box::new(hub.counter.waiter()))
    }

    fn read_page_text(&self) -> Result<String, String> {
        let focused = self.focused_app().map_err(|err| err.to_string())?;
        super::page_reader::read_focused_page_text(self, &focused)
    }

    fn perform_press(&self, el: &Element) -> Result<bool, String> {
        let Some(PlatformElementHandle::MacAx { key }) = el.platform_handle.as_ref() else {
            return Ok(false);
        };
        let Some(handle) = self.handles.get(*key) else {
            return Ok(false);
        };
        let action = CFString::new("AXPress");
        let err = unsafe { AXUIElementPerformAction(handle, action.as_concrete_TypeRef()) };
        // Any non-success means no press happened, so the synthetic-click
        // fallback is always safe; a press that "succeeded" without effect is
        // caught by the caller's post-action verification.
        Ok(err == AX_ERROR_SUCCESS)
    }

    fn set_value(&self, el: &Element, text: &str) -> Result<bool, String> {
        let Some(PlatformElementHandle::MacAx { key }) = el.platform_handle.as_ref() else {
            return Ok(false);
        };
        let Some(handle) = self.handles.get(*key) else {
            return Ok(false);
        };

        // Focus first: bail out before writing any text unless keyboard focus
        // verifiably lands on the field. A following "key Return" must go to
        // this field, and a half-finished set_value may leave at worst a
        // focused field — never stray text the synthetic fallback would
        // double up on.
        let focused_attr = CFString::new("AXFocused");
        let truthy = CFBoolean::true_value();
        unsafe {
            let _ = AXUIElementSetAttributeValue(
                handle,
                focused_attr.as_concrete_TypeRef(),
                truthy.as_CFTypeRef(),
            );
        }
        let focused = copy_bool_attribute(handle, "AXFocused").map_err(|err| err.to_string())?;
        if focused != Some(true) {
            return Ok(false);
        }

        let value = CFString::new(text);
        let value_attr = CFString::new("AXValue");
        let err = unsafe {
            AXUIElementSetAttributeValue(
                handle,
                value_attr.as_concrete_TypeRef(),
                value.as_CFTypeRef(),
            )
        };
        if err != AX_ERROR_SUCCESS {
            return Ok(false);
        }

        // Read back: WebKit bodies and JS-backed inputs accept the write and
        // silently ignore it; only a matching read-back counts as success.
        let written =
            copy_value_string_attribute(handle, "AXValue").map_err(|err| err.to_string())?;
        let matches = match written.as_deref() {
            Some(value) => value.trim() == text.trim(),
            None => text.trim().is_empty(),
        };
        Ok(matches)
    }

    fn press_menu_path(&self, path: &[String]) -> Result<MenuPressOutcome, String> {
        ensure_accessibility_permission_with(system_accessibility_trusted, prompt_accessibility)
            .map_err(|err| err.to_string())?;

        let focused_app = frontmost_application_info()
            .map_err(|err| err.to_string())?
            .ok_or("no frontmost application")?;
        let pid = focused_app.pid.ok_or("no frontmost application pid")?;
        let app = unsafe { AXUIElementCreateApplication(pid) };
        if app.is_null() {
            return Err("no frontmost application".into());
        }
        let _app_ref = OwnedCf::new(app);

        let Some(menu_bar) = copy_attribute(app, "AXMenuBar").map_err(|err| err.to_string())?
        else {
            return Ok(MenuPressOutcome::NotFound {
                depth: 0,
                available: Vec::new(),
            });
        };

        let mut current = menu_bar;
        let mut resolved_path = Vec::with_capacity(path.len());
        for (depth, wanted) in path.iter().enumerate() {
            let items = menu_level_items(current.as_type_ref())?;
            let titles = items.iter().map(|(_, title)| title.clone()).collect::<Vec<_>>();
            let Some(index) = match_menu_title(&titles, wanted) else {
                return Ok(MenuPressOutcome::NotFound {
                    depth,
                    available: titles
                        .into_iter()
                        .filter(|title| !title.is_empty())
                        .take(MAX_MENU_TITLES_IN_FEEDBACK)
                        .collect(),
                });
            };
            let (element, title) = items
                .into_iter()
                .nth(index)
                .expect("matched index is within items");
            resolved_path.push(title);
            current = element;
        }

        if copy_bool_attribute(current.as_type_ref(), "AXEnabled")
            .map_err(|err| err.to_string())?
            == Some(false)
        {
            return Err(format!(
                "menu item '{}' is disabled right now",
                resolved_path.join(" > ")
            ));
        }

        // Pressing a deep AXMenuItem triggers it without opening the parent
        // menus on screen — the tree is fully readable while menus are closed.
        let action = CFString::new("AXPress");
        let err =
            unsafe { AXUIElementPerformAction(current.as_type_ref(), action.as_concrete_TypeRef()) };
        if err != AX_ERROR_SUCCESS {
            return Err(format!(
                "pressing menu item '{}' failed (AX error {err})",
                resolved_path.join(" > ")
            ));
        }
        Ok(MenuPressOutcome::Pressed { resolved_path })
    }

    fn search_menu_tree(&self, query: &str, max_results: usize) -> Result<MenuScanResult, String> {
        ensure_accessibility_permission_with(system_accessibility_trusted, prompt_accessibility)
            .map_err(|err| err.to_string())?;

        let focused_app = frontmost_application_info()
            .map_err(|err| err.to_string())?
            .ok_or("no frontmost application")?;
        let pid = focused_app.pid.ok_or("no frontmost application pid")?;
        let app = unsafe { AXUIElementCreateApplication(pid) };
        if app.is_null() {
            return Err("no frontmost application".into());
        }
        let _app_ref = OwnedCf::new(app);

        let Some(menu_bar) = copy_attribute(app, "AXMenuBar").map_err(|err| err.to_string())?
        else {
            return Ok(MenuScanResult::default());
        };

        Ok(scan_menu_tree(menu_bar.as_type_ref(), query, max_results))
    }

    fn refresh_element(&self, el: &Element) -> Option<Element> {
        match el.source {
            ElementSource::Ax => {
                if let Some(PlatformElementHandle::MacAx { key }) = el.platform_handle.as_ref() {
                    if let Some(handle) = self.handles.get(*key) {
                        if let Ok(snapshot) = copy_element_snapshot(
                            handle,
                            el.id,
                            Some(PlatformElementHandle::MacAx { key: *key }),
                        ) {
                            if element_identity_matches(el, &snapshot) {
                                return Some(snapshot);
                            }
                        }
                    }
                }
            }
            ElementSource::Web => {
                if let Some(PlatformElementHandle::SafariDom { agent_id }) =
                    el.platform_handle.as_ref()
                {
                    let web_area = match self.current_safari_web_area() {
                        Ok(Some(area)) => area,
                        Ok(None) => {
                            eprintln!(
                                "[screenie] agent safari-dom refresh: no web area while refreshing '{}'",
                                el.name
                            );
                            return None;
                        }
                        Err(err) => {
                            eprintln!(
                                "[screenie] agent safari-dom refresh: web area lookup failed while refreshing '{}': {err}",
                                el.name
                            );
                            return None;
                        }
                    };
                    match safari_dom::refresh_safari_dom_element(agent_id, web_area, el.id) {
                        Ok(Some(refreshed)) => {
                            if element_identity_matches(el, &refreshed) {
                                return Some(refreshed);
                            }
                            eprintln!(
                                "[screenie] agent safari-dom refresh: identity mismatch for '{}' (role '{}' -> '{}', name '{}' -> '{}')",
                                el.name, el.role, refreshed.role, el.name, refreshed.name
                            );
                        }
                        Ok(None) => {
                            eprintln!(
                                "[screenie] agent safari-dom refresh: element '{}' no longer found",
                                el.name
                            );
                        }
                        Err(err) => {
                            eprintln!(
                                "[screenie] agent safari-dom refresh failed for '{}': {err}",
                                el.name
                            );
                        }
                    }
                    return None;
                }
            }
            ElementSource::VisionDetected | ElementSource::VisionCoordinate => {}
        }

        self.refresh_by_signature(el)
    }
}

impl FocusedAppProvider for MacObserver {
    fn focused_app(&self) -> Result<FocusedApp, ObservationError> {
        frontmost_application_info()?.ok_or(ObservationError::NoFrontmostApp)
    }
}

impl CalibrationProbe for MacObserver {
    fn hit_test(&self, point: ClickPoint) -> Result<Option<TargetSummary>, String> {
        MacObserver::hit_test(self, point).map_err(|err| err.to_string())
    }
}

impl ObservationMetadataProvider for MacObserver {
    fn observation_metadata(&self) -> ObservationMetadata {
        ObservationMetadata::default()
    }
}

#[derive(Default)]
struct AxTreeWalker {
    candidates: Vec<ObservedCandidate>,
    handles: AxHandleRegistry,
    safari_mode: bool,
    web_areas: Vec<Rect>,
}

impl AxTreeWalker {
    fn new(handles: AxHandleRegistry, safari_mode: bool) -> Self {
        Self {
            candidates: Vec::new(),
            handles,
            safari_mode,
            web_areas: Vec::new(),
        }
    }

    fn walk_children_array(
        &mut self,
        array: &OwnedCf,
        depth: usize,
        inside_web_area: bool,
    ) -> Result<(), ObservationError> {
        let count = cf_array_len(array)?;
        let capped_count = count.min(MAX_AX_CHILDREN_PER_NODE);
        for index in 0..capped_count {
            let child = unsafe {
                CFArrayGetValueAtIndex(array.as_array_ref()?, index as isize) as AXUIElementRef
            };
            if !child.is_null() {
                self.walk_element(child, depth, inside_web_area)?;
            }
        }
        Ok(())
    }

    fn walk_element(
        &mut self,
        element: AXUIElementRef,
        depth: usize,
        inside_web_area: bool,
    ) -> Result<(), ObservationError> {
        let max_depth = if inside_web_area {
            MAX_AX_WEB_DEPTH
        } else {
            MAX_AX_DEPTH
        };
        if depth > max_depth {
            return Ok(());
        }

        let role = copy_string_attribute(element, "AXRole")?.unwrap_or_default();
        let title = copy_nonempty_string_attribute(element, "AXTitle")?;
        let description = copy_nonempty_string_attribute(element, "AXDescription")?;
        let name = title.or(description).unwrap_or_default();
        // Secure fields: never read the value, even though macOS usually
        // masks it — belt and braces against contents reaching logs/prompts.
        let value = if is_secure_text_role(&role) {
            None
        } else {
            copy_value_string_attribute(element, "AXValue")?
        };
        let enabled = copy_bool_attribute(element, "AXEnabled")?.unwrap_or(true);
        let focused = copy_bool_attribute(element, "AXFocused")?.unwrap_or(false);
        // Selection only matters (and is only cheap to read) for the focused
        // element; it feeds the semantic state hash so select-all/deselect
        // counts as UI progress.
        let selected_text = if focused && !is_secure_text_role(&role) {
            copy_nonempty_string_attribute(element, "AXSelectedText")?
        } else {
            None
        };
        let bounds = copy_bounds(element)?;

        let enters_web_area = role == "AXWebArea";
        let now_inside_web_area = inside_web_area || enters_web_area;
        if enters_web_area && rect_has_visible_bounds(bounds) {
            self.web_areas.push(bounds);
            if self.safari_mode {
                return Ok(());
            }
        }

        if !role.is_empty() && !is_structural_container_role(&role) {
            let platform_handle = Some(self.handles.register(element));
            self.candidates.push(ObservedCandidate {
                role: role.clone(),
                name,
                value,
                bounds,
                enabled,
                focused,
                selected_text,
                platform_handle,
            });
        }

        if depth == max_depth {
            return Ok(());
        }

        if let Some(children) = copy_array_attribute(element, "AXChildren")? {
            let child_depth = if now_inside_web_area && is_structural_container_role(&role) {
                depth
            } else {
                depth + 1
            };
            self.walk_children_array(&children, child_depth, now_inside_web_area)?;
        }

        Ok(())
    }
}

#[derive(Clone, Default)]
struct AxHandleRegistry {
    inner: Rc<RefCell<AxHandleRegistryInner>>,
}

impl AxHandleRegistry {
    fn clear(&self) {
        self.inner.borrow_mut().handles.clear();
    }

    fn register(&self, element: AXUIElementRef) -> PlatformElementHandle {
        let retained = unsafe { CFRetain(element) };
        let mut inner = self.inner.borrow_mut();
        let key = inner.next_key;
        inner.next_key = inner.next_key.saturating_add(1).max(1);
        inner.handles.insert(key, OwnedCf::new(retained));
        PlatformElementHandle::MacAx { key }
    }

    fn get(&self, key: u64) -> Option<AXUIElementRef> {
        self.inner
            .borrow()
            .handles
            .get(&key)
            .map(|handle| handle.as_type_ref())
    }
}

impl fmt::Debug for AxHandleRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AxHandleRegistry")
            .field("len", &self.inner.borrow().handles.len())
            .finish()
    }
}

#[derive(Default)]
struct AxHandleRegistryInner {
    next_key: u64,
    handles: HashMap<u64, OwnedCf>,
}

struct OwnedCf(CFTypeRef);

impl OwnedCf {
    fn new(value: CFTypeRef) -> Self {
        Self(value)
    }

    fn as_type_ref(&self) -> CFTypeRef {
        self.0
    }

    fn type_id(&self) -> CFTypeID {
        unsafe { CFGetTypeID(self.0) }
    }

    fn has_type_id(&self, expected: CFTypeID) -> bool {
        !self.0.is_null() && self.type_id() == expected
    }

    fn as_array_ref(&self) -> Result<CFArrayRef, ObservationError> {
        if self.has_type_id(unsafe { CFArrayGetTypeID() }) {
            Ok(self.0 as CFArrayRef)
        } else {
            Err(ObservationError::AxReadFailed(
                "AX attribute was not a CFArray".into(),
            ))
        }
    }
}

impl Drop for OwnedCf {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { CFRelease(self.0) };
        }
    }
}

fn system_accessibility_trusted() -> bool {
    unsafe { AXIsProcessTrusted() != 0 }
}

/// One unlock attempt per pid per observer lifetime; repeated writes are
/// wasted IPC round-trips into the target app.
fn should_attempt_ax_unlock(unlocked: &mut HashSet<Pid>, pid: Pid) -> bool {
    unlocked.insert(pid)
}

const MAX_MENU_TITLES_IN_FEEDBACK: usize = 30;
const MAX_MENU_WRAPPER_DEPTH: usize = 2;

/// Children of a menu element, with the interposed `AXMenu` wrapper unwrapped:
/// an `AXMenuBarItem`/`AXMenuItem` parents a single `AXMenu` whose children
/// are the actual items.
fn menu_level_items(element: AXUIElementRef) -> Result<Vec<(OwnedCf, String)>, String> {
    let mut items = Vec::new();
    collect_menu_children(element, 0, &mut items)?;
    Ok(items)
}

fn collect_menu_children(
    element: AXUIElementRef,
    wrapper_depth: usize,
    items: &mut Vec<(OwnedCf, String)>,
) -> Result<(), String> {
    let Some(children) =
        copy_array_attribute(element, "AXChildren").map_err(|err| err.to_string())?
    else {
        return Ok(());
    };
    let count = cf_array_len(&children)
        .map_err(|err| err.to_string())?
        .min(MAX_AX_CHILDREN_PER_NODE);
    for index in 0..count {
        let child = unsafe {
            CFArrayGetValueAtIndex(
                children.as_array_ref().map_err(|err| err.to_string())?,
                index as isize,
            ) as AXUIElementRef
        };
        if child.is_null() {
            continue;
        }
        let role = copy_string_attribute(child, "AXRole")
            .map_err(|err| err.to_string())?
            .unwrap_or_default();
        if role == "AXMenu" {
            if wrapper_depth < MAX_MENU_WRAPPER_DEPTH {
                collect_menu_children(child, wrapper_depth + 1, items)?;
            }
            continue;
        }
        let title = copy_nonempty_string_attribute(child, "AXTitle")
            .map_err(|err| err.to_string())?
            .unwrap_or_default();
        let retained = unsafe { CFRetain(child) };
        items.push((OwnedCf::new(retained), title));
    }
    Ok(())
}

const MAX_MENU_SCAN_DEPTH: usize = 4;
const MAX_MENU_SCAN_NODES: usize = 1500;
const MENU_SCAN_BUDGET_MS: u64 = 350;

/// Read-only fuzzy search over the whole menu tree. The tree is fully
/// readable while menus stay closed (same property press_menu_path relies
/// on), so this never changes anything on screen. Bounded by depth, node
/// count, and wall clock so apps with huge menus (Xcode) stay cheap;
/// exceeding a budget sets `truncated` instead of failing.
fn scan_menu_tree(menu_bar: AXUIElementRef, query: &str, max_results: usize) -> MenuScanResult {
    let started = Instant::now();
    let budget = Duration::from_millis(MENU_SCAN_BUDGET_MS);
    let mut scored: Vec<(u32, MenuMatch)> = Vec::new();
    let mut truncated = false;
    let mut visited = 0usize;

    let mut stack: Vec<(OwnedCf, Vec<String>, usize)> = menu_level_items(menu_bar)
        .unwrap_or_default()
        .into_iter()
        .filter(|(_, title)| !title.trim().is_empty())
        .map(|(element, title)| (element, vec![title], 1))
        .collect();
    // Reverse so the leftmost menu (File before Help) is scanned first when
    // budgets bite.
    stack.reverse();

    while let Some((element, path, depth)) = stack.pop() {
        visited += 1;
        if visited > MAX_MENU_SCAN_NODES || started.elapsed() >= budget {
            truncated = true;
            break;
        }

        let title = path.last().map(String::as_str).unwrap_or_default();
        // Score the item title and the joined path so multi-level queries
        // ("safari settings advanced") can match too.
        let score = match (score_match(title, query), score_match(&path.join(" "), query)) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        };
        if let Some(score) = score {
            // AXEnabled read only for hits — it is the expensive part.
            let enabled = copy_bool_attribute(element.as_type_ref(), "AXEnabled")
                .ok()
                .flatten()
                .unwrap_or(true);
            scored.push((
                score,
                MenuMatch {
                    path: path.clone(),
                    enabled,
                },
            ));
        }

        if depth < MAX_MENU_SCAN_DEPTH {
            // One unreadable submenu shouldn't kill the scan.
            let children = menu_level_items(element.as_type_ref()).unwrap_or_default();
            for (child, child_title) in children.into_iter().rev() {
                if !child_title.trim().is_empty() {
                    let mut child_path = path.clone();
                    child_path.push(child_title);
                    stack.push((child, child_path, depth + 1));
                }
            }
        }
    }

    scored.sort_by(|a, b| b.0.cmp(&a.0));
    MenuScanResult {
        matches: scored
            .into_iter()
            .take(max_results)
            .map(|(_, entry)| entry)
            .collect(),
        truncated,
    }
}

/// Match a model-supplied menu title against the level's real titles:
/// normalized exact match first, then a unique prefix match. Returns the
/// index of the matched title.
pub(crate) fn match_menu_title(titles: &[String], wanted: &str) -> Option<usize> {
    let wanted = normalize_menu_title(wanted);
    if wanted.is_empty() {
        return None;
    }
    if let Some(index) = titles
        .iter()
        .position(|title| normalize_menu_title(title) == wanted)
    {
        return Some(index);
    }
    let prefix_matches = titles
        .iter()
        .enumerate()
        .filter(|(_, title)| normalize_menu_title(title).starts_with(&wanted))
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    if prefix_matches.len() == 1 {
        return Some(prefix_matches[0]);
    }
    None
}

fn normalize_menu_title(title: &str) -> String {
    title
        .trim()
        .trim_end_matches('…')
        .trim_end_matches("...")
        .trim()
        .to_ascii_lowercase()
}

fn prompt_accessibility() {
    let key = CFString::new("AXTrustedCheckOptionPrompt");
    let value = CFBoolean::true_value();
    let options = CFDictionary::from_CFType_pairs(&[(key, value)]);
    unsafe {
        let _ = AXIsProcessTrustedWithOptions(options.as_concrete_TypeRef());
    }
}

pub(crate) fn is_safari_app(app: &FocusedApp) -> bool {
    if let Some(bundle_id) = app.bundle_id.as_deref() {
        return bundle_id == "com.apple.Safari";
    }
    app.name.eq_ignore_ascii_case("Safari")
}

fn largest_visible_web_area(web_areas: &[Rect]) -> Option<Rect> {
    web_areas
        .iter()
        .copied()
        .filter(|rect| rect_has_visible_bounds(*rect))
        .max_by(|a, b| {
            rect_area(*a)
                .partial_cmp(&rect_area(*b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

fn rect_has_visible_bounds(rect: Rect) -> bool {
    rect.x.is_finite()
        && rect.y.is_finite()
        && rect.width.is_finite()
        && rect.height.is_finite()
        && rect.width > 0.0
        && rect.height > 0.0
}

fn rect_area(rect: Rect) -> f64 {
    rect.width.max(0.0) * rect.height.max(0.0)
}

fn rects_overlap(a: Rect, b: Rect) -> bool {
    a.x < b.x + b.width && b.x < a.x + a.width && a.y < b.y + b.height && b.y < a.y + a.height
}

/// One log line per observation when a converted Safari DOM element lands
/// entirely off the display — the smoke signal for a wrong web-area origin
/// (e.g. an unhandled AXWebArea shape after scrolling).
fn log_suspect_safari_dom_geometry(web_area: Rect, web_elements: &[Element]) {
    let Some(display) = main_display_bounds() else {
        return;
    };
    if let Some(stray) = web_elements
        .iter()
        .find(|element| !rects_overlap(element.bounds, display))
    {
        eprintln!(
            "[screenie] agent safari-dom geometry suspect web_area={:?} element={:?} bounds={:?}",
            web_area, stray.name, stray.bounds
        );
    }
}

fn frontmost_application_info() -> Result<Option<FocusedApp>, ObservationError> {
    unsafe {
        let workspace_class = objc_getClass(cstr_ptr(b"NSWorkspace\0"));
        if workspace_class.is_null() {
            return Err(ObservationError::AxReadFailed(
                "NSWorkspace class unavailable".into(),
            ));
        }

        let workspace = objc_send_id(workspace_class, selector(b"sharedWorkspace\0"));
        if workspace.is_null() {
            return Ok(None);
        }

        let app = objc_send_id(workspace, selector(b"frontmostApplication\0"));
        if app.is_null() {
            return Ok(None);
        }

        let pid = objc_send_i32(app, selector(b"processIdentifier\0"));
        if pid <= 0 {
            Ok(None)
        } else {
            let name = objc_send_nsstring(app, selector(b"localizedName\0"))
                .or_else(|| objc_send_nsstring(app, selector(b"bundleIdentifier\0")))
                .unwrap_or_else(|| "Unknown".into());
            let bundle_id = objc_send_nsstring(app, selector(b"bundleIdentifier\0"))
                .filter(|value| !value.trim().is_empty());
            Ok(Some(FocusedApp {
                bundle_id,
                name,
                pid: Some(pid),
            }))
        }
    }
}

unsafe fn objc_send_id(receiver: ObjcId, selector: Sel) -> ObjcId {
    let send: unsafe extern "C" fn(ObjcId, Sel) -> ObjcId =
        mem::transmute(objc_msgSend as *const ());
    send(receiver, selector)
}

unsafe fn objc_send_i32(receiver: ObjcId, selector: Sel) -> i32 {
    let send: unsafe extern "C" fn(ObjcId, Sel) -> i32 = mem::transmute(objc_msgSend as *const ());
    send(receiver, selector)
}

unsafe fn objc_send_nsstring(receiver: ObjcId, sel: Sel) -> Option<String> {
    let nsstring = objc_send_id(receiver, sel);
    if nsstring.is_null() {
        return None;
    }
    let send: unsafe extern "C" fn(ObjcId, Sel) -> *const c_char =
        mem::transmute(objc_msgSend as *const ());
    let ptr = send(nsstring, selector(b"UTF8String\0"));
    if ptr.is_null() {
        None
    } else {
        Some(CStr::from_ptr(ptr).to_string_lossy().into_owned())
    }
}

fn selector(name: &'static [u8]) -> Sel {
    unsafe { sel_registerName(cstr_ptr(name)) }
}

fn cstr_ptr(bytes: &'static [u8]) -> *const c_char {
    bytes.as_ptr().cast()
}

fn copy_attribute(
    element: AXUIElementRef,
    attribute: &str,
) -> Result<Option<OwnedCf>, ObservationError> {
    let attribute = CFString::new(attribute);
    let mut value = ptr::null();
    let err = unsafe {
        AXUIElementCopyAttributeValue(element, attribute.as_concrete_TypeRef(), &mut value)
    };

    match err {
        AX_ERROR_SUCCESS if value.is_null() => Ok(None),
        AX_ERROR_SUCCESS => Ok(Some(OwnedCf::new(value))),
        other if ax_error_means_missing_or_stale(other) => Ok(None),
        AX_ERROR_CANNOT_COMPLETE => Err(ObservationError::AxReadFailed(
            "AX server could not complete the request".into(),
        )),
        other => Err(ObservationError::AxReadFailed(format!(
            "AXUIElementCopyAttributeValue({}) returned {}",
            attribute,
            other
        ))),
    }
}

fn ax_error_means_missing_or_stale(err: i32) -> bool {
    matches!(
        err,
        AX_ERROR_INVALID_UI_ELEMENT | AX_ERROR_ATTRIBUTE_UNSUPPORTED | AX_ERROR_NO_VALUE
    )
}

fn copy_string_attribute(
    element: AXUIElementRef,
    attribute: &str,
) -> Result<Option<String>, ObservationError> {
    let Some(value) = copy_attribute(element, attribute)? else {
        return Ok(None);
    };
    cf_string_value(&value)
}

fn copy_nonempty_string_attribute(
    element: AXUIElementRef,
    attribute: &str,
) -> Result<Option<String>, ObservationError> {
    Ok(copy_string_attribute(element, attribute)?
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty()))
}

fn copy_value_string_attribute(
    element: AXUIElementRef,
    attribute: &str,
) -> Result<Option<String>, ObservationError> {
    let Some(value) = copy_attribute(element, attribute)? else {
        return Ok(None);
    };
    Ok(cf_value_to_string(&value)
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty()))
}

fn copy_bool_attribute(
    element: AXUIElementRef,
    attribute: &str,
) -> Result<Option<bool>, ObservationError> {
    let Some(value) = copy_attribute(element, attribute)? else {
        return Ok(None);
    };
    if !value.has_type_id(unsafe { CFBooleanGetTypeID() }) {
        return Ok(None);
    }
    Ok(Some(unsafe {
        CFBooleanGetValue(value.as_type_ref() as CFBooleanRef)
    }))
}

fn copy_number_attribute(
    element: AXUIElementRef,
    attribute: &str,
) -> Result<Option<f64>, ObservationError> {
    let Some(value) = copy_attribute(element, attribute)? else {
        return Ok(None);
    };
    if !value.has_type_id(unsafe { CFNumberGetTypeID() }) {
        return Ok(None);
    }
    let mut number = 0.0_f64;
    let ok = unsafe {
        CFNumberGetValue(
            value.as_type_ref() as CFNumberRef,
            kCFNumberFloat64Type,
            (&mut number as *mut f64).cast(),
        )
    };
    Ok(ok.then_some(number))
}

// Safari's tab-bar subtree alone can consume hundreds of pre-order nodes
// before the DFS reaches the web content group, and the web area sits deep
// in the window hierarchy — caps sized so chrome cannot starve the search.
const SCROLL_CONTAINER_MAX_DEPTH: usize = 12;
const SCROLL_CONTAINER_MAX_NODES: usize = 1000;

struct ScrollContainerCandidate {
    handle: OwnedCf,
    bounds: Rect,
    kind: ScrollContainerKind,
    rank: (u8, f64),
}

/// Rank a scroll-container candidate for anchoring: an AXWebArea beats any
/// AXScrollArea (Safari chrome — the tab bar — is an AXScrollArea), and ties
/// compare by the area actually visible on screen (bounds ∩ window ∩
/// screen), not raw area, so a mostly-offscreen full-document web area is
/// judged by its visible slice. `None` = not anchorable (degenerate bounds
/// or disjoint from the window/screen).
fn scroll_container_rank(
    kind: ScrollContainerKind,
    bounds: Rect,
    window: Option<Rect>,
    screen: Option<Rect>,
) -> Option<(u8, f64)> {
    if !rect_has_visible_bounds(bounds) {
        return None;
    }
    let mut visible = bounds;
    for clamp in [window, screen].into_iter().flatten() {
        visible = intersect_rects(visible, clamp)?;
    }
    let priority = match kind {
        ScrollContainerKind::WebArea => 1,
        ScrollContainerKind::ScrollArea => 0,
    };
    Some((priority, rect_area(visible)))
}

/// Depth-first, node-capped search for the best-ranked visible scroll
/// container (AXWebArea or AXScrollArea) under `element`, retaining the
/// winning handle.
fn find_largest_scroll_container(
    element: AXUIElementRef,
    depth: usize,
    window: Option<Rect>,
    screen: Option<Rect>,
    best: &mut Option<ScrollContainerCandidate>,
    visited: &mut usize,
) -> Result<(), ObservationError> {
    if depth > SCROLL_CONTAINER_MAX_DEPTH || *visited >= SCROLL_CONTAINER_MAX_NODES {
        return Ok(());
    }
    *visited += 1;
    let role = copy_string_attribute(element, "AXRole")?.unwrap_or_default();
    let kind = match role.as_str() {
        "AXWebArea" => Some(ScrollContainerKind::WebArea),
        "AXScrollArea" => Some(ScrollContainerKind::ScrollArea),
        _ => None,
    };
    if let Some(kind) = kind {
        let bounds = copy_bounds(element)?;
        if let Some(rank) = scroll_container_rank(kind, bounds, window, screen) {
            let replace = best
                .as_ref()
                .map(|current| rank > current.rank)
                .unwrap_or(true);
            if replace {
                let retained = unsafe { CFRetain(element) };
                *best = Some(ScrollContainerCandidate {
                    handle: OwnedCf::new(retained),
                    bounds,
                    kind,
                    rank,
                });
            }
        }
        // The page scroller found; nested scroll areas inside it are not
        // the one the wheel should target.
        if kind == ScrollContainerKind::WebArea {
            return Ok(());
        }
    }
    if let Some(children) = copy_array_attribute(element, "AXChildren")? {
        let count = (cf_array_len(&children)? as usize).min(MAX_AX_CHILDREN_PER_NODE);
        for index in 0..count {
            let child = unsafe {
                CFArrayGetValueAtIndex(children.as_array_ref()?, index as isize) as AXUIElementRef
            };
            if !child.is_null() {
                find_largest_scroll_container(child, depth + 1, window, screen, best, visited)?;
            }
        }
    }
    Ok(())
}

/// AXVerticalScrollBar value of a scroll container, 0.0 (top) ..= 1.0.
fn container_vertical_position(container: AXUIElementRef) -> Result<Option<f64>, ObservationError> {
    let Some(scrollbar) = copy_attribute(container, "AXVerticalScrollBar")? else {
        return Ok(None);
    };
    copy_number_attribute(scrollbar.as_type_ref(), "AXValue")
}

fn main_display_bounds() -> Option<Rect> {
    // TODO multi-display: clamp against the display actually hosting the
    // focused window instead of the main display.
    let bounds = unsafe { CGDisplayBounds(CGMainDisplayID()) };
    (bounds.size.width > 0.0 && bounds.size.height > 0.0).then_some(Rect {
        x: bounds.origin.x,
        y: bounds.origin.y,
        width: bounds.size.width,
        height: bounds.size.height,
    })
}

fn copy_array_attribute(
    element: AXUIElementRef,
    attribute: &str,
) -> Result<Option<OwnedCf>, ObservationError> {
    let Some(value) = copy_attribute(element, attribute)? else {
        return Ok(None);
    };
    if value.has_type_id(unsafe { CFArrayGetTypeID() }) {
        Ok(Some(value))
    } else {
        Err(ObservationError::AxReadFailed(format!(
            "{attribute} was not a CFArray"
        )))
    }
}

fn copy_bounds(element: AXUIElementRef) -> Result<Rect, ObservationError> {
    let position = copy_point_attribute(element, "AXPosition")?;
    let size = copy_size_attribute(element, "AXSize")?;
    Ok(match (position, size) {
        (Some(position), Some(size)) => Rect {
            x: position.x,
            y: position.y,
            width: size.width,
            height: size.height,
        },
        _ => Rect {
            x: 0.0,
            y: 0.0,
            width: 0.0,
            height: 0.0,
        },
    })
}

fn copy_target_summary(element: AXUIElementRef) -> Result<TargetSummary, ObservationError> {
    let snapshot = copy_element_snapshot(element, 0, None)?;
    Ok(TargetSummary::from(&snapshot))
}

fn copy_element_at_position(
    root: AXUIElementRef,
    point: ClickPoint,
) -> Result<Option<OwnedCf>, ObservationError> {
    let mut hit = ptr::null();
    let err =
        unsafe { AXUIElementCopyElementAtPosition(root, point.x as f32, point.y as f32, &mut hit) };
    match err {
        AX_ERROR_SUCCESS if hit.is_null() => Ok(None),
        AX_ERROR_SUCCESS => Ok(Some(OwnedCf::new(hit))),
        other if ax_error_means_missing_or_stale(other) => Ok(None),
        AX_ERROR_CANNOT_COMPLETE => Err(ObservationError::AxReadFailed(
            "AX hit-test could not complete the request".into(),
        )),
        other => Err(ObservationError::AxReadFailed(format!(
            "AXUIElementCopyElementAtPosition({}, {}) returned {}",
            point.x, point.y, other
        ))),
    }
}

fn element_pid(element: AXUIElementRef) -> Option<Pid> {
    let mut pid: Pid = 0;
    let err = unsafe { AXUIElementGetPid(element, &mut pid) };
    (err == AX_ERROR_SUCCESS && pid > 0).then_some(pid)
}

fn copy_element_snapshot(
    element: AXUIElementRef,
    id: u32,
    platform_handle: Option<PlatformElementHandle>,
) -> Result<Element, ObservationError> {
    let role = copy_string_attribute(element, "AXRole")?.unwrap_or_default();
    let title = copy_nonempty_string_attribute(element, "AXTitle")?;
    let description = copy_nonempty_string_attribute(element, "AXDescription")?;
    let name = title.or(description).unwrap_or_default();
    let value = if is_secure_text_role(&role) {
        None
    } else {
        copy_value_string_attribute(element, "AXValue")?
    };
    let enabled = copy_bool_attribute(element, "AXEnabled")?.unwrap_or(true);
    let focused = copy_bool_attribute(element, "AXFocused")?.unwrap_or(false);
    let bounds = copy_bounds(element)?;

    let element = Element::new(
        id,
        role,
        name,
        value,
        bounds,
        enabled,
        focused,
        CoordinateSpace::AxPoints,
        ElementSource::Ax,
    );
    Ok(if let Some(platform_handle) = platform_handle {
        element.with_platform_handle(platform_handle)
    } else {
        element
    })
}

fn element_identity_matches(original: &Element, refreshed: &Element) -> bool {
    original.source == refreshed.source
        && original.role == refreshed.role
        && normalize_signature_name(&original.name, original.source)
            == normalize_signature_name(&refreshed.name, refreshed.source)
        && original.value.is_some() == refreshed.value.is_some()
}

fn copy_point_attribute(
    element: AXUIElementRef,
    attribute: &str,
) -> Result<Option<CGPoint>, ObservationError> {
    let Some(value) = copy_attribute(element, attribute)? else {
        return Ok(None);
    };
    if !value.has_type_id(unsafe { AXValueGetTypeID() }) {
        return Ok(None);
    }
    if unsafe { AXValueGetType(value.as_type_ref()) } != K_AX_VALUE_CGPOINT_TYPE {
        return Ok(None);
    }
    let mut point = CGPoint::default();
    let ok = unsafe {
        AXValueGetValue(
            value.as_type_ref(),
            K_AX_VALUE_CGPOINT_TYPE,
            (&mut point as *mut CGPoint).cast(),
        )
    };
    Ok((ok != 0).then_some(point))
}

fn copy_size_attribute(
    element: AXUIElementRef,
    attribute: &str,
) -> Result<Option<CGSize>, ObservationError> {
    let Some(value) = copy_attribute(element, attribute)? else {
        return Ok(None);
    };
    if !value.has_type_id(unsafe { AXValueGetTypeID() }) {
        return Ok(None);
    }
    if unsafe { AXValueGetType(value.as_type_ref()) } != K_AX_VALUE_CGSIZE_TYPE {
        return Ok(None);
    }
    let mut size = CGSize::default();
    let ok = unsafe {
        AXValueGetValue(
            value.as_type_ref(),
            K_AX_VALUE_CGSIZE_TYPE,
            (&mut size as *mut CGSize).cast(),
        )
    };
    Ok((ok != 0).then_some(size))
}

fn cf_array_len(array: &OwnedCf) -> Result<usize, ObservationError> {
    Ok(unsafe { CFArrayGetCount(array.as_array_ref()?) as usize })
}

fn cf_string_value(value: &OwnedCf) -> Result<Option<String>, ObservationError> {
    if !value.has_type_id(unsafe { CFStringGetTypeID() }) {
        return Ok(None);
    }
    let string =
        unsafe { CFString::wrap_under_get_rule(value.as_type_ref() as CFStringRef).to_string() };
    Ok(Some(string))
}

fn cf_value_to_string(value: &OwnedCf) -> Option<String> {
    if let Ok(Some(string)) = cf_string_value(value) {
        return Some(string);
    }
    if value.has_type_id(unsafe { CFBooleanGetTypeID() }) {
        let bool_value = unsafe { CFBooleanGetValue(value.as_type_ref() as CFBooleanRef) };
        return Some(bool_value.to_string());
    }

    let description = unsafe { CFCopyDescription(value.as_type_ref()) };
    if description.is_null() {
        return None;
    }
    let description = OwnedCf::new(description.cast());
    cf_string_value(&description).ok().flatten()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ax_signal_runloop_is_created_once_and_reused() {
        let first = ax_signal_runloop();
        let second = ax_signal_runloop();
        assert!(first.is_some());
        assert_eq!(first, second);
    }

    #[test]
    fn invalid_ui_element_is_treated_as_missing_or_stale() {
        assert!(ax_error_means_missing_or_stale(AX_ERROR_INVALID_UI_ELEMENT));
        assert!(ax_error_means_missing_or_stale(
            AX_ERROR_ATTRIBUTE_UNSUPPORTED
        ));
        assert!(ax_error_means_missing_or_stale(AX_ERROR_NO_VALUE));
        assert!(!ax_error_means_missing_or_stale(AX_ERROR_CANNOT_COMPLETE));
    }

    #[test]
    fn safari_detection_prefers_bundle_id_and_falls_back_to_name() {
        assert!(is_safari_app(&FocusedApp {
            bundle_id: Some("com.apple.Safari".into()),
            name: "Anything".into(),
            pid: Some(1),
        }));
        assert!(!is_safari_app(&FocusedApp {
            bundle_id: Some("com.example.SafariLike".into()),
            name: "Safari".into(),
            pid: Some(1),
        }));
        assert!(is_safari_app(&FocusedApp {
            bundle_id: None,
            name: "Safari".into(),
            pid: Some(1),
        }));
    }

    #[test]
    fn ax_unlock_is_attempted_once_per_pid() {
        let mut unlocked = HashSet::new();
        assert!(should_attempt_ax_unlock(&mut unlocked, 100));
        assert!(!should_attempt_ax_unlock(&mut unlocked, 100));
        assert!(should_attempt_ax_unlock(&mut unlocked, 200));
    }

    #[test]
    fn menu_title_match_normalizes_ellipsis_case_and_unique_prefix() {
        let titles = vec![
            "File".to_string(),
            "Export as PDF…".to_string(),
            "Export All".to_string(),
            "Print...".to_string(),
        ];

        assert_eq!(match_menu_title(&titles, "export as pdf"), Some(1));
        assert_eq!(match_menu_title(&titles, "Export as PDF…"), Some(1));
        assert_eq!(match_menu_title(&titles, "Print"), Some(3));
        assert_eq!(match_menu_title(&titles, "Export as"), Some(1));
        // "Export" prefixes two entries — ambiguous, no match.
        assert_eq!(match_menu_title(&titles, "Export"), None);
        assert_eq!(match_menu_title(&titles, "Quit"), None);
        assert_eq!(match_menu_title(&titles, "  "), None);
    }

    #[test]
    fn largest_visible_web_area_ignores_empty_bounds() {
        let selected = largest_visible_web_area(&[
            Rect {
                x: 0.0,
                y: 0.0,
                width: 0.0,
                height: 100.0,
            },
            Rect {
                x: 20.0,
                y: 30.0,
                width: 400.0,
                height: 300.0,
            },
            Rect {
                x: 10.0,
                y: 20.0,
                width: 200.0,
                height: 100.0,
            },
        ])
        .unwrap();

        assert_eq!(
            selected,
            Rect {
                x: 20.0,
                y: 30.0,
                width: 400.0,
                height: 300.0,
            }
        );
    }

    #[test]
    fn scroll_container_rank_prefers_web_area_over_larger_scroll_area() {
        // Safari's tab bar is an AXScrollArea spanning the window's full
        // width; the page's AXWebArea must outrank it regardless of area.
        let window = Some(Rect {
            x: 0.0,
            y: 0.0,
            width: 1440.0,
            height: 900.0,
        });
        let screen = window;
        let tab_bar = scroll_container_rank(
            ScrollContainerKind::ScrollArea,
            Rect {
                x: 3.0,
                y: 0.0,
                width: 1437.0,
                height: 94.0,
            },
            window,
            screen,
        )
        .unwrap();
        let web_area = scroll_container_rank(
            ScrollContainerKind::WebArea,
            Rect {
                x: 0.0,
                y: 90.0,
                width: 1440.0,
                height: 810.0,
            },
            window,
            screen,
        )
        .unwrap();
        assert!(web_area > tab_bar);

        // Even a tiny web area outranks a huge scroll area: role first.
        let tiny_web_area = scroll_container_rank(
            ScrollContainerKind::WebArea,
            Rect {
                x: 0.0,
                y: 90.0,
                width: 200.0,
                height: 100.0,
            },
            window,
            screen,
        )
        .unwrap();
        assert!(tiny_web_area > tab_bar);
    }

    #[test]
    fn scroll_container_rank_uses_visible_not_raw_area() {
        // A full-document web area scrolled mostly above the screen is
        // judged by its visible slice, not its (huge) raw area.
        let window = Some(Rect {
            x: 0.0,
            y: 0.0,
            width: 1440.0,
            height: 900.0,
        });
        let screen = window;
        let mostly_offscreen = scroll_container_rank(
            ScrollContainerKind::WebArea,
            Rect {
                x: 0.0,
                y: -5000.0,
                width: 1440.0,
                height: 5100.0,
            },
            window,
            screen,
        )
        .unwrap();
        let fully_visible = scroll_container_rank(
            ScrollContainerKind::WebArea,
            Rect {
                x: 0.0,
                y: 90.0,
                width: 1440.0,
                height: 810.0,
            },
            window,
            screen,
        )
        .unwrap();
        assert!(fully_visible > mostly_offscreen);
        // Visible slice = 1440 × 100.
        assert_eq!(mostly_offscreen.1, 1440.0 * 100.0);
    }

    #[test]
    fn scroll_container_rank_rejects_disjoint_candidates() {
        let window = Some(Rect {
            x: 0.0,
            y: 0.0,
            width: 1440.0,
            height: 900.0,
        });
        // Fully above the screen: no visible slice, not anchorable.
        assert_eq!(
            scroll_container_rank(
                ScrollContainerKind::WebArea,
                Rect {
                    x: 0.0,
                    y: -6000.0,
                    width: 1440.0,
                    height: 500.0,
                },
                window,
                window,
            ),
            None
        );
        // Degenerate bounds are rejected outright.
        assert_eq!(
            scroll_container_rank(
                ScrollContainerKind::ScrollArea,
                Rect {
                    x: 0.0,
                    y: 0.0,
                    width: 0.0,
                    height: 100.0,
                },
                window,
                window,
            ),
            None
        );
        // Missing window/screen pieces are skipped, not fatal.
        assert!(scroll_container_rank(
            ScrollContainerKind::ScrollArea,
            Rect {
                x: 0.0,
                y: 0.0,
                width: 100.0,
                height: 100.0,
            },
            None,
            None,
        )
        .is_some());
    }
}
