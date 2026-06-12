//! The ONLY unsafe AX FFI surface of the perception layer.
//!
//! Everything above this file works with [`Cf`]-owned references and plain
//! Rust values. The extern block is module-local on purpose — that is the
//! repo convention (lib.rs and macos_observer.rs each carry their own) and
//! it keeps this file the complete audit surface for perception's unsafe
//! code.
//!
//! Two calls here are new to this codebase and load-bearing for the perf
//! gates: `AXUIElementCopyMultipleAttributeValues` (one IPC round trip per
//! element instead of one per attribute) and
//! `AXUIElementSetMessagingTimeout` (a hung target app costs ~250ms, not the
//! 6s system default).

use core_foundation::array::CFArray;
use core_foundation::base::TCFType;
use core_foundation::string::CFString;
use core_foundation_sys::array::{
    CFArrayGetCount, CFArrayGetTypeID, CFArrayGetValueAtIndex, CFArrayRef,
};
use core_foundation_sys::base::{Boolean, CFGetTypeID, CFRelease, CFRetain, CFTypeID, CFTypeRef};
use core_foundation_sys::dictionary::CFDictionaryRef;
use core_foundation_sys::number::{
    kCFNumberFloat64Type, CFBooleanGetTypeID, CFBooleanGetValue, CFBooleanRef, CFNumberGetTypeID,
    CFNumberGetValue, CFNumberRef,
};
use core_foundation_sys::string::{CFStringGetTypeID, CFStringRef};
use std::ffi::c_void;

pub(crate) const AX_ERROR_SUCCESS: i32 = 0;
const AX_ERROR_API_DISABLED: i32 = -25211;
const AX_ERROR_CANNOT_COMPLETE: i32 = -25204;

const K_AX_VALUE_CGPOINT_TYPE: i32 = 1;
const K_AX_VALUE_CGSIZE_TYPE: i32 = 2;
/// Entries of a `CopyMultipleAttributeValues` result that carry a per-slot
/// failure are AXValues of this type.
const K_AX_VALUE_AX_ERROR_TYPE: i32 = 5;

type AXUIElementRef = CFTypeRef;
type AXValueRef = CFTypeRef;
type Pid = i32;
type CGWindowID = u32;

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
    fn AXUIElementCopyElementAtPosition(
        application: AXUIElementRef,
        x: f32,
        y: f32,
        element: *mut AXUIElementRef,
    ) -> i32;
    fn AXUIElementCopyAttributeValue(
        element: AXUIElementRef,
        attribute: CFStringRef,
        value: *mut CFTypeRef,
    ) -> i32;
    fn AXUIElementCopyMultipleAttributeValues(
        element: AXUIElementRef,
        attributes: CFArrayRef,
        options: u32,
        values: *mut CFArrayRef,
    ) -> i32;
    fn AXUIElementCopyActionNames(element: AXUIElementRef, names: *mut CFArrayRef) -> i32;
    fn AXUIElementSetAttributeValue(
        element: AXUIElementRef,
        attribute: CFStringRef,
        value: CFTypeRef,
    ) -> i32;
    fn AXUIElementSetMessagingTimeout(element: AXUIElementRef, timeout_seconds: f32) -> i32;
    fn AXValueGetType(value: AXValueRef) -> i32;
    fn AXValueGetTypeID() -> CFTypeID;
    fn AXValueGetValue(value: AXValueRef, value_type: i32, value_ptr: *mut c_void) -> Boolean;
    /// Private but long-stable (macOSPrivateApi is already enabled app-wide):
    /// maps an AX window element to its CGWindowID so snapshots can join
    /// against the capture stack's window enumeration. Callers must tolerate
    /// failure — there is a frame/title fallback in the see pipeline.
    #[link_name = "_AXUIElementGetWindow"]
    fn AXUIElementGetWindow(element: AXUIElementRef, out: *mut CGWindowID) -> i32;
}

/// Owned CF reference: releases on drop. Perception's equivalent of the
/// observer's private `OwnedCf`, duplicated here so all of perception's
/// unsafe stays in this one file.
pub struct Cf(CFTypeRef);

impl Cf {
    /// Takes ownership of a +1 reference.
    fn owned(value: CFTypeRef) -> Option<Self> {
        if value.is_null() {
            None
        } else {
            Some(Self(value))
        }
    }

    /// Retains a borrowed reference.
    fn retained(value: CFTypeRef) -> Option<Self> {
        if value.is_null() {
            None
        } else {
            Some(Self(unsafe { CFRetain(value) }))
        }
    }

    fn type_id(&self) -> CFTypeID {
        unsafe { CFGetTypeID(self.0) }
    }
}

impl Drop for Cf {
    fn drop(&mut self) {
        unsafe { CFRelease(self.0) };
    }
}

// AXUIElementRef is documented thread-safe to *send* between threads (calls
// IPC into the target app); perception only ever uses a ref from one thread
// at a time (the walker thread that created it).
pub struct AxElementRef(Cf);

unsafe impl Send for AxElementRef {}

impl AxElementRef {
    fn from_cf(cf: Cf) -> Self {
        Self(cf)
    }

    fn raw(&self) -> AXUIElementRef {
        self.0 .0
    }
}

/// `AXIsProcessTrusted`, no prompt.
pub fn process_trusted() -> bool {
    unsafe { AXIsProcessTrusted() != 0 }
}

/// `AXIsProcessTrustedWithOptions` with the prompt option — first-run flow.
pub fn process_trusted_with_prompt() -> bool {
    use core_foundation::boolean::CFBoolean;
    use core_foundation::dictionary::CFDictionary;
    let key = CFString::new("AXTrustedCheckOptionPrompt");
    let options =
        CFDictionary::from_CFType_pairs(&[(key.as_CFType(), CFBoolean::true_value().as_CFType())]);
    unsafe { AXIsProcessTrustedWithOptions(options.as_concrete_TypeRef() as CFDictionaryRef) != 0 }
}

/// App root element for a pid, with the perception messaging timeout applied
/// so one hung app cannot stall a walk for the 6s system default.
pub fn app_element(pid: i32, messaging_timeout_secs: f32) -> Option<AxElementRef> {
    let raw = unsafe { AXUIElementCreateApplication(pid) };
    let cf = Cf::owned(raw)?;
    let element = AxElementRef::from_cf(cf);
    set_messaging_timeout(&element, messaging_timeout_secs);
    Some(element)
}

/// Re-tune the per-app AX messaging timeout. Window acquisition uses a
/// longer timeout than the walk: a just-launched app's AX server can take
/// most of a second to answer its first request, which is not the "hung
/// app" case the walk timeout guards against.
pub fn set_messaging_timeout(element: &AxElementRef, timeout_secs: f32) {
    unsafe {
        // Best-effort: a failure here only means default timeouts.
        let _ = AXUIElementSetMessagingTimeout(element.raw(), timeout_secs);
    }
}

/// One attribute slot of a batched fetch.
pub enum AttrValue {
    Missing,
    String(String),
    Bool(bool),
    Number(f64),
    Point(f64, f64),
    Size(f64, f64),
    /// Element-array attributes (AXChildren rides in the same batched round
    /// trip as the scalar attributes — the walk's biggest IPC saving).
    Elements(Vec<AxElementRef>),
    /// Present but not a type the walker consumes (kept for AXValue counts
    /// etc. where a numeric/string projection exists on other attributes).
    Other,
}

/// Batched per-element fetch: one IPC round trip for all `attributes`.
/// Returns one [`AttrValue`] per requested attribute, `Missing` for
/// unsupported or value-less slots; element-array slots are capped at
/// `element_array_cap` entries. `Err` only on transport-level failure
/// (`kAXErrorCannotComplete`, API disabled, invalid element) — callers treat
/// that as "element gone".
pub fn copy_attributes(
    element: &AxElementRef,
    attributes: &[&str],
    element_array_cap: usize,
) -> Result<Vec<AttrValue>, i32> {
    let names: Vec<CFString> = attributes.iter().map(|a| CFString::new(a)).collect();
    let array = CFArray::from_CFTypes(&names);
    let mut out: CFArrayRef = std::ptr::null();
    let err = unsafe {
        AXUIElementCopyMultipleAttributeValues(
            element.raw(),
            array.as_concrete_TypeRef(),
            0,
            &mut out,
        )
    };
    if err != AX_ERROR_SUCCESS {
        return Err(err);
    }
    let Some(values) = Cf::owned(out as CFTypeRef) else {
        return Err(AX_ERROR_CANNOT_COMPLETE);
    };
    let array_ref = values.0 as CFArrayRef;
    let count = unsafe { CFArrayGetCount(array_ref) } as usize;
    let mut result = Vec::with_capacity(attributes.len());
    for index in 0..attributes.len() {
        if index >= count {
            result.push(AttrValue::Missing);
            continue;
        }
        let raw = unsafe { CFArrayGetValueAtIndex(array_ref, index as isize) } as CFTypeRef;
        result.push(coerce_attr(raw, element_array_cap));
    }
    Ok(result)
}

/// Borrowed-ref coercion for one batched slot. Scalars copy out; element
/// arrays retain each entry (the parent array dies with this call).
fn coerce_attr(raw: CFTypeRef, element_array_cap: usize) -> AttrValue {
    if raw.is_null() {
        return AttrValue::Missing;
    }
    let type_id = unsafe { CFGetTypeID(raw) };
    unsafe {
        if type_id == AXValueGetTypeID() {
            match AXValueGetType(raw) {
                K_AX_VALUE_AX_ERROR_TYPE => AttrValue::Missing,
                K_AX_VALUE_CGPOINT_TYPE => {
                    let mut point = CGPoint::default();
                    if AXValueGetValue(
                        raw,
                        K_AX_VALUE_CGPOINT_TYPE,
                        &mut point as *mut _ as *mut c_void,
                    ) != 0
                    {
                        AttrValue::Point(point.x, point.y)
                    } else {
                        AttrValue::Missing
                    }
                }
                K_AX_VALUE_CGSIZE_TYPE => {
                    let mut size = CGSize::default();
                    if AXValueGetValue(
                        raw,
                        K_AX_VALUE_CGSIZE_TYPE,
                        &mut size as *mut _ as *mut c_void,
                    ) != 0
                    {
                        AttrValue::Size(size.width, size.height)
                    } else {
                        AttrValue::Missing
                    }
                }
                _ => AttrValue::Other,
            }
        } else if type_id == CFStringGetTypeID() {
            AttrValue::String(CFString::wrap_under_get_rule(raw as CFStringRef).to_string())
        } else if type_id == CFBooleanGetTypeID() {
            AttrValue::Bool(CFBooleanGetValue(raw as CFBooleanRef))
        } else if type_id == CFNumberGetTypeID() {
            let mut value: f64 = 0.0;
            if CFNumberGetValue(
                raw as CFNumberRef,
                kCFNumberFloat64Type,
                &mut value as *mut _ as *mut c_void,
            ) {
                AttrValue::Number(value)
            } else {
                AttrValue::Missing
            }
        } else if type_id == CFArrayGetTypeID() {
            let array_ref = raw as CFArrayRef;
            let count = (CFArrayGetCount(array_ref) as usize).min(element_array_cap);
            let mut refs = Vec::with_capacity(count);
            for index in 0..count {
                let entry = CFArrayGetValueAtIndex(array_ref, index as isize) as CFTypeRef;
                if let Some(child) = Cf::retained(entry) {
                    refs.push(AxElementRef::from_cf(child));
                }
            }
            AttrValue::Elements(refs)
        } else {
            AttrValue::Other
        }
    }
}

/// Single-attribute element fetch (used for AXFocusedWindow / child role
/// probes where batching buys nothing).
fn copy_element_attribute(element: &AxElementRef, attribute: &str) -> Option<AxElementRef> {
    let name = CFString::new(attribute);
    let mut out: CFTypeRef = std::ptr::null();
    let err = unsafe {
        AXUIElementCopyAttributeValue(element.raw(), name.as_concrete_TypeRef(), &mut out)
    };
    if err != AX_ERROR_SUCCESS {
        return None;
    }
    Cf::owned(out).map(AxElementRef::from_cf)
}

/// The app's focused window, falling back to the first of `AXWindows`.
pub fn focused_window(app: &AxElementRef) -> Option<AxElementRef> {
    if let Some(window) = copy_element_attribute(app, "AXFocusedWindow") {
        return Some(window);
    }
    windows(app).into_iter().next()
}

/// All windows of an app element.
pub fn windows(app: &AxElementRef) -> Vec<AxElementRef> {
    element_array_attribute(app, "AXWindows", usize::MAX)
}

fn element_array_attribute(
    element: &AxElementRef,
    attribute: &str,
    cap: usize,
) -> Vec<AxElementRef> {
    let name = CFString::new(attribute);
    let mut out: CFTypeRef = std::ptr::null();
    let err = unsafe {
        AXUIElementCopyAttributeValue(element.raw(), name.as_concrete_TypeRef(), &mut out)
    };
    if err != AX_ERROR_SUCCESS {
        return Vec::new();
    }
    let Some(owned) = Cf::owned(out) else {
        return Vec::new();
    };
    if owned.type_id() != unsafe { CFArrayGetTypeID() } {
        return Vec::new();
    }
    let array_ref = owned.0 as CFArrayRef;
    let count = (unsafe { CFArrayGetCount(array_ref) } as usize).min(cap);
    let mut refs = Vec::with_capacity(count);
    for index in 0..count {
        let raw = unsafe { CFArrayGetValueAtIndex(array_ref, index as isize) } as CFTypeRef;
        if let Some(child) = Cf::retained(raw) {
            refs.push(AxElementRef::from_cf(child));
        }
    }
    refs
}

/// Most action names an element keeps in the index.
const ACTION_NAMES_CAP: usize = 8;

/// `AXUIElementCopyActionNames` — the element's action vocabulary. Only
/// standard `AX…` names are kept: Finder toolbar items (and other custom
/// controls) advertise free-text custom actions ("Name:Move next\nTarget:…")
/// that would bloat the index without being invocable.
pub fn action_names(element: &AxElementRef) -> Vec<String> {
    let mut out: CFArrayRef = std::ptr::null();
    let err = unsafe { AXUIElementCopyActionNames(element.raw(), &mut out) };
    if err != AX_ERROR_SUCCESS {
        return Vec::new();
    }
    let Some(owned) = Cf::owned(out as CFTypeRef) else {
        return Vec::new();
    };
    let array_ref = owned.0 as CFArrayRef;
    let count = unsafe { CFArrayGetCount(array_ref) } as usize;
    let mut names = Vec::with_capacity(count.min(ACTION_NAMES_CAP));
    for index in 0..count {
        if names.len() >= ACTION_NAMES_CAP {
            break;
        }
        let raw = unsafe { CFArrayGetValueAtIndex(array_ref, index as isize) } as CFTypeRef;
        if raw.is_null() {
            continue;
        }
        if unsafe { CFGetTypeID(raw) } == unsafe { CFStringGetTypeID() } {
            let name = unsafe { CFString::wrap_under_get_rule(raw as CFStringRef) }.to_string();
            if name.starts_with("AX") {
                names.push(name);
            }
        }
    }
    names
}

/// Cheap single-attribute role probe used only when a node's child list
/// exceeds the cap and actionable-first ordering is needed.
pub fn role_of(element: &AxElementRef) -> Option<String> {
    let name = CFString::new("AXRole");
    let mut out: CFTypeRef = std::ptr::null();
    let err = unsafe {
        AXUIElementCopyAttributeValue(element.raw(), name.as_concrete_TypeRef(), &mut out)
    };
    if err != AX_ERROR_SUCCESS {
        return None;
    }
    let owned = Cf::owned(out)?;
    if owned.type_id() == unsafe { CFStringGetTypeID() } {
        Some(unsafe { CFString::wrap_under_get_rule(owned.0 as CFStringRef) }.to_string())
    } else {
        None
    }
}

/// Set a boolean AX attribute (the Electron `AXManualAccessibility` /
/// `AXEnhancedUserInterface` unlock). Returns whether the app accepted it.
pub fn set_bool_attribute(element: &AxElementRef, attribute: &str, value: bool) -> bool {
    use core_foundation::boolean::CFBoolean;
    let name = CFString::new(attribute);
    let flag = if value {
        CFBoolean::true_value()
    } else {
        CFBoolean::false_value()
    };
    let err = unsafe {
        AXUIElementSetAttributeValue(
            element.raw(),
            name.as_concrete_TypeRef(),
            flag.as_CFTypeRef(),
        )
    };
    if std::env::var_os("SCREENIE_PERCEPTION_DEBUG").is_some() {
        eprintln!("[screenie] perception set {attribute}={value} -> {err}");
    }
    err == AX_ERROR_SUCCESS
}

/// System-wide AX hit test: the deepest element under a global point. Used
/// by the hit-test harness to validate that an indexed frame's center still
/// lands on the element it came from.
pub fn hit_test(x: f64, y: f64) -> Option<AxElementRef> {
    let system = Cf::owned(unsafe { AXUIElementCreateSystemWide() })?;
    let mut out: AXUIElementRef = std::ptr::null();
    let err = unsafe { AXUIElementCopyElementAtPosition(system.0, x as f32, y as f32, &mut out) };
    if err != AX_ERROR_SUCCESS {
        return None;
    }
    Cf::owned(out).map(AxElementRef::from_cf)
}

/// CGWindowID of an AX window element via the private `_AXUIElementGetWindow`.
pub fn window_id(window: &AxElementRef) -> Option<u32> {
    let mut id: CGWindowID = 0;
    let err = unsafe { AXUIElementGetWindow(window.raw(), &mut id) };
    if err == AX_ERROR_SUCCESS && id != 0 {
        Some(id)
    } else {
        None
    }
}

/// Whether an AX transport error means "accessibility is off for this
/// process" as opposed to "this element/app is gone".
pub fn is_api_disabled(err: i32) -> bool {
    err == AX_ERROR_API_DISABLED
}
