use super::executor::{CalibrationProbe, ClickPoint, TargetSummary};
use super::observer::{
    ensure_accessibility_permission_with, filter_candidates, is_structural_container_role,
    ObservedCandidate, MAX_OBSERVED_ELEMENTS,
};
use super::safari_dom;
use super::types::{
    normalize_signature_name, CoordinateSpace, Element, ElementSource, FocusedApp,
    FocusedAppProvider, ObservationError, PlatformElementHandle, Rect, ScreenObserver,
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
use core_foundation_sys::number::{CFBooleanGetTypeID, CFBooleanGetValue, CFBooleanRef};
use core_foundation_sys::string::{CFStringGetTypeID, CFStringRef};
use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr};
use std::fmt;
use std::mem;
use std::ptr;
use std::rc::Rc;

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
    fn AXUIElementCopyElementAtPosition(
        application: AXUIElementRef,
        x: f32,
        y: f32,
        element: *mut AXUIElementRef,
    ) -> i32;
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

/// macOS Accessibility observer.
///
/// Bounds returned by this observer are macOS Accessibility screen points
/// with a top-left origin. For Milestone 3, those AX points and enigo's
/// CGEvent global positioning are assumed to be 1:1. Screenshot pixel
/// conversion belongs to a later milestone.
#[derive(Clone, Debug, Default)]
pub struct MacObserver {
    handles: AxHandleRegistry,
}

impl MacObserver {
    pub fn new() -> Self {
        Self::default()
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

        let mut hit = ptr::null();
        let err = unsafe {
            AXUIElementCopyElementAtPosition(system_wide, point.x as f32, point.y as f32, &mut hit)
        };
        match err {
            AX_ERROR_SUCCESS if hit.is_null() => Ok(None),
            AX_ERROR_SUCCESS => {
                let _hit_ref = OwnedCf::new(hit);
                Ok(Some(copy_target_summary(hit)?))
            }
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
}

impl ScreenObserver for MacObserver {
    fn observe(&self) -> Result<Vec<Element>, ObservationError> {
        self.observe_frontmost_app()
    }

    fn read_page_text(&self) -> Result<String, String> {
        let focused = self.focused_app().map_err(|err| err.to_string())?;
        super::page_reader::read_focused_page_text(self, &focused)
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
                    let web_area = self.current_safari_web_area().ok().flatten()?;
                    if let Ok(Some(refreshed)) =
                        safari_dom::refresh_safari_dom_element(agent_id, web_area, el.id)
                    {
                        if element_identity_matches(el, &refreshed) {
                            return Some(refreshed);
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
        let value = copy_value_string_attribute(element, "AXValue")?;
        let enabled = copy_bool_attribute(element, "AXEnabled")?.unwrap_or(true);
        let focused = copy_bool_attribute(element, "AXFocused")?.unwrap_or(false);
        // Selection only matters (and is only cheap to read) for the focused
        // element; it feeds the semantic state hash so select-all/deselect
        // counts as UI progress.
        let selected_text = if focused {
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
            return Ok(None);
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
            attribute.to_string(),
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

fn copy_element_snapshot(
    element: AXUIElementRef,
    id: u32,
    platform_handle: Option<PlatformElementHandle>,
) -> Result<Element, ObservationError> {
    let role = copy_string_attribute(element, "AXRole")?.unwrap_or_default();
    let title = copy_nonempty_string_attribute(element, "AXTitle")?;
    let description = copy_nonempty_string_attribute(element, "AXDescription")?;
    let name = title.or(description).unwrap_or_default();
    let value = copy_value_string_attribute(element, "AXValue")?;
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
}
