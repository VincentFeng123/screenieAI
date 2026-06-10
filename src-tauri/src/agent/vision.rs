use super::types::{
    Action, CoordinateSpace, Element, ElementSource, FocusedApp, FocusedAppProvider,
    MenuPressOutcome, MenuScanResult, ObservationError, Rect, ScreenObserver,
};
use ab_glyph::FontArc;
use base64::{engine::general_purpose::STANDARD, Engine as _};
use image::{DynamicImage, GrayImage, ImageFormat, Luma, Rgba, RgbaImage};
use imageproc::drawing::{draw_filled_rect_mut, draw_hollow_rect_mut, draw_text_mut};
use imageproc::edges::canny;
use imageproc::rect::Rect as ImageProcRect;
use serde::Serialize;
use std::cell::RefCell;
use std::collections::{BTreeMap, VecDeque};
use std::io::Cursor;
use std::rc::Rc;

const DEFAULT_VISION_FALLBACK_MIN_ELEMENTS: u32 = 3;
const DEFAULT_VISION_FALLBACK_MIN_WINDOW_AREA_POINTS: f64 = 120_000.0;
const MAX_VISION_CANDIDATES: usize = 80;
const MARK_FONT: &[u8] = include_bytes!("../../assets/fonts/NotoSans-Bold.ttf");
const FNV_1A_64_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_1A_64_PRIME: u64 = 0x100000001b3;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct VisionFallbackOptions {
    pub min_elements: u32,
    pub min_window_area_points: f64,
    pub coordinate_fallback: bool,
}

impl Default for VisionFallbackOptions {
    fn default() -> Self {
        Self {
            min_elements: DEFAULT_VISION_FALLBACK_MIN_ELEMENTS,
            min_window_area_points: DEFAULT_VISION_FALLBACK_MIN_WINDOW_AREA_POINTS,
            coordinate_fallback: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
#[derive(Default)]
pub enum ObservationSource {
    #[default]
    Ax,
    VisionMarks,
    VisionCoordinate,
    VisionGrounding,
}


#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CaptureSize {
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ObservationMetadata {
    pub source: ObservationSource,
    pub candidate_count: Option<u32>,
    pub trigger_reason: Option<String>,
    pub capture_size: Option<CaptureSize>,
    pub detector_kind: Option<String>,
    #[serde(skip_serializing)]
    pub visual_state_hash: Option<String>,
}

impl Default for ObservationMetadata {
    fn default() -> Self {
        Self {
            source: ObservationSource::Ax,
            candidate_count: None,
            trigger_reason: None,
            capture_size: None,
            detector_kind: None,
            visual_state_hash: None,
        }
    }
}

pub trait ObservationMetadataProvider {
    fn observation_metadata(&self) -> ObservationMetadata;

    fn synthetic_elements_for_action(&self, _action: &Action) -> Vec<Element> {
        Vec::new()
    }

    fn vision_fallback_context(&self) -> Option<VisionFallbackContext> {
        None
    }

    fn observe_for_visual_replan(
        &self,
        _ax: &[Element],
        _trigger_reason: &str,
    ) -> Result<Option<Vec<Element>>, ObservationError> {
        Ok(None)
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct VisionFallbackState {
    inner: Rc<RefCell<VisionFallbackStateInner>>,
}

impl VisionFallbackState {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn clear(&self) {
        *self.inner.borrow_mut() = VisionFallbackStateInner::default();
    }

    pub(crate) fn context(&self) -> Option<VisionFallbackContext> {
        self.inner.borrow().context.clone()
    }

    pub(crate) fn metadata(&self) -> ObservationMetadata {
        self.inner.borrow().metadata.clone()
    }

    pub(crate) fn set_context(
        &self,
        context: VisionFallbackContext,
        metadata: ObservationMetadata,
    ) {
        let mut inner = self.inner.borrow_mut();
        inner.context = Some(context);
        inner.metadata = metadata;
        inner.pending_synthetic.clear();
    }

    pub(crate) fn set_pending_synthetic(&self, elements: Vec<Element>) {
        self.inner.borrow_mut().pending_synthetic = elements;
    }

    pub(crate) fn synthetic_for_action(&self, action: &Action) -> Vec<Element> {
        let ids = action.target_ids();
        if ids.is_empty() {
            return Vec::new();
        }
        self.inner
            .borrow()
            .pending_synthetic
            .iter()
            .filter(|element| ids.contains(&element.id))
            .cloned()
            .collect()
    }
}

#[derive(Clone, Debug, Default)]
struct VisionFallbackStateInner {
    context: Option<VisionFallbackContext>,
    metadata: ObservationMetadata,
    pending_synthetic: Vec<Element>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct VisionFallbackContext {
    pub mode: VisionFallbackMode,
    pub image_png_b64: String,
    pub capture_width: u32,
    pub capture_height: u32,
    pub window_origin_x: f64,
    pub window_origin_y: f64,
    pub scale_factor: f64,
    pub candidate_count: usize,
    pub trigger_reason: String,
    pub detector_kind: String,
    pub element_pixel_bounds: BTreeMap<u32, Rect>,
}

impl VisionFallbackContext {
    pub(crate) fn coordinate_space(&self) -> CoordinateSpace {
        CoordinateSpace::WindowPixels {
            origin_x: self.window_origin_x,
            origin_y: self.window_origin_y,
            scale_factor: self.scale_factor,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VisionFallbackMode {
    Marks,
    Grounding,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FocusedWindowInfo {
    pub origin_x: f64,
    pub origin_y: f64,
    pub width_pixels: u32,
    pub height_pixels: u32,
    pub scale_factor: f64,
}

impl FocusedWindowInfo {
    fn point_area(&self) -> f64 {
        let scale = sane_scale(self.scale_factor);
        (self.width_pixels as f64 / scale) * (self.height_pixels as f64 / scale)
    }

    fn with_capture_size(&self, width: u32, height: u32) -> Self {
        Self {
            width_pixels: width,
            height_pixels: height,
            ..self.clone()
        }
    }
}

pub(crate) trait VisionWindowCapturer {
    type Window;

    fn focused_window(
        &self,
        focused_pid: Option<i32>,
    ) -> Result<Option<Self::Window>, ObservationError>;

    fn window_info(&self, window: &Self::Window) -> Result<FocusedWindowInfo, ObservationError>;

    fn capture_window(&self, window: &Self::Window) -> Result<RgbaImage, ObservationError>;

    fn capture_observation_surface(
        &self,
        focused_pid: Option<i32>,
    ) -> Result<Option<(RgbaImage, FocusedWindowInfo)>, ObservationError> {
        let Some(window) = self.focused_window(focused_pid)? else {
            return Ok(None);
        };
        let window_info = self.window_info(&window)?;
        let capture = self.capture_window(&window)?;
        let info = window_info.with_capture_size(capture.width(), capture.height());
        Ok(Some((capture, info)))
    }
}

pub(crate) trait VisionCandidateDetector {
    fn kind(&self) -> &'static str;

    fn detect(&self, image: &RgbaImage) -> Vec<Rect>;
}

#[derive(Clone, Debug)]
pub(crate) struct VisionFallbackObserver<O, C, D> {
    base: O,
    capturer: C,
    detector: D,
    state: VisionFallbackState,
    options: VisionFallbackOptions,
}

impl<O, C, D> VisionFallbackObserver<O, C, D> {
    pub(crate) fn new(
        base: O,
        capturer: C,
        detector: D,
        state: VisionFallbackState,
        options: VisionFallbackOptions,
    ) -> Self {
        Self {
            base,
            capturer,
            detector,
            state,
            options,
        }
    }
}

#[cfg(target_os = "macos")]
impl
    VisionFallbackObserver<super::macos_observer::MacObserver, XcapWindowCapturer, CoarseCvDetector>
{
    pub(crate) fn macos(
        base: super::macos_observer::MacObserver,
        state: VisionFallbackState,
        options: VisionFallbackOptions,
    ) -> Self {
        Self::new(base, XcapWindowCapturer, CoarseCvDetector, state, options)
    }
}

impl<O, C, D> ScreenObserver for VisionFallbackObserver<O, C, D>
where
    O: ScreenObserver + FocusedAppProvider,
    C: VisionWindowCapturer,
    D: VisionCandidateDetector,
{
    fn read_page_text(&self) -> Result<String, String> {
        self.base.read_page_text()
    }

    // Change notifications come from the platform AX layer regardless of
    // which fallback produced the current observation.
    fn change_signal(&self) -> Option<Box<dyn super::types::UiChangeSignal>> {
        self.base.change_signal()
    }

    fn observe(&self) -> Result<Vec<Element>, ObservationError> {
        let ax = match self.base.observe() {
            Ok(ax) => ax,
            Err(err) => return self.observe_ax_error_fallback(err),
        };
        if ax
            .iter()
            .any(|element| element.source == ElementSource::Web)
        {
            self.state.clear();
            return Ok(ax);
        }
        let ax_count = ax.len() as u32;
        if ax_count >= self.options.min_elements {
            self.state.clear();
            return Ok(ax);
        }

        let trigger_reason = if ax.is_empty() {
            "ax-empty".to_string()
        } else if ax_count < self.options.min_elements {
            format!("ax-weak:{ax_count}<{}", self.options.min_elements)
        } else {
            format!("ax-visual-augment:{ax_count}")
        };
        let focused = self.base.focused_app().ok();
        let focused_pid = focused.as_ref().and_then(|app| app.pid);
        let window = match self.capturer.focused_window(focused_pid) {
            Ok(Some(window)) => window,
            Ok(None) => {
                self.state.clear();
                return Ok(ax);
            }
            Err(err) if ax.is_empty() => return Err(err),
            Err(_) => {
                self.state.clear();
                return Ok(ax);
            }
        };
        let window_info = match self.capturer.window_info(&window) {
            Ok(info) => info,
            Err(err) if ax.is_empty() => return Err(err),
            Err(_) => {
                self.state.clear();
                return Ok(ax);
            }
        };

        if !ax.is_empty() && window_info.point_area() < self.options.min_window_area_points {
            self.state.clear();
            return Ok(ax);
        }

        let capture = match self.capturer.capture_window(&window) {
            Ok(image) => image,
            Err(err) if ax.is_empty() => return Err(err),
            Err(_) => {
                self.state.clear();
                return Ok(ax);
            }
        };
        let info = window_info.with_capture_size(capture.width(), capture.height());

        self.observe_grounding_fallback(ax, capture, info, trigger_reason)
    }

    fn refresh_element(&self, el: &Element) -> Option<Element> {
        match el.source {
            ElementSource::Ax | ElementSource::Web => self.base.refresh_element(el),
            ElementSource::VisionDetected => self.refresh_detected_element(el),
            ElementSource::VisionCoordinate => self.refresh_coordinate_element(el),
        }
    }

    // Semantic actions only make sense on real AX elements; vision-derived
    // synthetic elements have no platform handle, so they fall back cleanly.
    fn perform_press(&self, el: &Element) -> Result<bool, String> {
        if el.source == ElementSource::Ax {
            self.base.perform_press(el)
        } else {
            Ok(false)
        }
    }

    fn set_value(&self, el: &Element, text: &str) -> Result<bool, String> {
        if el.source == ElementSource::Ax {
            self.base.set_value(el, text)
        } else {
            Ok(false)
        }
    }

    fn press_menu_path(&self, path: &[String]) -> Result<MenuPressOutcome, String> {
        self.base.press_menu_path(path)
    }

    fn search_menu_tree(&self, query: &str, max_results: usize) -> Result<MenuScanResult, String> {
        self.base.search_menu_tree(query, max_results)
    }
}

impl<O, C, D> FocusedAppProvider for VisionFallbackObserver<O, C, D>
where
    O: FocusedAppProvider,
{
    fn focused_app(&self) -> Result<FocusedApp, ObservationError> {
        self.base.focused_app()
    }
}

impl<O, C, D> ObservationMetadataProvider for VisionFallbackObserver<O, C, D>
where
    O: FocusedAppProvider,
    C: VisionWindowCapturer,
    D: VisionCandidateDetector,
{
    fn observation_metadata(&self) -> ObservationMetadata {
        self.state.metadata()
    }

    fn synthetic_elements_for_action(&self, action: &Action) -> Vec<Element> {
        self.state.synthetic_for_action(action)
    }

    fn vision_fallback_context(&self) -> Option<VisionFallbackContext> {
        self.state.context()
    }

    fn observe_for_visual_replan(
        &self,
        ax: &[Element],
        trigger_reason: &str,
    ) -> Result<Option<Vec<Element>>, ObservationError> {
        self.observe_marked_visual_context(ax.to_vec(), trigger_reason.to_string())
    }
}

impl<O, C, D> VisionFallbackObserver<O, C, D>
where
    O: FocusedAppProvider,
    C: VisionWindowCapturer,
    D: VisionCandidateDetector,
{
    fn observe_grounding_fallback(
        &self,
        ax: Vec<Element>,
        capture: RgbaImage,
        info: FocusedWindowInfo,
        trigger_reason: String,
    ) -> Result<Vec<Element>, ObservationError> {
        // An all-black capture mid-run almost always means a broken Screen
        // Recording grant (e.g. the binary moved); tag it so step reports and
        // logs point at the real cause instead of a generic vision failure.
        let trigger_reason = if crate::capture::rgba_is_blank(&capture) {
            eprintln!(
                "[screenie] agent vision capture is blank; Screen Recording permission is likely broken for this binary (trigger was: {trigger_reason})"
            );
            "capture-blank-check-screen-recording".to_string()
        } else {
            trigger_reason
        };
        let image_png_b64 = png_b64(&capture)?;
        let visual_state_hash = image_state_hash(&capture, &info);
        let metadata = grounding_observation_metadata(
            &trigger_reason,
            &info,
            "local-http-grounder",
            visual_state_hash,
        );
        self.state.set_context(
            VisionFallbackContext {
                mode: VisionFallbackMode::Grounding,
                image_png_b64,
                capture_width: info.width_pixels,
                capture_height: info.height_pixels,
                window_origin_x: info.origin_x,
                window_origin_y: info.origin_y,
                scale_factor: sane_scale(info.scale_factor),
                candidate_count: 0,
                trigger_reason,
                detector_kind: "local-http-grounder".into(),
                element_pixel_bounds: BTreeMap::new(),
            },
            metadata,
        );
        Ok(ax)
    }

    fn observe_ax_error_fallback(
        &self,
        err: ObservationError,
    ) -> Result<Vec<Element>, ObservationError> {
        let trigger_reason = format!("ax-error:{}", compact_trigger_reason(&err.to_string()));
        match self.capture_focused_window() {
            Ok(Some((capture, info))) => {
                self.observe_grounding_fallback(Vec::new(), capture, info, trigger_reason)
            }
            Ok(None) => {
                self.state.clear();
                Err(err)
            }
            Err(fallback_err) => {
                self.state.clear();
                Err(ObservationError::AxReadFailed(format!(
                    "{err}; visual fallback failed: {fallback_err}"
                )))
            }
        }
    }

    fn observe_partial_ax_fallback(
        &self,
        ax: Vec<Element>,
        capture: RgbaImage,
        info: FocusedWindowInfo,
        trigger_reason: String,
    ) -> Result<Vec<Element>, ObservationError> {
        let detector_kind = self.detector.kind().to_string();
        let boxes = self.detector.detect(&capture);
        let (elements, mark_bounds, dropped_offscreen) =
            augment_ax_with_visual_candidates(ax, &boxes, &info);
        if dropped_offscreen > 0 {
            eprintln!(
                "[screenie] agent marks dropped_offscreen={} visible={}",
                dropped_offscreen,
                mark_bounds.len()
            );
        }

        if mark_bounds.is_empty() {
            self.state.clear();
            return Ok(elements);
        }

        let image_png_b64 = render_marked_png_b64(&capture, &mark_bounds)?;
        let metadata = observation_metadata(
            ObservationSource::VisionMarks,
            mark_bounds.len(),
            &trigger_reason,
            &info,
            &format!("ax+{detector_kind}"),
        );
        self.state.set_context(
            VisionFallbackContext {
                mode: VisionFallbackMode::Marks,
                image_png_b64,
                capture_width: info.width_pixels,
                capture_height: info.height_pixels,
                window_origin_x: info.origin_x,
                window_origin_y: info.origin_y,
                scale_factor: sane_scale(info.scale_factor),
                candidate_count: mark_bounds.len(),
                trigger_reason,
                detector_kind: format!("ax+{detector_kind}"),
                element_pixel_bounds: mark_bounds,
            },
            metadata,
        );
        Ok(elements)
    }

    fn observe_marked_visual_context(
        &self,
        ax: Vec<Element>,
        trigger_reason: String,
    ) -> Result<Option<Vec<Element>>, ObservationError> {
        let Some((capture, info)) = self.capture_focused_window()? else {
            self.state.clear();
            return Ok(None);
        };
        if ax.is_empty() {
            self.observe_empty_ax_fallback(capture, info, trigger_reason)
                .map(Some)
        } else {
            self.observe_partial_ax_fallback(ax, capture, info, trigger_reason)
                .map(Some)
        }
    }

    fn observe_empty_ax_fallback(
        &self,
        capture: RgbaImage,
        info: FocusedWindowInfo,
        trigger_reason: String,
    ) -> Result<Vec<Element>, ObservationError> {
        let detector_kind = self.detector.kind().to_string();
        let boxes = self.detector.detect(&capture);
        if !boxes.is_empty() {
            let elements = detected_boxes_to_elements(&boxes, &info);
            let mark_bounds = elements
                .iter()
                .map(|element| (element.id, element.bounds))
                .collect::<BTreeMap<_, _>>();
            let image_png_b64 = render_marked_png_b64(&capture, &mark_bounds)?;
            let metadata = observation_metadata(
                ObservationSource::VisionMarks,
                elements.len(),
                &trigger_reason,
                &info,
                &detector_kind,
            );
            self.state.set_context(
                VisionFallbackContext {
                    mode: VisionFallbackMode::Marks,
                    image_png_b64,
                    capture_width: info.width_pixels,
                    capture_height: info.height_pixels,
                    window_origin_x: info.origin_x,
                    window_origin_y: info.origin_y,
                    scale_factor: sane_scale(info.scale_factor),
                    candidate_count: elements.len(),
                    trigger_reason,
                    detector_kind,
                    element_pixel_bounds: mark_bounds,
                },
                metadata,
            );
            return Ok(elements);
        }

        if !self.options.coordinate_fallback {
            self.state.clear();
            return Ok(Vec::new());
        }

        let image_png_b64 = png_b64(&capture)?;
        let metadata = observation_metadata(
            ObservationSource::VisionGrounding,
            0,
            &trigger_reason,
            &info,
            &detector_kind,
        );
        self.state.set_context(
            VisionFallbackContext {
                mode: VisionFallbackMode::Grounding,
                image_png_b64,
                capture_width: info.width_pixels,
                capture_height: info.height_pixels,
                window_origin_x: info.origin_x,
                window_origin_y: info.origin_y,
                scale_factor: sane_scale(info.scale_factor),
                candidate_count: 0,
                trigger_reason,
                detector_kind,
                element_pixel_bounds: BTreeMap::new(),
            },
            metadata,
        );
        Ok(Vec::new())
    }

    fn refresh_detected_element(&self, el: &Element) -> Option<Element> {
        let (capture, info) = self.capture_focused_window().ok()??;
        let boxes = self.detector.detect(&capture);
        let matches = detected_boxes_to_elements(&boxes, &info)
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

    fn refresh_coordinate_element(&self, el: &Element) -> Option<Element> {
        let (_capture, info) = self.capture_focused_window().ok()??;
        let context = self.state.context()?;
        if !window_capture_compatible(&context, &info) || !rect_in_capture(el.bounds, &info) {
            return None;
        }

        Some(Element::new(
            el.id,
            el.role.clone(),
            el.name.clone(),
            el.value.clone(),
            el.bounds,
            el.enabled,
            el.focused,
            info.coordinate_space(),
            ElementSource::VisionCoordinate,
        ))
    }

    fn capture_focused_window(
        &self,
    ) -> Result<Option<(RgbaImage, FocusedWindowInfo)>, ObservationError> {
        let focused = self.base.focused_app().ok();
        let focused_pid = focused.as_ref().and_then(|app| app.pid);
        self.capturer.capture_observation_surface(focused_pid)
    }
}

impl FocusedWindowInfo {
    fn coordinate_space(&self) -> CoordinateSpace {
        CoordinateSpace::WindowPixels {
            origin_x: self.origin_x,
            origin_y: self.origin_y,
            scale_factor: sane_scale(self.scale_factor),
        }
    }
}

fn window_capture_compatible(context: &VisionFallbackContext, info: &FocusedWindowInfo) -> bool {
    approx_equal(context.window_origin_x, info.origin_x)
        && approx_equal(context.window_origin_y, info.origin_y)
        && approx_equal(
            sane_scale(context.scale_factor),
            sane_scale(info.scale_factor),
        )
        && context.capture_width == info.width_pixels
        && context.capture_height == info.height_pixels
}

fn compact_trigger_reason(value: &str) -> String {
    const MAX_CHARS: usize = 96;
    let normalized = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.chars().count() <= MAX_CHARS {
        return normalized;
    }
    let mut truncated = normalized
        .chars()
        .take(MAX_CHARS.saturating_sub(3))
        .collect::<String>();
    truncated.push_str("...");
    truncated
}

fn rect_in_capture(rect: Rect, info: &FocusedWindowInfo) -> bool {
    rect_is_finite(rect)
        && rect.x >= 0.0
        && rect.y >= 0.0
        && rect.x + rect.width <= info.width_pixels as f64
        && rect.y + rect.height <= info.height_pixels as f64
}

fn approx_equal(a: f64, b: f64) -> bool {
    const EPSILON: f64 = 0.5;
    (a - b).abs() <= EPSILON
}

fn observation_metadata(
    source: ObservationSource,
    candidate_count: usize,
    trigger_reason: &str,
    info: &FocusedWindowInfo,
    detector_kind: &str,
) -> ObservationMetadata {
    ObservationMetadata {
        source,
        candidate_count: Some(candidate_count as u32),
        trigger_reason: Some(trigger_reason.to_string()),
        capture_size: Some(CaptureSize {
            width: info.width_pixels,
            height: info.height_pixels,
        }),
        detector_kind: Some(detector_kind.to_string()),
        visual_state_hash: None,
    }
}

fn grounding_observation_metadata(
    trigger_reason: &str,
    info: &FocusedWindowInfo,
    detector_kind: &str,
    visual_state_hash: String,
) -> ObservationMetadata {
    ObservationMetadata {
        source: ObservationSource::VisionGrounding,
        candidate_count: Some(0),
        trigger_reason: Some(trigger_reason.to_string()),
        capture_size: Some(CaptureSize {
            width: info.width_pixels,
            height: info.height_pixels,
        }),
        detector_kind: Some(detector_kind.to_string()),
        visual_state_hash: Some(visual_state_hash),
    }
}

fn image_state_hash(image: &RgbaImage, info: &FocusedWindowInfo) -> String {
    let mut hash = FNV_1A_64_OFFSET;
    fn hash_bytes(hash: &mut u64, bytes: &[u8]) {
        for byte in bytes {
            *hash ^= *byte as u64;
            *hash = hash.wrapping_mul(FNV_1A_64_PRIME);
        }
    }

    hash_bytes(&mut hash, &info.width_pixels.to_le_bytes());
    hash_bytes(&mut hash, &info.height_pixels.to_le_bytes());
    hash_bytes(&mut hash, &info.origin_x.to_le_bytes());
    hash_bytes(&mut hash, &info.origin_y.to_le_bytes());
    hash_bytes(&mut hash, &sane_scale(info.scale_factor).to_le_bytes());
    hash_bytes(&mut hash, image.as_raw());
    format!("{hash:016x}")
}

pub(crate) fn click_point_from_rect(bounds: Rect, coordinate_space: CoordinateSpace) -> (i32, i32) {
    let (x, y) = bounds.center();
    match coordinate_space {
        CoordinateSpace::AxPoints => (x.round() as i32, y.round() as i32),
        CoordinateSpace::WindowPixels {
            origin_x,
            origin_y,
            scale_factor,
        } => {
            let scale = sane_scale(scale_factor);
            (
                (origin_x + (x / scale)).round() as i32,
                (origin_y + (y / scale)).round() as i32,
            )
        }
    }
}

pub(crate) fn coordinate_element(
    id: u32,
    x: u32,
    y: u32,
    context: &VisionFallbackContext,
) -> Element {
    let half = 2.0;
    Element::new(
        id,
        "VisionCoordinate".into(),
        format!("Vision coordinate ({x}, {y})"),
        None,
        Rect {
            x: (x as f64 - half).max(0.0),
            y: (y as f64 - half).max(0.0),
            width: half * 2.0,
            height: half * 2.0,
        },
        true,
        false,
        context.coordinate_space(),
        ElementSource::VisionCoordinate,
    )
}

pub(crate) fn detected_boxes_to_elements(boxes: &[Rect], info: &FocusedWindowInfo) -> Vec<Element> {
    detected_boxes_to_elements_with_start_id(boxes, info, 1)
}

fn detected_boxes_to_elements_with_start_id(
    boxes: &[Rect],
    info: &FocusedWindowInfo,
    start_id: u32,
) -> Vec<Element> {
    let coordinate_space = CoordinateSpace::WindowPixels {
        origin_x: info.origin_x,
        origin_y: info.origin_y,
        scale_factor: sane_scale(info.scale_factor),
    };
    boxes
        .iter()
        .take(MAX_VISION_CANDIDATES)
        .enumerate()
        .map(|(index, bounds)| {
            let id = start_id.saturating_add(index as u32);
            Element::new(
                id,
                "VisionCandidate".into(),
                format!("Visual candidate {id}"),
                None,
                *bounds,
                true,
                false,
                coordinate_space,
                ElementSource::VisionDetected,
            )
        })
        .collect()
}

fn augment_ax_with_visual_candidates(
    ax: Vec<Element>,
    visual_boxes: &[Rect],
    info: &FocusedWindowInfo,
) -> (Vec<Element>, BTreeMap<u32, Rect>, usize) {
    let mut mark_bounds = ax
        .iter()
        .filter_map(|element| {
            ax_points_to_window_pixels(element.bounds, info)
                .and_then(|bounds| clip_rect(bounds, info.width_pixels, info.height_pixels))
                .map(|bounds| (element.id, bounds))
        })
        .collect::<BTreeMap<_, _>>();

    let ax_pixel_bounds = mark_bounds.values().copied().collect::<Vec<_>>();
    let filtered_visual_boxes = visual_boxes
        .iter()
        .copied()
        .filter(|visual| {
            !ax_pixel_bounds.iter().any(|ax_bounds| {
                rects_overlap(*visual, *ax_bounds) && overlap_ratio(*visual, *ax_bounds) >= 0.55
            })
        })
        .collect::<Vec<_>>();

    let max_id = ax.iter().map(|element| element.id).max().unwrap_or(0);
    let visual_elements =
        detected_boxes_to_elements_with_start_id(&filtered_visual_boxes, info, max_id + 1);

    for element in &visual_elements {
        mark_bounds.insert(element.id, element.bounds);
    }

    // In marks mode the planner grounds ids by reading the numbered chips
    // off the screenshot; an element outside the capture has no chip, so
    // offering its id invites the wrong-id clicks of the failing run. Drop
    // them for this observation — they come back with the next AX snapshot.
    let mut elements = Vec::with_capacity(ax.len() + visual_elements.len());
    let mut dropped_offscreen = 0_usize;
    for element in ax {
        if mark_bounds.contains_key(&element.id) {
            elements.push(element);
        } else {
            dropped_offscreen += 1;
        }
    }
    elements.extend(visual_elements);
    (elements, mark_bounds, dropped_offscreen)
}

fn ax_points_to_window_pixels(bounds: Rect, info: &FocusedWindowInfo) -> Option<Rect> {
    let scale = sane_scale(info.scale_factor);
    let rect = Rect {
        x: (bounds.x - info.origin_x) * scale,
        y: (bounds.y - info.origin_y) * scale,
        width: bounds.width * scale,
        height: bounds.height * scale,
    };
    rect_is_finite(rect).then_some(rect)
}

fn clip_rect(bounds: Rect, image_width: u32, image_height: u32) -> Option<Rect> {
    if !rect_is_finite(bounds) {
        return None;
    }
    let min_x = bounds.x.max(0.0).min(image_width as f64);
    let min_y = bounds.y.max(0.0).min(image_height as f64);
    let max_x = (bounds.x + bounds.width).max(0.0).min(image_width as f64);
    let max_y = (bounds.y + bounds.height).max(0.0).min(image_height as f64);
    let clipped = Rect {
        x: min_x,
        y: min_y,
        width: max_x - min_x,
        height: max_y - min_y,
    };
    (!clipped.is_empty()).then_some(clipped)
}

fn rect_is_finite(rect: Rect) -> bool {
    rect.x.is_finite()
        && rect.y.is_finite()
        && rect.width.is_finite()
        && rect.height.is_finite()
        && rect.width > 0.0
        && rect.height > 0.0
}

fn render_marked_png_b64(
    image: &RgbaImage,
    marks: &BTreeMap<u32, Rect>,
) -> Result<String, ObservationError> {
    let mut out = image.clone();
    let font = FontArc::try_from_slice(MARK_FONT).ok();
    for (id, bounds) in marks {
        draw_mark(&mut out, *id, *bounds, font.as_ref());
    }
    png_b64(&out)
}

fn draw_mark(image: &mut RgbaImage, id: u32, bounds: Rect, font: Option<&FontArc>) {
    let x = bounds.x.round().max(0.0) as i32;
    let y = bounds.y.round().max(0.0) as i32;
    let w = bounds.width.round().max(1.0) as u32;
    let h = bounds.height.round().max(1.0) as u32;
    let rect = ImageProcRect::at(x, y).of_size(w, h);
    let stroke = Rgba([255, 64, 64, 255]);
    for inset in 0..2 {
        if w > inset * 2 && h > inset * 2 {
            draw_hollow_rect_mut(
                image,
                ImageProcRect::at(x + inset as i32, y + inset as i32)
                    .of_size(w - inset * 2, h - inset * 2),
                stroke,
            );
        }
    }

    let label = id.to_string();
    let chip_w = (label.chars().count() as u32 * 10 + 12).max(22);
    let chip_h = 20;
    // The chip sits INSIDE the box's top-left corner: a chip floating above
    // the box reads as the previous element's label on dense pages (and
    // edge-clamping used to pile chips on top of each other).
    let chip_x = x.max(0);
    let chip_y = y.max(0);
    draw_filled_rect_mut(
        image,
        ImageProcRect::at(chip_x, chip_y).of_size(chip_w, chip_h),
        Rgba([255, 64, 64, 235]),
    );
    if let Some(font) = font {
        draw_text_mut(
            image,
            Rgba([255, 255, 255, 255]),
            chip_x + 5,
            chip_y + 1,
            16.0,
            font,
            &label,
        );
    }
    draw_hollow_rect_mut(image, rect, stroke);
}

fn png_b64(image: &RgbaImage) -> Result<String, ObservationError> {
    let mut bytes = Vec::new();
    DynamicImage::ImageRgba8(image.clone())
        .write_to(&mut Cursor::new(&mut bytes), ImageFormat::Png)
        .map_err(|err| {
            ObservationError::AxReadFailed(format!("vision PNG encode failed: {err}"))
        })?;
    Ok(STANDARD.encode(bytes))
}

fn sane_scale(scale_factor: f64) -> f64 {
    if scale_factor.is_finite() && scale_factor > 0.0 {
        scale_factor
    } else {
        1.0
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CoarseCvDetector;

impl VisionCandidateDetector for CoarseCvDetector {
    fn kind(&self) -> &'static str {
        "coarseCv"
    }

    fn detect(&self, image: &RgbaImage) -> Vec<Rect> {
        detect_coarse_boxes(image)
    }
}

fn detect_coarse_boxes(image: &RgbaImage) -> Vec<Rect> {
    if image.width() < 8 || image.height() < 8 {
        return Vec::new();
    }

    let gray = rgba_to_luma(image);
    let edges = canny(&gray, 30.0, 90.0);
    let mut boxes = connected_edge_boxes(&edges);
    let contrast_boxes = contrasting_region_boxes(image);
    boxes.extend(contrast_boxes.iter().copied());
    boxes.extend(group_inline_components(&contrast_boxes));
    normalize_detected_boxes(boxes, image.width(), image.height())
}

fn rgba_to_luma(image: &RgbaImage) -> GrayImage {
    let mut gray = GrayImage::new(image.width(), image.height());
    for (x, y, pixel) in image.enumerate_pixels() {
        let [r, g, b, _] = pixel.0;
        let value = (0.299 * r as f32 + 0.587 * g as f32 + 0.114 * b as f32).round() as u8;
        gray.put_pixel(x, y, Luma([value]));
    }
    gray
}

fn connected_edge_boxes(edges: &GrayImage) -> Vec<Rect> {
    let width = edges.width() as usize;
    let height = edges.height() as usize;
    let mut visited = vec![false; width.saturating_mul(height)];
    let mut boxes = Vec::new();
    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if visited[idx] || edges.get_pixel(x as u32, y as u32)[0] == 0 {
                continue;
            }
            visited[idx] = true;
            let mut queue = VecDeque::from([(x, y)]);
            let mut min_x = x;
            let mut min_y = y;
            let mut max_x = x;
            let mut max_y = y;
            let mut count = 0usize;
            while let Some((cx, cy)) = queue.pop_front() {
                count += 1;
                min_x = min_x.min(cx);
                min_y = min_y.min(cy);
                max_x = max_x.max(cx);
                max_y = max_y.max(cy);
                for ny in cy.saturating_sub(1)..=(cy + 1).min(height - 1) {
                    for nx in cx.saturating_sub(1)..=(cx + 1).min(width - 1) {
                        let nidx = ny * width + nx;
                        if visited[nidx] || edges.get_pixel(nx as u32, ny as u32)[0] == 0 {
                            continue;
                        }
                        visited[nidx] = true;
                        queue.push_back((nx, ny));
                    }
                }
            }
            if count >= 8 {
                boxes.push(Rect {
                    x: min_x as f64,
                    y: min_y as f64,
                    width: (max_x - min_x + 1) as f64,
                    height: (max_y - min_y + 1) as f64,
                });
            }
        }
    }
    boxes
}

fn contrasting_region_boxes(image: &RgbaImage) -> Vec<Rect> {
    let width = image.width() as usize;
    let height = image.height() as usize;
    let background = estimate_background_rgb(image);
    let mut visited = vec![false; width.saturating_mul(height)];
    let mut boxes = Vec::new();

    for y in 0..height {
        for x in 0..width {
            let idx = y * width + x;
            if visited[idx]
                || !pixel_contrasts_with_background(image, x as u32, y as u32, background)
            {
                continue;
            }
            visited[idx] = true;
            let mut queue = VecDeque::from([(x, y)]);
            let mut min_x = x;
            let mut min_y = y;
            let mut max_x = x;
            let mut max_y = y;
            let mut count = 0usize;
            while let Some((cx, cy)) = queue.pop_front() {
                count += 1;
                min_x = min_x.min(cx);
                min_y = min_y.min(cy);
                max_x = max_x.max(cx);
                max_y = max_y.max(cy);
                for ny in cy.saturating_sub(1)..=(cy + 1).min(height - 1) {
                    for nx in cx.saturating_sub(1)..=(cx + 1).min(width - 1) {
                        let nidx = ny * width + nx;
                        if visited[nidx]
                            || !pixel_contrasts_with_background(
                                image, nx as u32, ny as u32, background,
                            )
                        {
                            continue;
                        }
                        visited[nidx] = true;
                        queue.push_back((nx, ny));
                    }
                }
            }

            if count >= 24 {
                boxes.push(Rect {
                    x: min_x as f64,
                    y: min_y as f64,
                    width: (max_x - min_x + 1) as f64,
                    height: (max_y - min_y + 1) as f64,
                });
            }
        }
    }

    boxes
}

fn estimate_background_rgb(image: &RgbaImage) -> [u8; 3] {
    let samples = [
        (0, 0),
        (image.width().saturating_sub(1), 0),
        (0, image.height().saturating_sub(1)),
        (
            image.width().saturating_sub(1),
            image.height().saturating_sub(1),
        ),
        (image.width() / 2, 0),
        (image.width() / 2, image.height().saturating_sub(1)),
        (0, image.height() / 2),
        (image.width().saturating_sub(1), image.height() / 2),
    ];
    let mut r = 0u32;
    let mut g = 0u32;
    let mut b = 0u32;
    for &(x, y) in &samples {
        let [pr, pg, pb, _] = image.get_pixel(x, y).0;
        r += pr as u32;
        g += pg as u32;
        b += pb as u32;
    }
    [
        (r / samples.len() as u32) as u8,
        (g / samples.len() as u32) as u8,
        (b / samples.len() as u32) as u8,
    ]
}

fn pixel_contrasts_with_background(image: &RgbaImage, x: u32, y: u32, background: [u8; 3]) -> bool {
    let [r, g, b, a] = image.get_pixel(x, y).0;
    if a < 32 {
        return false;
    }
    let dr = r as i32 - background[0] as i32;
    let dg = g as i32 - background[1] as i32;
    let db = b as i32 - background[2] as i32;
    (dr * dr + dg * dg + db * db) >= 32_i32.pow(2)
}

fn group_inline_components(components: &[Rect]) -> Vec<Rect> {
    let mut groups = Vec::new();
    for (index, first) in components.iter().enumerate() {
        let mut group = *first;
        let mut count = 1usize;
        for second in components.iter().skip(index + 1) {
            if count >= 8 {
                break;
            }
            if inline_component_neighbors(group, *second) {
                group = union_rect(group, *second);
                count += 1;
            }
        }
        if count >= 2 && group.width >= 24.0 && group.height >= 10.0 && group.height <= 72.0 {
            groups.push(group);
        }
    }
    groups
}

fn inline_component_neighbors(a: Rect, b: Rect) -> bool {
    let vertical_overlap = interval_overlap(a.y, a.y + a.height, b.y, b.y + b.height);
    let min_height = a.height.min(b.height).max(1.0);
    let gap_x = if a.x + a.width < b.x {
        b.x - (a.x + a.width)
    } else if b.x + b.width < a.x {
        a.x - (b.x + b.width)
    } else {
        0.0
    };
    vertical_overlap / min_height >= 0.45 && gap_x <= 18.0
}

fn normalize_detected_boxes(
    mut boxes: Vec<Rect>,
    image_width: u32,
    image_height: u32,
) -> Vec<Rect> {
    boxes = merge_nearby_boxes(boxes, 4.0);
    boxes.retain(|rect| candidate_box_allowed(*rect, image_width, image_height));
    boxes = suppress_duplicate_boxes(boxes);
    boxes.sort_by(|a, b| {
        a.y.partial_cmp(&b.y)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal))
    });
    boxes.truncate(MAX_VISION_CANDIDATES);
    boxes
}

fn suppress_duplicate_boxes(mut boxes: Vec<Rect>) -> Vec<Rect> {
    boxes.sort_by(|a, b| {
        rect_area(*b)
            .partial_cmp(&rect_area(*a))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut kept = Vec::<Rect>::new();
    'candidate: for rect in boxes {
        for kept_rect in &kept {
            let overlap = overlap_ratio(rect, *kept_rect);
            let candidate_area = rect_area(rect);
            let kept_area = rect_area(*kept_rect);
            let contained_by_kept = rect_contains_rect(*kept_rect, rect);
            let contains_kept = rect_contains_rect(rect, *kept_rect);
            let similar_nested = (contained_by_kept || contains_kept)
                && candidate_area.min(kept_area) / candidate_area.max(kept_area).max(1.0) >= 0.50;
            if overlap >= 0.65 || similar_nested {
                continue 'candidate;
            }
        }
        kept.push(rect);
    }
    kept
}

fn merge_nearby_boxes(mut boxes: Vec<Rect>, padding: f64) -> Vec<Rect> {
    let mut changed = true;
    while changed {
        changed = false;
        let mut merged = Vec::new();
        while let Some(rect) = boxes.pop() {
            if let Some(index) = boxes.iter().position(|other| {
                rects_overlap(expand_rect(rect, padding), expand_rect(*other, padding))
            }) {
                let other = boxes.swap_remove(index);
                boxes.push(union_rect(rect, other));
                changed = true;
            } else {
                merged.push(rect);
            }
        }
        boxes = merged;
    }
    boxes
}

fn candidate_box_allowed(rect: Rect, image_width: u32, image_height: u32) -> bool {
    let area = rect.width * rect.height;
    let image_area = image_width as f64 * image_height as f64;
    rect.width >= 10.0
        && rect.height >= 10.0
        && area >= 100.0
        && area <= image_area * 0.60
        && rect.width <= image_width as f64 * 0.92
        && rect.height <= image_height as f64 * 0.92
}

fn expand_rect(rect: Rect, padding: f64) -> Rect {
    Rect {
        x: rect.x - padding,
        y: rect.y - padding,
        width: rect.width + padding * 2.0,
        height: rect.height + padding * 2.0,
    }
}

fn rects_overlap(a: Rect, b: Rect) -> bool {
    a.x < b.x + b.width && a.x + a.width > b.x && a.y < b.y + b.height && a.y + a.height > b.y
}

fn interval_overlap(a1: f64, a2: f64, b1: f64, b2: f64) -> f64 {
    (a2.min(b2) - a1.max(b1)).max(0.0)
}

fn overlap_area(a: Rect, b: Rect) -> f64 {
    interval_overlap(a.x, a.x + a.width, b.x, b.x + b.width)
        * interval_overlap(a.y, a.y + a.height, b.y, b.y + b.height)
}

fn overlap_ratio(a: Rect, b: Rect) -> f64 {
    let overlap = overlap_area(a, b);
    if overlap <= 0.0 {
        return 0.0;
    }
    overlap / rect_area(a).min(rect_area(b)).max(1.0)
}

fn rect_area(rect: Rect) -> f64 {
    (rect.width.max(0.0)) * (rect.height.max(0.0))
}

fn rect_contains_rect(outer: Rect, inner: Rect) -> bool {
    inner.x >= outer.x
        && inner.y >= outer.y
        && inner.x + inner.width <= outer.x + outer.width
        && inner.y + inner.height <= outer.y + outer.height
}

fn union_rect(a: Rect, b: Rect) -> Rect {
    let x1 = a.x.min(b.x);
    let y1 = a.y.min(b.y);
    let x2 = (a.x + a.width).max(b.x + b.width);
    let y2 = (a.y + a.height).max(b.y + b.height);
    Rect {
        x: x1,
        y: y1,
        width: x2 - x1,
        height: y2 - y1,
    }
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct XcapWindowCapturer;

#[cfg(target_os = "macos")]
impl VisionWindowCapturer for XcapWindowCapturer {
    type Window = xcap::Window;

    fn focused_window(
        &self,
        focused_pid: Option<i32>,
    ) -> Result<Option<Self::Window>, ObservationError> {
        let windows = xcap::Window::all().map_err(|err| {
            ObservationError::AxReadFailed(format!("xcap window list failed: {err}"))
        })?;
        let mut visible = windows
            .into_iter()
            .filter(|window| !window.is_minimized().unwrap_or(true))
            .collect::<Vec<_>>();

        if let Some(index) = visible
            .iter()
            .position(|window| window.is_focused().unwrap_or(false))
        {
            return Ok(Some(visible.swap_remove(index)));
        }

        if let Some(pid) = focused_pid {
            if let Some(index) = visible
                .iter()
                .position(|window| window.pid().ok().map(|value| value as i32) == Some(pid))
            {
                return Ok(Some(visible.swap_remove(index)));
            }
        }

        Ok(visible.into_iter().next())
    }

    fn window_info(&self, window: &Self::Window) -> Result<FocusedWindowInfo, ObservationError> {
        let monitor = window.current_monitor().map_err(|err| {
            ObservationError::AxReadFailed(format!("xcap monitor lookup failed: {err}"))
        })?;
        let scale_factor = monitor
            .scale_factor()
            .map(|value| value as f64)
            .unwrap_or(1.0);
        Ok(FocusedWindowInfo {
            origin_x: window.x().unwrap_or(0) as f64,
            origin_y: window.y().unwrap_or(0) as f64,
            width_pixels: window.width().unwrap_or(0),
            height_pixels: window.height().unwrap_or(0),
            scale_factor,
        })
    }

    fn capture_window(&self, window: &Self::Window) -> Result<RgbaImage, ObservationError> {
        window.capture_image().map_err(|err| {
            ObservationError::AxReadFailed(format!("xcap window capture failed: {err}"))
        })
    }

    fn capture_observation_surface(
        &self,
        focused_pid: Option<i32>,
    ) -> Result<Option<(RgbaImage, FocusedWindowInfo)>, ObservationError> {
        let Some(window) = self.focused_window(focused_pid)? else {
            return Ok(None);
        };
        let monitor = window.current_monitor().map_err(|err| {
            ObservationError::AxReadFailed(format!("xcap monitor lookup failed: {err}"))
        })?;
        let scale_factor = monitor
            .scale_factor()
            .map(|value| value as f64)
            .unwrap_or(1.0);
        let image = monitor.capture_image().map_err(|err| {
            ObservationError::AxReadFailed(format!("xcap monitor capture failed: {err}"))
        })?;
        let info = FocusedWindowInfo {
            origin_x: monitor.x().unwrap_or(0) as f64,
            origin_y: monitor.y().unwrap_or(0) as f64,
            width_pixels: image.width(),
            height_pixels: image.height(),
            scale_factor,
        };
        Ok(Some((image, info)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::types::{ChangeWait, UiChangeSignal};
    use std::cell::Cell;

    #[test]
    fn coordinate_conversion_keeps_ax_points_and_maps_retina_pixels() {
        assert_eq!(
            click_point_from_rect(
                Rect {
                    x: 10.0,
                    y: 20.0,
                    width: 100.0,
                    height: 40.0,
                },
                CoordinateSpace::AxPoints,
            ),
            (60, 40)
        );
        assert_eq!(
            click_point_from_rect(
                Rect {
                    x: 40.0,
                    y: 60.0,
                    width: 20.0,
                    height: 20.0,
                },
                CoordinateSpace::WindowPixels {
                    origin_x: 100.0,
                    origin_y: 200.0,
                    scale_factor: 2.0,
                },
            ),
            (125, 235)
        );
    }

    #[test]
    fn detected_boxes_become_pixel_space_elements_with_stable_ids() {
        let info = FocusedWindowInfo {
            origin_x: 10.0,
            origin_y: 20.0,
            width_pixels: 200,
            height_pixels: 100,
            scale_factor: 2.0,
        };
        let elements = detected_boxes_to_elements(
            &[
                Rect {
                    x: 5.0,
                    y: 10.0,
                    width: 20.0,
                    height: 12.0,
                },
                Rect {
                    x: 40.0,
                    y: 10.0,
                    width: 30.0,
                    height: 12.0,
                },
            ],
            &info,
        );

        assert_eq!(elements.len(), 2);
        assert_eq!(elements[0].id, 1);
        assert_eq!(elements[0].source, ElementSource::VisionDetected);
        assert_eq!(
            elements[0].coordinate_space,
            CoordinateSpace::WindowPixels {
                origin_x: 10.0,
                origin_y: 20.0,
                scale_factor: 2.0,
            }
        );
        assert_eq!(elements[1].id, 2);
    }

    #[test]
    fn mark_rendering_preserves_mapping_and_clips_boxes() {
        let image = RgbaImage::from_pixel(50, 40, Rgba([255, 255, 255, 255]));
        let info = FocusedWindowInfo {
            origin_x: 100.0,
            origin_y: 200.0,
            width_pixels: 50,
            height_pixels: 40,
            scale_factor: 2.0,
        };
        let clipped = ax_points_to_window_pixels(
            Rect {
                x: 95.0,
                y: 205.0,
                width: 20.0,
                height: 10.0,
            },
            &info,
        )
        .and_then(|rect| clip_rect(rect, info.width_pixels, info.height_pixels))
        .unwrap();
        assert_eq!(
            clipped,
            Rect {
                x: 0.0,
                y: 10.0,
                width: 30.0,
                height: 20.0,
            }
        );

        let marks = BTreeMap::from([(7, clipped)]);
        let png = render_marked_png_b64(&image, &marks).unwrap();
        assert!(!png.is_empty());
    }

    #[test]
    fn coarse_detector_finds_simple_rectangles() {
        let mut image = RgbaImage::from_pixel(120, 80, Rgba([255, 255, 255, 255]));
        for x in 20..70 {
            image.put_pixel(x, 20, Rgba([0, 0, 0, 255]));
            image.put_pixel(x, 50, Rgba([0, 0, 0, 255]));
        }
        for y in 20..50 {
            image.put_pixel(20, y, Rgba([0, 0, 0, 255]));
            image.put_pixel(70, y, Rgba([0, 0, 0, 255]));
        }

        let boxes = CoarseCvDetector.detect(&image);
        assert!(!boxes.is_empty());
        assert!(boxes[0].x <= 22.0);
        assert!(boxes[0].y <= 22.0);
    }

    #[test]
    fn coarse_detector_finds_borderless_filled_buttons() {
        let mut image = RgbaImage::from_pixel(180, 100, Rgba([246, 246, 246, 255]));
        fill_rect(
            &mut image,
            Rect {
                x: 30.0,
                y: 24.0,
                width: 76.0,
                height: 28.0,
            },
            Rgba([30, 110, 220, 255]),
        );

        let boxes = CoarseCvDetector.detect(&image);

        assert!(boxes
            .iter()
            .any(|rect| rect_contains_point(*rect, 68.0, 38.0)));
    }

    #[test]
    fn coarse_detector_finds_input_like_rectangles_and_canvas_regions() {
        let mut image = RgbaImage::from_pixel(360, 240, Rgba([242, 243, 245, 255]));
        stroke_rect(
            &mut image,
            Rect {
                x: 24.0,
                y: 28.0,
                width: 180.0,
                height: 34.0,
            },
            Rgba([140, 146, 156, 255]),
        );
        stroke_rect(
            &mut image,
            Rect {
                x: 30.0,
                y: 86.0,
                width: 250.0,
                height: 120.0,
            },
            Rgba([90, 98, 112, 255]),
        );
        fill_rect(
            &mut image,
            Rect {
                x: 72.0,
                y: 122.0,
                width: 34.0,
                height: 34.0,
            },
            Rgba([230, 90, 70, 255]),
        );

        let boxes = CoarseCvDetector.detect(&image);

        assert!(boxes
            .iter()
            .any(|rect| rect_contains_point(*rect, 110.0, 45.0)));
        assert!(boxes
            .iter()
            .any(|rect| rect_contains_point(*rect, 155.0, 146.0)));
    }

    #[test]
    fn coarse_detector_rejects_tiny_noise_and_full_window_containers() {
        let mut image = RgbaImage::from_pixel(160, 120, Rgba([255, 255, 255, 255]));
        stroke_rect(
            &mut image,
            Rect {
                x: 1.0,
                y: 1.0,
                width: 158.0,
                height: 118.0,
            },
            Rgba([0, 0, 0, 255]),
        );
        for i in 0..6 {
            image.put_pixel(20 + i, 20 + i, Rgba([0, 0, 0, 255]));
        }

        let boxes = CoarseCvDetector.detect(&image);

        assert!(boxes.is_empty());
    }

    #[test]
    fn augment_ax_appends_non_overlapping_visual_candidates() {
        let info = FocusedWindowInfo {
            origin_x: 0.0,
            origin_y: 0.0,
            width_pixels: 800,
            height_pixels: 600,
            scale_factor: 1.0,
        };
        let ax = vec![element(9)];
        let (elements, marks, dropped) = augment_ax_with_visual_candidates(
            ax,
            &[
                Rect {
                    x: 10.0,
                    y: 10.0,
                    width: 80.0,
                    height: 24.0,
                },
                Rect {
                    x: 240.0,
                    y: 120.0,
                    width: 90.0,
                    height: 32.0,
                },
            ],
            &info,
        );

        assert_eq!(elements.len(), 2);
        assert_eq!(elements[0].id, 9);
        assert_eq!(elements[0].source, ElementSource::Ax);
        assert_eq!(elements[1].id, 10);
        assert_eq!(elements[1].source, ElementSource::VisionDetected);
        assert!(marks.contains_key(&9));
        assert!(marks.contains_key(&10));
        assert_eq!(dropped, 0);
    }

    #[test]
    fn augment_ax_drops_offscreen_elements_from_marks_observation() {
        let info = FocusedWindowInfo {
            origin_x: 0.0,
            origin_y: 0.0,
            width_pixels: 800,
            height_pixels: 600,
            scale_factor: 1.0,
        };
        let visible = element(9);
        let mut offscreen = element(10);
        // Scrolled above the viewport: negative AX y, fully outside the
        // capture — the failing run kept such ids choosable with no mark.
        offscreen.bounds = Rect {
            x: 100.0,
            y: -500.0,
            width: 80.0,
            height: 24.0,
        };

        let (elements, marks, dropped) =
            augment_ax_with_visual_candidates(vec![visible, offscreen], &[], &info);

        assert_eq!(dropped, 1);
        assert!(elements.iter().any(|element| element.id == 9));
        assert!(
            !elements.iter().any(|element| element.id == 10),
            "an element with no visible mark must not stay choosable in marks mode"
        );
        assert!(marks.contains_key(&9));
        assert!(!marks.contains_key(&10));
    }

    #[test]
    fn draw_mark_places_chip_inside_its_own_box() {
        let mut image = RgbaImage::from_pixel(200, 200, Rgba([255, 255, 255, 255]));
        draw_mark(
            &mut image,
            7,
            Rect {
                x: 40.0,
                y: 60.0,
                width: 100.0,
                height: 30.0,
            },
            None,
        );

        let chip = Rgba([255, 64, 64, 235]);
        // The chip sits inside the box's top-left corner...
        assert_eq!(*image.get_pixel(45, 65), chip);
        // ...not floating above the box, where on dense pages it reads as
        // the previous element's label.
        assert_ne!(*image.get_pixel(45, 45), chip);
    }

    #[test]
    fn usable_ax_skips_vision_grounding() {
        let state = VisionFallbackState::new();
        let capturer = FakeCapturer::large();
        let capture_calls = capturer.capture_calls.clone();
        let observer = VisionFallbackObserver::new(
            FakeBaseObserver::new(vec![Ok(vec![element(1), element(2), element(3)])]),
            capturer,
            FakeDetector::new(vec![Rect {
                x: 220.0,
                y: 16.0,
                width: 300.0,
                height: 36.0,
            }]),
            state.clone(),
            VisionFallbackOptions::default(),
        );

        let obs = observer.observe().unwrap();

        assert_eq!(obs.len(), 3);
        assert_eq!(capture_calls.get(), 0);
        assert_eq!(obs[0].source, ElementSource::Ax);
        assert!(state.context().is_none());
        assert_eq!(state.metadata().source, ObservationSource::Ax);
    }

    #[test]
    fn visual_replan_forces_marked_context_with_usable_ax() {
        let state = VisionFallbackState::new();
        let capturer = FakeCapturer::large();
        let capture_calls = capturer.capture_calls.clone();
        let observer = VisionFallbackObserver::new(
            FakeBaseObserver::new(vec![]),
            capturer,
            FakeDetector::new(vec![Rect {
                x: 220.0,
                y: 16.0,
                width: 300.0,
                height: 36.0,
            }]),
            state.clone(),
            VisionFallbackOptions::default(),
        );
        let ax = vec![element(1), element(2), element(3)];

        let obs = observer
            .observe_for_visual_replan(&ax, "after-no-op:test")
            .unwrap()
            .unwrap();

        assert_eq!(capture_calls.get(), 1);
        assert!(obs.len() >= ax.len());
        assert_eq!(state.metadata().source, ObservationSource::VisionMarks);
        assert_eq!(
            state.metadata().trigger_reason.as_deref(),
            Some("after-no-op:test")
        );
        let context = state.context().unwrap();
        assert_eq!(context.mode, VisionFallbackMode::Marks);
        assert!(!context.image_png_b64.is_empty());
        assert!(!context.element_pixel_bounds.is_empty());
    }

    #[test]
    fn web_elements_skip_vision_augmentation() {
        let state = VisionFallbackState::new();
        let capturer = FakeCapturer::large();
        let capture_calls = capturer.capture_calls.clone();
        let observer = VisionFallbackObserver::new(
            FakeBaseObserver::new(vec![Ok(vec![element(1), web_element(2)])]),
            capturer,
            FakeDetector::new(vec![Rect {
                x: 220.0,
                y: 16.0,
                width: 300.0,
                height: 36.0,
            }]),
            state.clone(),
            VisionFallbackOptions::default(),
        );

        let obs = observer.observe().unwrap();

        assert_eq!(obs.len(), 2);
        assert_eq!(capture_calls.get(), 0);
        assert_eq!(obs[1].source, ElementSource::Web);
        assert!(state.context().is_none());
        assert_eq!(state.metadata().source, ObservationSource::Ax);
    }

    #[test]
    fn ax_error_creates_unmarked_grounding_context() {
        let state = VisionFallbackState::new();
        let capturer = FakeCapturer::large();
        let capture_calls = capturer.capture_calls.clone();
        let observer = VisionFallbackObserver::new(
            FakeBaseObserver::new(vec![Err(ObservationError::AxReadFailed(
                "AX server could not complete the request".into(),
            ))]),
            capturer,
            FakeDetector::new(vec![]),
            state.clone(),
            VisionFallbackOptions::default(),
        );

        let obs = observer.observe().unwrap();

        assert!(obs.is_empty());
        assert_eq!(capture_calls.get(), 1);
        assert_eq!(state.metadata().source, ObservationSource::VisionGrounding);
        assert_eq!(state.metadata().candidate_count, Some(0));
        assert!(state
            .metadata()
            .trigger_reason
            .as_deref()
            .unwrap_or("")
            .starts_with("ax-error:"));
        assert!(state.metadata().visual_state_hash.is_some());
        assert_eq!(state.context().unwrap().mode, VisionFallbackMode::Grounding);
    }

    #[test]
    fn empty_ax_creates_unmarked_grounding_context() {
        let state = VisionFallbackState::new();
        let capturer = FakeCapturer::large();
        let capture_calls = capturer.capture_calls.clone();
        let observer = VisionFallbackObserver::new(
            FakeBaseObserver::new(vec![Ok(Vec::new())]),
            capturer,
            FakeDetector::new(vec![Rect {
                x: 10.0,
                y: 12.0,
                width: 30.0,
                height: 20.0,
            }]),
            state.clone(),
            VisionFallbackOptions::default(),
        );

        let obs = observer.observe().unwrap();

        assert_eq!(capture_calls.get(), 1);
        assert!(obs.is_empty());
        assert_eq!(state.metadata().source, ObservationSource::VisionGrounding);
        assert_eq!(state.metadata().candidate_count, Some(0));
        assert!(state.metadata().visual_state_hash.is_some());
        let context = state.context().unwrap();
        assert_eq!(context.mode, VisionFallbackMode::Grounding);
        assert!(!context.image_png_b64.is_empty());
        assert!(context.element_pixel_bounds.is_empty());
    }

    #[test]
    fn weak_ax_creates_unmarked_grounding_context() {
        let state = VisionFallbackState::new();
        let capturer = FakeCapturer::large();
        let capture_calls = capturer.capture_calls.clone();
        let observer = VisionFallbackObserver::new(
            FakeBaseObserver::new(vec![Ok(vec![element(9)])]),
            capturer,
            FakeDetector::new(vec![]),
            state.clone(),
            VisionFallbackOptions::default(),
        );

        let obs = observer.observe().unwrap();

        assert_eq!(capture_calls.get(), 1);
        assert_eq!(obs.len(), 1);
        assert_eq!(obs[0].source, ElementSource::Ax);
        let context = state.context().unwrap();
        assert_eq!(context.mode, VisionFallbackMode::Grounding);
        assert!(context.element_pixel_bounds.is_empty());
        assert_eq!(state.metadata().source, ObservationSource::VisionGrounding);
        assert_eq!(
            state.metadata().trigger_reason.as_deref(),
            Some("ax-weak:1<3")
        );
    }

    #[test]
    fn partial_ax_small_window_does_not_capture() {
        let state = VisionFallbackState::new();
        let capturer = FakeCapturer::small();
        let capture_calls = capturer.capture_calls.clone();
        let observer = VisionFallbackObserver::new(
            FakeBaseObserver::new(vec![Ok(vec![element(1)])]),
            capturer,
            FakeDetector::new(vec![]),
            state.clone(),
            VisionFallbackOptions::default(),
        );

        let obs = observer.observe().unwrap();

        assert_eq!(obs.len(), 1);
        assert_eq!(capture_calls.get(), 0);
        assert!(state.context().is_none());
    }

    #[test]
    fn empty_ax_uses_grounding_regardless_of_detector_results() {
        let state = VisionFallbackState::new();
        let observer = VisionFallbackObserver::new(
            FakeBaseObserver::new(vec![Ok(Vec::new())]),
            FakeCapturer::large(),
            FakeDetector::new(vec![]),
            state.clone(),
            VisionFallbackOptions::default(),
        );

        let obs = observer.observe().unwrap();

        assert!(obs.is_empty());
        assert_eq!(state.metadata().source, ObservationSource::VisionGrounding);
        assert_eq!(state.context().unwrap().mode, VisionFallbackMode::Grounding);
    }

    #[test]
    fn coordinate_fallback_option_no_longer_enables_direct_coordinate_planning() {
        let state = VisionFallbackState::new();
        let observer = VisionFallbackObserver::new(
            FakeBaseObserver::new(vec![Ok(Vec::new())]),
            FakeCapturer::large(),
            FakeDetector::new(vec![]),
            state.clone(),
            VisionFallbackOptions {
                coordinate_fallback: true,
                ..Default::default()
            },
        );

        let obs = observer.observe().unwrap();

        assert!(obs.is_empty());
        assert_eq!(state.metadata().source, ObservationSource::VisionGrounding);
        assert_eq!(state.context().unwrap().mode, VisionFallbackMode::Grounding);
    }

    #[test]
    fn vision_detected_refresh_redetects_and_updates_bounds_by_signature() {
        let info = FocusedWindowInfo {
            origin_x: 0.0,
            origin_y: 0.0,
            width_pixels: 800,
            height_pixels: 600,
            scale_factor: 1.0,
        };
        let original = detected_boxes_to_elements(
            &[Rect {
                x: 10.0,
                y: 12.0,
                width: 30.0,
                height: 20.0,
            }],
            &info,
        )
        .remove(0);
        let observer = VisionFallbackObserver::new(
            FakeBaseObserver::new(vec![Ok(Vec::new())]),
            FakeCapturer::large(),
            FakeDetector::new(vec![Rect {
                x: 14.0,
                y: 16.0,
                width: 30.0,
                height: 20.0,
            }]),
            VisionFallbackState::new(),
            VisionFallbackOptions::default(),
        );

        let refreshed = observer.refresh_element(&original).unwrap();

        assert_eq!(refreshed.id, original.id);
        assert_eq!(
            refreshed.bounds,
            Rect {
                x: 14.0,
                y: 16.0,
                width: 30.0,
                height: 20.0,
            }
        );
    }

    #[test]
    fn change_signal_forwards_to_the_base_observer() {
        let observer = VisionFallbackObserver::new(
            FakeBaseObserver::new(Vec::new()).with_change_signal(),
            FakeCapturer::large(),
            FakeDetector::new(Vec::new()),
            VisionFallbackState::new(),
            VisionFallbackOptions::default(),
        );

        assert!(observer.change_signal().is_some());

        let without_signal = VisionFallbackObserver::new(
            FakeBaseObserver::new(Vec::new()),
            FakeCapturer::large(),
            FakeDetector::new(Vec::new()),
            VisionFallbackState::new(),
            VisionFallbackOptions::default(),
        );

        assert!(without_signal.change_signal().is_none());
    }

    struct NeverChangeSignal;

    impl UiChangeSignal for NeverChangeSignal {
        fn wait_for_change(&mut self, _timeout: std::time::Duration) -> ChangeWait {
            ChangeWait::TimedOut
        }
    }

    struct FakeBaseObserver {
        observations: RefCell<VecDeque<Result<Vec<Element>, ObservationError>>>,
        offers_change_signal: bool,
    }

    impl FakeBaseObserver {
        fn new(observations: Vec<Result<Vec<Element>, ObservationError>>) -> Self {
            Self {
                observations: RefCell::new(observations.into()),
                offers_change_signal: false,
            }
        }

        fn with_change_signal(mut self) -> Self {
            self.offers_change_signal = true;
            self
        }
    }

    impl ScreenObserver for FakeBaseObserver {
        fn observe(&self) -> Result<Vec<Element>, ObservationError> {
            self.observations
                .borrow_mut()
                .pop_front()
                .unwrap_or_else(|| Ok(Vec::new()))
        }

        fn change_signal(&self) -> Option<Box<dyn UiChangeSignal>> {
            self.offers_change_signal
                .then(|| Box::new(NeverChangeSignal) as Box<dyn UiChangeSignal>)
        }
    }

    impl FocusedAppProvider for FakeBaseObserver {
        fn focused_app(&self) -> Result<FocusedApp, ObservationError> {
            Ok(FocusedApp {
                bundle_id: Some("com.example.app".into()),
                name: "Example".into(),
                pid: Some(42),
            })
        }
    }

    #[derive(Clone)]
    struct FakeCapturer {
        info: FocusedWindowInfo,
        capture_calls: Rc<Cell<u32>>,
    }

    impl FakeCapturer {
        fn large() -> Self {
            Self {
                info: FocusedWindowInfo {
                    origin_x: 0.0,
                    origin_y: 0.0,
                    width_pixels: 800,
                    height_pixels: 600,
                    scale_factor: 1.0,
                },
                capture_calls: Rc::new(Cell::new(0)),
            }
        }

        fn small() -> Self {
            Self {
                info: FocusedWindowInfo {
                    width_pixels: 100,
                    height_pixels: 100,
                    ..Self::large().info
                },
                capture_calls: Rc::new(Cell::new(0)),
            }
        }
    }

    impl VisionWindowCapturer for FakeCapturer {
        type Window = ();

        fn focused_window(
            &self,
            _focused_pid: Option<i32>,
        ) -> Result<Option<Self::Window>, ObservationError> {
            Ok(Some(()))
        }

        fn window_info(
            &self,
            _window: &Self::Window,
        ) -> Result<FocusedWindowInfo, ObservationError> {
            Ok(self.info.clone())
        }

        fn capture_window(&self, _window: &Self::Window) -> Result<RgbaImage, ObservationError> {
            self.capture_calls.set(self.capture_calls.get() + 1);
            Ok(RgbaImage::from_pixel(
                self.info.width_pixels,
                self.info.height_pixels,
                Rgba([255, 255, 255, 255]),
            ))
        }
    }

    #[derive(Clone)]
    struct FakeDetector {
        boxes: Vec<Rect>,
    }

    impl FakeDetector {
        fn new(boxes: Vec<Rect>) -> Self {
            Self { boxes }
        }
    }

    impl VisionCandidateDetector for FakeDetector {
        fn kind(&self) -> &'static str {
            "fake"
        }

        fn detect(&self, _image: &RgbaImage) -> Vec<Rect> {
            self.boxes.clone()
        }
    }

    fn element(id: u32) -> Element {
        Element::new(
            id,
            "AXButton".into(),
            format!("Element {id}"),
            None,
            Rect {
                x: 10.0,
                y: 10.0,
                width: 80.0,
                height: 24.0,
            },
            true,
            false,
            CoordinateSpace::AxPoints,
            ElementSource::Ax,
        )
    }

    fn web_element(id: u32) -> Element {
        Element::new(
            id,
            "AXButton".into(),
            format!("Web Element {id}"),
            None,
            Rect {
                x: 120.0,
                y: 140.0,
                width: 80.0,
                height: 24.0,
            },
            true,
            false,
            CoordinateSpace::AxPoints,
            ElementSource::Web,
        )
    }

    fn fill_rect(image: &mut RgbaImage, rect: Rect, color: Rgba<u8>) {
        let x1 = rect.x.max(0.0) as u32;
        let y1 = rect.y.max(0.0) as u32;
        let x2 = (rect.x + rect.width).min(image.width() as f64) as u32;
        let y2 = (rect.y + rect.height).min(image.height() as f64) as u32;
        for y in y1..y2 {
            for x in x1..x2 {
                image.put_pixel(x, y, color);
            }
        }
    }

    fn stroke_rect(image: &mut RgbaImage, rect: Rect, color: Rgba<u8>) {
        let x1 = rect.x.max(0.0) as u32;
        let y1 = rect.y.max(0.0) as u32;
        let x2 = (rect.x + rect.width).min(image.width().saturating_sub(1) as f64) as u32;
        let y2 = (rect.y + rect.height).min(image.height().saturating_sub(1) as f64) as u32;
        for x in x1..=x2 {
            image.put_pixel(x, y1, color);
            image.put_pixel(x, y2, color);
        }
        for y in y1..=y2 {
            image.put_pixel(x1, y, color);
            image.put_pixel(x2, y, color);
        }
    }

    fn rect_contains_point(rect: Rect, x: f64, y: f64) -> bool {
        x >= rect.x && y >= rect.y && x <= rect.x + rect.width && y <= rect.y + rect.height
    }
}
