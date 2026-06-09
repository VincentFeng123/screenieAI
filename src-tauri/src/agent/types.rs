use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

const SIGNATURE_RECT_BUCKET_SIZE: f64 = 24.0;
const FNV_1A_64_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_1A_64_PRIME: u64 = 0x100000001b3;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Rect {
    /// Screen-space rectangle in platform logical points.
    ///
    /// `MacObserver` returns macOS Accessibility screen points with a
    /// top-left origin. For Milestone 3, those AX points and enigo's CGEvent
    /// global positioning are assumed to be 1:1. Screenshot pixel conversion
    /// belongs to a later milestone.
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

impl Rect {
    pub fn center(self) -> (f64, f64) {
        (self.x + (self.width / 2.0), self.y + (self.height / 2.0))
    }

    pub fn is_empty(self) -> bool {
        self.width <= 0.0 || self.height <= 0.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum CoordinateSpace {
    AxPoints,
    WindowPixels {
        origin_x: f64,
        origin_y: f64,
        scale_factor: f64,
    },
}

impl Default for CoordinateSpace {
    fn default() -> Self {
        Self::AxPoints
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ElementSource {
    #[default]
    Ax,
    Web,
    VisionDetected,
    VisionCoordinate,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum PlatformElementHandle {
    MacAx { key: u64 },
    SafariDom { agent_id: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Element {
    pub id: u32,
    pub signature: String,
    pub role: String,
    pub name: String,
    pub value: Option<String>,
    pub bounds: Rect,
    pub enabled: bool,
    pub focused: bool,
    /// Text currently selected inside the element. Only captured for the
    /// focused element; excluded from `element_signature` and `PartialEq` so
    /// selection changes refresh the semantic state hash without churning
    /// element identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_text: Option<String>,
    #[serde(default)]
    pub coordinate_space: CoordinateSpace,
    #[serde(default)]
    pub source: ElementSource,
    #[serde(default, skip)]
    pub platform_handle: Option<PlatformElementHandle>,
}

impl Element {
    pub fn new(
        id: u32,
        role: String,
        name: String,
        value: Option<String>,
        bounds: Rect,
        enabled: bool,
        focused: bool,
        coordinate_space: CoordinateSpace,
        source: ElementSource,
    ) -> Self {
        let signature = element_signature(&role, &name, value.is_some(), source, bounds);
        Self {
            id,
            signature,
            role,
            name,
            value,
            bounds,
            enabled,
            focused,
            selected_text: None,
            coordinate_space,
            source,
            platform_handle: None,
        }
    }

    pub fn with_platform_handle(mut self, platform_handle: PlatformElementHandle) -> Self {
        self.platform_handle = Some(platform_handle);
        self
    }

    pub fn with_selected_text(mut self, selected_text: Option<String>) -> Self {
        self.selected_text = selected_text;
        self
    }

    pub fn refresh_signature(&mut self) {
        self.signature = element_signature(
            &self.role,
            &self.name,
            self.value.is_some(),
            self.source,
            self.bounds,
        );
    }
}

impl PartialEq for Element {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.signature == other.signature
            && self.role == other.role
            && self.name == other.name
            && self.value == other.value
            && self.bounds == other.bounds
            && self.enabled == other.enabled
            && self.focused == other.focused
            && self.coordinate_space == other.coordinate_space
            && self.source == other.source
    }
}

pub fn element_signature(
    role: &str,
    name: &str,
    value_present: bool,
    source: ElementSource,
    bounds: Rect,
) -> String {
    let normalized_name = normalize_signature_name(name, source);
    let coarse = coarse_rect_key(bounds);
    fnv1a_64_hex(&format!(
        "{role}\u{1f}{normalized_name}\u{1f}{value_present}\u{1f}{source:?}\u{1f}{coarse}"
    ))
}

/// Secure (password) fields: their values are never read or logged, and
/// typing into one always requires explicit confirmation.
pub fn is_secure_text_role(role: &str) -> bool {
    role == "AXSecureTextField"
}

pub fn normalize_signature_name(name: &str, source: ElementSource) -> String {
    let normalized = name.split_whitespace().collect::<Vec<_>>().join(" ");
    match source {
        ElementSource::VisionDetected if normalized.starts_with("Visual candidate ") => {
            "Visual candidate".into()
        }
        ElementSource::VisionCoordinate if normalized.starts_with("Vision coordinate ") => {
            "Vision coordinate".into()
        }
        _ => normalized,
    }
}

fn coarse_rect_key(bounds: Rect) -> String {
    format!(
        "{},{},{},{}",
        coarse_bucket(bounds.x),
        coarse_bucket(bounds.y),
        coarse_bucket(bounds.width),
        coarse_bucket(bounds.height)
    )
}

fn coarse_bucket(value: f64) -> i64 {
    if value.is_finite() {
        (value / SIGNATURE_RECT_BUCKET_SIZE).floor() as i64
    } else {
        0
    }
}

fn fnv1a_64_hex(value: &str) -> String {
    let mut hash = FNV_1A_64_OFFSET;
    for byte in value.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(FNV_1A_64_PRIME);
    }
    format!("{hash:016x}")
}

#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum ObservationError {
    #[error("Accessibility permission is required to inspect the frontmost app")]
    AccessibilityPermissionMissing,
    #[error("No frontmost application is available")]
    NoFrontmostApp,
    #[error("Accessibility tree read failed: {0}")]
    AxReadFailed(String),
    #[error("{0}")]
    SafariDomReadFailed(String),
    #[error("Screen observation is not supported on this platform")]
    UnsupportedPlatform,
}

/// Result of resolving and pressing a menu-bar title path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MenuPressOutcome {
    Pressed {
        resolved_path: Vec<String>,
    },
    /// `path[depth]` did not match; `available` lists that level's real
    /// titles so the planner can self-correct.
    NotFound {
        depth: usize,
        available: Vec<String>,
    },
}

pub trait ScreenObserver {
    fn observe(&self) -> Result<Vec<Element>, ObservationError>;

    /// Extract readable text from the focused page/window for the agent's
    /// readPage action. Platform observers override this; the default keeps
    /// stub and test observers working without page-reading support.
    fn read_page_text(&self) -> Result<String, String> {
        Err("page reading is not supported by this observer".into())
    }

    /// Try the platform's semantic press (kAXPressAction on macOS) on the
    /// element's retained platform handle. `Ok(true)` means the press was
    /// delivered; `Ok(false)` means no handle or unsupported here, and the
    /// caller falls back to a synthetic click. Effect verification stays with
    /// the caller either way.
    fn perform_press(&self, _el: &Element) -> Result<bool, String> {
        Ok(false)
    }

    /// Try writing `text` straight into the element's value (AXValue on
    /// macOS) instead of synthesizing keystrokes. `Ok(true)` only when the
    /// write stuck (read-back matches) and keyboard focus landed on the
    /// element; `Ok(false)` falls back to click+type.
    fn set_value(&self, _el: &Element, _text: &str) -> Result<bool, String> {
        Ok(false)
    }

    /// Resolve a menu-bar title path on the frontmost app and press the leaf
    /// item. Platform observers override this.
    fn press_menu_path(&self, _path: &[String]) -> Result<MenuPressOutcome, String> {
        Err("menu actions are not supported by this observer".into())
    }

    fn refresh_element(&self, el: &Element) -> Option<Element> {
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
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FocusedApp {
    pub bundle_id: Option<String>,
    pub name: String,
    pub pid: Option<i32>,
}

pub trait FocusedAppProvider {
    fn focused_app(&self) -> Result<FocusedApp, ObservationError>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ActionTargetRef {
    Id(u32),
    Target(String),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    ActivateApp { app: String },
    Click { id: u32 },
    ClickTarget { target: String },
    DoubleClick { id: u32 },
    DoubleClickTarget { target: String },
    Type { id: u32, text: String },
    TypeTarget { target: String, text: String },
    TypeFocused { text: String },
    RightClick { id: u32 },
    Move { id: u32 },
    Drag { from_id: u32, to_id: u32 },
    Key { combo: String },
    /// Press one item in the frontmost app's menu bar by title path, e.g.
    /// `["File", "Export as PDF…"]`.
    Menu { path: Vec<String> },
    Scroll { dx: i32, dy: i32 },
    ScrollAt { id: u32, dx: i32, dy: i32 },
    Wait { ms: u64 },
    /// Open a URL in the default browser in one deterministic step.
    OpenUrl { url: String },
    /// Run a web search in the default browser in one deterministic step.
    WebSearch { query: String },
    /// Extract the visible text of the focused page/window into the
    /// planner's next prompt. Emits no input events.
    ReadPage,
    /// Pause and ask the user one short question; the answer arrives in the
    /// planner's history. Emits no input events.
    Ask { question: String, options: Vec<String> },
    /// Run an AppleScript snippet. Gated by a user setting (default off) and
    /// always requires explicit approval; `do shell script` is rejected.
    AppleScript { script: String },
    /// Run a Shortcuts.app shortcut by name. Same gating as AppleScript.
    RunShortcut { name: String, input: Option<String> },
    /// Move a file to the Trash via Finder. Permanent deletion does not
    /// exist as a primitive.
    MoveToTrash { path: String },
    Done,
    Fail { reason: String },
}

impl Serialize for Action {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let field_count = match self {
            Self::Done | Self::ReadPage => 1,
            Self::ActivateApp { .. }
            | Self::Click { .. }
            | Self::ClickTarget { .. }
            | Self::DoubleClick { .. }
            | Self::DoubleClickTarget { .. }
            | Self::TypeFocused { .. }
            | Self::RightClick { .. }
            | Self::Move { .. }
            | Self::Key { .. }
            | Self::Menu { .. }
            | Self::Scroll { .. }
            | Self::Wait { .. }
            | Self::OpenUrl { .. }
            | Self::WebSearch { .. }
            | Self::AppleScript { .. }
            | Self::MoveToTrash { .. }
            | Self::Fail { .. } => 2,
            Self::Type { .. } | Self::TypeTarget { .. } => 3,
            Self::Drag { .. } => 3,
            Self::Ask { .. } => 3,
            Self::RunShortcut { input, .. } => 2 + usize::from(input.is_some()),
            Self::ScrollAt { .. } => 4,
        };
        let mut state = serializer.serialize_struct("Action", field_count)?;
        match self {
            Self::ActivateApp { app } => {
                state.serialize_field("action", "activateApp")?;
                state.serialize_field("app", app)?;
            }
            Self::Click { id } => {
                state.serialize_field("action", "click")?;
                state.serialize_field("id", id)?;
            }
            Self::ClickTarget { target } => {
                state.serialize_field("action", "click")?;
                state.serialize_field("target", target)?;
            }
            Self::DoubleClick { id } => {
                state.serialize_field("action", "doubleClick")?;
                state.serialize_field("id", id)?;
            }
            Self::DoubleClickTarget { target } => {
                state.serialize_field("action", "doubleClick")?;
                state.serialize_field("target", target)?;
            }
            Self::Type { id, text } => {
                state.serialize_field("action", "type")?;
                state.serialize_field("id", id)?;
                state.serialize_field("text", text)?;
            }
            Self::TypeTarget { target, text } => {
                state.serialize_field("action", "type")?;
                state.serialize_field("target", target)?;
                state.serialize_field("text", text)?;
            }
            Self::TypeFocused { text } => {
                state.serialize_field("action", "typeFocused")?;
                state.serialize_field("text", text)?;
            }
            Self::RightClick { id } => {
                state.serialize_field("action", "rightClick")?;
                state.serialize_field("id", id)?;
            }
            Self::Move { id } => {
                state.serialize_field("action", "move")?;
                state.serialize_field("id", id)?;
            }
            Self::Drag { from_id, to_id } => {
                state.serialize_field("action", "drag")?;
                state.serialize_field("fromId", from_id)?;
                state.serialize_field("toId", to_id)?;
            }
            Self::Key { combo } => {
                state.serialize_field("action", "key")?;
                state.serialize_field("combo", combo)?;
            }
            Self::Menu { path } => {
                state.serialize_field("action", "menu")?;
                state.serialize_field("path", path)?;
            }
            Self::Scroll { dx, dy } => {
                state.serialize_field("action", "scroll")?;
                state.serialize_field("dx", dx)?;
                state.serialize_field("dy", dy)?;
            }
            Self::ScrollAt { id, dx, dy } => {
                state.serialize_field("action", "scroll")?;
                state.serialize_field("id", id)?;
                state.serialize_field("dx", dx)?;
                state.serialize_field("dy", dy)?;
            }
            Self::Wait { ms } => {
                state.serialize_field("action", "wait")?;
                state.serialize_field("ms", ms)?;
            }
            Self::OpenUrl { url } => {
                state.serialize_field("action", "openUrl")?;
                state.serialize_field("url", url)?;
            }
            Self::WebSearch { query } => {
                state.serialize_field("action", "webSearch")?;
                state.serialize_field("query", query)?;
            }
            Self::ReadPage => {
                state.serialize_field("action", "readPage")?;
            }
            Self::Ask { question, options } => {
                state.serialize_field("action", "ask")?;
                state.serialize_field("question", question)?;
                state.serialize_field("options", options)?;
            }
            Self::AppleScript { script } => {
                state.serialize_field("action", "applescript")?;
                state.serialize_field("script", script)?;
            }
            Self::RunShortcut { name, input } => {
                state.serialize_field("action", "shortcut")?;
                state.serialize_field("name", name)?;
                if let Some(input) = input {
                    state.serialize_field("input", input)?;
                }
            }
            Self::MoveToTrash { path } => {
                state.serialize_field("action", "moveToTrash")?;
                state.serialize_field("file", path)?;
            }
            Self::Done => {
                state.serialize_field("action", "done")?;
            }
            Self::Fail { reason } => {
                state.serialize_field("action", "fail")?;
                state.serialize_field("reason", reason)?;
            }
        }
        state.end()
    }
}

impl<'de> Deserialize<'de> for Action {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct RawAction {
            action: String,
            app: Option<String>,
            id: Option<u32>,
            #[serde(rename = "fromId")]
            from_id: Option<u32>,
            #[serde(rename = "toId")]
            to_id: Option<u32>,
            target: Option<String>,
            text: Option<String>,
            combo: Option<String>,
            path: Option<Vec<String>>,
            question: Option<String>,
            options: Option<Vec<String>>,
            script: Option<String>,
            name: Option<String>,
            input: Option<String>,
            file: Option<String>,
            dx: Option<i32>,
            dy: Option<i32>,
            ms: Option<u64>,
            url: Option<String>,
            query: Option<String>,
            reason: Option<String>,
            x: Option<serde_json::Value>,
            y: Option<serde_json::Value>,
        }

        let raw = RawAction::deserialize(deserializer)?;
        if raw.x.is_some() || raw.y.is_some() {
            return Err(serde::de::Error::custom(
                "raw coordinates are not valid action targets",
            ));
        }

        fn reject_extra<E>(
            action: &str,
            present: &[(&str, bool)],
            allowed: &[&str],
        ) -> Result<(), E>
        where
            E: serde::de::Error,
        {
            let extras = present
                .iter()
                .filter_map(|(field, is_present)| {
                    (*is_present && !allowed.contains(field)).then_some(*field)
                })
                .collect::<Vec<_>>();
            if extras.is_empty() {
                Ok(())
            } else {
                Err(E::custom(format!(
                    "{action} does not allow field(s): {}",
                    extras.join(", ")
                )))
            }
        }

        let present = [
            ("app", raw.app.is_some()),
            ("id", raw.id.is_some()),
            ("fromId", raw.from_id.is_some()),
            ("toId", raw.to_id.is_some()),
            ("target", raw.target.is_some()),
            ("text", raw.text.is_some()),
            ("combo", raw.combo.is_some()),
            ("path", raw.path.is_some()),
            ("question", raw.question.is_some()),
            ("options", raw.options.is_some()),
            ("script", raw.script.is_some()),
            ("name", raw.name.is_some()),
            ("input", raw.input.is_some()),
            ("file", raw.file.is_some()),
            ("dx", raw.dx.is_some()),
            ("dy", raw.dy.is_some()),
            ("ms", raw.ms.is_some()),
            ("url", raw.url.is_some()),
            ("query", raw.query.is_some()),
            ("reason", raw.reason.is_some()),
        ];

        match raw.action.as_str() {
            "activateApp" | "activate_app" => {
                reject_extra::<D::Error>(&raw.action, &present, &["app"])?;
                let app = required_string::<D::Error>("app", raw.app)?;
                Ok(Self::ActivateApp { app })
            }
            "click" => {
                reject_extra::<D::Error>(&raw.action, &present, &["id", "target"])?;
                match (raw.id, raw.target) {
                    (Some(id), None) => Ok(Self::Click { id }),
                    (None, Some(target)) => Ok(Self::ClickTarget {
                        target: required_string::<D::Error>("target", Some(target))?,
                    }),
                    (Some(_), Some(_)) => Err(serde::de::Error::custom(
                        "click requires either id or target, not both",
                    )),
                    (None, None) => Err(serde::de::Error::custom("click requires id or target")),
                }
            }
            "doubleClick" | "double_click" => {
                reject_extra::<D::Error>(&raw.action, &present, &["id", "target"])?;
                match (raw.id, raw.target) {
                    (Some(id), None) => Ok(Self::DoubleClick { id }),
                    (None, Some(target)) => Ok(Self::DoubleClickTarget {
                        target: required_string::<D::Error>("target", Some(target))?,
                    }),
                    (Some(_), Some(_)) => Err(serde::de::Error::custom(
                        "doubleClick requires either id or target, not both",
                    )),
                    (None, None) => Err(serde::de::Error::custom(
                        "doubleClick requires id or target",
                    )),
                }
            }
            "type" => {
                reject_extra::<D::Error>(&raw.action, &present, &["id", "target", "text"])?;
                let text = raw
                    .text
                    .ok_or_else(|| serde::de::Error::custom("type requires text"))?;
                match (raw.id, raw.target) {
                    (Some(id), None) => Ok(Self::Type { id, text }),
                    (None, Some(target)) => Ok(Self::TypeTarget {
                        target: required_string::<D::Error>("target", Some(target))?,
                        text,
                    }),
                    (Some(_), Some(_)) => Err(serde::de::Error::custom(
                        "type requires either id or target, not both",
                    )),
                    (None, None) => Err(serde::de::Error::custom("type requires id or target")),
                }
            }
            "typeFocused" | "type_focused" => {
                reject_extra::<D::Error>(&raw.action, &present, &["text"])?;
                let text = raw
                    .text
                    .ok_or_else(|| serde::de::Error::custom("typeFocused requires text"))?;
                Ok(Self::TypeFocused { text })
            }
            "rightClick" | "right_click" => {
                reject_extra::<D::Error>(&raw.action, &present, &["id"])?;
                Ok(Self::RightClick {
                    id: raw
                        .id
                        .ok_or_else(|| serde::de::Error::custom("rightClick requires id"))?,
                })
            }
            "move" => {
                reject_extra::<D::Error>(&raw.action, &present, &["id"])?;
                Ok(Self::Move {
                    id: raw
                        .id
                        .ok_or_else(|| serde::de::Error::custom("move requires id"))?,
                })
            }
            "drag" => {
                reject_extra::<D::Error>(&raw.action, &present, &["fromId", "toId"])?;
                Ok(Self::Drag {
                    from_id: raw
                        .from_id
                        .ok_or_else(|| serde::de::Error::custom("drag requires fromId"))?,
                    to_id: raw
                        .to_id
                        .ok_or_else(|| serde::de::Error::custom("drag requires toId"))?,
                })
            }
            "key" => {
                reject_extra::<D::Error>(&raw.action, &present, &["combo"])?;
                let combo = required_string::<D::Error>("combo", raw.combo)?;
                Ok(Self::Key { combo })
            }
            "menu" => {
                reject_extra::<D::Error>(&raw.action, &present, &["path"])?;
                let path = raw
                    .path
                    .unwrap_or_default()
                    .into_iter()
                    .map(|part| part.trim().to_string())
                    .filter(|part| !part.is_empty())
                    .collect::<Vec<_>>();
                if path.is_empty() {
                    return Err(serde::de::Error::custom("menu requires a non-empty path"));
                }
                Ok(Self::Menu { path })
            }
            "scroll" => {
                reject_extra::<D::Error>(&raw.action, &present, &["id", "dx", "dy"])?;
                match raw.id {
                    Some(id) => Ok(Self::ScrollAt {
                        id,
                        dx: raw.dx.unwrap_or(0),
                        dy: raw.dy.unwrap_or(0),
                    }),
                    None => Ok(Self::Scroll {
                        dx: raw.dx.unwrap_or(0),
                        dy: raw.dy.unwrap_or(0),
                    }),
                }
            }
            "wait" => {
                reject_extra::<D::Error>(&raw.action, &present, &["ms"])?;
                Ok(Self::Wait {
                    ms: raw
                        .ms
                        .ok_or_else(|| serde::de::Error::custom("wait requires ms"))?,
                })
            }
            "openUrl" | "open_url" => {
                reject_extra::<D::Error>(&raw.action, &present, &["url"])?;
                let url = required_string::<D::Error>("url", raw.url)?;
                Ok(Self::OpenUrl { url })
            }
            "webSearch" | "web_search" => {
                reject_extra::<D::Error>(&raw.action, &present, &["query"])?;
                let query = required_string::<D::Error>("query", raw.query)?;
                Ok(Self::WebSearch { query })
            }
            "readPage" | "read_page" => {
                reject_extra::<D::Error>(&raw.action, &present, &[])?;
                Ok(Self::ReadPage)
            }
            "ask" => {
                reject_extra::<D::Error>(&raw.action, &present, &["question", "options"])?;
                let question = required_string::<D::Error>("question", raw.question)?;
                let options = raw
                    .options
                    .unwrap_or_default()
                    .into_iter()
                    .map(|option| option.trim().to_string())
                    .filter(|option| !option.is_empty())
                    .collect();
                Ok(Self::Ask { question, options })
            }
            "applescript" | "apple_script" => {
                reject_extra::<D::Error>(&raw.action, &present, &["script"])?;
                let script = required_string::<D::Error>("script", raw.script)?;
                Ok(Self::AppleScript { script })
            }
            "shortcut" | "run_shortcut" | "runShortcut" => {
                reject_extra::<D::Error>(&raw.action, &present, &["name", "input"])?;
                let name = required_string::<D::Error>("name", raw.name)?;
                let input = raw.input.filter(|input| !input.trim().is_empty());
                Ok(Self::RunShortcut { name, input })
            }
            "moveToTrash" | "move_to_trash" => {
                reject_extra::<D::Error>(&raw.action, &present, &["file"])?;
                let path = required_string::<D::Error>("file", raw.file)?;
                Ok(Self::MoveToTrash { path })
            }
            "done" => {
                reject_extra::<D::Error>(&raw.action, &present, &[])?;
                Ok(Self::Done)
            }
            "fail" => {
                reject_extra::<D::Error>(&raw.action, &present, &["reason"])?;
                let reason = required_string::<D::Error>("reason", raw.reason)?;
                Ok(Self::Fail { reason })
            }
            other => Err(serde::de::Error::custom(format!(
                "unknown action '{other}'"
            ))),
        }
    }
}

fn required_string<E>(field: &str, value: Option<String>) -> Result<String, E>
where
    E: serde::de::Error,
{
    value
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| E::custom(format!("{field} is required")))
}

impl Action {
    pub fn from_json_strict(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    pub fn target_ids(&self) -> Vec<u32> {
        match self {
            Self::Click { id }
            | Self::DoubleClick { id }
            | Self::Type { id, .. }
            | Self::RightClick { id }
            | Self::Move { id }
            | Self::ScrollAt { id, .. } => vec![*id],
            Self::Drag { from_id, to_id } => vec![*from_id, *to_id],
            Self::ActivateApp { .. }
            | Self::ClickTarget { .. }
            | Self::DoubleClickTarget { .. }
            | Self::TypeTarget { .. }
            | Self::TypeFocused { .. }
            | Self::Key { .. }
            | Self::Menu { .. }
            | Self::Scroll { .. }
            | Self::Wait { .. }
            | Self::OpenUrl { .. }
            | Self::WebSearch { .. }
            | Self::ReadPage
            | Self::Ask { .. }
            | Self::AppleScript { .. }
            | Self::RunShortcut { .. }
            | Self::MoveToTrash { .. }
            | Self::Done
            | Self::Fail { .. } => Vec::new(),
        }
    }

    pub fn target_ref(&self) -> Option<ActionTargetRef> {
        match self {
            Self::Click { id }
            | Self::DoubleClick { id }
            | Self::Type { id, .. }
            | Self::RightClick { id }
            | Self::Move { id }
            | Self::ScrollAt { id, .. }
            | Self::Drag { from_id: id, .. } => Some(ActionTargetRef::Id(*id)),
            Self::ClickTarget { target }
            | Self::DoubleClickTarget { target }
            | Self::TypeTarget { target, .. } => Some(ActionTargetRef::Target(target.clone())),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlannerDecision {
    pub reason: String,
    pub action: Action,
    /// Optional batched follow-up actions (≤2) the planner is confident
    /// about; each runs only if the previous action verifiably progressed,
    /// without another model call.
    pub followups: Vec<Action>,
    /// Fact the planner wants remembered across steps (prices, names, URLs).
    pub note: Option<String>,
    /// Short phrase the planner expects to be visible after the action.
    pub expect: Option<String>,
    /// True when the planner judges the current milestone visibly complete.
    pub milestone_done: bool,
}

impl PlannerDecision {
    pub fn new(reason: impl Into<String>, action: Action) -> Self {
        Self {
            reason: reason.into(),
            action,
            followups: Vec::new(),
            note: None,
            expect: None,
            milestone_done: false,
        }
    }

    pub fn with_followups(mut self, followups: Vec<Action>) -> Self {
        self.followups = followups;
        self
    }

    pub fn with_note(mut self, note: Option<String>) -> Self {
        self.note = note;
        self
    }

    pub fn with_expect(mut self, expect: Option<String>) -> Self {
        self.expect = expect;
        self
    }

    pub fn with_milestone_done(mut self, milestone_done: bool) -> Self {
        self.milestone_done = milestone_done;
        self
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlannerHistoryEntry {
    pub action: Action,
    pub reason: String,
    pub result: String,
}

impl PlannerHistoryEntry {
    pub fn new(action: Action, reason: impl Into<String>, result: impl Into<String>) -> Self {
        Self {
            action,
            reason: reason.into(),
            result: result.into(),
        }
    }
}

#[async_trait::async_trait(?Send)]
pub trait Planner {
    async fn next_action(
        &self,
        goal: &str,
        obs: &[Element],
        history: &[PlannerHistoryEntry],
    ) -> PlannerDecision;

    /// One-shot decomposition of the goal into 3-6 short observable
    /// milestones, run once at task start. The default (used by stub and
    /// test planners) returns no milestones, which disables the plan block.
    async fn plan_milestones(&self, _goal: &str) -> Vec<String> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        element_signature, Action, CoordinateSpace, Element, ElementSource, PlatformElementHandle,
        Rect, ScreenObserver,
    };

    #[test]
    fn action_json_contract_uses_action_tag() {
        let action = Action::DoubleClick { id: 42 };
        assert_eq!(
            action.to_json().unwrap(),
            r#"{"action":"doubleClick","id":42}"#
        );
    }

    #[test]
    fn action_json_contract_supports_target_ref_without_changing_id_shape() {
        let action = Action::ClickTarget {
            target: "the blue Send button".into(),
        };
        assert_eq!(
            action.to_json().unwrap(),
            r#"{"action":"click","target":"the blue Send button"}"#
        );
        assert_eq!(
            Action::from_json_strict(r#"{"action":"click","target":"the blue Send button"}"#)
                .unwrap(),
            action
        );
    }

    #[test]
    fn action_json_contract_serializes_app_activation() {
        let action = Action::ActivateApp {
            app: "Safari".into(),
        };
        assert_eq!(
            action.to_json().unwrap(),
            r#"{"action":"activateApp","app":"Safari"}"#
        );
    }

    #[test]
    fn action_json_contract_rejects_raw_coordinates() {
        let json = r#"{"action":"click","id":7,"x":10,"y":20}"#;
        assert!(Action::from_json_strict(json).is_err());
    }

    #[test]
    fn action_json_contract_parses_done() {
        let action = Action::from_json_strict(r#"{"action":"done"}"#).unwrap();
        assert_eq!(action, Action::Done);
    }

    #[test]
    fn action_json_contract_round_trips_menu_paths() {
        let menu = Action::Menu {
            path: vec!["File".into(), "Export as PDF…".into()],
        };
        let json = menu.to_json().unwrap();
        assert_eq!(
            json,
            r#"{"action":"menu","path":["File","Export as PDF…"]}"#
        );
        assert_eq!(Action::from_json_strict(&json).unwrap(), menu);

        assert!(Action::from_json_strict(r#"{"action":"menu"}"#).is_err());
        assert!(Action::from_json_strict(r#"{"action":"menu","path":[]}"#).is_err());
        assert!(Action::from_json_strict(r#"{"action":"menu","path":["  "]}"#).is_err());
    }

    #[test]
    fn action_json_contract_round_trips_macro_actions() {
        let open = Action::OpenUrl {
            url: "https://example.com".into(),
        };
        let open_json = open.to_json().unwrap();
        assert_eq!(
            open_json,
            r#"{"action":"openUrl","url":"https://example.com"}"#
        );
        assert_eq!(Action::from_json_strict(&open_json).unwrap(), open);

        let search = Action::WebSearch {
            query: "refurbished mac mini".into(),
        };
        let search_json = search.to_json().unwrap();
        assert_eq!(
            search_json,
            r#"{"action":"webSearch","query":"refurbished mac mini"}"#
        );
        assert_eq!(Action::from_json_strict(&search_json).unwrap(), search);

        let read = Action::ReadPage;
        let read_json = read.to_json().unwrap();
        assert_eq!(read_json, r#"{"action":"readPage"}"#);
        assert_eq!(Action::from_json_strict(&read_json).unwrap(), read);

        assert!(Action::from_json_strict(r#"{"action":"openUrl"}"#).is_err());
        assert!(Action::from_json_strict(r#"{"action":"readPage","url":"x"}"#).is_err());
    }

    #[test]
    fn element_signature_is_stable_across_ids_and_sensitive_to_semantics() {
        let bounds = Rect {
            x: 10.0,
            y: 20.0,
            width: 80.0,
            height: 24.0,
        };
        let a = Element::new(
            1,
            "AXButton".into(),
            " Save ".into(),
            None,
            bounds,
            true,
            false,
            CoordinateSpace::AxPoints,
            ElementSource::Ax,
        );
        let b = Element::new(
            99,
            "AXButton".into(),
            "Save".into(),
            None,
            bounds,
            true,
            false,
            CoordinateSpace::AxPoints,
            ElementSource::Ax,
        );

        assert_eq!(a.signature, b.signature);
        assert_ne!(
            a.signature,
            element_signature("AXLink", "Save", false, ElementSource::Ax, bounds)
        );
        assert_ne!(
            a.signature,
            element_signature("AXButton", "Save", true, ElementSource::Ax, bounds)
        );
        assert_ne!(
            a.signature,
            element_signature(
                "AXButton",
                "Save",
                false,
                ElementSource::Ax,
                Rect { x: 50.0, ..bounds },
            )
        );
    }

    #[test]
    fn vision_candidate_signature_ignores_display_sequence_label() {
        let bounds = Rect {
            x: 10.0,
            y: 20.0,
            width: 80.0,
            height: 24.0,
        };

        assert_eq!(
            element_signature(
                "VisionCandidate",
                "Visual candidate 1",
                false,
                ElementSource::VisionDetected,
                bounds,
            ),
            element_signature(
                "VisionCandidate",
                "Visual candidate 8",
                false,
                ElementSource::VisionDetected,
                bounds,
            )
        );
    }

    #[test]
    fn element_equality_ignores_platform_handle() {
        let base = Element::new(
            1,
            "AXButton".into(),
            "Save".into(),
            None,
            Rect {
                x: 10.0,
                y: 20.0,
                width: 80.0,
                height: 24.0,
            },
            true,
            false,
            CoordinateSpace::AxPoints,
            ElementSource::Ax,
        );
        let with_handle = base
            .clone()
            .with_platform_handle(PlatformElementHandle::MacAx { key: 42 });

        assert_eq!(base, with_handle);
    }

    #[test]
    fn default_refresh_fallback_requires_unique_signature_match() {
        let original = element(7, "Save");
        let mut refreshed = element(42, "Save");
        refreshed.bounds.x = 12.0;
        refreshed.refresh_signature();
        let unique = StaticObserver(vec![refreshed.clone()]);

        let result = unique.refresh_element(&original).unwrap();
        assert_eq!(result.id, original.id);
        assert_eq!(result.signature, original.signature);

        let missing = StaticObserver(vec![element(1, "Cancel")]);
        assert!(missing.refresh_element(&original).is_none());

        let ambiguous = StaticObserver(vec![element(1, "Save"), element(2, "Save")]);
        assert!(ambiguous.refresh_element(&original).is_none());
    }

    struct StaticObserver(Vec<Element>);

    impl ScreenObserver for StaticObserver {
        fn observe(&self) -> Result<Vec<Element>, super::ObservationError> {
            Ok(self.0.clone())
        }
    }

    fn element(id: u32, name: &str) -> Element {
        Element::new(
            id,
            "AXButton".into(),
            name.into(),
            None,
            Rect {
                x: 10.0,
                y: 20.0,
                width: 80.0,
                height: 24.0,
            },
            true,
            false,
            CoordinateSpace::AxPoints,
            ElementSource::Ax,
        )
    }
}
