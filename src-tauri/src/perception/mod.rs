//! Structured perception: `see` (walk + index) → `read` (query) → `look`
//! (set-of-marks frame).
//!
//! The agent's default sense is a structured query over the latest snapshot
//! — kilobytes of element text in milliseconds. Screenshots are the
//! fallback tier, not the default. Element IDs are namespaced and
//! snapshot-scoped (`ax:B3` + snapshot id); the executor resolves them
//! through [`ids`]'s resolver, which rejects stale or wrong-namespace refs
//! with typed errors.

pub mod annotate;
pub mod ax;
pub mod commands;
pub mod geometry;
pub mod ids;
pub mod index;
pub mod serialize;

pub use geometry::RectPt;
pub use ids::{ElementRow, ResolvedElement};
pub use index::{ElementQuery, SnapshotMeta};

/// Namespaces the resolver understands enough to redirect: `ax:` is handled
/// here, `dom:` (browser extension) and `vis:` (vision marks) get a typed
/// WrongNamespace pointing the caller at the right channel.
pub const RESOLVE_NAMESPACES: &[&str] = &["ax", "dom", "vis"];

use serde::ser::SerializeStruct;
use serde::{Deserialize, Serialize, Serializer};

/// What a `see` walks. Frontmost window is the default and the cheap path.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "kind")]
pub enum SeeScope {
    #[default]
    FrontmostWindow,
    /// One app by pid. `bundle` (when known) drives the browser check that
    /// decides whether web areas are boundaries or walked content.
    App {
        pid: i32,
        #[serde(default)]
        bundle: Option<String>,
    },
    /// All on-screen windows, element budget shared across them.
    Screen,
}

/// Browsers whose web content the `dom:` namespace (browser extension path)
/// owns — their AXWebAreas are recorded as boundaries. Everything else
/// (Electron and friends) gets its web areas walked: no DOM channel can
/// exist for them.
pub const BROWSER_BUNDLES: &[&str] = &[
    "com.apple.Safari",
    "com.google.Chrome",
    "org.mozilla.firefox",
    "com.microsoft.edgemac",
    "company.thebrowser.Browser",
    "com.brave.Browser",
    "com.operasoftware.Opera",
    "com.vivaldi.Vivaldi",
];

/// `None` (unknown bundle) is treated as not-a-browser: descending a web
/// area needlessly costs budget; treating a browser's page as unreachable
/// costs the task.
pub fn is_known_browser(bundle: Option<&str>) -> bool {
    bundle
        .map(|bundle| BROWSER_BUNDLES.iter().any(|known| known == &bundle))
        .unwrap_or(false)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct SeeOptions {
    pub scope: SeeScope,
    pub element_budget: usize,
    pub max_depth: u8,
    pub descend_web_areas: bool,
}

impl Default for SeeOptions {
    fn default() -> Self {
        Self {
            scope: SeeScope::FrontmostWindow,
            element_budget: 1500,
            max_depth: 30,
            descend_web_areas: false,
        }
    }
}

/// Which permission a [`PerceptionError::Permission`] is about; carries the
/// System Settings deep link so the frontend can one-click the right pane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PermissionKind {
    Accessibility,
    ScreenRecording,
}

impl PermissionKind {
    pub fn which(&self) -> &'static str {
        match self {
            PermissionKind::Accessibility => "accessibility",
            PermissionKind::ScreenRecording => "screen_recording",
        }
    }

    pub fn settings_url(&self) -> &'static str {
        match self {
            PermissionKind::Accessibility => {
                "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
            }
            PermissionKind::ScreenRecording => {
                "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture"
            }
        }
    }
}

/// Perception's error contract. Serializes structured (not a bare string —
/// unlike `AiError`) because the permission case must carry `which` +
/// `settings_url` for the frontend's one-click flow (Phase 6 contract).
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum PerceptionError {
    #[error("{0:?} permission is required")]
    Permission(PermissionKind),
    /// AX reads degraded (API disabled / no readable tree). Carries
    /// `ax_empty` semantics: the agent's ladder falls through to vision.
    #[error("Accessibility tree unavailable: {0}")]
    AxEmpty(String),
    #[error("No frontmost application window to walk")]
    NoTarget,
    #[error("Snapshot {snapshot_id} is no longer the latest for its window")]
    StaleSnapshot { snapshot_id: String },
    #[error("Element id {eid} belongs to the {ns} namespace, not this resolver")]
    WrongNamespace { eid: String, ns: String },
    #[error("Unknown element id {eid} in snapshot {snapshot_id}")]
    UnknownElement { eid: String, snapshot_id: String },
    #[error("Snapshot {0} not found")]
    UnknownSnapshot(String),
    #[error("Index error: {0}")]
    Index(String),
    #[error("Capture error: {0}")]
    Capture(String),
    #[error("Invalid query: {0}")]
    InvalidQuery(String),
    #[error("Perception is not supported on this platform")]
    Unsupported,
}

impl Serialize for PerceptionError {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut out = serializer.serialize_struct("PerceptionError", 4)?;
        match self {
            PerceptionError::Permission(kind) => {
                out.serialize_field("kind", "permission")?;
                out.serialize_field("which", kind.which())?;
                out.serialize_field("settingsUrl", kind.settings_url())?;
                out.serialize_field("message", &self.to_string())?;
            }
            PerceptionError::AxEmpty(_) => {
                out.serialize_field("kind", "axEmpty")?;
                out.serialize_field("axEmpty", &true)?;
                out.serialize_field("message", &self.to_string())?;
            }
            PerceptionError::StaleSnapshot { snapshot_id } => {
                out.serialize_field("kind", "staleSnapshot")?;
                out.serialize_field("snapshotId", snapshot_id)?;
                out.serialize_field("message", &self.to_string())?;
            }
            PerceptionError::WrongNamespace { ns, .. } => {
                out.serialize_field("kind", "wrongNamespace")?;
                out.serialize_field("ns", ns)?;
                out.serialize_field("message", &self.to_string())?;
            }
            _ => {
                out.serialize_field("kind", "error")?;
                out.serialize_field("message", &self.to_string())?;
            }
        }
        out.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permission_error_serializes_structured() {
        let err = PerceptionError::Permission(PermissionKind::Accessibility);
        let json = serde_json::to_value(&err).unwrap();
        assert_eq!(json["kind"], "permission");
        assert_eq!(json["which"], "accessibility");
        assert_eq!(
            json["settingsUrl"],
            "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
        );
    }

    #[test]
    fn see_options_default_from_empty_json() {
        let opts: SeeOptions = serde_json::from_str("{}").unwrap();
        assert_eq!(opts.element_budget, 1500);
        assert_eq!(opts.max_depth, 30);
        assert!(!opts.descend_web_areas);
        assert_eq!(opts.scope, SeeScope::FrontmostWindow);
    }
}
