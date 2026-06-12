//! Agent-facing perception seam.
//!
//! The agent loop's `read`/`look` actions perceive through this trait —
//! injected like the capture engine — so the loop never touches Tauri state
//! and tests swap in a stub. The concrete implementation drives
//! `perception::commands::{read_impl, look_impl}` against the same on-disk
//! index the Tauri commands use.
//!
//! Errors are Strings: they become planner feedback in the step result,
//! never run failures (the webLookup convention).

use std::path::PathBuf;

/// Filters for a perception read, platform-neutral so the executor's
/// prepared actions never depend on macOS-only index types.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PerceptionReadQuery {
    pub text: Option<String>,
    /// Class letters ("B,T") or AX roles, comma-separated.
    pub roles: Option<String>,
    pub actionable_only: bool,
    pub limit: Option<u32>,
    pub snapshot_id: Option<String>,
}

/// What a read hands the planner: the serialized SNAP block plus enough
/// metadata for the (<220 char) step-result line.
#[derive(Clone, Debug)]
pub struct PerceptionReadFeed {
    pub snapshot_id: String,
    /// SNAP header, FOCUS, element lines, truncation footer.
    pub block: String,
    pub total: u64,
    pub returned: usize,
    /// The read transparently re-ran `see` (stale or missing snapshot).
    pub refreshed: bool,
}

/// What a look hands the planner: the set-of-marks frame as base64 PNG,
/// ready for the capture-attachment path.
#[derive(Clone, Debug)]
pub struct PerceptionLookFeed {
    pub snapshot_id: String,
    pub marked_png_b64: String,
    pub width: u32,
    pub height: u32,
    pub marks_drawn: usize,
    pub marks_capped: usize,
}

#[async_trait::async_trait(?Send)]
pub trait PerceptionTools {
    /// Query the element index (re-seeing stale snapshots — C3) and return
    /// the serialized block for the planner's goal context.
    async fn read(&self, query: &PerceptionReadQuery) -> Result<PerceptionReadFeed, String>;

    /// Set-of-marks frame for the current (or named) snapshot; box labels
    /// are element ids of that same snapshot.
    async fn look(&self, snapshot_id: Option<&str>) -> Result<PerceptionLookFeed, String>;
}

/// The real seam: `read_impl`/`look_impl` over the index under the app-data
/// directory (the same `base_dir` the `perception_*` Tauri commands use).
pub struct IndexPerceptionTools {
    base_dir: Option<PathBuf>,
}

impl IndexPerceptionTools {
    pub fn new(base_dir: Option<PathBuf>) -> Self {
        Self { base_dir }
    }
}

#[cfg(target_os = "macos")]
fn split_roles(roles: Option<&str>) -> Option<Vec<String>> {
    let roles: Vec<String> = roles?
        .split([',', ' '])
        .map(str::trim)
        .filter(|role| !role.is_empty())
        .map(str::to_string)
        .collect();
    (!roles.is_empty()).then_some(roles)
}

/// Planner-feedback rendering of a perception error. AxEmpty keeps the
/// literal `ax_empty` marker the prompt's look rule is taught to react to.
#[cfg(target_os = "macos")]
fn feedback(err: crate::perception::PerceptionError) -> String {
    use crate::perception::PerceptionError;
    match err {
        PerceptionError::AxEmpty(detail) => format!("ax_empty ({detail}); use look instead"),
        other => other.to_string(),
    }
}

#[async_trait::async_trait(?Send)]
impl PerceptionTools for IndexPerceptionTools {
    async fn read(&self, query: &PerceptionReadQuery) -> Result<PerceptionReadFeed, String> {
        #[cfg(target_os = "macos")]
        {
            let base = self
                .base_dir
                .clone()
                .ok_or_else(|| "perception index unavailable (no app-data directory)".to_string())?;
            let element_query = crate::perception::index::ElementQuery {
                snapshot_id: query.snapshot_id.clone(),
                roles: split_roles(query.roles.as_deref()),
                text: query.text.clone(),
                region: None,
                actionable_only: query.actionable_only,
                focused_only: false,
                limit: query
                    .limit
                    .unwrap_or(crate::perception::index::DEFAULT_READ_LIMIT)
                    .clamp(1, crate::perception::index::MAX_READ_LIMIT),
            };
            let result = tokio::task::spawn_blocking(move || {
                crate::perception::commands::read_impl(&base, &element_query)
            })
            .await
            .map_err(|err| format!("read join: {err}"))?
            .map_err(feedback)?;
            Ok(PerceptionReadFeed {
                snapshot_id: result.snapshot_id,
                block: result.text,
                total: result.total,
                returned: result.returned,
                refreshed: result.refreshed,
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = query;
            Err("the element index is not available on this platform".into())
        }
    }

    async fn look(&self, snapshot_id: Option<&str>) -> Result<PerceptionLookFeed, String> {
        #[cfg(target_os = "macos")]
        {
            use base64::Engine as _;
            let base = self
                .base_dir
                .clone()
                .ok_or_else(|| "perception index unavailable (no app-data directory)".to_string())?;
            // Same long edge as the agent's captureFrame path: readable UI
            // text within vision-model token budgets.
            let result = crate::perception::commands::look_impl(&base, snapshot_id, Some(1280))
                .await
                .map_err(feedback)?;
            let png = std::fs::read(&result.marked_png)
                .map_err(|err| format!("read marked frame: {err}"))?;
            Ok(PerceptionLookFeed {
                snapshot_id: result.snapshot_id,
                marked_png_b64: base64::engine::general_purpose::STANDARD.encode(png),
                width: result.width,
                height: result.height,
                marks_drawn: result.marks_drawn,
                marks_capped: result.marks_capped,
            })
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = snapshot_id;
            Err("the element index is not available on this platform".into())
        }
    }
}

/// Default for the test-only loop wrapper: declines every operation with
/// planner-readable feedback.
#[cfg(test)]
pub struct NoopPerceptionTools;

#[cfg(test)]
#[async_trait::async_trait(?Send)]
impl PerceptionTools for NoopPerceptionTools {
    async fn read(&self, _query: &PerceptionReadQuery) -> Result<PerceptionReadFeed, String> {
        Err("the element index is not available in this configuration".into())
    }

    async fn look(&self, _snapshot_id: Option<&str>) -> Result<PerceptionLookFeed, String> {
        Err("the element index is not available in this configuration".into())
    }
}
