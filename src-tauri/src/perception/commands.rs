//! Command surface and orchestration for `see` → index → `read` / `look`.
//!
//! Freshness contract (C3): `read` and `look` with no snapshot id resolve
//! the frontmost window's latest snapshot; if none exists, the window is
//! marked dirty, or the AX change counter moved since the walk, they
//! transparently run `see` first. The agent never reasons about freshness.
//!
//! The Tauri commands are thin wrappers over the `*_impl` functions so the
//! agent loop (and see_probe) can drive the same paths without a window.

use crate::perception::index::{self, ElementQuery, SnapshotMeta};
use crate::perception::serialize::{self, SnapHeader};
use crate::perception::{PerceptionError, SeeOptions};
use serde::Serialize;
use std::path::Path;

#[cfg(target_os = "macos")]
use crate::perception::ax::walker;
#[cfg(target_os = "macos")]
use crate::perception::ids::{self, SnapshotRecord, WindowKey};
#[cfg(target_os = "macos")]
use crate::perception::{annotate, PermissionKind, RectPt};

/// Windows allowed to drive perception. The overlay is deliberately absent —
/// it runs at reduced privilege.
const ALLOWED_WINDOWS: &[&str] = &["main", "chat", "quick_tooltip"];

/// `read_sql` ships enabled (read-only connection + query_only + row caps);
/// this env var is the off switch the spec's flag-gating asks for.
const READ_SQL_ENV: &str = "SCREENIE_PERCEPTION_SQL";

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SeeResult {
    pub snapshot_id: String,
    pub app_bundle: Option<String>,
    pub app_name: Option<String>,
    pub app_pid: Option<i32>,
    pub window_title: Option<String>,
    pub window_id: Option<u32>,
    pub element_count: u32,
    pub partial: bool,
    pub took_ms: u64,
    /// The serialized SNAP block (top-K elements + truncation footer).
    pub top_elements: String,
    pub raw_png: Option<String>,
    pub marked_png: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReadResult {
    pub snapshot_id: String,
    /// The serialized block: SNAP header, FOCUS, element lines, footer.
    pub text: String,
    pub total: u64,
    pub returned: usize,
    /// Set when this read transparently re-ran `see`.
    pub refreshed: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LookResult {
    pub snapshot_id: String,
    pub marked_png: String,
    pub raw_png: String,
    pub width: u32,
    pub height: u32,
    pub marks_drawn: usize,
    pub marks_capped: usize,
    /// eid → label pairs for every drawn mark.
    pub legend: Vec<crate::perception::annotate::LegendEntry>,
}

// ---------------------------------------------------------------------------
// Orchestration (macOS)
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
fn accessibility_preflight() -> Result<(), PerceptionError> {
    if crate::perception::ax::ffi::process_trusted() {
        Ok(())
    } else {
        // Trigger the system prompt on the first denial so the user gets the
        // native flow; callers still receive the structured error with the
        // settings deep link.
        let _ = crate::perception::ax::ffi::process_trusted_with_prompt();
        Err(PerceptionError::Permission(PermissionKind::Accessibility))
    }
}

#[cfg(target_os = "macos")]
fn window_key_for(pid: i32, window_id: Option<u32>, title: Option<&str>) -> WindowKey {
    match window_id {
        Some(id) => WindowKey::WindowId(id),
        None => WindowKey::PidTitle(pid, title.unwrap_or_default().to_string()),
    }
}

/// Walk per `opts`, assign ids, publish to the registry, ingest into the
/// index, and serialize the top-K block.
#[cfg(target_os = "macos")]
pub fn see_impl(base_dir: &Path, opts: &SeeOptions) -> Result<SeeResult, PerceptionError> {
    use crate::perception::SeeScope;

    accessibility_preflight()?;
    let index = index::init(base_dir)?;

    let frontmost = crate::agent::frontmost_application_info().ok().flatten();
    let (pid, app_bundle, app_name) = match &opts.scope {
        SeeScope::App { pid } => (*pid, None, None),
        SeeScope::FrontmostWindow | SeeScope::Screen => {
            let app = frontmost.ok_or(PerceptionError::NoTarget)?;
            let pid = app.pid.ok_or(PerceptionError::NoTarget)?;
            (pid, app.bundle_id.clone(), Some(app.name.clone()))
        }
    };

    let config = walker::WalkConfig {
        element_budget: opts.element_budget.clamp(50, 5000),
        max_depth: opts.max_depth.clamp(2, 60),
        descend_web_areas: opts.descend_web_areas,
        ..walker::WalkConfig::default()
    };
    // Screen scope walks every window of the frontmost app under the shared
    // budget (each as its own snapshot) and reports on the focused one;
    // walking every app on screen is deferred to the multi-app milestone.
    let outcomes = match &opts.scope {
        SeeScope::Screen => walker::walk_app_windows(pid, &config),
        _ => walker::walk_app_focused_window(pid, &config)
            .map(|outcome| vec![outcome])
            .unwrap_or_default(),
    };
    if outcomes.is_empty() {
        // The app answered AX with nothing walkable — degrade to ax_empty so
        // the caller's ladder falls through to vision (Phase 6 contract).
        return Err(PerceptionError::AxEmpty(format!(
            "no walkable AX window for pid {pid}"
        )));
    }

    let change_counter = crate::agent::perception_change_counter();
    let mut primary: Option<SeeResult> = None;
    for outcome in outcomes {
        let rows = ids::assign_rows(&outcome);
        let snapshot_id = ids::new_snapshot_id("ax");
        let window_key = window_key_for(pid, outcome.window_id, outcome.window_title.as_deref());

        let meta = SnapshotMeta {
            id: snapshot_id.clone(),
            ns: "ax".into(),
            created_at_ms: index::store::now_ms(),
            app_bundle: app_bundle.clone(),
            app_pid: Some(pid),
            window_id: outcome.window_id,
            window_title: outcome.window_title.clone(),
            display_id: None,
            // Points-per-point placeholder; the real px/pt scale is measured
            // at look (capture) time and written back onto the row.
            scale: 1.0,
            cap_rect: Some(outcome.window_frame),
            raw_png: None,
            marked_png: None,
            element_count: rows.len() as u32,
            took_ms: outcome.took_ms,
            partial: outcome.partial,
            diff_region: None,
        };
        index.store.ingest(meta, rows.clone())?;
        ids::registry().publish(
            SnapshotRecord::new(snapshot_id.clone(), window_key, rows.clone()),
            change_counter,
            Some(pid),
        );

        let ranked = serialize::rank_rows(&rows);
        let top: Vec<_> = ranked
            .into_iter()
            .take(index::DEFAULT_READ_LIMIT as usize)
            .cloned()
            .collect();
        let header = SnapHeader {
            snapshot_id: snapshot_id.clone(),
            app: app_bundle.clone().or_else(|| app_name.clone()),
            window_title: outcome.window_title.clone(),
            frame: outcome.window_frame,
            scale: 1.0,
            total: rows.len(),
            took_ms: outcome.took_ms,
        };
        let result = SeeResult {
            snapshot_id,
            app_bundle: app_bundle.clone(),
            app_name: app_name.clone(),
            app_pid: Some(pid),
            window_title: outcome.window_title.clone(),
            window_id: outcome.window_id,
            element_count: header.total as u32,
            partial: outcome.partial,
            took_ms: outcome.took_ms,
            top_elements: serialize::serialize_block(&header, &top),
            raw_png: None,
            marked_png: None,
        };
        // Screen scope: the focused window's walk comes first from
        // AXFocusedWindow ordering isn't guaranteed — prefer the outcome
        // with a focused element, else the first.
        let has_focus = result.top_elements.contains("\nFOCUS ");
        match &primary {
            None => primary = Some(result),
            Some(existing) if has_focus && !existing.top_elements.contains("\nFOCUS ") => {
                primary = Some(result)
            }
            _ => {}
        }
    }
    primary.ok_or(PerceptionError::NoTarget)
}

/// Resolve which snapshot a read/look targets, re-seeing when stale (C3).
/// Returns (snapshot_id, refreshed).
#[cfg(target_os = "macos")]
fn ensure_fresh_snapshot(
    base_dir: &Path,
    requested: Option<&str>,
    opts: &SeeOptions,
) -> Result<(String, bool), PerceptionError> {
    let registry = ids::registry();
    let live_counter = crate::agent::perception_change_counter();

    if let Some(snapshot_id) = requested {
        if let Some(record) = registry.snapshot_by_id(snapshot_id) {
            let is_latest =
                registry.latest_snapshot_id(&record.window_key).as_deref() == Some(snapshot_id);
            if !is_latest {
                // An explicitly-named older snapshot is a historical read.
                return Ok((snapshot_id.to_string(), false));
            }
            if registry
                .fresh_snapshot(&record.window_key, live_counter)
                .is_some()
            {
                return Ok((snapshot_id.to_string(), false));
            }
            // The window's latest, but dirty / changed since the walk:
            // transparently re-see.
        } else {
            // Not in the hot registry: serve from the index if it exists
            // (historical read), else it's unknown.
            let index = index::init(base_dir)?;
            return match index.reader.snapshot_header(snapshot_id)? {
                Some(_) => Ok((snapshot_id.to_string(), false)),
                None => Err(PerceptionError::UnknownSnapshot(snapshot_id.to_string())),
            };
        }
    } else if let Some(pid) = crate::agent::frontmost_application_info()
        .ok()
        .flatten()
        .and_then(|app| app.pid)
    {
        if let Some(fresh) = registry.fresh_snapshot_for_pid(pid, live_counter) {
            return Ok((fresh.snapshot_id.clone(), false));
        }
    }

    let seen = see_impl(base_dir, opts)?;
    Ok((seen.snapshot_id, true))
}

#[cfg(target_os = "macos")]
pub fn read_impl(base_dir: &Path, query: &ElementQuery) -> Result<ReadResult, PerceptionError> {
    let (snapshot_id, refreshed) = ensure_fresh_snapshot(
        base_dir,
        query.snapshot_id.as_deref(),
        &SeeOptions::default(),
    )?;
    let index = index::init(base_dir)?;
    let header_row = index
        .reader
        .snapshot_header(&snapshot_id)?
        .ok_or_else(|| PerceptionError::UnknownSnapshot(snapshot_id.clone()))?;
    let result = index.reader.query(&snapshot_id, query)?;
    let header = SnapHeader {
        snapshot_id: snapshot_id.clone(),
        app: header_row.app_bundle.clone(),
        window_title: header_row.window_title.clone(),
        frame: header_row.cap_rect.unwrap_or_default(),
        scale: header_row.scale,
        total: result.total as usize,
        took_ms: header_row.took_ms,
    };
    Ok(ReadResult {
        snapshot_id,
        text: serialize::serialize_block(&header, &result.rows),
        total: result.total,
        returned: result.rows.len(),
        refreshed,
    })
}

#[cfg(target_os = "macos")]
pub async fn look_impl(
    base_dir: &Path,
    requested_snapshot: Option<&str>,
    long_edge: Option<u32>,
) -> Result<LookResult, PerceptionError> {
    use crate::capture::engine::{self, CaptureTarget, FrameOpts};
    use base64::Engine as _;

    if !crate::capture::engine_macos::screen_capture_access_granted() {
        return Err(PerceptionError::Permission(PermissionKind::ScreenRecording));
    }

    let base = base_dir.to_path_buf();
    let requested = requested_snapshot.map(|s| s.to_string());
    let (snapshot_id, _) = tokio::task::spawn_blocking({
        let base = base.clone();
        move || ensure_fresh_snapshot(&base, requested.as_deref(), &SeeOptions::default())
    })
    .await
    .map_err(|err| PerceptionError::Index(format!("look join: {err}")))??;

    let index = index::init(&base)?;
    let header = index
        .reader
        .snapshot_header(&snapshot_id)?
        .ok_or_else(|| PerceptionError::UnknownSnapshot(snapshot_id.clone()))?;
    let mut rows = match ids::registry().snapshot_by_id(&snapshot_id) {
        Some(record) => record.rows.clone(),
        None => index.reader.all_rows(&snapshot_id)?,
    };
    let mut cap_rect = header
        .cap_rect
        .ok_or_else(|| PerceptionError::Capture("snapshot has no window frame".into()))?;

    // Drift correction: if the same window moved since the walk, shift all
    // frames by the delta instead of forcing a re-walk.
    if let (Some(pid), Some(snapshot_window)) = (header.app_pid, header.window_id) {
        let current = tokio::task::spawn_blocking(move || walker::focused_window_frame(pid))
            .await
            .ok()
            .flatten();
        if let Some((current_frame, Some(current_id))) = current {
            if current_id == snapshot_window && current_frame != cap_rect {
                let (dx, dy) = (current_frame.x - cap_rect.x, current_frame.y - cap_rect.y);
                for row in &mut rows {
                    row.frame.x += dx;
                    row.frame.y += dy;
                }
                cap_rect = RectPt::new(current_frame.x, current_frame.y, cap_rect.w, cap_rect.h);
            }
        }
    }

    let target = match header.window_id {
        Some(id) => CaptureTarget::Window { id },
        None => CaptureTarget::Region {
            x: cap_rect.x,
            y: cap_rect.y,
            w: cap_rect.w,
            h: cap_rect.h,
        },
    };
    let frame = engine::capture_frame(
        target,
        FrameOpts {
            max_dimension: 4096,
            persist: false,
            exclude_self: true,
            show_cursor: false,
        },
        None,
    )
    .await
    .map_err(|err| PerceptionError::Capture(err.to_string()))?;
    if frame.blank {
        // The TCC all-black placeholder — the ground truth for a broken
        // Screen Recording grant even when CGPreflight says yes.
        return Err(PerceptionError::Permission(PermissionKind::ScreenRecording));
    }
    let raw_bytes = base64::engine::general_purpose::STANDARD
        .decode(&frame.png_base64)
        .map_err(|err| PerceptionError::Capture(format!("decode capture: {err}")))?;
    let scale = if cap_rect.w > 0.0 {
        f64::from(frame.width) / cap_rect.w
    } else {
        1.0
    };

    let annotation = {
        let rows = rows.clone();
        let raw = raw_bytes.clone();
        tokio::task::spawn_blocking(move || {
            annotate::annotate(&raw, &rows, cap_rect, scale, long_edge)
        })
        .await
        .map_err(|err| PerceptionError::Index(format!("annotate join: {err}")))??
    };

    let file_stem = snapshot_id.replace(':', "-");
    let raw_path = index
        .store
        .frames_dir()
        .join(format!("{file_stem}-raw.png"));
    let marked_path = index
        .store
        .frames_dir()
        .join(format!("{file_stem}-marked.png"));
    std::fs::write(&raw_path, &raw_bytes)
        .map_err(|err| PerceptionError::Index(format!("write raw png: {err}")))?;
    std::fs::write(&marked_path, &annotation.png)
        .map_err(|err| PerceptionError::Index(format!("write marked png: {err}")))?;
    index.store.set_png_paths(
        &snapshot_id,
        Some(raw_path.to_string_lossy().into_owned()),
        Some(marked_path.to_string_lossy().into_owned()),
        Some(scale),
    )?;

    Ok(LookResult {
        snapshot_id,
        marked_png: marked_path.to_string_lossy().into_owned(),
        raw_png: raw_path.to_string_lossy().into_owned(),
        width: annotation.width,
        height: annotation.height,
        marks_drawn: annotation.marks_drawn,
        marks_capped: annotation.marks_capped,
        legend: annotation.legend,
    })
}

// ---------------------------------------------------------------------------
// Tauri commands
// ---------------------------------------------------------------------------

fn guard_window(window: &tauri::WebviewWindow) -> Result<(), PerceptionError> {
    if ALLOWED_WINDOWS.contains(&window.label()) {
        Ok(())
    } else {
        Err(PerceptionError::InvalidQuery(format!(
            "perception commands are not available to the {} window",
            window.label()
        )))
    }
}

fn data_dir(app: &tauri::AppHandle) -> Result<std::path::PathBuf, PerceptionError> {
    use tauri::Manager;
    app.path()
        .app_data_dir()
        .map_err(|err| PerceptionError::Index(format!("app data dir: {err}")))
}

#[tauri::command]
pub async fn perception_see(
    window: tauri::WebviewWindow,
    app: tauri::AppHandle,
    opts: Option<SeeOptions>,
) -> Result<SeeResult, PerceptionError> {
    guard_window(&window)?;
    let base = data_dir(&app)?;
    #[cfg(target_os = "macos")]
    {
        let opts = opts.unwrap_or_default();
        tauri::async_runtime::spawn_blocking(move || see_impl(&base, &opts))
            .await
            .map_err(|err| PerceptionError::Index(format!("see join: {err}")))?
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (base, opts);
        Err(PerceptionError::Unsupported)
    }
}

#[tauri::command]
pub async fn perception_read(
    window: tauri::WebviewWindow,
    app: tauri::AppHandle,
    query: Option<ElementQuery>,
) -> Result<ReadResult, PerceptionError> {
    guard_window(&window)?;
    let base = data_dir(&app)?;
    #[cfg(target_os = "macos")]
    {
        let query = query.unwrap_or_default();
        tauri::async_runtime::spawn_blocking(move || read_impl(&base, &query))
            .await
            .map_err(|err| PerceptionError::Index(format!("read join: {err}")))?
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (base, query);
        Err(PerceptionError::Unsupported)
    }
}

#[tauri::command]
pub async fn perception_read_sql(
    window: tauri::WebviewWindow,
    app: tauri::AppHandle,
    sql: String,
) -> Result<index::SqlRows, PerceptionError> {
    guard_window(&window)?;
    if matches!(
        std::env::var(READ_SQL_ENV).ok().as_deref(),
        Some("0") | Some("false") | Some("off")
    ) {
        return Err(PerceptionError::InvalidQuery(format!(
            "read_sql is disabled ({READ_SQL_ENV}=0)"
        )));
    }
    let base = data_dir(&app)?;
    let index = index::init(&base)?;
    tauri::async_runtime::spawn_blocking(move || index.reader.read_sql(&sql))
        .await
        .map_err(|err| PerceptionError::Index(format!("read_sql join: {err}")))?
}

#[tauri::command]
pub async fn perception_look(
    window: tauri::WebviewWindow,
    app: tauri::AppHandle,
    snapshot_id: Option<String>,
    long_edge: Option<u32>,
) -> Result<LookResult, PerceptionError> {
    guard_window(&window)?;
    let base = data_dir(&app)?;
    #[cfg(target_os = "macos")]
    {
        look_impl(&base, snapshot_id.as_deref(), long_edge).await
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (base, snapshot_id, long_edge);
        Err(PerceptionError::Unsupported)
    }
}

#[tauri::command]
pub async fn perception_resolve(
    window: tauri::WebviewWindow,
    eid: String,
    snapshot_id: String,
    allow_stale: Option<bool>,
) -> Result<crate::perception::ResolvedElement, PerceptionError> {
    guard_window(&window)?;
    crate::perception::ids::resolve(&eid, &snapshot_id, allow_stale.unwrap_or(false))
}
