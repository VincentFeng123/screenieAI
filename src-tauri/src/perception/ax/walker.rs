//! BFS AX-tree walker with hard budgets.
//!
//! Budgets are the walk's safety contract: element budget (default 1500),
//! depth cap (30), per-node child cap (200), a per-app messaging timeout
//! (0.25s, set in ffi) and a wall-clock cap (2s) that returns a *partial*
//! snapshot instead of failing. Web areas are recorded as boundary elements
//! and not descended by default — the `dom:` namespace owns those subtrees.

use super::classify;
use super::ffi;
use crate::perception::geometry::RectPt;
use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Hard ceiling on value text carried into the index.
const VALUE_TRUNCATE_CHARS: usize = 256;
/// Redaction marker for secure fields (C6); never the real value.
pub const REDACTED_VALUE: &str = "«redacted»";
/// When a node's child list overflows the cap, at most this many children
/// get the single-attribute role probe that decides actionable-first
/// ordering; the rest are dropped.
const CHILD_ROLE_PROBE_LIMIT: usize = 600;

/// The batched attribute set — one IPC round trip per element, children
/// included.
const BATCH_ATTRIBUTES: &[&str] = &[
    "AXRole",
    "AXSubrole",
    "AXTitle",
    "AXDescription",
    "AXValue",
    "AXEnabled",
    "AXFocused",
    "AXPosition",
    "AXSize",
    "AXHidden",
    "AXChildren",
];

#[derive(Clone, Copy, Debug)]
pub struct WalkConfig {
    pub element_budget: usize,
    pub max_depth: u8,
    pub child_cap: usize,
    pub descend_web_areas: bool,
    pub walk_timeout: Duration,
    pub messaging_timeout_secs: f32,
}

impl Default for WalkConfig {
    fn default() -> Self {
        Self {
            element_budget: 1500,
            max_depth: 30,
            child_cap: 200,
            descend_web_areas: false,
            walk_timeout: Duration::from_secs(2),
            messaging_timeout_secs: 0.25,
        }
    }
}

/// One walked element. `eid`, `fp`, and `parent_eid` are assigned by
/// `ids::assign` after the walk; the walker fills everything else.
#[derive(Clone, Debug, Default)]
pub struct AxElement {
    pub eid: String,
    pub fp: u64,
    pub role: String,
    pub subrole: Option<String>,
    pub title: Option<String>,
    pub descr: Option<String>,
    pub value: Option<String>,
    pub actions: Vec<String>,
    pub actionable: bool,
    pub enabled: bool,
    pub focused: bool,
    pub frame: RectPt,
    pub depth: u8,
    pub parent_eid: Option<String>,
    pub is_web_boundary: bool,
    /// Index of the parent within the walk's element vec; `ids::assign`
    /// turns it into `parent_eid`.
    pub parent_index: Option<usize>,
}

#[derive(Debug, Default)]
pub struct WalkOutcome {
    pub elements: Vec<AxElement>,
    /// Set when a budget or the wall clock cut the walk short.
    pub partial: bool,
    pub took_ms: u64,
    pub window_title: Option<String>,
    pub window_id: Option<u32>,
    /// The window root's frame in global points.
    pub window_frame: RectPt,
}

struct PendingNode {
    element: ffi::AxElementRef,
    depth: u8,
    parent_index: Option<usize>,
    secure_ancestor: bool,
}

/// Walk one window subtree breadth-first.
pub fn walk_window(window: ffi::AxElementRef, config: &WalkConfig) -> WalkOutcome {
    let started = Instant::now();
    let mut outcome = WalkOutcome {
        window_id: ffi::window_id(&window),
        ..WalkOutcome::default()
    };

    let mut queue: VecDeque<PendingNode> = VecDeque::new();
    queue.push_back(PendingNode {
        element: window,
        depth: 0,
        parent_index: None,
        secure_ancestor: false,
    });

    while let Some(node) = queue.pop_front() {
        if outcome.elements.len() >= config.element_budget {
            outcome.partial = true;
            break;
        }
        if started.elapsed() > config.walk_timeout {
            outcome.partial = true;
            break;
        }

        let child_fetch_cap = config.child_cap.saturating_add(CHILD_ROLE_PROBE_LIMIT);
        let Ok(attrs) = ffi::copy_attributes(&node.element, BATCH_ATTRIBUTES, child_fetch_cap)
        else {
            // Transport failure: the element (or its app) is gone — skip the
            // subtree, the walk itself continues.
            continue;
        };
        let mut read = BatchRead::from(attrs);

        if read.hidden {
            continue;
        }

        let frame = read.frame();
        let is_root = node.parent_index.is_none();
        if is_root {
            outcome.window_frame = frame.unwrap_or_default();
            outcome.window_title = read.title.clone();
        } else if let Some(frame) = frame {
            // Prune: degenerate frames and frames fully outside the window's
            // visible bounds cost budget without ever being actionable. The
            // intersect prune only applies when the root reported a real
            // frame — a zero window frame must not silently drop the tree.
            let outside_window =
                !outcome.window_frame.is_degenerate() && !frame.intersects(&outcome.window_frame);
            if frame.is_degenerate() || outside_window {
                continue;
            }
        }

        let role = read.role.clone().unwrap_or_default();
        let secure =
            node.secure_ancestor || classify::is_secure_field(&role, read.subrole.as_deref());
        let is_web_boundary = role == "AXWebArea" && !config.descend_web_areas;

        // AXStaticText is never actionable and dominates node counts; skipping
        // its action fetch saves one IPC round trip on the largest class.
        let actions = if role == "AXStaticText" {
            Vec::new()
        } else {
            ffi::action_names(&node.element)
        };
        let class = classify::classify(&role, &actions);
        let actionable = classify::is_actionable(class, &actions);

        let value = if secure {
            read.value.is_some().then(|| REDACTED_VALUE.to_string())
        } else {
            read.value
                .clone()
                .map(|v| truncate_chars(&v, VALUE_TRUNCATE_CHARS))
        };

        outcome.elements.push(AxElement {
            eid: String::new(),
            fp: 0,
            role: role.clone(),
            subrole: read.subrole.clone(),
            title: read.title.clone(),
            descr: read.descr.clone(),
            value,
            actions,
            actionable,
            enabled: read.enabled,
            focused: read.focused,
            frame: frame.unwrap_or_default(),
            depth: node.depth,
            parent_eid: None,
            is_web_boundary,
            parent_index: node.parent_index,
        });
        let index = outcome.elements.len() - 1;

        if node.depth >= config.max_depth || is_web_boundary {
            continue;
        }
        for child in capped_children(std::mem::take(&mut read.children), config.child_cap) {
            queue.push_back(PendingNode {
                element: child,
                depth: node.depth + 1,
                parent_index: Some(index),
                secure_ancestor: secure,
            });
        }
    }

    outcome.took_ms = started.elapsed().as_millis() as u64;
    outcome
}

/// Messaging timeout for window acquisition only: a just-launched app's AX
/// server can take most of a second to answer its first request.
const WINDOW_ACQUIRE_TIMEOUT_SECS: f32 = 1.0;

/// Walk an app's focused window (or its first window when no focus).
pub fn walk_app_focused_window(pid: i32, config: &WalkConfig) -> Option<WalkOutcome> {
    let app = ffi::app_element(pid, WINDOW_ACQUIRE_TIMEOUT_SECS)?;
    let window = ffi::focused_window(&app)?;
    ffi::set_messaging_timeout(&app, config.messaging_timeout_secs);
    Some(walk_window(window, config))
}

/// All windows of an app, walked under one shared element budget.
pub fn walk_app_windows(pid: i32, config: &WalkConfig) -> Vec<WalkOutcome> {
    let Some(app) = ffi::app_element(pid, WINDOW_ACQUIRE_TIMEOUT_SECS) else {
        return Vec::new();
    };
    ffi::set_messaging_timeout(&app, config.messaging_timeout_secs);
    let mut remaining = config.element_budget;
    let mut outcomes = Vec::new();
    for window in ffi::windows(&app) {
        if remaining == 0 {
            break;
        }
        let window_config = WalkConfig {
            element_budget: remaining,
            ..*config
        };
        let outcome = walk_window(window, &window_config);
        remaining = remaining.saturating_sub(outcome.elements.len());
        outcomes.push(outcome);
    }
    outcomes
}

/// Current focused-window frame + CGWindowID for a pid. The look path uses
/// this to shift element frames by the window's movement since the walk
/// (same window id), so marks stay aligned without a re-walk.
pub fn focused_window_frame(pid: i32) -> Option<(RectPt, Option<u32>)> {
    let app = ffi::app_element(pid, WINDOW_ACQUIRE_TIMEOUT_SECS)?;
    let window = ffi::focused_window(&app)?;
    let window_id = ffi::window_id(&window);
    let attrs = ffi::copy_attributes(&window, &["AXPosition", "AXSize"], 0).ok()?;
    let mut position = None;
    let mut size = None;
    for (index, attr) in attrs.into_iter().enumerate() {
        match (index, attr) {
            (0, ffi::AttrValue::Point(x, y)) => position = Some((x, y)),
            (1, ffi::AttrValue::Size(w, h)) => size = Some((w, h)),
            _ => {}
        }
    }
    let (x, y) = position?;
    let (w, h) = size?;
    Some((RectPt::new(x, y, w, h), window_id))
}

/// Children with the cap applied. When the list overflows, actionable-role
/// children keep their slots first (Electron tab strips and web remnants
/// produce thousands of inert siblings that would otherwise crowd out the
/// controls).
fn capped_children(mut children: Vec<ffi::AxElementRef>, cap: usize) -> Vec<ffi::AxElementRef> {
    if children.len() <= cap {
        return children;
    }
    let mut actionable_first: Vec<(usize, bool)> = children
        .iter()
        .enumerate()
        .map(|(index, child)| {
            let probe_actionable = index < CHILD_ROLE_PROBE_LIMIT
                && ffi::role_of(child)
                    .map(|role| classify::classify(&role, &[]) != classify::ElementClass::Static)
                    .unwrap_or(false);
            (index, probe_actionable)
        })
        .collect();
    // Stable partition: actionable roles first, original order within each
    // group.
    actionable_first.sort_by_key(|(index, actionable)| (!*actionable, *index));
    let keep: Vec<usize> = actionable_first
        .into_iter()
        .take(cap)
        .map(|(index, _)| index)
        .collect();
    let mut keep_sorted = keep;
    keep_sorted.sort_unstable();
    let mut kept = Vec::with_capacity(cap);
    for index in keep_sorted.into_iter().rev() {
        kept.push(children.swap_remove(index));
    }
    kept.reverse();
    kept
}

fn truncate_chars(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        value.to_string()
    } else {
        let truncated: String = value.chars().take(max_chars).collect();
        format!("{truncated}…")
    }
}

/// Decoded batch-fetch result in walk order.
#[derive(Default)]
struct BatchRead {
    role: Option<String>,
    subrole: Option<String>,
    title: Option<String>,
    descr: Option<String>,
    value: Option<String>,
    enabled: bool,
    focused: bool,
    position: Option<(f64, f64)>,
    size: Option<(f64, f64)>,
    hidden: bool,
    children: Vec<ffi::AxElementRef>,
}

impl BatchRead {
    fn from(attrs: Vec<ffi::AttrValue>) -> Self {
        let mut read = BatchRead {
            enabled: true,
            ..BatchRead::default()
        };
        for (index, attr) in attrs.into_iter().enumerate() {
            match (BATCH_ATTRIBUTES[index], attr) {
                ("AXRole", ffi::AttrValue::String(s)) => read.role = nonempty(s),
                ("AXSubrole", ffi::AttrValue::String(s)) => read.subrole = nonempty(s),
                ("AXTitle", ffi::AttrValue::String(s)) => read.title = nonempty(s),
                ("AXDescription", ffi::AttrValue::String(s)) => read.descr = nonempty(s),
                ("AXValue", ffi::AttrValue::String(s)) => read.value = nonempty(s),
                ("AXValue", ffi::AttrValue::Number(n)) => read.value = Some(format_number(n)),
                ("AXValue", ffi::AttrValue::Bool(b)) => {
                    read.value = Some(if b { "1" } else { "0" }.into())
                }
                ("AXEnabled", ffi::AttrValue::Bool(b)) => read.enabled = b,
                ("AXFocused", ffi::AttrValue::Bool(b)) => read.focused = b,
                ("AXPosition", ffi::AttrValue::Point(x, y)) => read.position = Some((x, y)),
                ("AXSize", ffi::AttrValue::Size(w, h)) => read.size = Some((w, h)),
                ("AXHidden", ffi::AttrValue::Bool(b)) => read.hidden = b,
                ("AXChildren", ffi::AttrValue::Elements(children)) => read.children = children,
                _ => {}
            }
        }
        read
    }

    /// Frame in global points; `None` when either half is missing (menu-bar
    /// descendants and other unrealized elements) — recorded as a zero rect,
    /// never pruned for it.
    fn frame(&self) -> Option<RectPt> {
        let (x, y) = self.position?;
        let (w, h) = self.size?;
        Some(RectPt::new(x, y, w, h))
    }
}

fn nonempty(s: String) -> Option<String> {
    if s.trim().is_empty() {
        None
    } else {
        Some(s)
    }
}

fn format_number(n: f64) -> String {
    if (n.fract()).abs() < f64::EPSILON && n.abs() < 1e15 {
        format!("{}", n as i64)
    } else {
        format!("{n}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncation_is_char_safe() {
        // Multi-byte codepoints must not be split (CJK from OCR answers is a
        // known hazard elsewhere in this codebase).
        let long = "日本語テキスト".repeat(60);
        let truncated = truncate_chars(&long, VALUE_TRUNCATE_CHARS);
        assert_eq!(truncated.chars().count(), VALUE_TRUNCATE_CHARS + 1); // + ellipsis
        assert!(truncated.ends_with('…'));
        assert_eq!(truncate_chars("short", 256), "short");
    }

    #[test]
    fn number_formatting() {
        assert_eq!(format_number(3.0), "3");
        assert_eq!(format_number(0.5), "0.5");
    }
}
