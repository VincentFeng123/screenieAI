//! Element identity: eid allocation, stable fingerprints, and the resolver.
//!
//! IDs are namespaced and snapshot-scoped (C4): `ax:B3` is only meaningful
//! paired with its snapshot id. The namespace travels inside the ID string
//! and [`resolve`] dispatches on it, so a `dom:` or `vis:` ref handed to
//! this resolver fails with a typed error instead of silently hitting the
//! wrong element — that kills the cross-namespace collision class of bugs.
//!
//! Fingerprints deliberately exclude `value` and `frame` so they survive
//! typing and window moves; they answer "is this the same Save button as
//! last turn" across snapshots and runs.

use crate::perception::geometry::RectPt;
use crate::perception::{PerceptionError, RESOLVE_NAMESPACES};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

/// Snapshot records kept resolvable in memory after they stop being a
/// window's latest (`allow_stale` resolution). The SQLite index keeps more
/// history; this bound only caps the hot path.
const CACHED_SNAPSHOT_RECORDS: usize = 40;

/// Platform-neutral element row: what the walker hands the index, the
/// resolver, the serializer, and the annotation renderer. The `ax:` walker
/// produces these today; `uia:` (Windows) and `dom:` producers slot in
/// without touching anything downstream.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ElementRow {
    pub eid: String,
    /// Stable fingerprint as i64-compatible u64 (SQLite stores i64).
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
    /// Class prefix letter (B/T/L/C/M/S/I/G/X) — denormalized for query
    /// filters and mark salience.
    pub class_prefix: char,
}

/// What [`resolve`] returns: everything the executor needs to act on an
/// element without re-walking.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ResolvedElement {
    pub eid: String,
    pub snapshot_id: String,
    pub role: String,
    pub title: Option<String>,
    pub value: Option<String>,
    /// Global screen points, top-left origin — CGEvent space (C5).
    pub frame: RectPt,
    /// Frame center, the default click point.
    pub click_point: (f64, f64),
    pub actionable: bool,
    pub enabled: bool,
    pub focused: bool,
    pub fp: u64,
    pub is_web_boundary: bool,
    pub actions: Vec<String>,
    /// True when the snapshot resolved against is no longer the window's
    /// latest (only possible with `allow_stale`).
    pub stale: bool,
}

/// Identity of the window a snapshot belongs to. CGWindowID when the
/// private API provided one, else pid + title (good enough to tell two
/// windows of one app apart for staleness tracking).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum WindowKey {
    WindowId(u32),
    PidTitle(i32, String),
}

/// One published snapshot, resolvable by id.
pub struct SnapshotRecord {
    pub snapshot_id: String,
    pub window_key: WindowKey,
    pub rows: Vec<ElementRow>,
    by_eid: HashMap<String, usize>,
}

impl SnapshotRecord {
    pub fn new(snapshot_id: String, window_key: WindowKey, rows: Vec<ElementRow>) -> Self {
        let by_eid = rows
            .iter()
            .enumerate()
            .map(|(index, row)| (row.eid.clone(), index))
            .collect();
        Self {
            snapshot_id,
            window_key,
            rows,
            by_eid,
        }
    }

    pub fn row(&self, eid: &str) -> Option<&ElementRow> {
        self.by_eid.get(eid).map(|index| &self.rows[*index])
    }
}

struct WindowState {
    latest: Arc<SnapshotRecord>,
    dirty: bool,
    /// AX change-counter value observed when the snapshot was taken; freshness
    /// checks compare against the live counter (C3).
    change_counter: Option<u64>,
}

/// Process-global registry of resolvable snapshots. The SQLite index is the
/// durable record; this is the hot path resolve()/read() consult so neither
/// ever blocks on disk.
#[derive(Default)]
pub struct SnapshotRegistry {
    state: Mutex<RegistryState>,
}

#[derive(Default)]
struct RegistryState {
    windows: HashMap<WindowKey, WindowState>,
    by_id: HashMap<String, Arc<SnapshotRecord>>,
    /// Publication order for eviction.
    order: VecDeque<String>,
}

static REGISTRY: OnceLock<SnapshotRegistry> = OnceLock::new();

pub fn registry() -> &'static SnapshotRegistry {
    REGISTRY.get_or_init(SnapshotRegistry::default)
}

/// The registry is process-global and tests run in parallel: every test in
/// any module that touches it must hold this guard.
#[cfg(test)]
pub(crate) fn registry_test_guard() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    LOCK.lock().unwrap_or_else(PoisonError::into_inner)
}

impl SnapshotRegistry {
    /// Publish a snapshot as its window's latest (clears the dirty flag).
    pub fn publish(
        &self,
        record: SnapshotRecord,
        change_counter: Option<u64>,
    ) -> Arc<SnapshotRecord> {
        let record = Arc::new(record);
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state
            .by_id
            .insert(record.snapshot_id.clone(), Arc::clone(&record));
        state.order.push_back(record.snapshot_id.clone());
        while state.order.len() > CACHED_SNAPSHOT_RECORDS {
            if let Some(evicted) = state.order.pop_front() {
                state.by_id.remove(&evicted);
            }
        }
        state.windows.insert(
            record.window_key.clone(),
            WindowState {
                latest: Arc::clone(&record),
                dirty: false,
                change_counter,
            },
        );
        record
    }

    /// The diff hook (C3): a significant change over a window marks its
    /// latest snapshot dirty; the next read transparently re-runs see.
    pub fn mark_dirty(&self, window_key: &WindowKey) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(window) = state.windows.get_mut(window_key) {
            window.dirty = true;
        }
    }

    /// Latest snapshot for a window if it is fresh: not dirty, and (when
    /// both sides have one) the AX change counter still matches.
    pub fn fresh_snapshot(
        &self,
        window_key: &WindowKey,
        live_change_counter: Option<u64>,
    ) -> Option<Arc<SnapshotRecord>> {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let window = state.windows.get(window_key)?;
        if window.dirty {
            return None;
        }
        if let (Some(seen), Some(live)) = (window.change_counter, live_change_counter) {
            if seen != live {
                return None;
            }
        }
        Some(Arc::clone(&window.latest))
    }

    pub fn snapshot_by_id(&self, snapshot_id: &str) -> Option<Arc<SnapshotRecord>> {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.by_id.get(snapshot_id).cloned()
    }

    /// Whether `snapshot_id` is the latest non-dirty snapshot of its window.
    fn is_latest(&self, record: &SnapshotRecord) -> bool {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state
            .windows
            .get(&record.window_key)
            .map(|window| !window.dirty && window.latest.snapshot_id == record.snapshot_id)
            .unwrap_or(false)
    }

    #[cfg(test)]
    fn clear(&self) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        *state = RegistryState::default();
    }
}

/// New snapshot id: namespaced, short, unique enough for the bounded cache
/// and the SQLite retention window.
pub fn new_snapshot_id(ns: &str) -> String {
    let uuid = uuid::Uuid::new_v4().simple().to_string();
    format!("{ns}:{}", &uuid[..8])
}

/// Resolve `(eid, snapshot_id)` to an element. Namespace-dispatched on the
/// eid's `ns:` prefix; this resolver owns `ax:` and returns typed errors
/// pointing `dom:`/`vis:` callers at the right channel. `StaleSnapshot`
/// when the snapshot is no longer its window's latest, unless `allow_stale`.
pub fn resolve(
    eid: &str,
    snapshot_id: &str,
    allow_stale: bool,
) -> Result<ResolvedElement, PerceptionError> {
    let ns = eid.split(':').next().unwrap_or("");
    if ns != "ax" {
        return Err(if RESOLVE_NAMESPACES.contains(&ns) {
            PerceptionError::WrongNamespace {
                eid: eid.to_string(),
                ns: ns.to_string(),
            }
        } else {
            PerceptionError::InvalidQuery(format!(
                "element id \"{eid}\" has no recognized namespace prefix"
            ))
        });
    }

    let registry = registry();
    let Some(record) = registry.snapshot_by_id(snapshot_id) else {
        return Err(PerceptionError::UnknownSnapshot(snapshot_id.to_string()));
    };
    let is_latest = registry.is_latest(&record);
    if !is_latest && !allow_stale {
        return Err(PerceptionError::StaleSnapshot {
            snapshot_id: snapshot_id.to_string(),
        });
    }
    let Some(row) = record.row(eid) else {
        return Err(PerceptionError::UnknownElement {
            eid: eid.to_string(),
            snapshot_id: snapshot_id.to_string(),
        });
    };
    Ok(ResolvedElement {
        eid: row.eid.clone(),
        snapshot_id: snapshot_id.to_string(),
        role: row.role.clone(),
        title: row.title.clone(),
        value: row.value.clone(),
        frame: row.frame,
        click_point: row.frame.center(),
        actionable: row.actionable,
        enabled: row.enabled,
        focused: row.focused,
        fp: row.fp,
        is_web_boundary: row.is_web_boundary,
        actions: row.actions.clone(),
        stale: !is_latest,
    })
}

// ---------------------------------------------------------------------------
// Fingerprints
// ---------------------------------------------------------------------------

/// Fixed seed: fingerprints must be stable across runs (per-app hint caches
/// and cross-run identity depend on it). Changing this invalidates every
/// stored fingerprint — bump only with a migration.
const FP_SEED: u64 = 0x53_43_52_4e_5f_46_50_31; // "SCRN_FP1"
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv1a_seeded(parts: &[&str]) -> u64 {
    let mut hash = FP_SEED;
    for part in parts {
        for byte in part.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        // Field separator: "ab"+"c" must differ from "a"+"bc".
        hash ^= 0x1f;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// `fp = hash(role | subrole | title-or-descr | ancestor role path |
/// same-role sibling index)` — value and frame excluded by design.
pub fn fingerprint(
    role: &str,
    subrole: Option<&str>,
    title_or_descr: Option<&str>,
    ancestor_role_path: &str,
    same_role_sibling_index: usize,
) -> u64 {
    fnv1a_seeded(&[
        role,
        subrole.unwrap_or(""),
        title_or_descr.unwrap_or(""),
        ancestor_role_path,
        &same_role_sibling_index.to_string(),
    ])
}

// ---------------------------------------------------------------------------
// eid assignment (ax: producer)
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
/// Turn a walk into indexed rows: per-snapshot per-class counters in tree
/// order, parent eids, and fingerprints.
pub fn assign_rows(outcome: &crate::perception::ax::WalkOutcome) -> Vec<ElementRow> {
    use crate::perception::ax::classify;

    let elements = &outcome.elements;
    let mut counters: HashMap<char, u32> = HashMap::new();
    let mut eids: Vec<String> = Vec::with_capacity(elements.len());
    let mut role_paths: Vec<String> = Vec::with_capacity(elements.len());
    // (parent_index, role) → count so far, for the same-role sibling index.
    let mut sibling_counts: HashMap<(Option<usize>, &str), usize> = HashMap::new();
    let mut rows = Vec::with_capacity(elements.len());

    for element in elements.iter() {
        let class = classify::classify(&element.role, &element.actions);
        let prefix = class.prefix();
        let counter = counters.entry(prefix).or_insert(0);
        *counter += 1;
        let eid = format!("ax:{prefix}{counter}");
        eids.push(eid.clone());

        // BFS guarantees parents precede children, so the parent's path is
        // always already built.
        let role_path = match element.parent_index {
            Some(parent) => format!("{}/{}", role_paths[parent], element.role),
            None => element.role.clone(),
        };
        let ancestor_path = match element.parent_index {
            Some(parent) => role_paths[parent].as_str(),
            None => "",
        };

        let sibling_slot = sibling_counts
            .entry((element.parent_index, element.role.as_str()))
            .or_insert(0);
        let sibling_index = *sibling_slot;
        *sibling_slot += 1;

        let fp = fingerprint(
            &element.role,
            element.subrole.as_deref(),
            element.title.as_deref().or(element.descr.as_deref()),
            ancestor_path,
            sibling_index,
        );
        role_paths.push(role_path);

        rows.push(ElementRow {
            eid,
            fp,
            role: element.role.clone(),
            subrole: element.subrole.clone(),
            title: element.title.clone(),
            descr: element.descr.clone(),
            value: element.value.clone(),
            actions: element.actions.clone(),
            actionable: element.actionable,
            enabled: element.enabled,
            focused: element.focused,
            frame: element.frame,
            depth: element.depth,
            parent_eid: element.parent_index.map(|parent| eids[parent].clone()),
            is_web_boundary: element.is_web_boundary,
            class_prefix: prefix,
        });
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(eid: &str) -> ElementRow {
        ElementRow {
            eid: eid.to_string(),
            fp: 1,
            role: "AXButton".into(),
            subrole: None,
            title: Some("Save".into()),
            descr: None,
            value: None,
            actions: vec!["AXPress".into()],
            actionable: true,
            enabled: true,
            focused: false,
            frame: RectPt::new(10.0, 10.0, 40.0, 20.0),
            depth: 1,
            parent_eid: None,
            is_web_boundary: false,
            class_prefix: 'B',
        }
    }

    #[test]
    fn fingerprint_stability_contract() {
        let base = fingerprint("AXButton", None, Some("Save"), "AXWindow/AXGroup", 0);
        // Same UI → same fp (deterministic across calls/runs: fixed seed).
        assert_eq!(
            base,
            fingerprint("AXButton", None, Some("Save"), "AXWindow/AXGroup", 0)
        );
        // Value/frame are not inputs, so "value change → same fp" holds by
        // construction. Title change → different fp:
        assert_ne!(
            base,
            fingerprint("AXButton", None, Some("Cancel"), "AXWindow/AXGroup", 0)
        );
        // Ancestor path and sibling index discriminate:
        assert_ne!(
            base,
            fingerprint("AXButton", None, Some("Save"), "AXWindow/AXToolbar", 0)
        );
        assert_ne!(
            base,
            fingerprint("AXButton", None, Some("Save"), "AXWindow/AXGroup", 1)
        );
        // Field boundaries matter: ("AXB","utton") ≠ ("AXBu","tton").
        assert_ne!(
            fingerprint("AXB", Some("utton"), None, "", 0),
            fingerprint("AXBu", Some("tton"), None, "", 0)
        );
    }

    #[test]
    fn resolver_rejects_wrong_namespace_and_unknown() {
        let _guard = registry_test_guard();
        registry().clear();
        match resolve("dom:L3", "ax:abcd1234", false) {
            Err(PerceptionError::WrongNamespace { ns, .. }) => assert_eq!(ns, "dom"),
            other => panic!("expected WrongNamespace, got {other:?}"),
        }
        match resolve("vis:G1", "ax:abcd1234", false) {
            Err(PerceptionError::WrongNamespace { ns, .. }) => assert_eq!(ns, "vis"),
            other => panic!("expected WrongNamespace, got {other:?}"),
        }
        assert!(matches!(
            resolve("B3", "ax:abcd1234", false),
            Err(PerceptionError::InvalidQuery(_))
        ));
        assert!(matches!(
            resolve("ax:B3", "ax:missing00", false),
            Err(PerceptionError::UnknownSnapshot(_))
        ));
    }

    #[test]
    fn resolver_staleness_contract() {
        let _guard = registry_test_guard();
        registry().clear();
        let window = WindowKey::WindowId(7);
        let first = SnapshotRecord::new("ax:firstsnap".into(), window.clone(), vec![row("ax:B1")]);
        registry().publish(first, Some(10));

        // Latest + clean resolves.
        let resolved = resolve("ax:B1", "ax:firstsnap", false).unwrap();
        assert!(!resolved.stale);
        assert_eq!(resolved.click_point, (30.0, 20.0));

        // Unknown eid in a known snapshot.
        assert!(matches!(
            resolve("ax:B2", "ax:firstsnap", false),
            Err(PerceptionError::UnknownElement { .. })
        ));

        // A newer snapshot supersedes: old id is stale unless allowed.
        let second =
            SnapshotRecord::new("ax:secondsnap".into(), window.clone(), vec![row("ax:B1")]);
        registry().publish(second, Some(11));
        assert!(matches!(
            resolve("ax:B1", "ax:firstsnap", false),
            Err(PerceptionError::StaleSnapshot { .. })
        ));
        let stale = resolve("ax:B1", "ax:firstsnap", true).unwrap();
        assert!(stale.stale);

        // Dirty latest is also stale (the window changed since the walk).
        registry().mark_dirty(&window);
        assert!(matches!(
            resolve("ax:B1", "ax:secondsnap", false),
            Err(PerceptionError::StaleSnapshot { .. })
        ));
        assert!(registry().fresh_snapshot(&window, Some(11)).is_none());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn assign_rows_counts_per_class_in_tree_order() {
        use crate::perception::ax::walker::AxElement;
        use crate::perception::ax::WalkOutcome;

        let element = |role: &str, title: &str, parent: Option<usize>, actions: &[&str]| {
            // Mirror the walker: it computes class + actionable at walk time
            // and assign_rows carries them through.
            let actions: Vec<String> = actions.iter().map(|a| a.to_string()).collect();
            let class = crate::perception::ax::classify::classify(role, &actions);
            AxElement {
                role: role.into(),
                title: (!title.is_empty()).then(|| title.to_string()),
                actionable: crate::perception::ax::classify::is_actionable(class, &actions),
                actions,
                enabled: true,
                frame: RectPt::new(0.0, 0.0, 10.0, 10.0),
                parent_index: parent,
                ..AxElement::default()
            }
        };
        let outcome = WalkOutcome {
            elements: vec![
                element("AXWindow", "Doc", None, &["AXRaise"]),
                element("AXButton", "Save", Some(0), &["AXPress"]),
                element("AXButton", "Cancel", Some(0), &["AXPress"]),
                element("AXTextField", "Name", Some(0), &[]),
                element("AXGroup", "", Some(0), &["AXPress"]),
                element("AXStaticText", "hint", Some(4), &[]),
            ],
            ..WalkOutcome::default()
        };
        let rows = assign_rows(&outcome);
        let eids: Vec<&str> = rows.iter().map(|row| row.eid.as_str()).collect();
        // Window: Static (X); buttons count per-class in tree order; the
        // pressable group is Generic.
        assert_eq!(eids, ["ax:X1", "ax:B1", "ax:B2", "ax:T1", "ax:G1", "ax:X2"]);
        assert_eq!(rows[1].parent_eid.as_deref(), Some("ax:X1"));
        assert_eq!(rows[5].parent_eid.as_deref(), Some("ax:G1"));
        // Same-role same-title siblings still get distinct fingerprints via
        // the sibling index; differently-titled ones differ by title alone.
        assert_ne!(rows[1].fp, rows[2].fp);
        assert_eq!(rows[1].class_prefix, 'B');
        assert!(rows[1].actionable);
        assert!(!rows[5].actionable);
    }

    #[test]
    fn freshness_uses_change_counter() {
        let _guard = registry_test_guard();
        registry().clear();
        let window = WindowKey::WindowId(9);
        let record = SnapshotRecord::new("ax:counter01".into(), window.clone(), vec![row("ax:B1")]);
        registry().publish(record, Some(5));
        assert!(registry().fresh_snapshot(&window, Some(5)).is_some());
        // Counter moved → not fresh (read must re-see).
        assert!(registry().fresh_snapshot(&window, Some(6)).is_none());
        // Counter unavailable on either side → counter check is skipped.
        assert!(registry().fresh_snapshot(&window, None).is_some());
    }
}
