//! Per-app cache of verified UI navigation paths, so a path the agent had to
//! hunt for (menu scan or web lookup) is reused on later runs with zero
//! searching. One JSON file per app under `<app_data>/agent/hints/`.
//!
//! Safety properties:
//! - Structured fields only — no freeform prose field exists, so a hostile
//!   web page can at worst plant a wrong menu title, which fails harmlessly
//!   through the menu action's not-found feedback.
//! - Hints are written exclusively from ground truth the executor observed
//!   (an AX-resolved menu path or a key combo that verified as progress),
//!   never from parsed web text. Callers enforce that contract.

use super::search::{normalize_for_match, score_match};
use super::types::FocusedApp;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

const HINTS_FILE_VERSION: u32 = 1;
const MAX_HINTS_PER_APP: usize = 20;
const MAX_HINT_FEATURE_CHARS: usize = 64;
const MAX_HINT_MENU_COMPONENTS: usize = 4;
const MAX_HINT_TEXT_CHARS: usize = 80;
/// Hints older than ~6 months are ignored on read — app UIs move.
const HINT_STALE_MS: u64 = 183 * 24 * 60 * 60 * 1000;

pub(crate) fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as u64)
        .unwrap_or(0)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) enum UiHintKind {
    Menu,
    Shortcut,
    Settings,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct UiHint {
    /// Normalized feature phrase the hint answers, e.g. "export pdf".
    pub feature: String,
    pub kind: UiHintKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub menu_path: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub combo: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub settings_pane: Option<String>,
    pub verified_at_ms: u64,
    /// "menuScan" or "webLookup" — provenance for debugging, never rendered
    /// into prompts.
    pub source: String,
}

impl UiHint {
    /// Render for a findUi result or the run-start known-paths line. The
    /// output is template-driven on purpose: hint content never reaches the
    /// model as free text.
    pub(crate) fn render(&self, now_ms: u64) -> Option<String> {
        let what = match self.kind {
            UiHintKind::Menu => format!("menu {}", self.menu_path.as_ref()?.join(" > ")),
            UiHintKind::Shortcut => format!("shortcut {}", self.combo.as_deref()?),
            UiHintKind::Settings => format!("settings {}", self.settings_pane.as_deref()?),
        };
        Some(format!("{what} ({})", verified_age(self.verified_at_ms, now_ms)))
    }

    fn is_fresh(&self, now_ms: u64) -> bool {
        now_ms.saturating_sub(self.verified_at_ms) < HINT_STALE_MS
    }

    fn is_valid(&self) -> bool {
        let feature_ok = !self.feature.trim().is_empty()
            && self.feature.chars().count() <= MAX_HINT_FEATURE_CHARS;
        let payload_ok = match self.kind {
            UiHintKind::Menu => self.menu_path.as_ref().is_some_and(|path| {
                !path.is_empty()
                    && path.len() <= MAX_HINT_MENU_COMPONENTS
                    && path.iter().all(|title| {
                        !title.trim().is_empty() && title.chars().count() <= MAX_HINT_TEXT_CHARS
                    })
            }),
            UiHintKind::Shortcut => self.combo.as_ref().is_some_and(|combo| {
                !combo.trim().is_empty() && combo.chars().count() <= MAX_HINT_TEXT_CHARS
            }),
            UiHintKind::Settings => self.settings_pane.as_ref().is_some_and(|pane| {
                !pane.trim().is_empty() && pane.chars().count() <= MAX_HINT_TEXT_CHARS
            }),
        };
        feature_ok && payload_ok
    }
}

fn verified_age(verified_at_ms: u64, now_ms: u64) -> String {
    let days = now_ms.saturating_sub(verified_at_ms) / (24 * 60 * 60 * 1000);
    if days == 0 {
        "verified today".into()
    } else {
        format!("verified {days}d ago")
    }
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct HintFile {
    version: u32,
    hints: Vec<UiHint>,
}

/// Per-app hint persistence. `dir: None` (tests, headless) makes every
/// operation a cheap no-op. Reads are cached per app key; only this process
/// writes the files.
pub(crate) struct HintStore {
    dir: Option<PathBuf>,
    cache: RefCell<HashMap<String, Vec<UiHint>>>,
}

impl HintStore {
    pub(crate) fn new(dir: Option<PathBuf>) -> Self {
        Self {
            dir,
            cache: RefCell::new(HashMap::new()),
        }
    }

    /// File key for an app: sanitized bundle id, falling back to the app
    /// name. `None` when neither identifies the app.
    pub(crate) fn key_for(app: &FocusedApp) -> Option<String> {
        let raw = app
            .bundle_id
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| app.name.trim());
        let sanitized: String = raw
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        (!sanitized.trim_matches('_').is_empty()).then_some(sanitized)
    }

    fn file_path(&self, key: &str) -> Option<PathBuf> {
        Some(self.dir.as_ref()?.join(format!("{key}.json")))
    }

    fn load(&self, key: &str) -> Vec<UiHint> {
        if let Some(cached) = self.cache.borrow().get(key) {
            return cached.clone();
        }
        let hints = self
            .file_path(key)
            .and_then(|path| fs::read_to_string(path).ok())
            .and_then(|raw| serde_json::from_str::<HintFile>(&raw).ok())
            .map(|file| file.hints)
            .unwrap_or_default();
        self.cache.borrow_mut().insert(key.to_string(), hints.clone());
        hints
    }

    fn save(&self, key: &str, hints: Vec<UiHint>) {
        if let Some(path) = self.file_path(key) {
            if let Some(parent) = path.parent() {
                let _ = fs::create_dir_all(parent);
            }
            let file = HintFile {
                version: HINTS_FILE_VERSION,
                hints: hints.clone(),
            };
            if let Ok(serialized) = serde_json::to_string_pretty(&file) {
                // tmp+rename so a crash never leaves a torn file.
                let tmp = path.with_extension("json.tmp");
                if fs::write(&tmp, serialized).is_ok() {
                    let _ = fs::rename(&tmp, &path);
                }
            }
        }
        self.cache.borrow_mut().insert(key.to_string(), hints);
    }

    /// Fresh hints matching `query`, best match first. Matching runs both
    /// directions so a short feature phrase ("web inspector") also matches a
    /// long query that contains it (the user's whole goal).
    pub(crate) fn lookup(
        &self,
        app: &FocusedApp,
        query: &str,
        max_results: usize,
        now_ms: u64,
    ) -> Vec<UiHint> {
        let Some(key) = Self::key_for(app) else {
            return Vec::new();
        };
        let mut scored: Vec<(u32, UiHint)> = self
            .load(&key)
            .into_iter()
            .filter(|hint| hint.is_fresh(now_ms) && hint.is_valid())
            .filter_map(|hint| {
                let forward = score_match(&hint.feature, query);
                let reverse = score_match(query, &hint.feature);
                let score = match (forward, reverse) {
                    (Some(a), Some(b)) => a.max(b),
                    (a, b) => a.or(b)?,
                };
                Some((score, hint))
            })
            .collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0));
        scored
            .into_iter()
            .take(max_results)
            .map(|(_, hint)| hint)
            .collect()
    }

    /// Upsert by normalized feature phrase; invalid hints are refused.
    /// Capped per app, dropping the oldest verification first.
    pub(crate) fn record(&self, app: &FocusedApp, hint: UiHint) -> bool {
        let Some(key) = Self::key_for(app) else {
            return false;
        };
        let mut hint = hint;
        // Collapse whitespace too so "Web   Inspector" and "web inspector"
        // upsert the same entry.
        hint.feature = normalize_for_match(&hint.feature)
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        if !hint.is_valid() {
            return false;
        }
        let mut hints = self.load(&key);
        hints.retain(|existing| {
            !(existing.feature == hint.feature && existing.kind == hint.kind)
        });
        hints.push(hint);
        hints.sort_by(|a, b| b.verified_at_ms.cmp(&a.verified_at_ms));
        hints.truncate(MAX_HINTS_PER_APP);
        self.save(&key, hints);
        true
    }

    /// Self-healing: a hinted menu path that came back not-found is deleted
    /// immediately so it stops being suggested.
    pub(crate) fn remove_menu_path(&self, app: &FocusedApp, path: &[String]) {
        let Some(key) = Self::key_for(app) else {
            return;
        };
        let hints = self.load(&key);
        let filtered: Vec<UiHint> = hints
            .into_iter()
            .filter(|hint| hint.menu_path.as_deref() != Some(path))
            .collect();
        self.save(&key, filtered);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app() -> FocusedApp {
        FocusedApp {
            bundle_id: Some("com.apple.Safari".into()),
            name: "Safari".into(),
            pid: Some(42),
        }
    }

    fn menu_hint(feature: &str, verified_at_ms: u64) -> UiHint {
        UiHint {
            feature: feature.into(),
            kind: UiHintKind::Menu,
            menu_path: Some(vec!["Develop".into(), "Show Web Inspector".into()]),
            combo: None,
            settings_pane: None,
            verified_at_ms,
            source: "menuScan".into(),
        }
    }

    fn store(dir: &std::path::Path) -> HintStore {
        HintStore::new(Some(dir.to_path_buf()))
    }

    #[test]
    fn record_and_lookup_round_trip_across_instances() {
        let dir = std::env::temp_dir().join(format!("screenie-hints-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let now = now_ms();

        assert!(store(&dir).record(&app(), menu_hint("web inspector", now)));
        let fresh_store = store(&dir);
        let hits = fresh_store.lookup(&app(), "web inspector", 3, now);
        assert_eq!(hits.len(), 1);
        assert_eq!(
            hits[0].menu_path.as_deref(),
            Some(&["Develop".to_string(), "Show Web Inspector".to_string()][..])
        );
        assert_eq!(
            hits[0].render(now).unwrap(),
            "menu Develop > Show Web Inspector (verified today)"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn upsert_dedupes_by_feature_and_caps_per_app() {
        let dir = std::env::temp_dir().join(format!("screenie-hints-cap-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let now = now_ms();
        let store = store(&dir);

        assert!(store.record(&app(), menu_hint("Web   Inspector", now - 1000)));
        assert!(store.record(&app(), menu_hint("web inspector", now)));
        assert_eq!(store.lookup(&app(), "web inspector", 5, now).len(), 1);

        for index in 0..(MAX_HINTS_PER_APP + 5) {
            assert!(store.record(&app(), menu_hint(&format!("feature {index}"), now)));
        }
        let all = store.load(&HintStore::key_for(&app()).unwrap());
        assert_eq!(all.len(), MAX_HINTS_PER_APP);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stale_and_invalid_hints_are_ignored_and_corrupt_files_tolerated() {
        let dir = std::env::temp_dir().join(format!("screenie-hints-stale-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let now = now_ms();
        let store = store(&dir);

        assert!(store.record(&app(), menu_hint("old feature", now - HINT_STALE_MS - 1)));
        assert!(store.lookup(&app(), "old feature", 3, now).is_empty());

        // Invalid payloads are refused at write time.
        let mut empty_path = menu_hint("broken", now);
        empty_path.menu_path = Some(Vec::new());
        assert!(!store.record(&app(), empty_path));
        let mut no_combo = menu_hint("no combo", now);
        no_combo.kind = UiHintKind::Shortcut;
        assert!(!store.record(&app(), no_combo));

        // A corrupt file reads as empty instead of erroring.
        let key = HintStore::key_for(&app()).unwrap();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(format!("{key}.json")), "{not json").unwrap();
        let corrupt_store = HintStore::new(Some(dir.clone()));
        assert!(corrupt_store.lookup(&app(), "anything", 3, now).is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn key_sanitizes_bundle_id_and_falls_back_to_name() {
        assert_eq!(
            HintStore::key_for(&app()).as_deref(),
            Some("com.apple.Safari")
        );
        let odd = FocusedApp {
            bundle_id: Some("com.evil/../../escape".into()),
            name: "Evil".into(),
            pid: None,
        };
        assert_eq!(
            HintStore::key_for(&odd).as_deref(),
            Some("com.evil_.._.._escape")
        );
        let unnamed = FocusedApp {
            bundle_id: None,
            name: "My App".into(),
            pid: None,
        };
        assert_eq!(HintStore::key_for(&unnamed).as_deref(), Some("My_App"));
    }

    #[test]
    fn failed_menu_path_is_removed() {
        let dir = std::env::temp_dir().join(format!("screenie-hints-heal-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let now = now_ms();
        let store = store(&dir);

        assert!(store.record(&app(), menu_hint("web inspector", now)));
        store.remove_menu_path(
            &app(),
            &["Develop".to_string(), "Show Web Inspector".to_string()],
        );
        assert!(store.lookup(&app(), "web inspector", 3, now).is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
