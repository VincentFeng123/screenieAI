//! Cross-run agent memory: one durable fact per completed goal, written
//! only when a run ends in `Done` AND the model explicitly provided a
//! `remember` field on its done decision. A single global JSON file at
//! `<app_data>/agent/memory/memory.json`; matching entries are injected at
//! run start when they fuzzy-match the new goal.
//!
//! Safety properties (vs. hints.rs's verified-facts-only stance):
//! - Only the bounded `remember` field is free text, and it is the model's
//!   OWN words (same trust class as the existing `note` field, which
//!   already round-trips into prompts) — never page text.
//! - Rendering is template-driven and framed as unverified; entries inject
//!   only when the goal matches, never globally.
//! - Every write surfaces to the user as an audit card with a delete path.

use super::search::score_match;
#[cfg(test)]
use super::hints::now_ms;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::fs;
use std::path::PathBuf;

const MEMORY_FILE_VERSION: u32 = 1;
const MEMORY_FILE: &str = "memory.json";
const MAX_ENTRIES: usize = 40;
pub(crate) const MAX_GOAL_SUMMARY_CHARS: usize = 120;
pub(crate) const MAX_REMEMBER_CHARS: usize = 200;
/// Memory is model-authored free text about moving UIs — stale faster than
/// executor-verified hints (6 months); ignore entries older than ~90 days.
const MEMORY_STALE_MS: u64 = 90 * 24 * 60 * 60 * 1000;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MemoryEntry {
    pub id: String,
    /// Truncated goal text, the lookup key.
    pub goal_summary: String,
    /// HintStore-style sanitized app key of the app focused at completion.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub app_key: Option<String>,
    /// The model's own durable takeaway (bounded free text).
    pub remember: String,
    pub outcome: String,
    pub steps: u32,
    pub created_at_ms: u64,
}

impl MemoryEntry {
    fn is_fresh(&self, now_ms: u64) -> bool {
        now_ms.saturating_sub(self.created_at_ms) < MEMORY_STALE_MS
    }

    fn is_valid(&self) -> bool {
        !self.id.trim().is_empty()
            && !self.goal_summary.trim().is_empty()
            && self.goal_summary.chars().count() <= MAX_GOAL_SUMMARY_CHARS
            && !self.remember.trim().is_empty()
            && self.remember.chars().count() <= MAX_REMEMBER_CHARS
    }

    /// Template-driven goal-block line; free text never escapes its slot.
    pub(crate) fn render(&self, now_ms: u64) -> String {
        let days = now_ms.saturating_sub(self.created_at_ms) / (24 * 60 * 60 * 1000);
        let age = if days == 0 {
            "today".to_string()
        } else {
            format!("{days}d ago")
        };
        let app = self.app_key.as_deref().unwrap_or("any app");
        format!(
            "- [{age}, {app}] goal: {} — learned: {}",
            self.goal_summary, self.remember
        )
    }
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct MemoryFile {
    version: u32,
    entries: Vec<MemoryEntry>,
}

/// Single-file memory persistence. `dir: None` (tests, headless) makes
/// every operation a cheap no-op, like HintStore.
pub(crate) struct MemoryStore {
    dir: Option<PathBuf>,
    cache: RefCell<Option<Vec<MemoryEntry>>>,
}

impl MemoryStore {
    pub(crate) fn new(dir: Option<PathBuf>) -> Self {
        Self {
            dir,
            cache: RefCell::new(None),
        }
    }

    fn file_path(&self) -> Option<PathBuf> {
        Some(self.dir.as_ref()?.join(MEMORY_FILE))
    }

    fn load(&self) -> Vec<MemoryEntry> {
        if let Some(cached) = self.cache.borrow().as_ref() {
            return cached.clone();
        }
        let entries = self
            .file_path()
            .and_then(|path| fs::read_to_string(path).ok())
            .and_then(|raw| serde_json::from_str::<MemoryFile>(&raw).ok())
            .map(|file| file.entries)
            .unwrap_or_default();
        self.cache.borrow_mut().replace(entries.clone());
        entries
    }

    fn save(&self, entries: Vec<MemoryEntry>) {
        if let Some(path) = self.file_path() {
            if let Some(parent) = path.parent() {
                let _ = fs::create_dir_all(parent);
            }
            let file = MemoryFile {
                version: MEMORY_FILE_VERSION,
                entries: entries.clone(),
            };
            if let Ok(serialized) = serde_json::to_string_pretty(&file) {
                // tmp+rename so a crash never leaves a torn file.
                let tmp = path.with_extension("json.tmp");
                if fs::write(&tmp, serialized).is_ok() {
                    let _ = fs::rename(&tmp, &path);
                }
            }
        }
        self.cache.borrow_mut().replace(entries);
    }

    /// Persist one completed-run takeaway; invalid entries are refused.
    /// Capped globally, dropping the oldest first.
    pub(crate) fn record(&self, entry: MemoryEntry) -> bool {
        if !entry.is_valid() {
            return false;
        }
        let mut entries = self.load();
        entries.push(entry);
        entries.sort_by(|a, b| b.created_at_ms.cmp(&a.created_at_ms));
        entries.truncate(MAX_ENTRIES);
        self.save(entries);
        true
    }

    /// Fresh entries whose goal or takeaway matches the new goal, best
    /// match first. Bidirectional like hint lookup.
    pub(crate) fn lookup(&self, goal: &str, max_results: usize, now_ms: u64) -> Vec<MemoryEntry> {
        let mut scored: Vec<(u32, MemoryEntry)> = self
            .load()
            .into_iter()
            .filter(|entry| entry.is_fresh(now_ms) && entry.is_valid())
            .filter_map(|entry| {
                let haystack = format!("{} {}", entry.goal_summary, entry.remember);
                let forward = score_match(&haystack, goal);
                let reverse = score_match(goal, &entry.goal_summary);
                let score = match (forward, reverse) {
                    (Some(a), Some(b)) => a.max(b),
                    (a, b) => a.or(b)?,
                };
                Some((score, entry))
            })
            .collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0));
        scored
            .into_iter()
            .take(max_results)
            .map(|(_, entry)| entry)
            .collect()
    }

    pub(crate) fn list(&self) -> Vec<MemoryEntry> {
        self.load()
    }

    pub(crate) fn delete(&self, id: &str) -> bool {
        let entries = self.load();
        let before = entries.len();
        let remaining: Vec<MemoryEntry> = entries
            .into_iter()
            .filter(|entry| entry.id != id)
            .collect();
        let deleted = remaining.len() != before;
        if deleted {
            self.save(remaining);
        }
        deleted
    }

    pub(crate) fn clear(&self) {
        self.save(Vec::new());
    }
}

/// Truncate the raw goal into the stored summary key.
pub(crate) fn summarize_goal(goal: &str) -> String {
    let normalized = goal.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.chars().count() <= MAX_GOAL_SUMMARY_CHARS {
        return normalized;
    }
    let mut truncated: String = normalized
        .chars()
        .take(MAX_GOAL_SUMMARY_CHARS.saturating_sub(3))
        .collect();
    truncated.push_str("...");
    truncated
}

pub(crate) fn new_memory_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(goal: &str, remember: &str, created_at_ms: u64) -> MemoryEntry {
        MemoryEntry {
            id: new_memory_id(),
            goal_summary: goal.into(),
            app_key: Some("com.apple.Safari".into()),
            remember: remember.into(),
            outcome: "done".into(),
            steps: 7,
            created_at_ms,
        }
    }

    fn store(dir: &std::path::Path) -> MemoryStore {
        MemoryStore::new(Some(dir.to_path_buf()))
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("screenie-memory-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn record_and_lookup_round_trip_across_instances() {
        let dir = temp_dir("roundtrip");
        let now = now_ms();

        assert!(store(&dir).record(entry(
            "compare refurbished mac mini prices",
            "B&H lists refurbs under Computers > Used; prices beat Amazon",
            now,
        )));
        let fresh = store(&dir);
        let hits = fresh.lookup("compare refurbished mac mini prices", 3, now);
        assert_eq!(hits.len(), 1);
        assert!(hits[0].render(now).starts_with("- [today, com.apple.Safari] goal:"));

        // Unrelated goals do not match.
        assert!(fresh.lookup("write an email to Ana", 3, now).is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn caps_at_max_entries_dropping_oldest() {
        let dir = temp_dir("cap");
        let now = now_ms();
        let store = store(&dir);
        for index in 0..(MAX_ENTRIES + 5) {
            assert!(store.record(entry(
                &format!("goal number {index}"),
                "fact",
                now + index as u64,
            )));
        }
        let all = store.list();
        assert_eq!(all.len(), MAX_ENTRIES);
        // Oldest dropped: the smallest created_at_ms values are gone.
        assert!(all.iter().all(|entry| entry.created_at_ms >= now + 5));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stale_invalid_and_corrupt_are_tolerated() {
        let dir = temp_dir("stale");
        let now = now_ms();
        let store = store(&dir);

        assert!(store.record(entry("old goal", "old fact", now - MEMORY_STALE_MS - 1)));
        assert!(store.lookup("old goal", 3, now).is_empty());

        // Invalid entries refused at write time.
        assert!(!store.record(entry("g", "", now)));
        let long = "x".repeat(MAX_REMEMBER_CHARS + 1);
        assert!(!store.record(entry("g", &long, now)));

        // Corrupt file reads as empty.
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(MEMORY_FILE), "{broken").unwrap();
        assert!(MemoryStore::new(Some(dir.clone())).list().is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn delete_and_clear_remove_entries() {
        let dir = temp_dir("delete");
        let now = now_ms();
        let store = store(&dir);

        let kept = entry("goal a", "fact a", now);
        let gone = entry("goal b", "fact b", now);
        let gone_id = gone.id.clone();
        assert!(store.record(kept));
        assert!(store.record(gone));

        assert!(store.delete(&gone_id));
        assert!(!store.delete(&gone_id));
        assert_eq!(store.list().len(), 1);

        store.clear();
        assert!(store.list().is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn summarize_goal_normalizes_and_truncates() {
        assert_eq!(summarize_goal("  open   Safari  "), "open Safari");
        let long = "word ".repeat(60);
        let summary = summarize_goal(&long);
        assert!(summary.chars().count() <= MAX_GOAL_SUMMARY_CHARS);
        assert!(summary.ends_with("..."));
    }
}
