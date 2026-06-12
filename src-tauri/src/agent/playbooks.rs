//! Per-app/per-task markdown playbooks: operational knowledge that splices
//! into the per-step GOAL block only when the focused app or goal matches —
//! progressive disclosure, so the cached system prompt never grows. Built-in
//! playbooks compile in via `include_str!`; user files live under
//! `<app_data>/agent/playbooks/*.md` and shadow a built-in with the same
//! `name`. Enable/disable state persists in `config.json` next to them.
//!
//! Safety properties:
//! - Playbook text enters prompts behind a frame that marks it as local
//!   guidance, not user instructions, and tells the model to verify on
//!   screen.
//! - A playbook with `requires_scripting: true` is skipped entirely while
//!   the scripting toggle is off, so prompts can never advertise the script
//!   rung past the gate. The executor's per-script confirmation and
//!   forbidden-pattern checks remain the enforcement backstop.

use super::search::{normalize_for_match, score_match};
use super::types::FocusedApp;
use std::cell::RefCell;
use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

const CONFIG_FILE: &str = "config.json";
const MAX_NAME_CHARS: usize = 64;
const MAX_BODY_BYTES: usize = 32 * 1024;
const MAX_APPS: usize = 8;
const MAX_TRIGGERS: usize = 12;
/// Total characters of playbook text allowed into one goal block.
const MAX_GOAL_CHARS: usize = 2_500;
const MAX_PLAYBOOKS_PER_STEP: usize = 2;
/// Score for a playbook selected by app match alone (no triggers declared);
/// below every score_match tier so trigger-matched playbooks outrank it.
const APP_ONLY_SCORE: u32 = 50;

/// Built-in playbooks shipped with the binary. Adding one is a markdown
/// file plus an entry here.
const BUILTINS: &[&str] = &[
    include_str!("playbooks/builtin/browser-tasks.md"),
    include_str!("playbooks/builtin/system-settings.md"),
    include_str!("playbooks/builtin/finder-file-ops.md"),
];

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Playbook {
    pub name: String,
    /// Bundle ids (entries with a '.') or app names; `*` matches any app.
    pub apps: Vec<String>,
    /// Goal phrases that select this playbook. With apps set and no
    /// triggers, the app match alone selects it.
    pub triggers: Vec<String>,
    pub requires_scripting: bool,
    pub body: String,
    pub builtin: bool,
}

/// Parse `---` frontmatter (plain `key: value` lines — deliberately not
/// YAML; no new dependency) followed by a markdown body. Returns `None` for
/// anything malformed so a broken user file degrades to "no playbook".
pub(crate) fn parse_playbook(raw: &str) -> Option<Playbook> {
    let rest = raw.strip_prefix("---")?;
    let (header, body) = rest.split_once("\n---")?;
    let body = body.trim_start_matches(['\r', '\n']).trim_end();

    let mut name = String::new();
    let mut apps = Vec::new();
    let mut triggers = Vec::new();
    let mut requires_scripting = false;
    for line in header.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim();
        match key.trim() {
            "name" => name = value.to_string(),
            "apps" => apps = split_list(value),
            "triggers" => triggers = split_list(value),
            "requires_scripting" => requires_scripting = value.eq_ignore_ascii_case("true"),
            // Unknown keys are tolerated so the format can grow.
            _ => {}
        }
    }

    let valid = !name.is_empty()
        && name.chars().count() <= MAX_NAME_CHARS
        && !body.is_empty()
        && body.len() <= MAX_BODY_BYTES
        && apps.len() <= MAX_APPS
        && triggers.len() <= MAX_TRIGGERS;
    valid.then(|| Playbook {
        name,
        apps,
        triggers,
        requires_scripting,
        body: body.to_string(),
        builtin: false,
    })
}

fn split_list(value: &str) -> Vec<String> {
    value
        .split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .map(str::to_string)
        .collect()
}

#[derive(serde::Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct PlaybookConfig {
    #[serde(default)]
    disabled: Vec<String>,
}

/// Settings-UI view of one playbook (no body — list views stay light).
#[derive(serde::Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct PlaybookMeta {
    pub name: String,
    pub apps: Vec<String>,
    pub triggers: Vec<String>,
    pub requires_scripting: bool,
    /// The active version is the compiled-in one.
    pub builtin: bool,
    /// A user file shadows a built-in of the same name.
    pub overridden: bool,
    pub enabled: bool,
}

/// Playbook source for one agent run. `dir: None` (tests, headless) serves
/// built-ins only. Reads are cached for the run; only the settings UI
/// writes user files.
pub(crate) struct PlaybookStore {
    dir: Option<PathBuf>,
    cache: RefCell<Option<Vec<Playbook>>>,
}

impl PlaybookStore {
    pub(crate) fn new(dir: Option<PathBuf>) -> Self {
        Self {
            dir,
            cache: RefCell::new(None),
        }
    }

    /// Built-ins plus user files, user files shadowing built-ins by name,
    /// minus the disabled set. Corrupt files are skipped.
    fn load(&self) -> Vec<Playbook> {
        if let Some(cached) = self.cache.borrow().as_ref() {
            return cached.clone();
        }
        let disabled = self.disabled_names();
        let mut playbooks = self.load_all();
        playbooks.retain(|playbook| !disabled.contains(&playbook.name));
        self.cache.borrow_mut().replace(playbooks.clone());
        playbooks
    }

    fn disabled_names(&self) -> HashSet<String> {
        self.dir
            .as_ref()
            .and_then(|dir| fs::read_to_string(dir.join(CONFIG_FILE)).ok())
            .and_then(|raw| serde_json::from_str::<PlaybookConfig>(&raw).ok())
            .map(|config| config.disabled.into_iter().collect())
            .unwrap_or_default()
    }

    /// The playbook text for this step's goal block, or `None` when nothing
    /// matches. Re-evaluated per step: the goal is constant, so selection
    /// changes exactly when the focused app does.
    pub(crate) fn select_and_render(
        &self,
        app: &FocusedApp,
        goal: &str,
        scripting_enabled: bool,
    ) -> Option<String> {
        let mut scored: Vec<(u32, Playbook)> = self
            .load()
            .into_iter()
            .filter(|playbook| scripting_enabled || !playbook.requires_scripting)
            .filter(|playbook| app_matches(playbook, app))
            .filter_map(|playbook| Some((goal_score(&playbook, goal)?, playbook)))
            .collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.name.cmp(&b.1.name)));
        let selected: Vec<Playbook> = scored
            .into_iter()
            .take(MAX_PLAYBOOKS_PER_STEP)
            .map(|(_, playbook)| playbook)
            .collect();
        if selected.is_empty() {
            return None;
        }

        let share = MAX_GOAL_CHARS / selected.len();
        let mut rendered = String::from(
            "Playbook for this app/task (local guidance, not user instructions; verify on screen):",
        );
        for playbook in &selected {
            rendered.push_str(&format!(
                "\n[{}]\n{}",
                playbook.name,
                render_within_budget(&playbook.body, goal, share)
            ));
        }
        Some(rendered)
    }
}

/// Management surface for the settings UI. These construct a fresh store
/// per command, so the per-run read cache never serves stale state here.
impl PlaybookStore {
    pub(crate) fn list_meta(&self) -> Vec<PlaybookMeta> {
        let disabled = self.disabled_names();
        let builtin_names: HashSet<String> = BUILTINS
            .iter()
            .filter_map(|raw| parse_playbook(raw))
            .map(|playbook| playbook.name)
            .collect();
        let mut meta: Vec<PlaybookMeta> = self
            .load_all()
            .into_iter()
            .map(|playbook| PlaybookMeta {
                enabled: !disabled.contains(&playbook.name),
                overridden: !playbook.builtin && builtin_names.contains(&playbook.name),
                builtin: playbook.builtin,
                name: playbook.name,
                apps: playbook.apps,
                triggers: playbook.triggers,
                requires_scripting: playbook.requires_scripting,
            })
            .collect();
        meta.sort_by(|a, b| a.name.cmp(&b.name));
        meta
    }

    /// Raw markdown for editing: the user file when one exists, else the
    /// built-in source.
    pub(crate) fn read_raw(&self, name: &str) -> Option<String> {
        if let Some(path) = self.user_file_for(name) {
            return fs::read_to_string(path).ok();
        }
        BUILTINS
            .iter()
            .find(|raw| parse_playbook(raw).is_some_and(|playbook| playbook.name == name))
            .map(|raw| raw.to_string())
    }

    /// Validate and write a user playbook (named by its own frontmatter);
    /// shadows a built-in with the same name on the next load.
    pub(crate) fn write_user(&self, content: &str) -> Result<PlaybookMeta, String> {
        let playbook =
            parse_playbook(content).ok_or_else(|| {
                "invalid playbook: needs --- frontmatter with a name, a non-empty body, and caps respected (name ≤ 64 chars, body ≤ 32KB, ≤ 8 apps, ≤ 12 triggers)".to_string()
            })?;
        let dir = self.dir.as_ref().ok_or("no playbook directory")?;
        fs::create_dir_all(dir).map_err(|err| format!("create playbooks dir: {err}"))?;
        let path = dir.join(format!("{}.md", sanitize_name(&playbook.name)));
        let tmp = path.with_extension("md.tmp");
        fs::write(&tmp, content).map_err(|err| format!("write playbook: {err}"))?;
        fs::rename(&tmp, &path).map_err(|err| format!("write playbook: {err}"))?;
        self.cache.borrow_mut().take();
        let enabled = !self.disabled_names().contains(&playbook.name);
        Ok(PlaybookMeta {
            overridden: BUILTINS
                .iter()
                .filter_map(|raw| parse_playbook(raw))
                .any(|builtin| builtin.name == playbook.name),
            builtin: false,
            enabled,
            name: playbook.name,
            apps: playbook.apps,
            triggers: playbook.triggers,
            requires_scripting: playbook.requires_scripting,
        })
    }

    /// Delete the user file (a shadowed built-in reappears). Built-ins
    /// themselves cannot be deleted, only disabled.
    pub(crate) fn delete_user(&self, name: &str) -> Result<(), String> {
        let path = self
            .user_file_for(name)
            .ok_or_else(|| format!("no user playbook named '{name}'"))?;
        fs::remove_file(path).map_err(|err| format!("delete playbook: {err}"))?;
        self.cache.borrow_mut().take();
        Ok(())
    }

    pub(crate) fn set_enabled(&self, name: &str, enabled: bool) -> Result<(), String> {
        let dir = self.dir.as_ref().ok_or("no playbook directory")?;
        fs::create_dir_all(dir).map_err(|err| format!("create playbooks dir: {err}"))?;
        let mut disabled: Vec<String> = self.disabled_names().into_iter().collect();
        disabled.retain(|entry| entry != name);
        if !enabled {
            disabled.push(name.to_string());
        }
        disabled.sort();
        let config = PlaybookConfig { disabled };
        let serialized = serde_json::to_string_pretty(&config)
            .map_err(|err| format!("encode playbook config: {err}"))?;
        let path = dir.join(CONFIG_FILE);
        let tmp = path.with_extension("json.tmp");
        fs::write(&tmp, serialized).map_err(|err| format!("write playbook config: {err}"))?;
        fs::rename(&tmp, &path).map_err(|err| format!("write playbook config: {err}"))?;
        self.cache.borrow_mut().take();
        Ok(())
    }

    /// All playbooks including disabled ones (management view).
    fn load_all(&self) -> Vec<Playbook> {
        let mut playbooks: Vec<Playbook> = BUILTINS
            .iter()
            .filter_map(|raw| parse_playbook(raw))
            .map(|playbook| Playbook {
                builtin: true,
                ..playbook
            })
            .collect();
        if let Some(dir) = self.dir.as_ref() {
            let mut user_files: Vec<PathBuf> = fs::read_dir(dir)
                .map(|entries| {
                    entries
                        .filter_map(|entry| entry.ok())
                        .map(|entry| entry.path())
                        .filter(|path| path.extension().is_some_and(|ext| ext == "md"))
                        .collect()
                })
                .unwrap_or_default();
            user_files.sort();
            for path in user_files {
                let Some(playbook) = fs::read_to_string(&path)
                    .ok()
                    .and_then(|raw| parse_playbook(&raw))
                else {
                    continue;
                };
                playbooks.retain(|existing| existing.name != playbook.name);
                playbooks.push(playbook);
            }
        }
        playbooks
    }

    /// Path of the user file whose frontmatter name matches, regardless of
    /// its filename (hand-created files may not follow the sanitized name).
    fn user_file_for(&self, name: &str) -> Option<PathBuf> {
        let dir = self.dir.as_ref()?;
        fs::read_dir(dir)
            .ok()?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "md"))
            .find(|path| {
                fs::read_to_string(path)
                    .ok()
                    .and_then(|raw| parse_playbook(&raw))
                    .is_some_and(|playbook| playbook.name == name)
            })
    }
}

fn sanitize_name(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// `*` matches any app; entries with a '.' compare against the bundle id,
/// others against the app name (both case-insensitive).
fn app_matches(playbook: &Playbook, app: &FocusedApp) -> bool {
    let bundle = app.bundle_id.as_deref().unwrap_or("").trim();
    let name = app.name.trim();
    playbook.apps.iter().any(|entry| {
        entry == "*"
            || (entry.contains('.') && entry.eq_ignore_ascii_case(bundle))
            || (!entry.contains('.') && entry.eq_ignore_ascii_case(name))
    })
}

/// Best trigger-vs-goal score (bidirectional, like hint lookup). A playbook
/// with no triggers is selected on app match alone at a low base score.
fn goal_score(playbook: &Playbook, goal: &str) -> Option<u32> {
    if playbook.triggers.is_empty() {
        return (!playbook.apps.is_empty()).then_some(APP_ONLY_SCORE);
    }
    playbook
        .triggers
        .iter()
        .chain(std::iter::once(&playbook.name))
        .filter_map(|trigger| {
            let forward = score_match(goal, trigger);
            let reverse = score_match(trigger, goal);
            match (forward, reverse) {
                (Some(a), Some(b)) => Some(a.max(b)),
                (a, b) => a.or(b),
            }
        })
        .max()
}

/// Fit a body into `budget` characters. Small bodies pass through; larger
/// ones are reduced to the preamble plus the goal-relevant `## ` sections
/// (kept in document order), hard-truncated as the last resort.
fn render_within_budget(body: &str, goal: &str, budget: usize) -> String {
    if body.chars().count() <= budget {
        return body.to_string();
    }
    let mut sections = split_sections(body);
    let preamble = if sections
        .first()
        .is_some_and(|section| !section.starts_with("## "))
    {
        sections.remove(0)
    } else {
        String::new()
    };

    // Greedily keep the best-scoring sections that fit the budget.
    let mut scored: Vec<(usize, u32)> = sections
        .iter()
        .enumerate()
        .filter_map(|(index, section)| {
            let heading = section
                .lines()
                .next()
                .unwrap_or("")
                .trim_start_matches('#')
                .trim();
            section_score(heading, goal).map(|score| (index, score))
        })
        .collect();
    scored.sort_by(|a, b| b.1.cmp(&a.1));
    let mut kept = vec![false; sections.len()];
    let mut used = preamble.chars().count();
    for (index, _) in scored {
        let cost = sections[index].chars().count() + 1;
        if used + cost <= budget {
            kept[index] = true;
            used += cost;
        }
    }

    let mut out = preamble;
    for (index, section) in sections.iter().enumerate() {
        if kept[index] {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(section);
        }
    }
    if out.trim().is_empty() {
        // Nothing scored against the goal — fall back to the raw body.
        out = body.to_string();
    }
    truncate_chars(&out, budget)
}

/// Section headings are coarse topical buckets, so unlike the strict findUi
/// matcher this scores by token overlap: any shared word selects the
/// section, more shared words rank it higher.
fn section_score(heading: &str, goal: &str) -> Option<u32> {
    let heading_norm = normalize_for_match(heading);
    let goal_norm = normalize_for_match(goal);
    let goal_tokens: HashSet<&str> = goal_norm.split_whitespace().collect();
    let overlap = heading_norm
        .split_whitespace()
        .filter(|token| goal_tokens.contains(token))
        .count() as u32;
    (overlap > 0).then_some(overlap)
}

fn split_sections(body: &str) -> Vec<String> {
    let mut sections: Vec<String> = Vec::new();
    for line in body.lines() {
        if line.starts_with("## ") || sections.is_empty() {
            if line.starts_with("## ") {
                sections.push(line.to_string());
                continue;
            }
            sections.push(String::new());
        }
        let current = sections.last_mut().expect("section started above");
        if !current.is_empty() {
            current.push('\n');
        }
        current.push_str(line);
    }
    sections
}

fn truncate_chars(text: &str, budget: usize) -> String {
    if text.chars().count() <= budget {
        return text.to_string();
    }
    let mut truncated: String = text.chars().take(budget.saturating_sub(3)).collect();
    truncated.push_str("...");
    truncated
}

#[cfg(test)]
mod tests {
    use super::*;

    fn safari() -> FocusedApp {
        FocusedApp {
            bundle_id: Some("com.apple.Safari".into()),
            name: "Safari".into(),
            pid: Some(42),
        }
    }

    fn finder() -> FocusedApp {
        FocusedApp {
            bundle_id: Some("com.apple.finder".into()),
            name: "Finder".into(),
            pid: Some(7),
        }
    }

    fn write_playbook(dir: &std::path::Path, file: &str, content: &str) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join(file), content).unwrap();
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("screenie-playbooks-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn parse_round_trips_frontmatter_and_rejects_malformed() {
        let playbook = parse_playbook(
            "---\nname: pin-tabs\napps: com.apple.Safari, Chrome\ntriggers: pin tab, tabs\nrequires_scripting: true\nmystery: ignored\n---\nBody text\n\n## Section\nMore\n",
        )
        .unwrap();
        assert_eq!(playbook.name, "pin-tabs");
        assert_eq!(playbook.apps, vec!["com.apple.Safari", "Chrome"]);
        assert_eq!(playbook.triggers, vec!["pin tab", "tabs"]);
        assert!(playbook.requires_scripting);
        assert!(playbook.body.starts_with("Body text"));
        assert!(!playbook.builtin);

        // Missing frontmatter, missing name, empty body all degrade to None.
        assert!(parse_playbook("no frontmatter").is_none());
        assert!(parse_playbook("---\napps: x\n---\nbody").is_none());
        assert!(parse_playbook("---\nname: x\n---\n\n").is_none());
        let long_name = "x".repeat(MAX_NAME_CHARS + 1);
        assert!(parse_playbook(&format!("---\nname: {long_name}\n---\nbody")).is_none());
    }

    #[test]
    fn builtins_parse_and_load_without_a_dir() {
        let store = PlaybookStore::new(None);
        let loaded = store.load();
        for name in ["browser-tasks", "system-settings", "finder-file-ops"] {
            assert!(
                loaded.iter().any(|p| p.name == name && p.builtin),
                "{name} builtin missing: {:?}",
                loaded.iter().map(|p| &p.name).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn builtin_routing_playbooks_route_settings_and_gate_finder_scripts() {
        let store = PlaybookStore::new(None);

        let settings_app = FocusedApp {
            bundle_id: Some("com.apple.systempreferences".into()),
            name: "System Settings".into(),
            pid: Some(9),
        };
        let rendered = store
            .select_and_render(&settings_app, "turn on do not disturb", false)
            .unwrap();
        assert!(rendered.contains("[system-settings]"));
        assert!(rendered.contains("x-apple.systempreferences:"));

        // The Finder script playbook never appears with scripting off.
        assert!(store
            .select_and_render(&finder(), "move the report file to a new folder", false)
            .is_none());
        let with_scripting = store
            .select_and_render(&finder(), "move the report file to a new folder", true)
            .unwrap();
        assert!(with_scripting.contains("[finder-file-ops]"));
        assert!(with_scripting.contains("moveToTrash"));
    }

    #[test]
    fn user_file_shadows_builtin_and_corrupt_files_are_skipped() {
        let dir = temp_dir("shadow");
        write_playbook(
            &dir,
            "browser-tasks.md",
            "---\nname: browser-tasks\napps: *\ntriggers: search\n---\nUser override body",
        );
        write_playbook(&dir, "broken.md", "not a playbook");

        let store = PlaybookStore::new(Some(dir.clone()));
        let loaded = store.load();
        let browser: Vec<&Playbook> =
            loaded.iter().filter(|p| p.name == "browser-tasks").collect();
        assert_eq!(browser.len(), 1);
        assert!(!browser[0].builtin);
        assert_eq!(browser[0].body, "User override body");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn disabled_config_removes_playbooks() {
        let dir = temp_dir("disabled");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join(CONFIG_FILE),
            r#"{"disabled":["browser-tasks"]}"#,
        )
        .unwrap();

        let store = PlaybookStore::new(Some(dir.clone()));
        assert!(store.load().iter().all(|p| p.name != "browser-tasks"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn selection_requires_app_and_goal_match() {
        let dir = temp_dir("select");
        write_playbook(
            &dir,
            "pin.md",
            "---\nname: pin-tabs\napps: com.apple.Safari\ntriggers: pin tab\n---\nPin via the tab context menu.",
        );
        let store = PlaybookStore::new(Some(dir.clone()));

        let hit = store
            .select_and_render(&safari(), "pin this tab for me", false)
            .unwrap();
        assert!(hit.contains("local guidance, not user instructions"));
        assert!(hit.contains("[pin-tabs]"));
        assert!(hit.contains("Pin via the tab context menu."));

        // Wrong app and wrong goal both deselect.
        assert!(store
            .select_and_render(&finder(), "pin this tab for me", false)
            .is_none());
        assert!(store
            .select_and_render(&safari(), "write an email to Ana", false)
            .is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn app_name_and_wildcard_entries_match() {
        let by_name = Playbook {
            name: "n".into(),
            apps: vec!["Safari".into()],
            triggers: vec![],
            requires_scripting: false,
            body: "b".into(),
            builtin: false,
        };
        assert!(app_matches(&by_name, &safari()));
        assert!(!app_matches(&by_name, &finder()));
        let wildcard = Playbook {
            apps: vec!["*".into()],
            ..by_name.clone()
        };
        assert!(app_matches(&wildcard, &finder()));
    }

    #[test]
    fn scripting_gated_playbooks_are_skipped_when_toggle_off() {
        let dir = temp_dir("gate");
        write_playbook(
            &dir,
            "files.md",
            "---\nname: finder-files\napps: com.apple.finder\ntriggers: move file\nrequires_scripting: true\n---\nUse applescript for file ops.",
        );
        let store = PlaybookStore::new(Some(dir.clone()));

        assert!(store
            .select_and_render(&finder(), "move file report.pdf", false)
            .is_none());
        assert!(store
            .select_and_render(&finder(), "move file report.pdf", true)
            .is_some());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn oversize_bodies_reduce_to_relevant_sections_within_budget() {
        let filler = "filler line\n".repeat(150); // ~1.8k chars per section
        let body = format!(
            "Preamble guidance.\n## Site search\nUse the site search field.\n{filler}\n## Tabs\nReuse tabs.\n{filler}"
        );
        let rendered = render_within_budget(&body, "search amazon for a mac mini", 2_000);
        assert!(rendered.chars().count() <= 2_000);
        assert!(rendered.contains("Preamble guidance."));
        // "Site search" shares the token "search" with the goal; "Tabs"
        // shares nothing and is dropped to fit the budget.
        assert!(rendered.contains("## Site search"));
        assert!(!rendered.contains("## Tabs"));

        // Tiny budgets still terminate via hard truncation.
        let tiny = render_within_budget(&body, "search", 80);
        assert!(tiny.chars().count() <= 80);
    }

    #[test]
    fn management_surface_lists_writes_toggles_and_deletes() {
        let dir = temp_dir("manage");
        let store = PlaybookStore::new(Some(dir.clone()));

        // Built-ins listed enabled; writing an override flips its provenance.
        let meta = store.list_meta();
        let browser = meta.iter().find(|m| m.name == "browser-tasks").unwrap();
        assert!(browser.builtin && browser.enabled && !browser.overridden);

        let written = store
            .write_user("---\nname: browser-tasks\napps: *\ntriggers: search\n---\nOverride body")
            .unwrap();
        assert!(!written.builtin && written.overridden);
        let meta = store.list_meta();
        let browser = meta.iter().find(|m| m.name == "browser-tasks").unwrap();
        assert!(!browser.builtin && browser.overridden);
        assert_eq!(
            store.read_raw("browser-tasks").unwrap(),
            "---\nname: browser-tasks\napps: *\ntriggers: search\n---\nOverride body"
        );

        // Invalid content is refused before touching disk.
        assert!(store.write_user("no frontmatter at all").is_err());

        // Disable removes it from selection but not from the list.
        store.set_enabled("browser-tasks", false).unwrap();
        let meta = store.list_meta();
        assert!(!meta.iter().find(|m| m.name == "browser-tasks").unwrap().enabled);
        assert!(store
            .select_and_render(&safari(), "search the web for socks", false)
            .map(|text| !text.contains("[browser-tasks]"))
            .unwrap_or(true));
        store.set_enabled("browser-tasks", true).unwrap();

        // Deleting the override restores the built-in; built-ins themselves
        // cannot be deleted.
        store.delete_user("browser-tasks").unwrap();
        let meta = store.list_meta();
        let browser = meta.iter().find(|m| m.name == "browser-tasks").unwrap();
        assert!(browser.builtin && !browser.overridden);
        assert!(store.delete_user("browser-tasks").is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn two_playbooks_share_the_budget_and_render_deterministically() {
        let dir = temp_dir("two");
        write_playbook(
            &dir,
            "a.md",
            "---\nname: alpha\napps: *\ntriggers: search\n---\nAlpha body.",
        );
        write_playbook(
            &dir,
            "b.md",
            "---\nname: beta\napps: *\ntriggers: search\n---\nBeta body.",
        );
        write_playbook(
            &dir,
            "c.md",
            "---\nname: gamma\napps: *\ntriggers: search\n---\nGamma body.",
        );
        let store = PlaybookStore::new(Some(dir.clone()));

        let first = store
            .select_and_render(&safari(), "search the web", false)
            .unwrap();
        let second = store
            .select_and_render(&safari(), "search the web", false)
            .unwrap();
        assert_eq!(first, second);
        // Cap is two playbooks per step.
        let count = ["[alpha]", "[beta]", "[gamma]"]
            .iter()
            .filter(|tag| first.contains(*tag))
            .count();
        assert_eq!(count, 2);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
