//! Single source of truth for the planner's model-visible action contract.
//!
//! Every representation of an action that is written as a *string* — the
//! decision-schema enum, the per-action field whitelists, the alias/synonym
//! table, the batchable-`next` list, and the prompt's example objects — is
//! declared once here and consumed (or drift-checked) by the planner and the
//! provider schema builders. The `Action` enum in `types.rs` and the
//! exhaustive matches in `executor.rs` deliberately stay hand-written: the
//! compiler enforcing every variant there is a feature, not duplication.
//!
//! Adding a model-visible action now means: the `Action` variant (the
//! compiler forces the executor arms), one `ActionSpec` entry here, one parse
//! arm in `parse_raw_planner_action` (the example round-trip test forces it),
//! the `types.rs` ser/de arms, and a prose rule in the system prompt.
#![allow(dead_code)] // consumed incrementally; fully wired by the registry migration

/// When an action is advertised to the model. Gating is prompt-level only:
/// the schema always lists every action, and the executor enforces the real
/// gates (scripting toggle, per-script confirmation) regardless of prompts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Gate {
    Always,
    /// Advertised only when the user's scripting toggle is on.
    Scripting,
    /// Advertised only when the webLookup toggle is on and the provider
    /// supports server-side search.
    WebLookup,
}

/// One action-scoped JSON field. `required` documents the parse contract;
/// the flat decision schema cannot express per-action requiredness, so the
/// parse arms remain the enforcement point.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FieldSpec {
    pub name: &'static str,
    pub required: bool,
}

const fn req(name: &'static str) -> FieldSpec {
    FieldSpec {
        name,
        required: true,
    }
}

const fn opt(name: &'static str) -> FieldSpec {
    FieldSpec {
        name,
        required: false,
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ActionSpec {
    /// Canonical model-facing name; must equal the `types.rs` Serialize tag.
    pub name: &'static str,
    /// Accepted alternate spellings. Lowercase entries are canonicalized via
    /// the synonym table; mixed-case entries (e.g. `runShortcut`) pass the
    /// case-preserving fallthrough and are matched by the parse arm directly.
    pub aliases: &'static [&'static str],
    /// Action-scoped fields, in the whitelist's order. Envelope fields
    /// (reason, note, expect, milestone_done, target_name, target_role,
    /// next) apply to every action and are not repeated here.
    pub fields: &'static [FieldSpec],
    /// May ride in the `next` follow-up batch.
    pub batchable: bool,
    pub gate: Gate,
    /// Id-targeted actions must echo the element's name as `target_name`.
    pub requires_target_name: bool,
    /// The exact example object the system prompt teaches. Doubles as the
    /// drift test's parse fixture, so it must always parse.
    pub prompt_example: &'static str,
}

/// Declared in the exact order of the decision schema's action enum; schema
/// generation iterates this slice, so reordering changes provider request
/// bytes and busts the Anthropic prompt cache mid-run. Append, don't sort.
pub(crate) static ACTIONS: &[ActionSpec] = &[
    ActionSpec {
        name: "activateApp",
        aliases: &["activate_app"],
        fields: &[req("app")],
        batchable: false,
        gate: Gate::Always,
        requires_target_name: false,
        prompt_example: r#"{"reason":"brief reason","action":"activateApp","app":"Safari"}"#,
    },
    ActionSpec {
        name: "click",
        aliases: &["left_click", "leftclick", "click_element", "tap"],
        fields: &[req("id")],
        batchable: true,
        gate: Gate::Always,
        requires_target_name: true,
        prompt_example: r#"{"reason":"brief reason","action":"click","id":14,"target_name":"Add to Bag"}"#,
    },
    ActionSpec {
        name: "clickText",
        aliases: &[
            "clicktext",
            "click_text",
            "click_by_text",
            "clickbytext",
            "click_label",
        ],
        fields: &[req("text"), opt("role"), opt("nth")],
        batchable: false,
        gate: Gate::Always,
        requires_target_name: false,
        prompt_example: r#"{"reason":"select largest display","action":"clickText","text":"iPhone 17 Pro Max","role":"radio"}"#,
    },
    ActionSpec {
        name: "doubleClick",
        aliases: &["double_click"],
        fields: &[req("id")],
        batchable: false,
        gate: Gate::Always,
        requires_target_name: true,
        prompt_example: r#"{"reason":"brief reason","action":"doubleClick","id":14,"target_name":"report.pdf"}"#,
    },
    ActionSpec {
        name: "type",
        aliases: &[
            "type_text",
            "input",
            "input_text",
            "enter_text",
            "set_text",
            "settext",
        ],
        fields: &[req("id"), req("text")],
        batchable: true,
        gate: Gate::Always,
        requires_target_name: true,
        prompt_example: r#"{"reason":"brief reason","action":"type","id":9,"target_name":"Address and Search","text":"..."}"#,
    },
    ActionSpec {
        name: "key",
        aliases: &["press", "press_key", "hotkey", "keypress", "key_press"],
        fields: &[req("combo")],
        batchable: true,
        gate: Gate::Always,
        requires_target_name: false,
        prompt_example: r#"{"reason":"brief reason","action":"key","combo":"cmd+s"}"#,
    },
    ActionSpec {
        name: "menu",
        aliases: &[
            "menu_click",
            "menuclick",
            "click_menu",
            "menu_item",
            "menuitem",
            "select_menu",
            "menu_select",
        ],
        fields: &[req("path")],
        batchable: false,
        gate: Gate::Always,
        requires_target_name: false,
        prompt_example: r#"{"reason":"brief reason","action":"menu","path":["File","Export as PDF"]}"#,
    },
    ActionSpec {
        name: "scroll",
        // dx/dy are individually optional; the parse arm requires at least
        // one to be non-zero.
        aliases: &[],
        fields: &[opt("dx"), opt("dy")],
        batchable: true,
        gate: Gate::Always,
        requires_target_name: false,
        prompt_example: r#"{"reason":"brief reason","action":"scroll","dx":0,"dy":300}"#,
    },
    ActionSpec {
        name: "wait",
        aliases: &[],
        fields: &[req("ms")],
        batchable: true,
        gate: Gate::Always,
        requires_target_name: false,
        prompt_example: r#"{"reason":"brief reason","action":"wait","ms":200}"#,
    },
    ActionSpec {
        name: "openUrl",
        aliases: &["open_url", "openurl", "navigate", "goto", "go_to_url"],
        fields: &[req("url")],
        batchable: false,
        gate: Gate::Always,
        requires_target_name: false,
        prompt_example: r#"{"reason":"brief reason","action":"openUrl","url":"https://example.com"}"#,
    },
    ActionSpec {
        name: "webSearch",
        aliases: &["web_search", "websearch", "search"],
        fields: &[req("query")],
        batchable: false,
        gate: Gate::Always,
        requires_target_name: false,
        prompt_example: r#"{"reason":"brief reason","action":"webSearch","query":"refurbished mac mini"}"#,
    },
    ActionSpec {
        name: "readPage",
        aliases: &["read_page", "readpage", "read"],
        fields: &[],
        batchable: false,
        gate: Gate::Always,
        requires_target_name: false,
        prompt_example: r#"{"reason":"brief reason","action":"readPage"}"#,
    },
    ActionSpec {
        name: "findUi",
        aliases: &[
            "find_ui",
            "findui",
            "search_ui",
            "find_element",
            "findelement",
        ],
        fields: &[req("query")],
        batchable: false,
        gate: Gate::Always,
        requires_target_name: false,
        prompt_example: r#"{"reason":"export control not visible","action":"findUi","query":"export as pdf"}"#,
    },
    ActionSpec {
        name: "webLookup",
        aliases: &["web_lookup", "weblookup", "lookup"],
        fields: &[req("query")],
        batchable: false,
        gate: Gate::WebLookup,
        requires_target_name: false,
        prompt_example: r#"{"reason":"findUi found nothing","action":"webLookup","query":"export note as PDF"}"#,
    },
    ActionSpec {
        name: "ask",
        aliases: &["ask_user", "askuser", "ask_human", "question"],
        fields: &[req("question"), opt("options")],
        batchable: false,
        gate: Gate::Always,
        requires_target_name: false,
        prompt_example: r#"{"reason":"two drafts match","action":"ask","question":"Which draft should I send?","options":["Budget v2","Budget final"]}"#,
    },
    ActionSpec {
        name: "applescript",
        aliases: &[
            "apple_script",
            "osascript",
            "run_applescript",
            "runapplescript",
            "run_script",
        ],
        fields: &[req("script")],
        batchable: false,
        gate: Gate::Scripting,
        requires_target_name: false,
        prompt_example: r#"{"reason":"brief reason","action":"applescript","script":"tell application \"Notes\" to make new note with properties {body:\"hi\"}"}"#,
    },
    ActionSpec {
        name: "shortcut",
        aliases: &["run_shortcut", "runShortcut"],
        fields: &[req("name"), opt("input")],
        batchable: false,
        gate: Gate::Scripting,
        requires_target_name: false,
        prompt_example: r#"{"reason":"brief reason","action":"shortcut","name":"Set Do Not Disturb"}"#,
    },
    ActionSpec {
        name: "moveToTrash",
        aliases: &["move_to_trash", "delete_file", "trash_file", "trash"],
        fields: &[req("file")],
        batchable: false,
        gate: Gate::Scripting,
        requires_target_name: false,
        prompt_example: r#"{"reason":"brief reason","action":"moveToTrash","file":"/Users/me/Desktop/old.dmg"}"#,
    },
    ActionSpec {
        name: "captureFrame",
        aliases: &[
            "capture_frame",
            "captureframe",
            "screenshot",
            "take_screenshot",
            "capture_screen",
        ],
        fields: &[opt("scope")],
        batchable: false,
        gate: Gate::Always,
        requires_target_name: false,
        prompt_example: r#"{"reason":"need to see the chart's colors","action":"captureFrame","scope":"window"}"#,
    },
    ActionSpec {
        name: "recordClip",
        aliases: &["record_clip", "recordclip", "record_video", "record_screen"],
        fields: &[req("seconds"), opt("scope")],
        batchable: false,
        gate: Gate::Always,
        requires_target_name: false,
        prompt_example: r#"{"reason":"user asked for a 10s demo clip","action":"recordClip","seconds":10,"scope":"screen"}"#,
    },
    ActionSpec {
        name: "startRecording",
        aliases: &["start_recording", "startrecording"],
        fields: &[opt("scope")],
        batchable: false,
        gate: Gate::Always,
        requires_target_name: false,
        prompt_example: r#"{"reason":"user asked to record until the task is done","action":"startRecording","scope":"screen"}"#,
    },
    ActionSpec {
        name: "stopRecording",
        aliases: &["stop_recording", "stoprecording"],
        fields: &[],
        batchable: false,
        gate: Gate::Always,
        requires_target_name: false,
        prompt_example: r#"{"reason":"flow finished; save the recording","action":"stopRecording"}"#,
    },
    ActionSpec {
        name: "capturePermission",
        aliases: &["capture_permission", "capturepermission", "check_permissions"],
        fields: &[],
        batchable: false,
        gate: Gate::Always,
        requires_target_name: false,
        prompt_example: r#"{"reason":"capture may be blocked","action":"capturePermission"}"#,
    },
    ActionSpec {
        name: "done",
        aliases: &["finish", "complete", "end", "stop", "terminate"],
        fields: &[],
        batchable: false,
        gate: Gate::Always,
        requires_target_name: false,
        prompt_example: r#"{"reason":"brief reason","action":"done"}"#,
    },
    ActionSpec {
        name: "fail",
        aliases: &[],
        fields: &[req("reason_detail")],
        batchable: false,
        gate: Gate::Always,
        requires_target_name: false,
        prompt_example: r#"{"reason":"brief reason","action":"fail","reason_detail":"..."}"#,
    },
];

pub(crate) fn spec_for(name: &str) -> Option<&'static ActionSpec> {
    ACTIONS.iter().find(|spec| spec.name == name)
}

/// Canonical name for any accepted spelling: exact canonical names pass
/// through; lowercase synonyms map case-insensitively (matching the historic
/// `canonical_action_name` behavior); mixed-case aliases match verbatim.
pub(crate) fn canonical_name_for(raw: &str) -> Option<&'static str> {
    let trimmed = raw.trim();
    if let Some(spec) = spec_for(trimmed) {
        return Some(spec.name);
    }
    let lowered = trimmed.to_ascii_lowercase();
    for spec in ACTIONS {
        for alias in spec.aliases {
            let matches = if alias.chars().any(|c| c.is_ascii_uppercase()) {
                *alias == trimmed
            } else {
                *alias == lowered
            };
            if matches {
                return Some(spec.name);
            }
        }
    }
    None
}

pub(crate) fn model_action_names() -> Vec<&'static str> {
    ACTIONS.iter().map(|spec| spec.name).collect()
}

pub(crate) fn batchable_action_names() -> Vec<&'static str> {
    ACTIONS
        .iter()
        .filter(|spec| spec.batchable)
        .map(|spec| spec.name)
        .collect()
}

pub(crate) fn field_names(spec: &ActionSpec) -> Vec<&'static str> {
    spec.fields.iter().map(|field| field.name).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::planner::{
        action_specific_fields, batchable_followup, build_system_prompt, parse_planner_decision,
        planner_response_schema,
    };
    use crate::agent::types::{CoordinateSpace, Element, ElementSource, Rect};

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

    fn example_observation() -> Vec<Element> {
        vec![element(14, "Add to Bag"), element(9, "Address and Search")]
    }

    fn schema_action_enum() -> Vec<String> {
        planner_response_schema()["properties"]["action"]["enum"]
            .as_array()
            .expect("action enum")
            .iter()
            .map(|value| value.as_str().expect("enum entry").to_string())
            .collect()
    }

    #[test]
    fn registry_names_and_aliases_are_collision_free() {
        let mut seen = std::collections::HashSet::new();
        for spec in ACTIONS {
            assert!(seen.insert(spec.name), "duplicate action name {}", spec.name);
        }
        let mut alias_seen = std::collections::HashSet::new();
        for spec in ACTIONS {
            for alias in spec.aliases {
                assert_ne!(alias, &spec.name, "{}: alias equals its own name", spec.name);
                assert!(
                    alias_seen.insert(*alias),
                    "alias '{alias}' appears in more than one spec"
                );
                // The shortcut->key bug class: an accepted spelling of one
                // action must never be another action's canonical name.
                assert!(
                    spec_for(alias).is_none(),
                    "alias '{alias}' of {} collides with a canonical action name",
                    spec.name
                );
            }
        }
    }

    #[test]
    fn registry_matches_planner_schema_action_enum_in_order() {
        assert_eq!(schema_action_enum(), model_action_names());
    }

    /// Byte-stability pin: the generated enum must keep the exact historical
    /// order, or provider request bodies shift and the Anthropic prompt
    /// cache is busted. Append new actions at the end of ACTIONS only.
    #[test]
    fn schema_action_enum_order_is_pinned() {
        assert_eq!(
            schema_action_enum(),
            [
                "activateApp",
                "click",
                "clickText",
                "doubleClick",
                "type",
                "key",
                "menu",
                "scroll",
                "wait",
                "openUrl",
                "webSearch",
                "readPage",
                "findUi",
                "webLookup",
                "ask",
                "applescript",
                "shortcut",
                "moveToTrash",
                "captureFrame",
                "recordClip",
                "startRecording",
                "stopRecording",
                "capturePermission",
                "done",
                "fail"
            ]
        );
        assert_eq!(
            planner_response_schema()["properties"]["next"]["items"]["properties"]["action"]
                ["enum"],
            serde_json::json!(["click", "type", "key", "scroll", "wait"])
        );
    }

    #[test]
    fn registry_batchables_match_next_enum_and_batchable_followup() {
        let next_enum: Vec<String> = planner_response_schema()["properties"]["next"]["items"]
            ["properties"]["action"]["enum"]
            .as_array()
            .expect("next action enum")
            .iter()
            .map(|value| value.as_str().expect("enum entry").to_string())
            .collect();
        assert_eq!(next_enum, batchable_action_names());

        let obs = example_observation();
        for spec in ACTIONS {
            let decision =
                parse_planner_decision(spec.prompt_example, &obs).unwrap_or_else(|err| {
                    panic!("{}: prompt example failed to parse: {err}", spec.name)
                });
            assert_eq!(
                batchable_followup(&decision.action),
                spec.batchable,
                "{}: batchable flag drifted from batchable_followup",
                spec.name
            );
        }
    }

    #[test]
    fn registry_fields_match_whitelists_and_schema_properties() {
        let schema = planner_response_schema();
        let properties = schema["properties"].as_object().expect("properties");
        for spec in ACTIONS {
            assert_eq!(
                action_specific_fields(spec.name),
                Some(field_names(spec)),
                "{}: registry fields drifted from action_specific_fields",
                spec.name
            );
            for field in spec.fields {
                assert!(
                    properties.contains_key(field.name),
                    "{}: field '{}' missing from the decision schema",
                    spec.name,
                    field.name
                );
            }
        }
    }

    /// The keystone drift test: the exact object the prompt teaches must
    /// parse through the real pipeline and round-trip to the same action
    /// name. A new action cannot ship a prompt example without a parse arm.
    #[test]
    fn prompt_examples_round_trip_through_the_real_parser() {
        let obs = example_observation();
        for spec in ACTIONS {
            let decision =
                parse_planner_decision(spec.prompt_example, &obs).unwrap_or_else(|err| {
                    panic!("{}: prompt example failed to parse: {err}", spec.name)
                });
            let serialized = serde_json::to_value(&decision.action).expect("serialize action");
            assert_eq!(
                serialized["action"].as_str(),
                Some(spec.name),
                "{}: example parsed to a different action",
                spec.name
            );
        }
    }

    /// Every accepted alias must parse to the same action as the canonical
    /// name, end to end (canonicalization + whitelists + parse arm).
    #[test]
    fn alias_swapped_examples_parse_to_the_same_action() {
        let obs = example_observation();
        for spec in ACTIONS {
            let canonical_tag = format!(r#""action":"{}""#, spec.name);
            for alias in spec.aliases {
                assert_eq!(
                    canonical_name_for(alias),
                    Some(spec.name),
                    "alias '{alias}' does not canonicalize to {}",
                    spec.name
                );
                let swapped = spec
                    .prompt_example
                    .replace(&canonical_tag, &format!(r#""action":"{alias}""#));
                assert_ne!(
                    swapped, spec.prompt_example,
                    "{}: example does not contain its own action tag",
                    spec.name
                );
                let decision = parse_planner_decision(&swapped, &obs).unwrap_or_else(|err| {
                    panic!("{}: alias '{alias}' failed to parse: {err}", spec.name)
                });
                let serialized = serde_json::to_value(&decision.action).expect("serialize action");
                assert_eq!(
                    serialized["action"].as_str(),
                    Some(spec.name),
                    "{}: alias '{alias}' parsed to a different action",
                    spec.name
                );
            }
        }
    }

    #[test]
    fn gated_actions_match_prompt_advertisement() {
        let base = build_system_prompt(false, false);
        let full = build_system_prompt(true, true);
        for spec in ACTIONS {
            assert!(
                full.contains(spec.prompt_example),
                "{}: example missing from the fully-enabled prompt",
                spec.name
            );
            let advertised_when_off = base.contains(spec.prompt_example);
            assert_eq!(
                advertised_when_off,
                spec.gate == Gate::Always,
                "{}: prompt advertisement does not match its gate",
                spec.name
            );
        }
        let gated: Vec<&str> = ACTIONS
            .iter()
            .filter(|spec| spec.gate != Gate::Always)
            .map(|spec| spec.name)
            .collect();
        assert_eq!(
            gated,
            vec!["webLookup", "applescript", "shortcut", "moveToTrash"]
        );
    }

    /// Provider drift guard: the OpenAI strict schema is derived mechanically
    /// from the planner schema, so it must cover every registry action and
    /// the target-intent fields. This was a live bug — the previous
    /// hand-maintained copy had 14 of 25 actions and no target_name, making
    /// the OpenAI provider unable to emit a valid click at all.
    #[test]
    fn openai_strict_schema_covers_registry() {
        let schema = crate::ai::decision::openai_strictify(&planner_response_schema());
        let openai_enum: Vec<String> = schema["properties"]["action"]["enum"]
            .as_array()
            .expect("action enum")
            .iter()
            .map(|value| value.as_str().expect("enum entry").to_string())
            .collect();
        assert_eq!(openai_enum, model_action_names());
        let properties = schema["properties"].as_object().expect("properties");
        assert!(properties.contains_key("target_name"));
        assert!(properties.contains_key("target_role"));
        let required: Vec<&str> = schema["required"]
            .as_array()
            .expect("required")
            .iter()
            .filter_map(serde_json::Value::as_str)
            .collect();
        for spec in ACTIONS {
            for field in spec.fields {
                assert!(
                    properties.contains_key(field.name),
                    "{}: field '{}' missing from the OpenAI strict schema",
                    spec.name,
                    field.name
                );
                // Strict mode requires every property; optional ones must be
                // nullable so the model can omit them by emitting null.
                assert!(required.contains(&field.name));
            }
        }
    }
}
