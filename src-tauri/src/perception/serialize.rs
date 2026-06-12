//! LLM-facing serialization — the exact line grammar of the tool contract:
//!
//! ```text
//! SNAP ax:k7f3a app=com.apple.Safari win="GitHub — screenieAI" 1512x982pt scale=2 n=347 t=143ms
//! FOCUS ax:T1
//! ax:B1 button "Back" (12,48 28x28) [press]
//! ax:T1 textfield "Address and Search" (120,44 900x36) [press,confirm] val="github.com/Vincent…"
//! ax:G4 group "Sidebar" (0,90 220x860) [press] web⊥
//! … 312 more — call read with role/text/region filters
//! ```
//!
//! Frames are `(x,y wxh)` in points; `val=` appears only when present, ≤40
//! chars after truncation; `web⊥` marks web-content boundaries; the footer
//! appears whenever fewer than all matching elements are shown.

use crate::perception::geometry::RectPt;
use crate::perception::ids::ElementRow;

/// Longest rendered label (title/descr) per line.
const LABEL_CHARS: usize = 60;
/// Longest rendered value per line (the grammar's 40-char rule).
const VALUE_CHARS: usize = 40;
/// Most action names rendered per line.
const ACTIONS_SHOWN: usize = 4;

/// Header fields for the `SNAP` line.
#[derive(Clone, Debug, Default)]
pub struct SnapHeader {
    pub snapshot_id: String,
    /// Bundle id preferred, app name fallback.
    pub app: Option<String>,
    pub window_title: Option<String>,
    /// Window frame in points (only w×h is rendered).
    pub frame: RectPt,
    pub scale: f64,
    /// Total elements in the snapshot (or matching the read's filters).
    pub total: usize,
    pub took_ms: u64,
}

/// Render the SNAP header, FOCUS line, element lines for `rows` (already
/// ranked by the caller), and the truncation footer when `rows` shows fewer
/// than `header.total`.
pub fn serialize_block(header: &SnapHeader, rows: &[ElementRow]) -> String {
    let mut out = String::with_capacity(64 + rows.len() * 80);
    out.push_str(&format!(
        "SNAP {} app={} win=\"{}\" {}x{}pt scale={} n={} t={}ms",
        header.snapshot_id,
        header.app.as_deref().unwrap_or("?"),
        escape_quotes(&truncate(
            header.window_title.as_deref().unwrap_or(""),
            LABEL_CHARS
        )),
        round(header.frame.w),
        round(header.frame.h),
        format_scale(header.scale),
        header.total,
        header.took_ms,
    ));
    if let Some(focused) = rows.iter().find(|row| row.focused) {
        out.push_str(&format!("\nFOCUS {}", focused.eid));
    }
    for row in rows {
        out.push('\n');
        out.push_str(&element_line(row));
    }
    if rows.len() < header.total {
        out.push_str(&format!(
            "\n… {} more — call read with role/text/region filters",
            header.total - rows.len()
        ));
    }
    out
}

/// One element line of the grammar.
pub fn element_line(row: &ElementRow) -> String {
    let mut line = format!(
        "{} {} \"{}\" ({},{} {}x{})",
        row.eid,
        class_word(row),
        escape_quotes(&truncate(
            row.title.as_deref().or(row.descr.as_deref()).unwrap_or(""),
            LABEL_CHARS
        )),
        round(row.frame.x),
        round(row.frame.y),
        round(row.frame.w),
        round(row.frame.h),
    );
    let actions = render_actions(&row.actions);
    if !actions.is_empty() {
        line.push_str(&format!(" [{actions}]"));
    }
    if let Some(value) = row.value.as_deref() {
        let value = truncate(value, VALUE_CHARS);
        if is_bare_value(&value) {
            line.push_str(&format!(" val={value}"));
        } else {
            line.push_str(&format!(" val=\"{}\"", escape_quotes(&value)));
        }
    }
    if !row.enabled {
        line.push_str(" disabled");
    }
    if row.is_web_boundary {
        line.push_str(" web⊥");
    }
    line
}

/// Salience ranking shared by `see`'s top-K and `read`'s ordering when rows
/// arrive unranked: focused first, then actionable by class priority, then
/// larger area, ties broken by tree order.
pub fn rank_rows(rows: &[ElementRow]) -> Vec<&ElementRow> {
    use crate::perception::ax::classify::ElementClass;
    let mut ranked: Vec<(usize, &ElementRow)> = rows.iter().enumerate().collect();
    ranked.sort_by_key(|(index, row)| {
        (
            !row.focused,
            !row.actionable,
            ElementClass::from_prefix(row.class_prefix)
                .map(|class| class.salience_rank())
                .unwrap_or(u8::MAX),
            std::cmp::Reverse(row.frame.area() as i64),
            *index,
        )
    });
    ranked.into_iter().map(|(_, row)| row).collect()
}

/// Display word: AX role minus the `AX` prefix, lowercased (`AXTextField` →
/// `textfield`); class-derived fallback when the role is empty.
fn class_word(row: &ElementRow) -> String {
    let stripped = row.role.strip_prefix("AX").unwrap_or(&row.role);
    if stripped.is_empty() {
        return match row.class_prefix {
            'B' => "button",
            'T' => "textfield",
            'L' => "link",
            'C' => "checkbox",
            'M' => "menuitem",
            'S' => "slider",
            'I' => "image",
            'G' => "group",
            _ => "element",
        }
        .to_string();
    }
    stripped.to_lowercase()
}

fn render_actions(actions: &[String]) -> String {
    actions
        .iter()
        .take(ACTIONS_SHOWN)
        .map(|action| action.strip_prefix("AX").unwrap_or(action).to_lowercase())
        .collect::<Vec<_>>()
        .join(",")
}

fn is_bare_value(value: &str) -> bool {
    !value.is_empty()
        && value
            .chars()
            .all(|ch| ch.is_ascii_digit() || ch == '.' || ch == '-')
}

fn escape_quotes(text: &str) -> String {
    text.replace('"', "\\\"")
}

fn truncate(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        text.to_string()
    } else {
        let mut out: String = text.chars().take(max_chars.saturating_sub(1)).collect();
        out.push('…');
        out
    }
}

fn round(value: f64) -> i64 {
    value.round() as i64
}

fn format_scale(scale: f64) -> String {
    if (scale - scale.round()).abs() < 1e-9 {
        format!("{}", scale.round() as i64)
    } else {
        format!("{scale:.1}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(eid: &str, class: char, role: &str, title: &str) -> ElementRow {
        ElementRow {
            eid: eid.into(),
            fp: 1,
            role: role.into(),
            subrole: None,
            title: (!title.is_empty()).then(|| title.to_string()),
            descr: None,
            value: None,
            actions: Vec::new(),
            actionable: class != 'X' && class != 'G',
            enabled: true,
            focused: false,
            frame: RectPt::new(12.0, 48.0, 28.0, 28.0),
            depth: 1,
            parent_eid: None,
            is_web_boundary: false,
            class_prefix: class,
        }
    }

    #[test]
    fn grammar_matches_contract_examples() {
        let mut back = row("ax:B1", 'B', "AXButton", "Back");
        back.actions = vec!["AXPress".into()];
        assert_eq!(
            element_line(&back),
            "ax:B1 button \"Back\" (12,48 28x28) [press]"
        );

        let mut field = row("ax:T1", 'T', "AXTextField", "Address and Search");
        field.actions = vec!["AXPress".into(), "AXConfirm".into()];
        field.frame = RectPt::new(120.0, 44.0, 900.0, 36.0);
        field.value = Some("github.com/Vincent-Feng-2027/screenieAI/pulls".into());
        let line = element_line(&field);
        assert!(
            line.starts_with(
                "ax:T1 textfield \"Address and Search\" (120,44 900x36) [press,confirm] val=\""
            ),
            "{line}"
        );
        // 40-char value truncation with ellipsis.
        assert!(line.contains('…'), "{line}");

        let mut toggle = row("ax:C1", 'C', "AXCheckBox", "Remember me");
        toggle.actions = vec!["AXPress".into()];
        toggle.frame = RectPt::new(402.0, 310.0, 18.0, 18.0);
        toggle.value = Some("0".into());
        assert_eq!(
            element_line(&toggle),
            "ax:C1 checkbox \"Remember me\" (402,310 18x18) [press] val=0"
        );

        let mut sidebar = row("ax:G4", 'G', "AXGroup", "Sidebar");
        sidebar.actions = vec!["AXPress".into()];
        sidebar.frame = RectPt::new(0.0, 90.0, 220.0, 860.0);
        sidebar.is_web_boundary = true;
        assert_eq!(
            element_line(&sidebar),
            "ax:G4 group \"Sidebar\" (0,90 220x860) [press] web⊥"
        );
    }

    #[test]
    fn block_header_focus_and_footer() {
        let mut focused = row("ax:T1", 'T', "AXTextField", "Search");
        focused.focused = true;
        let rows = vec![row("ax:B1", 'B', "AXButton", "Back"), focused];
        let header = SnapHeader {
            snapshot_id: "ax:k7f3a".into(),
            app: Some("com.apple.Safari".into()),
            window_title: Some("GitHub — screenieAI".into()),
            frame: RectPt::new(0.0, 0.0, 1512.0, 982.0),
            scale: 2.0,
            total: 347,
            took_ms: 143,
        };
        let block = serialize_block(&header, &rows);
        let lines: Vec<&str> = block.lines().collect();
        assert_eq!(
            lines[0],
            "SNAP ax:k7f3a app=com.apple.Safari win=\"GitHub — screenieAI\" 1512x982pt scale=2 n=347 t=143ms"
        );
        assert_eq!(lines[1], "FOCUS ax:T1");
        assert_eq!(
            *lines.last().unwrap(),
            "… 345 more — call read with role/text/region filters"
        );
        // No footer when everything is shown.
        let complete = serialize_block(
            &SnapHeader {
                total: 2,
                ..header.clone()
            },
            &rows,
        );
        assert!(!complete.contains("more — call read"));
    }

    #[test]
    fn ranking_orders_focus_class_area() {
        let mut rows = vec![
            row("ax:G1", 'G', "AXGroup", "g"),
            row("ax:B1", 'B', "AXButton", "small"),
            row("ax:B2", 'B', "AXButton", "big"),
            row("ax:X1", 'X', "AXStaticText", "text"),
            row("ax:T1", 'T', "AXTextField", "field"),
        ];
        rows[2].frame = RectPt::new(0.0, 0.0, 400.0, 40.0);
        rows[4].focused = true;
        let ranked: Vec<&str> = rank_rows(&rows)
            .iter()
            .map(|row| row.eid.as_str())
            .collect();
        assert_eq!(ranked, ["ax:T1", "ax:B2", "ax:B1", "ax:G1", "ax:X1"]);
    }

    #[test]
    fn disabled_and_quote_escaping() {
        let mut quoted = row("ax:B1", 'B', "AXButton", "Say \"hi\"");
        quoted.enabled = false;
        let line = element_line(&quoted);
        assert_eq!(
            line,
            "ax:B1 button \"Say \\\"hi\\\"\" (12,48 28x28) disabled"
        );
    }
}
