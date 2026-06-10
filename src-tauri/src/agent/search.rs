//! Fuzzy matching for the `findUi` action: scores candidate strings (menu
//! item titles, hint feature phrases, element names) against a short feature
//! query. Pure and cross-platform so the ranking is unit-testable everywhere.

/// Normalize for matching: trim, drop trailing ellipses, lowercase. Mirrors
/// the menu-title normalization used by the macOS menu press path.
pub(crate) fn normalize_for_match(text: &str) -> String {
    text.trim()
        .trim_end_matches('…')
        .trim_end_matches("...")
        .trim()
        .to_lowercase()
}

/// Score a candidate against the query. Higher is better; `None` means no
/// match. Tiers: exact (400) > prefix (300) > query-is-substring (200) >
/// every query token appears somewhere in the candidate (100). Within a
/// tier, shorter candidates rank higher (tighter match).
pub(crate) fn score_match(candidate: &str, query: &str) -> Option<u32> {
    let candidate_norm = normalize_for_match(candidate);
    let query_norm = normalize_for_match(query);
    if candidate_norm.is_empty() || query_norm.is_empty() {
        return None;
    }
    let tier = if candidate_norm == query_norm {
        400
    } else if candidate_norm.starts_with(&query_norm) {
        300
    } else if candidate_norm.contains(&query_norm) {
        200
    } else {
        let tokens: Vec<&str> = query_norm.split_whitespace().collect();
        if !tokens.is_empty() && tokens.iter().all(|token| candidate_norm.contains(token)) {
            100
        } else {
            return None;
        }
    };
    // Tie-break inside a tier: penalize long candidates, max penalty 99 so
    // tiers never overlap.
    let penalty = (candidate_norm.len() as u32).min(99);
    Some(tier + 99 - penalty)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match_outranks_prefix_and_substring() {
        let exact = score_match("Export as PDF…", "export as pdf").unwrap();
        let prefix = score_match("Export as PDF to Folder", "export as pdf").unwrap();
        let substring = score_match("Quick Export as PDF", "export as pdf").unwrap();
        assert!(exact > prefix, "{exact} vs {prefix}");
        assert!(prefix > substring, "{prefix} vs {substring}");
    }

    #[test]
    fn all_tokens_match_when_scattered() {
        let scattered = score_match("Export Notes as a PDF Document", "pdf export").unwrap();
        let substring = score_match("Bulk pdf export", "pdf export").unwrap();
        assert!(substring > scattered);
    }

    #[test]
    fn no_match_when_a_token_is_missing() {
        assert_eq!(score_match("Export as HTML", "export pdf"), None);
        assert_eq!(score_match("", "export"), None);
        assert_eq!(score_match("Export", "   "), None);
    }

    #[test]
    fn normalization_handles_case_and_ellipsis() {
        assert_eq!(
            score_match("Settings…", "settings"),
            score_match("settings", "settings")
        );
        assert!(score_match("PRINT", "print").is_some());
    }

    #[test]
    fn shorter_candidate_wins_within_a_tier() {
        let short = score_match("Export PDF", "export").unwrap();
        let long = score_match("Export Everything To Some Other Place", "export").unwrap();
        assert!(short > long);
    }
}
