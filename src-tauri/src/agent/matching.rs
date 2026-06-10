//! Fuzzy label matching shared by the target-intent gate (WI-1) and
//! text-targeted resolution (WI-2). Scores are ordered exact > containment >
//! token coverage > token Jaccard > Jaro-Winkler, so a verbatim echo of an
//! observation name always wins and unrelated labels land far below the
//! acceptance threshold.

/// Minimum score at which an expected label is considered to match a
/// candidate element's label.
pub(crate) const TARGET_MATCH_ACCEPT_THRESHOLD: f64 = 0.65;

/// Lowercase, strip punctuation to spaces, collapse whitespace.
pub(crate) fn normalize_label(label: &str) -> String {
    let mut normalized = String::with_capacity(label.len());
    let mut pending_space = false;
    for ch in label.chars() {
        if ch.is_alphanumeric() {
            if pending_space && !normalized.is_empty() {
                normalized.push(' ');
            }
            pending_space = false;
            normalized.extend(ch.to_lowercase());
        } else {
            pending_space = true;
        }
    }
    normalized
}

/// Similarity in [0.0, 1.0] between the label the planner asked for and a
/// candidate element label.
pub(crate) fn label_match_score(expected: &str, candidate: &str) -> f64 {
    let expected = normalize_label(expected);
    let candidate = normalize_label(candidate);
    if expected.is_empty() || candidate.is_empty() {
        return 0.0;
    }
    if expected == candidate {
        return 1.0;
    }

    // Substring containment: a short query inside a longer label (or vice
    // versa) is a strong signal, scaled by how much of the longer label the
    // shorter one covers so near-full overlap outranks a tiny fragment.
    let mut score: f64 = 0.0;
    if expected.contains(&candidate) || candidate.contains(&expected) {
        let shorter = expected.len().min(candidate.len()) as f64;
        let longer = expected.len().max(candidate.len()) as f64;
        score = 0.75 + 0.2 * (shorter / longer);
    }

    let expected_tokens: std::collections::BTreeSet<&str> =
        expected.split_whitespace().collect();
    let candidate_tokens: std::collections::BTreeSet<&str> =
        candidate.split_whitespace().collect();
    let intersection = expected_tokens.intersection(&candidate_tokens).count();
    if intersection > 0 {
        // Full coverage of the expected tokens (the query is a token-subset
        // of the candidate) matches the "name plus suffix" shape of AX
        // labels; partial overlap is scaled down hard so sibling options
        // ("Buy, iPhone 17 Pro" vs the Pro Max radio) stay below threshold.
        if intersection == expected_tokens.len() && expected_tokens.len() >= 2 {
            score = score.max(0.85);
        }
        let union = expected_tokens.union(&candidate_tokens).count();
        let jaccard = intersection as f64 / union as f64;
        score = score.max(jaccard * 0.75);
    }

    score.max(strsim::jaro_winkler(&expected, &candidate) * 0.6)
}

/// Whether a planner-supplied role hint ("link", "AXRadioButton", "radio")
/// names the same role as an observed AX role string.
pub(crate) fn role_matches_hint(hint: &str, actual_role: &str) -> bool {
    let normalize_role = |role: &str| {
        let lowered: String = role
            .chars()
            .filter(|ch| ch.is_ascii_alphanumeric())
            .collect::<String>()
            .to_ascii_lowercase();
        lowered.strip_prefix("ax").map(str::to_string).unwrap_or(lowered)
    };
    let hint = normalize_role(hint);
    let actual = normalize_role(actual_role);
    if hint.is_empty() || actual.is_empty() {
        return false;
    }
    // "radio" matches "radiobutton"; "popup" matches "popupbutton". A bare
    // prefix must be a word-ish stem, so require at least 4 chars to avoid
    // "b" matching everything.
    hint == actual || (hint.len() >= 4 && actual.starts_with(&hint))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_lowercases_strips_punctuation_and_collapses_whitespace() {
        assert_eq!(
            normalize_label("iPhone 17 Pro Max 6.9-inch display"),
            "iphone 17 pro max 6 9 inch display"
        );
        assert_eq!(
            normalize_label("  Connect with a Specialist(Opens in a new window) "),
            "connect with a specialist opens in a new window"
        );
        assert_eq!(normalize_label("Add to Bag"), "add to bag");
        assert_eq!(normalize_label(""), "");
    }

    #[test]
    fn verbatim_echo_scores_exact_match() {
        assert_eq!(
            label_match_score("Buy, iPhone 17 Pro", "Buy, iPhone 17 Pro"),
            1.0
        );
        // Case and punctuation differences still count as exact.
        assert_eq!(
            label_match_score("buy iphone 17 pro", "Buy, iPhone 17 Pro"),
            1.0
        );
    }

    #[test]
    fn query_contained_in_longer_label_passes_threshold() {
        let score = label_match_score(
            "iPhone 17 Pro Max",
            "iPhone 17 Pro Max 6.9-inch display",
        );
        assert!(
            score >= TARGET_MATCH_ACCEPT_THRESHOLD,
            "containment score {score} should pass threshold"
        );
        assert!(score < 1.0);
    }

    #[test]
    fn step_14_mismatch_scores_far_below_threshold() {
        // The failing run's step 14: planner said "iPhone 17 Pro Max
        // 6.9-inch display" while id 29 resolved to the specialist link.
        let score = label_match_score(
            "iPhone 17 Pro Max 6.9-inch display",
            "Connect with a Specialist(Opens in a new window)",
        );
        assert!(
            score < 0.5,
            "unrelated labels must score well below threshold, got {score}"
        );
    }

    #[test]
    fn step_16_mismatch_scores_below_threshold() {
        // The failing run's step 16: intent "iPhone 17 Pro" vs the
        // "Show more Need help choosing a model?" button.
        let score = label_match_score("iPhone 17 Pro", "Show more Need help choosing a model?");
        assert!(score < TARGET_MATCH_ACCEPT_THRESHOLD, "got {score}");
    }

    #[test]
    fn sibling_option_with_partial_overlap_is_rejected() {
        // "Buy, iPhone 17 Pro" shares tokens with the Pro Max radio but is a
        // different element; the gate must not let it through.
        let score = label_match_score(
            "iPhone 17 Pro Max 6.9-inch display",
            "Buy, iPhone 17 Pro",
        );
        assert!(score < TARGET_MATCH_ACCEPT_THRESHOLD, "got {score}");
    }

    #[test]
    fn reordered_tokens_still_match() {
        let score = label_match_score("Save Draft", "Draft Save");
        assert!(score >= TARGET_MATCH_ACCEPT_THRESHOLD, "got {score}");
    }

    #[test]
    fn empty_labels_score_zero() {
        assert_eq!(label_match_score("", "Add to Bag"), 0.0);
        assert_eq!(label_match_score("Add to Bag", ""), 0.0);
        assert_eq!(label_match_score("  ", "   "), 0.0);
    }

    #[test]
    fn role_hint_matches_ax_role_variants() {
        assert!(role_matches_hint("AXRadioButton", "AXRadioButton"));
        assert!(role_matches_hint("radio", "AXRadioButton"));
        assert!(role_matches_hint("radioButton", "AXRadioButton"));
        assert!(role_matches_hint("link", "AXLink"));
        assert!(role_matches_hint("button", "AXButton"));
        assert!(role_matches_hint("checkbox", "AXCheckBox"));
        assert!(!role_matches_hint("button", "AXLink"));
        assert!(!role_matches_hint("radio", "AXButton"));
        assert!(!role_matches_hint("", "AXButton"));
    }
}
