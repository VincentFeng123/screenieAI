//! Role → element-class mapping and actionability rules.
//!
//! The class drives the eid prefix (`ax:B3`), the set-of-marks palette, and
//! mark salience ordering. The table is deliberately closed: anything not
//! listed falls to Generic (when pressable) or Static.

/// Element class, ordered by mark salience (Phase 4 cap ranking).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ElementClass {
    Button,
    TextInput,
    Link,
    Toggle,
    MenuItem,
    Slider,
    Image,
    Generic,
    Static,
}

impl ElementClass {
    /// The eid prefix letter — per-snapshot counters are per-class.
    pub fn prefix(self) -> char {
        match self {
            ElementClass::Button => 'B',
            ElementClass::TextInput => 'T',
            ElementClass::Link => 'L',
            ElementClass::Toggle => 'C',
            ElementClass::MenuItem => 'M',
            ElementClass::Slider => 'S',
            ElementClass::Image => 'I',
            ElementClass::Generic => 'G',
            ElementClass::Static => 'X',
        }
    }

    pub fn from_prefix(prefix: char) -> Option<Self> {
        Some(match prefix {
            'B' => ElementClass::Button,
            'T' => ElementClass::TextInput,
            'L' => ElementClass::Link,
            'C' => ElementClass::Toggle,
            'M' => ElementClass::MenuItem,
            'S' => ElementClass::Slider,
            'I' => ElementClass::Image,
            'G' => ElementClass::Generic,
            'X' => ElementClass::Static,
            _ => return None,
        })
    }

    /// Salience rank for the mark cap: focused elements are handled
    /// separately; among the rest, B > T > L > C > M > S > G > I, X never
    /// marked.
    pub fn salience_rank(self) -> u8 {
        match self {
            ElementClass::Button => 0,
            ElementClass::TextInput => 1,
            ElementClass::Link => 2,
            ElementClass::Toggle => 3,
            ElementClass::MenuItem => 4,
            ElementClass::Slider => 5,
            ElementClass::Generic => 6,
            ElementClass::Image => 7,
            ElementClass::Static => 8,
        }
    }
}

/// Actions that make an otherwise-unclassified element actionable.
const ACTIONABLE_ACTIONS: &[&str] = &["AXPress", "AXConfirm", "AXShowMenu"];

fn role_class(role: &str) -> Option<ElementClass> {
    Some(match role {
        "AXButton" | "AXPopUpButton" | "AXMenuButton" | "AXDisclosureTriangle" => {
            ElementClass::Button
        }
        "AXTextField" | "AXTextArea" | "AXSearchField" | "AXComboBox" => ElementClass::TextInput,
        "AXLink" => ElementClass::Link,
        // Switch subroles (AXSwitch / AXToggle) ride on AXCheckBox.
        "AXCheckBox" | "AXRadioButton" => ElementClass::Toggle,
        "AXMenuItem" | "AXMenuBarItem" => ElementClass::MenuItem,
        "AXSlider" | "AXIncrementor" => ElementClass::Slider,
        "AXImage" => ElementClass::Image,
        _ => return None,
    })
}

/// Classify an element given its AX role and the actions it advertises.
/// Static is strictly the non-actionable rest: any unlisted role carrying an
/// actionable action (press, confirm, show-menu) is Generic — otherwise a
/// menu-bearing AXList would be unmarked and class-filtered out despite
/// being interactive.
pub fn classify(role: &str, actions: &[String]) -> ElementClass {
    if let Some(class) = role_class(role) {
        return class;
    }
    if actions
        .iter()
        .any(|a| ACTIONABLE_ACTIONS.iter().any(|known| known == a))
    {
        ElementClass::Generic
    } else {
        ElementClass::Static
    }
}

/// `actionable = actions ∩ {AXPress, AXConfirm, AXShowMenu} ≠ ∅ or class ∈
/// {B,T,L,C,M,S}`.
pub fn is_actionable(class: ElementClass, actions: &[String]) -> bool {
    matches!(
        class,
        ElementClass::Button
            | ElementClass::TextInput
            | ElementClass::Link
            | ElementClass::Toggle
            | ElementClass::MenuItem
            | ElementClass::Slider
    ) || actions
        .iter()
        .any(|a| ACTIONABLE_ACTIONS.iter().any(|known| known == a))
}

/// Secure text fields (and everything beneath one) get `value` redacted (C6).
pub fn is_secure_field(role: &str, subrole: Option<&str>) -> bool {
    role == "AXSecureTextField" || subrole == Some("AXSecureTextField")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(items: &[&str]) -> Vec<String> {
        items.iter().map(|i| i.to_string()).collect()
    }

    #[test]
    fn class_table_maps_per_spec() {
        for role in [
            "AXButton",
            "AXPopUpButton",
            "AXMenuButton",
            "AXDisclosureTriangle",
        ] {
            assert_eq!(classify(role, &[]), ElementClass::Button, "{role}");
        }
        for role in ["AXTextField", "AXTextArea", "AXSearchField", "AXComboBox"] {
            assert_eq!(classify(role, &[]), ElementClass::TextInput, "{role}");
        }
        assert_eq!(classify("AXLink", &[]), ElementClass::Link);
        for role in ["AXCheckBox", "AXRadioButton"] {
            assert_eq!(classify(role, &[]), ElementClass::Toggle, "{role}");
        }
        for role in ["AXMenuItem", "AXMenuBarItem"] {
            assert_eq!(classify(role, &[]), ElementClass::MenuItem, "{role}");
        }
        for role in ["AXSlider", "AXIncrementor"] {
            assert_eq!(classify(role, &[]), ElementClass::Slider, "{role}");
        }
        assert_eq!(classify("AXImage", &[]), ElementClass::Image);
    }

    #[test]
    fn unknown_roles_split_on_actionable_actions() {
        assert_eq!(classify("AXGroup", &s(&["AXPress"])), ElementClass::Generic);
        assert_eq!(classify("AXGroup", &[]), ElementClass::Static);
        assert_eq!(classify("AXStaticText", &[]), ElementClass::Static);
        // A pressable static text is still Generic by the table's rule.
        assert_eq!(
            classify("AXStaticText", &s(&["AXPress"])),
            ElementClass::Generic
        );
        // Static is the NON-ACTIONABLE rest: menu-bearing and confirmable
        // roles are Generic (Finder's icon-view AXList carries AXShowMenu).
        assert_eq!(
            classify("AXList", &s(&["AXShowMenu"])),
            ElementClass::Generic
        );
        assert_eq!(classify("AXRow", &s(&["AXConfirm"])), ElementClass::Generic);
        // Non-actionable custom actions don't promote.
        assert_eq!(classify("AXRow", &s(&["AXOpen"])), ElementClass::Static);
    }

    #[test]
    fn actionability_rules() {
        assert!(is_actionable(ElementClass::Button, &[]));
        assert!(is_actionable(ElementClass::TextInput, &[]));
        assert!(is_actionable(ElementClass::Slider, &[]));
        assert!(!is_actionable(ElementClass::Static, &[]));
        assert!(!is_actionable(ElementClass::Image, &[]));
        assert!(is_actionable(ElementClass::Image, &s(&["AXPress"])));
        assert!(is_actionable(ElementClass::Generic, &s(&["AXShowMenu"])));
        assert!(is_actionable(ElementClass::Generic, &s(&["AXConfirm"])));
        assert!(!is_actionable(
            ElementClass::Generic,
            &s(&["AXScrollToVisible"])
        ));
    }

    #[test]
    fn prefix_roundtrip() {
        for class in [
            ElementClass::Button,
            ElementClass::TextInput,
            ElementClass::Link,
            ElementClass::Toggle,
            ElementClass::MenuItem,
            ElementClass::Slider,
            ElementClass::Image,
            ElementClass::Generic,
            ElementClass::Static,
        ] {
            assert_eq!(ElementClass::from_prefix(class.prefix()), Some(class));
        }
    }

    #[test]
    fn secure_field_detection() {
        assert!(is_secure_field("AXSecureTextField", None));
        assert!(is_secure_field("AXTextField", Some("AXSecureTextField")));
        assert!(!is_secure_field("AXTextField", None));
    }
}
