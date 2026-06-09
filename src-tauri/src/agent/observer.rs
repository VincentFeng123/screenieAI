use super::types::{
    CoordinateSpace, Element, ElementSource, ObservationError, PlatformElementHandle, Rect,
};

pub(crate) const MAX_OBSERVED_ELEMENTS: usize = 250;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ObservedCandidate {
    pub role: String,
    pub name: String,
    pub value: Option<String>,
    pub bounds: Rect,
    pub enabled: bool,
    pub focused: bool,
    pub selected_text: Option<String>,
    pub platform_handle: Option<PlatformElementHandle>,
}

pub(crate) fn ensure_accessibility_permission_with(
    is_trusted: impl FnOnce() -> bool,
    prompt: impl FnOnce(),
) -> Result<(), ObservationError> {
    if is_trusted() {
        Ok(())
    } else {
        prompt();
        Err(ObservationError::AccessibilityPermissionMissing)
    }
}

pub(crate) fn filter_candidates(candidates: Vec<ObservedCandidate>) -> Vec<Element> {
    candidates
        .into_iter()
        .filter(|candidate| candidate.enabled)
        .filter(|candidate| is_actionable_role(&candidate.role))
        .filter(|candidate| has_observable_bounds(candidate.bounds))
        .take(MAX_OBSERVED_ELEMENTS)
        .enumerate()
        .map(|(idx, candidate)| {
            let element = Element::new(
                idx as u32 + 1,
                candidate.role,
                candidate.name,
                candidate.value,
                candidate.bounds,
                candidate.enabled,
                candidate.focused,
                CoordinateSpace::AxPoints,
                ElementSource::Ax,
            )
            .with_selected_text(candidate.selected_text);
            if let Some(platform_handle) = candidate.platform_handle {
                element.with_platform_handle(platform_handle)
            } else {
                element
            }
        })
        .collect()
}

pub(crate) fn is_actionable_role(role: &str) -> bool {
    matches!(
        role,
        "AXButton"
            | "AXMenuButton"
            | "AXPopUpButton"
            | "AXTextField"
            | "AXTextArea"
            | "AXCheckBox"
            | "AXRadioButton"
            | "AXMenuItem"
            | "AXLink"
            | "AXComboBox"
            | "AXSlider"
    )
}

pub(crate) fn is_structural_container_role(role: &str) -> bool {
    matches!(
        role,
        "AXApplication"
            | "AXWindow"
            | "AXGroup"
            | "AXSplitGroup"
            | "AXScrollArea"
            | "AXLayoutArea"
            | "AXLayoutItem"
            | "AXToolbar"
            | "AXSheet"
            | "AXDrawer"
            | "AXPopover"
            | "AXTabGroup"
            | "AXTable"
            | "AXOutline"
            | "AXRow"
            | "AXColumn"
            | "AXCell"
            | "AXList"
            | "AXBrowser"
            | "AXWebArea"
            | "AXUnknown"
    )
}

fn has_observable_bounds(bounds: Rect) -> bool {
    bounds.x.is_finite()
        && bounds.y.is_finite()
        && bounds.width.is_finite()
        && bounds.height.is_finite()
        && !bounds.is_empty()
}

#[cfg(test)]
mod tests {
    use super::{ensure_accessibility_permission_with, filter_candidates, ObservedCandidate, Rect};
    use crate::agent::types::ObservationError;
    use std::cell::Cell;

    fn candidate(role: &str, enabled: bool, bounds: Rect) -> ObservedCandidate {
        ObservedCandidate {
            role: role.to_string(),
            name: format!("{role} name"),
            value: None,
            bounds,
            enabled,
            focused: false,
            selected_text: None,
            platform_handle: None,
        }
    }

    fn visible_bounds() -> Rect {
        Rect {
            x: 10.0,
            y: 20.0,
            width: 100.0,
            height: 40.0,
        }
    }

    #[test]
    fn agent_observer_filters_roles_and_assigns_stable_ids() {
        let elements = filter_candidates(vec![
            candidate("AXGroup", true, visible_bounds()),
            candidate("AXButton", true, visible_bounds()),
            candidate("AXTextField", true, visible_bounds()),
            candidate("AXButton", false, visible_bounds()),
        ]);

        assert_eq!(elements.len(), 2);
        assert_eq!(elements[0].id, 1);
        assert_eq!(elements[0].role, "AXButton");
        assert_eq!(elements[1].id, 2);
        assert_eq!(elements[1].role, "AXTextField");
    }

    #[test]
    fn agent_observer_drops_empty_bounds_and_preserves_negative_coordinates() {
        let elements = filter_candidates(vec![
            candidate(
                "AXButton",
                true,
                Rect {
                    width: 0.0,
                    ..visible_bounds()
                },
            ),
            candidate(
                "AXButton",
                true,
                Rect {
                    x: -20.0,
                    y: 10.0,
                    width: 10.0,
                    height: 10.0,
                },
            ),
            candidate("AXButton", true, visible_bounds()),
        ]);

        assert_eq!(elements.len(), 2);
        assert_eq!(elements[0].id, 1);
        assert_eq!(
            elements[0].bounds,
            Rect {
                x: -20.0,
                y: 10.0,
                width: 10.0,
                height: 10.0,
            }
        );
        assert_eq!(elements[1].id, 2);
        assert_eq!(elements[1].bounds, visible_bounds());
    }

    #[test]
    fn agent_observer_permission_branch_returns_clear_error_and_prompts() {
        let prompted = Cell::new(false);
        let result = ensure_accessibility_permission_with(
            || false,
            || {
                prompted.set(true);
            },
        );

        assert_eq!(
            result,
            Err(ObservationError::AccessibilityPermissionMissing)
        );
        assert!(prompted.get());
    }
}
