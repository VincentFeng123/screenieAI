//! macOS Accessibility walker: the `ax:` namespace of the perception layer.
//!
//! Layering: `ffi` is the single unsafe surface, `walker` drives traversal
//! with budgets, `classify` maps roles to element classes. Everything is
//! macOS-only; the Windows/UIA port is a separate namespace-equal module.

#![cfg(target_os = "macos")]

pub mod classify;
pub mod ffi;
pub mod walker;

pub use classify::ElementClass;
pub use walker::{AxElement, WalkConfig, WalkOutcome};
