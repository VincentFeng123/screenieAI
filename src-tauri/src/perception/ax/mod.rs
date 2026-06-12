//! macOS Accessibility walker: the `ax:` namespace of the perception layer.
//!
//! Layering: `ffi` is the single unsafe surface, `walker` drives traversal
//! with budgets, `classify` maps roles to element classes. The FFI and the
//! walker are macOS-only; classification is platform-neutral (the Windows
//! UIA port maps its roles onto the same class table).

pub mod classify;
#[cfg(target_os = "macos")]
pub mod ffi;
#[cfg(target_os = "macos")]
pub mod walker;

pub use classify::ElementClass;
#[cfg(target_os = "macos")]
pub use walker::{AxElement, WalkConfig, WalkOutcome};
