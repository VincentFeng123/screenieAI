mod actions;
mod capture_tools;
mod executor;
mod grounding;
mod hints;
#[cfg(target_os = "macos")]
mod macos_observer;
mod matching;
mod observer;
#[cfg(target_os = "macos")]
mod page_reader;
mod planner;
mod playbooks;
#[cfg(target_os = "macos")]
mod safari_dom;
mod search;
mod types;
mod vision;

pub(crate) use executor::run_stub_agent_loop_with_grounder;
#[cfg(target_os = "macos")]
pub(crate) use executor::EnigoBackendFactory;
pub use executor::{
    element_to_click_point, ActionMechanism, AgentAbortState, AgentConfirmationRequest,
    AgentQuestionRequest, AgentRunReport, AgentRunStatus, AgentStepReport, CalibrationReport,
    ClickPoint, ClickPreflightReport, ClickPreflightStatus, ConfirmationOutcome,
    ConfirmationRequester, ConfirmationStatus, ExecutionPolicy, ResolvedStubAgentOptions,
    SafetyDecision, SafetyGateReport, SettleStatus, StubAgentOptions, TargetSummary,
    UserAnswerOutcome, UserAnswerStatus, VerificationReport, VerificationStatus,
};
pub(crate) use capture_tools::CaptureEngine;
pub(crate) use grounding::GrounderManager;
pub use grounding::{GroundingMode, GroundingPixel, GroundingReport};
#[cfg(target_os = "macos")]
pub use macos_observer::MacObserver;
pub(crate) use planner::ContextAwareLlmPlanner;
pub use playbooks::PlaybookMeta;
pub(crate) use playbooks::PlaybookStore;
pub use planner::{NameResolvingStubPlanner, StubPlanner};
pub use types::{
    Action, CaptureScope, CoordinateSpace, Element, ElementSource, FocusedApp,
    FocusedAppProvider, MenuMatch, MenuPressOutcome, MenuScanResult, ObservationError, Planner,
    PlannerDecision, PlannerHistoryEntry, Rect, ScreenObserver, ScrollContext,
};
pub use vision::{CaptureSize, ObservationMetadata, ObservationSource};
pub(crate) use vision::{VisionFallbackObserver, VisionFallbackOptions, VisionFallbackState};
