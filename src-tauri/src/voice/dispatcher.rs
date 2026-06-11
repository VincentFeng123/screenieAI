//! Transcript routing: kill-word check -> spoken-artifact normalization ->
//! serial FIFO dispatch into the EXISTING agent runtime.
//!
//! The dispatcher thread runs the exact code path the typed input uses — the
//! private lib.rs helpers wrapped by `start_agent_task`
//! (`agent_task_options_from_goal` -> `prepare_stub_agent_run` ->
//! `run_prepared_stub_agent`) — one task at a time, blocking on each run.
//! The planner contract, executor, and safety gate are reused byte-for-byte;
//! kill words go through `crate::abort_agent_runs`, the same switch the Stop
//! button and the Cmd+Alt+Esc hotkey trip.

use std::collections::VecDeque;
use std::sync::{Condvar, Mutex};
use std::sync::Arc;

use tauri::AppHandle;

use super::{emit_voice, AgentRunSettings};
use crate::lock_poison_safe;

/// Queue cap; overflow is rejected with a `voice:error` (code `queue_full`).
pub const QUEUE_CAP: usize = 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KillKind {
    /// Abort the running task and clear the queue; keep listening.
    Kill,
    /// Same, then end the listening session ("stop listening").
    KillAndStopListening,
}

/// Case-insensitive whole-utterance kill-word match, tolerant of surrounding
/// whitespace and a single trailing `.`/`!`/`?` (whisper likes to punctuate).
pub fn kill_word(raw: &str) -> Option<KillKind> {
    let trimmed = raw.trim();
    let without_punct = trimmed
        .strip_suffix(['.', '!', '?'])
        .unwrap_or(trimmed)
        .trim_end();
    let lowered = without_punct.to_lowercase();
    let collapsed = lowered.split_whitespace().collect::<Vec<_>>().join(" ");
    match collapsed.as_str() {
        "stop" | "cancel" | "abort" | "kill it" => Some(KillKind::Kill),
        "stop listening" => Some(KillKind::KillAndStopListening),
        _ => None,
    }
}

/// Normalize spoken artifacts so transcripts type real values:
/// 1. a standalone "dot" between two words fuses them with a "."
///    ("amazon dot com" -> "amazon.com"); a leading "dot word" becomes
///    ".word".
/// 2. a leading "please " is stripped.
/// 3. ONE trailing "." is stripped when the result is < 6 words (short
///    commands shouldn't end in a period; real sentences keep theirs).
pub fn normalize_transcript(raw: &str) -> String {
    let tokens: Vec<&str> = raw.split_whitespace().collect();

    // 1. Fuse spoken "dot" with its neighbors. Chains fuse repeatedly
    //    because the fused token stays at the tail of `out`.
    let mut out: Vec<String> = Vec::with_capacity(tokens.len());
    let mut i = 0;
    while i < tokens.len() {
        let token = tokens[i];
        if token.eq_ignore_ascii_case("dot") && i + 1 < tokens.len() {
            let next = tokens[i + 1];
            match out.last_mut() {
                Some(prev) => {
                    prev.push('.');
                    prev.push_str(next);
                }
                None => out.push(format!(".{next}")),
            }
            i += 2;
            continue;
        }
        out.push(token.to_string());
        i += 1;
    }

    // 2. Strip a single leading "please" (whole word only).
    if out.first().is_some_and(|t| t.eq_ignore_ascii_case("please")) {
        out.remove(0);
    }

    // 3. Short commands lose ONE trailing period; sentences keep theirs.
    let mut text = out.join(" ");
    if out.len() < 6 {
        if let Some(stripped) = text.strip_suffix('.') {
            text.truncate(stripped.len());
        }
    }
    text
}

#[derive(Debug)]
pub struct VoiceTask {
    pub id: String,
    /// Post-normalization command text — becomes the agent goal verbatim.
    pub text: String,
    pub settings: AgentRunSettings,
}

/// Queue state guarded by ONE mutex: `paused` must be observed under the
/// same lock the condvar waits on, or a resume could slip between the
/// worker's pause check and its wait (lost wakeup).
struct QueueInner {
    tasks: VecDeque<VoiceTask>,
    paused: bool,
}

/// Shared between the STT thread (producer via [`route`]) and the
/// long-lived dispatcher thread (consumer). Outlives listening sessions.
pub struct DispatchShared {
    inner: Mutex<QueueInner>,
    wake: Condvar,
    /// Id of the task currently inside an agent run, for kill events.
    running_id: Mutex<Option<String>>,
    /// Bumped (under the queue lock) by every kill. The worker snapshots it
    /// when popping and re-checks after `prepare_stub_agent_run` — which
    /// RESETS the shared abort flag — so a kill landing between pop and
    /// reset can't be silently erased.
    kill_generation: std::sync::atomic::AtomicU64,
}

impl DispatchShared {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(QueueInner {
                tasks: VecDeque::new(),
                paused: false,
            }),
            wake: Condvar::new(),
            running_id: Mutex::new(None),
            kill_generation: std::sync::atomic::AtomicU64::new(0),
        }
    }

    pub fn kill_generation(&self) -> u64 {
        self.kill_generation.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Kill path: bumps the generation and empties the queue atomically
    /// (same lock), returning the dropped tasks in order. Deliberately
    /// ignores pause — kill words must always work.
    pub fn kill_drain(&self) -> Vec<VoiceTask> {
        let mut inner = lock_poison_safe(&self.inner);
        self.kill_generation
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        inner.tasks.drain(..).collect()
    }

    /// Blocks until a task is available AND dispatch is not paused; returns
    /// it with the kill generation sampled under the same lock acquisition.
    fn pop_blocking(&self) -> (VoiceTask, u64) {
        let mut inner = lock_poison_safe(&self.inner);
        loop {
            if !inner.paused {
                if let Some(task) = inner.tasks.pop_front() {
                    let generation = self
                        .kill_generation
                        .load(std::sync::atomic::Ordering::SeqCst);
                    return (task, generation);
                }
            }
            inner = condvar_wait(&self.wake, inner);
        }
    }

    /// FIFO push, capped at [`QUEUE_CAP`]; the task is handed back on
    /// overflow so the caller can report exactly which utterance dropped.
    /// Pushing while paused still queues — only dispatch is held back.
    pub fn try_push(&self, task: VoiceTask) -> Result<(), VoiceTask> {
        let mut inner = lock_poison_safe(&self.inner);
        if inner.tasks.len() >= QUEUE_CAP {
            return Err(task);
        }
        inner.tasks.push_back(task);
        drop(inner);
        self.wake.notify_one();
        Ok(())
    }

    /// Pauses/resumes dispatch. The task currently inside an agent run is
    /// untouched — pause only stops the worker from popping the next one.
    pub fn set_paused(&self, paused: bool) {
        let mut inner = lock_poison_safe(&self.inner);
        inner.paused = paused;
        drop(inner);
        if !paused {
            self.wake.notify_all();
        }
    }

    pub fn is_paused(&self) -> bool {
        lock_poison_safe(&self.inner).paused
    }

    /// Drops every QUEUED task and returns them in order. Unlike
    /// [`kill_drain`] this neither bumps the kill generation nor aborts the
    /// running task — it is the UI's "clear queued" button, not a kill.
    pub fn clear_queued(&self) -> Vec<VoiceTask> {
        lock_poison_safe(&self.inner).tasks.drain(..).collect()
    }

    /// Removes one queued task by id. `None` means it already popped (or
    /// never existed) — the caller should tell the user it's too late.
    pub fn remove_queued(&self, id: &str) -> Option<VoiceTask> {
        let mut inner = lock_poison_safe(&self.inner);
        let idx = inner.tasks.iter().position(|t| t.id == id)?;
        inner.tasks.remove(idx)
    }

    /// Rewrites the text of one queued task. False means it already popped
    /// (or never existed).
    pub fn edit_queued(&self, id: &str, text: &str) -> bool {
        let mut inner = lock_poison_safe(&self.inner);
        match inner.tasks.iter_mut().find(|t| t.id == id) {
            Some(task) => {
                task.text = text.to_string();
                true
            }
            None => false,
        }
    }

    pub fn running_task_id(&self) -> Option<String> {
        lock_poison_safe(&self.running_id).clone()
    }
}

fn task_event(app: &AppHandle, id: &str, state: &'static str) {
    emit_voice(app, "voice:task", serde_json::json!({ "id": id, "state": state }));
}

/// Routes one surviving transcript, in spec order: kill-word check first
/// (raw text, before normalization), then normalize, then enqueue. Called
/// inline on the STT thread — cheap string work only.
pub fn route(
    app: &AppHandle,
    shared: &Arc<DispatchShared>,
    settings: &AgentRunSettings,
    utterance_id: String,
    raw: String,
) {
    if let Some(kind) = kill_word(&raw) {
        // Transcripts are user speech: log the verdict, never the words.
        eprintln!("[screenie] voice kill word detected ({kind:?})");
        // The existing global kill switch: trips agent_abort, which the
        // executor observes mid-step AND mid-confirmation-wait.
        crate::abort_agent_runs(app);
        if let Some(running) = shared.running_task_id() {
            task_event(app, &running, "killed");
        }
        for task in &shared.kill_drain() {
            task_event(app, &task.id, "killed");
        }
        if kind == KillKind::KillAndStopListening {
            let state = tauri::Manager::state::<crate::AppState>(app);
            let taken = lock_poison_safe(&state.voice_session).take();
            if let Some(handle) = taken {
                super::stop_session(app, &handle);
            }
        }
        return;
    }

    let text = normalize_transcript(&raw);
    if text.is_empty() {
        return;
    }
    emit_voice(
        app,
        "voice:utterance",
        serde_json::json!({ "id": utterance_id, "text": text }),
    );

    let task = VoiceTask {
        id: utterance_id.clone(),
        text,
        settings: settings.clone(),
    };
    match shared.try_push(task) {
        Ok(()) => task_event(app, &utterance_id, "queued"),
        Err(rejected) => {
            emit_voice(
                app,
                "voice:error",
                serde_json::json!({
                    "code": "queue_full",
                    "id": rejected.id,
                    "message": format!("Voice task queue is full ({QUEUE_CAP}); command dropped"),
                }),
            );
            task_event(app, &rejected.id, "failed");
        }
    }
}

/// Long-lived worker: pops ONE task at a time and blocks on the full agent
/// run before popping the next — voice's strict serial FIFO. Spawned once at
/// first mic-on and kept for the process lifetime (it owns no audio).
pub fn spawn_dispatcher(
    app: AppHandle,
    shared: Arc<DispatchShared>,
) -> Result<std::thread::JoinHandle<()>, String> {
    std::thread::Builder::new()
        .name("screenie-voice-dispatcher".into())
        .spawn(move || loop {
            let (task, kill_generation) = shared.pop_blocking();
            *lock_poison_safe(&shared.running_id) = Some(task.id.clone());
            task_event(&app, &task.id, "running");
            let outcome = run_agent_task(&app, &task, &shared, kill_generation);
            // A kill may already have emitted "killed" via route(); emitting
            // the terminal state again with the same value is harmless, and
            // mapping Aborted -> "killed" keeps the two paths consistent.
            task_event(&app, &task.id, outcome);
            *lock_poison_safe(&shared.running_id) = None;
        })
        .map_err(|e| format!("spawn voice dispatcher: {e}"))
}

fn condvar_wait<'a, T>(
    cv: &Condvar,
    guard: std::sync::MutexGuard<'a, T>,
) -> std::sync::MutexGuard<'a, T> {
    match cv.wait(guard) {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Runs one voice task through the typed-input code path and maps the
/// outcome onto the voice:task state vocabulary. `kill_generation` is the
/// snapshot taken when the task was popped — if it advances before the run
/// starts, a kill word landed in the pop window and the task must die even
/// though `prepare_stub_agent_run` reset the shared abort flag.
#[cfg(target_os = "macos")]
fn run_agent_task(
    app: &AppHandle,
    task: &VoiceTask,
    shared: &Arc<DispatchShared>,
    kill_generation: u64,
) -> &'static str {
    use std::sync::atomic::{AtomicBool, Ordering};
    use tauri::{Emitter, Listener, Manager};

    let settings = &task.settings;
    let Some(mut options) = crate::agent_task_options_from_goal(
        &task.text,
        settings.provider.clone(),
        settings.model.clone(),
        settings.vision_provider.clone(),
        settings.vision_model.clone(),
        settings.autonomy.clone(),
        settings.scripting_enabled,
        settings.web_lookup_enabled,
    ) else {
        return "failed";
    };

    let fail = |msg: String| {
        eprintln!("[screenie] voice task start failed: {msg}");
        emit_voice(
            app,
            "voice:error",
            serde_json::json!({ "code": "agent", "message": msg }),
        );
    };
    if let Err(e) = crate::request_agent_permissions(app) {
        fail(e);
        return "failed";
    }
    if let Err(e) = crate::agent_capture_health() {
        fail(e);
        return "failed";
    }
    let Some(window) = app.get_webview_window(super::VOICE_WINDOW) else {
        fail("quick_tooltip window missing".into());
        return "failed";
    };
    // Same as the typed path: release keyboard routing so the agent types
    // into the target app, not the tooltip.
    let _ = crate::set_quick_tooltip_keyboard_mode_on_main(&window, false, false);
    let (text_config, vision_config, abort) =
        match crate::prepare_stub_agent_run(app, &mut options) {
            Ok(v) => v,
            Err(e) => {
                fail(e);
                return "failed";
            }
        };
    // prepare_stub_agent_run just RESET the shared abort flag; if a kill
    // word arrived after this task was popped, honor it now instead of
    // letting the reset swallow it.
    if shared.kill_generation() != kill_generation {
        return "killed";
    }
    // Same settle the typed path takes before its run (lib.rs thread body):
    // lets the keyboard-mode/focus toggle land before the agent acts.
    std::thread::sleep(std::time::Duration::from_millis(180));

    // blocked_on_confirmation tracking without touching the safety gate:
    // the confirmation requester broadcasts `agent-confirmation-requested`,
    // and any subsequent step update means the gate resolved.
    let blocked = Arc::new(AtomicBool::new(false));
    let confirmation_listener = {
        let app = app.clone();
        let id = task.id.clone();
        let blocked = blocked.clone();
        move |_event: tauri::Event| {
            blocked.store(true, Ordering::Relaxed);
            task_event(&app, &id, "blocked_on_confirmation");
        }
    };
    let step_listener = {
        let app = app.clone();
        let id = task.id.clone();
        let blocked = blocked.clone();
        move |_event: tauri::Event| {
            if blocked.swap(false, Ordering::Relaxed) {
                task_event(&app, &id, "running");
            }
        }
    };
    let confirmation_id = app.listen_any("agent-confirmation-requested", confirmation_listener);
    let step_id = app.listen_any("agent-step-update", step_listener);

    let panic_report_options = options.clone().resolve();
    let report = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        tauri::async_runtime::block_on(crate::run_prepared_stub_agent(
            app.clone(),
            window.clone(),
            options,
            text_config,
            vision_config,
            abort,
        ))
    })) {
        Ok(report) => report,
        Err(payload) => {
            let reason = format!(
                "agent task panicked: {}",
                crate::panic_payload_message(payload.as_ref())
            );
            eprintln!("[screenie] {reason}");
            crate::agent::AgentRunReport {
                status: crate::agent::AgentRunStatus::Failed,
                options: panic_report_options,
                steps: Vec::new(),
                failure_reason: Some(reason),
            }
        }
    };

    app.unlisten(confirmation_id);
    app.unlisten(step_id);

    // Mirror the typed path's completion event so the existing tooltip UI
    // treats voice runs identically (lib.rs start_agent_task thread tail).
    if let Err(e) = window.emit("agent-task-finished", report.clone()) {
        eprintln!("[screenie] emit agent task finished failed: {e}");
    }
    eprintln!(
        "[screenie] voice agent task finished status={:?} steps={} failure={:?}",
        report.status,
        report.steps.len(),
        report.failure_reason
    );
    match report.status {
        crate::agent::AgentRunStatus::Done => "done",
        crate::agent::AgentRunStatus::Aborted => "killed",
        _ => "failed",
    }
}

#[cfg(not(target_os = "macos"))]
fn run_agent_task(
    app: &AppHandle,
    _task: &VoiceTask,
    _shared: &Arc<DispatchShared>,
    _kill_generation: u64,
) -> &'static str {
    emit_voice(
        app,
        "voice:error",
        serde_json::json!({
            "code": "agent",
            "message": crate::agent::ObservationError::UnsupportedPlatform.to_string(),
        }),
    );
    "failed"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kill_words_match_whole_utterance_case_insensitively() {
        assert_eq!(kill_word("stop"), Some(KillKind::Kill));
        assert_eq!(kill_word("Stop."), Some(KillKind::Kill));
        assert_eq!(kill_word(" CANCEL "), Some(KillKind::Kill));
        assert_eq!(kill_word("abort!"), Some(KillKind::Kill));
        assert_eq!(kill_word("kill it"), Some(KillKind::Kill));
        assert_eq!(kill_word("Kill it?"), Some(KillKind::Kill));
        assert_eq!(
            kill_word("stop listening"),
            Some(KillKind::KillAndStopListening)
        );
        assert_eq!(
            kill_word("Stop Listening."),
            Some(KillKind::KillAndStopListening)
        );
    }

    #[test]
    fn kill_words_do_not_match_inside_sentences() {
        assert_eq!(kill_word("stop the music"), None);
        assert_eq!(kill_word("please stop"), None);
        assert_eq!(kill_word("cancel the subscription"), None);
        assert_eq!(kill_word("abort the mission later"), None);
        assert_eq!(kill_word(""), None);
    }

    #[test]
    fn dot_between_words_fuses_them() {
        assert_eq!(normalize_transcript("type amazon dot com"), "type amazon.com");
        assert_eq!(
            normalize_transcript("go to google dot com and search"),
            "go to google.com and search"
        );
        // Chained dots fuse repeatedly.
        assert_eq!(
            normalize_transcript("open docs dot rs dot com please now yes"),
            "open docs.rs.com please now yes"
        );
    }

    #[test]
    fn dot_handles_capitalization_and_punctuation() {
        assert_eq!(normalize_transcript("type Amazon Dot Com"), "type Amazon.Com");
        // Whisper-style trailing period on the fused token, short command:
        assert_eq!(normalize_transcript("type amazon dot com."), "type amazon.com");
    }

    #[test]
    fn leading_dot_fuses_with_following_word() {
        assert_eq!(normalize_transcript("dot com"), ".com");
    }

    #[test]
    fn standalone_or_trailing_dot_is_left_alone() {
        assert_eq!(normalize_transcript("dot"), "dot");
        assert_eq!(normalize_transcript("connect the dot"), "connect the dot");
    }

    #[test]
    fn leading_please_is_stripped() {
        assert_eq!(normalize_transcript("please open safari"), "open safari");
        assert_eq!(normalize_transcript("Please open Safari"), "open Safari");
        // Only the leading one, and only as a whole word.
        assert_eq!(
            normalize_transcript("pleasebot run now then stop again ok"),
            "pleasebot run now then stop again ok"
        );
    }

    #[test]
    fn short_commands_lose_one_trailing_period() {
        assert_eq!(normalize_transcript("Open a new tab."), "Open a new tab");
        assert_eq!(normalize_transcript("Open safari.."), "Open safari.");
    }

    #[test]
    fn long_sentences_keep_their_trailing_period() {
        assert_eq!(
            normalize_transcript("open the settings panel and check for updates."),
            "open the settings panel and check for updates."
        );
    }

    #[test]
    fn whitespace_is_trimmed_and_collapsed() {
        assert_eq!(normalize_transcript("  open   safari  "), "open safari");
    }

    fn task(id: &str) -> VoiceTask {
        VoiceTask {
            id: id.into(),
            text: format!("task {id}"),
            settings: AgentRunSettings::default(),
        }
    }

    #[test]
    fn queue_preserves_fifo_order() {
        let shared = DispatchShared::new();
        shared.try_push(task("a")).unwrap();
        shared.try_push(task("b")).unwrap();
        shared.try_push(task("c")).unwrap();
        let order: Vec<String> = shared.kill_drain().into_iter().map(|t| t.id).collect();
        assert_eq!(order, ["a", "b", "c"]);
    }

    #[test]
    fn queue_rejects_overflow_beyond_cap() {
        let shared = DispatchShared::new();
        for i in 0..QUEUE_CAP {
            shared.try_push(task(&i.to_string())).unwrap();
        }
        let rejected = shared.try_push(task("overflow")).unwrap_err();
        assert_eq!(rejected.id, "overflow");
        assert_eq!(shared.kill_drain().len(), QUEUE_CAP);
    }

    #[test]
    fn kill_drain_leaves_the_queue_reusable() {
        let shared = DispatchShared::new();
        shared.try_push(task("a")).unwrap();
        assert_eq!(shared.kill_drain().len(), 1);
        assert!(shared.kill_drain().is_empty());
        shared.try_push(task("b")).unwrap();
        assert_eq!(shared.kill_drain().len(), 1);
    }

    #[test]
    fn running_id_starts_empty() {
        assert_eq!(DispatchShared::new().running_task_id(), None);
    }

    #[test]
    fn kill_drain_bumps_generation_and_empties_queue() {
        let shared = DispatchShared::new();
        shared.try_push(task("a")).unwrap();
        shared.try_push(task("b")).unwrap();
        let before = shared.kill_generation();
        let drained = shared.kill_drain();
        assert_eq!(drained.len(), 2);
        assert_eq!(shared.kill_generation(), before + 1);
        assert!(shared.kill_drain().is_empty());
    }

    #[test]
    fn pause_blocks_pop_until_resume() {
        use std::sync::mpsc;
        use std::time::Duration;

        let shared = Arc::new(DispatchShared::new());
        shared.set_paused(true);
        shared.try_push(task("a")).unwrap();

        let (tx, rx) = mpsc::channel();
        let worker = {
            let shared = shared.clone();
            std::thread::spawn(move || {
                let (popped, _) = shared.pop_blocking();
                tx.send(popped.id).unwrap();
            })
        };
        // Paused: the worker must NOT pop even though a task is queued.
        assert!(rx.recv_timeout(Duration::from_millis(150)).is_err());

        shared.set_paused(false);
        assert_eq!(rx.recv_timeout(Duration::from_secs(5)).unwrap(), "a");
        worker.join().unwrap();
    }

    #[test]
    fn push_while_paused_still_queues() {
        let shared = DispatchShared::new();
        shared.set_paused(true);
        shared.try_push(task("a")).unwrap();
        assert!(shared.is_paused());
        assert_eq!(shared.clear_queued().len(), 1);
    }

    #[test]
    fn clear_queued_does_not_bump_generation() {
        let shared = DispatchShared::new();
        shared.try_push(task("a")).unwrap();
        shared.try_push(task("b")).unwrap();
        let before = shared.kill_generation();
        let cleared: Vec<String> = shared.clear_queued().into_iter().map(|t| t.id).collect();
        assert_eq!(cleared, ["a", "b"]);
        assert_eq!(shared.kill_generation(), before);
        assert!(shared.clear_queued().is_empty());
    }

    #[test]
    fn remove_queued_hits_only_the_given_id() {
        let shared = DispatchShared::new();
        shared.try_push(task("a")).unwrap();
        shared.try_push(task("b")).unwrap();
        shared.try_push(task("c")).unwrap();
        assert_eq!(shared.remove_queued("b").map(|t| t.id), Some("b".into()));
        assert!(shared.remove_queued("b").is_none());
        assert!(shared.remove_queued("nope").is_none());
        let order: Vec<String> = shared.clear_queued().into_iter().map(|t| t.id).collect();
        assert_eq!(order, ["a", "c"]);
    }

    #[test]
    fn edit_queued_rewrites_text_for_queued_ids_only() {
        let shared = DispatchShared::new();
        shared.try_push(task("a")).unwrap();
        assert!(shared.edit_queued("a", "new text"));
        assert!(!shared.edit_queued("gone", "x"));
        let drained = shared.clear_queued();
        assert_eq!(drained[0].text, "new text");
    }

    #[test]
    fn kill_drain_works_while_paused() {
        let shared = DispatchShared::new();
        shared.set_paused(true);
        shared.try_push(task("a")).unwrap();
        let before = shared.kill_generation();
        assert_eq!(shared.kill_drain().len(), 1);
        assert_eq!(shared.kill_generation(), before + 1);
        assert!(shared.is_paused(), "kill must not silently resume dispatch");
    }

    #[test]
    fn pop_snapshots_the_current_generation() {
        let shared = DispatchShared::new();
        shared.kill_drain(); // generation 1
        shared.try_push(task("a")).unwrap();
        let (popped, generation) = shared.pop_blocking();
        assert_eq!(popped.id, "a");
        assert_eq!(generation, shared.kill_generation());
        // A kill after the pop advances the generation past the snapshot.
        shared.kill_drain();
        assert_ne!(generation, shared.kill_generation());
    }
}
