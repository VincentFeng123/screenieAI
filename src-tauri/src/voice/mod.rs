//! Voice command mode: mic capture -> VAD endpointing -> on-device whisper
//! transcription -> serial dispatch into the existing agent runtime.
//!
//! The agent loop, planner contract, and safety gate are reused unchanged —
//! the dispatcher calls the same private helpers `start_agent_task` wraps
//! (`agent_task_options_from_goal`, `prepare_stub_agent_run`,
//! `run_prepared_stub_agent`). See docs in each submodule.

pub mod capture;
pub mod dispatcher;
pub mod model;
pub mod segmenter;
pub mod stt;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager};

use crate::{lock_poison_safe, AppState};

/// Window that owns all voice UI; every voice event targets it.
pub const VOICE_WINDOW: &str = "quick_tooltip";

/// Samples per VAD frame. Silero V5 only accepts 512-sample chunks at 16 kHz.
pub const FRAME_SAMPLES: usize = 512;
/// One frame is 512 / 16_000 s = 32 ms.
pub const FRAME_MS: u32 = 32;
/// Pipeline sample rate. Whisper and Silero both want 16 kHz mono.
pub const SAMPLE_RATE: u32 = 16_000;

/// One mono 16 kHz frame. Boxed so channel sends move a pointer, not 2 KiB.
pub type Frame = Box<[f32; FRAME_SAMPLES]>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum VoiceStatus {
    Idle,
    Listening,
    Transcribing,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WhisperModel {
    #[serde(rename = "tiny.en")]
    TinyEn,
    #[serde(rename = "base.en")]
    BaseEn,
    #[serde(rename = "small.en")]
    SmallEn,
}

/// Persisted voice settings (`voice.json` in the app data dir). Rust is the
/// single source of truth — the segmenter consumes these at runtime, so they
/// deliberately do not live in frontend localStorage.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VoiceConfig {
    /// Trailing silence that ends an utterance. Clamped to 400..=1000.
    pub silence_ms: u32,
    pub model: WhisperModel,
    /// Listening auto-stops after this many seconds without detected speech.
    pub auto_stop_s: u32,
}

impl Default for VoiceConfig {
    fn default() -> Self {
        Self {
            silence_ms: 700,
            model: WhisperModel::BaseEn,
            auto_stop_s: 90,
        }
    }
}

impl WhisperModel {
    /// The wire format used by `voice_set_config` / settings UI.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "tiny.en" => Some(Self::TinyEn),
            "base.en" => Some(Self::BaseEn),
            "small.en" => Some(Self::SmallEn),
            _ => None,
        }
    }
}

impl VoiceConfig {
    /// Merge optional overrides, clamping each field to its valid range.
    /// Out-of-range values are clamped rather than rejected so a stale or
    /// hand-edited config can never wedge the pipeline.
    pub fn merged_with(
        self,
        silence_ms: Option<u32>,
        model: Option<WhisperModel>,
        auto_stop_s: Option<u32>,
    ) -> Self {
        Self {
            silence_ms: silence_ms.unwrap_or(self.silence_ms).clamp(400, 1000),
            model: model.unwrap_or(self.model),
            auto_stop_s: auto_stop_s.unwrap_or(self.auto_stop_s).clamp(10, 600),
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum VoiceError {
    #[error("microphone permission denied")]
    MicDenied,
    #[error("no audio input device available")]
    NoDevice,
    #[error("speech model not downloaded: {0}")]
    ModelMissing(String),
    #[error("audio: {0}")]
    Audio(String),
    #[error("stt: {0}")]
    Stt(String),
    #[error("download: {0}")]
    Download(String),
    /// The input device streams exact digital zeros (Bluetooth mic idle in
    /// its case, hardware mute). Message is user-facing as written.
    #[error("{0}")]
    InputSilent(String),
}

impl VoiceError {
    /// Stable machine-readable code carried by `voice:error` events.
    pub fn code(&self) -> &'static str {
        match self {
            VoiceError::MicDenied => "mic_denied",
            VoiceError::NoDevice => "no_device",
            VoiceError::ModelMissing(_) => "model_missing",
            VoiceError::Audio(_) => "audio",
            VoiceError::Stt(_) => "stt",
            VoiceError::Download(_) => "download",
            VoiceError::InputSilent(_) => "input_silent",
        }
    }
}

/// Agent settings captured at mic-on, mirroring exactly what the typed
/// input passes to `start_agent_task`; every dispatched voice task runs
/// with these.
#[derive(Clone, Debug, Default)]
pub struct AgentRunSettings {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub vision_provider: Option<String>,
    pub vision_model: Option<String>,
    pub autonomy: Option<String>,
    pub scripting_enabled: Option<bool>,
    pub web_lookup_enabled: Option<bool>,
}

/// Live listening session. Held in `AppState.voice_session`; a handle whose
/// `stop` flag is set is stale (auto-stopped or torn down) — `is_active`
/// distinguishes, so a stale handle left in state is harmless.
pub struct VoiceSessionHandle {
    stop: Arc<AtomicBool>,
    status: Arc<Mutex<VoiceStatus>>,
}

impl VoiceSessionHandle {
    pub fn is_active(&self) -> bool {
        !self.stop.load(Ordering::Relaxed)
    }

    pub fn status(&self) -> VoiceStatus {
        if self.is_active() {
            *lock_poison_safe(&self.status)
        } else {
            VoiceStatus::Idle
        }
    }
}

/// All backend → webview voice events funnel through here, targeted at the
/// tooltip window only.
fn emit_voice<S: Serialize + Clone>(app: &AppHandle, event: &str, payload: S) {
    if let Err(e) = app.emit_to(VOICE_WINDOW, event, payload) {
        eprintln!("[screenie] voice emit {event} failed: {e}");
    }
}

fn emit_voice_error(app: &AppHandle, err: &VoiceError) {
    emit_voice(
        app,
        "voice:error",
        serde_json::json!({ "code": err.code(), "message": err.to_string() }),
    );
}

/// Frame-channel bound between capture and segmenter: ~2 s of audio.
const FRAME_CHAN_BOUND: usize = 64;

/// Starts a listening session: mic-permission preflight, capture thread,
/// segmenter thread. Blocking (TCC prompt wait, stream startup) — call from
/// `spawn_blocking`, never the main thread.
pub fn start_session(
    app: &AppHandle,
    cfg: VoiceConfig,
    run_settings: AgentRunSettings,
) -> Result<VoiceSessionHandle, VoiceError> {
    capture::ensure_mic_permission()?;
    if !model::is_model_present(app, cfg.model) {
        // Drives the frontend's first-use download prompt.
        return Err(VoiceError::ModelMissing(cfg.model.wire_name().into()));
    }
    let model_path = model::model_path(app, cfg.model).map_err(VoiceError::Stt)?;

    let stop = Arc::new(AtomicBool::new(false));
    let status = Arc::new(Mutex::new(VoiceStatus::Listening));
    let (frame_tx, frame_rx) = mpsc::sync_channel::<Frame>(FRAME_CHAN_BOUND);
    // Unbounded: utterances are rare and large; the segmenter must never
    // block on a slow decode.
    let (utterance_tx, utterance_rx) = mpsc::channel::<segmenter::Utterance>();

    let on_error: Arc<dyn Fn(VoiceError) + Send + Sync> = {
        let app = app.clone();
        Arc::new(move |e: VoiceError| {
            eprintln!("[screenie] voice error: {e}");
            emit_voice_error(&app, &e);
        })
    };

    // Everything fallible happens BEFORE the mic-owning capture thread
    // spawns, and the spawn order is downstream-first (STT, segmenter,
    // capture last): if any spawn fails, the already-running stages tear
    // down through their closing channels — no thread is left holding the
    // microphone with nothing to feed.
    let detector = segmenter::SileroDetector::new()?;
    let dispatch = ensure_dispatcher(app)?;

    {
        let on_transcript = {
            let app = app.clone();
            let settings = run_settings;
            move |utterance: segmenter::Utterance, text: String| {
                dispatcher::route(&app, &dispatch, &settings, utterance.id, text);
            }
        };
        let on_status = {
            let app = app.clone();
            let stop = stop.clone();
            let status = status.clone();
            move |s: VoiceStatus| {
                if !stop.load(Ordering::Relaxed) {
                    *lock_poison_safe(&status) = s;
                    emit_voice(&app, "voice:status", s);
                }
            }
        };
        let on_stt_error = {
            let app = app.clone();
            move |e: VoiceError| {
                eprintln!("[screenie] voice stt error: {e}");
                emit_voice_error(&app, &e);
            }
        };
        stt::spawn_stt(
            stop.clone(),
            model_path,
            utterance_rx,
            on_transcript,
            on_status,
            on_stt_error,
        )?;
    }

    let seg_cfg = segmenter::SegmenterConfig::from_voice_config(&cfg);
    {
        let app = app.clone();
        let stop = stop.clone();
        let status = status.clone();
        let auto_stop_s = cfg.auto_stop_s;
        std::thread::Builder::new()
            .name("screenie-voice-segmenter".into())
            .spawn(move || {
                segmenter_thread(
                    app,
                    stop,
                    status,
                    seg_cfg,
                    Box::new(detector),
                    frame_rx,
                    utterance_tx,
                    auto_stop_s,
                )
            })
            // The STT thread unwinds via the dropped utterance channel.
            .map_err(|e| VoiceError::Audio(format!("spawn segmenter thread: {e}")))?;
    }

    let capture_result = capture::spawn_capture(stop.clone(), frame_tx, on_error);
    let spec = match capture_result {
        Ok((_join, spec)) => spec,
        Err(e) => {
            // Belt and braces: the failed spawn already dropped frame_tx
            // (unwinding segmenter -> STT through channel closes), but make
            // the teardown explicit for any thread mid-recv.
            stop.store(true, Ordering::Relaxed);
            return Err(e);
        }
    };
    eprintln!(
        "[screenie] voice capture started: '{}' {} Hz, {} ch -> 16 kHz mono",
        spec.device_name, spec.device_rate, spec.channels
    );

    emit_voice(app, "voice:status", VoiceStatus::Listening);
    Ok(VoiceSessionHandle { stop, status })
}

/// The dispatcher thread is process-lifetime: spawned at first mic-on and
/// shared by every later session (it owns no audio, only the task queue).
fn ensure_dispatcher(app: &AppHandle) -> Result<Arc<dispatcher::DispatchShared>, VoiceError> {
    let state = app.state::<AppState>();
    let mut guard = lock_poison_safe(&state.voice_dispatch);
    if let Some(shared) = guard.as_ref() {
        return Ok(shared.clone());
    }
    let shared = Arc::new(dispatcher::DispatchShared::new());
    dispatcher::spawn_dispatcher(app.clone(), shared.clone()).map_err(VoiceError::Audio)?;
    *guard = Some(shared.clone());
    Ok(shared)
}

/// Stops capture immediately. Any partial utterance is discarded with the
/// segmenter; queued/running agent tasks deliberately keep going — kill
/// words and the existing Stop paths are the only things that abort work.
pub fn stop_session(app: &AppHandle, handle: &VoiceSessionHandle) {
    handle.stop.store(true, Ordering::Relaxed);
    *lock_poison_safe(&handle.status) = VoiceStatus::Idle;
    emit_voice(app, "voice:status", VoiceStatus::Idle);
}

#[allow(clippy::too_many_arguments)]
fn segmenter_thread(
    app: AppHandle,
    stop: Arc<AtomicBool>,
    status: Arc<Mutex<VoiceStatus>>,
    seg_cfg: segmenter::SegmenterConfig,
    detector: Box<dyn segmenter::SpeechDetector>,
    frame_rx: mpsc::Receiver<Frame>,
    utterance_tx: mpsc::Sender<segmenter::Utterance>,
    auto_stop_s: u32,
) {
    let mut seg = segmenter::UtteranceSegmenter::new(seg_cfg, detector);
    let mut out = Vec::new();
    // SCREENIE_VOICE_DEBUG=1: once a second, log the peak speech probability
    // and RMS so a dead pipeline is diagnosable from the dev terminal —
    // rms ≈ 0.000000 means no real audio (denied mic / wrong device); real
    // probability that never crosses 0.6 means a VAD threshold problem.
    let debug = std::env::var("SCREENIE_VOICE_DEBUG").is_ok();
    let mut debug_frames = 0u32;
    // SCREENIE_VOICE_TAP=<path>: record the exact 16 kHz mono stream the
    // VAD/STT see (raw little-endian f32) for offline inspection.
    let mut tap = std::env::var("SCREENIE_VOICE_TAP")
        .ok()
        .and_then(|path| std::fs::File::create(path).ok());
    while let Ok(frame) = frame_rx.recv() {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        if let Some(file) = tap.as_mut() {
            use std::io::Write;
            let bytes: Vec<u8> = frame.iter().flat_map(|s| s.to_le_bytes()).collect();
            let _ = file.write_all(&bytes);
        }
        if debug {
            debug_frames += 1;
            if debug_frames >= 31 {
                debug_frames = 0;
                let (max_prob, max_rms) = seg.take_debug_stats();
                eprintln!(
                    "[screenie] voice debug: max_prob={max_prob:.3} max_rms={max_rms:.6}"
                );
            }
        }
        seg.push(frame, &mut out);
        for event in out.drain(..) {
            match event {
                segmenter::SegmenterOutput::Utterance(utterance) => {
                    eprintln!(
                        "[screenie] utterance captured: {:.1}s",
                        utterance.samples.len() as f32 / SAMPLE_RATE as f32
                    );
                    let _ = utterance_tx.send(utterance);
                }
                segmenter::SegmenterOutput::Level { rms } => {
                    emit_voice(&app, "voice:level", serde_json::json!({ "rms": rms }));
                }
                segmenter::SegmenterOutput::IdleTimeout => {
                    eprintln!("[screenie] voice auto-stop: no speech for {auto_stop_s}s");
                    stop.store(true, Ordering::Relaxed);
                    *lock_poison_safe(&status) = VoiceStatus::Idle;
                    emit_voice(&app, "voice:status", VoiceStatus::Idle);
                    emit_voice(
                        &app,
                        "voice:error",
                        serde_json::json!({
                            "code": "idle_timeout",
                            "message": format!("Mic turned off — no speech detected for {auto_stop_s}s"),
                        }),
                    );
                }
            }
        }
    }
    // Capture thread exits on the stop flag and drops `frame_tx`, which ends
    // the recv loop above. Partial utterance state drops with the segmenter.
}

fn current_config(app: &AppHandle) -> VoiceConfig {
    let state = app.state::<AppState>();
    let mut cfg = lock_poison_safe(&state.voice_config);
    *cfg.get_or_insert_with(|| model::load_voice_config(app))
}

// ---------------------------------------------------------------------------
// Tauri commands. Registered in lib.rs's generate_handler! as voice::…
// ---------------------------------------------------------------------------

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VoiceStatusPayload {
    pub status: VoiceStatus,
    pub model_present: bool,
    pub mic_permission: &'static str,
    pub config: VoiceConfig,
}

/// Async so the bounded waits inside (TCC prompt, stream startup) never run
/// on the main thread; the heavy lifting goes through `spawn_blocking`.
#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn voice_start_listening(
    app: AppHandle,
    window: tauri::WebviewWindow,
    provider: Option<String>,
    model: Option<String>,
    vision_provider: Option<String>,
    vision_model: Option<String>,
    autonomy: Option<String>,
    scripting_enabled: Option<bool>,
    web_lookup_enabled: Option<bool>,
) -> Result<(), String> {
    crate::require_window(&window, VOICE_WINDOW)?;
    let run_settings = AgentRunSettings {
        provider,
        model,
        vision_provider,
        vision_model,
        autonomy,
        scripting_enabled,
        web_lookup_enabled,
    };
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        // `start_session` blocks (TCC prompt, stream startup), so the
        // is-active check below can't be held under a lock for its whole
        // duration; this latch closes the window where two concurrent
        // invokes would both pass the check and double-open the mic.
        if state
            .voice_starting
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Ok(()); // another start is mid-flight; idempotent
        }
        let result = (|| {
            // Idempotent: a second start while listening is a no-op.
            if lock_poison_safe(&state.voice_session)
                .as_ref()
                .is_some_and(VoiceSessionHandle::is_active)
            {
                return Ok(());
            }
            let cfg = current_config(&app);
            match start_session(&app, cfg, run_settings) {
                Ok(handle) => {
                    *lock_poison_safe(&state.voice_session) = Some(handle);
                    Ok(())
                }
                Err(e) => {
                    emit_voice_error(&app, &e);
                    Err(e.to_string())
                }
            }
        })();
        state.voice_starting.store(false, Ordering::SeqCst);
        result
    })
    .await
    .map_err(|e| format!("voice start: {e}"))?
}

#[tauri::command]
pub fn voice_stop_listening(
    app: AppHandle,
    window: tauri::WebviewWindow,
    state: tauri::State<'_, AppState>,
) -> Result<(), String> {
    crate::require_window(&window, VOICE_WINDOW)?;
    if let Some(handle) = lock_poison_safe(&state.voice_session).take() {
        stop_session(&app, &handle);
    }
    Ok(())
}

#[tauri::command]
pub fn voice_get_status(
    app: AppHandle,
    window: tauri::WebviewWindow,
    state: tauri::State<'_, AppState>,
) -> Result<VoiceStatusPayload, String> {
    // Read-only; also used by the Settings window to seed the Voice section.
    let label = window.label();
    if label != VOICE_WINDOW && label != "main" {
        return Err("command not allowed from this window".into());
    }
    let status = lock_poison_safe(&state.voice_session)
        .as_ref()
        .map(VoiceSessionHandle::status)
        .unwrap_or(VoiceStatus::Idle);
    let config = current_config(&app);
    Ok(VoiceStatusPayload {
        status,
        model_present: model::is_model_present(&app, config.model),
        mic_permission: capture::mic_permission().as_str(),
        config,
    })
}

/// Async: streams the download on the async runtime, resolving when the
/// model is verified in place. Progress arrives via `voice:model_download`.
#[tauri::command]
pub async fn voice_download_model(
    app: AppHandle,
    window: tauri::WebviewWindow,
    model: Option<String>,
) -> Result<(), String> {
    let label = window.label();
    if label != VOICE_WINDOW && label != "main" {
        return Err("command not allowed from this window".into());
    }
    let model = match model {
        Some(name) => {
            WhisperModel::parse(&name).ok_or_else(|| format!("unknown voice model: {name}"))?
        }
        None => current_config(&app).model,
    };
    // Replace-on-new cancel flag (cf. the per-window ai cancel slots): a
    // second download request supersedes the in-flight one.
    let cancel: crate::ai::CancelFlag = Arc::new(std::sync::atomic::AtomicBool::new(false));
    {
        let state = app.state::<AppState>();
        let mut guard = lock_poison_safe(&state.voice_download_cancel);
        if let Some(previous) = guard.replace(cancel.clone()) {
            previous.store(true, Ordering::Relaxed);
        }
    }
    match model::download_model(app.clone(), model, cancel).await {
        Ok(()) => Ok(()),
        Err(e) => {
            emit_voice_error(&app, &e);
            Err(e.to_string())
        }
    }
}

#[tauri::command]
pub fn voice_set_config(
    app: AppHandle,
    window: tauri::WebviewWindow,
    state: tauri::State<'_, AppState>,
    silence_ms: Option<u32>,
    model: Option<String>,
    auto_stop_s: Option<u32>,
) -> Result<VoiceConfig, String> {
    let label = window.label();
    if label != VOICE_WINDOW && label != "main" {
        return Err("command not allowed from this window".into());
    }
    let model = match model {
        Some(name) => Some(
            WhisperModel::parse(&name).ok_or_else(|| format!("unknown voice model: {name}"))?,
        ),
        None => None,
    };
    let merged = current_config(&app).merged_with(silence_ms, model, auto_stop_s);
    model::save_voice_config(&app, &merged)?;
    *lock_poison_safe(&state.voice_config) = Some(merged);
    // Applies to the NEXT listening session; an active one keeps its config.
    Ok(merged)
}

/// Minimal WAV reader for integration-test fixtures: scans for the `data`
/// chunk and decodes 32-bit-float little-endian samples (the format
/// `say --data-format=LEF32@16000` produces).
#[cfg(test)]
pub(crate) fn read_wav_f32_mono_16k(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).expect("read wav");
    let data_pos = bytes
        .windows(4)
        .position(|w| w == b"data")
        .expect("no data chunk");
    let len = u32::from_le_bytes(bytes[data_pos + 4..data_pos + 8].try_into().unwrap());
    let start = data_pos + 8;
    let end = (start + len as usize).min(bytes.len());
    bytes[start..end]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_names_round_trip() {
        assert_eq!(WhisperModel::parse("tiny.en"), Some(WhisperModel::TinyEn));
        assert_eq!(WhisperModel::parse("base.en"), Some(WhisperModel::BaseEn));
        assert_eq!(WhisperModel::parse("small.en"), Some(WhisperModel::SmallEn));
        assert_eq!(WhisperModel::parse("large"), None);
        assert_eq!(WhisperModel::parse(""), None);
    }

    #[test]
    fn merged_with_overrides_and_clamps() {
        let base = VoiceConfig::default();
        let merged = base.merged_with(Some(450), Some(WhisperModel::TinyEn), Some(120));
        assert_eq!(merged.silence_ms, 450);
        assert_eq!(merged.model, WhisperModel::TinyEn);
        assert_eq!(merged.auto_stop_s, 120);

        // Out-of-range values clamp instead of erroring.
        let clamped = base.merged_with(Some(50), None, Some(100_000));
        assert_eq!(clamped.silence_ms, 400);
        assert_eq!(clamped.auto_stop_s, 600);
        assert_eq!(clamped.model, base.model);

        // None leaves fields untouched.
        let same = base.merged_with(None, None, None);
        assert_eq!(same, base);
    }

    #[test]
    fn config_serializes_model_as_wire_name() {
        let json = serde_json::to_string(&VoiceConfig::default()).unwrap();
        assert!(json.contains("\"base.en\""), "got {json}");
        assert!(json.contains("\"silenceMs\":700"), "got {json}");
        let back: VoiceConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back, VoiceConfig::default());
    }
}
