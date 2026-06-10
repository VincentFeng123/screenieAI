//! Microphone capture: cpal input stream at the device's native rate,
//! downmixed to mono and resampled to 16 kHz, chunked into 512-sample frames.
//!
//! The cpal callback does no allocation and no heavy work — it recycles
//! buffers through a pool channel and `try_send`s raw blocks; downmix,
//! resampling, and chunking happen on the capture thread, which also owns
//! the `cpal::Stream` (it is `!Send` and must never cross threads).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::FromSample;
use rubato::Resampler;

use super::{Frame, VoiceError, FRAME_SAMPLES, SAMPLE_RATE};

/// Mono device-rate samples fed to the resampler per call.
const RESAMPLE_CHUNK_IN: usize = 1024;
/// Raw-block channel bound: callback drops blocks instead of blocking.
const RAW_CHAN_BOUND: usize = 32;
/// Recycled callback buffers. 8 blocks of ~100 ms each is far more headroom
/// than the capture loop ever leaves unprocessed.
const POOL_SIZE: usize = 8;

#[derive(Clone)]
pub struct CaptureSpec {
    pub device_rate: u32,
    pub channels: u16,
    pub device_name: String,
}

/// Spawns the `screenie-voice-capture` thread and returns once its stream is
/// playing (or failed to start). The thread exits when `stop` is set or every
/// consumer of `frame_tx` is gone; dropping `frame_tx` on exit closes the
/// segmenter downstream.
pub fn spawn_capture(
    stop: Arc<AtomicBool>,
    frame_tx: mpsc::SyncSender<Frame>,
    on_error: Arc<dyn Fn(VoiceError) + Send + Sync>,
) -> Result<(std::thread::JoinHandle<()>, CaptureSpec), VoiceError> {
    let (ready_tx, ready_rx) = mpsc::channel::<Result<CaptureSpec, VoiceError>>();
    let handle = std::thread::Builder::new()
        .name("screenie-voice-capture".into())
        .spawn(move || capture_thread(stop, frame_tx, on_error, ready_tx))
        .map_err(|e| VoiceError::Audio(format!("spawn capture thread: {e}")))?;
    match ready_rx.recv_timeout(Duration::from_secs(10)) {
        Ok(Ok(spec)) => Ok((handle, spec)),
        Ok(Err(e)) => Err(e),
        Err(_) => Err(VoiceError::Audio(
            "capture thread did not start in time".into(),
        )),
    }
}

/// One opened input device: its (!Send) stream, the callback channels, and
/// the rate-matched resampler. Replaced wholesale on silent-device fallback.
struct OpenCapture {
    /// Held for its lifetime only; dropping it stops the cpal callback.
    _stream: cpal::Stream,
    raw_rx: mpsc::Receiver<Vec<f32>>,
    pool_tx: mpsc::SyncSender<Vec<f32>>,
    resampler: Option<rubato::SincFixedIn<f32>>,
    resampled: Vec<Vec<f32>>,
    spec: CaptureSpec,
}

fn open_device(
    device: &cpal::Device,
    on_error: Arc<dyn Fn(VoiceError) + Send + Sync>,
) -> Result<OpenCapture, VoiceError> {
    let device_name = device
        .description()
        .map(|d| d.name().to_string())
        .unwrap_or_else(|_| "unknown input".into());
    let default_config = device
        .default_input_config()
        .map_err(|e| VoiceError::Audio(format!("input config ({device_name}): {e}")))?;
    let sample_format = default_config.sample_format();
    let config: cpal::StreamConfig = default_config.into();
    let device_rate = config.sample_rate;
    let channels = config.channels;

    let (pool_tx, pool_rx) = mpsc::sync_channel::<Vec<f32>>(POOL_SIZE);
    let (raw_tx, raw_rx) = mpsc::sync_channel::<Vec<f32>>(RAW_CHAN_BOUND);
    // ~100 ms of interleaved samples per recycled buffer; the callback only
    // reallocates if a device delivers larger blocks, and the bigger buffer
    // then stays in the pool.
    let block_capacity = (device_rate as usize / 10).max(2048) * channels.max(1) as usize;
    for _ in 0..POOL_SIZE {
        let _ = pool_tx.try_send(Vec::with_capacity(block_capacity));
    }

    let stream = build_stream(device, &config, sample_format, pool_rx, raw_tx, on_error)
        .map_err(|e| VoiceError::Audio(format!("build input stream ({device_name}): {e}")))?;
    stream
        .play()
        .map_err(|e| VoiceError::Audio(format!("start input stream ({device_name}): {e}")))?;

    let resampler = if device_rate == SAMPLE_RATE {
        None
    } else {
        let params = rubato::SincInterpolationParameters {
            sinc_len: 128,
            f_cutoff: rubato::calculate_cutoff(128, rubato::WindowFunction::Blackman2),
            interpolation: rubato::SincInterpolationType::Linear,
            oversampling_factor: 128,
            window: rubato::WindowFunction::Blackman2,
        };
        Some(
            rubato::SincFixedIn::<f32>::new(
                f64::from(SAMPLE_RATE) / f64::from(device_rate),
                1.0,
                params,
                RESAMPLE_CHUNK_IN,
                1,
            )
            .map_err(|e| VoiceError::Audio(format!("resampler init: {e}")))?,
        )
    };
    let resampled = resampler
        .as_ref()
        .map(|r| r.output_buffer_allocate(true))
        .unwrap_or_default();

    Ok(OpenCapture {
        _stream: stream,
        raw_rx,
        pool_tx,
        resampler,
        resampled,
        spec: CaptureSpec {
            device_rate,
            channels,
            device_name,
        },
    })
}

/// Three seconds of raw interleaved samples at the device's native rate.
fn silence_watch_for(spec: &CaptureSpec) -> SilenceWatch {
    SilenceWatch::new(u64::from(spec.device_rate) * u64::from(spec.channels.max(1)) * 3)
}

/// Fallback input when the default delivers digital silence: prefer the
/// built-in microphone (a Bluetooth headset idle in its case is the classic
/// silent default), then any plain microphone, then anything else.
fn pick_fallback_input(host: &cpal::Host, exclude: &str) -> Option<(cpal::Device, String)> {
    let mut plain_mic: Option<(cpal::Device, String)> = None;
    let mut any_other: Option<(cpal::Device, String)> = None;
    for device in host.input_devices().ok()? {
        let Ok(description) = device.description() else {
            continue;
        };
        let name = description.name().to_string();
        if name == exclude {
            continue;
        }
        if description.device_type() == cpal::DeviceType::Microphone {
            if name.contains("MacBook") {
                return Some((device, name));
            }
            if plain_mic.is_none() {
                plain_mic = Some((device, name));
            }
        } else if any_other.is_none() {
            any_other = Some((device, name));
        }
    }
    plain_mic.or(any_other)
}

fn capture_thread(
    stop: Arc<AtomicBool>,
    frame_tx: mpsc::SyncSender<Frame>,
    on_error: Arc<dyn Fn(VoiceError) + Send + Sync>,
    ready_tx: mpsc::Sender<Result<CaptureSpec, VoiceError>>,
) {
    let host = cpal::default_host();
    let Some(device) = host.default_input_device() else {
        let _ = ready_tx.send(Err(VoiceError::NoDevice));
        return;
    };
    let mut capture = match open_device(&device, on_error.clone()) {
        Ok(c) => c,
        Err(e) => {
            let _ = ready_tx.send(Err(e));
            return;
        }
    };
    let _ = ready_tx.send(Ok(capture.spec.clone()));

    let mut watch = silence_watch_for(&capture.spec);
    let mut fallback_used = false;
    let mut mono = Vec::new();
    let mut staging: Vec<f32> = Vec::with_capacity(RESAMPLE_CHUNK_IN * 4);
    let mut fifo: Vec<f32> = Vec::with_capacity(FRAME_SAMPLES * 8);
    let mut frames_dropped_logged = false;

    loop {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        let block = match capture.raw_rx.recv_timeout(Duration::from_millis(50)) {
            Ok(b) => b,
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        };

        if watch.observe(&block) {
            let silent_name = capture.spec.device_name.clone();
            eprintln!("[screenie] voice capture: '{silent_name}' delivered 3s of digital silence");
            let fallback = if fallback_used {
                None
            } else {
                pick_fallback_input(&host, &silent_name)
            };
            match fallback {
                Some((fb_device, fb_name)) => {
                    fallback_used = true;
                    match open_device(&fb_device, on_error.clone()) {
                        Ok(next) => {
                            on_error(VoiceError::InputSilent(format!(
                                "Microphone “{silent_name}” is producing only silence — switched to “{fb_name}”."
                            )));
                            eprintln!(
                                "[screenie] voice capture: fallback to '{}' ({} Hz, {} ch)",
                                next.spec.device_name, next.spec.device_rate, next.spec.channels
                            );
                            staging.clear();
                            fifo.clear();
                            watch = silence_watch_for(&next.spec);
                            // Old stream drops here, on its owning thread.
                            capture = next;
                            continue;
                        }
                        Err(e) => on_error(e),
                    }
                }
                None => {
                    on_error(VoiceError::InputSilent(format!(
                        "Microphone “{silent_name}” is producing only silence — choose a working input in System Settings → Sound."
                    )));
                }
            }
        }

        downmix_into(&block, capture.spec.channels, &mut mono);
        let _ = capture.pool_tx.try_send(block);
        staging.extend_from_slice(&mono);

        match &mut capture.resampler {
            None => {
                fifo.extend_from_slice(&staging);
                staging.clear();
            }
            Some(rs) => {
                while staging.len() >= RESAMPLE_CHUNK_IN {
                    let wave_in = [&staging[..RESAMPLE_CHUNK_IN]];
                    match rs.process_into_buffer(&wave_in, &mut capture.resampled, None) {
                        Ok((consumed, produced)) => {
                            fifo.extend_from_slice(&capture.resampled[0][..produced]);
                            staging.drain(..consumed);
                        }
                        Err(e) => {
                            on_error(VoiceError::Audio(format!("resample: {e}")));
                            return;
                        }
                    }
                }
            }
        }

        while fifo.len() >= FRAME_SAMPLES {
            let mut frame: Frame = Box::new([0.0; FRAME_SAMPLES]);
            frame.copy_from_slice(&fifo[..FRAME_SAMPLES]);
            fifo.drain(..FRAME_SAMPLES);
            if frame_tx.try_send(frame).is_err() && !frames_dropped_logged {
                frames_dropped_logged = true;
                eprintln!("[screenie] voice capture: segmenter backpressure, dropping frames");
            }
        }
    }
    // The stream drops here, on the thread that created it; `frame_tx`
    // drops with the thread and closes the segmenter's input.
}

/// Builds the typed input stream for whatever sample format the device
/// reports, converting to f32 in the callback (a per-sample cast — copy-tier
/// work, the only processing allowed there).
fn build_stream(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    sample_format: cpal::SampleFormat,
    pool_rx: mpsc::Receiver<Vec<f32>>,
    raw_tx: mpsc::SyncSender<Vec<f32>>,
    on_error: Arc<dyn Fn(VoiceError) + Send + Sync>,
) -> Result<cpal::Stream, cpal::Error> {
    match sample_format {
        cpal::SampleFormat::F32 => typed_stream::<f32>(device, config, pool_rx, raw_tx, on_error),
        cpal::SampleFormat::I16 => typed_stream::<i16>(device, config, pool_rx, raw_tx, on_error),
        cpal::SampleFormat::U16 => typed_stream::<u16>(device, config, pool_rx, raw_tx, on_error),
        cpal::SampleFormat::I32 => typed_stream::<i32>(device, config, pool_rx, raw_tx, on_error),
        other => {
            eprintln!("[screenie] voice capture: unusual sample format {other}, requesting f32");
            typed_stream::<f32>(device, config, pool_rx, raw_tx, on_error)
        }
    }
}

fn typed_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    pool_rx: mpsc::Receiver<Vec<f32>>,
    raw_tx: mpsc::SyncSender<Vec<f32>>,
    on_error: Arc<dyn Fn(VoiceError) + Send + Sync>,
) -> Result<cpal::Stream, cpal::Error>
where
    T: cpal::SizedSample,
    f32: cpal::FromSample<T>,
{
    // `spare` keeps a buffer the callback couldn't hand off (raw channel
    // full) so the failure path frees nothing and the buffer is reused.
    let mut spare: Option<Vec<f32>> = None;
    let mut starved_logged = false;
    device.build_input_stream(
        config.clone(),
        move |data: &[T], _: &cpal::InputCallbackInfo| {
            let buf = spare.take().or_else(|| pool_rx.try_recv().ok());
            let Some(mut buf) = buf else {
                if !starved_logged {
                    starved_logged = true;
                    eprintln!("[screenie] voice capture: buffer pool starved, dropping a block");
                }
                return;
            };
            buf.clear();
            buf.extend(data.iter().map(|&s| f32::from_sample_(s)));
            if let Err(mpsc::TrySendError::Full(b) | mpsc::TrySendError::Disconnected(b)) =
                raw_tx.try_send(buf)
            {
                spare = Some(b);
            }
        },
        move |err| on_error(VoiceError::Audio(format!("input stream: {err}"))),
        None,
    )
}

/// Detects a dead input: a real microphone always shows a noise floor, so a
/// stream of EXACT digital zeros means the device is delivering nothing —
/// a Bluetooth mic idle in its case, a TCC-denied process, or a hardware
/// mute. Fires once when `threshold_samples` of pure zeros have elapsed
/// from (re)arming with not a single nonzero sample; any nonzero sample
/// disarms it for good.
pub struct SilenceWatch {
    threshold_samples: u64,
    seen_samples: u64,
    heard_nonzero: bool,
    fired: bool,
}

impl SilenceWatch {
    pub fn new(threshold_samples: u64) -> Self {
        Self {
            threshold_samples,
            seen_samples: 0,
            heard_nonzero: false,
            fired: false,
        }
    }

    /// Feed one raw block; true exactly once, at the silence threshold.
    pub fn observe(&mut self, block: &[f32]) -> bool {
        if self.heard_nonzero || self.fired {
            return false;
        }
        if block.iter().any(|s| *s != 0.0) {
            self.heard_nonzero = true;
            return false;
        }
        self.seen_samples += block.len() as u64;
        if self.seen_samples >= self.threshold_samples {
            self.fired = true;
            return true;
        }
        false
    }
}

/// Downmix interleaved multi-channel samples to mono by channel average.
/// Reuses `mono_out` (cleared, then filled) so the steady state is
/// allocation-free once the buffer has grown to a block's size.
pub fn downmix_into(interleaved: &[f32], channels: u16, mono_out: &mut Vec<f32>) {
    mono_out.clear();
    let ch = channels.max(1) as usize;
    if ch == 1 {
        mono_out.extend_from_slice(interleaved);
        return;
    }
    let inv = 1.0 / ch as f32;
    mono_out.extend(
        interleaved
            .chunks_exact(ch)
            .map(|frame| frame.iter().sum::<f32>() * inv),
    );
}

// ---------------------------------------------------------------------------
// Microphone permission. cpal on macOS delivers *silence*, not an error, when
// permission is denied, so listening must be gated on an explicit preflight.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MicPermission {
    Authorized,
    Denied,
    Undetermined,
    /// Platform without a TCC-style preflight (Windows); rely on cpal errors.
    #[cfg_attr(target_os = "macos", allow(dead_code))]
    Unknown,
}

impl MicPermission {
    pub fn as_str(self) -> &'static str {
        match self {
            MicPermission::Authorized => "authorized",
            MicPermission::Denied => "denied",
            MicPermission::Undetermined => "undetermined",
            MicPermission::Unknown => "unknown",
        }
    }
}

#[cfg(target_os = "macos")]
mod macos_permission {
    use std::ffi::c_void;
    use std::time::Duration;

    extern "C" {
        // src/voice_macos.m — AVCaptureDevice authorization wrappers.
        fn screenie_voice_mic_auth_status() -> i32;
        fn screenie_voice_request_mic_access(
            cb: extern "C" fn(bool, *mut c_void),
            ctx: *mut c_void,
        );
    }

    pub fn status() -> super::MicPermission {
        match unsafe { screenie_voice_mic_auth_status() } {
            0 => super::MicPermission::Undetermined,
            3 => super::MicPermission::Authorized,
            _ => super::MicPermission::Denied, // restricted (1) or denied (2)
        }
    }

    /// Triggers the TCC prompt and waits (bounded) for the user's answer.
    pub fn request_blocking(timeout: Duration) -> bool {
        extern "C" fn trampoline(granted: bool, ctx: *mut c_void) {
            let tx = unsafe { Box::from_raw(ctx.cast::<std::sync::mpsc::Sender<bool>>()) };
            let _ = tx.send(granted);
        }
        let (tx, rx) = std::sync::mpsc::channel::<bool>();
        let ctx = Box::into_raw(Box::new(tx)).cast::<c_void>();
        unsafe { screenie_voice_request_mic_access(trampoline, ctx) };
        rx.recv_timeout(timeout).unwrap_or(false)
    }
}

pub fn mic_permission() -> MicPermission {
    #[cfg(target_os = "macos")]
    {
        macos_permission::status()
    }
    #[cfg(not(target_os = "macos"))]
    {
        MicPermission::Unknown
    }
}

/// Preflight for `voice_start_listening`: prompts if undetermined, errors if
/// denied. Always called off the main thread (the wait is bounded but long —
/// the user may take a while to answer the TCC dialog).
pub fn ensure_mic_permission() -> Result<(), VoiceError> {
    match mic_permission() {
        MicPermission::Authorized | MicPermission::Unknown => Ok(()),
        MicPermission::Denied => Err(VoiceError::MicDenied),
        MicPermission::Undetermined => {
            #[cfg(target_os = "macos")]
            {
                if macos_permission::request_blocking(Duration::from_secs(60)) {
                    Ok(())
                } else {
                    Err(VoiceError::MicDenied)
                }
            }
            #[cfg(not(target_os = "macos"))]
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hardware probe: captures ~2 s from EVERY input device and reports its
    /// peak RMS. Diagnostic for "voice hears nothing" reports — a device
    /// printing exactly 0.000000 delivers digital silence (TCC denial, or a
    /// Bluetooth mic that is idle/in its case). Run manually:
    ///   cargo test probe_input_devices -- --ignored --nocapture
    #[test]
    #[ignore = "hardware probe; run manually with --nocapture"]
    fn probe_input_devices_rms() {
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
        let host = cpal::default_host();
        let default_name = host
            .default_input_device()
            .and_then(|d| d.description().ok())
            .map(|d| d.name().to_string())
            .unwrap_or_default();
        for device in host.input_devices().expect("enumerate inputs") {
            let name = device
                .description()
                .map(|d| d.name().to_string())
                .unwrap_or_else(|_| "?".into());
            let Ok(config) = device.default_input_config() else {
                eprintln!("mic probe: {name}: no input config");
                continue;
            };
            let peak = std::sync::Arc::new(std::sync::Mutex::new(0.0f32));
            let peak_in = peak.clone();
            let stream = device.build_input_stream(
                config.clone().into(),
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let rms = (data.iter().map(|s| s * s).sum::<f32>()
                        / data.len().max(1) as f32)
                        .sqrt();
                    let mut p = peak_in.lock().unwrap();
                    *p = p.max(rms);
                },
                |e| eprintln!("mic probe stream error: {e}"),
                None,
            );
            match stream {
                Ok(s) => {
                    let _ = s.play();
                    std::thread::sleep(std::time::Duration::from_millis(2000));
                    drop(s);
                    let is_default = if name == default_name { " (DEFAULT)" } else { "" };
                    eprintln!(
                        "mic probe: {name}{is_default}: {} Hz, peak_rms={:.6}",
                        config.sample_rate(),
                        *peak.lock().unwrap()
                    );
                }
                Err(e) => eprintln!("mic probe: {name}: build failed: {e}"),
            }
        }
    }

    /// Raw-callback spectral probe of one device (default: the built-in
    /// MacBook mic): captures ~3 s straight from the cpal callback — no
    /// pool, no resampler — and prints per-frequency power, so hum or
    /// periodic artifacts can be attributed to the DEVICE vs the capture
    /// pipeline. Run:
    ///   SCREENIE_PROBE_DEVICE="MacBook Pro Microphone" \
    ///     cargo test probe_device_spectrum -- --ignored --nocapture
    #[test]
    #[ignore = "hardware probe; run manually with --nocapture"]
    fn probe_device_spectrum() {
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
        let want = std::env::var("SCREENIE_PROBE_DEVICE")
            .unwrap_or_else(|_| "MacBook Pro Microphone".into());
        let host = cpal::default_host();
        let device = host
            .input_devices()
            .expect("enumerate")
            .find(|d| {
                d.description()
                    .map(|desc| desc.name() == want)
                    .unwrap_or(false)
            })
            .unwrap_or_else(|| panic!("device {want:?} not found"));
        let config = device.default_input_config().expect("config");
        let rate = config.sample_rate();
        let captured = std::sync::Arc::new(std::sync::Mutex::new(Vec::<f32>::new()));
        let captured_in = captured.clone();
        let stream = device
            .build_input_stream(
                config.into(),
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    captured_in.lock().unwrap().extend_from_slice(data);
                },
                |e| eprintln!("spectrum probe stream error: {e}"),
                None,
            )
            .expect("build stream");
        stream.play().expect("play");
        std::thread::sleep(std::time::Duration::from_millis(3000));
        drop(stream);
        let samples = captured.lock().unwrap().clone();
        eprintln!("spectrum probe: {want}: {} samples at {rate} Hz", samples.len());
        let n = 16384.min(samples.len());
        let window = &samples[samples.len() - n..];
        let rms = (window.iter().map(|s| s * s).sum::<f32>() / n as f32).sqrt();
        eprintln!("rms={rms:.6}");
        for freq in [60.0f32, 120.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0] {
            // Goertzel power at freq.
            let k = (0.5 + n as f32 * freq / rate as f32) as usize;
            let w = 2.0 * std::f32::consts::PI * k as f32 / n as f32;
            let c = 2.0 * w.cos();
            let (mut s0, mut s1, mut s2) = (0.0f32, 0.0f32, 0.0f32);
            for x in window {
                s0 = x + c * s1 - s2;
                s2 = s1;
                s1 = s0;
            }
            let power = (s2 * s2 + s1 * s1 - c * s1 * s2) / n as f32;
            eprintln!("{freq:6.0} Hz power={power:.6}");
        }
    }

    #[test]
    fn silence_watch_fires_once_at_threshold() {
        let mut watch = SilenceWatch::new(100);
        assert!(!watch.observe(&[0.0; 60]));
        assert!(watch.observe(&[0.0; 60])); // 120 >= 100
        assert!(!watch.observe(&[0.0; 60])); // latched
    }

    #[test]
    fn silence_watch_disarms_on_any_nonzero_sample() {
        let mut watch = SilenceWatch::new(100);
        assert!(!watch.observe(&[0.0, 1e-6, 0.0]));
        assert!(!watch.observe(&[0.0; 500])); // disarmed forever
    }

    #[test]
    fn silence_watch_does_not_fire_before_threshold() {
        let mut watch = SilenceWatch::new(1000);
        for _ in 0..9 {
            assert!(!watch.observe(&[0.0; 100]));
        }
        assert!(watch.observe(&[0.0; 100]));
    }

    #[test]
    fn stereo_downmix_averages_channel_pairs() {
        let mut out = Vec::new();
        downmix_into(&[0.2, 0.4, -1.0, 1.0, 0.5, 0.5], 2, &mut out);
        let expected = [0.3f32, 0.0, 0.5];
        assert_eq!(out.len(), 3);
        for (got, want) in out.iter().zip(expected) {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
    }

    #[test]
    fn mono_downmix_is_a_passthrough() {
        let mut out = Vec::new();
        downmix_into(&[0.1, -0.2, 0.3], 1, &mut out);
        assert_eq!(out, vec![0.1, -0.2, 0.3]);
    }

    #[test]
    fn downmix_clears_previous_contents() {
        let mut out = vec![9.0; 8];
        downmix_into(&[0.5, 0.5], 2, &mut out);
        assert_eq!(out, vec![0.5]);
    }

    #[test]
    fn downmix_ignores_trailing_partial_frame() {
        // 5 samples at 2 channels: the dangling half-frame is dropped.
        let mut out = Vec::new();
        downmix_into(&[1.0, 1.0, 2.0, 2.0, 3.0], 2, &mut out);
        assert_eq!(out, vec![1.0, 2.0]);
    }
}
