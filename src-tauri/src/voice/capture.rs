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

pub struct CaptureSpec {
    pub device_rate: u32,
    pub channels: u16,
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
    let default_config = match device.default_input_config() {
        Ok(c) => c,
        Err(e) => {
            let _ = ready_tx.send(Err(VoiceError::Audio(format!("input config: {e}"))));
            return;
        }
    };
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

    let stream = match build_stream(
        &device,
        &config,
        sample_format,
        pool_rx,
        raw_tx,
        on_error.clone(),
    ) {
        Ok(s) => s,
        Err(e) => {
            let _ = ready_tx.send(Err(VoiceError::Audio(format!("build input stream: {e}"))));
            return;
        }
    };
    if let Err(e) = stream.play() {
        let _ = ready_tx.send(Err(VoiceError::Audio(format!("start input stream: {e}"))));
        return;
    }

    let mut resampler = if device_rate == SAMPLE_RATE {
        None
    } else {
        let params = rubato::SincInterpolationParameters {
            sinc_len: 128,
            f_cutoff: rubato::calculate_cutoff(128, rubato::WindowFunction::Blackman2),
            interpolation: rubato::SincInterpolationType::Linear,
            oversampling_factor: 128,
            window: rubato::WindowFunction::Blackman2,
        };
        match rubato::SincFixedIn::<f32>::new(
            f64::from(SAMPLE_RATE) / f64::from(device_rate),
            1.0,
            params,
            RESAMPLE_CHUNK_IN,
            1,
        ) {
            Ok(r) => Some(r),
            Err(e) => {
                let _ = ready_tx.send(Err(VoiceError::Audio(format!("resampler init: {e}"))));
                return;
            }
        }
    };
    let mut resampled = resampler
        .as_ref()
        .map(|r| r.output_buffer_allocate(true))
        .unwrap_or_default();

    let _ = ready_tx.send(Ok(CaptureSpec {
        device_rate,
        channels,
    }));

    let mut mono = Vec::with_capacity(block_capacity);
    let mut staging: Vec<f32> = Vec::with_capacity(RESAMPLE_CHUNK_IN * 4);
    let mut fifo: Vec<f32> = Vec::with_capacity(FRAME_SAMPLES * 8);
    let mut frames_dropped_logged = false;

    loop {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        let block = match raw_rx.recv_timeout(Duration::from_millis(50)) {
            Ok(b) => b,
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        };
        downmix_into(&block, channels, &mut mono);
        let _ = pool_tx.try_send(block);
        staging.extend_from_slice(&mono);

        match &mut resampler {
            None => {
                fifo.extend_from_slice(&staging);
                staging.clear();
            }
            Some(rs) => {
                while staging.len() >= RESAMPLE_CHUNK_IN {
                    let wave_in = [&staging[..RESAMPLE_CHUNK_IN]];
                    match rs.process_into_buffer(&wave_in, &mut resampled, None) {
                        Ok((consumed, produced)) => {
                            fifo.extend_from_slice(&resampled[0][..produced]);
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
    // `stream` drops here, on the thread that created it; `frame_tx` drops
    // with the thread and closes the segmenter's input.
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
