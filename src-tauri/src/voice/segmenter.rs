//! Silero VAD state machine that turns a stream of 32 ms frames into
//! pause-delimited utterances. Pure and synchronous: the thread wrapper just
//! pumps frames through [`UtteranceSegmenter::push`], so the endpointing
//! logic is fully unit-tested with a scripted detector.

use std::collections::VecDeque;
use std::time::Instant;

use super::{VoiceConfig, Frame, FRAME_MS, FRAME_SAMPLES};

/// Probability-of-speech source for one frame. Abstracted so tests script
/// probabilities and so the Silero crate can be swapped for `webrtc-vad` /
/// `earshot` without touching the state machine (spec's fallback ladder).
pub trait SpeechDetector: Send {
    fn predict(&mut self, frame: &[f32; FRAME_SAMPLES]) -> f32;
}

/// Production detector over the bundled Silero V5 ONNX model.
pub struct SileroDetector {
    inner: voice_activity_detector::VoiceActivityDetector,
}

impl SileroDetector {
    pub fn new() -> Result<Self, super::VoiceError> {
        let inner = voice_activity_detector::VoiceActivityDetector::builder()
            .sample_rate(super::SAMPLE_RATE as i64)
            .chunk_size(FRAME_SAMPLES)
            .build()
            .map_err(|e| super::VoiceError::Audio(format!("silero vad init: {e}")))?;
        Ok(Self { inner })
    }
}

impl SpeechDetector for SileroDetector {
    fn predict(&mut self, frame: &[f32; FRAME_SAMPLES]) -> f32 {
        self.inner.predict(frame.iter().copied())
    }
}

/// All counts are in 32 ms frames.
#[derive(Clone, Copy, Debug)]
pub struct SegmenterConfig {
    /// P(speech) above this counts as a voiced frame.
    pub speech_threshold: f32,
    /// Consecutive voiced frames required to enter Speech.
    pub speech_start_frames: usize,
    /// Trailing non-voiced frames that end an utterance.
    pub silence_frames: usize,
    /// Utterances with fewer voiced frames are discarded.
    pub min_voiced_frames: usize,
    /// Force-emit when the utterance buffer reaches this many frames.
    pub max_utterance_frames: usize,
    /// Frames kept ahead of the trigger so speech onset isn't clipped.
    pub pre_roll_frames: usize,
    /// Emit [`SegmenterOutput::IdleTimeout`] after this many frames without
    /// a voiced frame. `u64::MAX` disables.
    pub idle_timeout_frames: u64,
    /// Emit an RMS level reading every Nth frame (3 ≈ 10.4 Hz).
    pub level_every: u64,
}

impl SegmenterConfig {
    pub fn from_voice_config(cfg: &VoiceConfig) -> Self {
        Self {
            silence_frames: (cfg.silence_ms.clamp(400, 1000) as usize).div_ceil(FRAME_MS as usize),
            idle_timeout_frames: u64::from(cfg.auto_stop_s) * 1000 / u64::from(FRAME_MS),
            ..Self::default()
        }
    }
}

impl Default for SegmenterConfig {
    fn default() -> Self {
        Self {
            speech_threshold: 0.6,
            speech_start_frames: 3,
            silence_frames: 700 / FRAME_MS as usize,  // 21 ≈ 700 ms
            min_voiced_frames: 250 / FRAME_MS as usize + 1, // 8 ≈ 250 ms
            max_utterance_frames: 30_000 / FRAME_MS as usize, // 937 ≈ 30 s
            pre_roll_frames: 10, // 320 ms
            idle_timeout_frames: 90_000 / u64::from(FRAME_MS),
            level_every: 3,
        }
    }
}

/// A completed, pause-delimited utterance ready for transcription.
pub struct Utterance {
    /// Shared id across voice:utterance / voice:task events (read from M3).
    #[allow(dead_code)] // consumed by the dispatcher starting M3
    pub id: String,
    /// Mono 16 kHz, pre-roll included, trailing silence trimmed to a short pad.
    pub samples: Vec<f32>,
    /// End-of-speech moment, for the end-of-speech → transcript latency log.
    pub captured_at: Instant,
}

pub enum SegmenterOutput {
    Utterance(Utterance),
    Level { rms: f32 },
    IdleTimeout,
}

enum VadState {
    Idle,
    /// Voiced frames seen, not yet enough to confirm speech.
    MaybeSpeech { run: usize },
    Speech { voiced: usize, trailing: usize },
}

pub struct UtteranceSegmenter {
    cfg: SegmenterConfig,
    detector: Box<dyn SpeechDetector>,
    /// Pre-roll ring: holds the last `pre_roll_frames + speech_start_frames`
    /// frames while not in Speech, drained into the utterance on confirm.
    ring: VecDeque<Frame>,
    state: VadState,
    buf: Vec<f32>,
    buf_frames: usize,
    frames_since_voice: u64,
    idle_fired: bool,
    frame_counter: u64,
    /// Peak values since the last `take_debug_stats`, for the
    /// SCREENIE_VOICE_DEBUG pipeline-health trace.
    debug_max_prob: f32,
    debug_max_rms: f32,
}

/// Trailing silence kept on an emitted utterance so whisper sees a natural
/// release instead of an abrupt cut.
const SILENCE_PAD_FRAMES: usize = 3; // ~96 ms

impl UtteranceSegmenter {
    pub fn new(cfg: SegmenterConfig, detector: Box<dyn SpeechDetector>) -> Self {
        let ring_cap = cfg.pre_roll_frames + cfg.speech_start_frames;
        Self {
            cfg,
            detector,
            ring: VecDeque::with_capacity(ring_cap),
            state: VadState::Idle,
            buf: Vec::new(),
            buf_frames: 0,
            frames_since_voice: 0,
            idle_fired: false,
            frame_counter: 0,
            debug_max_prob: 0.0,
            debug_max_rms: 0.0,
        }
    }

    /// Peak (speech probability, rms) observed since the previous call;
    /// resets on read. Diagnostic only — distinguishes "no audio reaching
    /// the pipeline" (rms ≈ 0, e.g. TCC-denied zeros or a dead input
    /// device) from "audio flows but VAD never crosses the threshold".
    pub fn take_debug_stats(&mut self) -> (f32, f32) {
        let stats = (self.debug_max_prob, self.debug_max_rms);
        self.debug_max_prob = 0.0;
        self.debug_max_rms = 0.0;
        stats
    }

    /// Feed one frame; any resulting events are appended to `out`.
    pub fn push(&mut self, frame: Frame, out: &mut Vec<SegmenterOutput>) {
        let p = self.detector.predict(&frame);
        let is_voiced = p > self.cfg.speech_threshold;
        self.debug_max_prob = self.debug_max_prob.max(p);

        self.frame_counter += 1;
        if self.cfg.level_every > 0 && self.frame_counter % self.cfg.level_every == 0 {
            let rms =
                (frame.iter().map(|s| s * s).sum::<f32>() / FRAME_SAMPLES as f32).sqrt();
            self.debug_max_rms = self.debug_max_rms.max(rms);
            out.push(SegmenterOutput::Level { rms });
        }

        if is_voiced {
            self.frames_since_voice = 0;
            self.idle_fired = false;
        } else {
            self.frames_since_voice = self.frames_since_voice.saturating_add(1);
            if !self.idle_fired && self.frames_since_voice >= self.cfg.idle_timeout_frames {
                self.idle_fired = true;
                out.push(SegmenterOutput::IdleTimeout);
            }
        }

        match self.state {
            VadState::Idle => {
                self.push_ring(frame);
                if is_voiced {
                    self.state = VadState::MaybeSpeech { run: 1 };
                    self.confirm_if_started();
                }
            }
            VadState::MaybeSpeech { run } => {
                self.push_ring(frame);
                if is_voiced {
                    self.state = VadState::MaybeSpeech { run: run + 1 };
                    self.confirm_if_started();
                } else {
                    self.state = VadState::Idle;
                }
            }
            VadState::Speech { voiced, trailing } => {
                self.buf.extend_from_slice(&frame[..]);
                self.buf_frames += 1;
                let (voiced, trailing) = if is_voiced {
                    (voiced + 1, 0)
                } else {
                    (voiced, trailing + 1)
                };
                self.state = VadState::Speech { voiced, trailing };
                if self.buf_frames >= self.cfg.max_utterance_frames {
                    self.finalize(out, 0, true);
                } else if trailing >= self.cfg.silence_frames {
                    self.finalize(out, trailing, false);
                }
            }
        }
    }

    fn push_ring(&mut self, frame: Frame) {
        let cap = self.cfg.pre_roll_frames + self.cfg.speech_start_frames;
        if self.ring.len() >= cap {
            self.ring.pop_front();
        }
        self.ring.push_back(frame);
    }

    /// MaybeSpeech -> Speech once enough consecutive voiced frames arrived;
    /// the utterance starts from the drained pre-roll ring (which also holds
    /// the trigger frames themselves).
    fn confirm_if_started(&mut self) {
        let VadState::MaybeSpeech { run } = self.state else {
            return;
        };
        if run < self.cfg.speech_start_frames {
            return;
        }
        self.buf.clear();
        self.buf_frames = 0;
        while let Some(f) = self.ring.pop_front() {
            self.buf.extend_from_slice(&f[..]);
            self.buf_frames += 1;
        }
        self.state = VadState::Speech {
            voiced: self.cfg.speech_start_frames,
            trailing: 0,
        };
    }

    fn finalize(&mut self, out: &mut Vec<SegmenterOutput>, trailing: usize, forced: bool) {
        let VadState::Speech { voiced, .. } = self.state else {
            return;
        };
        if !forced {
            let strip = trailing.saturating_sub(SILENCE_PAD_FRAMES);
            self.buf
                .truncate((self.buf_frames - strip) * FRAME_SAMPLES);
        }
        if forced || voiced >= self.cfg.min_voiced_frames {
            out.push(SegmenterOutput::Utterance(Utterance {
                id: uuid::Uuid::new_v4().to_string(),
                samples: std::mem::take(&mut self.buf),
                // End-of-speech as detected (i.e. when the trailing-silence
                // window closed); STT latency is measured from here.
                captured_at: Instant::now(),
            }));
        } else {
            self.buf.clear();
        }
        self.buf_frames = 0;
        self.ring.clear();
        self.state = VadState::Idle;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Returns scripted probabilities in order; repeats the last entry.
    struct Scripted {
        probs: Vec<f32>,
        i: usize,
    }

    impl SpeechDetector for Scripted {
        fn predict(&mut self, _frame: &[f32; FRAME_SAMPLES]) -> f32 {
            let p = self.probs[self.i.min(self.probs.len() - 1)];
            self.i += 1;
            p
        }
    }

    fn cfg() -> SegmenterConfig {
        SegmenterConfig {
            speech_threshold: 0.6,
            speech_start_frames: 3,
            silence_frames: 22,
            min_voiced_frames: 8,
            max_utterance_frames: 938,
            pre_roll_frames: 10,
            idle_timeout_frames: u64::MAX,
            level_every: 3,
        }
    }

    /// Script helper: `(count, probability)` runs concatenated.
    fn script(runs: &[(usize, f32)]) -> Vec<f32> {
        runs.iter()
            .flat_map(|&(n, p)| std::iter::repeat(p).take(n))
            .collect()
    }

    /// Feeds `n` frames; frame `k` is filled with the value `k as f32` so
    /// tests can identify which frames ended up in an utterance.
    fn run(cfg: SegmenterConfig, probs: Vec<f32>) -> Vec<SegmenterOutput> {
        let n = probs.len();
        let mut seg = UtteranceSegmenter::new(cfg, Box::new(Scripted { probs, i: 0 }));
        let mut out = Vec::new();
        for k in 0..n {
            seg.push(Box::new([k as f32; FRAME_SAMPLES]), &mut out);
        }
        out
    }

    fn utterances(out: &[SegmenterOutput]) -> Vec<&Utterance> {
        out.iter()
            .filter_map(|o| match o {
                SegmenterOutput::Utterance(u) => Some(u),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn three_bursts_with_pauses_yield_exactly_three_utterances() {
        let probs = script(&[
            (15, 0.05),
            (40, 0.9),
            (30, 0.05),
            (40, 0.9),
            (30, 0.05),
            (40, 0.9),
            (30, 0.05),
        ]);
        let out = run(cfg(), probs);
        assert_eq!(utterances(&out).len(), 3);
    }

    #[test]
    fn pure_silence_yields_zero_utterances() {
        // ~30 s of low-probability frames.
        let out = run(cfg(), script(&[(940, 0.1)]));
        assert_eq!(utterances(&out).len(), 0);
    }

    #[test]
    fn blip_shorter_than_min_voiced_is_discarded() {
        // 5 voiced frames (160 ms) < min_voiced_frames 8 (250 ms).
        let out = run(cfg(), script(&[(15, 0.05), (5, 0.9), (30, 0.05)]));
        assert_eq!(utterances(&out).len(), 0);
    }

    #[test]
    fn two_voiced_frames_never_confirm_speech() {
        let out = run(cfg(), script(&[(15, 0.05), (2, 0.9), (30, 0.05)]));
        assert_eq!(utterances(&out).len(), 0);
    }

    #[test]
    fn continuous_speech_force_emits_at_cap_then_segments_remainder() {
        let out = run(cfg(), script(&[(15, 0.05), (950, 0.9), (30, 0.05)]));
        let utts = utterances(&out);
        assert_eq!(utts.len(), 2);
        // Force-emit fires exactly at the frame cap.
        assert_eq!(utts[0].samples.len(), 938 * FRAME_SAMPLES);
        assert!(!utts[1].samples.is_empty());
    }

    #[test]
    fn pre_roll_is_prepended_to_the_utterance() {
        // Speech starts at frame 20; trigger confirms at frame 22. The ring
        // holds pre_roll(10) + start(3) = 13 frames, so the utterance must
        // begin with frame 10 and contain speech-onset frame 20.
        let out = run(cfg(), script(&[(20, 0.05), (40, 0.9), (30, 0.05)]));
        let utts = utterances(&out);
        assert_eq!(utts.len(), 1);
        let samples = &utts[0].samples;
        assert_eq!(samples[0], 10.0);
        assert_eq!(samples[(20 - 10) * FRAME_SAMPLES], 20.0);
    }

    #[test]
    fn trailing_silence_is_stripped_to_a_short_pad() {
        // Ring at confirm: frames 5..=17 (13). Speech appends frames 18..=54
        // (37). Silence appends 22 trailing frames before finalize.
        // Kept = 13 + 37 + 3 pad = 53 frames.
        let out = run(cfg(), script(&[(15, 0.05), (40, 0.9), (25, 0.05)]));
        let utts = utterances(&out);
        assert_eq!(utts.len(), 1);
        assert_eq!(utts[0].samples.len(), 53 * FRAME_SAMPLES);
    }

    #[test]
    fn idle_timeout_fires_exactly_once() {
        let mut c = cfg();
        c.idle_timeout_frames = 50;
        let out = run(c, script(&[(120, 0.05)]));
        let timeouts = out
            .iter()
            .filter(|o| matches!(o, SegmenterOutput::IdleTimeout))
            .count();
        assert_eq!(timeouts, 1);
    }

    #[test]
    fn voiced_frames_reset_the_idle_counter() {
        let mut c = cfg();
        c.idle_timeout_frames = 50;
        let out = run(c, script(&[(40, 0.05), (10, 0.9), (45, 0.05)]));
        let timeouts = out
            .iter()
            .filter(|o| matches!(o, SegmenterOutput::IdleTimeout))
            .count();
        assert_eq!(timeouts, 0);
    }

    #[test]
    fn level_events_follow_the_configured_cadence() {
        let out = run(cfg(), script(&[(9, 0.05)]));
        let levels = out
            .iter()
            .filter(|o| matches!(o, SegmenterOutput::Level { .. }))
            .count();
        assert_eq!(levels, 3);
    }

    #[test]
    fn utterance_ids_are_unique_and_nonempty() {
        let probs = script(&[(15, 0.05), (40, 0.9), (30, 0.05), (40, 0.9), (30, 0.05)]);
        let out = run(cfg(), probs);
        let utts = utterances(&out);
        assert_eq!(utts.len(), 2);
        assert!(!utts[0].id.is_empty());
        assert_ne!(utts[0].id, utts[1].id);
    }

    /// End-to-end check with the REAL Silero model on synthesized speech.
    /// Ignored by default (needs a wav fixture); run manually with e.g.:
    ///   say --file-format=WAVE --data-format=LEF32@16000 -o /tmp/vad.wav \
    ///     "open a new tab [[slnc 1300]] type amazon dot com [[slnc 1300]] third phrase"
    ///   SCREENIE_VAD_WAV=/tmp/vad.wav SCREENIE_VAD_EXPECT=3 \
    ///     cargo test real_silero -- --ignored --nocapture
    #[test]
    #[ignore = "needs SCREENIE_VAD_WAV fixture; see doc comment"]
    fn real_silero_on_synthesized_speech() {
        let path = std::env::var("SCREENIE_VAD_WAV").expect("SCREENIE_VAD_WAV not set");
        let expect: usize = std::env::var("SCREENIE_VAD_EXPECT")
            .expect("SCREENIE_VAD_EXPECT not set")
            .parse()
            .unwrap();
        let samples = crate::voice::read_wav_f32_mono_16k(&path);
        assert!(!samples.is_empty(), "no samples decoded from {path}");

        let detector = SileroDetector::new().expect("silero init");
        let mut seg = UtteranceSegmenter::new(SegmenterConfig::default(), Box::new(detector));
        let mut out = Vec::new();
        for chunk in samples.chunks(FRAME_SAMPLES) {
            let mut frame = [0.0f32; FRAME_SAMPLES];
            frame[..chunk.len()].copy_from_slice(chunk);
            seg.push(Box::new(frame), &mut out);
        }
        // Flush: feed trailing silence so the last utterance endpoints.
        for _ in 0..SegmenterConfig::default().silence_frames + 2 {
            seg.push(Box::new([0.0f32; FRAME_SAMPLES]), &mut out);
        }
        let utts = utterances(&out);
        for u in &utts {
            eprintln!(
                "utterance captured: {:.1}s",
                u.samples.len() as f32 / 16_000.0
            );
        }
        assert_eq!(utts.len(), expect);
    }

    #[test]
    fn config_derivation_from_voice_config() {
        let vc = VoiceConfig {
            silence_ms: 700,
            auto_stop_s: 90,
            ..Default::default()
        };
        let sc = SegmenterConfig::from_voice_config(&vc);
        assert_eq!(sc.silence_frames, 22); // ceil(700 / 32)
        assert_eq!(sc.idle_timeout_frames, 2812); // 90_000 / 32
        // Out-of-range silence_ms is clamped before conversion.
        let vc = VoiceConfig {
            silence_ms: 5000,
            ..Default::default()
        };
        assert_eq!(SegmenterConfig::from_voice_config(&vc).silence_frames, 32); // ceil(1000/32)
    }
}
