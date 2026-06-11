//! whisper.cpp transcription on a dedicated thread. The model context loads
//! lazily on the first utterance of a session and is reused until the session
//! ends; decoding is blocking and heavy, which is why it gets its own thread
//! and why the segmenter never waits on it (unbounded utterance channel).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};

use super::segmenter::Utterance;
use super::{VoiceError, VoiceStatus, SAMPLE_RATE};

/// Exact (case-insensitive, post-trim) transcripts whisper is known to
/// hallucinate on silence, breath, and non-speech noise. One const so the
/// list is easy to extend during tuning.
pub const HALLUCINATION_BLOCKLIST: &[&str] = &[
    "thank you.",
    "thank you",
    "thanks for watching",
    "thanks for watching!",
    "you",
    "bye",
    "bye.",
    "bye!",
    "[blank_audio]",
    "(music)",
    "[music]",
    "(silence)",
    ".",
];

/// whisper.cpp misbehaves on inputs shorter than ~1 s; shorter utterances
/// are zero-padded to this length (1.1 s at 16 kHz).
const MIN_DECODE_SAMPLES: usize = 17_600;

/// Trim, drop too-short results, drop exact hallucination-blocklist matches.
/// Returns the cleaned transcript, or None when it should be discarded.
pub fn post_filter(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.chars().count() < 2 {
        return None;
    }
    let lowered = trimmed.to_lowercase();
    if HALLUCINATION_BLOCKLIST.contains(&lowered.as_str()) {
        return None;
    }
    Some(trimmed.to_string())
}

pub struct SttEngine {
    ctx: whisper_rs::WhisperContext,
    n_threads: i32,
}

impl SttEngine {
    pub fn load(model_path: &std::path::Path) -> Result<Self, VoiceError> {
        let path = model_path
            .to_str()
            .ok_or_else(|| VoiceError::Stt("model path is not valid UTF-8".into()))?;
        let ctx = whisper_rs::WhisperContext::new_with_params(
            path,
            whisper_rs::WhisperContextParameters::default(),
        )
        .map_err(|e| VoiceError::Stt(format!("load model: {e}")))?;
        let n_threads = std::thread::available_parallelism()
            .map(|n| (n.get() as i32 - 2).max(2))
            .unwrap_or(4);
        Ok(Self { ctx, n_threads })
    }

    /// Blocking greedy decode. `Ok(None)` means the post-filter discarded
    /// the result (silence/hallucination/too short).
    pub fn transcribe(&self, samples: &[f32]) -> Result<Option<String>, VoiceError> {
        let mut state = self
            .ctx
            .create_state()
            .map_err(|e| VoiceError::Stt(format!("create state: {e}")))?;

        // Beam search buys a real accuracy step over greedy on short
        // commands; with Metal-accelerated turbo/base models the extra
        // decode cost stays well under perceived latency.
        let mut params = whisper_rs::FullParams::new(whisper_rs::SamplingStrategy::BeamSearch {
            beam_size: 5,
            patience: -1.0,
        });
        params.set_language(Some("en"));
        params.set_translate(false);
        params.set_n_threads(self.n_threads);
        params.set_no_context(true);
        params.set_single_segment(true);
        params.set_no_timestamps(true);
        params.set_token_timestamps(false);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_suppress_blank(true);
        // Non-speech token suppression ("nst"): kills (music), [laughter], …
        params.set_suppress_nst(true);

        let padded;
        let data: &[f32] = if samples.len() < MIN_DECODE_SAMPLES {
            padded = {
                let mut v = samples.to_vec();
                v.resize(MIN_DECODE_SAMPLES, 0.0);
                v
            };
            &padded
        } else {
            samples
        };

        state
            .full(params, data)
            .map_err(|e| VoiceError::Stt(format!("decode: {e}")))?;

        let mut text = String::new();
        for i in 0..state.full_n_segments() {
            if let Some(segment) = state.get_segment(i) {
                if let Ok(s) = segment.to_str_lossy() {
                    text.push_str(&s);
                }
            }
        }
        Ok(post_filter(&text))
    }
}

/// Spawns the `screenie-voice-stt` thread. Exits when the utterance channel
/// closes (segmenter teardown) or `stop` is observed between utterances; an
/// utterance already mid-decode finishes (its transcript still routes — the
/// dispatcher's kill-word check must see "stop listening" itself).
pub fn spawn_stt(
    stop: Arc<AtomicBool>,
    model_path: std::path::PathBuf,
    utterance_rx: mpsc::Receiver<Utterance>,
    mut on_transcript: impl FnMut(Utterance, String) + Send + 'static,
    on_status: impl Fn(VoiceStatus) + Send + 'static,
    on_error: impl Fn(VoiceError) + Send + 'static,
) -> Result<std::thread::JoinHandle<()>, VoiceError> {
    std::thread::Builder::new()
        .name("screenie-voice-stt".into())
        .spawn(move || {
            let mut engine: Option<SttEngine> = None;
            while let Ok(utterance) = utterance_rx.recv() {
                if stop.load(Ordering::Relaxed) {
                    break;
                }
                // Covers both the one-time model load and the decode; the UI
                // shows a brief spinner either way.
                on_status(VoiceStatus::Transcribing);
                if engine.is_none() {
                    let load_started = std::time::Instant::now();
                    match SttEngine::load(&model_path) {
                        Ok(e) => {
                            eprintln!(
                                "[screenie] stt: model loaded in {:.2?}",
                                load_started.elapsed()
                            );
                            engine = Some(e);
                        }
                        Err(e) => {
                            on_error(e);
                            on_status(VoiceStatus::Listening);
                            continue;
                        }
                    }
                }
                let result = engine
                    .as_ref()
                    .expect("engine loaded above")
                    .transcribe(&utterance.samples);
                if !stop.load(Ordering::Relaxed) {
                    on_status(VoiceStatus::Listening);
                }
                match result {
                    Ok(Some(text)) => {
                        // Transcripts are user speech — log size/latency,
                        // never the words (stderr lands in Console.app).
                        eprintln!(
                            "[screenie] stt latency {:.2?} for {:.1}s audio ({} chars)",
                            utterance.captured_at.elapsed(),
                            utterance.samples.len() as f32 / SAMPLE_RATE as f32,
                            text.chars().count()
                        );
                        on_transcript(utterance, text);
                    }
                    Ok(None) => {
                        eprintln!("[screenie] stt: transcript discarded by post-filter");
                    }
                    Err(e) => on_error(e),
                }
            }
        })
        .map_err(|e| VoiceError::Stt(format!("spawn stt thread: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_and_trims_ordinary_transcripts() {
        assert_eq!(
            post_filter("  open a new tab \n"),
            Some("open a new tab".to_string())
        );
    }

    #[test]
    fn discards_empty_and_too_short() {
        assert_eq!(post_filter(""), None);
        assert_eq!(post_filter("   "), None);
        assert_eq!(post_filter("a"), None);
        assert_eq!(post_filter(" m "), None);
    }

    #[test]
    fn discards_blocklist_matches_case_insensitively() {
        assert_eq!(post_filter("Thank you."), None);
        assert_eq!(post_filter(" THANKS FOR WATCHING "), None);
        assert_eq!(post_filter("You"), None);
        assert_eq!(post_filter("[BLANK_AUDIO]"), None);
        assert_eq!(post_filter("(Music)"), None);
        assert_eq!(post_filter("Bye."), None);
    }

    #[test]
    fn keeps_sentences_that_merely_contain_blocklist_phrases() {
        assert_eq!(
            post_filter("thank you for the report"),
            Some("thank you for the report".to_string())
        );
        assert_eq!(
            post_filter("open you tube"),
            Some("open you tube".to_string())
        );
    }

    /// Full pipeline on real audio: Silero VAD segmentation -> whisper
    /// transcription, no microphone involved. Ignored by default (needs a
    /// downloaded model + wav fixture); run manually with:
    ///   say --file-format=WAVE --data-format=LEF32@16000 -o /tmp/stt.wav \
    ///     "open a new tab [[slnc 1300]] type amazon dot com [[slnc 1300]] third phrase"
    ///   SCREENIE_STT_WAV=/tmp/stt.wav \
    ///   SCREENIE_STT_MODEL="$HOME/Library/Application Support/com.screenieai.app/voice/models/ggml-base.en.bin" \
    ///     cargo test real_whisper -- --ignored --nocapture
    #[test]
    #[ignore = "needs SCREENIE_STT_MODEL + SCREENIE_STT_WAV; see doc comment"]
    fn real_whisper_transcribes_segmented_speech() {
        use crate::voice::segmenter::{
            SegmenterConfig, SegmenterOutput, SileroDetector, UtteranceSegmenter,
        };
        use crate::voice::FRAME_SAMPLES;

        let wav = std::env::var("SCREENIE_STT_WAV").expect("SCREENIE_STT_WAV not set");
        let model = std::env::var("SCREENIE_STT_MODEL").expect("SCREENIE_STT_MODEL not set");
        let samples = crate::voice::read_wav_f32_mono_16k(&wav);

        // Segment with the real VAD.
        let detector = SileroDetector::new().expect("silero init");
        let mut seg = UtteranceSegmenter::new(SegmenterConfig::default(), Box::new(detector));
        let mut out = Vec::new();
        for chunk in samples.chunks(FRAME_SAMPLES) {
            let mut frame = [0.0f32; FRAME_SAMPLES];
            frame[..chunk.len()].copy_from_slice(chunk);
            seg.push(Box::new(frame), &mut out);
        }
        for _ in 0..SegmenterConfig::default().silence_frames + 2 {
            seg.push(Box::new([0.0f32; FRAME_SAMPLES]), &mut out);
        }
        let utterances: Vec<_> = out
            .into_iter()
            .filter_map(|o| match o {
                SegmenterOutput::Utterance(u) => Some(u),
                _ => None,
            })
            .collect();
        assert_eq!(utterances.len(), 3, "expected 3 VAD-segmented utterances");

        // Transcribe each with the real model; measure decode latency.
        let engine = SttEngine::load(std::path::Path::new(&model)).expect("load model");
        let mut transcripts = Vec::new();
        for u in &utterances {
            let started = std::time::Instant::now();
            let text = engine.transcribe(&u.samples).expect("decode");
            eprintln!(
                "decode {:.2?} for {:.1}s audio -> {text:?}",
                started.elapsed(),
                u.samples.len() as f32 / 16_000.0
            );
            transcripts.push(text.unwrap_or_default().to_lowercase());
        }
        assert!(transcripts[0].contains("open a new tab"), "{transcripts:?}");
        assert!(transcripts[1].contains("amazon"), "{transcripts:?}");
        assert!(transcripts[2].contains("third"), "{transcripts:?}");
    }
}
