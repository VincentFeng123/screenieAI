//! Whisper model manager (download, presence/size verification) and
//! `voice.json` config persistence. Models live in
//! `<app_data>/voice/models/`; config follows the hotkeys.json pattern
//! (read-with-fallback, clamp on load, `.tmp` + rename on save).

use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::time::Duration;

use futures_util::StreamExt;
use tauri::AppHandle;
use tokio::io::AsyncWriteExt;

use super::{emit_voice, VoiceConfig, VoiceError, WhisperModel};
use crate::ai::CancelFlag;

pub const MODEL_BASE_URL: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/";
pub const VOICE_CONFIG_FILE: &str = "voice.json";

pub struct ModelSpec {
    pub model: WhisperModel,
    pub file_name: &'static str,
    /// Generous static bounds (the upstream files are ~75 / ~142 / ~466 /
    /// ~574 MB); the exact `Content-Length` is also enforced when the server
    /// sends it.
    pub min_bytes: u64,
    pub max_bytes: u64,
}

pub const MODEL_SPECS: &[ModelSpec] = &[
    ModelSpec {
        model: WhisperModel::TinyEn,
        file_name: "ggml-tiny.en.bin",
        min_bytes: 70_000_000,
        max_bytes: 90_000_000,
    },
    ModelSpec {
        model: WhisperModel::BaseEn,
        file_name: "ggml-base.en.bin",
        min_bytes: 135_000_000,
        max_bytes: 165_000_000,
    },
    ModelSpec {
        model: WhisperModel::SmallEn,
        file_name: "ggml-small.en.bin",
        min_bytes: 450_000_000,
        max_bytes: 510_000_000,
    },
    ModelSpec {
        model: WhisperModel::LargeV3Turbo,
        // Upstream file is exactly 574,041,195 bytes today.
        file_name: "ggml-large-v3-turbo-q5_0.bin",
        min_bytes: 540_000_000,
        max_bytes: 610_000_000,
    },
];

impl WhisperModel {
    pub fn spec(self) -> &'static ModelSpec {
        MODEL_SPECS
            .iter()
            .find(|s| s.model == self)
            .expect("every WhisperModel variant has a spec")
    }

    pub fn wire_name(self) -> &'static str {
        match self {
            WhisperModel::TinyEn => "tiny.en",
            WhisperModel::BaseEn => "base.en",
            WhisperModel::SmallEn => "small.en",
            WhisperModel::LargeV3Turbo => "large-v3-turbo",
        }
    }
}

/// True when `len` looks like a complete download for `spec`: within the
/// static bounds AND equal to the server-reported length when one is known.
pub fn download_size_ok(spec: &ModelSpec, len: u64, content_length: Option<u64>) -> bool {
    let bounds_ok = (spec.min_bytes..=spec.max_bytes).contains(&len);
    let server_ok = content_length.is_none_or(|cl| cl == len);
    bounds_ok && server_ok
}

/// Parse + clamp `voice.json` content; malformed or partial input falls back
/// to defaults so a stale or hand-edited file can never wedge the pipeline.
pub fn config_from_json(text: &str) -> VoiceConfig {
    serde_json::from_str::<VoiceConfig>(text)
        // merged_with(None, …) re-clamps every field on the way in.
        .map(|cfg| cfg.merged_with(None, None, None))
        .unwrap_or_default()
}

pub fn models_dir(app: &AppHandle) -> Result<PathBuf, String> {
    Ok(crate::app_data_dir(app)?.join("voice").join("models"))
}

pub fn model_path(app: &AppHandle, model: WhisperModel) -> Result<PathBuf, String> {
    Ok(models_dir(app)?.join(model.spec().file_name))
}

/// Present means "exists with a plausible complete size" — a truncated or
/// corrupted file reports missing and triggers a re-download.
pub fn is_model_present(app: &AppHandle, model: WhisperModel) -> bool {
    let Ok(path) = model_path(app, model) else {
        return false;
    };
    match std::fs::metadata(path) {
        Ok(meta) => download_size_ok(model.spec(), meta.len(), None),
        Err(_) => false,
    }
}

pub fn load_voice_config(app: &AppHandle) -> VoiceConfig {
    let Ok(dir) = crate::app_data_dir(app) else {
        return VoiceConfig::default();
    };
    match std::fs::read_to_string(dir.join(VOICE_CONFIG_FILE)) {
        Ok(text) => config_from_json(&text),
        Err(_) => VoiceConfig::default(),
    }
}

pub fn save_voice_config(app: &AppHandle, cfg: &VoiceConfig) -> Result<(), String> {
    let dir = crate::app_data_dir(app)?;
    std::fs::create_dir_all(&dir).map_err(|e| format!("create app data dir: {e}"))?;
    let path = dir.join(VOICE_CONFIG_FILE);
    let tmp = path.with_extension("json.tmp");
    let json =
        serde_json::to_vec_pretty(cfg).map_err(|e| format!("serialize voice config: {e}"))?;
    std::fs::write(&tmp, json).map_err(|e| format!("write voice config: {e}"))?;
    std::fs::rename(&tmp, &path).map_err(|e| format!("save voice config: {e}"))?;
    Ok(())
}

/// Streams the model from Hugging Face to `<file>.part`, emitting
/// `voice:model_download { pct, model }` on integer-percent changes, then
/// verifies the size and renames into place. Any failure (including cancel)
/// removes the partial file.
pub async fn download_model(
    app: AppHandle,
    model: WhisperModel,
    cancel: CancelFlag,
) -> Result<(), VoiceError> {
    let spec = model.spec();
    let dir = models_dir(&app).map_err(VoiceError::Download)?;
    tokio::fs::create_dir_all(&dir)
        .await
        .map_err(|e| VoiceError::Download(format!("create models dir: {e}")))?;
    let final_path = dir.join(spec.file_name);
    let part_path = final_path.with_extension("bin.part");

    let result = stream_to_part(&app, model, spec, &part_path, &cancel).await;
    let (len, content_length) = match result {
        Ok(v) => v,
        Err(e) => {
            let _ = tokio::fs::remove_file(&part_path).await;
            return Err(e);
        }
    };

    if !download_size_ok(spec, len, content_length) {
        let _ = tokio::fs::remove_file(&part_path).await;
        return Err(VoiceError::Download(format!(
            "unexpected size {len} bytes for {} (expected {}..{}, server said {:?})",
            spec.file_name, spec.min_bytes, spec.max_bytes, content_length
        )));
    }
    tokio::fs::rename(&part_path, &final_path)
        .await
        .map_err(|e| VoiceError::Download(format!("finalize model file: {e}")))?;
    eprintln!(
        "[screenie] voice model {} downloaded ({len} bytes)",
        spec.file_name
    );
    Ok(())
}

async fn stream_to_part(
    app: &AppHandle,
    model: WhisperModel,
    spec: &ModelSpec,
    part_path: &std::path::Path,
    cancel: &CancelFlag,
) -> Result<(u64, Option<u64>), VoiceError> {
    let url = format!("{MODEL_BASE_URL}{}", spec.file_name);
    // Deliberately NOT ai::cloud_client(): that client disables redirects,
    // and Hugging Face resolves model files through a CDN redirect.
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .read_timeout(Duration::from_secs(60))
        .timeout(Duration::from_secs(1800))
        .build()
        .map_err(|e| VoiceError::Download(e.to_string()))?;
    let response = client
        .get(&url)
        .send()
        .await
        .map_err(|e| VoiceError::Download(e.to_string()))?;
    if !response.status().is_success() {
        return Err(VoiceError::Download(format!(
            "download failed: HTTP {}",
            response.status()
        )));
    }
    let content_length = response.content_length();

    let mut file = tokio::fs::File::create(part_path)
        .await
        .map_err(|e| VoiceError::Download(format!("create part file: {e}")))?;
    let mut stream = response.bytes_stream();
    let mut downloaded: u64 = 0;
    let mut last_pct: i32 = -1;
    let wire = model.wire_name();
    while let Some(chunk) = stream.next().await {
        if cancel.load(Ordering::Relaxed) {
            return Err(VoiceError::Download("download cancelled".into()));
        }
        let chunk = chunk.map_err(|e| VoiceError::Download(e.to_string()))?;
        file.write_all(&chunk)
            .await
            .map_err(|e| VoiceError::Download(format!("write model file: {e}")))?;
        downloaded += chunk.len() as u64;
        let pct = match content_length {
            Some(total) if total > 0 => ((downloaded * 100) / total).min(100) as i32,
            _ => 0,
        };
        // Throttle UI updates: only emit on integer-percent changes.
        if pct > last_pct {
            last_pct = pct;
            emit_voice(
                app,
                "voice:model_download",
                serde_json::json!({ "pct": pct, "model": wire }),
            );
        }
    }
    file.flush()
        .await
        .map_err(|e| VoiceError::Download(format!("flush model file: {e}")))?;
    drop(file);
    Ok((downloaded, content_length))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_spec() -> &'static ModelSpec {
        WhisperModel::BaseEn.spec()
    }

    #[test]
    fn every_model_has_a_spec_and_wire_name() {
        for m in [
            WhisperModel::TinyEn,
            WhisperModel::BaseEn,
            WhisperModel::SmallEn,
            WhisperModel::LargeV3Turbo,
        ] {
            assert_eq!(m.spec().model, m);
            assert_eq!(WhisperModel::parse(m.wire_name()), Some(m));
        }
    }

    #[test]
    fn size_within_bounds_passes_without_content_length() {
        assert!(download_size_ok(base_spec(), 142_000_000, None));
    }

    #[test]
    fn size_outside_bounds_fails() {
        assert!(!download_size_ok(base_spec(), 10_000, None)); // truncated
        assert!(!download_size_ok(base_spec(), 999_000_000, None)); // wrong file
    }

    #[test]
    fn content_length_must_match_when_known() {
        assert!(download_size_ok(base_spec(), 142_000_000, Some(142_000_000)));
        assert!(!download_size_ok(base_spec(), 142_000_000, Some(142_000_001)));
    }

    #[test]
    fn config_round_trips_through_json() {
        let cfg = VoiceConfig {
            silence_ms: 450,
            model: WhisperModel::TinyEn,
            auto_stop_s: 120,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        assert_eq!(config_from_json(&json), cfg);
    }

    #[test]
    fn malformed_config_falls_back_to_defaults() {
        assert_eq!(config_from_json("not json"), VoiceConfig::default());
        assert_eq!(config_from_json(""), VoiceConfig::default());
        // Unknown model variant fails the parse entirely -> defaults.
        assert_eq!(
            config_from_json(r#"{"silenceMs":700,"model":"huge.en","autoStopS":90}"#),
            VoiceConfig::default()
        );
    }

    #[test]
    fn out_of_range_config_values_are_clamped_on_load() {
        let cfg =
            config_from_json(r#"{"silenceMs":5000,"model":"base.en","autoStopS":5}"#);
        assert_eq!(cfg.silence_ms, 1000);
        assert_eq!(cfg.auto_stop_s, 10);
    }
}
