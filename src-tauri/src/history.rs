//! Capture history persistence.
//!
//! Stores captured PNGs + metadata (provider, model, prompt, AI response,
//! timestamp) in the OS app-data directory so the user can revisit past
//! captures from the chat history pane and the Settings panel.
//!
//! Storage layout:
//!   <app_data>/history/index.json       — array of HistoryEntry records (newest first)
//!   <app_data>/history/<id>.png         — the cropped image bytes
//!   <app_data>/history/<id>.thumb.png   — a 240px-long-edge thumbnail for the list view
//!
//! The full PNG is stored separately from the index so the index stays
//! cheap to read even with many entries. The thumbnail keeps history list
//! rendering snappy without re-decoding the full bitmap.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

const MAX_ENTRIES: usize = 200;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct HistoryEntry {
    pub id: String,
    pub created_at_ms: u64,
    pub provider: String,
    pub model: String,
    pub prompt: String,
    pub response: String,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, thiserror::Error)]
pub enum HistoryError {
    #[error("io: {0}")]
    Io(String),
    #[error("decode: {0}")]
    Decode(String),
}

impl Serialize for HistoryError {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_string())
    }
}

impl From<std::io::Error> for HistoryError {
    fn from(e: std::io::Error) -> Self {
        HistoryError::Io(e.to_string())
    }
}

fn history_dir(app_data: &Path) -> PathBuf {
    app_data.join("history")
}

fn index_path(app_data: &Path) -> PathBuf {
    history_dir(app_data).join("index.json")
}

fn entry_png_path(app_data: &Path, id: &str) -> PathBuf {
    history_dir(app_data).join(format!("{}.png", id))
}

fn entry_thumb_path(app_data: &Path, id: &str) -> PathBuf {
    history_dir(app_data).join(format!("{}.thumb.png", id))
}

pub fn load_index(app_data: &Path) -> Result<Vec<HistoryEntry>, HistoryError> {
    let path = index_path(app_data);
    if !path.exists() {
        return Ok(Vec::new());
    }
    let bytes = std::fs::read(&path)?;
    serde_json::from_slice(&bytes).map_err(|e| HistoryError::Decode(e.to_string()))
}

fn write_index(app_data: &Path, index: &[HistoryEntry]) -> Result<(), HistoryError> {
    std::fs::create_dir_all(history_dir(app_data))?;
    let json = serde_json::to_vec_pretty(index).map_err(|e| HistoryError::Decode(e.to_string()))?;
    let tmp = index_path(app_data).with_extension("json.tmp");
    std::fs::write(&tmp, &json)?;
    std::fs::rename(&tmp, index_path(app_data))?;
    Ok(())
}

pub struct AddArgs {
    pub png_b64: String,
    pub width: u32,
    pub height: u32,
    pub provider: String,
    pub model: String,
    pub prompt: String,
    pub response: String,
}

pub fn add_entry(app_data: &Path, args: AddArgs) -> Result<HistoryEntry, HistoryError> {
    use base64::{engine::general_purpose::STANDARD, Engine as _};

    let id = uuid::Uuid::new_v4().to_string();
    let dir = history_dir(app_data);
    std::fs::create_dir_all(&dir)?;

    let bytes = STANDARD
        .decode(&args.png_b64)
        .map_err(|e| HistoryError::Decode(e.to_string()))?;
    std::fs::write(entry_png_path(app_data, &id), &bytes)?;

    // Thumbnail: 240px on the long edge. On encoder failure, skip the thumb
    // file rather than fall back to the full PNG — a 5K screenshot can be
    // 10+ MiB, and loading 200 of those into the history list view would
    // freeze Settings. `load_thumb_b64` returns Io error for missing file;
    // the frontend renders a placeholder for that case.
    match make_thumbnail(&bytes, 240) {
        Ok(thumb) => {
            std::fs::write(entry_thumb_path(app_data, &id), &thumb)?;
        }
        Err(e) => {
            eprintln!("[screenie] history thumb failed (skipping thumb file): {}", e);
        }
    }

    let entry = HistoryEntry {
        id,
        created_at_ms: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0),
        provider: args.provider,
        model: args.model,
        prompt: args.prompt,
        response: args.response,
        width: args.width,
        height: args.height,
    };

    let mut index = load_index(app_data).unwrap_or_default();
    index.insert(0, entry.clone());
    if index.len() > MAX_ENTRIES {
        // Drop the tail entries' files too to keep disk usage bounded.
        for old in index.drain(MAX_ENTRIES..) {
            let _ = std::fs::remove_file(entry_png_path(app_data, &old.id));
            let _ = std::fs::remove_file(entry_thumb_path(app_data, &old.id));
        }
    }
    write_index(app_data, &index)?;

    Ok(entry)
}

pub fn delete_entry(app_data: &Path, id: &str) -> Result<(), HistoryError> {
    let mut index = load_index(app_data).unwrap_or_default();
    let before = index.len();
    index.retain(|e| e.id != id);
    if index.len() < before {
        let _ = std::fs::remove_file(entry_png_path(app_data, id));
        let _ = std::fs::remove_file(entry_thumb_path(app_data, id));
        write_index(app_data, &index)?;
    }
    Ok(())
}

pub fn clear_all(app_data: &Path) -> Result<(), HistoryError> {
    let dir = history_dir(app_data);
    if dir.exists() {
        // Best-effort: clear all files inside; ignore any individual errors.
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for e in entries.flatten() {
                let _ = std::fs::remove_file(e.path());
            }
        }
    }
    Ok(())
}

/// Read the full PNG bytes for a given entry, base64-encoded — used by the
/// "open this entry in chat" flow to re-hydrate the cropped image.
pub fn load_image_b64(app_data: &Path, id: &str) -> Result<String, HistoryError> {
    use base64::{engine::general_purpose::STANDARD, Engine as _};
    let bytes = std::fs::read(entry_png_path(app_data, id))?;
    Ok(STANDARD.encode(&bytes))
}

/// Read the thumbnail bytes, base64-encoded — used by the history list view.
pub fn load_thumb_b64(app_data: &Path, id: &str) -> Result<String, HistoryError> {
    use base64::{engine::general_purpose::STANDARD, Engine as _};
    let bytes = std::fs::read(entry_thumb_path(app_data, id))?;
    Ok(STANDARD.encode(&bytes))
}

fn make_thumbnail(src_png: &[u8], max_long_edge: u32) -> Result<Vec<u8>, String> {
    let img = image::load_from_memory_with_format(src_png, image::ImageFormat::Png)
        .map_err(|e| e.to_string())?;
    let (w, h) = (img.width(), img.height());
    let long = w.max(h);
    let scale = if long > max_long_edge {
        max_long_edge as f64 / long as f64
    } else {
        1.0
    };
    let nw = ((w as f64) * scale).round().max(1.0) as u32;
    let nh = ((h as f64) * scale).round().max(1.0) as u32;
    let resized = if scale == 1.0 {
        img
    } else {
        img.resize_exact(nw, nh, image::imageops::FilterType::Triangle)
    };
    let mut out = Vec::with_capacity((nw * nh * 4) as usize);
    resized
        .write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
        .map_err(|e| e.to_string())?;
    Ok(out)
}
