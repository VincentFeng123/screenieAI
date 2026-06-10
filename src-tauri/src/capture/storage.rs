//! On-disk layout for capture artifacts.
//!
//! `<app_data>/captures/` (history.rs precedent — app data, never an
//! iCloud-synced location):
//!   frame_{yyyymmdd-hhmmss}_{8hex}.png   persisted perception frames
//!   clip_{yyyymmdd-hhmmss}_{8hex}.mp4    finished recordings
//!   .inprogress/<same name>              recordings still being written —
//!     an unfinalized mp4 has no moov atom and is unplayable garbage, so
//!     clips only move into captures/ after the writer finalizes; anything
//!     left in .inprogress/ at startup is a crash orphan and is deleted.
//!
//! Timestamps are UTC (machine-generated artifact names; avoids pulling a
//! chrono/time dependency into a repo that deliberately has neither).

#![allow(dead_code)] // consumed incrementally across the capture phases

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use super::CaptureError;

pub(crate) const CAPTURES_DIR: &str = "captures";
pub(crate) const INPROGRESS_DIR: &str = ".inprogress";
/// Default time-to-live for finished artifacts swept at startup.
pub(crate) const DEFAULT_TTL: Duration = Duration::from_secs(24 * 60 * 60);

pub(crate) fn captures_dir(app_data: &Path) -> PathBuf {
    app_data.join(CAPTURES_DIR)
}

pub(crate) fn inprogress_dir(app_data: &Path) -> PathBuf {
    captures_dir(app_data).join(INPROGRESS_DIR)
}

/// Days-since-epoch → (year, month, day). Howard Hinnant's civil-from-days.
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// `yyyymmdd-hhmmss` (UTC) for a unix timestamp.
fn compact_utc(secs_since_epoch: u64) -> String {
    let days = (secs_since_epoch / 86_400) as i64;
    let rem = secs_since_epoch % 86_400;
    let (y, m, d) = civil_from_days(days);
    format!(
        "{y:04}{m:02}{d:02}-{:02}{:02}{:02}",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

fn now_compact_utc() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    compact_utc(secs)
}

fn short_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()[..8].to_string()
}

pub(crate) fn frame_file_name() -> String {
    format!("frame_{}_{}.png", now_compact_utc(), short_id())
}

pub(crate) fn clip_file_name(extension: &str) -> String {
    format!("clip_{}_{}.{extension}", now_compact_utc(), short_id())
}

/// Write a persisted perception frame and return its path.
pub(crate) fn persist_frame(app_data: &Path, png_bytes: &[u8]) -> Result<PathBuf, CaptureError> {
    let dir = captures_dir(app_data);
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(frame_file_name());
    std::fs::write(&path, png_bytes)?;
    Ok(path)
}

/// Reserve paths for a new recording: (in-progress write path, final path).
/// The clip is renamed from the first to the second on successful finalize.
pub(crate) fn new_clip_paths(
    app_data: &Path,
    extension: &str,
) -> Result<(PathBuf, PathBuf), CaptureError> {
    let dir = captures_dir(app_data);
    let staging = inprogress_dir(app_data);
    std::fs::create_dir_all(&staging)?;
    let name = clip_file_name(extension);
    Ok((staging.join(&name), dir.join(&name)))
}

/// Promote a finalized recording out of `.inprogress/`.
pub(crate) fn promote_clip(inprogress: &Path, final_path: &Path) -> Result<(), CaptureError> {
    if let Some(parent) = final_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::rename(inprogress, final_path)?;
    Ok(())
}

/// Startup sweep: delete every `.inprogress/` leftover (crash orphans —
/// never playable) and finished artifacts older than `ttl`. Best-effort:
/// individual failures are logged and skipped so one stuck file can't wedge
/// startup. Never runs concurrently with a recording in this process —
/// callers invoke it once from setup before any session can start.
pub(crate) fn sweep(app_data: &Path, ttl: Duration) {
    let staging = inprogress_dir(app_data);
    if let Ok(entries) = std::fs::read_dir(&staging) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Err(e) = std::fs::remove_file(&path) {
                    eprintln!("[screenie] captures sweep: orphan {path:?}: {e}");
                }
            }
        }
    }

    let dir = captures_dir(app_data);
    let now = SystemTime::now();
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let expired = entry
            .metadata()
            .and_then(|m| m.modified())
            .ok()
            .and_then(|mtime| now.duration_since(mtime).ok())
            .is_some_and(|age| age > ttl);
        if expired {
            if let Err(e) = std::fs::remove_file(&path) {
                eprintln!("[screenie] captures sweep: expired {path:?}: {e}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_utc_known_dates() {
        assert_eq!(compact_utc(0), "19700101-000000");
        // 2026-06-11 12:34:56 UTC
        assert_eq!(compact_utc(1_781_181_296), "20260611-123456");
        // Leap day: 2024-02-29 00:00:00 UTC
        assert_eq!(compact_utc(1_709_164_800), "20240229-000000");
    }

    #[test]
    fn file_names_have_expected_shape() {
        let frame = frame_file_name();
        assert!(frame.starts_with("frame_"), "{frame}");
        assert!(frame.ends_with(".png"), "{frame}");
        assert_eq!(frame.len(), "frame_".len() + 15 + 1 + 8 + ".png".len());

        let clip = clip_file_name("mp4");
        assert!(clip.starts_with("clip_"), "{clip}");
        assert!(clip.ends_with(".mp4"), "{clip}");
    }

    #[test]
    fn sweep_deletes_orphans_and_expired_only() {
        let tmp = std::env::temp_dir().join(format!("screenie-sweep-test-{}", short_id()));
        let dir = captures_dir(&tmp);
        let staging = inprogress_dir(&tmp);
        std::fs::create_dir_all(&staging).unwrap();

        let orphan = staging.join("clip_x.mp4");
        let fresh = dir.join("clip_fresh.mp4");
        std::fs::write(&orphan, b"x").unwrap();
        std::fs::write(&fresh, b"x").unwrap();

        // ttl=0 expires everything with any age; fresh file mtime == now, so
        // use a 1h ttl to keep it and a backdated check via ttl=0 separately.
        sweep(&tmp, Duration::from_secs(3600));
        assert!(!orphan.exists(), "orphan should be deleted regardless of ttl");
        assert!(fresh.exists(), "fresh file should survive ttl sweep");

        // Ensure a measurable mtime age before expiring with ttl=0.
        std::thread::sleep(Duration::from_millis(20));
        sweep(&tmp, Duration::from_secs(0));
        assert!(!fresh.exists(), "ttl=0 expires any non-zero age");

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
