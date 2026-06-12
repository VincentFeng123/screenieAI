//! Single-writer ingestion for the perception index.
//!
//! One named thread owns the write connection; every mutation goes through
//! its channel and acks back, so writes serialize without ever blocking a
//! reader (WAL). Retention runs on open and after each ingest: latest 20
//! snapshots per (app_bundle, window_id), 24h hard expiry, PNGs deleted
//! with their rows.

use crate::perception::ids::ElementRow;
use crate::perception::{PerceptionError, RectPt};
use rusqlite::{params, Connection};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Snapshots kept per (app_bundle, window_id).
const RETAIN_PER_WINDOW: usize = 20;
/// Hard expiry for snapshot rows and their PNGs.
const RETAIN_MAX_AGE_MS: i64 = 24 * 60 * 60 * 1000;

/// Everything the snapshots row carries besides the element rows.
#[derive(Clone, Debug, Default)]
pub struct SnapshotMeta {
    pub id: String,
    pub ns: String,
    pub created_at_ms: i64,
    pub app_bundle: Option<String>,
    pub app_pid: Option<i32>,
    pub window_id: Option<u32>,
    pub window_title: Option<String>,
    pub display_id: Option<u32>,
    pub scale: f64,
    /// Capture rect in points (the annotation renderer's origin).
    pub cap_rect: Option<RectPt>,
    pub raw_png: Option<String>,
    pub marked_png: Option<String>,
    pub element_count: u32,
    pub took_ms: u64,
    pub partial: bool,
    /// Diff region from the change signal (v2 partial re-walk plumbing).
    pub diff_region: Option<RectPt>,
}

pub fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

enum Job {
    Ingest {
        meta: Box<SnapshotMeta>,
        rows: Vec<ElementRow>,
        ack: mpsc::Sender<Result<(), String>>,
    },
    SetPngPaths {
        snapshot_id: String,
        raw_png: Option<String>,
        marked_png: Option<String>,
        /// Real capture scale (px/pt), measured at look time — see ingests
        /// snapshots before any capture exists, so scale lands here.
        scale: Option<f64>,
        ack: mpsc::Sender<Result<(), String>>,
    },
    MarkDirty {
        snapshot_id: String,
    },
}

/// Handle to the index store: cheap to clone, all writes funnel to the one
/// writer thread.
#[derive(Clone)]
pub struct Store {
    sender: mpsc::Sender<Job>,
    db_path: PathBuf,
    frames_dir: PathBuf,
}

impl Store {
    /// Open (creating dirs/schema as needed) under
    /// `<base>/perception/{snapshots.db, frames/}` and run startup retention.
    pub fn open(base_dir: &Path) -> Result<Self, PerceptionError> {
        let perception_dir = base_dir.join("perception");
        let frames_dir = perception_dir.join("frames");
        std::fs::create_dir_all(&frames_dir)
            .map_err(|err| PerceptionError::Index(format!("create perception dir: {err}")))?;
        let db_path = perception_dir.join("snapshots.db");

        let conn = Connection::open(&db_path)
            .map_err(|err| PerceptionError::Index(format!("open index db: {err}")))?;
        super::schema::migrate(&conn)
            .map_err(|err| PerceptionError::Index(format!("migrate index db: {err}")))?;
        let startup_purge = purge(&conn, now_ms());
        if let Err(err) = startup_purge {
            eprintln!("[screenie] perception index startup purge failed: {err}");
        }

        let (sender, receiver) = mpsc::channel::<Job>();
        std::thread::Builder::new()
            .name("screenie-perception-store".into())
            .spawn(move || writer_loop(conn, receiver))
            .map_err(|err| PerceptionError::Index(format!("spawn store writer: {err}")))?;

        Ok(Self {
            sender,
            db_path,
            frames_dir,
        })
    }

    pub fn db_path(&self) -> &Path {
        &self.db_path
    }

    pub fn frames_dir(&self) -> &Path {
        &self.frames_dir
    }

    /// Ingest a snapshot (meta + element rows + FTS rows, one transaction).
    /// Synchronous ack: when this returns Ok the snapshot is queryable.
    pub fn ingest(&self, meta: SnapshotMeta, rows: Vec<ElementRow>) -> Result<(), PerceptionError> {
        let (ack, done) = mpsc::channel();
        self.sender
            .send(Job::Ingest {
                meta: Box::new(meta),
                rows,
                ack,
            })
            .map_err(|_| PerceptionError::Index("store writer thread is gone".into()))?;
        done.recv()
            .map_err(|_| PerceptionError::Index("store writer dropped the ack".into()))?
            .map_err(PerceptionError::Index)
    }

    /// Record rendered PNG paths (and the measured capture scale) on an
    /// existing snapshot row.
    pub fn set_png_paths(
        &self,
        snapshot_id: &str,
        raw_png: Option<String>,
        marked_png: Option<String>,
        scale: Option<f64>,
    ) -> Result<(), PerceptionError> {
        let (ack, done) = mpsc::channel();
        self.sender
            .send(Job::SetPngPaths {
                snapshot_id: snapshot_id.to_string(),
                raw_png,
                marked_png,
                scale,
                ack,
            })
            .map_err(|_| PerceptionError::Index("store writer thread is gone".into()))?;
        done.recv()
            .map_err(|_| PerceptionError::Index("store writer dropped the ack".into()))?
            .map_err(PerceptionError::Index)
    }

    /// Durable side of the dirty flag (the hot side lives in the registry).
    /// Fire-and-forget: freshness never blocks on the writer.
    pub fn mark_dirty(&self, snapshot_id: &str) {
        let _ = self.sender.send(Job::MarkDirty {
            snapshot_id: snapshot_id.to_string(),
        });
    }
}

fn writer_loop(mut conn: Connection, receiver: mpsc::Receiver<Job>) {
    while let Ok(job) = receiver.recv() {
        match job {
            Job::Ingest { meta, rows, ack } => {
                let result = ingest_txn(&mut conn, &meta, &rows)
                    .and_then(|_| retain_window(&conn, &meta))
                    .map_err(|err| err.to_string());
                let _ = ack.send(result);
            }
            Job::SetPngPaths {
                snapshot_id,
                raw_png,
                marked_png,
                scale,
                ack,
            } => {
                let result = conn
                    .execute(
                        "UPDATE snapshots SET raw_png = COALESCE(?2, raw_png),
                                              marked_png = COALESCE(?3, marked_png),
                                              scale = COALESCE(?4, scale)
                         WHERE id = ?1",
                        params![snapshot_id, raw_png, marked_png, scale],
                    )
                    .map(|_| ())
                    .map_err(|err| err.to_string());
                let _ = ack.send(result);
            }
            Job::MarkDirty { snapshot_id } => {
                let _ = conn.execute(
                    "UPDATE snapshots SET dirty = 1 WHERE id = ?1",
                    params![snapshot_id],
                );
            }
        }
    }
}

fn ingest_txn(
    conn: &mut Connection,
    meta: &SnapshotMeta,
    rows: &[ElementRow],
) -> rusqlite::Result<()> {
    let txn = conn.transaction()?;
    txn.execute(
        "INSERT OR REPLACE INTO snapshots(
            id, ns, created_at, app_bundle, app_pid, window_id, window_title,
            display_id, scale, cap_x, cap_y, cap_w, cap_h, raw_png, marked_png,
            element_count, took_ms, partial, dirty,
            diff_x, diff_y, diff_w, diff_h)
         VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,0,?19,?20,?21,?22)",
        params![
            meta.id,
            meta.ns,
            meta.created_at_ms,
            meta.app_bundle,
            meta.app_pid,
            meta.window_id,
            meta.window_title,
            meta.display_id,
            meta.scale,
            meta.cap_rect.map(|r| r.x),
            meta.cap_rect.map(|r| r.y),
            meta.cap_rect.map(|r| r.w),
            meta.cap_rect.map(|r| r.h),
            meta.raw_png,
            meta.marked_png,
            meta.element_count,
            meta.took_ms as i64,
            meta.partial,
            meta.diff_region.map(|r| r.x),
            meta.diff_region.map(|r| r.y),
            meta.diff_region.map(|r| r.w),
            meta.diff_region.map(|r| r.h),
        ],
    )?;

    {
        let mut element_stmt = txn.prepare_cached(
            "INSERT OR REPLACE INTO elements(
                snapshot_id, eid, fp, class, role, subrole, title, descr, value,
                actionable, enabled, focused, x, y, w, h, depth, parent_eid,
                actions, is_web_boundary)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,?20)",
        )?;
        let mut fts_stmt = txn.prepare_cached(
            "INSERT INTO elements_fts(snapshot_id, eid, title, descr, value)
             VALUES (?1,?2,?3,?4,?5)",
        )?;
        for row in rows {
            // fp is u64; SQLite stores i64 — bit-cast both ways.
            element_stmt.execute(params![
                meta.id,
                row.eid,
                row.fp as i64,
                row.class_prefix.to_string(),
                row.role,
                row.subrole,
                row.title,
                row.descr,
                row.value,
                row.actionable,
                row.enabled,
                row.focused,
                row.frame.x,
                row.frame.y,
                row.frame.w,
                row.frame.h,
                row.depth,
                row.parent_eid,
                serde_json::to_string(&row.actions).unwrap_or_else(|_| "[]".into()),
                row.is_web_boundary,
            ])?;
            if row.title.is_some() || row.descr.is_some() || row.value.is_some() {
                fts_stmt.execute(params![meta.id, row.eid, row.title, row.descr, row.value,])?;
            }
        }
    }
    txn.commit()
}

/// Keep only the newest [`RETAIN_PER_WINDOW`] snapshots of this snapshot's
/// window, deleting evicted rows' PNGs and FTS entries.
fn retain_window(conn: &Connection, meta: &SnapshotMeta) -> rusqlite::Result<()> {
    let evicted: Vec<(String, Option<String>, Option<String>)> = {
        let mut stmt = conn.prepare_cached(
            "SELECT id, raw_png, marked_png FROM snapshots
             WHERE (app_bundle IS ?1 OR app_bundle = ?1)
               AND (window_id IS ?2 OR window_id = ?2)
             ORDER BY created_at DESC
             LIMIT -1 OFFSET ?3",
        )?;
        let collected = stmt
            .query_map(
                params![meta.app_bundle, meta.window_id, RETAIN_PER_WINDOW],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )?
            .collect::<rusqlite::Result<_>>()?;
        collected
    };
    delete_snapshots(conn, &evicted)
}

/// Startup retention: hard 24h expiry.
fn purge(conn: &Connection, now_ms: i64) -> rusqlite::Result<()> {
    let cutoff = now_ms - RETAIN_MAX_AGE_MS;
    let expired: Vec<(String, Option<String>, Option<String>)> = {
        let mut stmt =
            conn.prepare("SELECT id, raw_png, marked_png FROM snapshots WHERE created_at < ?1")?;
        let collected = stmt
            .query_map(params![cutoff], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })?
            .collect::<rusqlite::Result<_>>()?;
        collected
    };
    delete_snapshots(conn, &expired)
}

fn delete_snapshots(
    conn: &Connection,
    snapshots: &[(String, Option<String>, Option<String>)],
) -> rusqlite::Result<()> {
    for (id, raw_png, marked_png) in snapshots {
        // CASCADE removes elements; FTS has no FK support, delete explicitly.
        conn.execute(
            "DELETE FROM elements_fts WHERE snapshot_id = ?1",
            params![id],
        )?;
        conn.execute("DELETE FROM snapshots WHERE id = ?1", params![id])?;
        for png in [raw_png, marked_png].into_iter().flatten() {
            // PNGs live with their rows (C6 retention); missing files are fine.
            let _ = std::fs::remove_file(png);
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn test_row(eid: &str, title: &str) -> ElementRow {
        ElementRow {
            eid: eid.into(),
            fp: 7,
            role: "AXButton".into(),
            subrole: None,
            title: Some(title.into()),
            descr: None,
            value: None,
            actions: vec!["AXPress".into()],
            actionable: true,
            enabled: true,
            focused: false,
            frame: RectPt::new(10.0, 20.0, 30.0, 40.0),
            depth: 2,
            parent_eid: None,
            is_web_boundary: false,
            class_prefix: 'B',
        }
    }

    pub(crate) fn test_meta(id: &str, window_id: u32, created_at_ms: i64) -> SnapshotMeta {
        SnapshotMeta {
            id: id.into(),
            ns: "ax".into(),
            created_at_ms,
            app_bundle: Some("com.test.app".into()),
            app_pid: Some(123),
            window_id: Some(window_id),
            window_title: Some("Test".into()),
            scale: 2.0,
            element_count: 1,
            took_ms: 5,
            ..SnapshotMeta::default()
        }
    }

    pub(crate) fn temp_base() -> PathBuf {
        let dir = std::env::temp_dir()
            .join("screenie-perception-tests")
            .join(uuid::Uuid::new_v4().simple().to_string());
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn ingest_then_retention_keeps_latest_20() {
        let base = temp_base();
        let store = Store::open(&base).unwrap();
        for index in 0..(RETAIN_PER_WINDOW as i64 + 3) {
            let meta = test_meta(&format!("ax:retain{index:03}"), 42, 1000 + index);
            store.ingest(meta, vec![test_row("ax:B1", "Save")]).unwrap();
        }
        let conn = Connection::open(store.db_path()).unwrap();
        let count: i64 = conn
            .query_row("SELECT count(*) FROM snapshots", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, RETAIN_PER_WINDOW as i64);
        // Oldest evicted, newest kept.
        let oldest: String = conn
            .query_row(
                "SELECT id FROM snapshots ORDER BY created_at ASC LIMIT 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(oldest, "ax:retain003");
        // Elements and FTS rows follow their snapshots out.
        let orphan_elements: i64 = conn
            .query_row(
                "SELECT count(*) FROM elements WHERE snapshot_id NOT IN (SELECT id FROM snapshots)",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let orphan_fts: i64 = conn
            .query_row(
                "SELECT count(*) FROM elements_fts WHERE snapshot_id NOT IN (SELECT id FROM snapshots)",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!((orphan_elements, orphan_fts), (0, 0));
        std::fs::remove_dir_all(base).ok();
    }

    #[test]
    fn startup_purge_expires_old_rows() {
        let base = temp_base();
        {
            let store = Store::open(&base).unwrap();
            let stale = test_meta("ax:ancient01", 7, now_ms() - RETAIN_MAX_AGE_MS - 1000);
            let fresh = test_meta("ax:current01", 7, now_ms());
            store.ingest(stale, vec![test_row("ax:B1", "Old")]).unwrap();
            store.ingest(fresh, vec![test_row("ax:B1", "New")]).unwrap();
        }
        // Reopen: startup purge drops the expired row.
        let store = Store::open(&base).unwrap();
        let conn = Connection::open(store.db_path()).unwrap();
        let ids: Vec<String> = conn
            .prepare("SELECT id FROM snapshots ORDER BY created_at")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap();
        assert_eq!(ids, vec!["ax:current01".to_string()]);
        std::fs::remove_dir_all(base).ok();
    }
}
