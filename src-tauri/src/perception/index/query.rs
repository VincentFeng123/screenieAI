//! Read side of the perception index: structured queries (the default
//! `read` path) and the restricted raw-SQL escape hatch.
//!
//! Structured reads run on one persistent read-only connection (C1: <5ms).
//! `read_sql` gets a fresh read-only connection per call so its interrupt
//! timer can never cut into anyone else's query, `PRAGMA query_only` on,
//! single SELECT statement only, results capped at 200 rows.

use crate::perception::ids::ElementRow;
use crate::perception::{PerceptionError, RectPt};
use rusqlite::{params_from_iter, Connection, OpenFlags};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Mutex, PoisonError};
use std::time::Duration;

/// Hard row cap for every read path.
pub const MAX_READ_LIMIT: u32 = 200;
/// Default top-K for `read` (the serialization contract's block size).
pub const DEFAULT_READ_LIMIT: u32 = 40;
/// Raw SQL wall clock before the interrupt fires.
const READ_SQL_TIMEOUT: Duration = Duration::from_millis(250);

fn default_limit() -> u32 {
    DEFAULT_READ_LIMIT
}

/// The structured query surface (the default `read` path).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ElementQuery {
    /// Default: latest snapshot for the frontmost window (the command layer
    /// resolves this and auto-sees when missing or dirty — C3).
    pub snapshot_id: Option<String>,
    /// Class letters ("B") or raw AX roles ("AXButton"), mixed freely.
    pub roles: Option<Vec<String>>,
    /// FTS5 match over title/descr/value.
    pub text: Option<String>,
    /// Frame intersects, points.
    pub region: Option<RectPt>,
    pub actionable_only: bool,
    pub focused_only: bool,
    #[serde(default = "default_limit")]
    pub limit: u32,
}

impl Default for ElementQuery {
    fn default() -> Self {
        Self {
            snapshot_id: None,
            roles: None,
            text: None,
            region: None,
            actionable_only: false,
            focused_only: false,
            limit: DEFAULT_READ_LIMIT,
        }
    }
}

/// Structured query result: rows plus enough to build the truncation footer.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct QueryRows {
    pub snapshot_id: String,
    pub rows: Vec<ElementRow>,
    /// Total rows matching the filters before the limit.
    pub total: u64,
}

/// Raw SQL result: column names + JSON values, row-capped.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SqlRows {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<serde_json::Value>>,
    pub truncated: bool,
}

/// The snapshots-table fields a serialized SNAP header needs.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SnapshotHeaderRow {
    pub snapshot_id: String,
    pub ns: String,
    pub app_bundle: Option<String>,
    pub app_pid: Option<i32>,
    pub window_id: Option<u32>,
    pub window_title: Option<String>,
    pub scale: f64,
    pub cap_rect: Option<RectPt>,
    pub element_count: u32,
    pub took_ms: u64,
    pub partial: bool,
}

fn row_to_element(row: &rusqlite::Row<'_>) -> rusqlite::Result<ElementRow> {
    Ok(ElementRow {
        eid: row.get(0)?,
        fp: row.get::<_, i64>(1)? as u64,
        class_prefix: row.get::<_, String>(2)?.chars().next().unwrap_or('X'),
        role: row.get::<_, Option<String>>(3)?.unwrap_or_default(),
        subrole: row.get(4)?,
        title: row.get(5)?,
        descr: row.get(6)?,
        value: row.get(7)?,
        actionable: row.get(8)?,
        enabled: row.get(9)?,
        focused: row.get(10)?,
        frame: RectPt::new(row.get(11)?, row.get(12)?, row.get(13)?, row.get(14)?),
        depth: row.get::<_, i64>(15)?.clamp(0, u8::MAX as i64) as u8,
        parent_eid: row.get(16)?,
        actions: row
            .get::<_, Option<String>>(17)?
            .and_then(|json| serde_json::from_str(&json).ok())
            .unwrap_or_default(),
        is_web_boundary: row.get(18)?,
    })
}

/// Read handle: persistent read-only connection for structured queries.
pub struct Reader {
    db_path: PathBuf,
    conn: Mutex<Connection>,
}

impl Reader {
    pub fn open(db_path: &Path) -> Result<Self, PerceptionError> {
        let conn = open_read_only(db_path)?;
        Ok(Self {
            db_path: db_path.to_path_buf(),
            conn: Mutex::new(conn),
        })
    }

    /// Structured query against one snapshot.
    pub fn query(&self, snapshot_id: &str, q: &ElementQuery) -> Result<QueryRows, PerceptionError> {
        let limit = q.limit.clamp(1, MAX_READ_LIMIT);
        let mut where_sql = String::from("snapshot_id = ?");
        let mut args: Vec<Box<dyn rusqlite::ToSql>> = vec![Box::new(snapshot_id.to_string())];

        if let Some(roles) = q.roles.as_ref().filter(|roles| !roles.is_empty()) {
            let mut classes: Vec<String> = Vec::new();
            let mut raw_roles: Vec<String> = Vec::new();
            for role in roles {
                if role.len() == 1 {
                    classes.push(role.to_uppercase());
                } else {
                    raw_roles.push(role.clone());
                }
            }
            let mut parts: Vec<String> = Vec::new();
            if !classes.is_empty() {
                parts.push(format!("class IN ({})", vec!["?"; classes.len()].join(",")));
                args.extend(
                    classes
                        .into_iter()
                        .map(|c| Box::new(c) as Box<dyn rusqlite::ToSql>),
                );
            }
            if !raw_roles.is_empty() {
                parts.push(format!(
                    "role IN ({})",
                    vec!["?"; raw_roles.len()].join(",")
                ));
                args.extend(
                    raw_roles
                        .into_iter()
                        .map(|r| Box::new(r) as Box<dyn rusqlite::ToSql>),
                );
            }
            where_sql.push_str(&format!(" AND ({})", parts.join(" OR ")));
        }

        if q.actionable_only {
            where_sql.push_str(" AND actionable = 1");
        }
        if q.focused_only {
            where_sql.push_str(" AND focused = 1");
        }
        if let Some(region) = q.region {
            where_sql.push_str(" AND x < ? AND x + w > ? AND y < ? AND y + h > ?");
            args.push(Box::new(region.x + region.w));
            args.push(Box::new(region.x));
            args.push(Box::new(region.y + region.h));
            args.push(Box::new(region.y));
        }
        if let Some(text) = q.text.as_ref().filter(|text| !text.trim().is_empty()) {
            where_sql.push_str(
                " AND eid IN (SELECT eid FROM elements_fts WHERE elements_fts MATCH ? AND snapshot_id = ?)",
            );
            args.push(Box::new(text.clone()));
            args.push(Box::new(snapshot_id.to_string()));
        }

        let conn = self.conn.lock().unwrap_or_else(PoisonError::into_inner);
        let total: u64 = conn
            .query_row(
                &format!("SELECT count(*) FROM elements WHERE {where_sql}"),
                params_from_iter(args.iter().map(|a| a.as_ref())),
                |row| row.get::<_, i64>(0),
            )
            .map(|count| count.max(0) as u64)
            .map_err(map_query_err)?;

        // Salience order (same ranking as marks and see's top-K): focused,
        // then actionable, then class priority, then area, then tree order —
        // so a LIMIT always returns the most useful rows, not the first in
        // tree order.
        let sql = format!(
            "SELECT eid, fp, class, role, subrole, title, descr, value, actionable,
                    enabled, focused, x, y, w, h, depth, parent_eid, actions,
                    is_web_boundary
             FROM elements WHERE {where_sql}
             ORDER BY focused DESC, actionable DESC,
                      CASE class WHEN 'B' THEN 0 WHEN 'T' THEN 1 WHEN 'L' THEN 2
                                 WHEN 'C' THEN 3 WHEN 'M' THEN 4 WHEN 'S' THEN 5
                                 WHEN 'G' THEN 6 WHEN 'I' THEN 7 ELSE 8 END,
                      (w*h) DESC, rowid
             LIMIT {limit}"
        );
        let mut stmt = conn.prepare_cached(&sql).map_err(map_query_err)?;
        let rows = stmt
            .query_map(
                params_from_iter(args.iter().map(|a| a.as_ref())),
                row_to_element,
            )
            .map_err(map_query_err)?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(map_query_err)?;

        Ok(QueryRows {
            snapshot_id: snapshot_id.to_string(),
            rows,
            total,
        })
    }

    /// The snapshots-table row a serialized header needs. `None` when the
    /// snapshot is unknown.
    pub fn snapshot_header(
        &self,
        snapshot_id: &str,
    ) -> Result<Option<SnapshotHeaderRow>, PerceptionError> {
        let conn = self.conn.lock().unwrap_or_else(PoisonError::into_inner);
        let row = conn
            .query_row(
                "SELECT ns, app_bundle, app_pid, window_id, window_title, scale,
                        cap_x, cap_y, cap_w, cap_h, element_count, took_ms, partial
                 FROM snapshots WHERE id = ?1",
                [snapshot_id],
                |row| {
                    Ok(SnapshotHeaderRow {
                        snapshot_id: snapshot_id.to_string(),
                        ns: row.get(0)?,
                        app_bundle: row.get(1)?,
                        app_pid: row.get(2)?,
                        window_id: row.get(3)?,
                        window_title: row.get(4)?,
                        scale: row.get(5)?,
                        cap_rect: match (
                            row.get::<_, Option<f64>>(6)?,
                            row.get::<_, Option<f64>>(7)?,
                            row.get::<_, Option<f64>>(8)?,
                            row.get::<_, Option<f64>>(9)?,
                        ) {
                            (Some(x), Some(y), Some(w), Some(h)) => Some(RectPt::new(x, y, w, h)),
                            _ => None,
                        },
                        element_count: row.get::<_, Option<i64>>(10)?.unwrap_or(0) as u32,
                        took_ms: row.get::<_, Option<i64>>(11)?.unwrap_or(0) as u64,
                        partial: row.get(12)?,
                    })
                },
            )
            .map(Some)
            .or_else(|err| match err {
                rusqlite::Error::QueryReturnedNoRows => Ok(None),
                other => Err(map_query_err(other)),
            })?;
        Ok(row)
    }

    /// Every element row of a snapshot in tree order — the annotation
    /// renderer's input (internal; not limit-clamped like `query`).
    pub fn all_rows(&self, snapshot_id: &str) -> Result<Vec<ElementRow>, PerceptionError> {
        let result = self.query(
            snapshot_id,
            &ElementQuery {
                limit: MAX_READ_LIMIT,
                ..ElementQuery::default()
            },
        )?;
        if result.total <= u64::from(MAX_READ_LIMIT) {
            return Ok(result.rows);
        }
        // Above the structured cap: pull the full set directly (bounded by
        // the walker's own element budget).
        let conn = self.conn.lock().unwrap_or_else(PoisonError::into_inner);
        let mut stmt = conn
            .prepare_cached(
                "SELECT eid, fp, class, role, subrole, title, descr, value, actionable,
                        enabled, focused, x, y, w, h, depth, parent_eid, actions,
                        is_web_boundary
                 FROM elements WHERE snapshot_id = ? ORDER BY rowid",
            )
            .map_err(map_query_err)?;
        let rows = stmt
            .query_map([snapshot_id], row_to_element)
            .map_err(map_query_err)?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(map_query_err)?;
        Ok(rows)
    }

    /// Restricted raw SQL (`read_sql`): fresh read-only connection,
    /// `query_only`, one SELECT, 250ms interrupt, hard 200-row cap via an
    /// outer LIMIT wrapper.
    pub fn read_sql(&self, sql: &str) -> Result<SqlRows, PerceptionError> {
        let trimmed = sql.trim().trim_end_matches(';').trim();
        if trimmed.is_empty() {
            return Err(PerceptionError::InvalidQuery("empty SQL".into()));
        }
        if trimmed.contains(';') {
            return Err(PerceptionError::InvalidQuery(
                "read_sql accepts a single statement".into(),
            ));
        }
        let lowered = trimmed.to_lowercase();
        if !(lowered.starts_with("select") || lowered.starts_with("with")) {
            return Err(PerceptionError::InvalidQuery(
                "read_sql accepts SELECT statements only".into(),
            ));
        }

        let conn = open_read_only(&self.db_path)?;
        conn.pragma_update(None, "query_only", "ON")
            .map_err(map_query_err)?;

        // The hard row cap wraps the statement, so an inner LIMIT still
        // applies and an absent one cannot return unbounded rows.
        let wrapped = format!("SELECT * FROM ({trimmed}) LIMIT {}", MAX_READ_LIMIT + 1);

        // Interrupt timer: fires only if the query is still running at the
        // deadline; the ack channel releases it early on completion.
        let interrupt = conn.get_interrupt_handle();
        let (done_tx, done_rx) = mpsc::channel::<()>();
        let timer = std::thread::Builder::new()
            .name("screenie-read-sql-timer".into())
            .spawn(move || {
                if done_rx.recv_timeout(READ_SQL_TIMEOUT).is_err() {
                    interrupt.interrupt();
                }
            });

        let result = (|| {
            let mut stmt = conn.prepare(&wrapped).map_err(map_query_err)?;
            if !stmt.readonly() {
                return Err(PerceptionError::InvalidQuery(
                    "read_sql accepts read-only statements only".into(),
                ));
            }
            let columns: Vec<String> = stmt
                .column_names()
                .into_iter()
                .map(|name| name.to_string())
                .collect();
            let column_count = columns.len();
            let mut rows_out: Vec<Vec<serde_json::Value>> = Vec::new();
            let mut rows = stmt.query([]).map_err(map_query_err)?;
            while let Some(row) = rows.next().map_err(map_query_err)? {
                let mut out = Vec::with_capacity(column_count);
                for index in 0..column_count {
                    out.push(sql_value_to_json(
                        row.get_ref(index).map_err(map_query_err)?,
                    ));
                }
                rows_out.push(out);
                if rows_out.len() > MAX_READ_LIMIT as usize {
                    break;
                }
            }
            let truncated = rows_out.len() > MAX_READ_LIMIT as usize;
            rows_out.truncate(MAX_READ_LIMIT as usize);
            Ok(SqlRows {
                columns,
                rows: rows_out,
                truncated,
            })
        })();

        let _ = done_tx.send(());
        if let Ok(handle) = timer {
            let _ = handle.join();
        }
        result
    }
}

fn open_read_only(db_path: &Path) -> Result<Connection, PerceptionError> {
    Connection::open_with_flags(
        db_path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|err| PerceptionError::Index(format!("open read connection: {err}")))
}

fn map_query_err(err: rusqlite::Error) -> PerceptionError {
    match &err {
        rusqlite::Error::SqliteFailure(code, _)
            if code.code == rusqlite::ErrorCode::OperationInterrupted =>
        {
            PerceptionError::InvalidQuery("query exceeded the 250ms budget".into())
        }
        _ => PerceptionError::Index(err.to_string()),
    }
}

fn sql_value_to_json(value: rusqlite::types::ValueRef<'_>) -> serde_json::Value {
    use rusqlite::types::ValueRef;
    match value {
        ValueRef::Null => serde_json::Value::Null,
        ValueRef::Integer(i) => serde_json::Value::from(i),
        ValueRef::Real(f) => serde_json::Value::from(f),
        ValueRef::Text(text) => serde_json::Value::from(String::from_utf8_lossy(text).to_string()),
        ValueRef::Blob(blob) => serde_json::Value::from(format!("<{} bytes>", blob.len())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perception::index::store::tests::{temp_base, test_meta, test_row};
    use crate::perception::index::store::Store;

    fn seeded() -> (PathBuf, Store, Reader) {
        let base = temp_base();
        let store = Store::open(&base).unwrap();
        let meta = test_meta(
            "ax:querysnap",
            11,
            crate::perception::index::store::now_ms(),
        );
        let mut save = test_row("ax:B1", "Save document");
        save.focused = true;
        let mut cancel = test_row("ax:B2", "Cancel");
        cancel.frame = RectPt::new(500.0, 20.0, 30.0, 40.0);
        let mut field = test_row("ax:T1", "Address and Search");
        field.role = "AXTextField".into();
        field.class_prefix = 'T';
        field.value = Some("github.com".into());
        let mut decoration = test_row("ax:X1", "");
        decoration.title = None;
        decoration.role = "AXStaticText".into();
        decoration.class_prefix = 'X';
        decoration.actionable = false;
        decoration.actions = Vec::new();
        store
            .ingest(meta, vec![save, cancel, field, decoration])
            .unwrap();
        let reader = Reader::open(store.db_path()).unwrap();
        (base, store, reader)
    }

    #[test]
    fn structured_filters_compose() {
        let (base, _store, reader) = seeded();
        let all = reader
            .query("ax:querysnap", &ElementQuery::default())
            .unwrap();
        assert_eq!(all.total, 4);
        assert_eq!(all.rows.len(), 4);

        // Class letter filter.
        let buttons = reader
            .query(
                "ax:querysnap",
                &ElementQuery {
                    roles: Some(vec!["B".into()]),
                    ..ElementQuery::default()
                },
            )
            .unwrap();
        assert_eq!(buttons.total, 2);

        // Raw role + class letter mix.
        let mixed = reader
            .query(
                "ax:querysnap",
                &ElementQuery {
                    roles: Some(vec!["B".into(), "AXTextField".into()]),
                    ..ElementQuery::default()
                },
            )
            .unwrap();
        assert_eq!(mixed.total, 3);

        // FTS text over title/value.
        let text = reader
            .query(
                "ax:querysnap",
                &ElementQuery {
                    text: Some("github".into()),
                    ..ElementQuery::default()
                },
            )
            .unwrap();
        assert_eq!(text.total, 1);
        assert_eq!(text.rows[0].eid, "ax:T1");

        // Region intersect: only Cancel sits past x=400.
        let region = reader
            .query(
                "ax:querysnap",
                &ElementQuery {
                    region: Some(RectPt::new(400.0, 0.0, 400.0, 400.0)),
                    ..ElementQuery::default()
                },
            )
            .unwrap();
        assert_eq!(region.total, 1);
        assert_eq!(region.rows[0].eid, "ax:B2");

        // Flags.
        let focused = reader
            .query(
                "ax:querysnap",
                &ElementQuery {
                    focused_only: true,
                    ..ElementQuery::default()
                },
            )
            .unwrap();
        assert_eq!(focused.rows[0].eid, "ax:B1");
        let actionable = reader
            .query(
                "ax:querysnap",
                &ElementQuery {
                    actionable_only: true,
                    ..ElementQuery::default()
                },
            )
            .unwrap();
        assert_eq!(actionable.total, 3);

        // Limit clamps and total still reports the full match count.
        let limited = reader
            .query(
                "ax:querysnap",
                &ElementQuery {
                    limit: 2,
                    ..ElementQuery::default()
                },
            )
            .unwrap();
        assert_eq!(limited.rows.len(), 2);
        assert_eq!(limited.total, 4);

        // Round trip fidelity.
        assert_eq!(actionable.rows[0].actions, vec!["AXPress".to_string()]);
        assert_eq!(
            actionable.rows[0].frame,
            RectPt::new(10.0, 20.0, 30.0, 40.0)
        );
        std::fs::remove_dir_all(base).ok();
    }

    #[test]
    fn read_sql_guards() {
        let (base, _store, reader) = seeded();
        // Plain select works and is row-capped by the wrapper.
        let ok = reader
            .read_sql("SELECT eid, title FROM elements ORDER BY eid")
            .unwrap();
        assert_eq!(ok.columns, vec!["eid".to_string(), "title".to_string()]);
        assert_eq!(ok.rows.len(), 4);
        assert!(!ok.truncated);

        // CTEs are SELECTs too.
        assert!(reader
            .read_sql("WITH b AS (SELECT eid FROM elements WHERE class='B') SELECT * FROM b")
            .is_ok());

        // Writes, multi-statements, and non-selects are rejected.
        assert!(matches!(
            reader.read_sql("DELETE FROM elements"),
            Err(PerceptionError::InvalidQuery(_))
        ));
        assert!(matches!(
            reader.read_sql("SELECT 1; SELECT 2"),
            Err(PerceptionError::InvalidQuery(_))
        ));
        assert!(matches!(
            reader.read_sql("PRAGMA user_version = 9"),
            Err(PerceptionError::InvalidQuery(_))
        ));
        assert!(matches!(
            reader.read_sql(""),
            Err(PerceptionError::InvalidQuery(_))
        ));
        std::fs::remove_dir_all(base).ok();
    }
}
