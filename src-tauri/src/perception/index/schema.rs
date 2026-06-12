//! DDL and migrations for the perception index.
//!
//! `user_version` gates migrations; v1 is the initial schema. The `elements`
//! schema is the build contract's read_sql surface — additive changes only,
//! and the `class` column is the one addition over the original spec
//! (denormalized eid prefix letter, so class filters don't string-slice).

use rusqlite::Connection;

const SCHEMA_VERSION: i32 = 1;

const DDL_V1: &str = "
CREATE TABLE IF NOT EXISTS snapshots(
  id TEXT PRIMARY KEY,
  ns TEXT NOT NULL DEFAULT 'ax',
  created_at INTEGER NOT NULL,
  app_bundle TEXT, app_pid INTEGER,
  window_id INTEGER, window_title TEXT,
  display_id INTEGER, scale REAL NOT NULL,
  cap_x REAL, cap_y REAL, cap_w REAL, cap_h REAL,
  raw_png TEXT, marked_png TEXT,
  element_count INTEGER, took_ms INTEGER,
  partial INTEGER DEFAULT 0,
  dirty INTEGER DEFAULT 0,
  -- Diff region (points) carried from the change signal so v2 can re-walk
  -- partially without an interface change.
  diff_x REAL, diff_y REAL, diff_w REAL, diff_h REAL
);

CREATE TABLE IF NOT EXISTS elements(
  snapshot_id TEXT NOT NULL REFERENCES snapshots(id) ON DELETE CASCADE,
  eid TEXT NOT NULL, fp INTEGER NOT NULL,
  class TEXT NOT NULL DEFAULT 'X',
  role TEXT, subrole TEXT, title TEXT, descr TEXT, value TEXT,
  actionable INTEGER, enabled INTEGER, focused INTEGER,
  x REAL, y REAL, w REAL, h REAL,
  depth INTEGER, parent_eid TEXT,
  actions TEXT, is_web_boundary INTEGER DEFAULT 0,
  PRIMARY KEY (snapshot_id, eid)
);

CREATE INDEX IF NOT EXISTS idx_el_role ON elements(snapshot_id, role);
CREATE INDEX IF NOT EXISTS idx_el_act  ON elements(snapshot_id, actionable);
CREATE INDEX IF NOT EXISTS idx_el_fp   ON elements(fp);
CREATE INDEX IF NOT EXISTS idx_snap_window ON snapshots(app_bundle, window_id, created_at);

CREATE VIRTUAL TABLE IF NOT EXISTS elements_fts USING fts5(
  snapshot_id UNINDEXED, eid UNINDEXED, title, descr, value
);
";

/// Connection-level pragmas every handle (reader and writer) needs.
pub fn apply_connection_pragmas(conn: &Connection) -> rusqlite::Result<()> {
    conn.pragma_update(None, "foreign_keys", "ON")?;
    Ok(())
}

/// One-time setup on the writer connection: WAL + schema migration.
pub fn migrate(conn: &Connection) -> rusqlite::Result<()> {
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.pragma_update(None, "synchronous", "NORMAL")?;
    apply_connection_pragmas(conn)?;
    let version: i32 = conn.pragma_query_value(None, "user_version", |row| row.get(0))?;
    if version < 1 {
        conn.execute_batch(DDL_V1)?;
    }
    if version < SCHEMA_VERSION {
        conn.pragma_update(None, "user_version", SCHEMA_VERSION)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migration_creates_schema_with_fts5() {
        let conn = Connection::open_in_memory().unwrap();
        migrate(&conn).unwrap();
        // FTS5 virtual table must exist — bundled SQLite ships FTS5.
        let fts_count: i64 = conn
            .query_row(
                "SELECT count(*) FROM sqlite_master WHERE name='elements_fts'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(fts_count, 1);
        // Idempotent.
        migrate(&conn).unwrap();
        let version: i32 = conn
            .pragma_query_value(None, "user_version", |row| row.get(0))
            .unwrap();
        assert_eq!(version, SCHEMA_VERSION);
    }
}
