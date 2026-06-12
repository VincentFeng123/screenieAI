//! The SQLite-backed perception index: one writer thread (store), read-only
//! query layer (query), DDL/migrations (schema). Lives under
//! `app_data_dir()/perception/`; nothing leaves the machine (C6).

pub mod query;
pub mod schema;
pub mod store;

pub use query::{ElementQuery, QueryRows, Reader, SqlRows, DEFAULT_READ_LIMIT, MAX_READ_LIMIT};
pub use store::{SnapshotMeta, Store};

use crate::perception::PerceptionError;
use std::path::Path;
use std::sync::OnceLock;

/// The process-wide index: store + reader over one database.
pub struct PerceptionIndex {
    pub store: Store,
    pub reader: Reader,
}

static INDEX: OnceLock<PerceptionIndex> = OnceLock::new();

/// Initialize the global index under `base_dir` (idempotent; first caller
/// wins). The command layer calls this with `app_data_dir()`.
pub fn init(base_dir: &Path) -> Result<&'static PerceptionIndex, PerceptionError> {
    if let Some(existing) = INDEX.get() {
        return Ok(existing);
    }
    let store = Store::open(base_dir)?;
    let reader = Reader::open(store.db_path())?;
    let _ = INDEX.set(PerceptionIndex { store, reader });
    INDEX
        .get()
        .ok_or_else(|| PerceptionError::Index("index initialization raced".into()))
}

/// The global index, if a command has initialized it.
pub fn get() -> Option<&'static PerceptionIndex> {
    INDEX.get()
}
