use crate::filter::IntFilterIndex;
use crate::merror::{DataError, FileError};
use crate::scalar::{ScalarStorage, NAMESPACE_WALS};
use crate::vecdb::{VdbUpsertArgs, VectorDatabase};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Lines, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::event;

const WAL_ID_KEY: &str = "__wal_id__";

#[derive(Serialize, Deserialize, Debug)]
pub enum WALOperation {
    Upsert,
    Delete,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct WALRecord {
    pub log_id: u64,
    pub version: String,
    pub operation: WALOperation,
    pub data: Vec<u8>,
}

impl TryFrom<&str> for WALRecord {
    type Error = DataError;

    fn try_from(line: &str) -> Result<Self, Self::Error> {
        let result = serde_json::from_str(line)
            .map_err(|e| DataError(format!("Failed to deserialize WAL log record: {e}")))?;
        Ok(result)
    }
}

pub struct Persistence {
    counter: AtomicU64,
    file_path: String,
    wal_writer: File,

    pub version: String,
}

pub struct WALRecordIter {
    lines: Lines<BufReader<File>>,
}

impl WALRecordIter {
    pub fn new(file_path: &str) -> Result<Self, FileError> {
        let file = File::open(file_path)
            .map_err(|e| FileError(format!("Failed to open WAL log file: {e}")))?;
        let reader = BufReader::new(file);
        Ok(WALRecordIter {
            lines: reader.lines(),
        })
    }
}

impl Iterator for WALRecordIter {
    type Item = Result<WALRecord, DataError>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(line) = self.lines.next() {
            let result = match line {
                Ok(l) => WALRecord::try_from(l.as_ref()),
                Err(e) => Err(DataError(format!(
                    "Failed to read line from WAL log file: {e}"
                ))),
            };

            Some(result)
        } else {
            None
        }
    }
}

fn new_persistence(
    wal_file_path: &str,
    version: &str,
    scalar_db: &impl ScalarStorage,
) -> Result<Persistence, FileError> {
    // try to get the counter from the scalar database
    let last_log_id: u64 = scalar_db
        .get(format!("{NAMESPACE_WALS}{WAL_ID_KEY}").as_bytes())
        .map_err(|e| FileError(format!("Failed to get WAL counter: {e}")))?
        .map(|bytes: Vec<u8>| {
            let bytes_u64: [u8; 8] = bytes.clone().try_into().map_err(|_| {
                FileError(format!(
                    "Failed to convert bytes with len {} to u64",
                    bytes.len()
                ))
            })?;
            Ok(u64::from_le_bytes(bytes_u64))
        })
        .transpose()?
        .unwrap_or(0);

    let p: &Path = Path::new(wal_file_path);
    if !p.exists() {
        fs::create_dir_all(p)
            .map_err(|e| FileError(format!("Failed to create WAL log directory: {}", e)))?;
    }

    let wal_file_writer =
        File::open(p).map_err(|e| FileError(format!("Failed to init WAL log file: {}", e)))?;

    Ok(Persistence {
        counter: AtomicU64::new(last_log_id),
        version: version.to_string(),
        wal_writer: wal_file_writer,
        file_path: wal_file_path.to_string(),
    })
}

impl Persistence {
    pub fn new(
        file_path: &str,
        version: &str,
        scalar_db: &impl ScalarStorage,
    ) -> Result<Self, FileError> {
        new_persistence(file_path, version, scalar_db)
    }

    pub fn get_log_id(&self) -> u64 {
        self.counter.load(Ordering::Acquire)
    }

    pub fn increment_log_id(&mut self) -> u64 {
        self.counter.fetch_add(1, Ordering::AcqRel)
    }

    pub fn set_log_id(&mut self, log_id: u64) {
        self.counter.store(log_id, Ordering::Release);
    }

    pub async fn write_wal(
        &mut self,
        operation_type: WALOperation,
        data: Vec<u8>,
    ) -> Result<(), FileError> {
        let record = WALRecord {
            log_id: self.increment_log_id(),
            version: self.version.clone(),
            operation: operation_type,
            data,
        };

        let json_record = serde_json::to_string(&record)
            .map_err(|e| FileError(format!("Failed to serialize WAL log record: {e}")))?;

        event!(
            tracing::Level::DEBUG,
            "Writing WAL log record: {}",
            json_record
        );

        writeln!(self.wal_writer, "{}", json_record)
            .map_err(|e| FileError(format!("Failed to write operation: {e}")))?;

        Ok(())
    }

    pub async fn get_wal_iterator(&self) -> Result<WALRecordIter, FileError> {
        WALRecordIter::new(&self.file_path)
    }
}

pub async fn apply_wal_record(
    record: WALRecord,
    vec_db: &mut VectorDatabase,
) -> Result<(), DataError> {
    match record.operation {
        WALOperation::Upsert => {
            let insert_args: VdbUpsertArgs = serde_json::from_slice(&record.data)
                .map_err(|e| DataError(format!("Failed to deserialize Upsert data: {e}")))?;

            vec_db
                .upsert(insert_args)
                .await
                .map_err(|e| DataError(format!("Failed to apply Upsert operation: {e}")))?;
        }
        WALOperation::Delete => {
            panic!(
                "Delete operation is not supported in this context. Please implement it if needed."
            );
        }
    }

    Ok(())
}

/// A guard that provides rollback semantics for operations that may need to be reverted if not committed.
///
/// `RollbackGuard` takes a closure that will be executed if the guard is dropped without being committed.
/// This is useful for ensuring cleanup or rollback logic is performed in case of early returns, panics, or errors.
///
/// # Example
/// ```rust
/// let guard = RollbackGuard::new(|| {
///     // rollback logic here
/// });
/// // ... perform operations ...
/// guard.commit(); // Prevents rollback on drop
/// ```
///
/// # Type Parameters
/// - `F`: A closure type that implements `FnOnce()`.
///
/// # Fields
/// - `rollback`: Optionally stores the rollback closure.
/// - `committed`: Indicates whether the guard has been committed.
///
/// # Methods
/// - `new(rollback: F) -> Self`: Creates a new guard with the given rollback closure.
/// - `commit(self)`: Marks the guard as committed, preventing rollback on drop.
///
/// # Drop
/// If the guard is dropped without being committed, the rollback closure is executed.
pub struct RollbackGuard<F: FnOnce()> {
    rollback: Option<F>,
    committed: bool,
}

impl<F: FnOnce()> RollbackGuard<F> {
    pub fn new(rollback: F) -> Self {
        RollbackGuard {
            rollback: Some(rollback),
            committed: false,
        }
    }

    pub fn commit(mut self) {
        self.committed = true;
        self.rollback = None;
    }
}

impl<F: FnOnce()> Drop for RollbackGuard<F> {
    fn drop(&mut self) {
        if !self.committed {
            if let Some(rollback) = self.rollback.take() {
                rollback();
            }
        }
    }
}
