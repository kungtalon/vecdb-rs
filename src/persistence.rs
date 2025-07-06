use crate::merror::FileError;
use crate::scalar::ScalarStorage;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::event;

const WAL_LOG_ID_KEY: &[u8] = b"__wal_log_id__";

#[derive(Serialize, Deserialize, Debug)]
pub enum WALLogOperation {
    Insert,
    Update,
    Delete,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct WALLogRecord<'a> {
    pub log_id: u64,
    pub version: String,
    pub operation: WALLogOperation,
    pub encoded_data: &'a [u8],
}

pub struct Persistence {
    pub counter: AtomicU64,
    pub wal_log_file: File,
}

fn new_persistence(
    wal_log_file_path: &str,
    scalar_db: &impl ScalarStorage,
) -> Result<Persistence, FileError> {
    // try to get the counter from the scalar database
    let init_counter: u64 = scalar_db
        .get(WAL_LOG_ID_KEY)
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

    let p: &Path = Path::new(wal_log_file_path);
    if !p.exists() {
        fs::create_dir_all(p)
            .map_err(|e| FileError(format!("Failed to create WAL log directory: {}", e)))?;
    }

    let wal_log_file =
        File::open(p).map_err(|e| FileError(format!("Failed to init WAL log file: {}", e)))?;

    Ok(Persistence {
        counter: AtomicU64::new(init_counter),
        wal_log_file,
    })
}

impl Persistence {
    pub fn new(
        wal_log_file_path: &str,
        scalar_db: &impl ScalarStorage,
    ) -> Result<Persistence, FileError> {
        new_persistence(wal_log_file_path, scalar_db)
    }

    pub fn get_counter(&self) -> u64 {
        self.counter.load(Ordering::Acquire)
    }

    pub fn increment_counter(&mut self) -> u64 {
        self.counter.fetch_add(1, Ordering::AcqRel)
    }

    pub fn write_wal_log(&mut self, record: &WALLogRecord) -> Result<(), FileError> {
        let json_record = serde_json::to_string(&record)
            .map_err(|e| FileError(format!("Failed to serialize WAL log record: {e}")))?;

        event!(
            tracing::Level::DEBUG,
            "Writing WAL log record: {}",
            json_record
        );

        writeln!(self.wal_log_file, "{}", json_record)
            .map_err(|e| FileError(format!("Failed to write operation: {e}")))?;

        Ok(())
    }
}
