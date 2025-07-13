use crate::merror::{DataError, FileError};
use crate::scalar::{ScalarStorage, NAMESPACE_WALLOGS};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Lines, Write};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::event;

const WAL_LOG_ID_KEY: &str = "__wal_log_id__";

#[derive(Serialize, Deserialize, Debug)]
pub enum WALLogOperation {
    Insert,
    Update,
    Delete,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct WALLogRecord {
    pub log_id: u64,
    pub version: String,
    pub operation: WALLogOperation,
    pub data: Vec<u8>,
}

impl TryFrom<&str> for WALLogRecord {
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
    wal_log_writer: File,

    pub version: String,
}

pub struct WALLogRecordIter {
    lines: Lines<BufReader<File>>,
}

impl WALLogRecordIter {
    pub fn new(file_path: &str) -> Result<Self, FileError> {
        let file = File::open(file_path)
            .map_err(|e| FileError(format!("Failed to open WAL log file: {e}")))?;
        let reader = BufReader::new(file);
        Ok(WALLogRecordIter {
            lines: reader.lines(),
        })
    }
}

impl Iterator for WALLogRecordIter {
    type Item = Result<WALLogRecord, DataError>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(line) = self.lines.next() {
            let result = match line {
                Ok(l) => WALLogRecord::try_from(l.as_ref()),
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
    wal_log_file_path: &str,
    version: &str,
    scalar_db: &impl ScalarStorage,
) -> Result<Persistence, FileError> {
    // try to get the counter from the scalar database
    let last_log_id: u64 = scalar_db
        .get(format!("{NAMESPACE_WALLOGS}{WAL_LOG_ID_KEY}").as_bytes())
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

    let wal_log_file_writer =
        File::open(p).map_err(|e| FileError(format!("Failed to init WAL log file: {}", e)))?;

    Ok(Persistence {
        counter: AtomicU64::new(last_log_id),
        version: version.to_string(),
        wal_log_writer: wal_log_file_writer,
        file_path: wal_log_file_path.to_string(),
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

    pub async fn write_wal_log(
        &mut self,
        operation_type: WALLogOperation,
        data: Vec<u8>,
    ) -> Result<(), FileError> {
        let record = WALLogRecord {
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

        writeln!(self.wal_log_writer, "{}", json_record)
            .map_err(|e| FileError(format!("Failed to write operation: {e}")))?;

        Ok(())
    }

    pub async fn get_wal_log_iterator(&self) -> Result<WALLogRecordIter, FileError> {
        WALLogRecordIter::new(&self.file_path)
    }
}
