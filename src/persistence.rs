use crate::merror::{DataError, FileError};
use crate::scalar::ScalarStorage;
use flate2::{write::GzEncoder, Compression};
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
    pub data: &'a [u8],
}

pub struct Persistence {
    pub counter: AtomicU64,
    pub wal_log_file: File,

    encoder: Option<Encoder>,
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
        encoder: None,
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

    pub fn with_encoder(self, encoder: Encoder) {
        self.encoder = Some(encoder.clone());
    }

    pub async fn write_wal_log(&mut self, record: WALLogRecord) -> Result<(), FileError> {
        if self.encoder.is_some() {
            if let Some(encoder) = &self.encoder {
                // Encode the data using the encoder
                record.data = &encoder
                    .encode(record.data)
                    .map_err(|e| FileError(format!("Failed to encode data for WAL log: {e}")))?;
            }
        }

        let json_record = serde_json::to_string(&record)
            .map_err(|e| DataError(format!("Failed to serialize WAL log record: {e}")))?;

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

pub enum Encoding {
    Gzip,
    // Add other encodings if needed
}

// Encoder is responsible for compressing and encoding the user data for WAL log
pub struct Encoder {
    pub encoding: Encoding,
}

impl Encoder {
    pub fn new(encoding: Encoding) -> Encoder {
        Encoder { encoding }
    }

    pub fn encode(&self, data: &[u8]) -> Result<Vec<u8>, DataError> {
        match self.encoding {
            Encoding::Gzip => Self::gzip_encode(data),
            // Add other encodings if needed
        }
    }

    fn gzip_encode(data: &[u8]) -> Result<Vec<u8>, DataError> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder
            .write_all(data)
            .map_err(|e| DataError(format!("Failed to write to encoder: {e}")))?;
        encoder
            .finish()
            .map_err(|e| DataError(format!("Failed to finish encoding: {e}")))
    }
}
