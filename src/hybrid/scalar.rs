use std::collections::HashMap;
use std::string;

use crate::merror::DBError;
use rocksdb::{Options, DB};
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::path::Path;
use uuid::Uuid;

type MDB = rocksdb::DBWithThreadMode<rocksdb::MultiThreaded>;

pub trait ScalarStorage {
    fn put(&mut self, index: u64, values: &[u8]) -> Result<(), DBError>;

    fn get(&self, index: u64) -> Result<Option<Vec<u8>>, DBError>;

    fn get_str(&self, index: u64) -> Result<Option<String>, DBError> {
        match self.get(index)? {
            Some(bytes) => {
                let value =
                    String::from_utf8(bytes).map_err(|e| DBError::GetError(e.to_string()))?;
                Ok(Some(value))
            }
            None => Ok(None),
        }
    }

    fn get_value(&self, index: u64) -> Result<Option<HashMap<String, Value>>, DBError> {
        match self.get(index)? {
            Some(bytes) => {
                let value = serde_json::from_slice::<HashMap<String, Value>>(&bytes)
                    .map_err(|e| DBError::GetError(e.to_string()))?;
                Ok(Some(value))
            }
            None => Ok(None),
        }
    }
}

pub struct SingleThreadRocksDB {
    db: DB,
}

impl SingleThreadRocksDB {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, DBError> {
        let mut options = Options::default();
        options.create_if_missing(true);
        let db = DB::open(&options, path).map_err(|e| DBError::CreateError(e.to_string()))?;
        Ok(SingleThreadRocksDB { db })
    }
}

impl ScalarStorage for SingleThreadRocksDB {
    fn put(&mut self, index: u64, values: &[u8]) -> Result<(), DBError> {
        let key = index.to_be_bytes();
        self.db
            .put(key, values)
            .map_err(|e| DBError::PutError(e.to_string()))?;
        Ok(())
    }

    fn get(&self, index: u64) -> Result<Option<Vec<u8>>, DBError> {
        let key = index.to_be_bytes();
        match self
            .db
            .get(key)
            .map_err(|e| DBError::GetError(e.to_string()))?
        {
            Some(value) => Ok(Some(value.to_vec())),
            None => Ok(None),
        }
    }
}

pub struct MultiThreadRocksDB {
    db: MDB,
}

impl MultiThreadRocksDB {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, DBError> {
        let mut options = Options::default();
        options.create_if_missing(true);
        let db = MDB::open(&options, path).map_err(|e| DBError::CreateError(e.to_string()))?;
        Ok(MultiThreadRocksDB { db })
    }
}

impl ScalarStorage for MultiThreadRocksDB {
    fn put(&mut self, index: u64, values: &[u8]) -> Result<(), DBError> {
        let key = index.to_be_bytes();
        self.db
            .put(key, values)
            .map_err(|e| DBError::PutError(e.to_string()))?;
        Ok(())
    }

    fn get(&self, index: u64) -> Result<Option<Vec<u8>>, DBError> {
        let key = index.to_be_bytes();
        match self
            .db
            .get(key)
            .map_err(|e| DBError::GetError(e.to_string()))?
        {
            Some(value) => Ok(Some(value.to_vec())),
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::{Path, PathBuf};

    fn setup(suffix: &str) -> PathBuf {
        let db_path = Path::new("/tmp/test_db").join(suffix);

        if fs::exists(&db_path).unwrap() {
            fs::remove_dir_all(&db_path).expect("failed to remove test directory");
        }

        fs::create_dir_all(&db_path).expect("failed to create test directory");

        return db_path;
    }

    fn test_db_get_str<D: ScalarStorage>(db: &mut D) {
        let key = 1u64;
        let value = b"Hello, world!";
        db.put(key, value).unwrap();

        let retrieved_value = db.get_str(key).unwrap().expect("failed to get value");
        assert_eq!(retrieved_value, String::from_utf8_lossy(value));
    }

    fn test_db_get_value<D: ScalarStorage>(db: &mut D) {
        let key = 2u64;
        let value = HashMap::from([
            ("name".to_string(), Value::String("Alice".to_string())),
            ("age".to_string(), Value::Number(30.into())),
            ("is_student".to_string(), Value::Bool(false)),
            (
                "grades".to_string(),
                Value::Array(vec![Value::Number(90.into()), Value::Number(85.into())]),
            ),
        ]);
        let value_bytes = serde_json::to_vec(&value).unwrap();
        db.put(key, &value_bytes).unwrap();

        let retrieved_value = db.get_value(key).unwrap().expect("failed to get value");
        assert_eq!(retrieved_value, value);
    }

    #[test]
    fn test_single_thread_rocksdb() {
        let path = setup(format!("single_thread_{}", Uuid::new_v4().to_string()).as_str());

        let mut db: SingleThreadRocksDB = SingleThreadRocksDB::new(&path).unwrap();

        test_db_get_str(&mut db);
        test_db_get_value(&mut db);

        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn test_multi_thread_rocksdb() {
        let path = setup(format!("multi_thread_{}", Uuid::new_v4().to_string()).as_str());

        let mut db = MultiThreadRocksDB::new(&path).unwrap();

        test_db_get_str(&mut db);
        test_db_get_value(&mut db);

        fs::remove_dir_all(path).unwrap();
    }
}
