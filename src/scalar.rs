use std::collections::HashMap;

use crate::merror::DBError;
use rocksdb::{Options, DB};
use serde_json::Value;
use std::path::Path;

type Mdb = rocksdb::DBWithThreadMode<rocksdb::MultiThreaded>;

pub trait ScalarStorage {
    fn put(&mut self, index: u64, values: &[u8]) -> Result<(), DBError>;

    fn get(&self, index: u64) -> Result<Option<Vec<u8>>, DBError>;

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

    fn multi_get_value(&self, indices: &[u64]) -> Result<Vec<HashMap<String, Value>>, DBError>;
}

pub fn new_scalar_storage<P: AsRef<Path>>(
    path: P,
    concurrent: bool,
) -> Result<Box<dyn ScalarStorage>, DBError> {
    if concurrent {
        let db = MultiThreadRocksDB::new(&path)?;
        return Ok(Box::new(db));
    }

    let db = SingleThreadRocksDB::new(&path)?;

    Ok(Box::new(db))
}

struct SingleThreadRocksDB {
    db: DB,
}

impl SingleThreadRocksDB {
    fn new<P: AsRef<Path>>(path: P) -> Result<Self, DBError> {
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

    fn multi_get_value(&self, indices: &[u64]) -> Result<Vec<HashMap<String, Value>>, DBError> {
        let mut result: Vec<HashMap<String, Value>> = Vec::new();

        let keys_byte = indices.iter().map(|i| i.to_be_bytes());

        let scalar_response = self.db.multi_get(keys_byte);

        for scalar in scalar_response {
            match scalar {
                Ok(Some(bytes)) => {
                    let value = serde_json::from_slice::<HashMap<String, Value>>(&bytes)
                        .map_err(|e| DBError::GetError(e.to_string()))?;

                    result.push(value);
                }
                Ok(None) => result.push(HashMap::new()),
                Err(e) => return Err(DBError::GetError(e.to_string())),
            }
        }

        Ok(result)
    }
}

struct MultiThreadRocksDB {
    db: Mdb,
}

impl MultiThreadRocksDB {
    fn new<P: AsRef<Path>>(path: P) -> Result<Self, DBError> {
        let mut options = Options::default();
        options.create_if_missing(true);
        let db = Mdb::open(&options, path).map_err(|e| DBError::CreateError(e.to_string()))?;
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

    fn multi_get_value(&self, indices: &[u64]) -> Result<Vec<HashMap<String, Value>>, DBError> {
        let mut result: Vec<HashMap<String, Value>> = Vec::new();

        let keys_byte = indices.iter().map(|i| i.to_be_bytes());

        let scalar_response = self.db.multi_get(keys_byte);

        for scalar in scalar_response {
            match scalar {
                Ok(Some(bytes)) => {
                    let value = serde_json::from_slice::<HashMap<String, Value>>(&bytes)
                        .map_err(|e| DBError::GetError(e.to_string()))?;

                    result.push(value);
                }
                Ok(None) => result.push(HashMap::new()),
                Err(e) => return Err(DBError::GetError(e.to_string())),
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::{Path, PathBuf};
    use uuid::Uuid;

    fn setup(suffix: &str) -> PathBuf {
        let db_path = Path::new("/tmp/test_db").join(suffix);

        if fs::exists(&db_path).unwrap() {
            fs::remove_dir_all(&db_path).expect("failed to remove test directory");
        }

        fs::create_dir_all(&db_path).expect("failed to create test directory");

        db_path
    }

    fn test_db_multi_get_value(db: &mut Box<dyn ScalarStorage>) {
        let key1 = 1u64;
        let msg1 = "Hello, world";
        let key2 = 2u64;
        let msg2 = "Goodbye, world";
        db.put(
            key1,
            serde_json::to_vec(&HashMap::from([("msg", msg1)]))
                .unwrap()
                .as_ref(),
        )
        .unwrap();
        db.put(
            key2,
            serde_json::to_vec(&HashMap::from([("msg", msg2)]))
                .unwrap()
                .as_ref(),
        )
        .unwrap();

        let retrieved_value = db
            .multi_get_value(&[key1, key2])
            .expect("failed to get value");
        assert_eq!(retrieved_value.len(), 2);
        assert_eq!(retrieved_value[0].get("msg").unwrap(), msg1);
        assert_eq!(retrieved_value[1].get("msg").unwrap(), msg2);
    }

    fn test_db_get_value(db: &mut Box<dyn ScalarStorage>) {
        let key = 3u64;
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
        let path = setup(format!("single_thread_{}", Uuid::new_v4()).as_str());

        let mut db = new_scalar_storage(&path, false).unwrap();

        test_db_multi_get_value(&mut db);
        test_db_get_value(&mut db);

        fs::remove_dir_all(path).unwrap();
    }

    #[test]
    fn test_multi_thread_rocksdb() {
        let path = setup(format!("multi_thread_{}", Uuid::new_v4()).as_str());

        let mut db = new_scalar_storage(&path, true).unwrap();

        test_db_multi_get_value(&mut db);
        test_db_get_value(&mut db);

        fs::remove_dir_all(path).unwrap();
    }
}
