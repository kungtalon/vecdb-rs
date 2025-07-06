use std::collections::HashMap;

use crate::merror::DBError;
use rocksdb::Options;
use serde_json::Value;
use std::path::Path;
use std::sync::Mutex;

type Mdb = rocksdb::DBWithThreadMode<rocksdb::MultiThreaded>;

const KEY_ID_MAX: &str = "__id_max__";

pub trait ScalarStorage: Sync + Send {
    fn put(&self, key: &[u8], values: &[u8]) -> Result<(), DBError>;

    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, DBError>;

    #[allow(unused)]
    fn get_value(&self, index: u64) -> Result<Option<HashMap<String, Value>>, DBError> {
        match self.get(&index.to_be_bytes())? {
            Some(bytes) => {
                let value = serde_json::from_slice::<HashMap<String, Value>>(&bytes)
                    .map_err(|e| DBError::GetError(e.to_string()))?;
                Ok(Some(value))
            }
            None => Ok(None),
        }
    }

    fn multi_get_value(&self, indices: &[u64]) -> Result<Vec<HashMap<String, Value>>, DBError>;

    // Generates a list of unique IDs starting from the last ID used
    fn gen_incr_ids(&self, num: usize) -> Result<Vec<u64>, DBError>;

    fn to_iter(&self) -> rocksdb::DBIteratorWithThreadMode<Mdb>;
}

pub fn new_scalar_storage<P: AsRef<Path>>(path: P) -> Result<impl ScalarStorage, DBError> {
    let db = MultiThreadRocksDB::new(&path)?;
    Ok(db)
}

struct MultiThreadRocksDB {
    id_mutex: Mutex<()>,
    db: Mdb,
}

impl MultiThreadRocksDB {
    fn new<P: AsRef<Path>>(path: P) -> Result<Self, DBError> {
        let mut options = Options::default();
        options.create_if_missing(true);
        let db = Mdb::open(&options, path).map_err(|e| DBError::CreateError(e.to_string()))?;
        Ok(MultiThreadRocksDB {
            id_mutex: Mutex::new(()),
            db,
        })
    }
}

impl ScalarStorage for MultiThreadRocksDB {
    fn put(&self, key: &[u8], values: &[u8]) -> Result<(), DBError> {
        self.db
            .put(key, values)
            .map_err(|e| DBError::PutError(e.to_string()))?;
        Ok(())
    }

    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>, DBError> {
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

    fn gen_incr_ids(&self, num: usize) -> Result<Vec<u64>, DBError> {
        let _guard = self
            .id_mutex
            .lock()
            .map_err(|e| DBError::GetError(format!("failed to acquire lock: {e:?}",)))?;

        gen_incr_ids(&self.db, num)
    }

    fn to_iter(&self) -> rocksdb::DBIteratorWithThreadMode<Mdb> {
        self.db.iterator(rocksdb::IteratorMode::Start)
    }
}

fn gen_incr_ids<T: rocksdb::ThreadMode>(
    db: &rocksdb::DBWithThreadMode<T>,
    num: usize,
) -> Result<Vec<u64>, DBError> {
    let max_id_as_bytes = db
        .get(KEY_ID_MAX.as_bytes())
        .map_err(|e| DBError::GetError(e.to_string()))?;

    let max_id: u64 =
        match max_id_as_bytes {
            Some(bytes) => u64::from_be_bytes(bytes.try_into().map_err(|e| {
                DBError::GetError(format!("failed to convert incr ID as u64: {e:?}"))
            })?),
            None => 0,
        };

    let new_max_id = max_id + num as u64;

    let ids: Vec<u64> = (max_id + 1..new_max_id + 1).collect::<Vec<u64>>();

    db.put(KEY_ID_MAX.as_bytes(), new_max_id.to_be_bytes())
        .map_err(|e| DBError::PutError(format!("failed to insert new generated max id: {e:?}")))?;

    Ok(ids)
}

#[cfg(debug_assertions)]
pub fn debug_print_scalar_db(ss: &dyn ScalarStorage) -> Result<(), DBError> {
    let iter = ss.to_iter();

    let special: [u8; 10] = [95, 95, 105, 100, 95, 109, 97, 120, 95, 95];

    for data in iter {
        let data_pair: (Box<[u8]>, Box<[u8]>) = data.unwrap();

        if data_pair.0.len() == special.len() {
            println!(
                "max id: {:?}",
                u64::from_be_bytes(data_pair.1.to_vec().try_into().unwrap())
            );

            continue;
        }

        let key_temp: Result<[u8; 8], _> = data_pair.0.to_vec().try_into();
        let mut key: u64 = 0;

        match key_temp {
            Ok(k) => key = u64::from_be_bytes(k),
            Err(e) => {
                println!("failed to convert key to u64: {e:?}")
            }
        }

        let value = String::from_utf8(data_pair.1.to_vec()).map_err(|e| {
            DBError::GetError("failed to get value from scalar db".to_string() + &e.to_string())
        })?;
        println!("Key: {key:?}, Value: {value:?}");
    }

    Ok(())
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

    fn test_db_multi_get_value(db: &mut impl ScalarStorage) {
        let key1 = 1u64;
        let msg1 = "Hello, world";
        let key2 = 2u64;
        let msg2 = "Goodbye, world";
        db.put(
            &key1.to_be_bytes(),
            serde_json::to_vec(&HashMap::from([("msg", msg1)]))
                .unwrap()
                .as_ref(),
        )
        .unwrap();
        db.put(
            &key2.to_be_bytes(),
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

    fn test_db_get_value(db: &mut impl ScalarStorage) {
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
        db.put(&key.to_be_bytes(), &value_bytes).unwrap();

        let retrieved_value = db.get_value(key).unwrap().expect("failed to get value");
        assert_eq!(retrieved_value, value);
    }

    #[test]
    fn test_single_thread_rocksdb() {
        let path = setup(format!("single_thread_{}", Uuid::new_v4()).as_str());

        let mut db = new_scalar_storage(&path).unwrap();

        test_db_multi_get_value(&mut db);
        test_db_get_value(&mut db);

        fs::remove_dir_all(&path).unwrap();
    }

    #[test]
    fn test_multi_thread_rocksdb() {
        let path = setup(format!("multi_thread_{}", Uuid::new_v4()).as_str());

        let mut db = new_scalar_storage(&path).unwrap();

        test_db_multi_get_value(&mut db);
        test_db_get_value(&mut db);

        fs::remove_dir_all(&path).unwrap();
    }
}
