use ndarray::array;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use crate::index::{
    FlatIndex, HnswIndex, HnswIndexOption, HnswSearchOption, Index, IndexType, InsertParams,
    MetricType, SearchQuery,
};
use crate::merror::DBError;
use crate::scalar::{new_scalar_storage, ScalarStorage};

pub struct VectorDatabase {
    db_path: PathBuf,
    scalar_storage: Box<dyn ScalarStorage>,
    vector_index: Box<dyn Index>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct IndexParams {
    dim: u32,
    metric_type: MetricType,
    index_type: IndexType,
    hnsw_params: Option<HnswIndexOption>,
}

impl VectorDatabase {
    pub fn new<D: AsRef<Path>>(
        db_path: D,
        index_params: IndexParams,
        concurrent: bool,
    ) -> Result<Self, DBError> {
        let db_path = PathBuf::new().join(db_path);
        let scalar_storage = new_scalar_storage(&db_path, concurrent)?;
        let vector_index: Box<dyn Index> = match index_params.index_type {
            IndexType::Flat => {
                // Create a flat index
                Box::new(
                    FlatIndex::new(index_params.dim, index_params.metric_type).map_err(|e| {
                        DBError::CreateError(format!("unable to create vector index: {}", e))
                    })?,
                )
            }
            IndexType::Hnsw => {
                // Create an HNSW index
                Box::new(
                    HnswIndex::new(
                        index_params.dim,
                        index_params.metric_type,
                        index_params.hnsw_params,
                    )
                    .map_err(|e| {
                        DBError::CreateError(format!("unable to create vector index: {}", e))
                    })?,
                )
            }
        };

        Ok(Self {
            db_path,
            scalar_storage,
            vector_index,
        })
    }

    pub fn upsert(
        &mut self,
        id: u64,
        insert_data: &InsertParams,
        doc: Option<HashMap<String, Value>>,
    ) -> Result<(), DBError> {
        self.vector_index
            .insert(insert_data)
            .map_err(|e| DBError::PutError(format!("unable to upsert vector data: {}", e)))?;

        let mut doc_map: HashMap<String, Value>;

        match doc {
            Some(m) => {
                doc_map = m;
            }
            None => {
                doc_map = HashMap::new();
            }
        }

        doc_map.insert("id".to_string(), Value::Number(id.into()));

        let doc_bytes = serde_json::to_vec(&doc_map)
            .map_err(|e| DBError::PutError(format!("unable to serialize doc data: {}", e)))?;

        self.scalar_storage
            .put(id, &doc_bytes)
            .map_err(|e| DBError::PutError(format!("unable to upsert scalar data: {}", e)))?;

        Ok(())
    }

    pub fn query(
        &mut self,
        query: &SearchQuery,
        k: usize,
    ) -> Result<Vec<HashMap<String, Value>>, DBError> {
        let search_result = self
            .vector_index
            .search(query, k)
            .map_err(|e| DBError::GetError(format!("unable to query vector data: {}", e)))?;

        println!("search result inside: {:?}", search_result);

        if search_result.labels.is_empty() {
            return Ok(vec![]);
        }

        let documents = self.scalar_storage.multi_get_value(&search_result.labels)?;

        Ok(documents)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn create_test_index_params(metric_type: MetricType, index_type: IndexType) -> IndexParams {
        IndexParams {
            dim: 3,
            metric_type,
            index_type,
            hnsw_params: None,
        }
    }

    struct TestPath {
        db_path: PathBuf,
    }

    impl TestPath {
        fn new() -> Self {
            let db_path = PathBuf::from("/tmp/test_db").join(Uuid::new_v4().to_string());
            fs::create_dir_all(&db_path).unwrap();
            TestPath { db_path }
        }
    }

    impl AsRef<Path> for TestPath {
        fn as_ref(&self) -> &Path {
            self.db_path.as_ref()
        }
    }

    impl Drop for TestPath {
        fn drop(&mut self) {
            if fs::exists(&self.db_path).unwrap() {
                std::fs::remove_dir_all(&self.db_path).unwrap();
            }
        }
    }

    macro_rules! vecdb_test_cases {
        ($($name:ident: $index_type:expr, $metric_type: expr)*) => {
        $(
            mod $name {
                use super::*;

                #[test]
                fn test_vector_database_new() {
                    let index_params = create_test_index_params($metric_type, $index_type);

                    let result = VectorDatabase::new(TestPath::new(), index_params, false);
                    assert!(result.is_ok());
                }

                #[test]
                fn test_vector_database_upsert() {
                    let index_params = create_test_index_params($metric_type, $index_type);
                    let mut db = VectorDatabase::new(TestPath::new(), index_params, false).unwrap();

                    let data_array = array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]];
                    let labels = vec![1, 2];
                    let insert_data = InsertParams::new(&data_array, &labels);
                    let doc = Some(HashMap::from([(
                        "key".to_string(),
                        Value::String("value".to_string()),
                    )]));

                    let result = db.upsert(1, &insert_data, doc);
                    assert!(result.is_ok());
                }

                #[test]
                fn test_vector_database_query() {
                    let index_params = create_test_index_params($metric_type, $index_type);
                    let mut db = VectorDatabase::new(TestPath::new(), index_params, false).unwrap();

                    let data_array = array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]];
                    let labels = vec![1, 2];
                    let insert_data = InsertParams::new(&data_array, &labels);
                    db.upsert(1, &insert_data, None).unwrap();

                    let mut query = SearchQuery::new(vec![0.1, 0.2, 0.3]);
                    if $index_type == IndexType::Hnsw {
                        query = query.with(&HnswSearchOption {
                            ef_search: 10,
                        });
                    }
                    let result = db.query(&query, 1);
                    assert!(result.is_ok());
                    assert_eq!(result.unwrap().len(), 1);
                }

                #[test]
                fn test_vector_database_upsert_with_wrong_dim() {
                    let index_params = create_test_index_params($metric_type, $index_type);
                    let mut db = VectorDatabase::new(TestPath::new(), index_params, false).unwrap();

                    let data_array = array![[0.1, 0.2, 0.3, 0.4], [0.4, 0.5, 0.6, 0.7]];
                    let labels = vec![1, 2];
                    let insert_data = InsertParams::new(&data_array, &labels);
                    let result = db.upsert(1, &insert_data, None);
                    assert!(result.is_err());

                    let data_array = array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]];
                    let labels = vec![1, 2, 3];
                    let insert_data = InsertParams::new(&data_array, &labels);
                    let result = db.upsert(2, &insert_data, None);
                    assert!(result.is_err());
                }

                #[test]
                fn test_vector_database_query_with_no_results() {
                    let index_params = create_test_index_params($metric_type, $index_type);
                    let mut db = VectorDatabase::new(TestPath::new(), index_params, false).unwrap();

                    let mut query = SearchQuery::new(vec![0.1, 0.2, 0.3]);
                    if $index_type == IndexType::Hnsw {
                        query = query.with(&HnswSearchOption {
                            ef_search: 10,
                        });
                    }
                    let result = db.query(&query, 1);
                    assert!(result.is_ok());
                    assert!(result.unwrap().is_empty());
                }

                #[test]
                fn test_vector_database_query_with_filter() {
                    let index_params = create_test_index_params($metric_type, $index_type);
                    let mut db = VectorDatabase::new(TestPath::new(), index_params, false).unwrap();

                    let mut query = SearchQuery::new(vec![0.1, 0.2, 0.3]);
                    if $index_type == IndexType::Hnsw {
                        query = query.with(&HnswSearchOption {
                            ef_search: 10,
                        });
                    }
                    let result = db.query(&query, 1);
                    assert!(result.is_ok());
                    assert!(result.unwrap().is_empty());
                }
            }
        )*
        };
    }

    vecdb_test_cases! {
        flat_l2: IndexType::Flat, MetricType::L2
        hnsw_l2: IndexType::Hnsw, MetricType::L2
        flat_inner_product: IndexType::Flat, MetricType::IP
        hnsw_inner_product: IndexType::Hnsw, MetricType::IP
    }
}
