use ndarray::Array;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::vec;
use tokio::task;
use tower::filter;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::{Arc, Mutex, RwLock};
use tracing::{event, Level};

use crate::filter::{IdFilter, IntFilterIndex, IntFilterInput};
use crate::index::*;
use crate::merror::DBError;
use crate::scalar::{new_scalar_storage, ScalarStorage};

pub type DocMap = HashMap<String, Value>;

pub struct VectorDatabase {
    params: Arc<IndexParams>,

    scalar_storage: Arc<RwLock<dyn ScalarStorage>>,
    vector_index: Arc<Mutex<dyn Index + Send>>,
    filter_index: Arc<RwLock<IntFilterIndex>>,
}

impl Clone for VectorDatabase {
    fn clone(&self) -> Self {
        Self {
            params: Arc::clone(&self.params),
            scalar_storage: Arc::clone(&self.scalar_storage),
            vector_index: Arc::clone(&self.vector_index),
            filter_index: Arc::clone(&self.filter_index),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct IndexParams {
    pub dim: u32,
    pub metric_type: MetricType,
    pub index_type: IndexType,
    pub hnsw_params: Option<HnswIndexOption>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct VectorInsertArgs {
    pub flat_data: Vec<f32>,
    pub data_row: usize,
    pub data_dim: usize,
    pub docs: Vec<Option<DocMap>>,
    pub attributes: Vec<Option<HashMap<String, Value>>>,

    pub hnsw_params: Option<HnswParams>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct VectorSearchArgs {
    pub query: Vec<f32>,
    pub k: usize,
    pub filter_inputs: Option<Vec<IntFilterInput>>,

    pub hnsw_params: Option<HnswSearchOption>,
}

impl VectorDatabase {
    pub fn new<D: AsRef<Path>>(db_path: D, index_params: IndexParams) -> Result<Self, DBError> {
        let index_params_copy = index_params.clone();

        let db_path = PathBuf::new().join(db_path);
        let scalar_storage = Arc::new(RwLock::new(new_scalar_storage(db_path)?));
        let vector_index: Arc<Mutex<dyn Index + Send>> = match index_params.index_type {
            IndexType::Flat => {
                // Create a flat index
                Arc::new(Mutex::new(
                    FlatIndex::new(index_params.dim, index_params.metric_type).map_err(|e| {
                        DBError::CreateError(format!("unable to create vector index: {}", e))
                    })?,
                ))
            }
            IndexType::Hnsw => {
                // Create an HNSW index
                Arc::new(Mutex::new(
                    HnswIndex::new(
                        index_params.dim,
                        index_params.metric_type,
                        index_params.hnsw_params,
                    )
                    .map_err(|e| {
                        DBError::CreateError(format!("unable to create vector index: {}", e))
                    })?,
                ))
            }
        };

        Ok(Self {
            params: Arc::new(index_params_copy),
            scalar_storage,
            vector_index,
            filter_index: Arc::new(RwLock::new(IntFilterIndex::new())),
        })
    }

    pub async fn upsert(&mut self, args: VectorInsertArgs) -> Result<(), DBError> {
        let (mismatch_field, mismatch_value, expect_value) = args.validate();

        if !mismatch_field.is_empty() {
            return Err(DBError::PutError(format!(
                "unexpected length of field {}: {}, expected length is {}",
                mismatch_field, mismatch_value, expect_value,
            )));
        }

        let insert_data = Array::from_shape_vec((args.data_row, args.data_dim), args.flat_data)
            .map_err(|e| {
                DBError::PutError(format!("unable to create array from flat data: {}", e))
            })?;

        let ids: Vec<u64>;
        {
            let mut scalar_storage_writer = self.scalar_storage.write().unwrap();

            ids = scalar_storage_writer.gen_incr_ids(args.data_row)?;
        }

        event!(Level::DEBUG, "upsert vector data with ids: {:?}", ids);

        {
            let vector_index_writer = Arc::clone(&self.vector_index);
            let insert_data_clone = insert_data.clone();
            let ids_clone = ids.clone();

            let res_async_insert = task::spawn_blocking(move || {
                let index_insert_params = InsertParams {
                    data: &insert_data_clone,
                    labels: &ids_clone,
                    hnsw_params: args.hnsw_params,
                };

                vector_index_writer
                    .lock()
                    .unwrap()
                    .insert(&index_insert_params)
                    .map_err(|e| DBError::PutError(format!("unable to upsert vector data: {}", e)))
            })
            .await;

            res_async_insert.map_err(|e| {
                DBError::PutError(format!(
                    "error while inserting vector database asynchronously: {}",
                    e
                ))
            })??;
        }

        for ((i, doc), attr) in args
            .docs
            .into_iter()
            .enumerate()
            .zip(args.attributes.into_iter())
        {
            let mut doc_map = match doc {
                Some(m) => m.clone().to_owned(),
                None => HashMap::new(),
            };

            let attr_map = attr.unwrap_or_default();

            self.insert_doc(&mut doc_map, &attr_map, ids[i]).await?;

            if !attr_map.is_empty() {
                self.insert_attribute(&attr_map, ids[i]).await?;
            }
        }

        Ok(())
    }

    async fn insert_doc(
        &mut self,
        doc: &mut DocMap,
        attributes: &HashMap<String, Value>,
        id: u64,
    ) -> Result<(), DBError> {
        doc.insert("id".to_string(), Value::Number(id.into()));
        doc.insert(
            "attributes".to_string(),
            serde_json::to_value(attributes).unwrap(),
        );

        let doc_bytes = serde_json::to_vec(&doc)
            .map_err(|e| DBError::PutError(format!("unable to serialize doc data: {}", e)))?;

        let scalar_storage = Arc::clone(&self.scalar_storage);
        task::spawn_blocking(move || {
            scalar_storage
                .write()
                .unwrap()
                .put(id, &doc_bytes)
                .map_err(|e| DBError::PutError(format!("unable to upsert scalar data: {}", e)))?;
            Ok(())
        })
        .await
        .map_err(|e| {
            DBError::PutError(format!(
                "error while inserting scalar data asynchronously: {}",
                e
            ))
        })??;

        Ok(())
    }

    async fn insert_attribute(
        &mut self,
        attr: &HashMap<String, Value>,
        id: u64,
    ) -> Result<(), DBError> {
        for (key, value) in attr {
            match value {
                Value::Number(num) => {
                    if let Some(num) = num.as_i64() {
                        self.filter_index.write().unwrap().upsert(key, num, id);
                    }
                }
                _ => {
                    return Err(DBError::PutError(format!(
                        "unsupported attribute type for key {}: {:?}",
                        key, value
                    )));
                }
            }
        }

        Ok(())
    }

    pub async fn query(&mut self, search_args: VectorSearchArgs) -> Result<Vec<DocMap>, DBError> {
        let mut query = SearchQuery::new(search_args.query);

        if query.vector.len() != self.params.dim as usize {
            return Err(DBError::GetError(format!(
                "query vector length {} does not match index dimension {}",
                query.vector.len(),
                self.params.dim,
            )));
        }

        if search_args.hnsw_params.is_some() {
            query = query.with(search_args.hnsw_params.as_ref().unwrap());
        }

        match &search_args.filter_inputs {
            Some(filter_inputs) if !filter_inputs.is_empty() => {
                let mut bitmap = roaring::RoaringBitmap::new();

                for filter in search_args.filter_inputs.unwrap() {
                    bitmap = self.filter_index.read().unwrap().apply(&filter, &bitmap);
                }

                query = query.with(&IdFilter::from(bitmap));
            }
            _ => {}
        }

        let vector_index = Arc::clone(&self.vector_index);

        let search_result = task::spawn_blocking(move || {
            vector_index
                .lock()
                .unwrap()
                .search(&query, search_args.k)
                .map_err(|e| DBError::GetError(format!("unable to query vector data: {}", e)))
        })
        .await
        .map_err(|e| {
            DBError::GetError(format!(
                "error while querying vector database asynchronously: {}",
                e
            ))
        })??;

        event!(Level::DEBUG, "search result inside: {:?}", search_result);

        if search_result.labels.is_empty() {
            return Ok(vec![]);
        }

        let documents = self
            .scalar_storage
            .read()
            .unwrap()
            .multi_get_value(&search_result.labels)?;

        Ok(documents)
    }
}

impl VectorInsertArgs {
    fn validate(&self) -> (&str, usize, usize) {
        if self.docs.len() != self.data_row {
            return ("docs", self.docs.len(), self.data_row);
        }

        if !self.attributes.is_empty() && self.attributes.len() != self.data_row {
            return ("attributes", self.attributes.len(), self.data_row);
        }

        if self.data_dim * self.data_row != self.flat_data.len() {
            return (
                "flat_data",
                self.flat_data.len(),
                self.data_dim * self.data_row,
            );
        }

        ("", 0, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::FilterOp;
    use ndarray::array;
    use std::fs;
    use uuid::Uuid;

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

    fn standardize_vecs(
        matrix: &Array<f32, ndarray::Dim<[usize; 2]>>,
    ) -> Array<f32, ndarray::Dim<[usize; 2]>> {
        let mut result = matrix.clone();

        for i in 0..matrix.shape()[0] {
            let row = matrix.row(i);
            let norm = row.dot(&row).sqrt();

            result.row_mut(i).assign(&row.map(|x| x / norm));
        }

        result
    }

    macro_rules! vecdb_test_cases {
        ($($name:ident: $index_type:expr, $metric_type: expr)*) => {
        $(
            mod $name {
                use super::*;

                #[test]
                fn test_vector_database_new() {
                    let index_params = create_test_index_params($metric_type, $index_type);

                    let result = VectorDatabase::new(TestPath::new(), index_params);
                    assert!(result.is_ok());
                }

                #[tokio::test]
                async fn test_vector_database_upsert() {
                    let index_params = create_test_index_params($metric_type, $index_type);
                    let mut db = VectorDatabase::new(TestPath::new(), index_params).unwrap();

                    let data_array = standardize_vecs(&array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]);
                    let doc = HashMap::from([(
                        "key".to_string(),
                        Value::String("value".to_string()),
                    )]);

                    let result = db.upsert(VectorInsertArgs{
                        flat_data: data_array.iter().map(|x| *x).collect(),
                        data_row: 2,
                        data_dim: 3,
                        docs: vec![Some(doc.clone()), Some(doc.clone())],
                        attributes: vec![],
                        hnsw_params: None,
                    }).await;

                    assert!(result.is_ok());
                }

                #[tokio::test]
                async fn test_vector_database_query() {
                    let index_params = create_test_index_params($metric_type, $index_type);
                    let mut db = VectorDatabase::new(TestPath::new(), index_params).unwrap();

                    let doc1 = HashMap::from([(
                        "key".to_string(),
                        Value::String("value1".to_string()),
                    )]);

                    let doc2 = HashMap::from([(
                        "key".to_string(),
                        Value::String("value2".to_string()),
                    )]);

                    let data_array = standardize_vecs(&array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]);
                    let res = db.upsert(VectorInsertArgs{
                        flat_data: data_array.iter().map(|x| *x).collect(),
                        data_row: 2,
                        data_dim: 3,
                        docs: vec![Some(doc1.clone()), Some(doc2)],
                        attributes: vec![],
                        hnsw_params: None,
                    }).await;

                    assert!(res.is_ok(), "upsert failed: {:?}", res.err().unwrap());

                    let mut search_args = VectorSearchArgs {
                        query: vec![0.1, 0.2, 0.3],
                        k: 2,
                        filter_inputs: None,
                        hnsw_params: None,
                    };
                    if $index_type == IndexType::Hnsw {
                        search_args.hnsw_params = Some(HnswSearchOption {
                            ef_search: 10,
                        });
                    }
                    let result = db.query(search_args).await;

                    assert!(result.is_ok());
                    let docs_result = result.unwrap();
                    assert_eq!(docs_result.len(), 2);
                    assert_eq!(doc1.get("key"), docs_result[0].get("key"))
                }

                #[tokio::test]
                async fn test_vector_database_upsert_with_wrong_dim() {
                    let index_params = create_test_index_params($metric_type, $index_type);
                    let mut db = VectorDatabase::new(TestPath::new(), index_params).unwrap();

                    let data_array = standardize_vecs(&array![[0.1, 0.2, 0.3, 0.4], [0.4, 0.5, 0.6, 0.7]]);
                    let result = db.upsert(VectorInsertArgs{
                        flat_data: data_array.iter().map(|x| *x).collect(),
                        data_row: 2,
                        data_dim: 4,
                        docs: vec![None, None],
                        attributes: vec![],
                        hnsw_params: None,
                    }).await;

                    assert!(result.is_err());

                    let data_array = standardize_vecs(&array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]);
                    let result = db.upsert(VectorInsertArgs{
                        flat_data: data_array.iter().map(|x| *x).collect(),
                        data_row: 3,
                        data_dim: 3,
                        docs: vec![None, None, None],
                        attributes: vec![],
                        hnsw_params: None,
                    }).await;

                    assert!(result.is_err());
                }

                #[tokio::test]
                async fn test_vector_database_query_with_no_results() {
                    let index_params = create_test_index_params($metric_type, $index_type);
                    let mut db = VectorDatabase::new(TestPath::new(), index_params).unwrap();

                    let mut search_args = VectorSearchArgs {
                        query: vec![0.1, 0.2, 0.3],
                        k: 1,
                        filter_inputs: None,
                        hnsw_params: None,
                    };
                    if $index_type == IndexType::Hnsw {
                        search_args.hnsw_params = Some(HnswSearchOption {
                            ef_search: 10,
                        });
                    }
                    let result = db.query(search_args).await;

                    assert!(result.is_ok());
                    assert!(result.unwrap().is_empty());
                }

                #[tokio::test]
                async fn test_vector_database_query_with_filter() {
                    let index_params = create_test_index_params($metric_type, $index_type);
                    let mut db = VectorDatabase::new(TestPath::new(), index_params).unwrap();

                    let doc1 = HashMap::from([(
                        "key".to_string(),
                        Value::String("value1".to_string()),
                    )]);

                    let doc2 = HashMap::from([(
                        "key".to_string(),
                        Value::String("value2".to_string()),
                    )]);

                    let data_array = standardize_vecs(&array![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]);
                    let res = db.upsert(VectorInsertArgs{
                        flat_data: data_array.iter().map(|x| *x).collect(),
                        data_row: 2,
                        data_dim: 3,
                        docs: vec![Some(doc1.clone()), Some(doc2.clone())],
                        attributes: vec![
                            Some(HashMap::from([("age".to_string(), Value::Number(10.into()))])),
                            Some(HashMap::from([("age".to_string(), Value::Number(20.into()))])),
                        ],
                        hnsw_params: None,
                    }).await;

                    assert!(res.is_ok(), "upsert failed: {:?}", res.err().unwrap());

                    let mut search_args = VectorSearchArgs {
                        query: vec![0.1, 0.2, 0.3],
                        k: 2,
                        filter_inputs: Some(vec![IntFilterInput {
                            field: "age".to_string(),
                            op: FilterOp::Equal,
                            target: 20,
                        }]),
                        hnsw_params: None,
                    };

                    if $index_type == IndexType::Hnsw {
                        search_args.hnsw_params = Some(HnswSearchOption {
                            ef_search: 10,
                        });
                    }

                    // without filter, the first doc should be returned
                    // filter the first doc, then only the second doc should be returned
                    let result = db.query(search_args.clone()).await;

                    assert!(result.is_ok());
                    let docs_result = result.unwrap();
                    assert_eq!(docs_result.len(), 1);
                    assert_eq!(doc2.get("key"), docs_result[0].get("key"));

                    // test with not equal filter
                    search_args.filter_inputs = Some(vec![IntFilterInput {
                        field: "age".to_string(),
                        op: FilterOp::NotEqual,
                        target: 20,
                    }]);
                    let result = db.query(search_args).await;

                    assert!(result.is_ok());
                    let docs_result = result.unwrap();
                    assert_eq!(docs_result.len(), 1);
                    assert_eq!(doc1.get("key"), docs_result[0].get("key"));
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
