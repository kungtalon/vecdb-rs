use hnsw_rs::hnsw;
use ndarray::{array, Array};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::vec;

use serde::{de, Deserialize, Serialize};
use serde_json::Value;
use std::marker::Sync;
use std::sync::{Arc, Mutex};
use uuid::Uuid;

use crate::filter::{FilterOp, IdFilter, IntFilterIndex, IntFilterInput};
use crate::index::*;
use crate::merror::DBError;
use crate::scalar::{new_concurrent_scalar_storage, ScalarStorage};

pub type DocMap = HashMap<String, Value>;

pub struct VectorDatabase {
    db_path: PathBuf,
    scalar_storage: Arc<Mutex<dyn ScalarStorage>>,
    vector_index: Box<dyn Index>,
    filter_index: IntFilterIndex,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct IndexParams {
    dim: u32,
    metric_type: MetricType,
    index_type: IndexType,
    hnsw_params: Option<HnswIndexOption>,
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
    pub fn new<D: AsRef<Path>>(
        db_path: D,
        index_params: IndexParams,
        concurrent: bool,
    ) -> Result<Self, DBError> {
        let db_path = PathBuf::new().join(db_path);
        let scalar_storage = new_concurrent_scalar_storage(&db_path, concurrent)?;
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
            filter_index: IntFilterIndex::new(),
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
            let mut scalar_storage_guard = self
                .scalar_storage
                .lock()
                .map_err(|e| DBError::PutError(format!("unable to lock scalar storage: {}", e)))?;

            ids = scalar_storage_guard.gen_incr_ids(args.data_row)?;
        }

        let mut index_insert_params = InsertParams::new(&insert_data, &ids);
        if args.hnsw_params.is_some() {
            index_insert_params = index_insert_params.with(args.hnsw_params.unwrap());
        }

        self.vector_index
            .insert(&index_insert_params)
            .map_err(|e| DBError::PutError(format!("unable to upsert vector data: {}", e)))?;

        for (i, doc) in args.docs.iter().enumerate() {
            let mut doc_map = match doc {
                Some(m) => m.clone().to_owned(),
                None => HashMap::new(),
            };
            self.insert_doc(&mut doc_map, ids[i]).await?;
        }

        for (i, attr) in args.attributes.iter().enumerate() {
            if let Some(attr_map) = attr {
                self.insert_attribute(attr_map, ids[i]).await?;
            }
        }

        Ok(())
    }

    async fn insert_doc(&mut self, doc: &mut DocMap, id: u64) -> Result<(), DBError> {
        doc.insert("id".to_string(), Value::Number(id.into()));

        let doc_bytes = serde_json::to_vec(&doc)
            .map_err(|e| DBError::PutError(format!("unable to serialize doc data: {}", e)))?;

        let mut scalar_storage_guard = self
            .scalar_storage
            .lock()
            .map_err(|e| DBError::PutError(format!("unable to lock scalar storage: {}", e)))?;

        scalar_storage_guard
            .put(id, &doc_bytes)
            .map_err(|e| DBError::PutError(format!("unable to upsert scalar data: {}", e)))?;

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
                        self.filter_index.upsert(&key, num, id);
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

    pub async fn query(
        &mut self,
        search_args: VectorSearchArgs, // should not use SearchQuery here
    ) -> Result<Vec<DocMap>, DBError> {
        let mut query = SearchQuery::new(&search_args.query);
        if search_args.hnsw_params.is_some() {
            query = query.with(search_args.hnsw_params.as_ref().unwrap());
        }
        if search_args.filter_inputs.is_some() {
            let mut bitmap = roaring::RoaringBitmap::new();
            for filter in search_args.filter_inputs.unwrap() {
                bitmap = self.filter_index.apply(&filter, &bitmap);
            }
            query = query.with(&IdFilter::from(bitmap));
        }

        let search_result = self
            .vector_index
            .search(&query, search_args.k)
            .map_err(|e| DBError::GetError(format!("unable to query vector data: {}", e)))?;

        println!("search result inside: {:?}", search_result);

        if search_result.labels.is_empty() {
            return Ok(vec![]);
        }

        let scalar_storage_guard = self
            .scalar_storage
            .lock()
            .map_err(|e| DBError::GetError(format!("unable to lock scalar storage: {}", e)))?;

        let documents = scalar_storage_guard.multi_get_value(&search_result.labels)?;

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

                    let result = VectorDatabase::new(TestPath::new(), index_params, false);
                    assert!(result.is_ok());
                }

                #[tokio::test]
                async fn test_vector_database_upsert() {
                    let index_params = create_test_index_params($metric_type, $index_type);
                    let mut db = VectorDatabase::new(TestPath::new(), index_params, false).unwrap();

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
                    let mut db = VectorDatabase::new(TestPath::new(), index_params, false).unwrap();

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
                    let mut db = VectorDatabase::new(TestPath::new(), index_params, false).unwrap();

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
                    let mut db = VectorDatabase::new(TestPath::new(), index_params, false).unwrap();

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
                    let mut db = VectorDatabase::new(TestPath::new(), index_params, false).unwrap();

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
