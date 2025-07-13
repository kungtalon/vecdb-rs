use ndarray::Array;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::vec;
use tokio::task;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::{Arc, Mutex, RwLock};
use tracing::{event, Level};

use crate::filter::{IdFilter, IntFilterIndex, IntFilterInput};
use crate::merror::DBError;
use crate::persistence::{Persistence, WALLogOperation, WALLogRecord};
use crate::scalar::{new_scalar_storage, ScalarStorage};
use crate::{index::*, scalar};

pub type DocMap = HashMap<String, Value>;

const SCALAR_DB_FILE_SUFFIX: &str = "scalar.db";
const INDEX_FILE_SUFFIX: &str = "index.bin";
const FILTER_FILE_SUFFIX: &str = "filter.bin";
const WAL_FILE_SUFFIX: &str = "vdb.log";

pub struct VectorDatabase {
    params: DatabaseParams,

    scalar_storage: Arc<dyn ScalarStorage>,
    vector_index: Arc<Mutex<dyn Index + Send>>,
    filter_index: RwLock<IntFilterIndex>,

    persistence: Arc<Persistence>,
}

unsafe impl Sync for VectorDatabase {}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DatabaseParams {
    pub dim: u32,
    pub metric_type: MetricType,
    pub index_type: IndexType,
    pub hnsw_params: Option<HnswIndexOption>,
    pub version: String,
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

fn new_index(index_params: DatabaseParams) -> Result<Arc<Mutex<dyn Index + Send>>, DBError> {
    let index: Arc<Mutex<dyn Index + Send>> = match index_params.index_type {
        IndexType::Flat => {
            // Create a flat index
            Arc::new(Mutex::new(
                FlatIndex::new(index_params.dim, index_params.metric_type).map_err(|e| {
                    DBError::CreateError(format!("unable to create vector index: {e}"))
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
                .map_err(|e| DBError::CreateError(format!("unable to create vector index: {e}")))?,
            ))
        }
    };

    Ok(index)
}

impl VectorDatabase {
    pub fn new<D: AsRef<Path>>(db_path: D, db_params: DatabaseParams) -> Result<Self, DBError> {
        let db_params_copy = db_params.clone();

        let scalar_db_path = PathBuf::new().join(&db_path).join(SCALAR_DB_FILE_SUFFIX);
        let scalar_storage = Arc::new(new_scalar_storage(scalar_db_path)?);
        let vector_index: Arc<Mutex<dyn Index + Send>> = new_index(db_params)?;
        let filter_index = RwLock::new(IntFilterIndex::new());

        let persistence_path = PathBuf::new().join(&db_path).join(WAL_FILE_SUFFIX);
        let persistence_path_str = persistence_path.to_str().ok_or(DBError::CreateError(
            "Failed to convert persistence path to str".into(),
        ))?;
        let persistence = Arc::new(
            Persistence::new(
                persistence_path_str,
                &db_params_copy.version,
                scalar_storage.as_ref(),
            )
            .map_err(|e| {
                DBError::CreateError(format!("unable to create persistence layer: {e}"))
            })?,
        );

        Ok(Self {
            params: db_params_copy,
            scalar_storage,
            vector_index,
            filter_index,
            persistence,
        })
    }

    pub async fn upsert(&mut self, args: VectorInsertArgs) -> Result<(), DBError> {
        let (mismatch_field, mismatch_value, expect_value) = args.validate();

        if !mismatch_field.is_empty() {
            event!(
                Level::ERROR,
                "unexpected length of field {}: {}, expected length is {}",
                mismatch_field,
                mismatch_value,
                expect_value,
            );

            return Err(DBError::PutError(format!(
                "unexpected length of field {mismatch_field}: {mismatch_value}, expected length is {expect_value}",
            )));
        }

        let ids: Vec<u64> = self
            .scalar_storage
            .gen_incr_ids(scalar::NAMESPACE_DOCS, args.data_row)?;

        event!(Level::DEBUG, "upsert vector data with ids: {:?}", ids);

        self.insert_vectors(ids.clone(), &args).await?;

        let mut attributes = args.attributes;
        if attributes.is_empty() {
            attributes = vec![Some(HashMap::new()); args.data_row];
        }

        for ((i, doc), attr) in args
            .docs
            .into_iter()
            .enumerate()
            .zip(attributes.into_iter())
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

    async fn insert_vectors(
        &mut self,
        ids: Vec<u64>,
        args: &VectorInsertArgs,
    ) -> Result<(), DBError> {
        let insert_data =
            Array::from_shape_vec((args.data_row, args.data_dim), args.flat_data.clone()).map_err(
                |e| DBError::PutError(format!("unable to create array from flat data: {e}")),
            )?;

        let vector_index_writer = Arc::clone(&self.vector_index);
        let hnsw_params = args.hnsw_params.clone();

        let res_async_insert = task::spawn_blocking(move || {
            let index_insert_params = InsertParams {
                data: &insert_data,
                labels: &ids,
                hnsw_params,
            };

            vector_index_writer
                .lock()
                .unwrap()
                .insert(&index_insert_params)
                .map_err(|e| DBError::PutError(format!("unable to upsert vector data: {e}")))
        })
        .await;

        res_async_insert.map_err(|e| {
            DBError::PutError(format!(
                "error while inserting vector database asynchronously: {e}",
            ))
        })??;

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
            .map_err(|e| DBError::PutError(format!("unable to serialize doc data: {e}")))?;

        let scalar_storage = Arc::clone(&self.scalar_storage);
        task::spawn_blocking(move || {
            scalar_storage
                .put(&id.to_be_bytes(), &doc_bytes)
                .map_err(|e| DBError::PutError(format!("unable to upsert scalar data: {e}")))?;
            Ok(())
        })
        .await
        .map_err(|e| {
            DBError::PutError(format!(
                "error while inserting scalar data asynchronously: {e}",
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
                        "unsupported attribute type for key {key}: {value:?}",
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
                .map_err(|e| DBError::GetError(format!("unable to query vector data: {e}")))
        })
        .await
        .map_err(|e| {
            DBError::GetError(format!(
                "error while querying vector database asynchronously: {e}",
            ))
        })??;

        event!(Level::DEBUG, "search result inside: {search_result:?}");

        if search_result.labels.is_empty() {
            return Ok(vec![]);
        }

        #[cfg(debug_assertions)]
        {
            use crate::scalar::debug_print_scalar_db;

            event!(
                Level::DEBUG,
                "start printing all contents in scalar storage",
            );
            debug_print_scalar_db(&*self.scalar_storage)?;
        }

        let documents = self.scalar_storage.multi_get_value(&search_result.labels)?;

        Ok(documents)
    }

    pub async fn recover_database(&mut self) -> Result<(), DBError> {
        event!(
            Level::INFO,
            "Recovering vector database from saved files..."
        );

        let wal_record_iter =
            self.persistence.get_wal_log_iterator().await.map_err(|e| {
                DBError::CreateError(format!("Failed to get WAL log iterator: {e}"))
            })?;
        for record in wal_record_iter {
            match record {
                Ok(record) => {
                    // Apply the WAL log record to the database
                    self.apply_wal_log_record(record);
                }
                Err(e) => {
                    event!(Level::ERROR, "Failed to read WAL log record: {e}");

                    return Err(DBError::CreateError(format!(
                        "Failed to read WAL log record: {e}"
                    )));
                }
            }
        }

        Ok(())
    }

    fn apply_wal_log_record(&mut self, record: WALLogRecord) {
        match record.operation {
            WALLogOperation::Insert => {
                // Apply insert operation
            }
            WALLogOperation::Update => {
                // Apply update operation
            }
            WALLogOperation::Delete => {
                // Apply delete operation
            }
        }
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
    use std::sync::Once;
    use std::time::SystemTime;
    use std::{fs, time::UNIX_EPOCH};
    use tracing::span;
    use tracing_subscriber::fmt;
    use uuid::Uuid;

    fn create_test_index_params(metric_type: MetricType, index_type: IndexType) -> DatabaseParams {
        DatabaseParams {
            dim: 3,
            metric_type,
            index_type,
            hnsw_params: None,
            version: "0.1.0".to_string(),
        }
    }

    struct TestPath {
        db_path: PathBuf,
    }

    impl TestPath {
        fn new() -> Self {
            let db_path = PathBuf::from("/tmp/test_db")
                .join(Uuid::new_v4().to_string())
                .join(
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs()
                        .to_string(),
                );
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

    static INIT: Once = Once::new();

    fn init_tracing(test_name: &str) -> tracing::Span {
        INIT.call_once(|| {
            fmt::Subscriber::builder()
                .with_max_level(Level::DEBUG)
                .with_span_events(fmt::format::FmtSpan::ENTER | fmt::format::FmtSpan::CLOSE)
                .with_line_number(true)
                .init();
        });

        span!(Level::DEBUG, "tracing for test", test_name)
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
                    let span = init_tracing("test_vector_database_upsert");
                    let _enter = span.enter();

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
                    let span = init_tracing("test_vector_database_query");
                    let _enter = span.enter();

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

                    let data_array = standardize_vecs(&array![[0.1, 0.2, 0.3], [-0.1, 0.2, -0.3]]);
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
                        k: 10,
                        filter_inputs: None,
                        hnsw_params: None,
                    };
                    if $index_type == IndexType::Hnsw {
                        search_args.hnsw_params = Some(HnswSearchOption {
                            ef_search: 200,
                        });
                    }
                    let result = db.query(search_args).await;

                    assert!(result.is_ok());
                    let docs_result = result.unwrap();

                    if $index_type == IndexType::Flat {
                        assert_eq!(docs_result.len(), 2);
                        assert_eq!(doc1.get("key"), docs_result[0].get("key"))
                    } else {
                        // HNSW index does not guarantee the number of results
                        assert!(!docs_result.is_empty());
                    }
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
                    let span = init_tracing("test_vector_database_query_with_no_results");
                    let _enter = span.enter();

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
                            ef_search: 200,
                        });
                    }
                    let result = db.query(search_args).await;

                    assert!(result.is_ok());
                    assert!(result.unwrap().is_empty());
                }

                #[tokio::test]
                async fn test_vector_database_query_with_filter() {
                    let span = init_tracing("test_vector_database_query_with_filter");
                    let _enter = span.enter();

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

                    let data_array = standardize_vecs(&array![[0.1, 0.2, 0.3], [0.1, -0.2, 0.3]]);
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
                        k: 10,
                        filter_inputs: Some(vec![IntFilterInput {
                            field: "age".to_string(),
                            op: FilterOp::Equal,
                            target: 20,
                        }]),
                        hnsw_params: None,
                    };

                    if $index_type == IndexType::Hnsw {
                        search_args.hnsw_params = Some(HnswSearchOption {
                            ef_search: 200,
                        });
                    }

                    // without filter, the first doc should be returned
                    // filter the first doc, then only the second doc should be returned
                    let result = db.query(search_args.clone()).await;

                    assert!(result.is_ok());
                    let docs_result = result.unwrap();
                    assert_eq!(docs_result.len(), 1);
                    if $index_type == IndexType::Flat {
                        assert_eq!(doc2.get("key"), docs_result[0].get("key"));
                    } else {
                        // HNSW index does not guarantee the number of results
                        assert!(!docs_result.is_empty());
                    }

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
                    if $index_type == IndexType::Flat {
                        assert_eq!(doc1.get("key"), docs_result[0].get("key"));
                    } else {
                        // HNSW index does not guarantee the number of results
                        assert!(!docs_result.is_empty());
                    }
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
