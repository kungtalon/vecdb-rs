use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::index::{
    FlatIndex, HnswIndex, HnswIndexOption, Index, IndexType, InsertParams, MetricType, SearchQuery,
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
            IndexType::HNSW => {
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
            vector_index: vector_index,
        })
    }

    pub fn upsert(
        &mut self,
        id: u64,
        insert_data: &InsertParams,
        meta: Option<HashMap<String, Value>>,
    ) -> Result<(), DBError> {
        self.vector_index
            .insert(insert_data)
            .map_err(|e| DBError::PutError(format!("unable to upsert vector data: {}", e)))?;

        let mut meta_map: HashMap<String, Value>;

        match meta {
            Some(m) => {
                meta_map = m;
            }
            None => {
                meta_map = HashMap::new();
            }
        }

        meta_map.insert("id".to_string(), Value::Number(id.into()));

        let meta_bytes = serde_json::to_vec(&meta_map)
            .map_err(|e| DBError::PutError(format!("unable to serialize meta data: {}", e)))?;

        self.scalar_storage
            .put(id, &meta_bytes)
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

        let documents = self.scalar_storage.multi_get_value(&search_result.labels)?;

        Ok(documents)
    }
}
