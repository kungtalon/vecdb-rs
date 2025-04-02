mod flat;
mod hnsw;
mod option;

use crate::merror::IndexError;
pub use flat::FlatIndex;
pub use hnsw::{HnswIndex, HnswIndexOption};
use hnsw_rs::hnsw::Neighbour;
pub use option::{HnswSearchOption, InsertParams, SearchQuery};
use serde::{Deserialize, Serialize};

pub trait Index {
    fn insert(&mut self, params: &option::InsertParams) -> Result<(), IndexError>;
    fn search(&mut self, query: &option::SearchQuery, k: usize)
        -> Result<SearchResult, IndexError>;
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub enum IndexType {
    Flat,
    Hnsw,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub distances: Vec<f32>,
    pub labels: Vec<u64>,
}

impl From<faiss::index::SearchResult> for SearchResult {
    fn from(value: faiss::index::SearchResult) -> Self {
        Self {
            distances: value.distances,
            labels: value.labels.iter().map(|i| i.to_native() as u64).collect(),
        }
    }
}

impl From<Vec<Neighbour>> for SearchResult {
    fn from(value: Vec<Neighbour>) -> Self {
        Self {
            distances: value.iter().map(|n| n.distance).collect(),
            labels: value.iter().map(|n| n.d_id as u64).collect(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MetricType {
    IP = 0,
    L2 = 1,
}

impl From<faiss::MetricType> for MetricType {
    fn from(value: faiss::MetricType) -> Self {
        match value {
            faiss::MetricType::InnerProduct => MetricType::IP,
            faiss::MetricType::L2 => MetricType::L2,
        }
    }
}

impl From<MetricType> for faiss::MetricType {
    fn from(value: MetricType) -> Self {
        match value {
            MetricType::IP => faiss::MetricType::InnerProduct,
            MetricType::L2 => faiss::MetricType::L2,
        }
    }
}
