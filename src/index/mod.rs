mod flat;
mod merror;

use merror::IndexError;
use ndarray::Array2 as NMatrix;

pub trait Index {
    fn insert(&mut self, data: &Vec<f32>, label: u64) -> Result<(), IndexError>;
    fn insert_many(&mut self, data: &NMatrix<f32>, labels: &Vec<u64>) -> Result<(), IndexError>;
    fn search(&mut self, query: &Vec<f32>, k: usize) -> Result<SearchResult, IndexError>;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
