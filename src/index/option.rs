use ndarray::Array2 as NMatrix;

use super::merror::IndexError;

#[derive(Debug, Clone)]
pub struct SearchQuery {
    pub vector: Vec<f32>,

    hnsw: Option<HnswSearchOption>,
}

pub trait SearchOption {
    fn set_query(&self, query: &mut SearchQuery) -> ();
}

#[derive(Debug, Clone)]
pub struct HnswSearchOption {
    pub ef_search: u32,
}

impl SearchOption for HnswSearchOption {
    fn set_query(&self, query: &mut SearchQuery) -> () {
        query.hnsw = Some(self.clone());
    }
}

impl SearchQuery {
    pub fn new(vector: Vec<f32>) -> Self {
        Self { vector, hnsw: None }
    }

    pub fn with(&mut self, option: impl SearchOption) -> &Self {
        option.set_query(self);
        return self;
    }

    pub fn get_hnsw(&self) -> Result<&HnswSearchOption, IndexError> {
        Ok(self.hnsw.as_ref().ok_or(IndexError::QueryError(
            "HNSW search option is not set".to_string(),
        ))?)
    }
}

pub struct InsertParams<'a> {
    pub data: &'a NMatrix<f32>,
    pub labels: &'a Vec<u64>,

    pub(crate) hnsw_params: Option<HnswParams>,
}

#[derive(Debug, Clone)]
pub struct HnswParams {
    pub parallel: bool,
}

impl<'a> InsertParams<'a> {
    pub fn new(data: &'a NMatrix<f32>, labels: &'a Vec<u64>) -> Self {
        Self {
            data,
            labels,
            hnsw_params: None,
        }
    }

    pub fn with(&mut self, option: impl InsertOption) -> &Self {
        option.set_params(self);
        return self;
    }
}

pub trait InsertOption {
    fn set_params(self, params: &mut InsertParams) -> ();
}

impl InsertOption for HnswParams {
    fn set_params(self, params: &mut InsertParams) -> () {
        params.hnsw_params = Some(self);
    }
}
