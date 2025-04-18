use ndarray::Array2 as NMatrix;

use crate::{filter::IdFilter, merror::IndexError};

#[derive(Debug)]
pub struct SearchQuery {
    pub vector: Vec<f32>,
    pub id_filter: Option<IdFilter>,

    hnsw: Option<HnswSearchOption>,
}

pub trait SearchOption {
    fn set_query(&self, query: &mut SearchQuery);
}

#[derive(Debug, Clone)]
pub struct HnswSearchOption {
    pub ef_search: u32,
}

impl SearchOption for HnswSearchOption {
    fn set_query(&self, query: &mut SearchQuery) {
        query.hnsw = Some(self.clone());
    }
}

impl SearchOption for IdFilter {
    fn set_query(&self, query: &mut SearchQuery) {
        query.id_filter = Some(self.clone());
    }
}

impl SearchQuery {
    pub fn new(vector: Vec<f32>) -> Self {
        Self {
            vector,
            hnsw: None,
            id_filter: None,
        }
    }

    pub fn with(mut self, option: &dyn SearchOption) -> Self {
        option.set_query(&mut self);
        self
    }

    pub fn get_hnsw(&self) -> Result<&HnswSearchOption, IndexError> {
        let result = self.hnsw.as_ref().ok_or(IndexError::QueryError(
            "HNSW search option is not set".to_string(),
        ))?;

        Ok(result)
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

    pub fn with(mut self, option: impl InsertOption) -> Self {
        option.set_params(&mut self);
        self
    }
}

pub trait InsertOption {
    fn set_params(self, params: &mut InsertParams);
}

impl InsertOption for HnswParams {
    fn set_params(self, params: &mut InsertParams) {
        params.hnsw_params = Some(self);
    }
}

struct MyStruct<'a> {
    data: &'a str, // A reference field with a limited lifetime
}

impl<'a> MyStruct<'a> {
    // Constructor
    fn new(data: &'a str) -> Self {
        Self { data }
    }

    // Setter method to update the reference field
    fn set_data(&mut self, new_data: &'a str) {
        self.data = new_data;
    }

    fn get_data(&self) -> &str {
        self.data
    }
}
