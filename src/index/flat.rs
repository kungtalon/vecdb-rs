use crate::index::{Index, MetricType, SearchResult};
use crate::merror::IndexError;
use faiss::Index as FIndex;
use faiss::{index_factory, IdMap};
use std::cmp::min;

use crate::index::option::{InsertParams, SearchQuery};

const FLAT_INDEX_OPTION: &str = "Flat";

pub struct FlatIndex {
    index: Box<dyn FIndex>,
}

impl FlatIndex {
    pub fn new(dim: u32, metric_type: MetricType) -> Result<Self, IndexError> {
        let index = index_factory(dim, FLAT_INDEX_OPTION, faiss::MetricType::from(metric_type))
            .map_err(|e| IndexError::InitializationError(e.to_string()))?;

        // id_map index allows us to use arbitrary labels instead of contiguous ids
        let id_map_index =
            IdMap::new(index).map_err(|e| IndexError::InitializationError(e.to_string()))?;

        Ok(Self {
            index: Box::new(id_map_index),
        })
    }
}

impl Index for FlatIndex {
    fn insert(&mut self, params: &InsertParams) -> Result<(), IndexError> {
        if params.data.nrows() != params.labels.len() {
            return Err(IndexError::InsertionError(format!(
                "data rows {} and labels {} do not match",
                params.data.nrows(),
                params.labels.len()
            )));
        }

        if params.data.ncols() != self.index.d() as usize {
            return Err(IndexError::InsertionError(format!(
                "data dimension {} does not match index dimension {}",
                params.data.ncols(),
                self.index.d()
            )));
        }

        let ids = params
            .labels
            .iter()
            .map(|&id| faiss::Idx::new(id))
            .collect::<Vec<_>>();
        let data_slice_opt = params.data.as_slice();

        match data_slice_opt {
            Some(data_slice) => {
                self.index
                    .add_with_ids(data_slice, ids.as_slice())
                    .map_err(|e| IndexError::InsertionError(e.to_string()))?;
                Ok(())
            }
            _ => Err(IndexError::InsertionError(format!(
                "Failed to convert data matrix to slice {:?}",
                params.data,
            ))),
        }
    }

    fn search(&mut self, query: &SearchQuery, k: usize) -> Result<SearchResult, IndexError> {
        // if k is bigger than ntotal, set k to ntotal
        // othewise faiss will return undefined data
        let target_k = min(k, self.index.ntotal() as usize);
        if target_k == 0 {
            return Ok(SearchResult {
                distances: vec![],
                labels: vec![],
            });
        }

        let search_res: SearchResult;
        if let Some(filter) = &query.id_filter {
            search_res = self
                .index
                .search_with_filter(query.vector.as_slice(), target_k, &filter.into_selector())
                .map(SearchResult::from)
                .map_err(|e| IndexError::UnexpectedError(e.to_string()))?;

            return Ok(search_res);
        }

        search_res = self
            .index
            .search(query.vector.as_slice(), target_k)
            .map(SearchResult::from)
            .map_err(|e| IndexError::UnexpectedError(e.to_string()))?;

        Ok(search_res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::IdFilter;
    use crate::index::option::{InsertParams, SearchOption};
    use ndarray::Array2 as NMatrix;
    use ndarray_rand::{rand_distr::Uniform, RandomExt};

    fn setup(nrow: u32, dim: u32, metric_type: MetricType) -> (FlatIndex, NMatrix<f32>, Vec<u64>) {
        let index = FlatIndex::new(dim, metric_type).expect("Failed to initialize index");

        let data_from_vec = NMatrix::from_shape_vec(
            (nrow as usize, dim as usize),
            (1..(dim * nrow + 1))
                .map(|e| e as f32)
                .collect::<Vec<f32>>(),
        );
        assert!(
            data_from_vec.is_ok(),
            "got error from converting vecs into matrix {:?}",
            data_from_vec.err()
        );
        let data = data_from_vec.unwrap();

        let labels = (1u64..(nrow + 1) as u64).collect::<Vec<u64>>();

        (index, data, labels)
    }

    #[test]
    fn test_insert_many() {
        let (mut index, data, labels) = setup(2, 4, MetricType::L2);
        let insert_result = index.insert(&InsertParams::new(&data, &labels));

        assert!(insert_result.is_ok());
        assert!(index.index.ntotal() == 2);
    }

    #[test]
    fn test_search() {
        let (mut index, data, labels) = setup(2, 4, MetricType::L2);
        let insert_result = index.insert(&InsertParams::new(&data, &labels));
        assert!(insert_result.is_ok());

        let query = vec![1.1, 2.1, 2.9, 3.9];
        let k: usize = 2;
        let result = index.search(&SearchQuery::new(query), k);

        assert!(result.is_ok());
        let search_result = result.unwrap();
        println!("{:?}", search_result);
        assert_eq!(search_result.labels.len(), k);
        assert_eq!(search_result.labels[0], labels[0]);
    }

    #[test]
    fn test_search_with_filter() {
        let (mut index, data, labels) = setup(4, 5, MetricType::L2);
        let insert_result = index.insert(&InsertParams::new(&data, &labels));
        assert!(
            insert_result.is_ok(),
            "error from insert {:?}",
            insert_result
        );

        let query = vec![1.1, 2.1, 2.9, 3.9, 5.0];
        let k: usize = 3;
        let original_result = index.search(&SearchQuery::new(query.clone()), k);
        assert!(
            original_result.is_ok(),
            "error from search {:?}",
            original_result.err()
        );
        assert_eq!(original_result.unwrap().labels[0], labels[0]);

        let mut filter = IdFilter::new();
        filter.add_all(&labels[1..]);
        let result = index.search(&SearchQuery::new(query).with(&filter), k);

        assert!(result.is_ok(), "error from search {:?}", result.err());
        let search_result = result.unwrap();
        assert_eq!(search_result.labels.len(), k);
        assert_ne!(search_result.labels[0], labels[0]);
    }
}
