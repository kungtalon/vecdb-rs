use crate::index::merror::IndexError;
use crate::index::{Index, MetricType, SearchResult};
use faiss::Index as FIndex;
use faiss::{index_factory, IdMap};
use ndarray::Array2 as NMatrix;
use std::{array, vec};

const FLAT_INDEX_OPTION: &str = "Flat";

pub struct FlatIndex {
    index: Box<dyn FIndex>,
}

impl FlatIndex {
    fn init(dim: u32, metric_type: MetricType) -> Result<Self, IndexError> {
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
    fn insert(&mut self, data: &Vec<f32>, label: u64) -> Result<(), IndexError> {
        let id = faiss::Idx::new(label);
        self.index
            .add_with_ids(data.as_slice(), &[id])
            .map_err(|e| IndexError::InsertionError(e.to_string()))?;

        Ok(())
    }

    fn insert_many(&mut self, data: &NMatrix<f32>, labels: &Vec<u64>) -> Result<(), IndexError> {
        let ids = labels
            .iter()
            .map(|&id| faiss::Idx::new(id))
            .collect::<Vec<_>>();
        let data_slice_opt = data.as_slice();

        match data_slice_opt {
            Some(data_slice) => {
                self.index
                    .add_with_ids(data_slice, ids.as_slice())
                    .map_err(|e| IndexError::InsertionError(e.to_string()))?;
                Ok(())
            }
            _ => Err(IndexError::InsertionError(format!(
                "Failed to convert data matrix to slice {:?}",
                data,
            ))),
        }
    }

    fn search(&mut self, query: &Vec<f32>, k: usize) -> Result<SearchResult, IndexError> {
        let search_res = self
            .index
            .search(query.as_slice(), k)
            .map(SearchResult::from)
            .map_err(|e| IndexError::UnexpectedError(e.to_string()))?;

        Ok(search_res)
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::format;

    use super::*;

    #[test]
    fn test_insert() {
        let dim = 4;
        let metric_type = MetricType::L2;
        let mut index = FlatIndex::init(dim, metric_type).expect("Failed to initialize index");

        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let label: u64 = 42;
        let result: Result<(), IndexError> = index.insert(&data, label);

        assert!(result.is_ok());
        assert!(index.index.ntotal() == 1);
    }

    #[test]
    fn test_insert_many() {
        let dim = 4;
        let mut index = FlatIndex::init(dim, MetricType::L2).expect("Failed to initialize index");

        let data_from_vec =
            NMatrix::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, -1.0, 2.0, -3.0, 4.0]);
        assert!(
            data_from_vec.is_ok(),
            "got error from converting vecs into matrix {:?}",
            data_from_vec.err()
        );
        let data = data_from_vec.unwrap();

        let labels = vec![42, 47];
        let insert_result = index.insert_many(&data, &labels);

        assert!(insert_result.is_ok());
        assert!(index.index.ntotal() == 2);
    }

    #[test]
    fn test_search() {
        let dim = 4;
        let mut index = FlatIndex::init(dim, MetricType::L2).expect("Failed to initialize index");

        let data_from_vec =
            NMatrix::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, -1.0, 2.0, -3.0, 4.0]);
        assert!(
            data_from_vec.is_ok(),
            "error from converting vecs into matrix {:?}",
            data_from_vec.err()
        );
        let data = data_from_vec.unwrap();

        let labels = vec![42, 47];
        let insert_result = index.insert_many(&data, &labels);
        assert!(insert_result.is_ok());

        let query = vec![1.1, 2.1, 2.9, 3.9];
        let k: usize = 2;
        let result = index.search(&query, k);

        assert!(result.is_ok());
        let search_result = result.unwrap();
        println!("{:?}", search_result);
        assert_eq!(search_result.labels.len(), k);
        assert_eq!(search_result.labels[0], labels[0]);
    }
}
