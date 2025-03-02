use crate::index::merror::IndexError;
use crate::index::{Index, MetricType, SearchResult};
use faiss::Index as FIndex;
use faiss::{index_factory, IdMap};

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
    }

    #[test]
    fn test_search() {
        let dim = 4;
        let mut index = FlatIndex::init(dim, MetricType::L2).expect("Failed to initialize index");

        let data = [vec![1.0, 2.0, 3.0, 4.0], vec![-1.0, 2.0, -3.0, 4.0]];
        let labels = &[42, 47];
        for i in 0..data.len() {
            index
                .insert(&data[i], labels[i])
                .expect("Failed to insert data");
        }

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
