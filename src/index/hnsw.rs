use crate::index::merror::IndexError;
use crate::index::{Index, MetricType, SearchResult};
use anndists::dist::{distances, Distance};
use hnsw_rs::api::{self as hnsw_api, AnnT};
use hnsw_rs::hnsw;

type FT = f32;

pub trait HnswIndexTrait: hnsw_api::AnnT<Val = FT> {
    fn get_nb_point(&self) -> usize;
}

impl<'a, D> HnswIndexTrait for hnsw::Hnsw<'a, FT, D>
where
    D: Distance<FT> + Send + Sync + 'static,
{
    fn get_nb_point(&self) -> usize {
        self.get_nb_point()
    }
}

pub struct HnswIndex {
    index: Box<dyn HnswIndexTrait>,
    dim: u32,
}

pub struct HnswIndexOption {
    pub ef_construction: u32,
    pub max_elements: u32,
    pub max_nb_connection: u32,
    pub max_layer: u32,
}

impl HnswIndex {
    fn new(dim: u32, metric_type: MetricType, option: HnswIndexOption) -> Result<Self, IndexError> {
        let index_box: Box<dyn HnswIndexTrait> = match metric_type {
            MetricType::IP => Box::new(hnsw::Hnsw::new(
                option.max_nb_connection as usize,
                option.max_elements as usize,
                option.max_layer as usize,
                option.ef_construction as usize,
                distances::DistDot,
            )),
            MetricType::L2 => Box::new(hnsw::Hnsw::new(
                option.max_nb_connection as usize,
                option.max_elements as usize,
                option.max_layer as usize,
                option.ef_construction as usize,
                distances::DistL2,
            )),
            _ => {
                return Err(IndexError::InitializationError(format!(
                    "Metric Type {:?} is not supported",
                    metric_type,
                )))
            }
        };

        Ok(Self {
            index: index_box,
            dim: dim,
        })
    }
}

impl Index for HnswIndex {
    fn insert(&mut self, params: &crate::index::option::InsertParams) -> Result<(), IndexError> {
        if params.data.nrows() != params.labels.len() {
            return Err(IndexError::InsertionError(format!(
                "Data and labels length mismatch: {} != {}",
                params.data.nrows(),
                params.labels.len()
            )));
        }

        if params.data.ncols() as u32 != self.dim {
            return Err(IndexError::InsertionError(format!(
                "Data dimension mismatch: {} != {}",
                params.data.ncols(),
                self.dim
            )));
        }

        let zipped_params = params
            .data
            .axis_iter(ndarray::Axis(0))
            .zip(params.labels.iter())
            .map(|(data, &label)| (data.to_vec(), label as usize))
            .collect::<Vec<_>>();

        if params.hnsw_params.is_some() && params.hnsw_params.as_ref().unwrap().parallel {
            let mut zipped_data_labels = Vec::<(&Vec<FT>, usize)>::new();

            for i in 0..zipped_params.len() {
                zipped_data_labels.push((&zipped_params[i].0, zipped_params[i].1));
            }

            self.index
                .parallel_insert_data(zipped_data_labels.as_slice());
        } else {
            // use insert data to add data sequentially
            for pair in zipped_params {
                self.index.insert_data(pair.0.as_slice(), pair.1);
            }
        }

        Ok(())
    }

    fn search(
        &mut self,
        query: &crate::index::option::SearchQuery,
        k: usize,
    ) -> Result<SearchResult, IndexError> {
        let hnsw_opt = query.get_hnsw()?;

        let neighbours =
            self.index
                .search_neighbours(&query.vector, k, hnsw_opt.ef_search as usize);

        Ok(SearchResult::from(neighbours))
    }
}

#[cfg(test)]
mod tests {
    use crate::index::option::HnswSearchOption;

    use super::*;
    use ndarray::Array2 as NMatrix;

    use crate::index::option::{HnswParams, InsertParams, SearchQuery};

    #[test]
    fn test_insert_many() {
        let dim = 4;
        let mut index = HnswIndex::new(
            dim,
            MetricType::L2,
            HnswIndexOption {
                ef_construction: 20,
                max_elements: 100,
                max_nb_connection: 100,
                max_layer: 16,
            },
        )
        .expect("Failed to initialize index");

        let data_from_vec =
            NMatrix::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, -1.0, 2.0, -3.0, 4.0]);
        assert!(
            data_from_vec.is_ok(),
            "got error from converting vecs into matrix {:?}",
            data_from_vec.err()
        );
        let data = data_from_vec.unwrap();

        let labels = vec![42, 47];
        use crate::index::option::InsertParams;
        let insert_result =
            index.insert(&InsertParams::new(&data, &labels).with(HnswParams { parallel: true }));

        assert!(insert_result.is_ok());
        assert!(index.index.get_nb_point() == 2);
    }

    #[test]
    fn test_search() {
        let dim = 4;
        let mut index = HnswIndex::new(
            dim,
            MetricType::L2,
            HnswIndexOption {
                ef_construction: 200,
                max_elements: 100,
                max_nb_connection: 100,
                max_layer: 16,
            },
        )
        .expect("Failed to initialize index");

        let data_from_vec =
            NMatrix::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, -1.0, 2.0, -3.0, 4.0]);
        assert!(
            data_from_vec.is_ok(),
            "error from converting vecs into matrix {:?}",
            data_from_vec.err()
        );
        let data = data_from_vec.unwrap();

        let labels = vec![42, 47];
        let insert_result = index.insert(&InsertParams::new(&data, &labels));
        assert!(insert_result.is_ok());

        let query = vec![1.1, 2.1, 2.9, 3.9];
        let k: usize = 2;
        let result = index.search(
            &SearchQuery::new(query).with(HnswSearchOption { ef_search: 20 }),
            k,
        );

        assert!(result.is_ok(), "error from search {:?}", result.err());
        let search_result = result.unwrap();
        println!("{:?}", search_result);
        assert_eq!(search_result.labels.len(), k);
        assert_eq!(search_result.labels[0], labels[0]);
    }
}
