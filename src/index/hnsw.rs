use core::alloc;

use crate::filter::IdFilter;
use crate::index::option::{InsertParams, SearchQuery};
use crate::index::{Index, MetricType, SearchResult};
use crate::merror::IndexError;
use anndists::dist::{distances, Distance};
use hnsw_rs::api::{self as hnsw_api, AnnT};
use hnsw_rs::filter::FilterT;
use hnsw_rs::hnsw;
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};

type FT = f32;

const DEFAULT_EF_CONSTRUCTION: u32 = 200;
const DEFAULT_MAX_ELEMENTS: u32 = 500;
const DEFAULT_MAX_NB_CONNECTION: u32 = 16;
const DEFAULT_MAX_LAYER: u32 = 3;

lazy_static! {
    static ref EF_CONSTRUCTION: u32 =
        read_hnsw_config_from_env("HNSW_EF_CONSTRUCTION", DEFAULT_EF_CONSTRUCTION);
    static ref MAX_ELEMENTS: u32 =
        read_hnsw_config_from_env("HNSW_MAX_ELEMENTS", DEFAULT_MAX_ELEMENTS);
    static ref MAX_NB_CONNECTION: u32 =
        read_hnsw_config_from_env("HNSW_MAX_NB_CONNECTION", DEFAULT_MAX_NB_CONNECTION);
    static ref MAX_LAYER: u32 = read_hnsw_config_from_env("HNSW_MAX_LAYER", DEFAULT_MAX_LAYER);
}

fn read_hnsw_config_from_env(name: &str, default: u32) -> u32 {
    match std::env::var(name) {
        Ok(v) => v.parse::<u32>().unwrap_or(default),
        Err(_) => default,
    }
}

pub trait HnswIndexTrait: hnsw_api::AnnT<Val = FT> {
    fn get_nb_point(&self) -> usize;

    fn search_filter(
        &self,
        data: &[FT],
        knbn: usize,
        ef_arg: usize,
        filter: &IdFilter,
    ) -> Vec<hnsw::Neighbour>;
}

impl<D> HnswIndexTrait for hnsw::Hnsw<'_, FT, D>
where
    D: Distance<FT> + Send + Sync + 'static,
{
    fn get_nb_point(&self) -> usize {
        self.get_nb_point()
    }

    fn search_filter(
        &self,
        data: &[FT],
        knbn: usize,
        ef_arg: usize,
        filter: &IdFilter,
    ) -> Vec<hnsw::Neighbour> {
        self.search_filter(data, knbn, ef_arg, Some(filter))
    }
}

pub struct HnswIndex {
    index: Box<dyn HnswIndexTrait>,
    dim: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HnswIndexOption {
    pub ef_construction: Option<u32>,
    pub max_elements: Option<u32>,
    pub max_nb_connection: Option<u32>,
    pub max_layer: Option<u32>,
}

struct HnswIndexSetting {
    ef_construction: u32,
    max_elements: u32,
    max_nb_connection: u32,
    max_layer: u32,
}

impl From<HnswIndexOption> for HnswIndexSetting {
    fn from(option: HnswIndexOption) -> Self {
        Self {
            ef_construction: option.ef_construction.unwrap_or(*EF_CONSTRUCTION),
            max_elements: option.max_elements.unwrap_or(*MAX_ELEMENTS),
            max_nb_connection: option.max_nb_connection.unwrap_or(*MAX_NB_CONNECTION),
            max_layer: option.max_layer.unwrap_or(*MAX_LAYER),
        }
    }
}

impl From<HnswIndexSetting> for Option<HnswIndexOption> {
    fn from(setting: HnswIndexSetting) -> Self {
        Some(HnswIndexOption {
            ef_construction: Some(setting.ef_construction),
            max_elements: Some(setting.max_elements),
            max_nb_connection: Some(setting.max_nb_connection),
            max_layer: Some(setting.max_layer),
        })
    }
}

impl HnswIndex {
    pub fn new(
        dim: u32,
        metric_type: MetricType,
        option: Option<HnswIndexOption>,
    ) -> Result<Self, IndexError> {
        let setting = match option {
            Some(opt) => HnswIndexSetting::from(opt),
            None => HnswIndexSetting {
                ef_construction: *EF_CONSTRUCTION,
                max_elements: *MAX_ELEMENTS,
                max_nb_connection: *MAX_NB_CONNECTION,
                max_layer: *MAX_LAYER,
            },
        };

        let index_box: Box<dyn HnswIndexTrait> = match metric_type {
            MetricType::IP => Box::new(hnsw::Hnsw::new(
                setting.max_nb_connection as usize,
                setting.max_elements as usize,
                setting.max_layer as usize,
                setting.ef_construction as usize,
                distances::DistDot,
            )),
            MetricType::L2 => Box::new(hnsw::Hnsw::new(
                setting.max_nb_connection as usize,
                setting.max_elements as usize,
                setting.max_layer as usize,
                setting.ef_construction as usize,
                distances::DistL2,
            )),
        };

        Ok(Self {
            index: index_box,
            dim,
        })
    }
}

impl Index for HnswIndex {
    fn insert(&mut self, params: &InsertParams) -> Result<(), IndexError> {
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

            for (data, label) in zipped_params.iter() {
                zipped_data_labels.push((data, *label));
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

    fn search(&mut self, query: &SearchQuery, k: usize) -> Result<SearchResult, IndexError> {
        let hnsw_opt = query.get_hnsw()?;

        if let Some(filter) = &query.id_filter {
            let neighbours =
                self.index
                    .search_filter(&query.vector, k, hnsw_opt.ef_search as usize, filter);

            return Ok(SearchResult::from(neighbours));
        }

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

    fn setup(nrow: u32, dim: u32, metric_type: MetricType) -> (HnswIndex, NMatrix<f32>, Vec<u64>) {
        let index = HnswIndex::new(
            dim,
            metric_type,
            HnswIndexSetting {
                ef_construction: 20,
                max_elements: 100,
                max_nb_connection: 100,
                max_layer: 16,
            }
            .into(),
        )
        .expect("Failed to initialize index");

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

        use crate::index::option::InsertParams;
        let insert_result =
            index.insert(&InsertParams::new(&data, &labels).with(HnswParams { parallel: true }));

        assert!(insert_result.is_ok());
        assert!(index.index.get_nb_point() == 2);
    }

    #[test]
    fn test_search() {
        let (mut index, data, labels) = setup(2, 4, MetricType::L2);
        let insert_result = index.insert(&InsertParams::new(&data, &labels));
        assert!(insert_result.is_ok());

        let query = vec![1.1, 2.1, 2.9, 3.9];
        let k: usize = 2;
        let result = index.search(
            &SearchQuery::new(&query).with(&HnswSearchOption { ef_search: 20 }),
            k,
        );

        assert!(result.is_ok(), "error from search {:?}", result.err());
        let search_result = result.unwrap();
        println!("{:?}", search_result);
        assert_eq!(search_result.labels.len(), k);
        assert_eq!(search_result.labels[0], labels[0]);
    }

    #[test]
    fn test_search_with_params() {
        let (mut index, data, labels) = setup(4, 5, MetricType::L2);
        let insert_result = index.insert(&InsertParams::new(&data, &labels));
        assert!(insert_result.is_ok());

        let query = vec![1.1, 2.1, 2.9, 3.9, 5.0];
        let k: usize = 3;
        let original_result = index.search(
            &SearchQuery::new(&query.clone()).with(&HnswSearchOption { ef_search: 20 }),
            k,
        );
        assert!(
            original_result.is_ok(),
            "error from search {:?}",
            original_result.err()
        );
        assert_eq!(original_result.unwrap().labels[0], labels[0]);

        let mut filter = IdFilter::new();
        filter.add_all(&labels[1..]);
        let result = index.search(
            &SearchQuery::new(&query)
                .with(&HnswSearchOption { ef_search: 20 })
                .with(&filter),
            k,
        );

        assert!(result.is_ok(), "error from search {:?}", result.err());
        let search_result = result.unwrap();
        assert_eq!(search_result.labels.len(), k);
        assert_ne!(search_result.labels[0], labels[0]);
    }
}
