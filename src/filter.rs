use faiss::index::Idx;
use faiss::selector::IdSelector;
use hnsw_rs::hnsw::FilterT;
use hnsw_rs::prelude::DataId;
use roaring::RoaringBitmap;
use std::fmt::Debug;

#[derive(Debug, Clone)]
pub struct IdFilter(RoaringBitmap);

impl IdFilter {
    pub fn new() -> Self {
        Self(RoaringBitmap::new())
    }

    pub fn add(&mut self, id: u64) {
        self.0.insert(id as u32);
    }

    pub fn add_all(&mut self, ids: &[u64]) {
        for id in ids {
            self.add(*id);
        }
    }

    pub fn filter(&self, id: &u64) -> bool {
        self.0.contains(*id as u32)
    }

    pub fn into_selector(&self) -> IdSelector {
        IdSelector::batch(
            &self
                .0
                .iter()
                .map(|e| Idx::new(e as u64))
                .collect::<Vec<Idx>>(),
        )
        .unwrap()
    }
}

impl FilterT for IdFilter {
    fn hnsw_filter(&self, id: &DataId) -> bool {
        self.filter(&(*id as u64))
    }
}
