use hnsw_rs::hnsw::FilterT;
use roaring::RoaringBitmap;

pub trait IdFilter {
    fn filter(&self, id: &u32) -> bool;
}

pub struct BitMapFilter {
    bitmap: RoaringBitmap,
}

impl IdFilter for BitMapFilter {
    fn filter(&self, id: &u32) -> bool {
        self.bitmap.contains(*id)
    }
}

impl FilterT for Box<&dyn IdFilter> {
    fn hnsw_filter(&self, id: &usize) -> bool {
        self.filter(&(*id as u32))
    }
}
