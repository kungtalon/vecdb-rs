use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

type AttrLookupTable = HashMap<String, HashMap<i64, RoaringBitmap>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FilterOp {
    Equal,
    NotEqual,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntFilterInput {
    pub field: String,
    pub op: FilterOp,
    pub target: i64,
}

pub struct IntFilterIndex {
    pub int_field_filters: AttrLookupTable,
}

unsafe impl Send for IntFilterIndex {}
unsafe impl Sync for IntFilterIndex {}

impl IntFilterIndex {
    pub fn new() -> Self {
        Self {
            int_field_filters: HashMap::new(),
        }
    }

    pub fn upsert(&mut self, field: &str, value: i64, id: u64) {
        let filter_map_by_value = self.int_field_filters.entry(field.to_string()).or_default();

        let bitmap = filter_map_by_value.entry(value).or_default();

        bitmap.insert(id as u32);
    }

    pub fn apply(&self, input: &IntFilterInput, bitmap: &RoaringBitmap) -> RoaringBitmap {
        if input.op == FilterOp::Equal {
            let cur_bitmap_opt = self
                .int_field_filters
                .get(&input.field)
                .and_then(|map: &HashMap<i64, RoaringBitmap>| map.get(&input.target));

            if cur_bitmap_opt.is_none() {
                return bitmap.clone();
            }

            let cur_bitmap = cur_bitmap_opt.unwrap();

            return bitmap | cur_bitmap;
        }

        if input.op == FilterOp::NotEqual {
            let value_to_bitmap = self.int_field_filters.get(&input.field);

            if value_to_bitmap.is_none() {
                return bitmap.clone();
            }

            let mut res_bitmap = bitmap.clone();

            for (value, cur_bitmap) in value_to_bitmap.unwrap() {
                if value == &input.target {
                    continue;
                }

                res_bitmap |= cur_bitmap;
            }

            return res_bitmap;
        }

        bitmap.clone()
    }
}
