use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::cell::CellId;

pub const DEFAULT_CELL_RADIUS: f32 = 0.45;

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Position {
    pub x: f32,
    pub y: f32,
}

impl Position {
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }

    pub fn wrapped(self, width: f32, height: f32) -> Option<Self> {
        if !self.is_finite()
            || !width.is_finite()
            || !height.is_finite()
            || width <= 0.0
            || height <= 0.0
        {
            return None;
        }
        Some(Self {
            x: self.x.rem_euclid(width),
            y: self.y.rem_euclid(height),
        })
    }
}

pub fn minimum_image_delta(delta: f32, extent: f32) -> f32 {
    debug_assert!(delta.is_finite());
    debug_assert!(extent.is_finite() && extent > 0.0);
    let mut wrapped = delta.rem_euclid(extent);
    if wrapped > extent * 0.5 {
        wrapped -= extent;
    }
    wrapped
}

pub fn minimum_image_displacement(
    from: Position,
    to: Position,
    width: f32,
    height: f32,
) -> Position {
    Position {
        x: minimum_image_delta(to.x - from.x, width),
        y: minimum_image_delta(to.y - from.y, height),
    }
}

pub fn toroidal_distance_squared(a: Position, b: Position, width: f32, height: f32) -> f32 {
    let delta = minimum_image_displacement(a, b, width, height);
    delta.x.mul_add(delta.x, delta.y * delta.y)
}

pub fn toroidal_distance(a: Position, b: Position, width: f32, height: f32) -> f32 {
    toroidal_distance_squared(a, b, width, height).sqrt()
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BilinearSample {
    pub x: usize,
    pub y: usize,
    pub weight: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BilinearStencil {
    pub samples: [BilinearSample; 4],
}

impl BilinearStencil {
    pub fn new(position: Position, width: usize, height: usize) -> Option<Self> {
        let wrapped = position.wrapped(width as f32, height as f32)?;
        let x0 = wrapped.x.floor() as usize;
        let y0 = wrapped.y.floor() as usize;
        let x1 = (x0 + 1) % width;
        let y1 = (y0 + 1) % height;
        let fx = wrapped.x - x0 as f32;
        let fy = wrapped.y - y0 as f32;
        let one_minus_fx = 1.0 - fx;
        let one_minus_fy = 1.0 - fy;
        Some(Self {
            samples: [
                BilinearSample {
                    x: x0,
                    y: y0,
                    weight: one_minus_fx * one_minus_fy,
                },
                BilinearSample {
                    x: x1,
                    y: y0,
                    weight: fx * one_minus_fy,
                },
                BilinearSample {
                    x: x0,
                    y: y1,
                    weight: one_minus_fx * fy,
                },
                BilinearSample {
                    x: x1,
                    y: y1,
                    weight: fx * fy,
                },
            ],
        })
    }

    pub fn weight_sum(self) -> f32 {
        self.samples.iter().map(|sample| sample.weight).sum()
    }
}

#[derive(Clone, Debug, Default)]
pub struct SpatialIndex {
    width: f32,
    height: f32,
    columns: usize,
    rows: usize,
    bucket_width: f32,
    bucket_height: f32,
    buckets: Vec<Vec<CellId>>,
    positions: Vec<Option<Position>>,
    active_ids: Vec<CellId>,
}

impl SpatialIndex {
    pub fn new(width: usize, height: usize) -> Self {
        let columns = width.max(1);
        let rows = height.max(1);
        let width = width.max(1) as f32;
        let height = height.max(1) as f32;
        Self {
            width,
            height,
            columns,
            rows,
            bucket_width: width / columns as f32,
            bucket_height: height / rows as f32,
            buckets: vec![Vec::new(); columns.saturating_mul(rows)],
            positions: Vec::new(),
            active_ids: Vec::new(),
        }
    }

    pub fn dimensions(&self) -> (f32, f32) {
        (self.width, self.height)
    }

    pub fn len(&self) -> usize {
        self.active_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.active_ids.is_empty()
    }

    pub fn active_ids(&self) -> &[CellId] {
        &self.active_ids
    }

    pub fn matches_dimensions(&self, width: usize, height: usize) -> bool {
        self.columns == width.max(1) && self.rows == height.max(1)
    }

    pub fn position(&self, cell_id: CellId) -> Option<Position> {
        self.positions.get(cell_id.index()).copied().flatten()
    }

    pub fn rebuild<I>(&mut self, entries: I)
    where
        I: IntoIterator<Item = (CellId, Position)>,
    {
        self.clear_entries();

        for (cell_id, position) in entries {
            let Some(position) = position.wrapped(self.width, self.height) else {
                continue;
            };
            if self.positions.len() <= cell_id.index() {
                self.positions.resize(cell_id.index() + 1, None);
            }
            if self.positions[cell_id.index()].is_some() {
                continue;
            }
            self.positions[cell_id.index()] = Some(position);
            let bucket = self.bucket_for(position);
            self.buckets[bucket].push(cell_id);
            self.active_ids.push(cell_id);
        }

        self.active_ids
            .sort_unstable_by_key(|cell_id| cell_id.index());
        for bucket in &mut self.buckets {
            if bucket.len() > 1 {
                bucket.sort_unstable_by_key(|cell_id| cell_id.index());
            }
        }
    }

    pub fn insert(&mut self, cell_id: CellId, position: Position) -> bool {
        let Some(position) = position.wrapped(self.width, self.height) else {
            return false;
        };
        if self.positions.len() <= cell_id.index() {
            self.positions.resize(cell_id.index() + 1, None);
        }
        if self.positions[cell_id.index()].is_some() {
            return false;
        }

        let bucket_index = self.bucket_for(position);
        let bucket_position = self.buckets[bucket_index]
            .binary_search_by_key(&cell_id.index(), |candidate| candidate.index())
            .unwrap_or_else(|index| index);
        self.buckets[bucket_index].insert(bucket_position, cell_id);
        let active_position = self
            .active_ids
            .binary_search_by_key(&cell_id.index(), |candidate| candidate.index())
            .unwrap_or_else(|index| index);
        self.active_ids.insert(active_position, cell_id);
        self.positions[cell_id.index()] = Some(position);
        true
    }

    pub fn remove(&mut self, cell_id: CellId) -> bool {
        let Some(position) = self.positions.get(cell_id.index()).copied().flatten() else {
            return false;
        };
        let bucket_index = self.bucket_for(position);
        let Ok(bucket_position) = self.buckets[bucket_index]
            .binary_search_by_key(&cell_id.index(), |candidate| candidate.index())
        else {
            return false;
        };
        let Ok(active_position) = self
            .active_ids
            .binary_search_by_key(&cell_id.index(), |candidate| candidate.index())
        else {
            return false;
        };

        self.positions[cell_id.index()] = None;
        self.buckets[bucket_index].remove(bucket_position);
        self.active_ids.remove(active_position);
        true
    }

    pub fn update_position(&mut self, cell_id: CellId, position: Position) -> bool {
        let Some(position) = position.wrapped(self.width, self.height) else {
            return false;
        };
        let Some(old_position) = self.position(cell_id) else {
            return false;
        };
        let old_bucket = self.bucket_for(old_position);
        let new_bucket = self.bucket_for(position);
        if old_bucket != new_bucket {
            let Ok(index) = self.buckets[old_bucket]
                .binary_search_by_key(&cell_id.index(), |candidate| candidate.index())
            else {
                return false;
            };
            self.buckets[old_bucket].remove(index);
            let new_index = self.buckets[new_bucket]
                .binary_search_by_key(&cell_id.index(), |candidate| candidate.index())
                .unwrap_or_else(|index| index);
            self.buckets[new_bucket].insert(new_index, cell_id);
        }
        self.positions[cell_id.index()] = Some(position);
        true
    }

    fn clear_entries(&mut self) {
        for cell_id in self.active_ids.iter().copied() {
            if let Some(position) = self.positions.get(cell_id.index()).copied().flatten() {
                let bucket = self.bucket_for(position);
                self.buckets[bucket].clear();
            }
        }
        self.positions.clear();
        self.active_ids.clear();
    }

    pub fn query_radius(&self, position: Position, radius: f32) -> Vec<CellId> {
        if radius < 0.0 || !radius.is_finite() {
            return Vec::new();
        }
        let Some(position) = position.wrapped(self.width, self.height) else {
            return Vec::new();
        };
        let center_column = (position.x / self.bucket_width).floor() as isize;
        let center_row = (position.y / self.bucket_height).floor() as isize;
        let column_radius = (radius / self.bucket_width).ceil() as isize + 1;
        let row_radius = (radius / self.bucket_height).ceil() as isize + 1;
        let mut bucket_indices = BTreeSet::new();
        for dx in -column_radius..=column_radius {
            for dy in -row_radius..=row_radius {
                let column = (center_column + dx).rem_euclid(self.columns as isize) as usize;
                let row = (center_row + dy).rem_euclid(self.rows as isize) as usize;
                bucket_indices.insert(self.bucket_index(column, row));
            }
        }

        let radius_squared = radius * radius;
        let mut result = Vec::new();
        for bucket_index in bucket_indices {
            for cell_id in self.buckets[bucket_index].iter().copied() {
                let Some(other) = self.position(cell_id) else {
                    continue;
                };
                if toroidal_distance_squared(position, other, self.width, self.height)
                    <= radius_squared
                {
                    result.push(cell_id);
                }
            }
        }
        result.sort_unstable_by_key(|cell_id| cell_id.index());
        result.dedup();
        result
    }

    pub fn unique_pairs_within(&self, radius: f32) -> Vec<(CellId, CellId)> {
        let mut pairs = Vec::new();
        for cell_a in self.active_ids.iter().copied() {
            let Some(position) = self.position(cell_a) else {
                continue;
            };
            for cell_b in self.query_radius(position, radius) {
                if cell_b.index() > cell_a.index() {
                    pairs.push((cell_a, cell_b));
                }
            }
        }
        pairs
    }

    fn bucket_for(&self, position: Position) -> usize {
        let column = ((position.x / self.bucket_width).floor() as usize).min(self.columns - 1);
        let row = ((position.y / self.bucket_height).floor() as usize).min(self.rows - 1);
        self.bucket_index(column, row)
    }

    fn bucket_index(&self, column: usize, row: usize) -> usize {
        column * self.rows + row
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(values: &[usize]) -> Vec<CellId> {
        values.iter().copied().map(CellId).collect()
    }

    #[test]
    fn canonical_wrap_handles_negative_and_overflow_coordinates() {
        assert_eq!(
            Position::new(-0.25, 12.5).wrapped(10.0, 8.0),
            Some(Position::new(9.75, 4.5))
        );
        assert_eq!(
            Position::new(20.0, -16.0).wrapped(10.0, 8.0),
            Some(Position::new(0.0, 0.0))
        );
        assert!(Position::new(f32::NAN, 0.0).wrapped(10.0, 8.0).is_none());
    }

    #[test]
    fn minimum_image_distance_crosses_both_seams() {
        let a = Position::new(0.1, 0.2);
        let b = Position::new(9.9, 7.8);
        let delta = minimum_image_displacement(a, b, 10.0, 8.0);
        assert!((delta.x + 0.2).abs() < 1.0e-5);
        assert!((delta.y + 0.4).abs() < 1.0e-5);
        assert!((toroidal_distance(a, b, 10.0, 8.0) - 0.2_f32.hypot(0.4)).abs() < 1.0e-5);
    }

    #[test]
    fn bilinear_weights_sum_to_one_and_integer_positions_collapse() {
        let fractional = BilinearStencil::new(Position::new(2.25, 3.75), 5, 6).unwrap();
        assert!((fractional.weight_sum() - 1.0).abs() < 1.0e-6);

        let integer = BilinearStencil::new(Position::new(2.0, 3.0), 5, 6).unwrap();
        assert_eq!(
            integer.samples[0],
            BilinearSample {
                x: 2,
                y: 3,
                weight: 1.0
            }
        );
        assert!(
            integer.samples[1..]
                .iter()
                .all(|sample| sample.weight == 0.0)
        );
    }

    #[test]
    fn bilinear_stencil_wraps_support_across_seams() {
        let stencil = BilinearStencil::new(Position::new(3.75, -0.25), 4, 3).unwrap();
        assert_eq!((stencil.samples[0].x, stencil.samples[0].y), (3, 2));
        assert_eq!((stencil.samples[1].x, stencil.samples[1].y), (0, 2));
        assert_eq!((stencil.samples[2].x, stencil.samples[2].y), (3, 0));
        assert_eq!((stencil.samples[3].x, stencil.samples[3].y), (0, 0));
        assert!((stencil.weight_sum() - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn radius_query_is_toroidal_deduplicated_and_sorted() {
        let mut index = SpatialIndex::new(10, 8);
        index.rebuild([
            (CellId(7), Position::new(9.8, 0.1)),
            (CellId(2), Position::new(0.2, 0.1)),
            (CellId(5), Position::new(4.0, 4.0)),
        ]);
        assert_eq!(
            index.query_radius(Position::new(0.0, 0.1), 0.3),
            ids(&[2, 7])
        );
    }

    #[test]
    fn tiny_world_queries_and_pairs_have_no_duplicates() {
        let mut index = SpatialIndex::new(1, 1);
        index.rebuild([
            (CellId(3), Position::new(0.1, 0.1)),
            (CellId(1), Position::new(0.7, 0.7)),
            (CellId(2), Position::new(0.4, 0.4)),
        ]);
        assert_eq!(
            index.query_radius(Position::new(0.5, 0.5), 1.0),
            ids(&[1, 2, 3])
        );
        assert_eq!(
            index.unique_pairs_within(1.0),
            vec![
                (CellId(1), CellId(2)),
                (CellId(1), CellId(3)),
                (CellId(2), CellId(3))
            ]
        );
    }

    #[test]
    fn rebuild_and_queries_are_deterministic() {
        let entries = [
            (CellId(8), Position::new(1.0, 1.0)),
            (CellId(1), Position::new(1.5, 1.5)),
            (CellId(4), Position::new(2.0, 2.0)),
        ];
        let mut forward = SpatialIndex::new(4, 4);
        forward.rebuild(entries);
        let mut reverse = SpatialIndex::new(4, 4);
        reverse.rebuild(entries.into_iter().rev());
        assert_eq!(forward.active_ids(), reverse.active_ids());
        assert_eq!(
            forward.query_radius(Position::new(1.5, 1.5), 2.0),
            reverse.query_radius(Position::new(1.5, 1.5), 2.0)
        );
    }

    #[test]
    fn rebuild_ignores_duplicate_ids_without_misplacing_them() {
        let first = Position::new(1.0, 1.0);
        let mut index = SpatialIndex::new(8, 6);
        index.rebuild([(CellId(3), first), (CellId(3), Position::new(6.0, 4.0))]);

        assert_eq!(index.len(), 1);
        assert_eq!(index.active_ids(), &[CellId(3)]);
        assert_eq!(index.position(CellId(3)), Some(first));
        assert_eq!(index.query_radius(first, 0.1), vec![CellId(3)]);
        assert!(index.query_radius(Position::new(6.0, 4.0), 0.1).is_empty());
    }

    #[test]
    fn incremental_insert_update_and_remove_preserve_sorted_queries() {
        let mut index = SpatialIndex::new(10, 8);
        assert!(index.matches_dimensions(10, 8));
        assert!(!index.matches_dimensions(8, 10));

        assert!(index.insert(CellId(7), Position::new(9.8, 0.1)));
        assert!(index.insert(CellId(2), Position::new(0.2, 0.1)));
        assert!(index.insert(CellId(5), Position::new(4.0, 4.0)));
        assert!(!index.insert(CellId(5), Position::new(6.0, 6.0)));
        assert_eq!(index.active_ids(), &ids(&[2, 5, 7]));
        assert_eq!(
            index.query_radius(Position::new(0.0, 0.1), 0.3),
            ids(&[2, 7])
        );

        assert!(index.update_position(CellId(5), Position::new(-0.1, 0.1)));
        assert_eq!(index.position(CellId(5)), Some(Position::new(9.9, 0.1)));
        assert_eq!(
            index.query_radius(Position::new(0.0, 0.1), 0.3),
            ids(&[2, 5, 7])
        );
        assert!(!index.update_position(CellId(9), Position::new(1.0, 1.0)));

        assert!(index.remove(CellId(2)));
        assert!(!index.remove(CellId(2)));
        assert_eq!(index.active_ids(), &ids(&[5, 7]));
        assert_eq!(
            index.query_radius(Position::new(0.0, 0.1), 0.3),
            ids(&[5, 7])
        );
    }

    #[test]
    fn rebuild_reuses_allocated_bucket_and_entry_storage() {
        let entries = [
            (CellId(0), Position::new(1.1, 1.1)),
            (CellId(1), Position::new(1.2, 1.2)),
            (CellId(2), Position::new(2.1, 2.1)),
        ];
        let mut index = SpatialIndex::new(320, 240);
        index.rebuild(entries);

        let buckets_ptr = index.buckets.as_ptr();
        let positions_ptr = index.positions.as_ptr();
        let active_ids_ptr = index.active_ids.as_ptr();
        let occupied_bucket = index.bucket_for(entries[0].1);
        let occupied_bucket_ptr = index.buckets[occupied_bucket].as_ptr();

        index.rebuild(entries.into_iter().rev());

        assert_eq!(index.buckets.as_ptr(), buckets_ptr);
        assert_eq!(index.positions.as_ptr(), positions_ptr);
        assert_eq!(index.active_ids.as_ptr(), active_ids_ptr);
        assert_eq!(index.buckets[occupied_bucket].as_ptr(), occupied_bucket_ptr);
        assert_eq!(index.active_ids(), &ids(&[0, 1, 2]));
    }

    #[test]
    fn spatial_index_matches_brute_force_fixture() {
        let entries = (0..32)
            .map(|id| {
                let x = ((id * 17) % 11) as f32 + 0.125;
                let y = ((id * 13) % 7) as f32 + 0.375;
                (CellId(id), Position::new(x, y))
            })
            .collect::<Vec<_>>();
        let mut index = SpatialIndex::new(11, 7);
        index.rebuild(entries.iter().copied());
        let center = Position::new(10.8, 6.9);
        let radius = 2.25;
        let mut brute_force = entries
            .iter()
            .filter_map(|(cell_id, position)| {
                (toroidal_distance(center, *position, 11.0, 7.0) <= radius).then_some(*cell_id)
            })
            .collect::<Vec<_>>();
        brute_force.sort_unstable_by_key(|cell_id| cell_id.index());
        assert_eq!(index.query_radius(center, radius), brute_force);
    }
}
