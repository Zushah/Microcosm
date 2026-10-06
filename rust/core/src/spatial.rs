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
            x: wrap_coordinate(self.x, width),
            y: wrap_coordinate(self.y, height),
        })
    }
}

pub fn wrap_coordinate(value: f32, extent: f32) -> f32 {
    let wrapped = rem_euclid(value, extent);
    if wrapped >= extent { 0.0 } else { wrapped }
}

#[inline]
fn rem_euclid(value: f32, extent: f32) -> f32 {
    if value >= 0.0 && value < extent {
        value
    } else if value < 0.0 && value > -extent {
        value + extent
    } else {
        value.rem_euclid(extent)
    }
}

pub fn minimum_image_delta(delta: f32, extent: f32) -> f32 {
    debug_assert!(delta.is_finite());
    debug_assert!(extent.is_finite() && extent > 0.0);
    let mut wrapped = rem_euclid(delta, extent);
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
    #[inline]
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

#[derive(Clone, Copy, Debug, PartialEq)]
struct SpatialEntry {
    id: CellId,
    position: Position,
}

#[derive(Clone, Debug, Default)]
pub struct SpatialIndex {
    width: f32,
    height: f32,
    columns: usize,
    rows: usize,
    bucket_width: f32,
    bucket_height: f32,
    buckets: Vec<Vec<SpatialEntry>>,
    len: usize,
}

const BUCKET_RANGE_EPSILON: f32 = 1.0e-3;

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
            len: 0,
        }
    }

    pub fn dimensions(&self) -> (f32, f32) {
        (self.width, self.height)
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn matches_dimensions(&self, width: usize, height: usize) -> bool {
        self.columns == width.max(1) && self.rows == height.max(1)
    }

    pub fn contains_at(&self, cell_id: CellId, position: Position) -> bool {
        let Some(position) = position.wrapped(self.width, self.height) else {
            return false;
        };
        let bucket = &self.buckets[self.bucket_for(position)];
        bucket
            .binary_search_by_key(&cell_id.index(), |entry| entry.id.index())
            .is_ok_and(|index| bucket[index].position == position)
    }

    pub fn rebuild<I>(&mut self, entries: I)
    where
        I: IntoIterator<Item = (CellId, Position)>,
    {
        for bucket in &mut self.buckets {
            bucket.clear();
        }
        self.len = 0;
        let mut entries = entries.into_iter().collect::<Vec<_>>();
        entries.sort_by_key(|(cell_id, _)| cell_id.index());
        entries.dedup_by_key(|(cell_id, _)| *cell_id);
        for (cell_id, position) in entries {
            self.insert(cell_id, position);
        }
    }

    #[cfg(test)]
    fn sorted_ids(&self) -> Vec<CellId> {
        let mut ids = self
            .buckets
            .iter()
            .flatten()
            .map(|entry| entry.id)
            .collect::<Vec<_>>();
        ids.sort_unstable_by_key(|cell_id| cell_id.index());
        ids
    }

    pub fn insert(&mut self, cell_id: CellId, position: Position) -> bool {
        let Some(position) = position.wrapped(self.width, self.height) else {
            return false;
        };
        let bucket_index = self.bucket_for(position);
        let bucket = &mut self.buckets[bucket_index];
        match bucket.binary_search_by_key(&cell_id.index(), |entry| entry.id.index()) {
            Ok(_) => false,
            Err(index) => {
                bucket.insert(
                    index,
                    SpatialEntry {
                        id: cell_id,
                        position,
                    },
                );
                self.len += 1;
                true
            }
        }
    }

    pub fn remove(&mut self, cell_id: CellId, position: Position) -> bool {
        let Some(position) = position.wrapped(self.width, self.height) else {
            return false;
        };
        let bucket_index = self.bucket_for(position);
        let bucket = &mut self.buckets[bucket_index];
        let Ok(index) = bucket.binary_search_by_key(&cell_id.index(), |entry| entry.id.index())
        else {
            return false;
        };
        bucket.remove(index);
        self.len -= 1;
        true
    }

    pub fn update_position(
        &mut self,
        cell_id: CellId,
        old_position: Position,
        position: Position,
    ) -> bool {
        let (Some(old_position), Some(position)) = (
            old_position.wrapped(self.width, self.height),
            position.wrapped(self.width, self.height),
        ) else {
            return false;
        };
        let old_bucket = self.bucket_for(old_position);
        let new_bucket = self.bucket_for(position);
        let Ok(index) = self.buckets[old_bucket]
            .binary_search_by_key(&cell_id.index(), |entry| entry.id.index())
        else {
            return false;
        };
        if old_bucket == new_bucket {
            self.buckets[old_bucket][index].position = position;
            return true;
        }
        self.buckets[old_bucket].remove(index);
        let bucket = &mut self.buckets[new_bucket];
        let new_index = bucket
            .binary_search_by_key(&cell_id.index(), |entry| entry.id.index())
            .unwrap_or_else(|index| index);
        bucket.insert(
            new_index,
            SpatialEntry {
                id: cell_id,
                position,
            },
        );
        true
    }

    pub fn find_within<F>(&self, position: Position, radius: f32, visit: F) -> Option<CellId>
    where
        F: FnMut(CellId, Position) -> bool,
    {
        if radius < 0.0 || !radius.is_finite() {
            return None;
        }
        let position = position.wrapped(self.width, self.height)?;
        self.find_within_canonical(position, radius, visit)
    }

    fn find_within_canonical<F>(
        &self,
        position: Position,
        radius: f32,
        mut visit: F,
    ) -> Option<CellId>
    where
        F: FnMut(CellId, Position) -> bool,
    {
        let radius_squared = radius * radius;
        let (column_start, column_count) =
            bucket_span(position.x, radius, self.bucket_width, self.columns);
        let (row_start, row_count) = bucket_span(position.y, radius, self.bucket_height, self.rows);
        for column_offset in 0..column_count {
            let column = wrap_bucket(column_start + column_offset as isize, self.columns);
            for row_offset in 0..row_count {
                let row = wrap_bucket(row_start + row_offset as isize, self.rows);
                for entry in &self.buckets[self.bucket_index(column, row)] {
                    if toroidal_distance_squared(position, entry.position, self.width, self.height)
                        <= radius_squared
                        && visit(entry.id, entry.position)
                    {
                        return Some(entry.id);
                    }
                }
            }
        }
        None
    }

    pub fn query_radius(&self, position: Position, radius: f32) -> Vec<CellId> {
        let mut result = Vec::new();
        self.find_within(position, radius, |cell_id, _| {
            result.push(cell_id);
            false
        });
        result.sort_unstable_by_key(|cell_id| cell_id.index());
        result
    }

    pub fn collect_pairs_within<I>(&self, radius: f32, cells: I, pairs: &mut Vec<(CellId, CellId)>)
    where
        I: IntoIterator<Item = (CellId, Position)>,
    {
        pairs.clear();
        if radius < 0.0 || !radius.is_finite() {
            return;
        }
        for (cell_a, position) in cells {
            let Some(position) = position.wrapped(self.width, self.height) else {
                continue;
            };
            self.find_within_canonical(position, radius, |cell_b, _| {
                if cell_b.index() > cell_a.index() {
                    pairs.push((cell_a, cell_b));
                }
                false
            });
        }
        pairs.sort_unstable_by_key(|(a, b)| (a.index(), b.index()));
    }

    pub fn unique_pairs_within(&self, radius: f32) -> Vec<(CellId, CellId)> {
        let entries = self
            .buckets
            .iter()
            .flatten()
            .map(|entry| (entry.id, entry.position))
            .collect::<Vec<_>>();
        let mut pairs = Vec::new();
        self.collect_pairs_within(radius, entries, &mut pairs);
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

#[inline]
fn wrap_bucket(index: isize, buckets: usize) -> usize {
    let buckets = buckets as isize;
    if index < 0 {
        (index + buckets) as usize
    } else if index >= buckets {
        (index - buckets) as usize
    } else {
        index as usize
    }
}

fn bucket_span(center: f32, radius: f32, bucket_size: f32, buckets: usize) -> (isize, usize) {
    let first = ((center - radius - BUCKET_RANGE_EPSILON) / bucket_size).floor() as isize;
    let last = ((center + radius + BUCKET_RANGE_EPSILON) / bucket_size).floor() as isize;
    let count = (last - first + 1) as usize;
    if count >= buckets {
        (0, buckets)
    } else {
        (first, count)
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
    fn wrapping_never_returns_the_extent_for_tiny_negative_coordinates() {
        for value in [
            -1.0e-5_f32,
            -1.1920929e-7,
            -f32::MIN_POSITIVE,
            1.99999 - 2.0,
        ] {
            assert_eq!(value.rem_euclid(320.0), 320.0);
            let wrapped = Position::new(value, value).wrapped(320.0, 240.0).unwrap();
            assert_eq!(wrapped.x, 0.0);
            assert!(wrapped.y < 240.0);
            for extent in [320.0, 240.0, 64.0, 7.0] {
                let coordinate = wrap_coordinate(value, extent);
                assert!((0.0..extent).contains(&coordinate));
            }
            let stencil = BilinearStencil::new(Position::new(value, value), 320, 240).unwrap();
            assert!(
                stencil
                    .samples
                    .iter()
                    .all(|sample| sample.x < 320 && sample.y < 240)
            );
        }
        assert_eq!(
            wrap_coordinate(-2.0e-5, 320.0),
            (-2.0e-5_f32).rem_euclid(320.0)
        );
        assert_eq!(wrap_coordinate(5.5, 320.0), 5.5);
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
        assert_eq!(forward.sorted_ids(), reverse.sorted_ids());
        assert_eq!(
            forward.query_radius(Position::new(1.5, 1.5), 2.0),
            reverse.query_radius(Position::new(1.5, 1.5), 2.0)
        );
        assert_eq!(
            forward.unique_pairs_within(1.0),
            reverse.unique_pairs_within(1.0)
        );
    }

    #[test]
    fn rebuild_ignores_duplicate_ids_without_misplacing_them() {
        let first = Position::new(1.0, 1.0);
        let mut index = SpatialIndex::new(8, 6);
        index.rebuild([(CellId(3), first), (CellId(3), Position::new(6.0, 4.0))]);

        assert_eq!(index.len(), 1);
        assert!(index.contains_at(CellId(3), first));
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
        assert!(!index.insert(CellId(5), Position::new(4.0, 4.0)));
        assert_eq!(index.sorted_ids(), ids(&[2, 5, 7]));
        assert_eq!(
            index.query_radius(Position::new(0.0, 0.1), 0.3),
            ids(&[2, 7])
        );

        assert!(index.update_position(
            CellId(5),
            Position::new(4.0, 4.0),
            Position::new(-0.1, 0.1)
        ));
        assert!(index.contains_at(CellId(5), Position::new(9.9, 0.1)));
        assert!(!index.contains_at(CellId(5), Position::new(4.0, 4.0)));
        assert_eq!(
            index.query_radius(Position::new(0.0, 0.1), 0.3),
            ids(&[2, 5, 7])
        );
        assert!(index.update_position(
            CellId(5),
            Position::new(9.9, 0.1),
            Position::new(9.95, 0.15)
        ));
        assert!(index.contains_at(CellId(5), Position::new(9.95, 0.15)));
        assert!(!index.update_position(
            CellId(9),
            Position::new(1.0, 1.0),
            Position::new(2.0, 2.0)
        ));

        assert!(index.remove(CellId(2), Position::new(0.2, 0.1)));
        assert!(!index.remove(CellId(2), Position::new(0.2, 0.1)));
        assert_eq!(index.sorted_ids(), ids(&[5, 7]));
        assert_eq!(index.len(), 2);
        assert_eq!(
            index.query_radius(Position::new(0.0, 0.1), 0.3),
            ids(&[5, 7])
        );
    }

    #[test]
    fn rebuild_reuses_allocated_bucket_storage() {
        let entries = [
            (CellId(0), Position::new(1.1, 1.1)),
            (CellId(1), Position::new(1.2, 1.2)),
            (CellId(2), Position::new(2.1, 2.1)),
        ];
        let mut index = SpatialIndex::new(320, 240);
        index.rebuild(entries);

        let buckets_ptr = index.buckets.as_ptr();
        let occupied_bucket = index.bucket_for(entries[0].1);
        let occupied_bucket_ptr = index.buckets[occupied_bucket].as_ptr();

        index.rebuild(entries.into_iter().rev());

        assert_eq!(index.buckets.as_ptr(), buckets_ptr);
        assert_eq!(index.buckets[occupied_bucket].as_ptr(), occupied_bucket_ptr);
        assert_eq!(index.sorted_ids(), ids(&[0, 1, 2]));
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

    #[test]
    fn queries_and_pairs_match_brute_force_for_random_clouds() {
        let mut rng = crate::rng::Rng::from_seed_str("spatial-brute-force");
        for (width, height, count) in [(40, 30, 400), (7, 5, 60), (2, 3, 12), (1, 1, 5)] {
            let (w, h) = (width as f32, height as f32);
            let entries = (0..count)
                .map(|id| {
                    (
                        CellId(id * 3 + 1),
                        Position::new(rng.range(0.0, w), rng.range(0.0, h)),
                    )
                })
                .collect::<Vec<_>>();
            let mut index = SpatialIndex::new(width, height);
            for (cell_id, position) in &entries {
                assert!(index.insert(*cell_id, *position));
            }
            for radius in [0.0, 0.45, 0.9, std::f32::consts::SQRT_2 + 1.0e-5, 2.5, 7.0] {
                for _ in 0..25 {
                    let center = Position::new(rng.range(-w, 2.0 * w), rng.range(-h, 2.0 * h));
                    let mut expected = entries
                        .iter()
                        .filter(|(_, position)| {
                            toroidal_distance_squared(
                                center.wrapped(w, h).unwrap(),
                                *position,
                                w,
                                h,
                            ) <= radius * radius
                        })
                        .map(|(cell_id, _)| *cell_id)
                        .collect::<Vec<_>>();
                    expected.sort_unstable_by_key(|cell_id| cell_id.index());
                    assert_eq!(index.query_radius(center, radius), expected);
                }
                let mut expected_pairs = Vec::new();
                for (left, (a, a_position)) in entries.iter().enumerate() {
                    for (b, b_position) in &entries[left + 1..] {
                        if toroidal_distance_squared(*a_position, *b_position, w, h)
                            <= radius * radius
                        {
                            expected_pairs.push(if a.index() < b.index() {
                                (*a, *b)
                            } else {
                                (*b, *a)
                            });
                        }
                    }
                }
                expected_pairs.sort_unstable_by_key(|(a, b)| (a.index(), b.index()));
                let mut pairs = Vec::new();
                index.collect_pairs_within(radius, entries.iter().copied(), &mut pairs);
                assert_eq!(pairs, expected_pairs, "{width}x{height} radius {radius}");
                assert_eq!(index.unique_pairs_within(radius), expected_pairs);
                let capacity = pairs.capacity();
                let pointer = pairs.as_ptr();
                index.collect_pairs_within(radius, entries.iter().copied(), &mut pairs);
                assert_eq!(pairs.capacity(), capacity);
                assert_eq!(pairs.as_ptr(), pointer);
            }
        }
    }
}
