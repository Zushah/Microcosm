use serde::{Deserialize, Serialize};

use crate::cell::CellId;

pub const REFERENCE_CELL_RADIUS: f32 = 0.45;
pub const REFERENCE_CELL_AREA: f32 =
    std::f32::consts::PI * REFERENCE_CELL_RADIUS * REFERENCE_CELL_RADIUS;

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

    #[inline]
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

#[inline]
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

#[inline]
pub fn minimum_image_delta(delta: f32, extent: f32) -> f32 {
    debug_assert!(delta.is_finite());
    debug_assert!(extent.is_finite() && extent > 0.0);
    let mut wrapped = rem_euclid(delta, extent);
    if wrapped > extent * 0.5 {
        wrapped -= extent;
    }
    wrapped
}

#[inline]
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

#[inline]
pub fn toroidal_distance_squared(a: Position, b: Position, width: f32, height: f32) -> f32 {
    let delta = minimum_image_displacement(a, b, width, height);
    fused_multiply_add(delta.x, delta.x, delta.y * delta.y)
}

#[inline]
pub(crate) fn fused_multiply_add(a: f32, b: f32, c: f32) -> f32 {
    let sum = f64::from(a) * f64::from(b) + f64::from(c);
    const F32_ROUNDING_BITS: u64 = (1 << 29) - 1;
    const HALFWAY: u64 = 1 << 28;
    const SMALLEST_NORMAL: f64 = f32::MIN_POSITIVE as f64;
    if sum.to_bits() & F32_ROUNDING_BITS == HALFWAY || sum.abs() < SMALLEST_NORMAL {
        return a.mul_add(b, c);
    }
    sum as f32
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

    #[cfg(test)]
    pub(crate) fn weight_sum(self) -> f32 {
        self.samples.iter().map(|sample| sample.weight).sum()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct SpatialEntry {
    id: CellId,
    position: Position,
    slot_hint: u32,
}

const NO_SLOT_HINT: u32 = u32::MAX;
const REMOVED_ENTRY: SpatialEntry = SpatialEntry {
    id: CellId(usize::MAX),
    position: Position::new(0.0, 0.0),
    slot_hint: NO_SLOT_HINT,
};
const BUCKET_RANGE_EPSILON: f32 = 1.0e-3;
const MAX_RECENT_ENTRIES: usize = 64;

#[derive(Clone, Debug, Default)]
pub struct SpatialIndex {
    width: f32,
    height: f32,
    columns: usize,
    rows: usize,
    starts: Vec<u32>,
    entries: Vec<SpatialEntry>,
    recent: Vec<SpatialEntry>,
    len: usize,
}

impl SpatialIndex {
    pub fn new(width: usize, height: usize) -> Self {
        let columns = width.max(1);
        let rows = height.max(1);
        Self {
            width: columns as f32,
            height: rows as f32,
            columns,
            rows,
            starts: vec![0; columns.saturating_mul(rows) + 1],
            entries: Vec::new(),
            recent: Vec::new(),
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
        let bucket = self.bucket_for(position);
        self.bucket_entries(bucket)
            .iter()
            .chain(&self.recent)
            .any(|entry| entry.id == cell_id && entry.position == position)
    }

    pub fn rebuild<I>(&mut self, entries: I)
    where
        I: IntoIterator<Item = (CellId, Position)>,
    {
        let mut entries = entries
            .into_iter()
            .filter_map(|(cell_id, position)| {
                Some((cell_id, position.wrapped(self.width, self.height)?))
            })
            .collect::<Vec<_>>();
        entries.sort_by_key(|(cell_id, _)| cell_id.index());
        entries.dedup_by_key(|(cell_id, _)| *cell_id);
        self.rebuild_canonical(entries.len(), |index| entries[index], false);
    }

    pub(crate) fn rebuild_canonical<F>(&mut self, count: usize, cell_at: F, hint_slots: bool)
    where
        F: Fn(usize) -> (CellId, Position),
    {
        let (columns, rows) = (self.columns, self.rows);
        let bucket_of = |position: Position| {
            (position.x as usize).min(columns - 1) * rows + (position.y as usize).min(rows - 1)
        };
        self.recent.clear();
        self.starts.clear();
        self.starts.resize(columns * rows + 1, 0);
        for index in 0..count {
            self.starts[bucket_of(cell_at(index).1)] += 1;
        }
        let mut running = 0_u32;
        for start in self.starts.iter_mut() {
            running += *start;
            *start = running;
        }
        self.entries.clear();
        self.entries.resize(count, REMOVED_ENTRY);
        for index in 0..count {
            let (id, position) = cell_at(index);
            let bucket = bucket_of(position);
            self.starts[bucket] -= 1;
            self.entries[self.starts[bucket] as usize] = SpatialEntry {
                id,
                position,
                slot_hint: if hint_slots {
                    index as u32
                } else {
                    NO_SLOT_HINT
                },
            };
        }
        self.len = count;
    }

    pub(crate) fn write_bucket_counts(&self, counts: &mut [u32]) {
        debug_assert!(self.recent.is_empty());
        for (count, range) in counts.iter_mut().zip(self.starts.windows(2)) {
            *count = range[1] - range[0];
        }
    }

    #[cfg(test)]
    fn sorted_ids(&self) -> Vec<CellId> {
        let mut ids = self
            .entries
            .iter()
            .chain(&self.recent)
            .filter(|entry| entry.id != REMOVED_ENTRY.id)
            .map(|entry| entry.id)
            .collect::<Vec<_>>();
        ids.sort_unstable_by_key(|cell_id| cell_id.index());
        ids
    }

    pub fn insert(&mut self, cell_id: CellId, position: Position) -> bool {
        let Some(position) = position.wrapped(self.width, self.height) else {
            return false;
        };
        let bucket = self.bucket_for(position);
        if self
            .bucket_entries(bucket)
            .iter()
            .chain(&self.recent)
            .any(|entry| entry.id == cell_id)
        {
            return false;
        }
        self.len += 1;
        self.push_recent(cell_id, position);
        true
    }

    fn push_recent(&mut self, id: CellId, position: Position) {
        self.recent.push(SpatialEntry {
            id,
            position,
            slot_hint: NO_SLOT_HINT,
        });
        if self.recent.len() > MAX_RECENT_ENTRIES {
            self.compact();
        }
    }

    fn compact(&mut self) {
        let live = self
            .entries
            .iter()
            .chain(&self.recent)
            .filter(|entry| entry.id != REMOVED_ENTRY.id)
            .map(|entry| (entry.id, entry.position))
            .collect::<Vec<_>>();
        self.rebuild_canonical(live.len(), |index| live[index], false);
    }

    fn take_entry(&mut self, cell_id: CellId, position: Position) -> bool {
        let bucket = self.bucket_for(position);
        let range = self.starts[bucket] as usize..self.starts[bucket + 1] as usize;
        if let Some(entry) = self.entries[range]
            .iter_mut()
            .find(|entry| entry.id == cell_id)
        {
            *entry = REMOVED_ENTRY;
            return true;
        }
        if let Some(index) = self.recent.iter().position(|entry| entry.id == cell_id) {
            self.recent.swap_remove(index);
            return true;
        }
        false
    }

    pub fn remove(&mut self, cell_id: CellId, position: Position) -> bool {
        let Some(position) = position.wrapped(self.width, self.height) else {
            return false;
        };
        if !self.take_entry(cell_id, position) {
            return false;
        }
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
        if !self.take_entry(cell_id, old_position) {
            return false;
        }
        self.push_recent(cell_id, position);
        true
    }

    pub fn find_within<F>(&self, position: Position, radius: f32, mut visit: F) -> Option<CellId>
    where
        F: FnMut(CellId, Position, u32) -> bool,
    {
        if radius < 0.0 || !radius.is_finite() {
            return None;
        }
        let position = position.wrapped(self.width, self.height)?;
        self.find_within_canonical(position, radius, |entry, _| {
            visit(entry.id, entry.position, entry.slot_hint)
        })
    }

    pub(crate) fn find_within_hinted<F>(
        &self,
        position: Position,
        radius: f32,
        mut visit: F,
    ) -> Option<CellId>
    where
        F: FnMut(CellId, f32, u32) -> bool,
    {
        if radius < 0.0 || !radius.is_finite() {
            return None;
        }
        let position = position.wrapped(self.width, self.height)?;
        self.find_within_canonical(position, radius, |entry, distance_squared| {
            visit(entry.id, distance_squared, entry.slot_hint)
        })
    }

    #[inline]
    fn find_within_canonical<F>(
        &self,
        position: Position,
        radius: f32,
        mut visit: F,
    ) -> Option<CellId>
    where
        F: FnMut(&SpatialEntry, f32) -> bool,
    {
        let radius_squared = radius * radius;
        let (column_start, column_count) = bucket_span(position.x, radius, self.columns);
        let (row_start, row_count) = bucket_span(position.y, radius, self.rows);
        let first_row = wrap_bucket(row_start, self.rows);
        let unwrapped_rows = row_count.min(self.rows - first_row);
        let spans = [(first_row, unwrapped_rows), (0, row_count - unwrapped_rows)];
        for column_offset in 0..column_count {
            let column = wrap_bucket(column_start + column_offset as isize, self.columns);
            let base = column * self.rows;
            for (row, rows) in spans {
                if rows == 0 {
                    continue;
                }
                let start = self.starts[base + row] as usize;
                let end = self.starts[base + row + rows] as usize;
                for entry in &self.entries[start..end] {
                    if entry.id == REMOVED_ENTRY.id {
                        continue;
                    }
                    let distance_squared = toroidal_distance_squared(
                        position,
                        entry.position,
                        self.width,
                        self.height,
                    );
                    if distance_squared <= radius_squared && visit(entry, distance_squared) {
                        return Some(entry.id);
                    }
                }
            }
        }
        for entry in &self.recent {
            let distance_squared =
                toroidal_distance_squared(position, entry.position, self.width, self.height);
            if distance_squared <= radius_squared && visit(entry, distance_squared) {
                return Some(entry.id);
            }
        }
        None
    }

    #[cfg(test)]
    pub(crate) fn query_radius(&self, position: Position, radius: f32) -> Vec<CellId> {
        let mut result = Vec::new();
        self.find_within(position, radius, |cell_id, _, _| {
            result.push(cell_id);
            false
        });
        result.sort_unstable_by_key(|cell_id| cell_id.index());
        result
    }

    #[cfg(test)]
    pub(crate) fn collect_pairs_within<I>(
        &self,
        radius: f32,
        cells: I,
        pairs: &mut Vec<(CellId, CellId)>,
    ) where
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
            self.find_within_canonical(position, radius, |entry, _| {
                if entry.id.index() > cell_a.index() {
                    pairs.push((cell_a, entry.id));
                }
                false
            });
        }
        pairs.sort_unstable_by_key(|(a, b)| (a.index(), b.index()));
    }

    #[cfg(test)]
    pub(crate) fn unique_pairs_within(&self, radius: f32) -> Vec<(CellId, CellId)> {
        let entries = self
            .entries
            .iter()
            .chain(&self.recent)
            .filter(|entry| entry.id != REMOVED_ENTRY.id)
            .map(|entry| (entry.id, entry.position))
            .collect::<Vec<_>>();
        let mut pairs = Vec::new();
        self.collect_pairs_within(radius, entries, &mut pairs);
        pairs
    }

    #[inline]
    fn bucket_for(&self, position: Position) -> usize {
        (position.x as usize).min(self.columns - 1) * self.rows
            + (position.y as usize).min(self.rows - 1)
    }

    fn bucket_entries(&self, bucket: usize) -> &[SpatialEntry] {
        &self.entries[self.starts[bucket] as usize..self.starts[bucket + 1] as usize]
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

#[inline]
fn floor_to_isize(value: f32) -> isize {
    let truncated = value as isize;
    if (truncated as f32) > value {
        truncated - 1
    } else {
        truncated
    }
}

#[inline]
fn bucket_span(center: f32, radius: f32, buckets: usize) -> (isize, usize) {
    let first = floor_to_isize(center - radius - BUCKET_RANGE_EPSILON);
    let last = floor_to_isize(center + radius + BUCKET_RANGE_EPSILON);
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

        let starts_ptr = index.starts.as_ptr();
        let entries_ptr = index.entries.as_ptr();

        index.rebuild(entries.into_iter().rev());

        assert_eq!(index.starts.as_ptr(), starts_ptr);
        assert_eq!(index.entries.as_ptr(), entries_ptr);
        assert_eq!(index.sorted_ids(), ids(&[0, 1, 2]));
    }

    #[test]
    fn fused_multiply_add_matches_the_library_bit_for_bit() {
        let mut rng = crate::rng::Rng::from_seed_str("fused-multiply-add");
        let mut values = vec![
            0.0_f32,
            -0.0,
            1.0,
            -1.0,
            0.5,
            1.5,
            f32::MIN_POSITIVE,
            1.0e-30,
            3.0e30,
        ];
        for _ in 0..40 {
            values.push(rng.range(-320.0, 320.0));
            values.push(rng.range(-1.0, 1.0) * 1.0e-3);
        }
        let check = |a: f32, b: f32, c: f32| {
            assert_eq!(
                fused_multiply_add(a, b, c).to_bits(),
                a.mul_add(b, c).to_bits(),
                "{a} * {b} + {c}"
            );
        };
        for &a in &values {
            for &b in &values {
                for &c in &values {
                    check(a, b, c);
                }
            }
        }
        for _ in 0..2_000_000 {
            let scale = 2.0_f32.powi(rng.usize(40) as i32 - 20);
            let a = rng.range(-1.0, 1.0) * scale;
            let c = rng.range(-1.0, 1.0) * scale * scale;
            check(a, a, c);
            check(a, rng.range(-1.0, 1.0), c);
        }
        for exponent in 0..24 {
            let a = 1.0 + f32::EPSILON * (1 << exponent) as f32;
            for offset in [1.0, 2.0, 0.5, -1.0, f32::EPSILON, 1.0 + f32::EPSILON] {
                check(a, a, offset);
                check(a, 1.0 / a, offset);
            }
        }
    }

    #[test]
    fn edits_between_rebuilds_match_a_brute_force_model() {
        let (width, height) = (23_usize, 17_usize);
        let mut rng = crate::rng::Rng::from_seed_str("spatial-edits");
        let mut index = SpatialIndex::new(width, height);
        let mut model: Vec<(CellId, Position)> = Vec::new();
        let mut next_id = 0;
        let random_position = |rng: &mut crate::rng::Rng| {
            Position::new(rng.range(0.0, width as f32), rng.range(0.0, height as f32))
        };
        for round in 0..4_000 {
            match rng.usize(5) {
                0 | 1 => {
                    let position = random_position(&mut rng);
                    assert!(index.insert(CellId(next_id), position));
                    assert!(!index.insert(CellId(next_id), position));
                    model.push((CellId(next_id), position));
                    next_id += 1;
                }
                2 if !model.is_empty() => {
                    let (id, position) = model.swap_remove(rng.usize(model.len()));
                    assert!(index.remove(id, position));
                    assert!(!index.remove(id, position));
                    assert!(!index.contains_at(id, position));
                }
                3 if !model.is_empty() => {
                    let entry = rng.usize(model.len());
                    let (id, old) = model[entry];
                    let position = random_position(&mut rng);
                    assert!(index.update_position(id, old, position));
                    model[entry].1 = position;
                    assert!(index.contains_at(id, position));
                }
                4 if round % 40 == 0 => {
                    let snapshot = model.clone();
                    index.rebuild_canonical(snapshot.len(), |slot| snapshot[slot], true);
                }
                _ => {}
            }
            assert_eq!(index.len(), model.len());
            let center = random_position(&mut rng);
            let radius = rng.range(0.0, 6.0);
            let mut expected = model
                .iter()
                .filter(|(_, position)| {
                    toroidal_distance_squared(center, *position, width as f32, height as f32)
                        <= radius * radius
                })
                .map(|(id, _)| *id)
                .collect::<Vec<_>>();
            expected.sort_unstable_by_key(|id| id.index());
            assert_eq!(
                index.query_radius(center, radius),
                expected,
                "round {round}"
            );
        }
    }

    #[test]
    fn hinted_rebuild_reports_slots_and_later_additions_have_no_hint() {
        let cells = [
            (CellId(9), Position::new(1.5, 1.5)),
            (CellId(4), Position::new(1.6, 1.5)),
        ];
        let mut index = SpatialIndex::new(8, 8);
        index.rebuild_canonical(cells.len(), |slot| cells[slot], true);
        assert!(index.insert(CellId(11), Position::new(1.7, 1.5)));
        let mut seen = Vec::new();
        index.find_within(Position::new(1.5, 1.5), 1.0, |id, _, hint| {
            seen.push((id.index(), hint));
            false
        });
        seen.sort_unstable();
        assert_eq!(seen, vec![(4, 1), (9, 0), (11, u32::MAX)]);
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
