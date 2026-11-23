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
