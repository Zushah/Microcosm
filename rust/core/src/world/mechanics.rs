use super::{
    CONTACT_LIST_RANGE_EPSILON, CONTACT_LIST_SKIN_GAPS, GEOMETRY_TOLERANCE,
    OVERLAP_RELAXATION_PASSES, TileId, World,
};
use crate::cell::CellId;
use crate::interaction::{self, Contact};
use crate::spatial::{
    Position, REFERENCE_CELL_AREA, SpatialIndex, fused_multiply_add, minimum_image_displacement,
    toroidal_distance_squared,
};

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct BinnedCell {
    position: Position,
    radius: f32,
    slot: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct CompactCell {
    position: Position,
    radius: f32,
    tile: u32,
    lineage: u64,
    id: CellId,
}

impl World {
    pub(super) fn rebuild_spatial_index(&mut self) {
        self.contact_list_current = false;
        self.spatial_sync_pending = false;
        self.refresh_max_cell_radius();
        if !self
            .spatial_index
            .matches_dimensions(self.width, self.height)
        {
            self.spatial_index = SpatialIndex::new(self.width, self.height);
        }
        self.spatial_index
            .rebuild(self.cells.iter().map(|cell| (cell.id, cell.position)));
        self.tile_cell_counts.resize(self.tile_count, 0);
        self.tile_cell_counts.fill(0);
        for cell in self.cells.iter() {
            let Some(position) = cell.position.wrapped(self.width as f32, self.height as f32)
            else {
                continue;
            };
            let tile_index =
                position.x.floor() as usize * self.height + position.y.floor() as usize;
            self.tile_cell_counts[tile_index] = self.tile_cell_counts[tile_index].saturating_add(1);
        }
    }

    pub(super) fn center_tile_id(&self, position: Position) -> Option<TileId> {
        let position = position.wrapped(self.width as f32, self.height as f32)?;
        Some(TileId(self.index_xy(
            position.x.floor() as usize,
            position.y.floor() as usize,
        )))
    }

    pub(super) fn increment_tile_cell_count(&mut self, position: Position) {
        let tile_id = self
            .center_tile_id(position)
            .expect("active cell position must map to a field tile");
        self.tile_cell_counts[tile_id.index()] =
            self.tile_cell_counts[tile_id.index()].saturating_add(1);
    }

    pub(super) fn decrement_tile_cell_count(&mut self, position: Position) {
        let tile_id = self
            .center_tile_id(position)
            .expect("active cell position must map to a field tile");
        let count = &mut self.tile_cell_counts[tile_id.index()];
        assert!(*count > 0, "derived tile cell count underflow");
        *count -= 1;
    }

    pub(super) fn canonical_tile_index(&self, position: Position) -> usize {
        debug_assert_eq!(
            position.wrapped(self.width as f32, self.height as f32),
            Some(position)
        );
        canonical_tile(position, self.width, self.height)
    }

    pub(super) fn update_cell_position(&mut self, cell_id: CellId, position: Position) {
        debug_assert!(!self.spatial_sync_pending);
        let old_position = self.cells[cell_id].position;
        let old_tile = self.canonical_tile_index(old_position);
        let new_tile = self.canonical_tile_index(position);
        if old_tile != new_tile {
            let count = &mut self.tile_cell_counts[old_tile];
            assert!(*count > 0, "derived tile cell count underflow");
            *count -= 1;
            self.tile_cell_counts[new_tile] = self.tile_cell_counts[new_tile].saturating_add(1);
        }
        assert!(
            self.spatial_index
                .update_position(cell_id, old_position, position)
        );
        self.cells[cell_id].position = position;
    }

    pub(super) fn collect_contacts(
        &self,
        cell_id: CellId,
        position: Position,
        radius: f32,
        contacts: &mut Vec<Contact>,
    ) -> f64 {
        contacts.clear();
        let gap = self.config.surface.contact_gap;
        let (width, height) = (self.width as f32, self.height as f32);
        let cells = &self.cells;
        let wide_radius = f64::from(radius);
        let wide_gap = f64::from(gap);
        let mut push_contact =
            |other_id: CellId, slot: usize, other_position: Position, other_radius: f32| {
                let other_radius = f64::from(other_radius);
                let delta = minimum_image_displacement(position, other_position, width, height);
                let (dx, dy) = (f64::from(delta.x), f64::from(delta.y));
                let limit = wide_radius + other_radius + wide_gap;
                if dx * dx + dy * dy > limit * limit * (1.0 + 1.0e-12) {
                    return;
                }
                let distance = dx.hypot(dy);
                if distance <= limit {
                    contacts.push(Contact {
                        cell_id: other_id,
                        slot,
                        distance,
                        share: interaction::perimeter_share(
                            wide_radius,
                            other_radius,
                            distance,
                            wide_gap,
                        ),
                    });
                }
            };
        if let Some(row) = self.listed_row(cell_id) {
            let start = self.contact_row_offsets[row] as usize;
            let end = self.contact_row_offsets[row + 1] as usize;
            for &slot in &self.contact_row_neighbors[start..end] {
                let slot = slot as usize;
                if self.contact_displacements[slot] > self.contact_loose_threshold {
                    continue;
                }
                let other_id = self.contact_row_owners[slot];
                if let Some((slot, other)) = cells.find_near_slot(other_id, slot) {
                    push_contact(other_id, slot, other.position, other.radius);
                }
            }
            let extras = self.contact_loose_offsets[row] as usize
                ..self.contact_loose_offsets[row + 1] as usize;
            for &(_, other_id) in &self.contact_loose_extras[extras] {
                if let Some(slot) = cells.slot(other_id) {
                    let other = cells.at_slot(slot);
                    push_contact(other_id, slot, other.position, other.radius);
                }
            }
        } else {
            self.spatial_index.find_within(
                position,
                radius + self.max_cell_radius + gap,
                |other_id, other_position, slot_hint| {
                    if other_id != cell_id
                        && let Some((slot, other)) =
                            cells.find_near_slot(other_id, slot_hint as usize)
                    {
                        push_contact(other_id, slot, other_position, other.radius);
                    }
                    false
                },
            );
        }
        contacts.sort_unstable_by_key(|contact| contact.cell_id.index());
        interaction::normalize_shares(contacts)
    }

    pub(super) fn listed_row(&self, cell_id: CellId) -> Option<usize> {
        if !self.contact_list_current {
            return None;
        }
        let row = self.cell_phase_row?;
        if self.contact_row_owners.get(row) != Some(&cell_id)
            || self.contact_displacements[row] > self.contact_loose_threshold
        {
            return None;
        }
        Some(row)
    }

    pub(super) fn refresh_max_cell_radius(&mut self) {
        self.max_cell_radius = self
            .cells
            .iter()
            .map(|cell| cell.radius)
            .fold(0.0, f32::max);
    }

    fn refresh_compact_cells(&mut self) {
        let (width, height) = (self.width, self.height);
        let compact = &mut self.compact_cells;
        compact.clear();
        let mut max_radius = 0.0_f32;
        compact.extend(self.cells.iter().map(|cell| {
            max_radius = max_radius.max(cell.radius);
            CompactCell {
                position: cell.position,
                radius: cell.radius,
                tile: canonical_tile(cell.position, width, height) as u32,
                lineage: cell.lineage_id.raw(),
                id: cell.id,
            }
        }));
        self.max_cell_radius = max_radius;
    }

    pub(super) fn collect_contact_list(&mut self, reach: f32) -> (u64, u64) {
        self.refresh_compact_cells();
        let gap = self.config.surface.contact_gap;
        let (width, height) = (self.width, self.height);
        let (world_width, world_height) = (width as f32, height as f32);
        let compact = &self.compact_cells;
        let tile_offsets = &mut self.contact_tile_offsets;
        let tile_cells = &mut self.contact_tile_cells;
        let mut found = std::mem::take(&mut self.contact_found);
        found.clear();
        let mut contacts = 0_u64;
        let mut cross_lineage = 0_u64;
        let binned = if compact.len() < 2 { 0 } else { compact.len() };
        if binned > 0 {
            tile_offsets.clear();
            tile_offsets.resize(self.tile_count + 1, 0);
            for cell in compact {
                tile_offsets[cell.tile as usize] += 1;
            }
            let mut running = 0_u32;
            for offset in tile_offsets[..self.tile_count].iter_mut() {
                running += *offset;
                *offset = running;
            }
            tile_offsets[self.tile_count] = running;
            tile_cells.clear();
            tile_cells.resize(compact.len(), BinnedCell::default());
            for (slot, cell) in compact.iter().enumerate() {
                let tile = cell.tile as usize;
                tile_offsets[tile] -= 1;
                tile_cells[tile_offsets[tile] as usize] = BinnedCell {
                    position: cell.position,
                    radius: cell.radius,
                    slot: slot as u32,
                };
            }
        }
        for (slot, cell) in compact.iter().enumerate().take(binned) {
            let (position, radius) = (cell.position, cell.radius);
            let range = 2.0 * radius + reach + CONTACT_LIST_RANGE_EPSILON;
            let (first_column, columns) = tile_span(position.x, range, width);
            let (first_row, rows) = tile_span(position.y, range, height);
            let row = wrap_tile(first_row, height);
            let unwrapped = rows.min(height - row);
            let spans = [(row, unwrapped), (0, rows - unwrapped)];
            for column_offset in 0..columns {
                let column = wrap_tile(first_column + column_offset as isize, width);
                let base = column * height;
                for (span_row, span_rows) in spans {
                    if span_rows == 0 {
                        continue;
                    }
                    let start = tile_offsets[base + span_row] as usize;
                    let end = tile_offsets[base + span_row + span_rows] as usize;
                    for other in &tile_cells[start..end] {
                        let other_slot = other.slot as usize;
                        if other.radius > radius || (other.radius == radius && other_slot <= slot) {
                            continue;
                        }
                        let distance_squared = toroidal_distance_squared(
                            position,
                            other.position,
                            world_width,
                            world_height,
                        );
                        let reach_separation = radius + other.radius + reach;
                        if distance_squared <= reach_separation * reach_separation {
                            found.push((slot as u32, other.slot));
                            let separation = radius + other.radius + gap;
                            if distance_squared <= separation * separation {
                                contacts += 1;
                                if compact[other_slot].lineage != cell.lineage {
                                    cross_lineage += 1;
                                }
                            }
                        }
                    }
                }
            }
        }

        let count = compact.len();
        self.contact_row_owners.clear();
        self.contact_row_owners
            .extend(compact.iter().map(|cell| cell.id));
        let offsets = &mut self.contact_row_offsets;
        offsets.clear();
        offsets.resize(count + 1, 0);
        for &(a, b) in &found {
            offsets[a as usize] += 1;
            offsets[b as usize] += 1;
        }
        let mut total = 0_u32;
        for offset in offsets.iter_mut() {
            let degree = *offset;
            *offset = total;
            total += degree;
        }
        let neighbors = &mut self.contact_row_neighbors;
        neighbors.clear();
        neighbors.resize(total as usize, 0);
        for &(a, b) in &found {
            neighbors[offsets[a as usize] as usize] = b;
            offsets[a as usize] += 1;
            neighbors[offsets[b as usize] as usize] = a;
            offsets[b as usize] += 1;
        }
        for row in (1..=count).rev() {
            offsets[row] = offsets[row - 1];
        }
        offsets[0] = 0;
        let row_slots = &mut self.contact_row_scratch;
        self.contact_pairs.clear();
        for slot in 0..count {
            row_slots.clear();
            let row = offsets[slot] as usize..offsets[slot + 1] as usize;
            row_slots.extend(
                neighbors[row]
                    .iter()
                    .copied()
                    .filter(|other| *other as usize > slot),
            );
            row_slots.sort_unstable();
            self.contact_pairs
                .extend(row_slots.iter().map(|other| (slot as u32, *other)));
        }
        self.contact_found = found;
        self.contact_displacements.clear();
        self.contact_displacements.resize(count, 0.0);
        self.contact_loose_extras.clear();
        self.contact_loose_threshold = f32::INFINITY;
        self.contact_list_current = true;
        (contacts, cross_lineage)
    }

    pub(super) fn add_overlapping_pairs_of_loose_cells(&mut self, threshold: f32) {
        if !self
            .contact_displacements
            .iter()
            .any(|displacement| *displacement > threshold)
        {
            return;
        }
        self.sync_spatial_index_from_compact();
        let compact = &self.compact_cells;
        let pairs = &mut self.contact_pairs;
        for (slot, cell) in compact.iter().enumerate() {
            if self.contact_displacements[slot] <= threshold {
                continue;
            }
            self.spatial_index.find_within_hinted(
                cell.position,
                cell.radius + self.max_cell_radius,
                |_, distance_squared, other_slot| {
                    let other_slot = other_slot as usize;
                    if other_slot == slot {
                        return false;
                    }
                    let separation = cell.radius + compact[other_slot].radius;
                    if distance_squared <= separation * separation {
                        let pair = (slot.min(other_slot) as u32, slot.max(other_slot) as u32);
                        if let Err(position) = pairs.binary_search(&pair) {
                            pairs.insert(position, pair);
                        }
                    }
                    false
                },
            );
        }
    }

    pub(super) fn collect_loose_extras(&mut self, threshold: f32, reach: f32) {
        let compact = &self.compact_cells;
        let displacements = &self.contact_displacements;
        let extras = &mut self.contact_loose_extras;
        extras.clear();
        for (slot, cell) in compact.iter().enumerate() {
            if displacements[slot] <= threshold {
                continue;
            }
            self.spatial_index.find_within_hinted(
                cell.position,
                cell.radius + self.max_cell_radius + reach,
                |other_id, distance_squared, other_slot| {
                    if other_id == cell.id {
                        return false;
                    }
                    let other_slot = other_slot as usize;
                    if displacements[other_slot] > threshold {
                        return false;
                    }
                    let separation = cell.radius + compact[other_slot].radius + reach;
                    if distance_squared <= separation * separation {
                        extras.push((other_slot as u32, cell.id));
                    }
                    false
                },
            );
        }
        extras.sort_unstable_by_key(|(row, cell_id)| (*row, cell_id.index()));
        let offsets = &mut self.contact_loose_offsets;
        offsets.clear();
        offsets.resize(compact.len() + 1, 0);
        for (row, _) in extras.iter() {
            offsets[*row as usize + 1] += 1;
        }
        let mut running = 0_u32;
        for offset in offsets.iter_mut() {
            running += *offset;
            *offset = running;
        }
        self.contact_loose_threshold = threshold;
    }

    #[cfg(test)]
    pub(super) fn sync_spatial_index(&mut self) {
        if !self.spatial_sync_pending {
            return;
        }
        self.refresh_compact_cells();
        self.sync_spatial_index_from_compact();
    }

    fn sync_spatial_index_from_compact(&mut self) {
        if !self.spatial_sync_pending {
            return;
        }
        let compact = &self.compact_cells;
        self.spatial_index.rebuild_canonical(
            compact.len(),
            |slot| (compact[slot].id, compact[slot].position),
            true,
        );
        self.spatial_index
            .write_bucket_counts(&mut self.tile_cell_counts);
        self.spatial_sync_pending = false;
    }

    #[cfg(test)]
    pub(super) fn drift_cells(&mut self) {
        self.drift_cells_unsynced();
        self.sync_spatial_index();
    }

    pub(super) fn drift_cells_unsynced(&mut self) {
        let diffusivity = f64::from(self.config.cell_motion.diffusivity);
        if diffusivity.is_nan() || diffusivity <= 0.0 {
            return;
        }
        self.spatial_sync_pending = true;
        let reference_radius = (f64::from(REFERENCE_CELL_AREA) / std::f64::consts::PI).sqrt();
        let (width, height) = (self.width as f32, self.height as f32);
        let dt_seconds = self.config.dt_seconds;
        for slot in 0..self.cells.len() {
            let cell = self.cells.at_slot(slot);
            let position = cell.position;
            let cell_diffusivity = diffusivity * reference_radius / f64::from(cell.radius);
            let sigma = (2.0 * cell_diffusivity * dt_seconds).sqrt();
            let dx = self.rng.gaussian(0.0, 1.0) * sigma;
            let dy = self.rng.gaussian(0.0, 1.0) * sigma;
            let Some(moved) = Position::new(position.x + dx as f32, position.y + dy as f32)
                .wrapped(width, height)
            else {
                continue;
            };
            self.cells.at_slot_mut(slot).position = moved;
        }
    }

    pub(super) fn resolve_overlaps(&mut self) {
        let gap = self.config.surface.contact_gap;
        let skin = gap * CONTACT_LIST_SKIN_GAPS;
        let (contacts, cross_lineage) = self.collect_contact_list(gap + skin);
        self.contact_pair_count = contacts;
        self.cross_lineage_contact_pair_count = cross_lineage;
        if self.cells.len() >= 2 {
            for pass in 0..OVERLAP_RELAXATION_PASSES {
                if pass > 0 {
                    self.add_overlapping_pairs_of_loose_cells(0.5 * (gap + skin));
                }
                if !self.overlap_pass() {
                    break;
                }
            }
        }
        self.sync_spatial_index_from_compact();
        self.collect_loose_extras(0.25 * skin, gap + skin);
    }

    pub(super) fn overlap_pass(&mut self) -> bool {
        let pairs = std::mem::take(&mut self.contact_pairs);
        self.operation_counters.spatial_candidate_checks += pairs.len() as u64;
        self.operation_counters.overlap_candidates += pairs.len() as u64;
        self.mechanics_corrections.clear();
        self.mechanics_corrections
            .resize(self.cells.len(), Position::default());
        let (width, height) = (self.width as f32, self.height as f32);
        let compact = &self.compact_cells;
        let corrections = &mut self.mechanics_corrections;
        let mut correction_count = 0_u64;
        for &(slot_a, slot_b) in &pairs {
            let (slot_a, slot_b) = (slot_a as usize, slot_b as usize);
            let a = &compact[slot_a];
            let b = &compact[slot_b];
            let displacement = minimum_image_displacement(a.position, b.position, width, height);
            let distance_squared = fused_multiply_add(
                displacement.x,
                displacement.x,
                displacement.y * displacement.y,
            );
            let required_distance = a.radius + b.radius;
            if distance_squared > required_distance * required_distance * (1.0 + 1.0e-5) {
                continue;
            }
            let (direction_x, direction_y, distance) =
                if distance_squared <= GEOMETRY_TOLERANCE * GEOMETRY_TOLERANCE {
                    let directions = [
                        (1.0, 0.0),
                        (
                            std::f32::consts::FRAC_1_SQRT_2,
                            std::f32::consts::FRAC_1_SQRT_2,
                        ),
                        (0.0, 1.0),
                        (
                            -std::f32::consts::FRAC_1_SQRT_2,
                            std::f32::consts::FRAC_1_SQRT_2,
                        ),
                        (-1.0, 0.0),
                        (
                            -std::f32::consts::FRAC_1_SQRT_2,
                            -std::f32::consts::FRAC_1_SQRT_2,
                        ),
                        (0.0, -1.0),
                        (
                            std::f32::consts::FRAC_1_SQRT_2,
                            -std::f32::consts::FRAC_1_SQRT_2,
                        ),
                    ];
                    let direction = directions[(a.id.index().wrapping_mul(31)
                        ^ b.id.index().wrapping_mul(17))
                        % directions.len()];
                    (direction.0, direction.1, 0.0)
                } else {
                    let distance = distance_squared.sqrt();
                    (
                        displacement.x / distance,
                        displacement.y / distance,
                        distance,
                    )
                };
            let penetration = required_distance - distance;
            if penetration <= GEOMETRY_TOLERANCE {
                continue;
            }
            let half = penetration * 0.5;
            corrections[slot_a].x -= direction_x * half;
            corrections[slot_a].y -= direction_y * half;
            corrections[slot_b].x += direction_x * half;
            corrections[slot_b].y += direction_y * half;
            correction_count += 1;
        }
        self.operation_counters.overlap_corrections += correction_count;
        self.contact_pairs = pairs;
        if correction_count == 0 {
            return false;
        }
        self.spatial_sync_pending = true;
        for slot in 0..self.cells.len() {
            let correction = self.mechanics_corrections[slot];
            if correction.x == 0.0 && correction.y == 0.0 {
                continue;
            }
            self.contact_displacements[slot] += correction.x.hypot(correction.y);
            let current = self.compact_cells[slot].position;
            let Some(position) = Position::new(current.x + correction.x, current.y + correction.y)
                .wrapped(width, height)
            else {
                continue;
            };
            self.cells.at_slot_mut(slot).position = position;
            self.compact_cells[slot].position = position;
        }
        true
    }
}

#[inline]
fn canonical_tile(position: Position, width: usize, height: usize) -> usize {
    (position.x as usize).min(width - 1) * height + (position.y as usize).min(height - 1)
}

#[inline]
fn tile_span(center: f32, range: f32, extent: usize) -> (isize, usize) {
    let floor = |value: f32| {
        let truncated = value as isize;
        if (truncated as f32) > value {
            truncated - 1
        } else {
            truncated
        }
    };
    let first = floor(center - range);
    let count = (floor(center + range) - first + 1) as usize;
    if count >= extent {
        (0, extent)
    } else {
        (first, count)
    }
}

#[inline]
fn wrap_tile(index: isize, extent: usize) -> usize {
    let extent = extent as isize;
    if index < 0 {
        (index + extent) as usize
    } else if index >= extent {
        (index - extent) as usize
    } else {
        index as usize
    }
}
