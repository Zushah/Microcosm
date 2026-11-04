use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::bio::EnzymeTerms;
use crate::cell::{Cell, CellFluxLog, CellId, CellStore};
use crate::chem::{ELEMENT_COUNT, ELEMENT_ORDER, ElementAmounts, ElementAmountsError};
use crate::config::{Config, ConfigError};
use crate::environment;
use crate::fingerprint::Fingerprint;
use crate::genome::{Genome, LineageId};
use crate::interaction::Contact;
use crate::rng::Rng;
use crate::spatial::{
    BilinearStencil, Position, REFERENCE_CELL_AREA, REFERENCE_CELL_RADIUS, SpatialIndex,
    toroidal_distance_squared,
};
use crate::stats::{
    EnergyLedger, EnvalLedger, InteractionCounters, OperationCounters, ReactionCounters,
    StepProfile,
};

mod cells;
mod census;
mod fields;
mod inspect;
mod invariants;
mod mechanics;
#[cfg(test)]
mod tests;

use mechanics::{BinnedCell, CompactCell};

pub use inspect::{
    CellDetailInspection, CellInspection, ContactInspection, EnzymeDetailInspection,
    FluxLogInspection, GenomeDetailInspection, GenomeEditResult, LineageListInspection,
    LineageSummaryInspection, TileInspection,
};
pub use invariants::InvariantError;

const LOCAL_ENVAL_RADIUS: usize = 2;
const LOCAL_ENVAL_WINDOW_DIAMETER: usize = LOCAL_ENVAL_RADIUS * 2 + 1;
const LOCAL_ENVAL_WINDOW_AREA: usize = LOCAL_ENVAL_WINDOW_DIAMETER * LOCAL_ENVAL_WINDOW_DIAMETER;
const MAX_TRANSPORT_FRACTION: f64 = 0.25;
const FOUNDER_PLACEMENT_ATTEMPTS: usize = 64;
pub const MIN_DAUGHTER_BODY: f64 = 0.05;
const OVERLAP_RELAXATION_PASSES: usize = 2;
const CONTACT_LIST_SKIN_GAPS: f32 = 1.0;
const CONTACT_LIST_RANGE_EPSILON: f32 = 1.0e-3;
const GEOMETRY_TOLERANCE: f32 = 1.0e-5;
pub const LYSIS_ATTRIBUTION_TICKS: u64 = 100;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TileId(pub usize);

impl TileId {
    pub const fn index(self) -> usize {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LineageCounters {
    pub births: u64,
    pub deaths: u64,
    pub population: u64,
}

type FounderMaterial = (ElementAmounts, ElementAmounts, [(usize, ElementAmounts); 4]);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct World {
    config: Config,
    rng: Rng,
    width: usize,
    height: usize,
    tile_count: usize,
    tick_count: u64,
    sim_time_seconds: f64,
    enval_sum: f64,
    avg_enval: f32,
    enval: Vec<f32>,
    #[serde(skip, default)]
    enval_next: Vec<f32>,
    element_fields: Vec<ElementAmounts>,
    #[serde(skip, default)]
    element_fields_next: Vec<ElementAmounts>,
    enval_landscape: Vec<f32>,
    cells: CellStore,
    #[serde(skip, default)]
    watched_cell_id: Option<CellId>,
    #[serde(skip, default)]
    flux_log: CellFluxLog,
    #[serde(skip, default)]
    cell_phase_scratch: Vec<CellId>,
    #[serde(skip, default)]
    mechanics_corrections: Vec<Position>,
    #[serde(skip, default)]
    contact_pairs: Vec<(u32, u32)>,
    #[serde(skip, default)]
    spatial_sync_pending: bool,
    #[serde(skip, default)]
    contact_found: Vec<(u32, u32)>,
    #[serde(skip, default)]
    enzyme_terms: Vec<EnzymeTerms>,
    #[serde(skip, default)]
    compact_cells: Vec<CompactCell>,
    #[serde(skip, default)]
    contact_tile_offsets: Vec<u32>,
    #[serde(skip, default)]
    contact_tile_cells: Vec<BinnedCell>,
    #[serde(skip, default)]
    contact_row_scratch: Vec<u32>,
    #[serde(skip, default)]
    contact_row_owners: Vec<CellId>,
    #[serde(skip, default)]
    contact_row_offsets: Vec<u32>,
    #[serde(skip, default)]
    contact_row_neighbors: Vec<u32>,
    #[serde(skip, default)]
    contact_displacements: Vec<f32>,
    #[serde(skip, default)]
    contact_loose_extras: Vec<(u32, CellId)>,
    #[serde(skip, default)]
    contact_loose_offsets: Vec<u32>,
    #[serde(skip, default)]
    contact_loose_threshold: f32,
    #[serde(skip, default)]
    contact_list_current: bool,
    #[serde(skip, default)]
    cell_phase_row: Option<usize>,
    #[serde(skip, default)]
    contact_scratch: Vec<Contact>,
    #[serde(skip, default)]
    tile_cell_counts: Vec<u32>,
    #[serde(skip, default)]
    enval_row_sums: Vec<f32>,
    #[serde(skip, default)]
    element_row_sums: Vec<[f32; ELEMENT_COUNT]>,
    #[serde(skip, default)]
    enval_smoothed: Vec<f32>,
    #[serde(skip, default)]
    enval_smoothing_scratch: Vec<f64>,
    #[serde(skip, default)]
    enval_smoothed_current: bool,
    #[serde(skip, default)]
    pending_enval_uptake: f64,
    #[serde(skip, default)]
    pending_enval_emission: f64,
    #[serde(skip, default)]
    pending_secretion: ElementAmounts,
    #[serde(skip, default)]
    pending_secretion_mask: u8,
    lineage_counters: BTreeMap<LineageId, LineageCounters>,
    birth_count: u64,
    death_count: u64,
    energy_death_count: u64,
    lysis_count: u64,
    reaction_counters: ReactionCounters,
    operation_counters: OperationCounters,
    interaction_counters: InteractionCounters,
    contact_pair_count: u64,
    cross_lineage_contact_pair_count: u64,
    energy_ledger: EnergyLedger,
    enval_ledger: EnvalLedger,
    #[serde(skip, default)]
    spatial_index: SpatialIndex,
    #[serde(skip, default)]
    max_cell_radius: f32,
}

impl World {
    pub fn new(config: Config) -> Result<Self, WorldError> {
        config.validate()?;
        let tile_count = config.tile_count().ok_or(ConfigError::TileCountOverflow)?;
        let mut rng = Rng::from_seed_str(&config.seed);
        let element_fields = environment::initial_element_fields(
            &config.element_fields,
            config.width,
            config.height,
            &mut rng,
        );
        let enval_landscape = environment::enval_landscape(
            &config.enval_landscape,
            config.width,
            config.height,
            &mut rng,
        );
        let enval = enval_landscape.clone();
        let enval_sum = enval.iter().map(|value| f64::from(*value)).sum::<f64>();

        let world = Self {
            width: config.width,
            height: config.height,
            tile_count,
            tick_count: 0,
            sim_time_seconds: 0.0,
            enval_sum,
            avg_enval: (enval_sum / tile_count as f64) as f32,
            enval_next: vec![0.0; tile_count],
            enval,
            element_fields_next: vec![ElementAmounts::ZERO; tile_count],
            element_fields,
            enval_landscape,
            cells: CellStore::default(),
            watched_cell_id: None,
            flux_log: CellFluxLog::default(),
            cell_phase_scratch: Vec::new(),
            mechanics_corrections: Vec::new(),
            contact_pairs: Vec::new(),
            spatial_sync_pending: false,
            contact_found: Vec::new(),
            enzyme_terms: Vec::new(),
            compact_cells: Vec::new(),
            contact_tile_offsets: Vec::new(),
            contact_tile_cells: Vec::new(),
            contact_row_scratch: Vec::new(),
            contact_row_owners: Vec::new(),
            contact_row_offsets: Vec::new(),
            contact_row_neighbors: Vec::new(),
            contact_displacements: Vec::new(),
            contact_loose_extras: Vec::new(),
            contact_loose_offsets: Vec::new(),
            contact_loose_threshold: 0.0,
            contact_list_current: false,
            cell_phase_row: None,
            contact_scratch: Vec::new(),
            tile_cell_counts: vec![0; tile_count],
            enval_row_sums: Vec::new(),
            element_row_sums: Vec::new(),
            enval_smoothed: Vec::new(),
            enval_smoothing_scratch: Vec::new(),
            enval_smoothed_current: false,
            pending_enval_uptake: 0.0,
            pending_enval_emission: 0.0,
            pending_secretion: ElementAmounts::ZERO,
            pending_secretion_mask: 0,
            lineage_counters: BTreeMap::new(),
            birth_count: 0,
            death_count: 0,
            energy_death_count: 0,
            lysis_count: 0,
            reaction_counters: ReactionCounters::default(),
            operation_counters: OperationCounters::default(),
            interaction_counters: InteractionCounters::default(),
            contact_pair_count: 0,
            cross_lineage_contact_pair_count: 0,
            energy_ledger: EnergyLedger::default(),
            enval_ledger: EnvalLedger {
                initial_total: enval_sum,
                ..EnvalLedger::default()
            },
            spatial_index: SpatialIndex::new(config.width, config.height),
            max_cell_radius: 0.0,
            rng,
            config,
        };
        Ok(world)
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn rng(&self) -> &Rng {
        &self.rng
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    pub fn tile_count(&self) -> usize {
        self.tile_count
    }

    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    pub fn tick_count(&self) -> u64 {
        self.tick_count
    }

    pub fn sim_time_seconds(&self) -> f64 {
        self.sim_time_seconds
    }

    pub fn enval_landscape(&self) -> &[f32] {
        &self.enval_landscape
    }

    pub fn average_enval(&self) -> f32 {
        self.avg_enval
    }

    pub fn operation_counters(&self) -> OperationCounters {
        self.operation_counters
    }

    pub fn reaction_counters(&self) -> ReactionCounters {
        self.reaction_counters
    }

    pub fn interaction_counters(&self) -> InteractionCounters {
        self.interaction_counters
    }

    pub fn lineage_counters(&self) -> &BTreeMap<LineageId, LineageCounters> {
        &self.lineage_counters
    }

    pub fn extant_lineage_count(&self) -> usize {
        self.lineage_counters
            .values()
            .filter(|counters| counters.population > 0)
            .count()
    }

    pub fn top_lineages(&self, limit: usize) -> Vec<(LineageId, LineageCounters)> {
        let mut entries = self
            .lineage_counters
            .iter()
            .filter(|(_, counters)| counters.population > 0)
            .map(|(lineage, counters)| (*lineage, *counters))
            .collect::<Vec<_>>();
        entries.sort_by(|a, b| {
            b.1.population
                .cmp(&a.1.population)
                .then_with(|| b.1.births.cmp(&a.1.births))
                .then_with(|| a.0.cmp(&b.0))
        });
        entries.truncate(limit);
        entries
    }

    pub fn energy_ledger(&self) -> EnergyLedger {
        self.energy_ledger
    }

    pub fn enval_ledger(&self) -> EnvalLedger {
        self.enval_ledger
    }

    pub fn total_live_cell_energy(&self) -> f64 {
        self.cells.iter().map(|cell| cell.energy).sum()
    }

    pub fn energy_ledger_residual(&self) -> f64 {
        self.total_live_cell_energy() - self.energy_ledger.expected_cell_energy()
    }

    #[cfg(test)]
    pub(crate) fn enval_ledger_residual(&self) -> f64 {
        let field_total = self
            .enval
            .iter()
            .map(|value| f64::from(*value))
            .sum::<f64>();
        field_total - self.enval_ledger.initial_total - self.enval_ledger.net_change()
    }

    pub fn set_all_live_cell_energy(&mut self, energy: f64) -> Result<(), WorldError> {
        if !energy.is_finite() || energy < 0.0 {
            return Err(WorldError::InvalidEnergyInput(energy));
        }
        for cell in self.cells.iter_mut() {
            let delta = energy - cell.energy;
            if delta >= 0.0 {
                self.energy_ledger.injected_energy += delta;
            } else {
                self.energy_ledger.extracted_energy -= delta;
            }
            cell.energy = energy;
        }
        Ok(())
    }

    pub fn tile_id(&self, x: usize, y: usize) -> Option<TileId> {
        if x < self.width && y < self.height {
            Some(TileId(self.index_xy(x, y)))
        } else {
            None
        }
    }

    pub fn wrapped_tile_id(&self, x: isize, y: isize) -> TileId {
        let wrapped_x = x.rem_euclid(self.width as isize) as usize;
        let wrapped_y = y.rem_euclid(self.height as isize) as usize;
        TileId(self.index_xy(wrapped_x, wrapped_y))
    }

    pub fn tile_xy(&self, tile_id: TileId) -> Option<(usize, usize)> {
        if tile_id.index() >= self.tile_count {
            None
        } else {
            Some((tile_id.index() / self.height, tile_id.index() % self.height))
        }
    }

    pub fn cell(&self, cell_id: CellId) -> Option<&Cell> {
        self.cells.get(cell_id)
    }

    pub fn pick_cell(&self, position: Position) -> Option<CellId> {
        let position = position.wrapped(self.width as f32, self.height as f32)?;
        let mut nearest: Option<(CellId, f32)> = None;
        self.spatial_index.find_within(
            position,
            self.max_cell_radius + GEOMETRY_TOLERANCE,
            |id, center, _| {
                let cell = &self.cells[id];
                let distance = toroidal_distance_squared(
                    position,
                    center,
                    self.width as f32,
                    self.height as f32,
                );
                if distance <= (cell.radius + GEOMETRY_TOLERANCE).powi(2)
                    && nearest.is_none_or(|(other, best)| {
                        distance < best || (distance == best && id.index() < other.index())
                    })
                {
                    nearest = Some((id, distance));
                }
                false
            },
        );
        nearest.map(|(id, _)| id)
    }

    pub fn spawn_founder_cells(&mut self, count: usize) -> Result<usize, WorldError> {
        let mut spawned = 0;
        for _ in 0..count {
            if self.spawn_random_cell()?.is_some() {
                spawned += 1;
            }
        }
        Ok(spawned)
    }

    pub fn spawn_random_cell(&mut self) -> Result<Option<CellId>, WorldError> {
        let Some(position) = self.random_non_overlapping_position(FOUNDER_PLACEMENT_ATTEMPTS)
        else {
            return Ok(None);
        };
        let genome = Genome::random_founder(&mut self.rng, self.avg_enval);
        self.spawn_cell_with_genome_at_position(position, genome)
            .map(Some)
    }

    #[cfg(test)]
    pub(crate) fn spawn_cell_with_genome_at(
        &mut self,
        tile_id: TileId,
        genome: Genome,
    ) -> Result<CellId, WorldError> {
        if tile_id.index() >= self.tile_count {
            return Err(WorldError::InvalidTile(tile_id));
        }
        let (x, y) = self
            .tile_xy(tile_id)
            .ok_or(WorldError::InvalidTile(tile_id))?;
        self.spawn_cell_with_genome_at_position(Position::new(x as f32, y as f32), genome)
    }

    pub fn spawn_cell_with_genome_at_position(
        &mut self,
        position: Position,
        genome: Genome,
    ) -> Result<CellId, WorldError> {
        let cell_id = self.insert_cell(position, genome)?;
        self.energy_ledger.founder_energy += self.cells[cell_id].energy;
        Ok(cell_id)
    }

    fn founder_material(&self, position: Position) -> Option<FounderMaterial> {
        let stencil = BilinearStencil::new(position, self.width, self.height)?;
        let composition = self.sample_element_fields_at(position)?;
        let total = composition.total();
        if total <= 0.0 {
            return None;
        }
        let body_total =
            f64::from(REFERENCE_CELL_AREA) * f64::from(self.config.cell_body.body_density);
        let boundary_total = std::f64::consts::TAU
            * f64::from(REFERENCE_CELL_RADIUS)
            * f64::from(self.config.cell_body.boundary_density);
        let mut body = ElementAmounts::ZERO;
        let mut boundary = ElementAmounts::ZERO;
        let mut withdrawals = [(0, ElementAmounts::ZERO); 4];
        let mut weights = [0.0_f64; 4];
        for (slot, sample) in stencil.samples.iter().enumerate() {
            withdrawals[slot].0 = self.index_xy(sample.x, sample.y);
            weights[slot] = f64::from(sample.weight);
            for earlier in 0..slot {
                if withdrawals[earlier].0 == withdrawals[slot].0 {
                    weights[earlier] += weights[slot];
                    weights[slot] = 0.0;
                    break;
                }
            }
        }
        for element in ELEMENT_ORDER {
            body[element] = (f64::from(composition[element]) / total * body_total) as f32;
            boundary[element] = (f64::from(composition[element]) / total * boundary_total) as f32;
            let mut remaining = f64::from(body[element]) + f64::from(boundary[element]);
            let mut available = [0.0_f64; 4];
            for slot in 0..4 {
                if weights[slot] > 0.0 {
                    available[slot] = f64::from(self.element_fields[withdrawals[slot].0][element]);
                }
            }
            if available.iter().sum::<f64>() < remaining {
                return None;
            }
            for _ in 0..4 {
                let weighted: f64 = (0..4).map(|slot| weights[slot] * available[slot]).sum();
                if weighted <= 0.0 {
                    break;
                }
                let requested = remaining;
                for slot in 0..4 {
                    let moved = (requested * weights[slot] * available[slot] / weighted)
                        .min(available[slot]);
                    available[slot] -= moved;
                    withdrawals[slot].1[element] += moved as f32;
                    remaining -= moved;
                }
                if remaining <= 1.0e-12 {
                    break;
                }
            }
            if remaining > 1.0e-7 {
                return None;
            }
        }
        Some((body, boundary, withdrawals))
    }

    fn insert_cell(
        &mut self,
        position: Position,
        mut genome: Genome,
    ) -> Result<CellId, WorldError> {
        let position = position
            .wrapped(self.width as f32, self.height as f32)
            .ok_or(WorldError::InvalidCellPosition(position))?;
        if !self.position_is_clear(position, REFERENCE_CELL_RADIUS, None) {
            return Err(WorldError::OverlappingCellPosition(position));
        }
        genome.enforce_enzyme_count_bounds(&mut self.rng);
        genome
            .validate()
            .map_err(|error| WorldError::GenomePatch(error.to_string()))?;
        let (body, boundary, withdrawals) = self
            .founder_material(position)
            .ok_or(WorldError::InsufficientFounderMaterial(position))?;
        for (index, amounts) in withdrawals {
            for element in ELEMENT_ORDER {
                self.element_fields[index][element] -= amounts[element];
            }
        }
        let cell = Cell::new(
            genome,
            position,
            self.sim_time_seconds,
            self.config.founder_energy,
            body,
            boundary,
            self.config.cell_body.body_density,
        );
        self.max_cell_radius = self.max_cell_radius.max(cell.radius);
        self.record_lineage_birth(cell.lineage_id);
        self.birth_count = self.birth_count.saturating_add(1);
        let cell_id = self.cells.insert(cell);
        assert!(self.spatial_index.insert(cell_id, position));
        self.increment_tile_cell_count(position);
        self.contact_list_current = false;
        Ok(cell_id)
    }

    pub fn state_fingerprint(&self) -> u64 {
        let mut fingerprint = Fingerprint::new();
        fingerprint.config(&self.config);
        self.rng.hash_state(&mut fingerprint);
        fingerprint.usize(self.width);
        fingerprint.usize(self.height);
        fingerprint.usize(self.tile_count);
        fingerprint.u64(self.tick_count);
        fingerprint.f64(self.sim_time_seconds);
        fingerprint.f64(self.enval_sum);
        fingerprint.f32(self.avg_enval);
        fingerprint.f32_slice(&self.enval);
        fingerprint.amounts_slice(&self.element_fields);
        fingerprint.f32_slice(&self.enval_landscape);
        fingerprint.usize(self.cells.issued_ids());
        fingerprint.usize(self.cells.len());
        for cell in self.cells.iter() {
            fingerprint.cell(cell);
        }
        fingerprint.usize(self.lineage_counters.len());
        for (lineage_id, counters) in &self.lineage_counters {
            let LineageCounters {
                births,
                deaths,
                population,
            } = counters;
            fingerprint.u64(lineage_id.raw());
            fingerprint.u64(*births);
            fingerprint.u64(*deaths);
            fingerprint.u64(*population);
        }
        fingerprint.u64(self.birth_count);
        fingerprint.u64(self.death_count);
        fingerprint.u64(self.energy_death_count);
        fingerprint.u64(self.lysis_count);
        fingerprint.reaction_counters(&self.reaction_counters);
        fingerprint.interaction_counters(&self.interaction_counters);
        fingerprint.u64(self.contact_pair_count);
        fingerprint.u64(self.cross_lineage_contact_pair_count);
        fingerprint.energy_ledger(&self.energy_ledger);
        fingerprint.enval_ledger(&self.enval_ledger);
        fingerprint.finish()
    }

    pub fn step(&mut self) {
        self.diffuse_element_fields();
        self.step_cells();
        self.drift_cells_unsynced();
        self.resolve_overlaps();
        self.enval_phase();
        self.advance_time();
    }

    pub fn step_many(&mut self, ticks: u32) {
        for _ in 0..ticks {
            self.step();
        }
    }

    pub fn step_profiled(&mut self) -> StepProfile {
        let counters_before = self.operation_counters;
        let total_start = Instant::now();

        let start = Instant::now();
        self.diffuse_element_fields();
        let element_field_diffusion = start.elapsed();

        let start = Instant::now();
        self.step_cells();
        let cell_step = start.elapsed();

        let start = Instant::now();
        self.drift_cells_unsynced();
        self.resolve_overlaps();
        let cell_mechanics = start.elapsed();

        let start = Instant::now();
        self.enval_phase();
        let enval_diffusion = start.elapsed();

        self.advance_time();

        StepProfile {
            element_field_diffusion,
            cell_step,
            cell_mechanics,
            enval_diffusion,
            total: total_start.elapsed(),
            counters: self.operation_counters.saturating_delta(counters_before),
        }
    }

    fn advance_time(&mut self) {
        self.tick_count = self.tick_count.wrapping_add(1);
        self.sim_time_seconds += self.config.dt_seconds;
    }

    pub fn rebuild_derived_caches(&mut self) {
        self.enval_next.resize(self.tile_count, 0.0);
        self.element_fields_next
            .resize(self.tile_count, ElementAmounts::ZERO);
        self.cell_phase_scratch.clear();
        self.mechanics_corrections.clear();
        self.contact_pairs.clear();
        self.contact_list_current = false;
        self.contact_scratch.clear();
        self.cells.rebuild_index();
        self.rebuild_spatial_index();
    }

    fn index_xy(&self, x: usize, y: usize) -> usize {
        x * self.height + y
    }

    #[cfg(test)]
    fn add_unledgered_energy(&mut self, cell_id: CellId, delta: f64) {
        self.cells[cell_id].energy += delta;
    }

    fn random_non_overlapping_position(&mut self, max_attempts: usize) -> Option<Position> {
        for _ in 0..max_attempts {
            let position = Position::new(
                self.rng.range(0.0, self.width as f32),
                self.rng.range(0.0, self.height as f32),
            );
            if self.position_is_clear(position, REFERENCE_CELL_RADIUS, None)
                && self.founder_material(position).is_some()
            {
                return Some(position);
            }
        }
        None
    }

    fn position_is_clear(
        &mut self,
        position: Position,
        radius: f32,
        ignored_cell: Option<CellId>,
    ) -> bool {
        if !position.is_finite() || !radius.is_finite() || radius <= 0.0 {
            return false;
        }
        let (width, height) = (self.width as f32, self.height as f32);
        let cells = &self.cells;
        let mut candidates = 0_u64;
        let blocker = self.spatial_index.find_within(
            position,
            radius + self.max_cell_radius,
            |other_id, other_position, _| {
                candidates += 1;
                if Some(other_id) == ignored_cell {
                    return false;
                }
                let other_radius = cells
                    .get(other_id)
                    .map_or(REFERENCE_CELL_RADIUS, |other| other.radius);
                let separation = radius + other_radius;
                toroidal_distance_squared(position, other_position, width, height)
                    < (separation - GEOMETRY_TOLERANCE).max(0.0).powi(2)
            },
        );
        self.operation_counters.spatial_candidate_checks = self
            .operation_counters
            .spatial_candidate_checks
            .saturating_add(candidates);
        blocker.is_none()
    }

    fn record_lineage_birth(&mut self, lineage_id: LineageId) {
        let counters = self.lineage_counters.entry(lineage_id).or_default();
        counters.births = counters.births.saturating_add(1);
        counters.population = counters.population.saturating_add(1);
    }

    fn record_lineage_death(&mut self, lineage_id: LineageId) {
        let counters = self.lineage_counters.entry(lineage_id).or_default();
        counters.deaths = counters.deaths.saturating_add(1);
        counters.population = counters.population.saturating_sub(1);
    }
}

#[derive(Debug)]
pub enum WorldError {
    Config(ConfigError),
    ElementAmounts(ElementAmountsError),
    InvalidTile(TileId),
    InvalidCell(CellId),
    InvalidCellPosition(Position),
    OverlappingCellPosition(Position),
    InsufficientFounderMaterial(Position),
    NonFiniteEnvalInput(f32),
    InvalidEnergyInput(f64),
    GenomePatch(String),
}

impl fmt::Display for WorldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(err) => write!(f, "invalid config: {err}"),
            Self::ElementAmounts(err) => write!(f, "invalid element amounts: {err}"),
            Self::InvalidTile(tile) => write!(f, "invalid tile id {}", tile.index()),
            Self::InvalidCell(cell) => write!(f, "invalid cell id {}", cell.index()),
            Self::InvalidCellPosition(position) => write!(
                f,
                "invalid continuous cell position ({}, {})",
                position.x, position.y
            ),
            Self::OverlappingCellPosition(position) => write!(
                f,
                "cell position ({}, {}) overlaps an active cell",
                position.x, position.y
            ),
            Self::InsufficientFounderMaterial(position) => write!(
                f,
                "insufficient founder material at ({}, {})",
                position.x, position.y
            ),
            Self::NonFiniteEnvalInput(value) => write!(f, "non-finite enval value {value}"),
            Self::InvalidEnergyInput(value) => write!(
                f,
                "invalid cell energy override: expected finite nonnegative value, got {value}"
            ),
            Self::GenomePatch(message) => write!(f, "invalid genome patch: {message}"),
        }
    }
}

impl Error for WorldError {}

impl From<ConfigError> for WorldError {
    fn from(value: ConfigError) -> Self {
        Self::Config(value)
    }
}

impl From<ElementAmountsError> for WorldError {
    fn from(value: ElementAmountsError) -> Self {
        Self::ElementAmounts(value)
    }
}
