use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::bio::{self, ReactionEnv};
use crate::cell::{Cell, CellFluxLog, CellId, CellStore, FluxRecord, ReactionSite};
use crate::chem::{ELEMENT_COUNT, ELEMENT_ORDER, Element, ElementAmounts, ElementAmountsError};
use crate::config::{Config, ConfigError};
use crate::environment;
use crate::genome::{
    Enzyme, Genome, GenomePatch, LineageId, MAX_CELL_ENZYMES, MIN_CELL_ENZYMES,
};
use crate::interaction::{self, Contact};
use crate::render_buffers::{RenderBuffers, RenderVisualState};
use crate::rng::Rng;
use crate::spatial::{
    BilinearStencil, Position, REFERENCE_CELL_AREA, REFERENCE_CELL_RADIUS, SpatialIndex,
    minimum_image_displacement, toroidal_distance_squared, wrap_coordinate,
};
use crate::stats::{
    ENZYME_COUNT_HISTOGRAM_LEN, EXPOSED_CELL_THRESHOLD, EnergyLedger, EnvalLedger,
    InteractionCounters, MIXED_POLARITY_THRESHOLD, OperationCounters,
    ReactionCounters, StepProfile, WorldStats, renewable_coverage,
};

const LOCAL_ENVAL_RADIUS: usize = 2;
const LOCAL_ENVAL_WINDOW_DIAMETER: usize = LOCAL_ENVAL_RADIUS * 2 + 1;
const LOCAL_ENVAL_WINDOW_AREA: usize = LOCAL_ENVAL_WINDOW_DIAMETER * LOCAL_ENVAL_WINDOW_DIAMETER;
const MOORE_WITH_CENTER_DX: [isize; 9] = [-1, -1, -1, 0, 0, 1, 1, 1, 0];
const MOORE_WITH_CENTER_DY: [isize; 9] = [-1, 0, 1, -1, 1, -1, 0, 1, 0];
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeighborIndices {
    pub left: TileId,
    pub right: TileId,
    pub up: TileId,
    pub down: TileId,
    pub up_left: TileId,
    pub up_right: TileId,
    pub down_left: TileId,
    pub down_right: TileId,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LineageCounters {
    pub births: u64,
    pub deaths: u64,
    pub population: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TileInspection {
    pub tile_id: TileId,
    pub x: usize,
    pub y: usize,
    pub enval: f32,
    pub enval_landscape_target: f32,
    pub cell_center_count: u32,
    pub element_concentrations: [f32; ELEMENT_COUNT],
    pub total_element_concentration: f32,
    pub mass_density: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CellInspection {
    pub cell_id: CellId,
    pub x: f32,
    pub y: f32,
    pub radius: f32,
    pub body_total: f64,
    pub boundary_total: f64,
    pub area: f64,
    pub strain: f64,
    pub membrane_polarity: f64,
    pub energy: f64,
    pub lineage_id: LineageId,
    pub enzyme_count: usize,
    pub total_internal_elements: f32,
    pub age_seconds: f64,
    pub mean_enval_optimum: f32,
    pub enval_optimum_spread: f32,
    pub local_enval_average: f32,
    pub field_share: f64,
    pub contacts: Vec<ContactInspection>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ContactInspection {
    pub cell_id: usize,
    pub lineage_id: LineageId,
    pub distance: f64,
    pub share: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EnzymeDetailInspection {
    pub index: usize,
    pub reactants: [f32; ELEMENT_COUNT],
    pub products: [f32; ELEMENT_COUNT],
    pub rate: f32,
    pub half_saturation: f32,
    pub energy_harvest_fraction: f32,
    pub secretion_fraction: f32,
    pub body_fraction: f32,
    pub boundary_fraction: f32,
    pub enval_optimum: f32,
    pub enval_sigma: f32,
    pub enval_throughput: f32,
    pub enval_energy_fraction: f32,
    pub enval_release_fraction: f32,
    pub enval_pump: f32,
    pub exposure: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GenomeDetailInspection {
    pub mutation_rate: f32,
    pub maintenance_cost_per_area: f64,
    pub lineage_id: LineageId,
    pub enzyme_count: usize,
    pub min_cell_enzymes: usize,
    pub max_cell_enzymes: usize,
    pub enzymes: Vec<EnzymeDetailInspection>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FluxLogInspection {
    pub available: bool,
    pub reason: &'static str,
    pub limit: usize,
    pub truncated: bool,
    pub flux_count: usize,
    pub returned_count: usize,
    pub order: &'static str,
    pub fluxes: Vec<FluxRecord>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GenomeEditResult {
    pub target: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cell_id: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub center_x: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub center_y: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub brush_width: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub brush_height: Option<usize>,
    pub visited_tile_count: usize,
    pub patched_cell_count: usize,
    pub changed_fields: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CellDetailInspection {
    pub cell: CellInspection,
    pub maintenance_cost_per_area: f64,
    pub catalyst_upkeep_per_sec: f64,
    pub genome: GenomeDetailInspection,
    pub body: [f32; ELEMENT_COUNT],
    pub boundary: [f32; ELEMENT_COUNT],
    pub internal_elements: [f32; ELEMENT_COUNT],
    pub total_internal_elements: f32,
    pub recent_fluxes: FluxLogInspection,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LineageSummaryInspection {
    pub lineage_id: LineageId,
    pub population: u64,
    pub births: u64,
    pub deaths: u64,
    pub extinct: bool,
    pub share: f64,
    pub average_energy: f64,
    pub average_enzyme_count: f64,
    pub mean_exposure: f64,
    pub mean_enval_optimum: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LineageListInspection {
    pub extant_lineage_count: usize,
    pub total_lineage_records: usize,
    pub limit: usize,
    pub truncated: bool,
    pub lineages: Vec<LineageSummaryInspection>,
}

#[derive(Clone, Copy, Debug, Default)]
struct BinnedCell {
    position: Position,
    radius: f32,
    slot: u32,
}

#[derive(Clone, Copy, Debug)]
struct TileStencil {
    tiles: [usize; 4],
    weights: [f32; 4],
}

#[derive(Clone, Copy, Debug)]
struct CellReactionContext {
    cell_id: CellId,
    slot: usize,
    position: Position,
    stencil: TileStencil,
    local_enval: f32,
    cell_area: f32,
}

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
    enval_next: Vec<f32>,
    element_fields: Vec<ElementAmounts>,
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
    indexed_positions: Vec<Position>,
    #[serde(skip, default)]
    spatial_sync_pending: bool,
    #[serde(skip, default)]
    contact_found: Vec<(u32, u32)>,
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
    neighbors: Vec<NeighborIndices>,
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
        let neighbors = build_neighbors(config.width, config.height);

        let world = Self {
            width: config.width,
            height: config.height,
            tile_count,
            tick_count: 0,
            sim_time_seconds: 0.0,
            enval_sum,
            avg_enval: (enval_sum / tile_count as f64) as f32,
            enval_next: enval.clone(),
            enval,
            element_fields_next: element_fields.clone(),
            element_fields,
            enval_landscape,
            cells: CellStore::default(),
            watched_cell_id: None,
            flux_log: CellFluxLog::default(),
            cell_phase_scratch: Vec::new(),
            mechanics_corrections: Vec::new(),
            contact_pairs: Vec::new(),
            indexed_positions: Vec::new(),
            spatial_sync_pending: false,
            contact_found: Vec::new(),
            contact_tile_offsets: Vec::new(),
            contact_tile_cells: Vec::new(),
            contact_row_scratch: Vec::new(),
            contact_row_owners: Vec::new(),
            contact_row_offsets: Vec::new(),
            contact_row_neighbors: Vec::new(),
            contact_displacements: Vec::new(),
            contact_loose_extras: Vec::new(),
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
            neighbors,
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

    pub fn enval_ledger_residual(&self) -> f64 {
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

    pub fn neighbors(&self, tile_id: TileId) -> Option<NeighborIndices> {
        self.neighbors.get(tile_id.index()).copied()
    }

    pub fn tile_enval(&self, tile_id: TileId) -> Option<f32> {
        self.enval.get(tile_id.index()).copied()
    }

    pub fn tile_element_amounts(&self, tile_id: TileId) -> Option<ElementAmounts> {
        self.element_fields.get(tile_id.index()).copied()
    }

    pub fn set_tile_element_amounts(
        &mut self,
        tile_id: TileId,
        amounts: ElementAmounts,
    ) -> Result<(), WorldError> {
        if tile_id.index() >= self.tile_count {
            return Err(WorldError::InvalidTile(tile_id));
        }
        amounts.validate_nonnegative("tile_element_amounts")?;
        self.element_fields[tile_id.index()] = amounts;
        self.element_fields_next[tile_id.index()] = amounts;
        Ok(())
    }

    pub fn element_field_totals(&self) -> [f64; ELEMENT_COUNT] {
        let mut totals = [0.0_f64; ELEMENT_COUNT];
        for amounts in &self.element_fields {
            for element in ELEMENT_ORDER {
                totals[element.index()] += f64::from(amounts[element]);
            }
        }
        totals
    }

    pub fn set_tile_enval(&mut self, tile_id: TileId, value: f32) -> Result<(), WorldError> {
        if tile_id.index() >= self.tile_count {
            return Err(WorldError::InvalidTile(tile_id));
        }
        if !value.is_finite() {
            return Err(WorldError::NonFiniteEnvalInput(value));
        }
        let old = self.enval[tile_id.index()];
        self.enval[tile_id.index()] = value;
        self.enval_next[tile_id.index()] = value;
        let delta = f64::from(value) - f64::from(old);
        self.enval_sum += delta;
        self.avg_enval = (self.enval_sum / self.tile_count as f64) as f32;
        self.enval_ledger.edits += delta;
        Ok(())
    }

    pub fn adjust_tile_enval(&mut self, tile_id: TileId, delta: f32) -> Result<(), WorldError> {
        let applied = self.adjust_tile_enval_unrecorded(tile_id, delta)?;
        self.enval_ledger.edits += applied;
        Ok(())
    }

    fn adjust_tile_enval_unrecorded(
        &mut self,
        tile_id: TileId,
        delta: f32,
    ) -> Result<f64, WorldError> {
        let applied = self.apply_tile_enval_delta(tile_id, delta)?;
        self.avg_enval = (self.enval_sum / self.tile_count as f64) as f32;
        Ok(applied)
    }

    fn apply_tile_enval_delta(&mut self, tile_id: TileId, delta: f32) -> Result<f64, WorldError> {
        if tile_id.index() >= self.tile_count {
            return Err(WorldError::InvalidTile(tile_id));
        }
        if !delta.is_finite() || delta == 0.0 {
            return Ok(0.0);
        }
        let old = self.enval[tile_id.index()];
        let value = old + delta;
        if !value.is_finite() {
            return Err(WorldError::NonFiniteEnvalInput(value));
        }
        self.enval[tile_id.index()] = value;
        self.enval_next[tile_id.index()] = value;
        let applied = f64::from(value) - f64::from(old);
        self.enval_sum += applied;
        Ok(applied)
    }

    pub fn set_all_enval(&mut self, value: f32) -> Result<(), WorldError> {
        if !value.is_finite() {
            return Err(WorldError::NonFiniteEnvalInput(value));
        }
        let before = self
            .enval
            .iter()
            .map(|value| f64::from(*value))
            .sum::<f64>();
        self.enval.fill(value);
        self.enval_next.fill(value);
        self.enval_sum = f64::from(value) * self.tile_count as f64;
        self.avg_enval = (self.enval_sum / self.tile_count as f64) as f32;
        self.enval_ledger.edits += self.enval_sum - before;
        Ok(())
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
            |id, center| {
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

    pub fn local_enval_average(&self, tile_id: TileId, radius: usize) -> Option<f32> {
        let (x, y) = self.tile_xy(tile_id)?;
        Some(self.local_enval_average_xy(x as isize, y as isize, radius))
    }

    pub fn local_enval_average_xy(&self, center_x: isize, center_y: isize, radius: usize) -> f32 {
        let radius = radius as isize;
        let mut sum = 0.0_f64;
        let mut count = 0_u32;
        for dx in -radius..=radius {
            for dy in -radius..=radius {
                let tile_id = self.wrapped_tile_id(center_x + dx, center_y + dy);
                sum += f64::from(self.enval[tile_id.index()]);
                count += 1;
            }
        }
        (sum / f64::from(count)) as f32
    }

    pub fn sample_enval_at(&self, position: Position) -> Option<f32> {
        let stencil = BilinearStencil::new(position, self.width, self.height)?;
        let mut value = 0.0_f64;
        for sample in stencil.samples {
            value +=
                f64::from(self.enval[self.index_xy(sample.x, sample.y)]) * f64::from(sample.weight);
        }
        Some(value as f32)
    }

    pub fn sample_element_fields_at(&self, position: Position) -> Option<ElementAmounts> {
        let stencil = BilinearStencil::new(position, self.width, self.height)?;
        let mut result = ElementAmounts::ZERO;
        for element in ELEMENT_ORDER {
            let mut value = 0.0_f64;
            for sample in stencil.samples {
                value += f64::from(self.element_fields[self.index_xy(sample.x, sample.y)][element])
                    * f64::from(sample.weight);
            }
            result[element] = value as f32;
        }
        Some(result)
    }

    pub fn local_enval_average_at(&self, position: Position, radius: usize) -> Option<f32> {
        position.wrapped(self.width as f32, self.height as f32)?;
        if radius == LOCAL_ENVAL_RADIUS {
            return Some(self.default_local_enval_average_at(position));
        }
        let radius = radius as isize;
        let mut sum = 0.0_f64;
        let mut count = 0_u32;
        for dx in -radius..=radius {
            for dy in -radius..=radius {
                let sample_position = Position::new(position.x + dx as f32, position.y + dy as f32);
                sum += f64::from(self.sample_enval_at(sample_position)?);
                count += 1;
            }
        }
        Some((sum / f64::from(count)) as f32)
    }

    fn default_local_enval_average_at(&self, position: Position) -> f32 {
        let axis = |center: f32, extent: usize| {
            let mut stencils = [(0_usize, 0_usize, 0.0_f32, 0.0_f32); LOCAL_ENVAL_WINDOW_DIAMETER];
            for (slot, offset) in
                (-(LOCAL_ENVAL_RADIUS as isize)..=LOCAL_ENVAL_RADIUS as isize).enumerate()
            {
                let wrapped = wrap_coordinate(center + offset as f32, extent as f32);
                let lower = wrapped.floor() as usize;
                let fraction = wrapped - lower as f32;
                stencils[slot] = (lower, (lower + 1) % extent, fraction, 1.0 - fraction);
            }
            stencils
        };
        let columns = axis(position.x, self.width);
        let rows = axis(position.y, self.height);
        let mut sum = 0.0_f64;
        for &(x0, x1, fx, one_minus_fx) in &columns {
            for &(y0, y1, fy, one_minus_fy) in &rows {
                let mut value = 0.0_f64;
                value += f64::from(self.enval[x0 * self.height + y0])
                    * f64::from(one_minus_fx * one_minus_fy);
                value +=
                    f64::from(self.enval[x1 * self.height + y0]) * f64::from(fx * one_minus_fy);
                value +=
                    f64::from(self.enval[x0 * self.height + y1]) * f64::from(one_minus_fx * fy);
                value += f64::from(self.enval[x1 * self.height + y1]) * f64::from(fx * fy);
                sum += f64::from(value as f32);
            }
        }
        (sum / LOCAL_ENVAL_WINDOW_AREA as f64) as f32
    }

    pub fn default_local_enval_average(&self, tile_id: TileId) -> Option<f32> {
        let tile_index = tile_id.index();
        if tile_index >= self.tile_count {
            return None;
        }

        let x = tile_index / self.height;
        let y = tile_index % self.height;
        if self.width > LOCAL_ENVAL_RADIUS * 2
            && self.height > LOCAL_ENVAL_RADIUS * 2
            && x >= LOCAL_ENVAL_RADIUS
            && x < self.width - LOCAL_ENVAL_RADIUS
            && y >= LOCAL_ENVAL_RADIUS
            && y < self.height - LOCAL_ENVAL_RADIUS
        {
            let start_y = y - LOCAL_ENVAL_RADIUS;
            let mut sum = 0.0_f64;
            for xx in (x - LOCAL_ENVAL_RADIUS)..=(x + LOCAL_ENVAL_RADIUS) {
                let base = xx * self.height + start_y;
                sum += f64::from(self.enval[base]);
                sum += f64::from(self.enval[base + 1]);
                sum += f64::from(self.enval[base + 2]);
                sum += f64::from(self.enval[base + 3]);
                sum += f64::from(self.enval[base + 4]);
            }
            return Some((sum / LOCAL_ENVAL_WINDOW_AREA as f64) as f32);
        }

        self.local_enval_average(tile_id, LOCAL_ENVAL_RADIUS)
    }

    fn tile_stencil(&self, position: Position) -> Option<TileStencil> {
        let position = position.wrapped(self.width as f32, self.height as f32)?;
        Some(self.canonical_tile_stencil(position))
    }

    fn canonical_tile_stencil(&self, position: Position) -> TileStencil {
        debug_assert_eq!(
            position.wrapped(self.width as f32, self.height as f32),
            Some(position)
        );
        let (x0, y0) = (position.x as usize, position.y as usize);
        let x1 = if x0 + 1 == self.width { 0 } else { x0 + 1 };
        let y1 = if y0 + 1 == self.height { 0 } else { y0 + 1 };
        let fx = position.x - x0 as f32;
        let fy = position.y - y0 as f32;
        let one_minus_fx = 1.0 - fx;
        let one_minus_fy = 1.0 - fy;
        TileStencil {
            tiles: [
                self.index_xy(x0, y0),
                self.index_xy(x1, y0),
                self.index_xy(x0, y1),
                self.index_xy(x1, y1),
            ],
            weights: [
                one_minus_fx * one_minus_fy,
                fx * one_minus_fy,
                one_minus_fx * fy,
                fx * fy,
            ],
        }
    }

    fn deposit_elements_at(
        &mut self,
        position: Position,
        amounts: ElementAmounts,
    ) -> Result<(), WorldError> {
        let stencil = self
            .tile_stencil(position)
            .ok_or(WorldError::InvalidCellPosition(position))?;
        self.deposit_elements_with(&stencil, amounts);
        Ok(())
    }

    fn deposit_elements_with(&mut self, stencil: &TileStencil, amounts: ElementAmounts) {
        for (tile, weight) in stencil.tiles.into_iter().zip(stencil.weights) {
            let field = &mut self.element_fields[tile];
            for element in ELEMENT_ORDER {
                field[element] += amounts[element] * weight;
            }
        }
    }

    fn adjust_enval_at(&mut self, position: Position, delta: f32) -> Result<f64, WorldError> {
        if !delta.is_finite() || delta == 0.0 {
            return Ok(0.0);
        }
        let stencil = self
            .tile_stencil(position)
            .ok_or(WorldError::NonFiniteEnvalInput(delta))?;
        self.adjust_enval_with(&stencil, delta)
    }

    fn adjust_enval_with(&mut self, stencil: &TileStencil, delta: f32) -> Result<f64, WorldError> {
        if !delta.is_finite() || delta == 0.0 {
            return Ok(0.0);
        }
        let mut applied = 0.0;
        let mut result = Ok(());
        for (tile, weight) in stencil.tiles.into_iter().zip(stencil.weights) {
            let tile_delta = delta * weight;
            if tile_delta == 0.0 {
                continue;
            }
            let old = self.enval[tile];
            let value = old + tile_delta;
            if !value.is_finite() {
                result = Err(WorldError::NonFiniteEnvalInput(value));
                break;
            }
            self.enval[tile] = value;
            applied += f64::from(value) - f64::from(old);
        }
        self.enval_sum += applied;
        self.avg_enval = (self.enval_sum / self.tile_count as f64) as f32;
        result.map(|()| applied)
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

    pub fn spawn_cell_with_genome_at(
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

    fn founder_material(
        &self,
        position: Position,
    ) -> Option<(ElementAmounts, ElementAmounts, [(usize, ElementAmounts); 4])> {
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

    fn enval_phase(&mut self) {
        self.diffuse_enval_field();
        let kappa = self.landscape_kappa();
        let outcome = environment::recharge_and_relax(
            &mut self.enval,
            &mut self.element_fields,
            &self.enval_landscape,
            f64::from(self.config.enval_recharge.rate_per_second),
            self.config.dt_seconds,
            kappa,
        );
        self.enval_ledger.recharge += outcome.recharge.enval_delta;
        self.enval_ledger.recharge_amount += outcome.recharge.amount;
        self.enval_ledger.recharge_energy += outcome.recharge.energy;
        self.enval_ledger.source_inflow += outcome.relaxation;
        self.enval_sum = outcome.enval_sum;
        self.avg_enval = (outcome.enval_sum / self.tile_count as f64) as f32;
    }

    fn landscape_kappa(&self) -> f64 {
        let rate = f64::from(self.config.enval_landscape.relaxation_per_second);
        if rate <= 0.0 {
            return 0.0;
        }
        1.0 - (-rate * self.config.dt_seconds).exp()
    }

    #[cfg(test)]
    fn apply_recharge(&mut self) -> bool {
        let recharge_rate = f64::from(self.config.enval_recharge.rate_per_second);
        if recharge_rate <= 0.0 {
            return false;
        }
        let outcome = environment::recharge(
            &mut self.enval,
            &mut self.element_fields,
            recharge_rate,
            self.config.dt_seconds,
        );
        self.enval_ledger.recharge += outcome.enval_delta;
        self.enval_ledger.recharge_amount += outcome.amount;
        self.enval_ledger.recharge_energy += outcome.energy;
        true
    }

    #[cfg(test)]
    fn relax_enval_landscape(&mut self) -> bool {
        let kappa = self.landscape_kappa();
        if kappa <= 0.0 {
            return false;
        }
        self.enval_ledger.source_inflow +=
            environment::relax_enval(&mut self.enval, &self.enval_landscape, kappa);
        true
    }

    fn refresh_enval_sum(&mut self) {
        let sum = self
            .enval
            .iter()
            .map(|value| f64::from(*value))
            .sum::<f64>();
        self.enval_sum = sum;
        self.avg_enval = (sum / self.tile_count as f64) as f32;
    }

    pub fn diffuse_enval(&mut self) {
        self.diffuse_enval_field();
        self.refresh_enval_sum();
    }

    fn diffuse_enval_field(&mut self) {
        let mix = (f64::from(self.config.enval_diffusivity_per_second) * self.config.dt_seconds
            / 9.0) as f32;
        let (width, height) = (self.width, self.height);
        let rows = &mut self.enval_row_sums;
        rows.resize(self.tile_count, 0.0);
        for x in 0..width {
            let left = if x == 0 { width - 1 } else { x - 1 } * height;
            let right = if x + 1 == width { 0 } else { x + 1 } * height;
            let here = x * height;
            for (sum, ((l, c), r)) in rows[here..here + height].iter_mut().zip(
                self.enval[left..left + height]
                    .iter()
                    .zip(&self.enval[here..here + height])
                    .zip(&self.enval[right..right + height]),
            ) {
                *sum = l + c + r;
            }
        }
        let mix_tile = |center: f32, up: f32, row: f32, down: f32| {
            let value = center + mix * (up + row + down - 9.0 * center);
            if value.is_finite() { value } else { 0.0 }
        };
        for x in 0..width {
            let here = x * height;
            let rows = &rows[here..here + height];
            let centers = &self.enval[here..here + height];
            let next = &mut self.enval_next[here..here + height];
            if height >= 3 {
                for (y, value) in next[1..height - 1].iter_mut().enumerate() {
                    *value = mix_tile(centers[y + 1], rows[y], rows[y + 1], rows[y + 2]);
                }
            }
            for y in [0, height - 1] {
                let up = if y == 0 { height - 1 } else { y - 1 };
                let down = if y + 1 == height { 0 } else { y + 1 };
                next[y] = mix_tile(centers[y], rows[up], rows[y], rows[down]);
            }
        }
        std::mem::swap(&mut self.enval, &mut self.enval_next);
    }

    pub fn diffuse_element_fields(&mut self) {
        let (width, height) = (self.width, self.height);
        let mut mix = [0.0_f32; ELEMENT_COUNT];
        for element in ELEMENT_ORDER {
            mix[element.index()] = (f64::from(self.config.element_fields.diffusivities[element])
                * self.config.dt_seconds
                / 9.0) as f32;
        }
        let rows = &mut self.element_row_sums;
        rows.resize(self.tile_count, [0.0; ELEMENT_COUNT]);
        for x in 0..width {
            let left = if x == 0 { width - 1 } else { x - 1 } * height;
            let right = if x + 1 == width { 0 } else { x + 1 } * height;
            let here = x * height;
            for (sum, ((l, c), r)) in rows[here..here + height].iter_mut().zip(
                self.element_fields[left..left + height]
                    .iter()
                    .zip(&self.element_fields[here..here + height])
                    .zip(&self.element_fields[right..right + height]),
            ) {
                let (l, c, r) = (l.as_array(), c.as_array(), r.as_array());
                *sum = std::array::from_fn(|element| l[element] + c[element] + r[element]);
            }
        }
        let mix_tile = |center: &ElementAmounts,
                        up: &[f32; ELEMENT_COUNT],
                        row: &[f32; ELEMENT_COUNT],
                        down: &[f32; ELEMENT_COUNT]| {
            let center = center.as_array();
            let mut next = [0.0_f32; ELEMENT_COUNT];
            for element in 0..ELEMENT_COUNT {
                next[element] = center[element]
                    + mix[element]
                        * (up[element] + row[element] + down[element] - 9.0 * center[element]);
            }
            ElementAmounts::new(next)
        };
        for x in 0..width {
            let here = x * height;
            let rows = &rows[here..here + height];
            let centers = &self.element_fields[here..here + height];
            let next = &mut self.element_fields_next[here..here + height];
            let mut up = rows[height - 1];
            let mut row = rows[0];
            for ((value, center), down) in next.iter_mut().zip(centers).zip(&rows[1..]) {
                *value = mix_tile(center, &up, &row, down);
                up = row;
                row = *down;
            }
            next[height - 1] = mix_tile(&centers[height - 1], &up, &row, &rows[0]);
        }

        self.operation_counters.element_field_diffusion_tiles = self
            .operation_counters
            .element_field_diffusion_tiles
            .saturating_add(self.tile_count as u64);

        std::mem::swap(&mut self.element_fields, &mut self.element_fields_next);
    }

    pub fn stats(&self) -> WorldStats {
        self.collect_stats(true)
    }

    pub fn compact_stats(&self) -> WorldStats {
        self.collect_stats(false)
    }

    fn collect_stats(&self, include_distribution_stats: bool) -> WorldStats {
        let mut min_enval = f32::INFINITY;
        let mut max_enval = f32::NEG_INFINITY;
        let mut enval_sum = 0.0_f64;
        let mut enval_sum_sq = 0.0_f64;
        let mut positive_enval_tile_count = 0_usize;
        let mut negative_enval_tile_count = 0_usize;
        let mut near_zero_enval_tile_count = 0_usize;
        for value in &self.enval {
            min_enval = min_enval.min(*value);
            max_enval = max_enval.max(*value);
            let as_f64 = f64::from(*value);
            enval_sum += as_f64;
            enval_sum_sq += as_f64 * as_f64;
            if *value > 1.0e-4 {
                positive_enval_tile_count += 1;
            } else if *value < -1.0e-4 {
                negative_enval_tile_count += 1;
            } else {
                near_zero_enval_tile_count += 1;
            }
        }
        let (enval_p05, enval_p50, enval_p95) = if include_distribution_stats {
            let mut sorted_enval = self.enval.clone();
            sorted_enval.sort_by(|a, b| a.total_cmp(b));
            (
                percentile_sorted_f32(&sorted_enval, 0.05),
                percentile_sorted_f32(&sorted_enval, 0.50),
                percentile_sorted_f32(&sorted_enval, 0.95),
            )
        } else {
            (0.0, 0.0, 0.0)
        };
        let enval_average = if self.tile_count > 0 {
            enval_sum / self.tile_count as f64
        } else {
            0.0
        };
        let enval_variance = if self.tile_count > 0 {
            (enval_sum_sq / self.tile_count as f64 - enval_average * enval_average).max(0.0)
        } else {
            0.0
        };
        let enval_std_dev = enval_variance.sqrt() as f32;
        if self.tile_count == 0 {
            min_enval = 0.0;
            max_enval = 0.0;
        }

        let mut extracellular_element_amounts = [0.0_f64; ELEMENT_COUNT];
        for amounts in &self.element_fields {
            for element in ELEMENT_ORDER {
                extracellular_element_amounts[element.index()] += f64::from(amounts[element]);
            }
        }
        let mut intracellular_element_amounts = [0.0_f64; ELEMENT_COUNT];
        for cell in self.cells.iter() {
            for element in ELEMENT_ORDER {
                intracellular_element_amounts[element.index()] +=
                    f64::from(cell.internal_elements[element])
                        + f64::from(cell.body[element])
                        + f64::from(cell.boundary[element]);
            }
        }
        let mut system_element_amounts = [0.0_f64; ELEMENT_COUNT];
        for element in ELEMENT_ORDER {
            system_element_amounts[element.index()] = extracellular_element_amounts
                [element.index()]
                + intracellular_element_amounts[element.index()];
        }
        let total_element_amount = system_element_amounts.iter().sum();

        let occupied_tile_count = self
            .tile_cell_counts
            .iter()
            .filter(|count| **count > 0)
            .count();
        let empty_tile_count = self.tile_count.saturating_sub(occupied_tile_count);
        let occupancy_fraction = if self.tile_count > 0 {
            occupied_tile_count as f64 / self.tile_count as f64
        } else {
            0.0
        };

        let mut live_cell_count = 0_usize;
        let mut energy_sum = 0.0_f64;
        let mut min_cell_energy = f64::INFINITY;
        let mut max_cell_energy = f64::NEG_INFINITY;
        let mut age_sum = 0.0_f64;
        let mut max_cell_age = 0.0_f64;
        let mut enzyme_sum = 0_usize;
        let mut min_enzyme_count = usize::MAX;
        let mut max_enzyme_count = 0_usize;
        let mut enzyme_count_histogram = [0_u64; ENZYME_COUNT_HISTOGRAM_LEN];
        let mut cells_at_enzyme_cap = 0_usize;
        let mut exposure_sum = 0.0_f64;
        let mut max_exposure = 0.0_f32;
        let mut exposed_cell_count = 0_usize;
        let mut mixed_polarity_cells = 0_usize;
        let mut enval_optimum_spread_sum = 0.0_f64;
        let mut niche_mismatch_sum = 0.0_f64;

        for cell in self.cells.iter() {
            live_cell_count += 1;
            energy_sum += cell.energy;
            min_cell_energy = min_cell_energy.min(cell.energy);
            max_cell_energy = max_cell_energy.max(cell.energy);
            let age = (self.sim_time_seconds - cell.birth_sim_time).max(0.0);
            age_sum += age;
            max_cell_age = max_cell_age.max(age);

            let enzyme_count = cell.genome.enzymes.len();
            enzyme_sum += enzyme_count;
            min_enzyme_count = min_enzyme_count.min(enzyme_count);
            max_enzyme_count = max_enzyme_count.max(enzyme_count);
            let histogram_index = enzyme_count.min(ENZYME_COUNT_HISTOGRAM_LEN - 1);
            enzyme_count_histogram[histogram_index] =
                enzyme_count_histogram[histogram_index].saturating_add(1);
            if enzyme_count >= MAX_CELL_ENZYMES {
                cells_at_enzyme_cap += 1;
            }
            let mut cell_max_exposure = 0.0_f32;
            let mut cell_min_optimum = f32::INFINITY;
            let mut cell_max_optimum = f32::NEG_INFINITY;
            for enzyme in &cell.genome.enzymes {
                exposure_sum += f64::from(enzyme.exposure);
                cell_max_exposure = cell_max_exposure.max(enzyme.exposure);
                cell_min_optimum = cell_min_optimum.min(enzyme.enval_optimum);
                cell_max_optimum = cell_max_optimum.max(enzyme.enval_optimum);
            }
            max_exposure = max_exposure.max(cell_max_exposure);
            if cell_max_exposure >= EXPOSED_CELL_THRESHOLD {
                exposed_cell_count += 1;
            }
            if cell_max_optimum > MIXED_POLARITY_THRESHOLD
                && cell_min_optimum < -MIXED_POLARITY_THRESHOLD
            {
                mixed_polarity_cells += 1;
            }
            if enzyme_count > 0 {
                enval_optimum_spread_sum += f64::from(cell_max_optimum - cell_min_optimum);
            }
            if include_distribution_stats {
                niche_mismatch_sum += self.cell_niche_mismatch(cell);
            }
        }

        if live_cell_count == 0 {
            min_cell_energy = 0.0;
            max_cell_energy = 0.0;
            min_enzyme_count = 0;
        }

        let total_lineage_records = self.lineage_counters.len();
        let extant_lineage_count = self.extant_lineage_count();
        let extinct_lineage_count = self
            .lineage_counters
            .values()
            .filter(|counters| counters.population == 0)
            .count();
        let mut dominant_lineage_id = 0_u64;
        let mut dominant_lineage_population = 0_u64;
        let mut lineage_entropy = 0.0_f64;
        for (lineage_id, counters) in &self.lineage_counters {
            if counters.population > dominant_lineage_population {
                dominant_lineage_id = lineage_id.raw();
                dominant_lineage_population = counters.population;
            }
            if live_cell_count > 0 && counters.population > 0 {
                let p = counters.population as f64 / live_cell_count as f64;
                lineage_entropy -= p * p.ln();
            }
        }
        let dominant_lineage_share = if live_cell_count > 0 {
            dominant_lineage_population as f64 / live_cell_count as f64
        } else {
            0.0
        };

        let mut radii = self
            .cells
            .iter()
            .map(|cell| cell.radius)
            .collect::<Vec<_>>();
        radii.sort_unstable_by(f32::total_cmp);
        let radius_median = percentile_sorted_f32(&radii, 0.5);
        let radius_p95 = percentile_sorted_f32(&radii, 0.95);
        let radius_max = radii.last().copied().unwrap_or(0.0);
        let mean_strain = self
            .cells
            .iter()
            .map(|cell| cell.strain(self.config.cell_body.boundary_density))
            .sum::<f64>()
            / live_cell_count.max(1) as f64;
        let live_cell_count_f64 = live_cell_count as f64;
        WorldStats {
            tick_count: self.tick_count,
            sim_time_seconds: self.sim_time_seconds,
            width: self.width,
            height: self.height,
            tile_count: self.tile_count,
            occupied_tile_count,
            empty_tile_count,
            occupancy_fraction,
            extracellular_element_amounts,
            intracellular_element_amounts,
            system_element_amounts,
            total_element_amount,
            average_enval: self.avg_enval,
            min_enval,
            max_enval,
            enval_std_dev,
            enval_p05,
            enval_p50,
            enval_p95,
            positive_enval_tile_count,
            negative_enval_tile_count,
            near_zero_enval_tile_count,
            cell_count: live_cell_count,
            live_cell_count,
            births: self.birth_count,
            deaths: self.death_count,
            fissions: self.reaction_counters.divisions,
            energy_deaths: self.energy_death_count,
            lyses: self.lysis_count,
            radius_median,
            radius_p95,
            radius_max,
            mean_strain,
            lineage_count: extant_lineage_count,
            extant_lineage_count,
            total_lineage_records,
            extinct_lineage_count,
            dominant_lineage_id,
            dominant_lineage_population,
            dominant_lineage_share,
            lineage_entropy,
            average_cell_energy: if live_cell_count > 0 {
                energy_sum / live_cell_count_f64
            } else {
                0.0
            },
            min_cell_energy,
            max_cell_energy,
            total_cell_energy: energy_sum,
            average_cell_age: if live_cell_count > 0 {
                age_sum / live_cell_count_f64
            } else {
                0.0
            },
            max_cell_age,
            average_enzyme_count: if live_cell_count > 0 {
                enzyme_sum as f64 / live_cell_count_f64
            } else {
                0.0
            },
            min_enzyme_count,
            max_enzyme_count,
            enzyme_count_histogram,
            cells_at_enzyme_cap,
            fraction_cells_at_enzyme_cap: if live_cell_count > 0 {
                cells_at_enzyme_cap as f64 / live_cell_count_f64
            } else {
                0.0
            },
            total_enzyme_count: enzyme_sum as u64,
            mean_exposure: if enzyme_sum > 0 {
                exposure_sum / enzyme_sum as f64
            } else {
                0.0
            },
            max_exposure,
            exposed_cell_count,
            mixed_polarity_cells,
            mean_enval_optimum_spread: enval_optimum_spread_sum / live_cell_count.max(1) as f64,
            mean_niche_mismatch: niche_mismatch_sum / live_cell_count.max(1) as f64,
            contact_pairs: self.contact_pair_count,
            cross_lineage_contact_pairs: self.cross_lineage_contact_pair_count,
            interaction: self.interaction_counters,
            reaction_counters: self.reaction_counters,
            operation_counters: self.operation_counters,
            energy_ledger: self.energy_ledger,
            enval_ledger: self.enval_ledger,
            energy_ledger_residual: energy_sum - self.energy_ledger.expected_cell_energy(),
            renewable_coverage: renewable_coverage(&self.energy_ledger, &self.enval_ledger),
        }
    }

    fn cell_niche_mismatch(&self, cell: &Cell) -> f64 {
        let local = f64::from(
            self.local_enval_average_at(cell.position, LOCAL_ENVAL_RADIUS)
                .unwrap_or(0.0),
        );
        let mut rate_total = 0.0_f64;
        let mut weighted_mismatch = 0.0_f64;
        for enzyme in &cell.genome.enzymes {
            let rate = f64::from(enzyme.rate);
            rate_total += rate;
            weighted_mismatch += rate * (local - f64::from(enzyme.enval_optimum)).abs();
        }
        if rate_total > 0.0 {
            weighted_mismatch / rate_total
        } else {
            0.0
        }
    }

    pub fn rebuild_derived_caches(&mut self) {
        self.neighbors = build_neighbors(self.width, self.height);
        self.cell_phase_scratch.clear();
        self.mechanics_corrections.clear();
        self.contact_pairs.clear();
        self.contact_list_current = false;
        self.contact_scratch.clear();
        self.cells.rebuild_index();
        self.rebuild_spatial_index();
    }

    fn rebuild_spatial_index(&mut self) {
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

    fn center_tile_id(&self, position: Position) -> Option<TileId> {
        let position = position.wrapped(self.width as f32, self.height as f32)?;
        Some(TileId(self.index_xy(
            position.x.floor() as usize,
            position.y.floor() as usize,
        )))
    }

    fn increment_tile_cell_count(&mut self, position: Position) {
        let tile_id = self
            .center_tile_id(position)
            .expect("active cell position must map to a field tile");
        self.tile_cell_counts[tile_id.index()] =
            self.tile_cell_counts[tile_id.index()].saturating_add(1);
    }

    fn decrement_tile_cell_count(&mut self, position: Position) {
        let tile_id = self
            .center_tile_id(position)
            .expect("active cell position must map to a field tile");
        let count = &mut self.tile_cell_counts[tile_id.index()];
        assert!(*count > 0, "derived tile cell count underflow");
        *count -= 1;
    }

    fn canonical_tile_index(&self, position: Position) -> usize {
        debug_assert_eq!(
            position.wrapped(self.width as f32, self.height as f32),
            Some(position)
        );
        (position.x as usize).min(self.width - 1) * self.height
            + (position.y as usize).min(self.height - 1)
    }

    fn update_cell_position(&mut self, cell_id: CellId, position: Position) {
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

    pub fn build_render_buffers(&self) -> RenderBuffers {
        let mut buffers = RenderBuffers::default();
        self.write_render_buffers(&mut buffers);
        buffers
    }

    pub fn write_render_buffers(&self, buffers: &mut RenderBuffers) {
        self.write_render_buffers_with_visual(buffers, &RenderVisualState::default());
    }

    pub fn write_render_buffers_with_visual(
        &self,
        buffers: &mut RenderBuffers,
        visual: &RenderVisualState,
    ) {
        buffers.clear_for_refresh();
        buffers.width = self.width.min(u32::MAX as usize) as u32;
        buffers.height = self.height.min(u32::MAX as usize) as u32;
        buffers.tick_count = self.tick_count;
        buffers.sim_time_seconds = self.sim_time_seconds;
        buffers.render_epoch = (self.tick_count & u64::from(u32::MAX)) as u32;

        buffers.tile_enval.reserve(self.tile_count);
        buffers.tile_cell_count.reserve(self.tile_count);
        buffers.tile_mass_density.reserve(self.tile_count);
        buffers.tile_total_elements.reserve(self.tile_count);
        buffers
            .tile_element_concentrations
            .reserve(self.tile_count.saturating_mul(ELEMENT_COUNT));
        buffers.cell_id.reserve(self.cells.len());
        buffers
            .cell_point_data
            .reserve(self.cells.len().saturating_mul(4));
        buffers
            .cell_rotation_data
            .reserve(self.cells.len().saturating_mul(4));
        buffers
            .cell_scale_data
            .reserve(self.cells.len().saturating_mul(4));
        buffers
            .cell_rgba
            .reserve(self.cells.len().saturating_mul(4));
        buffers.cell_radius.reserve(self.cells.len());
        buffers.cell_energy.reserve(self.cells.len());
        buffers.cell_lineage.reserve(self.cells.len());
        buffers.cell_flags.reserve(self.cells.len());
        buffers.cell_enzyme_count.reserve(self.cells.len());
        buffers.cell_age_seconds.reserve(self.cells.len());

        buffers.tile_enval.extend_from_slice(&self.enval);

        buffers
            .tile_cell_count
            .extend_from_slice(&self.tile_cell_counts);

        for tile_index in 0..self.tile_count {
            let amounts = self.element_fields[tile_index];
            buffers.tile_mass_density.push(amounts.mass() as f32);
            buffers.tile_total_elements.push(amounts.total() as f32);
            buffers
                .tile_element_concentrations
                .extend_from_slice(amounts.as_array());
        }

        for cell in self.cells.iter() {
            buffers
                .cell_id
                .push(cell.id.index().min(u32::MAX as usize) as u32);
            buffers.cell_point_data.extend_from_slice(&[
                cell.position.x + 0.5 - self.width as f32 * 0.5,
                self.height as f32 * 0.5 - cell.position.y - 0.5,
                0.5,
                0.0,
            ]);
            buffers
                .cell_rotation_data
                .extend_from_slice(&[0.0, 0.0, 0.0, 1.0]);
            buffers.cell_scale_data.extend_from_slice(&[
                cell.radius,
                cell.radius,
                cell.radius,
                0.0,
            ]);
            buffers.cell_radius.push(cell.radius);
            buffers.cell_energy.push(cell.energy as f32);
            buffers
                .cell_lineage
                .push(cell.lineage_id.raw().min(u64::from(u32::MAX)) as u32);
            buffers.cell_flags.push(1);
            buffers
                .cell_enzyme_count
                .push(cell.genome.enzymes.len().min(u32::MAX as usize) as u32);
            buffers
                .cell_age_seconds
                .push((self.sim_time_seconds - cell.birth_sim_time).max(0.0) as f32);
        }
        buffers.refresh_lattice_rgba(visual);
    }

    pub fn inspect_tile(&self, tile_id: TileId) -> Option<TileInspection> {
        if tile_id.index() >= self.tile_count {
            return None;
        }
        let (x, y) = self.tile_xy(tile_id)?;
        let amounts = self.element_fields[tile_id.index()];
        Some(TileInspection {
            tile_id,
            x,
            y,
            enval: self.enval[tile_id.index()],
            enval_landscape_target: self.enval_landscape[tile_id.index()],
            cell_center_count: self.tile_cell_counts[tile_id.index()],
            element_concentrations: *amounts.as_array(),
            total_element_concentration: amounts.total() as f32,
            mass_density: amounts.mass() as f32,
        })
    }

    pub fn inspect_tile_xy(&self, x: usize, y: usize) -> Option<TileInspection> {
        self.tile_id(x, y)
            .and_then(|tile_id| self.inspect_tile(tile_id))
    }

    pub fn inspect_cell(&self, cell_id: CellId) -> Option<CellInspection> {
        let cell = self.cells.get(cell_id)?;
        let mut contacts = Vec::new();
        let field_share = self.collect_contacts(cell_id, cell.position, cell.radius, &mut contacts);
        Some(CellInspection {
            cell_id,
            x: cell.position.x,
            y: cell.position.y,
            radius: cell.radius,
            body_total: cell.body.total(),
            boundary_total: cell.boundary.total(),
            area: cell.area(self.config.cell_body.body_density),
            strain: cell.strain(self.config.cell_body.boundary_density),
            membrane_polarity: cell.membrane_polarity(),
            energy: cell.energy,
            lineage_id: cell.lineage_id,
            enzyme_count: cell.genome.enzymes.len(),
            total_internal_elements: cell.internal_elements.total() as f32,
            age_seconds: (self.sim_time_seconds - cell.birth_sim_time).max(0.0),
            mean_enval_optimum: cell.genome.mean_enval_optimum(),
            enval_optimum_spread: cell.genome.enval_optimum_spread(),
            local_enval_average: self
                .local_enval_average_at(cell.position, LOCAL_ENVAL_RADIUS)
                .unwrap_or(0.0),
            field_share,
            contacts: contacts
                .iter()
                .map(|contact| ContactInspection {
                    cell_id: contact.cell_id.index(),
                    lineage_id: self.cells[contact.cell_id].lineage_id,
                    distance: contact.distance,
                    share: contact.share,
                })
                .collect(),
        })
    }

    pub fn inspect_cell_detail(
        &mut self,
        cell_id: CellId,
        flux_limit: usize,
    ) -> Option<CellDetailInspection> {
        let recent_fluxes = self.inspect_cell_fluxes(cell_id, flux_limit)?;
        let cell = self.cells.get(cell_id)?;
        let summary = self.inspect_cell(cell_id)?;
        Some(CellDetailInspection {
            cell: summary,
            maintenance_cost_per_area: cell.maintenance_cost_per_area,
            catalyst_upkeep_per_sec: self.catalyst_upkeep_per_sec(cell),
            genome: inspect_genome(&cell.genome),
            body: *cell.body.as_array(),
            boundary: *cell.boundary.as_array(),
            internal_elements: *cell.internal_elements.as_array(),
            total_internal_elements: cell.internal_elements.total() as f32,
            recent_fluxes,
        })
    }

    pub fn inspect_cell_fluxes(
        &mut self,
        cell_id: CellId,
        limit: usize,
    ) -> Option<FluxLogInspection> {
        self.cells.get(cell_id)?;
        if self.watched_cell_id != Some(cell_id) {
            self.watched_cell_id = Some(cell_id);
            self.flux_log.clear();
        }
        let flux_count = self.flux_log.flux_count();
        let fluxes = self
            .flux_log
            .fluxes_newest_first()
            .take(limit)
            .cloned()
            .collect::<Vec<_>>();
        Some(FluxLogInspection {
            available: true,
            reason: "recorded",
            limit,
            truncated: flux_count > fluxes.len(),
            flux_count,
            returned_count: fluxes.len(),
            order: "newest_first",
            fluxes,
        })
    }

    pub fn inspect_lineage(&self, lineage_id: LineageId) -> Option<LineageSummaryInspection> {
        let counters = *self.lineage_counters.get(&lineage_id)?;
        Some(self.lineage_summary(lineage_id, counters))
    }

    pub fn list_lineages(&self, limit: usize) -> LineageListInspection {
        let extant_lineage_count = self.extant_lineage_count();
        let entries = self.top_lineages(limit);
        let lineages = entries
            .into_iter()
            .map(|(lineage_id, counters)| self.lineage_summary(lineage_id, counters))
            .collect::<Vec<_>>();
        LineageListInspection {
            extant_lineage_count,
            total_lineage_records: self.lineage_counters.len(),
            limit,
            truncated: extant_lineage_count > limit,
            lineages,
        }
    }

    pub fn apply_cell_genome_patch(
        &mut self,
        cell_id: CellId,
        patch: &GenomePatch,
    ) -> Result<GenomeEditResult, WorldError> {
        let changed_fields = self.apply_genome_patch_to_cell(cell_id, patch)?;
        Ok(GenomeEditResult {
            target: "cell",
            cell_id: Some(cell_id.index()),
            center_x: None,
            center_y: None,
            brush_width: None,
            brush_height: None,
            visited_tile_count: 1,
            patched_cell_count: 1,
            changed_fields,
        })
    }

    pub fn apply_genome_brush(
        &mut self,
        center_x: usize,
        center_y: usize,
        brush_width: usize,
        brush_height: usize,
        patch: &GenomePatch,
    ) -> Result<GenomeEditResult, WorldError> {
        let width = brush_width.min(self.width).max(1);
        let height = brush_height.min(self.height).max(1);
        let start_x = center_x as isize - (width as isize / 2);
        let start_y = center_y as isize - (height as isize / 2);
        let mut covered_tiles = vec![false; self.tile_count];
        for dx in 0..width {
            for dy in 0..height {
                let tile_id = self.wrapped_tile_id(start_x + dx as isize, start_y + dy as isize);
                covered_tiles[tile_id.index()] = true;
            }
        }
        let cells = self
            .cells
            .iter()
            .filter(|cell| {
                self.center_tile_id(cell.position)
                    .is_some_and(|tile_id| covered_tiles[tile_id.index()])
            })
            .map(|cell| cell.id)
            .collect::<Vec<_>>();

        let mut changed = BTreeSet::new();
        for cell_id in cells.iter().copied() {
            let Some(cell) = self.cells.get(cell_id) else {
                return Err(WorldError::InvalidCell(cell_id));
            };
            let mut genome = cell.genome.clone();
            let changed_fields = patch
                .apply_to_genome(&mut genome)
                .map_err(|err| WorldError::GenomePatch(err.to_string()))?;
            for field in changed_fields {
                changed.insert(field);
            }
        }

        let mut patched_cell_count = 0_usize;
        for cell_id in cells.iter().copied() {
            self.apply_genome_patch_to_cell(cell_id, patch)?;
            patched_cell_count += 1;
        }

        Ok(GenomeEditResult {
            target: "brush",
            cell_id: None,
            center_x: Some(center_x % self.width),
            center_y: Some(center_y % self.height),
            brush_width: Some(width),
            brush_height: Some(height),
            visited_tile_count: width.saturating_mul(height),
            patched_cell_count,
            changed_fields: changed.into_iter().collect(),
        })
    }

    fn apply_genome_patch_to_cell(
        &mut self,
        cell_id: CellId,
        patch: &GenomePatch,
    ) -> Result<Vec<String>, WorldError> {
        let Some(cell) = self.cells.get(cell_id) else {
            return Err(WorldError::InvalidCell(cell_id));
        };
        let original_lineage = cell.lineage_id;
        let mut genome = cell.genome.clone();
        let changed_fields = patch
            .apply_to_genome(&mut genome)
            .map_err(|err| WorldError::GenomePatch(err.to_string()))?;
        genome.lineage_id = original_lineage;

        let cell = self
            .cells
            .get_mut(cell_id)
            .ok_or(WorldError::InvalidCell(cell_id))?;
        cell.genome = genome;
        cell.lineage_id = original_lineage;
        cell.maintenance_cost_per_area = cell.genome.maintenance_cost_per_area;
        Ok(changed_fields)
    }

    fn lineage_summary(
        &self,
        lineage_id: LineageId,
        counters: LineageCounters,
    ) -> LineageSummaryInspection {
        let mut live_count = 0_u64;
        let mut energy_sum = 0.0_f64;
        let mut enzyme_sum = 0_u64;
        let mut exposure_sum = 0.0_f64;
        let mut rate_sum = 0.0_f64;
        let mut rate_weighted_optimum_sum = 0.0_f64;
        for cell in self.cells.iter() {
            if cell.lineage_id != lineage_id {
                continue;
            }
            live_count = live_count.saturating_add(1);
            energy_sum += cell.energy;
            enzyme_sum = enzyme_sum.saturating_add(cell.genome.enzymes.len() as u64);
            for enzyme in &cell.genome.enzymes {
                exposure_sum += f64::from(enzyme.exposure);
                rate_sum += f64::from(enzyme.rate);
                rate_weighted_optimum_sum += f64::from(enzyme.rate) * f64::from(enzyme.enval_optimum);
            }
        }
        let denominator = live_count.max(1) as f64;
        let total_live_cells = self.cells.len().max(1) as f64;
        LineageSummaryInspection {
            lineage_id,
            population: counters.population,
            births: counters.births,
            deaths: counters.deaths,
            extinct: counters.population == 0,
            share: counters.population as f64 / total_live_cells,
            average_energy: if live_count > 0 {
                energy_sum / denominator
            } else {
                0.0
            },
            average_enzyme_count: if live_count > 0 {
                enzyme_sum as f64 / denominator
            } else {
                0.0
            },
            mean_exposure: if enzyme_sum > 0 {
                exposure_sum / enzyme_sum as f64
            } else {
                0.0
            },
            mean_enval_optimum: if rate_sum > 0.0 {
                rate_weighted_optimum_sum / rate_sum
            } else {
                0.0
            },
        }
    }

    pub fn check_invariants(&self) -> Result<(), InvariantError> {
        if self.neighbors.len() != self.tile_count
            || self.enval.len() != self.tile_count
            || self.enval_next.len() != self.tile_count
            || self.enval_landscape.len() != self.tile_count
            || self.element_fields.len() != self.tile_count
            || self.element_fields_next.len() != self.tile_count
            || self.tile_cell_counts.len() != self.tile_count
        {
            return Err(InvariantError::MismatchedWorldArrayLengths);
        }

        for tile_index in 0..self.tile_count {
            if !self.enval[tile_index].is_finite() {
                return Err(InvariantError::NonFiniteEnval(TileId(tile_index)));
            }
            for element in ELEMENT_ORDER {
                let value = self.element_fields[tile_index][element];
                if !value.is_finite() || value < 0.0 {
                    return Err(InvariantError::InvalidElementField {
                        tile: TileId(tile_index),
                        element,
                    });
                }
            }
            let neighbors = self.neighbors[tile_index];
            for neighbor in [
                neighbors.left,
                neighbors.right,
                neighbors.up,
                neighbors.down,
                neighbors.up_left,
                neighbors.up_right,
                neighbors.down_left,
                neighbors.down_right,
            ] {
                if neighbor.index() >= self.tile_count {
                    return Err(InvariantError::InvalidNeighbor {
                        tile: TileId(tile_index),
                        neighbor,
                    });
                }
            }
        }

        if !self.cells.is_consistent() {
            return Err(InvariantError::CellStoreMismatch);
        }
        let mut expected_tile_cell_counts = vec![0_u32; self.tile_count];
        for cell in self.cells.iter() {
            let cell_id = cell.id;
            if !cell.energy.is_finite() || cell.energy < 0.0 {
                return Err(InvariantError::NonFiniteCellEnergy(cell_id));
            }
            if !cell.position.is_finite()
                || !cell.radius.is_finite()
                || cell.body.total() <= 0.0
                || (cell.radius
                    - (cell.area(self.config.cell_body.body_density) / std::f64::consts::PI).sqrt()
                        as f32)
                    .abs()
                    > GEOMETRY_TOLERANCE
                || cell.radius <= 0.0
                || cell.position.wrapped(self.width as f32, self.height as f32)
                    != Some(cell.position)
            {
                return Err(InvariantError::InvalidCellGeometry(cell_id));
            }
            if !self.spatial_index.contains_at(cell_id, cell.position) {
                return Err(InvariantError::SpatialIndexMismatch(cell_id));
            }
            if cell.genome.enzymes.len() < MIN_CELL_ENZYMES
                || cell.genome.enzymes.len() > MAX_CELL_ENZYMES
            {
                return Err(InvariantError::InvalidGenomeEnzymeCount(cell_id));
            }
            for enzyme in &cell.genome.enzymes {
                if enzyme.validate().is_err() {
                    return Err(InvariantError::InvalidCatalyst(cell_id));
                }
            }

            for element in ELEMENT_ORDER {
                let amount = cell.internal_elements[element];
                if [amount, cell.body[element], cell.boundary[element]]
                    .iter()
                    .any(|value| !value.is_finite() || *value < 0.0)
                {
                    return Err(InvariantError::CellElementReservoirMismatch {
                        cell: cell_id,
                        element,
                    });
                }
            }
            let tile_id = self
                .center_tile_id(cell.position)
                .ok_or(InvariantError::InvalidCellGeometry(cell_id))?;
            expected_tile_cell_counts[tile_id.index()] =
                expected_tile_cell_counts[tile_id.index()].saturating_add(1);
        }
        if self.spatial_index.len() != self.cells.len() {
            return Err(InvariantError::SpatialIndexCountMismatch {
                expected: self.cells.len(),
                actual: self.spatial_index.len(),
            });
        }
        for (tile_index, (expected, actual)) in expected_tile_cell_counts
            .iter()
            .zip(&self.tile_cell_counts)
            .enumerate()
        {
            if expected != actual {
                return Err(InvariantError::TileCellCountMismatch {
                    tile: TileId(tile_index),
                    expected: *expected,
                    actual: *actual,
                });
            }
        }

        let mut actual_lineage_population = BTreeMap::<LineageId, u64>::new();
        for cell in self.cells.iter() {
            *actual_lineage_population
                .entry(cell.lineage_id)
                .or_default() += 1;
        }
        for (lineage_id, counters) in &self.lineage_counters {
            let actual = actual_lineage_population
                .remove(lineage_id)
                .unwrap_or_default();
            if counters.population != actual {
                return Err(InvariantError::LineagePopulationMismatch {
                    lineage: *lineage_id,
                    expected: actual,
                    actual: counters.population,
                });
            }
        }
        if let Some((lineage, actual)) = actual_lineage_population.into_iter().next() {
            return Err(InvariantError::LineagePopulationMismatch {
                lineage,
                expected: actual,
                actual: 0,
            });
        }

        for (term, value) in self.energy_ledger.terms() {
            if !value.is_finite() || value < 0.0 {
                return Err(InvariantError::InvalidEnergyLedgerTerm { term, value });
            }
        }
        let actual = self.total_live_cell_energy();
        let expected = self.energy_ledger.expected_cell_energy();
        let tolerance = self.energy_ledger.closure_tolerance();
        if !actual.is_finite() || (actual - expected).abs() > tolerance {
            return Err(InvariantError::EnergyLedgerMismatch {
                expected,
                actual,
                tolerance,
            });
        }

        Ok(())
    }

    fn catalyst_upkeep_per_sec(&self, cell: &Cell) -> f64 {
        self.config.catalyst_upkeep_per_sec * cell.genome.enzymes.len() as f64
    }

    fn index_xy(&self, x: usize, y: usize) -> usize {
        x * self.height + y
    }

    #[cfg(test)]
    fn add_unledgered_energy(&mut self, cell_id: CellId, delta: f64) {
        self.cells[cell_id].energy += delta;
    }

    fn step_cells(&mut self) {
        self.cell_phase_scratch.clear();
        self.cell_phase_scratch.extend(self.cells.ids());
        if self.cell_phase_scratch.is_empty() {
            return;
        }
        self.refresh_enval_smoothed();
        for phase_index in 0..self.cell_phase_scratch.len() {
            let cell_id = self.cell_phase_scratch[phase_index];
            if !self.cells.contains(cell_id) {
                continue;
            }
            self.cell_phase_row = Some(phase_index);
            self.step_cell(cell_id);
        }
        self.cell_phase_row = None;
        self.enval_smoothed_current = false;
    }

    fn refresh_enval_smoothed(&mut self) {
        let (width, height) = (self.width, self.height);
        let radius = LOCAL_ENVAL_RADIUS;
        let padded_height = height + 2 * radius;
        self.enval_smoothing_scratch
            .resize(self.tile_count + padded_height, 0.0);
        self.enval_smoothed.resize(self.tile_count, 0.0);
        let (column_sums, padded) = self.enval_smoothing_scratch.split_at_mut(self.tile_count);
        for x in 0..width {
            let column = &self.enval[x * height..(x + 1) * height];
            for (value, source) in padded[radius..radius + height].iter_mut().zip(column) {
                *value = f64::from(*source);
            }
            for offset in 0..radius {
                padded[offset] = f64::from(column[(offset + height * radius - radius) % height]);
                padded[radius + height + offset] = f64::from(column[offset % height]);
            }
            let sums = &mut column_sums[x * height..(x + 1) * height];
            for (y, sum) in sums.iter_mut().enumerate() {
                *sum = padded[y] + padded[y + 1] + padded[y + 2] + padded[y + 3] + padded[y + 4];
            }
        }
        let scale = 1.0 / LOCAL_ENVAL_WINDOW_AREA as f64;
        for x in 0..width {
            let columns: [usize; LOCAL_ENVAL_WINDOW_DIAMETER] = std::array::from_fn(|offset| {
                (x + width * radius + offset - radius) % width * height
            });
            let smoothed = &mut self.enval_smoothed[x * height..(x + 1) * height];
            for (y, value) in smoothed.iter_mut().enumerate() {
                let sum = column_sums[columns[0] + y]
                    + column_sums[columns[1] + y]
                    + column_sums[columns[2] + y]
                    + column_sums[columns[3] + y]
                    + column_sums[columns[4] + y];
                *value = (sum * scale) as f32;
            }
        }
        self.enval_smoothed_current = true;
    }

    #[cfg(test)]
    fn smoothed_local_enval_at(&self, position: Position) -> Option<f32> {
        let stencil = self.tile_stencil(position)?;
        self.smoothed_local_enval_with(position, &stencil)
    }

    fn smoothed_local_enval_with(&self, position: Position, stencil: &TileStencil) -> Option<f32> {
        if !self.enval_smoothed_current {
            return self.local_enval_average_at(position, LOCAL_ENVAL_RADIUS);
        }
        let mut value = 0.0_f64;
        for (tile, weight) in stencil.tiles.into_iter().zip(stencil.weights) {
            value += f64::from(self.enval_smoothed[tile]) * f64::from(weight);
        }
        Some(value as f32)
    }

    fn step_cell(&mut self, cell_id: CellId) {
        let Some(slot) = self.cells.slot(cell_id) else {
            return;
        };
        let position = self.cells.at_slot(slot).position;
        self.operation_counters.cell_steps = self.operation_counters.cell_steps.saturating_add(1);
        self.operation_counters.local_enval_average_calls = self
            .operation_counters
            .local_enval_average_calls
            .saturating_add(1);
        let stencil = self.canonical_tile_stencil(position);
        let local_enval = self
            .smoothed_local_enval_with(position, &stencil)
            .unwrap_or(0.0);
        let enzyme_count = self.cells.at_slot(slot).genome.enzymes.len();
        let radius = self.cells.at_slot(slot).radius;
        let context = CellReactionContext {
            cell_id,
            slot,
            position,
            stencil,
            local_enval,
            cell_area: std::f32::consts::PI * radius * radius,
        };

        for catalyst_index in 0..enzyme_count {
            let Some(enzyme) = self
                .cells
                .at_slot(slot)
                .genome
                .enzymes
                .get(catalyst_index)
                .copied()
            else {
                break;
            };
            self.operation_counters.enzyme_entries_seen = self
                .operation_counters
                .enzyme_entries_seen
                .saturating_add(1);
            self.count_reaction_attempt();
            let env = ReactionEnv {
                capacity_fraction: 1.0 - f64::from(enzyme.exposure),
                ..ReactionEnv::internal(local_enval, context.cell_area, self.config.cell_body)
            };
            let reservoir = self.cells.at_slot(slot).internal_elements;
            let energy_before = self.cells.at_slot(slot).energy;
            let flux = bio::compute_flux(
                &enzyme,
                reservoir,
                energy_before,
                self.config.dt_seconds,
                env,
            );
            let Some(outcome) = flux.as_ref() else {
                self.reaction_counters.no_substrate =
                    self.reaction_counters.no_substrate.saturating_add(1);
                continue;
            };

            let cell = self.cells.at_slot_mut(slot);
            for element in ELEMENT_ORDER {
                let before = cell.internal_elements[element];
                let consumed = outcome.consumed[element];
                debug_assert!(consumed <= before + 1.0e-5);
                cell.internal_elements[element] = if consumed >= before {
                    0.0
                } else {
                    before - consumed
                };
            }
            self.apply_reaction_outcome(
                context,
                catalyst_index,
                &enzyme,
                ReactionSite::Internal,
                outcome,
                energy_before,
            );
        }

        let secretion = std::mem::replace(&mut self.pending_secretion, ElementAmounts::ZERO);
        if secretion != ElementAmounts::ZERO {
            self.deposit_elements_with(&stencil, secretion);
        }
        self.step_surface_catalysis(context, enzyme_count);
        self.apply_enval_exchange(&context);

        self.cells
            .at_slot_mut(slot)
            .refresh_radius(self.config.cell_body.body_density);
        self.max_cell_radius = self.max_cell_radius.max(self.cells.at_slot(slot).radius);
        let maintenance_loss = (self.cells.at_slot_mut(slot).maintenance_cost_per_area
            * self
                .cells
                .at_slot(slot)
                .area(self.config.cell_body.body_density)
            + self.catalyst_upkeep_per_sec(self.cells.at_slot(slot)))
            * self.config.dt_seconds;
        if maintenance_loss > 0.0 {
            self.energy_ledger.maintenance +=
                maintenance_loss.min(self.cells.at_slot_mut(slot).energy);
            self.cells.at_slot_mut(slot).energy -= maintenance_loss;
        }
        if self.cells.at_slot_mut(slot).energy <= 0.0 {
            self.cells.at_slot_mut(slot).energy = 0.0;
            self.energy_death_count += 1;
            self.kill_cell_and_release(cell_id);
            return;
        }

        let (inward, outward) = self.transport_elements_with(slot, &stencil);
        self.reaction_counters.uptake_flux += inward;
        self.reaction_counters.leak_flux += outward;

        let strain = self.cells[cell_id].strain(self.config.cell_body.boundary_density);
        if strain < self.config.cell_body.lysis_strain {
            self.lysis_count += 1;
            if self.cells[cell_id].last_digested_tick.is_some_and(|tick| {
                self.tick_count.saturating_sub(tick) <= LYSIS_ATTRIBUTION_TICKS
            }) {
                self.interaction_counters.lyses_after_digestion += 1;
            }
            self.kill_cell_and_release(cell_id);
        } else if strain >= self.config.cell_body.fission_strain {
            self.fission_cell(cell_id);
        }
    }

    fn count_reaction_attempt(&mut self) {
        self.operation_counters.metabolic_enzyme_attempts = self
            .operation_counters
            .metabolic_enzyme_attempts
            .saturating_add(1);
        self.reaction_counters.attempts = self.reaction_counters.attempts.saturating_add(1);
    }

    fn apply_reaction_outcome(
        &mut self,
        context: CellReactionContext,
        catalyst_index: usize,
        enzyme: &Enzyme,
        site: ReactionSite,
        outcome: &bio::FluxOutcome,
        energy_before: f64,
    ) {
        let position = context.position;
        let cell = self.cells.at_slot_mut(context.slot);
        for element in ELEMENT_ORDER {
            cell.internal_elements[element] += outcome.retained_products[element];
            cell.body[element] += outcome.body_products[element];
            cell.boundary[element] += outcome.boundary_products[element];
        }
        let energy_after = energy_before + outcome.energy_delta;
        cell.energy = if energy_after < 0.0 && energy_after > -1.0e-9 {
            0.0
        } else {
            energy_after
        };
        debug_assert!(cell.energy >= 0.0);
        let energy_after = cell.energy;
        if outcome.secreted_products != ElementAmounts::ZERO {
            if site == ReactionSite::Internal {
                for element in ELEMENT_ORDER {
                    self.pending_secretion[element] += outcome.secreted_products[element];
                }
            } else {
                self.deposit_elements_with(&context.stencil, outcome.secreted_products);
            }
        }
        if outcome.chemical_energy_delta >= 0.0 {
            self.energy_ledger.chemical_harvest += outcome.chemical_energy_delta;
        } else {
            self.energy_ledger.chemical_cost -= outcome.chemical_energy_delta;
        }
        self.energy_ledger.enval_harvest += outcome.enval_energy;
        self.energy_ledger.pump_cost += outcome.pump_cost;
        self.energy_ledger.assembly_cost += outcome.assembly_cost;
        self.pending_enval_uptake += f64::from(outcome.enval_input);
        self.pending_enval_emission += f64::from(outcome.enval_output);

        self.operation_counters.reactions_succeeded = self
            .operation_counters
            .reactions_succeeded
            .saturating_add(1);
        self.reaction_counters.successes = self.reaction_counters.successes.saturating_add(1);
        self.reaction_counters.energy_delta += outcome.energy_delta;
        self.reaction_counters.enval_input += f64::from(outcome.enval_input);
        self.reaction_counters.enval_output += f64::from(outcome.enval_output);
        self.reaction_counters.executed_metabolic_flux += f64::from(outcome.executed_extent);
        self.reaction_counters.secretion_flux += outcome.secreted_products.total();

        if self.watched_cell_id == Some(context.cell_id) {
            self.flux_log.push_flux_record(FluxRecord {
                tick_count: self.tick_count,
                sim_time_seconds: self.sim_time_seconds,
                cell_id: context.cell_id.index(),
                x: position.x,
                y: position.y,
                catalyst_index,
                site,
                reactants: *enzyme.reactants.as_array(),
                products: *enzyme.products.as_array(),
                requested_extent: outcome.requested_extent,
                executed_extent: outcome.executed_extent,
                element_deltas: outcome.element_deltas,
                secreted_elements: *outcome.secreted_products.as_array(),
                energy_before,
                energy_after,
                delta_cell_energy: outcome.energy_delta,
                raw_chemical_energy: outcome.raw_chemical_energy,
                enval_energy: outcome.enval_energy,
                enval_input: outcome.enval_input,
                enval_output: outcome.enval_output,
                local_enval: context.local_enval,
                enval_optimum: enzyme.enval_optimum,
            });
        }
    }

    fn collect_contacts(
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
        let mut push_contact = |other_id: CellId, other_position: Position| {
            let other_radius = f64::from(cells[other_id].radius);
            let delta = minimum_image_displacement(position, other_position, width, height);
            let distance = f64::from(delta.x).hypot(f64::from(delta.y));
            if distance <= f64::from(radius) + other_radius + f64::from(gap) {
                contacts.push(Contact {
                    cell_id: other_id,
                    distance,
                    share: interaction::perimeter_share(
                        f64::from(radius),
                        other_radius,
                        distance,
                        f64::from(gap),
                    ),
                });
            }
        };
        if let Some(row) = self.listed_row(cell_id) {
            let start = self.contact_row_offsets[row] as usize;
            let end = self.contact_row_offsets[row + 1] as usize;
            for &slot in &self.contact_row_neighbors[start..end] {
                if self.contact_displacements[slot as usize] > self.contact_loose_threshold {
                    continue;
                }
                let other_id = self.contact_row_owners[slot as usize];
                if let Some(other) = cells.get(other_id) {
                    push_contact(other_id, other.position);
                }
            }
            let first = self
                .contact_loose_extras
                .partition_point(|(extra_row, _)| (*extra_row as usize) < row);
            for &(extra_row, other_id) in &self.contact_loose_extras[first..] {
                if extra_row as usize != row {
                    break;
                }
                if let Some(other) = cells.get(other_id) {
                    push_contact(other_id, other.position);
                }
            }
        } else {
            self.spatial_index.find_within(
                position,
                radius + self.max_cell_radius + gap,
                |other_id, other_position| {
                    if other_id != cell_id {
                        push_contact(other_id, other_position);
                    }
                    false
                },
            );
        }
        contacts.sort_unstable_by_key(|contact| contact.cell_id.index());
        interaction::normalize_shares(contacts)
    }

    fn listed_row(&self, cell_id: CellId) -> Option<usize> {
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

    fn step_surface_catalysis(&mut self, context: CellReactionContext, enzyme_count: usize) {
        let cell = self.cells.at_slot(context.slot);
        if !cell.genome.enzymes.iter().any(|enzyme| enzyme.exposure > 0.0) {
            return;
        }
        let radius = cell.radius;
        self.operation_counters.surface_contact_queries = self
            .operation_counters
            .surface_contact_queries
            .saturating_add(1);
        let mut contacts = std::mem::take(&mut self.contact_scratch);
        let field_share =
            self.collect_contacts(context.cell_id, context.position, radius, &mut contacts);
        for catalyst_index in 0..enzyme_count {
            let Some(enzyme) = self
                .cells
                .at_slot(context.slot)
                .genome
                .enzymes
                .get(catalyst_index)
                .copied()
            else {
                break;
            };
            if !(enzyme.exposure > 0.0) {
                continue;
            }
            let exposure = f64::from(enzyme.exposure);
            if field_share > 0.0 {
                self.surface_reaction_on_field(
                    context,
                    catalyst_index,
                    &enzyme,
                    exposure * field_share,
                );
            }
            for contact in &contacts {
                if contact.share > 0.0 {
                    self.surface_reaction_on_cell(
                        context,
                        catalyst_index,
                        &enzyme,
                        exposure * contact.share,
                        contact.cell_id,
                    );
                }
            }
        }
        self.contact_scratch = contacts;
    }

    fn surface_reaction_on_field(
        &mut self,
        context: CellReactionContext,
        catalyst_index: usize,
        enzyme: &Enzyme,
        capacity_fraction: f64,
    ) {
        let stencil = context.stencil;
        let tile_indices = stencil.tiles;
        let mut weighted = [[0.0_f64; 4]; ELEMENT_COUNT];
        let mut pool = ElementAmounts::ZERO;
        for element in ELEMENT_ORDER {
            let mut total = 0.0_f64;
            for (corner, weight) in stencil.weights.iter().enumerate() {
                let amount = f64::from(*weight)
                    * f64::from(self.element_fields[tile_indices[corner]][element]);
                weighted[element.index()][corner] = amount;
                total += amount;
            }
            pool[element] = total as f32;
        }

        self.count_reaction_attempt();
        let env = ReactionEnv {
            capacity_fraction,
            pool_extent: Some(1.0),
            ..ReactionEnv::internal(context.local_enval, context.cell_area, self.config.cell_body)
        };
        let energy_before = self.cells.at_slot(context.slot).energy;
        let flux = bio::compute_flux(
            enzyme,
            pool,
            energy_before,
            self.config.dt_seconds,
            env,
        );
        let Some(outcome) = flux.as_ref() else {
            self.reaction_counters.no_substrate =
                self.reaction_counters.no_substrate.saturating_add(1);
            return;
        };

        for element in ELEMENT_ORDER {
            let consumed = f64::from(outcome.consumed[element]);
            let shares = weighted[element.index()];
            let total = shares.iter().sum::<f64>();
            if consumed <= 0.0 || total <= 0.0 {
                continue;
            }
            let last = (0..4)
                .rev()
                .find(|corner| shares[*corner] > 0.0)
                .expect("a positive pool has a contributing tile");
            let mut remaining = consumed;
            for corner in 0..4 {
                if shares[corner] <= 0.0 {
                    continue;
                }
                let wanted = if corner == last {
                    remaining
                } else {
                    consumed * shares[corner] / total
                };
                let tile = &mut self.element_fields[tile_indices[corner]][element];
                let amount = (wanted as f32).min(*tile).max(0.0);
                *tile -= amount;
                remaining -= f64::from(amount);
            }
        }
        self.interaction_counters.field_surface_extent += f64::from(outcome.executed_extent);
        self.apply_reaction_outcome(
            context,
            catalyst_index,
            enzyme,
            ReactionSite::Field,
            outcome,
            energy_before,
        );
    }

    fn surface_reaction_on_cell(
        &mut self,
        context: CellReactionContext,
        catalyst_index: usize,
        enzyme: &Enzyme,
        capacity_fraction: f64,
        target: CellId,
    ) {
        let Some(target_slot) = self.cells.slot(target) else {
            return;
        };
        debug_assert_ne!(target, context.cell_id);
        let target_cell = self.cells.at_slot(target_slot);
        let pool = target_cell.boundary;
        let perimeter = std::f32::consts::TAU * target_cell.radius;

        self.count_reaction_attempt();
        let env = ReactionEnv {
            capacity_fraction,
            pool_extent: Some(perimeter),
            ..ReactionEnv::internal(context.local_enval, context.cell_area, self.config.cell_body)
        };
        let energy_before = self.cells.at_slot(context.slot).energy;
        let flux = bio::compute_flux(
            enzyme,
            pool,
            energy_before,
            self.config.dt_seconds,
            env,
        );
        let Some(outcome) = flux.as_ref() else {
            self.reaction_counters.no_substrate =
                self.reaction_counters.no_substrate.saturating_add(1);
            return;
        };

        let tick_count = self.tick_count;
        let target_cell = self.cells.at_slot_mut(target_slot);
        let mut digested = 0.0_f64;
        for element in ELEMENT_ORDER {
            let before = target_cell.boundary[element];
            let consumed = outcome.consumed[element];
            debug_assert!(consumed <= before + 1.0e-5);
            target_cell.boundary[element] = if consumed >= before {
                0.0
            } else {
                before - consumed
            };
            digested += f64::from(consumed.min(before));
        }
        target_cell.last_digested_tick = Some(tick_count);
        let target_lineage = target_cell.lineage_id;
        self.interaction_counters.cell_surface_extent += f64::from(outcome.executed_extent);
        self.interaction_counters.boundary_digested += digested;
        if target_lineage != self.cells.at_slot(context.slot).lineage_id {
            self.interaction_counters.cross_lineage_boundary_digested += digested;
        }
        self.apply_reaction_outcome(
            context,
            catalyst_index,
            enzyme,
            ReactionSite::Cell(target),
            outcome,
            energy_before,
        );
    }

    #[cfg(test)]
    fn transport_elements(&mut self, slot: usize, position: Position) -> (f64, f64) {
        let Some(stencil) = self.tile_stencil(position) else {
            return (0.0, 0.0);
        };
        self.transport_elements_with(slot, &stencil)
    }

    fn transport_elements_with(&mut self, slot: usize, stencil: &TileStencil) -> (f64, f64) {
        let radius = f64::from(self.cells.at_slot_mut(slot).radius);
        let area = std::f64::consts::PI * radius * radius;
        let perimeter = 2.0 * std::f64::consts::PI * radius;
        let base_fraction =
            f64::from(self.config.membrane_permeability) * perimeter * self.config.dt_seconds
                / (2.0 * area);
        let membrane_polarity = self.cells.at_slot(slot).membrane_polarity();
        let sigma = f64::from(self.config.cell_body.permeability_sigma);
        let tile_indices = stencil.tiles;

        let mut inward = 0.0_f64;
        let mut outward = 0.0_f64;
        for element in ELEMENT_ORDER {
            let distance = f64::from(element.properties().polarity) - membrane_polarity;
            let permeability_factor =
                0.25 + 1.5 * (-distance * distance / (2.0 * sigma * sigma)).exp();
            let fraction = (base_fraction * permeability_factor).min(MAX_TRANSPORT_FRACTION);
            let mut weighted_sources = [0.0_f64; 4];
            for (corner, weight) in stencil.weights.iter().enumerate() {
                weighted_sources[corner] = f64::from(*weight)
                    * f64::from(self.element_fields[tile_indices[corner]][element]);
            }
            let outside = weighted_sources.iter().sum::<f64>();
            let held = self.cells.at_slot_mut(slot).internal_elements[element];
            let inside = f64::from(held) / area;
            let delta = fraction * (outside - inside) * area;
            if delta > 0.0 && outside > 0.0 {
                let mut moved = 0.0_f32;
                for corner in 0..4 {
                    if weighted_sources[corner] <= 0.0 {
                        continue;
                    }
                    let tile = &mut self.element_fields[tile_indices[corner]][element];
                    let amount = ((delta * weighted_sources[corner] / outside) as f32).min(*tile);
                    *tile -= amount;
                    moved += amount;
                }
                self.cells.at_slot_mut(slot).internal_elements[element] = held + moved;
                inward += f64::from(moved);
            } else if delta < 0.0 {
                let amount = ((-delta) as f32).min(held);
                if amount <= 0.0 {
                    continue;
                }
                self.cells.at_slot_mut(slot).internal_elements[element] = held - amount;
                for (corner, weight) in stencil.weights.iter().enumerate() {
                    self.element_fields[tile_indices[corner]][element] += amount * weight;
                }
                outward += f64::from(amount);
            }
        }
        if inward > 0.0 || outward > 0.0 {
            self.operation_counters.element_uptake_events = self
                .operation_counters
                .element_uptake_events
                .saturating_add(1);
        }
        (inward, outward)
    }

    fn apply_enval_exchange(&mut self, context: &CellReactionContext) {
        let uptake = std::mem::take(&mut self.pending_enval_uptake) as f32;
        let emission = std::mem::take(&mut self.pending_enval_emission) as f32;
        if uptake != 0.0 {
            if let Ok(applied) = self.adjust_enval_with(&context.stencil, -uptake) {
                self.enval_ledger.cell_uptake += applied;
            }
        }
        if emission != 0.0 {
            if let Ok(applied) = self.add_enval_around(context.position, emission) {
                self.enval_ledger.cell_emission += applied;
            }
        }
    }

    fn add_enval_around(
        &mut self,
        position: Position,
        enval_delta: f32,
    ) -> Result<f64, WorldError> {
        if !enval_delta.is_finite() || enval_delta == 0.0 {
            return Ok(0.0);
        }
        let choice = self.rng.usize(9);
        let target = Position::new(
            position.x + MOORE_WITH_CENTER_DX[choice] as f32,
            position.y + MOORE_WITH_CENTER_DY[choice] as f32,
        );
        self.adjust_enval_at(target, enval_delta)
    }

    fn fission_cell(&mut self, cell_id: CellId) {
        let parent = &self.cells[cell_id];
        if parent.strain(self.config.cell_body.boundary_density)
            < self.config.cell_body.fission_strain
            || parent.body.total() * 0.5 < MIN_DAUGHTER_BODY
        {
            return;
        }
        let fraction = 0.5 + (self.rng.next_f64() - 0.5) * 0.1;
        if parent.body.total() * fraction.min(1.0 - fraction) < MIN_DAUGHTER_BODY {
            return;
        }
        let angle = self.rng.range(0.0, std::f32::consts::TAU);
        let axis = Position::new(angle.cos(), angle.sin());
        let old_position = parent.position;
        let lineage = parent.lineage_id;
        let genome = parent.genome.mutate(&mut self.rng);
        let child_energy = parent.energy * fraction;
        let mut body = ElementAmounts::ZERO;
        let mut boundary = ElementAmounts::ZERO;
        let mut free = ElementAmounts::ZERO;
        let parent = &mut self.cells[cell_id];
        for element in ELEMENT_ORDER {
            body[element] = (f64::from(parent.body[element]) * fraction) as f32;
            boundary[element] = (f64::from(parent.boundary[element]) * fraction) as f32;
            free[element] = (f64::from(parent.internal_elements[element]) * fraction) as f32;
            parent.body[element] -= body[element];
            parent.boundary[element] -= boundary[element];
            parent.internal_elements[element] -= free[element];
        }
        parent.energy -= child_energy;
        parent.refresh_radius(self.config.cell_body.body_density);
        let parent_radius = parent.radius;
        let mut child = Cell::new(
            genome,
            old_position,
            self.sim_time_seconds,
            child_energy,
            body,
            boundary,
            self.config.cell_body.body_density,
        );
        child.internal_elements = free;
        let parent_position = Position::new(
            old_position.x - axis.x * child.radius,
            old_position.y - axis.y * child.radius,
        )
        .wrapped(self.width as f32, self.height as f32)
        .expect("finite fission position");
        child.position = Position::new(
            old_position.x + axis.x * parent_radius,
            old_position.y + axis.y * parent_radius,
        )
        .wrapped(self.width as f32, self.height as f32)
        .expect("finite fission position");
        self.update_cell_position(cell_id, parent_position);
        let child_position = child.position;
        let child_id = self.cells.insert(child);
        assert!(self.spatial_index.insert(child_id, child_position));
        self.increment_tile_cell_count(child_position);
        self.record_lineage_birth(lineage);
        self.birth_count += 1;
        self.operation_counters.cell_divisions += 1;
        self.reaction_counters.divisions += 1;
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
            |other_id, other_position| {
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

    fn kill_cell_and_release(&mut self, cell_id: CellId) {
        let Some(cell) = self.cells.get(cell_id) else {
            return;
        };
        let position = cell.position;
        let mut released_elements = cell.internal_elements;
        for element in ELEMENT_ORDER {
            released_elements[element] += cell.body[element] + cell.boundary[element];
        }
        self.deposit_elements_at(position, released_elements)
            .expect("active cell position must support conservative death release");
        let cell = self.remove_live_cell(cell_id);
        self.energy_ledger.death_loss += cell.energy;
        self.record_lineage_death(cell.lineage_id);
        self.death_count = self.death_count.saturating_add(1);
        self.operation_counters.cell_deaths = self.operation_counters.cell_deaths.saturating_add(1);
    }

    fn remove_live_cell(&mut self, cell_id: CellId) -> Cell {
        let cell = self
            .cells
            .remove(cell_id)
            .expect("removed cell id must refer to a live cell");
        assert!(self.spatial_index.remove(cell_id, cell.position));
        self.decrement_tile_cell_count(cell.position);
        cell
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

    fn refresh_max_cell_radius(&mut self) {
        self.max_cell_radius = self
            .cells
            .iter()
            .map(|cell| cell.radius)
            .fold(0.0, f32::max);
    }

    fn collect_contact_list(&mut self, reach: f32) -> (u64, u64) {
        let gap = self.config.surface.contact_gap;
        let (width, height) = (self.width, self.height);
        let (world_width, world_height) = (width as f32, height as f32);
        let cells = &self.cells;
        let tile_of = |position: Position| {
            debug_assert_eq!(position.wrapped(world_width, world_height), Some(position));
            (position.x as usize).min(width - 1) * height + (position.y as usize).min(height - 1)
        };
        let tile_offsets = &mut self.contact_tile_offsets;
        let tile_cells = &mut self.contact_tile_cells;
        let mut found = std::mem::take(&mut self.contact_found);
        found.clear();
        let mut contacts = 0_u64;
        let mut cross_lineage = 0_u64;
        let binned = if cells.len() < 2 { 0 } else { cells.len() };
        if binned > 0 {
            tile_offsets.clear();
            tile_offsets.resize(self.tile_count + 1, 0);
            for cell in cells.iter() {
                tile_offsets[tile_of(cell.position)] += 1;
            }
            let mut running = 0_u32;
            for offset in tile_offsets[..self.tile_count].iter_mut() {
                running += *offset;
                *offset = running;
            }
            tile_offsets[self.tile_count] = running;
            tile_cells.clear();
            tile_cells.resize(cells.len(), BinnedCell::default());
            for (slot, cell) in cells.iter().enumerate() {
                let tile = tile_of(cell.position);
                tile_offsets[tile] -= 1;
                tile_cells[tile_offsets[tile] as usize] = BinnedCell {
                    position: cell.position,
                    radius: cell.radius,
                    slot: slot as u32,
                };
            }
        }
        for slot in 0..binned {
            let cell = cells.at_slot(slot);
            let (position, radius) = (cell.position, cell.radius);
            let range = 2.0 * radius + reach + CONTACT_LIST_RANGE_EPSILON;
            let (first_column, columns) = tile_span(cell.position.x, range, width);
            let (first_row, rows) = tile_span(cell.position.y, range, height);
            for column_offset in 0..columns {
                let column = wrap_tile(first_column + column_offset as isize, width);
                for row_offset in 0..rows {
                    let tile = column * height + wrap_tile(first_row + row_offset as isize, height);
                    let start = tile_offsets[tile] as usize;
                    let end = tile_offsets[tile + 1] as usize;
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
                                if cells.at_slot(other_slot).lineage_id != cell.lineage_id {
                                    cross_lineage += 1;
                                }
                            }
                        }
                    }
                }
            }
        }

        let count = cells.len();
        self.contact_row_owners.clear();
        self.contact_row_owners.extend(cells.ids());
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

    fn add_overlapping_pairs_of_loose_cells(&mut self, threshold: f32) {
        let (width, height) = (self.width as f32, self.height as f32);
        let cells = &self.cells;
        for slot in 0..cells.len() {
            if self.contact_displacements[slot] <= threshold {
                continue;
            }
            let cell = cells.at_slot(slot);
            let pairs = &mut self.contact_pairs;
            self.spatial_index.find_within(
                cell.position,
                cell.radius + self.max_cell_radius,
                |other_id, other_position| {
                    let Some(other_slot) = cells.slot(other_id) else {
                        return false;
                    };
                    if other_slot == slot {
                        return false;
                    }
                    let separation = cell.radius + cells.at_slot(other_slot).radius;
                    if toroidal_distance_squared(cell.position, other_position, width, height)
                        <= separation * separation
                    {
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

    fn collect_loose_extras(&mut self, threshold: f32, reach: f32) {
        let (width, height) = (self.width as f32, self.height as f32);
        let cells = &self.cells;
        let displacements = &self.contact_displacements;
        let extras = &mut self.contact_loose_extras;
        extras.clear();
        for slot in 0..cells.len() {
            if displacements[slot] <= threshold {
                continue;
            }
            let cell = cells.at_slot(slot);
            self.spatial_index.find_within(
                cell.position,
                cell.radius + self.max_cell_radius + reach,
                |other_id, other_position| {
                    if other_id == cell.id {
                        return false;
                    }
                    let Some(other_slot) = cells.slot(other_id) else {
                        return false;
                    };
                    if displacements[other_slot] > threshold {
                        return false;
                    }
                    let separation = cell.radius + cells.at_slot(other_slot).radius + reach;
                    if toroidal_distance_squared(cell.position, other_position, width, height)
                        <= separation * separation
                    {
                        extras.push((other_slot as u32, cell.id));
                    }
                    false
                },
            );
        }
        extras.sort_unstable_by_key(|(row, cell_id)| (*row, cell_id.index()));
        self.contact_loose_threshold = threshold;
    }

    fn begin_unsynced_moves(&mut self) {
        if self.spatial_sync_pending {
            return;
        }
        self.indexed_positions.clear();
        self.indexed_positions
            .extend(self.cells.iter().map(|cell| cell.position));
        self.spatial_sync_pending = true;
    }

    fn move_cell_unsynced(&mut self, slot: usize, position: Position) {
        let old_tile = self.canonical_tile_index(self.cells.at_slot(slot).position);
        let new_tile = self.canonical_tile_index(position);
        if old_tile != new_tile {
            let count = &mut self.tile_cell_counts[old_tile];
            assert!(*count > 0, "derived tile cell count underflow");
            *count -= 1;
            self.tile_cell_counts[new_tile] = self.tile_cell_counts[new_tile].saturating_add(1);
        }
        self.cells.at_slot_mut(slot).position = position;
    }

    fn sync_spatial_index(&mut self) {
        if !self.spatial_sync_pending {
            return;
        }
        for (slot, indexed) in self.indexed_positions.iter().enumerate() {
            let cell = self.cells.at_slot(slot);
            if cell.position != *indexed {
                assert!(
                    self.spatial_index
                        .update_position(cell.id, *indexed, cell.position)
                );
            }
        }
        self.spatial_sync_pending = false;
    }

    #[cfg(test)]
    fn drift_cells(&mut self) {
        self.drift_cells_unsynced();
        self.sync_spatial_index();
    }

    fn drift_cells_unsynced(&mut self) {
        let diffusivity = f64::from(self.config.cell_motion.diffusivity);
        if !(diffusivity > 0.0) {
            return;
        }
        self.begin_unsynced_moves();
        let reference_radius = (f64::from(REFERENCE_CELL_AREA) / std::f64::consts::PI).sqrt();
        let (width, height) = (self.width as f32, self.height as f32);
        for slot in 0..self.cells.len() {
            let cell = self.cells.at_slot(slot);
            let position = cell.position;
            let cell_diffusivity = diffusivity * reference_radius / f64::from(cell.radius);
            let sigma = (2.0 * cell_diffusivity * self.config.dt_seconds).sqrt();
            let dx = self.rng.gaussian(0.0, 1.0) * sigma;
            let dy = self.rng.gaussian(0.0, 1.0) * sigma;
            let Some(moved) =
                Position::new(position.x + dx as f32, position.y + dy as f32).wrapped(width, height)
            else {
                continue;
            };
            self.move_cell_unsynced(slot, moved);
        }
    }

    fn resolve_overlaps(&mut self) {
        self.refresh_max_cell_radius();
        let gap = self.config.surface.contact_gap;
        let skin = gap * CONTACT_LIST_SKIN_GAPS;
        let (contacts, cross_lineage) = self.collect_contact_list(gap + skin);
        self.contact_pair_count = contacts;
        self.cross_lineage_contact_pair_count = cross_lineage;
        if self.cells.len() >= 2 {
            for pass in 0..OVERLAP_RELAXATION_PASSES {
                if pass > 0 {
                    self.sync_spatial_index();
                    self.add_overlapping_pairs_of_loose_cells(0.5 * (gap + skin));
                }
                if !self.overlap_pass() {
                    break;
                }
            }
        }
        self.sync_spatial_index();
        self.collect_loose_extras(0.25 * skin, gap + skin);
    }

    fn overlap_pass(&mut self) -> bool {
        let pairs = std::mem::take(&mut self.contact_pairs);
        self.operation_counters.spatial_candidate_checks = self
            .operation_counters
            .spatial_candidate_checks
            .saturating_add(pairs.len() as u64);
        self.operation_counters.overlap_candidates = self
            .operation_counters
            .overlap_candidates
            .saturating_add(pairs.len() as u64);
        self.mechanics_corrections.clear();
        self.mechanics_corrections
            .resize(self.cells.len(), Position::default());
        let mut corrected = false;
        for &(slot_a, slot_b) in &pairs {
            let (slot_a, slot_b) = (slot_a as usize, slot_b as usize);
            let a = self.cells.at_slot(slot_a);
            let b = self.cells.at_slot(slot_b);
            let (cell_a, cell_b) = (a.id, b.id);
            let displacement = minimum_image_displacement(
                a.position,
                b.position,
                self.width as f32,
                self.height as f32,
            );
            let distance_squared = displacement
                .x
                .mul_add(displacement.x, displacement.y * displacement.y);
            let required_distance = a.radius + b.radius;
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
                    let direction = directions[(cell_a.index().wrapping_mul(31)
                        ^ cell_b.index().wrapping_mul(17))
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
            self.mechanics_corrections[slot_a].x -= direction_x * half;
            self.mechanics_corrections[slot_a].y -= direction_y * half;
            self.mechanics_corrections[slot_b].x += direction_x * half;
            self.mechanics_corrections[slot_b].y += direction_y * half;
            self.operation_counters.overlap_corrections = self
                .operation_counters
                .overlap_corrections
                .saturating_add(1);
            corrected = true;
        }
        self.contact_pairs = pairs;
        if !corrected {
            return false;
        }
        self.begin_unsynced_moves();
        for slot in 0..self.cells.len() {
            let correction = self.mechanics_corrections[slot];
            if correction.x == 0.0 && correction.y == 0.0 {
                continue;
            }
            self.contact_displacements[slot] += correction.x.hypot(correction.y);
            let cell = self.cells.at_slot(slot);
            let Some(position) = Position::new(
                cell.position.x + correction.x,
                cell.position.y + correction.y,
            )
            .wrapped(self.width as f32, self.height as f32) else {
                continue;
            };
            self.move_cell_unsynced(slot, position);
        }
        true
    }
}

fn inspect_genome(genome: &Genome) -> GenomeDetailInspection {
    GenomeDetailInspection {
        mutation_rate: genome.mutation_rate,
        maintenance_cost_per_area: genome.maintenance_cost_per_area,
        lineage_id: genome.lineage_id,
        enzyme_count: genome.enzymes.len(),
        min_cell_enzymes: MIN_CELL_ENZYMES,
        max_cell_enzymes: MAX_CELL_ENZYMES,
        enzymes: genome
            .enzymes
            .iter()
            .enumerate()
            .map(|(index, enzyme)| inspect_enzyme(index, enzyme))
            .collect(),
    }
}

fn inspect_enzyme(index: usize, enzyme: &Enzyme) -> EnzymeDetailInspection {
    EnzymeDetailInspection {
        index,
        reactants: *enzyme.reactants.as_array(),
        products: *enzyme.products.as_array(),
        rate: enzyme.rate,
        half_saturation: enzyme.half_saturation,
        energy_harvest_fraction: enzyme.energy_harvest_fraction,
        secretion_fraction: enzyme.secretion_fraction,
        body_fraction: enzyme.body_fraction,
        boundary_fraction: enzyme.boundary_fraction,
        enval_optimum: enzyme.enval_optimum,
        enval_sigma: enzyme.enval_sigma,
        enval_throughput: enzyme.enval_throughput,
        enval_energy_fraction: enzyme.enval_energy_fraction,
        enval_release_fraction: enzyme.enval_release_fraction,
        enval_pump: enzyme.enval_pump,
        exposure: enzyme.exposure,
    }
}

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

fn percentile_sorted_f32(sorted_values: &[f32], fraction: f64) -> f32 {
    if sorted_values.is_empty() {
        return 0.0;
    }
    let max_index = sorted_values.len() - 1;
    let index = ((max_index as f64) * fraction.clamp(0.0, 1.0)).round() as usize;
    sorted_values[index.min(max_index)]
}

fn build_neighbors(width: usize, height: usize) -> Vec<NeighborIndices> {
    let mut neighbors = Vec::with_capacity(width * height);
    for x in 0..width {
        let left_x = if x == 0 { width - 1 } else { x - 1 };
        let right_x = if x + 1 == width { 0 } else { x + 1 };
        for y in 0..height {
            let up_y = if y == 0 { height - 1 } else { y - 1 };
            let down_y = if y + 1 == height { 0 } else { y + 1 };
            let idx = |x: usize, y: usize| TileId(x * height + y);
            neighbors.push(NeighborIndices {
                left: idx(left_x, y),
                right: idx(right_x, y),
                up: idx(x, up_y),
                down: idx(x, down_y),
                up_left: idx(left_x, up_y),
                up_right: idx(right_x, up_y),
                down_left: idx(left_x, down_y),
                down_right: idx(right_x, down_y),
            });
        }
    }
    neighbors
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

#[derive(Clone, Debug, PartialEq)]
pub enum InvariantError {
    MismatchedWorldArrayLengths,
    NonFiniteEnval(TileId),
    InvalidElementField {
        tile: TileId,
        element: Element,
    },
    InvalidNeighbor {
        tile: TileId,
        neighbor: TileId,
    },
    CellStoreMismatch,
    NonFiniteCellEnergy(CellId),
    InvalidCellGeometry(CellId),
    SpatialIndexMismatch(CellId),
    SpatialIndexCountMismatch {
        expected: usize,
        actual: usize,
    },
    TileCellCountMismatch {
        tile: TileId,
        expected: u32,
        actual: u32,
    },
    InvalidGenomeEnzymeCount(CellId),
    InvalidCatalyst(CellId),
    CellElementReservoirMismatch {
        cell: CellId,
        element: Element,
    },
    LineagePopulationMismatch {
        lineage: LineageId,
        expected: u64,
        actual: u64,
    },
    InvalidEnergyLedgerTerm {
        term: &'static str,
        value: f64,
    },
    EnergyLedgerMismatch {
        expected: f64,
        actual: f64,
        tolerance: f64,
    },
}

impl fmt::Display for InvariantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MismatchedWorldArrayLengths => {
                f.write_str("world arrays have mismatched lengths")
            }
            Self::NonFiniteEnval(tile) => write!(f, "tile {} has non-finite enval", tile.index()),
            Self::InvalidElementField { tile, element } => write!(
                f,
                "tile {} has an invalid {} field value",
                tile.index(),
                element
            ),
            Self::InvalidNeighbor { tile, neighbor } => write!(
                f,
                "tile {} has invalid neighbor {}",
                tile.index(),
                neighbor.index()
            ),
            Self::CellStoreMismatch => f.write_str("live cell store ids, slots and index disagree"),
            Self::NonFiniteCellEnergy(cell) => {
                write!(f, "cell {} has invalid energy", cell.index())
            }
            Self::InvalidCellGeometry(cell) => write!(
                f,
                "cell {} has invalid or non-canonical continuous geometry",
                cell.index()
            ),
            Self::SpatialIndexMismatch(cell) => write!(
                f,
                "cell {} is missing or misplaced in the spatial index",
                cell.index()
            ),
            Self::SpatialIndexCountMismatch { expected, actual } => write!(
                f,
                "spatial index count mismatch: expected {expected}, got {actual}"
            ),
            Self::TileCellCountMismatch {
                tile,
                expected,
                actual,
            } => write!(
                f,
                "tile {} cell count mismatch: expected {expected}, got {actual}",
                tile.index()
            ),
            Self::InvalidGenomeEnzymeCount(cell) => write!(
                f,
                "cell {} genome enzyme count is outside [{}, {}]",
                cell.index(),
                MIN_CELL_ENZYMES,
                MAX_CELL_ENZYMES
            ),
            Self::InvalidCatalyst(cell) => {
                write!(
                    f,
                    "cell {} has an invalid catalyst definition",
                    cell.index()
                )
            }
            Self::CellElementReservoirMismatch { cell, element } => write!(
                f,
                "cell {} has an invalid continuous {} reservoir",
                cell.index(),
                element
            ),
            Self::LineagePopulationMismatch {
                lineage,
                expected,
                actual,
            } => write!(
                f,
                "lineage {} population mismatch: expected {}, got {}",
                lineage.raw(),
                expected,
                actual
            ),
            Self::InvalidEnergyLedgerTerm { term, value } => write!(
                f,
                "energy ledger term {term} must be finite and nonnegative, got {value}"
            ),
            Self::EnergyLedgerMismatch {
                expected,
                actual,
                tolerance,
            } => write!(
                f,
                "live cell energy {actual} does not match the energy ledger balance {expected} (residual {}, tolerance {tolerance})",
                actual - expected
            ),
        }
    }
}

impl Error for InvariantError {}

#[cfg(test)]
mod tests {
    use super::{
        BilinearStencil, GEOMETRY_TOLERANCE, MAX_TRANSPORT_FRACTION, MOORE_WITH_CENTER_DX,
        MOORE_WITH_CENTER_DY, RenderBuffers, TileId, World,
    };
    use crate::cell::CELL_FLUX_LOG_CAPACITY;
    use crate::chem::{ELEMENT_COUNT, ELEMENT_ORDER, Element, ElementAmounts};
    use crate::config::Config;
    use crate::genome::{
        Enzyme, EnzymeFieldPatch, EnzymePatchOperation, Genome, GenomeFieldPatch, GenomePatch,
        LineageId, MAX_CELL_ENZYMES, MIN_CELL_ENZYMES,
    };
    use crate::spatial::{
        Position, REFERENCE_CELL_RADIUS, minimum_image_displacement, toroidal_distance_squared,
    };

    fn live_ids(world: &World) -> Vec<crate::cell::CellId> {
        world.cells.ids().collect()
    }

    fn small_config(seed: &str, width: usize, height: usize) -> Config {
        Config {
            seed: seed.to_owned(),
            width,
            height,
            ..Config::default()
        }
    }

    fn only_a_config(seed: &str, width: usize, height: usize) -> Config {
        let mut config = small_config(seed, width, height);
        config.element_fields.initial_amounts = ElementAmounts::new([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        config.element_fields.heterogeneity = 0.0;
        config
    }

    fn inert_enzyme() -> Enzyme {
        let mut enzyme = Enzyme::founder_downhill();
        enzyme.enval_optimum = 50.0;
        enzyme.enval_sigma = 1.0e-4;
        enzyme
    }

    fn make_inert(genome: &mut Genome) {
        genome.enzymes = vec![inert_enzyme()];
    }

    fn set_enval_optimum(genome: &mut Genome, optimum: f32) {
        for enzyme in &mut genome.enzymes {
            enzyme.enval_optimum = optimum;
        }
    }

    fn assert_valid_reservoir(world: &World, cell_id: crate::cell::CellId) {
        let Some(cell) = world.cell(cell_id) else {
            return;
        };
        for element in ELEMENT_ORDER {
            assert!(cell.internal_elements[element].is_finite());
            assert!(cell.internal_elements[element] >= 0.0);
        }
    }

    #[test]
    fn watching_is_exclusive_resets_on_change_and_never_changes_trajectory() {
        let mut plain = World::new(small_config("watch-determinism", 16, 12)).unwrap();
        plain.spawn_founder_cells(8).unwrap();
        let mut watched = plain.clone();
        let first = live_ids(&watched)[0];
        let second = live_ids(&watched)[1];
        assert_eq!(
            watched
                .inspect_cell_detail(first, 64)
                .unwrap()
                .recent_fluxes
                .flux_count,
            0
        );
        for tick in 0..500 {
            plain.step();
            watched.step();
            if tick % 10 == 0 {
                let id = if tick < 250 { first } else { second };
                if watched.cell(id).is_some() {
                    watched.inspect_cell_detail(id, 64).unwrap();
                }
            }
            assert_eq!(
                bincode::serialize(&plain).unwrap(),
                bincode::serialize(&watched).unwrap()
            );
        }
        assert_eq!(plain.flux_log.flux_count(), 0);
        assert!(watched.flux_log.flux_count() > 0);
        assert!(
            watched
                .flux_log
                .fluxes_newest_first()
                .all(|record| record.cell_id == second.index())
        );
        assert_eq!(
            watched.inspect_cell_fluxes(first, 64).unwrap().flux_count,
            0
        );
        let bytes = crate::snapshot::to_bytes(&watched).unwrap();
        let mut restored = crate::snapshot::from_bytes(&bytes).unwrap();
        assert_eq!(restored.watched_cell_id, None);
        assert_eq!(restored.flux_log.flux_count(), 0);
        watched.step_many(100);
        restored.step_many(100);
        assert_eq!(
            bincode::serialize(&watched).unwrap(),
            bincode::serialize(&restored).unwrap()
        );
    }

    #[test]
    fn snapshot_cell_storage_is_under_one_point_five_kilobytes_per_live_cell() {
        let mut world = if let Ok(path) = std::env::var("MICROCOSM_REFERENCE_SNAPSHOT") {
            crate::snapshot::load_from_path(path).unwrap()
        } else {
            let mut world = World::new(small_config("body-snapshot-size", 64, 48)).unwrap();
            world.spawn_founder_cells(100).unwrap();
            world.inspect_cell_fluxes(live_ids(&world)[0], 64).unwrap();
            world.step_many(100);
            world
        };
        let populated = crate::snapshot::to_bytes(&world).unwrap().len();
        let count = world.cell_count();
        world.cells = crate::cell::CellStore::default();
        let empty = crate::snapshot::to_bytes(&world).unwrap().len();
        println!(
            "snapshot populated={populated} empty={empty} cells={count} bytes_per_cell={:.3}",
            (populated - empty) as f64 / count as f64
        );
        assert!(
            populated - empty <= 1_500 * count,
            "{} bytes / {count} cells",
            populated - empty
        );
    }

    #[test]
    fn mixed_radius_pairs_picking_and_placement_match_brute_force() {
        let mut world = World::new(small_config("mixed-radius", 32, 24)).unwrap();
        world.spawn_founder_cells(80).unwrap();
        for (index, cell) in world.cells.iter_mut().enumerate() {
            let scale = if index % 13 == 0 {
                36.0
            } else {
                0.25 + (index % 5) as f32 * 0.5
            };
            for element in ELEMENT_ORDER {
                cell.body[element] *= scale;
            }
            cell.refresh_radius(world.config.cell_body.body_density);
        }
        world.rebuild_derived_caches();
        let brute_force = |world: &World, reach: f32| {
            let mut expected = Vec::new();
            for a in world.cells.iter() {
                for b in world.cells.iter() {
                    if a.id.index() < b.id.index()
                        && toroidal_distance_squared(a.position, b.position, 32.0, 24.0)
                            <= (a.radius + b.radius + reach).powi(2)
                    {
                        expected.push((a.id, b.id));
                    }
                }
            }
            expected.sort_unstable_by_key(|(a, b)| (a.index(), b.index()));
            expected
        };
        for reach in [0.0, 0.1, 0.5] {
            let (contacts, _) = world.collect_contact_list(reach);
            let mut listed = world
                .contact_pairs
                .iter()
                .map(|&(a, b)| {
                    let (a, b) = (
                        world.cells.at_slot(a as usize).id,
                        world.cells.at_slot(b as usize).id,
                    );
                    if a.index() < b.index() {
                        (a, b)
                    } else {
                        (b, a)
                    }
                })
                .collect::<Vec<_>>();
            listed.sort_unstable_by_key(|(a, b)| (a.index(), b.index()));
            assert_eq!(listed, brute_force(&world, reach));
            assert!(world.contact_pairs.windows(2).all(|pair| pair[0] < pair[1]));
            if reach >= world.config.surface.contact_gap {
                assert_eq!(
                    contacts as usize,
                    brute_force(&world, world.config.surface.contact_gap).len()
                );
            }
            for (slot, owner) in world.contact_row_owners.iter().enumerate() {
                assert_eq!(world.cells.at_slot(slot).id, *owner);
                let start = world.contact_row_offsets[slot] as usize;
                let end = world.contact_row_offsets[slot + 1] as usize;
                let mut row = world.contact_row_neighbors[start..end]
                    .iter()
                    .map(|slot| world.contact_row_owners[*slot as usize])
                    .collect::<Vec<_>>();
                row.sort_unstable_by_key(|id| id.index());
                let mut expected = world
                    .contact_pairs
                    .iter()
                    .filter_map(|&(a, b)| {
                        let (a, b) = (a as usize, b as usize);
                        (a == slot).then_some(b).or((b == slot).then_some(a))
                    })
                    .map(|other| world.contact_row_owners[other])
                    .collect::<Vec<_>>();
                expected.sort_unstable_by_key(|id| id.index());
                assert_eq!(row, expected);
            }
            let capacity = world.contact_pairs.capacity();
            let pointer = world.contact_pairs.as_ptr();
            world.collect_contact_list(reach);
            assert_eq!(world.contact_pairs.capacity(), capacity);
            assert_eq!(world.contact_pairs.as_ptr(), pointer);
        }
        for index in 0..500 {
            let position = Position::new(
                (index * 17 % 320) as f32 * 0.1,
                (index * 13 % 240) as f32 * 0.1,
            );
            let expected = world
                .cells
                .iter()
                .filter_map(|cell| {
                    let distance = toroidal_distance_squared(position, cell.position, 32.0, 24.0);
                    (distance <= (cell.radius + GEOMETRY_TOLERANCE).powi(2))
                        .then_some((cell.id, distance))
                })
                .min_by(|(a, da), (b, db)| da.total_cmp(db).then_with(|| a.index().cmp(&b.index())))
                .map(|(id, _)| id);
            assert_eq!(world.pick_cell(position), expected);
            for radius in [0.1, 0.45, 2.5] {
                let clear = world.cells.iter().all(|cell| {
                    toroidal_distance_squared(position, cell.position, 32.0, 24.0)
                        >= (radius + cell.radius - GEOMETRY_TOLERANCE).max(0.0).powi(2)
                });
                assert_eq!(world.position_is_clear(position, radius, None), clear);
            }
        }
    }

    #[test]
    fn overlap_resolution_separates_disturbed_fission_daughters() {
        let mut world = World::new(small_config("daughter-overlap", 16, 12)).unwrap();
        let id = world.spawn_random_cell().unwrap().unwrap();
        for element in ELEMENT_ORDER {
            world.cells[id].boundary[element] *= 1.6;
        }
        world.fission_cell(id);
        let child_id = world.cells.ids().find(|other| *other != id).unwrap();
        let center = world.cells[id].position;
        world.update_cell_position(child_id, center);
        let rng_before = bincode::serialize(&world.rng).unwrap();
        for _ in 0..4 {
            world.resolve_overlaps();
        }
        let parent = &world.cells[id];
        let child = &world.cells[child_id];
        assert!(
            toroidal_distance_squared(parent.position, child.position, 16.0, 12.0).sqrt()
                >= parent.radius + child.radius - GEOMETRY_TOLERANCE
        );
        assert_eq!(bincode::serialize(&world.rng).unwrap(), rng_before);
        world.check_invariants().unwrap();
    }

    #[test]
    fn transport_equilibrium_preserves_body_size_for_mixed_areas() {
        for scale in [0.25, 1.0, 4.0, 16.0] {
            let mut world = World::new(only_a_config("body-transport", 16, 12)).unwrap();
            let genome = Genome::random_founder(&mut world.rng, 0.0);
            let position = Position::new(3.5, 4.5);
            let id = world
                .spawn_cell_with_genome_at_position(position, genome)
                .unwrap();
            for element in ELEMENT_ORDER {
                world.cells[id].body[element] *= scale;
            }
            world.cells[id].refresh_radius(world.config.cell_body.body_density);
            let radius = world.cells[id].radius;
            let body = world.cells[id].body;
            let boundary = world.cells[id].boundary;
            let total_before = world.stats().system_element_amounts;
            let rng_before = bincode::serialize(&world.rng).unwrap();
            for _ in 0..2_000 {
                world.transport_elements(world.cells.slot(id).unwrap(), position);
            }
            let inside =
                f64::from(world.cells[id].internal_elements[Element::A]) / cell_area(&world, id);
            let outside = f64::from(world.sample_element_fields_at(position).unwrap()[Element::A]);
            assert!(
                (inside - outside).abs() < 1.0e-5,
                "scale={scale} inside={inside} outside={outside}"
            );
            assert_eq!(world.cells[id].radius, radius);
            assert_eq!(world.cells[id].body, body);
            assert_eq!(world.cells[id].boundary, boundary);
            assert_eq!(bincode::serialize(&world.rng).unwrap(), rng_before);
            for element in ELEMENT_ORDER {
                assert!(
                    (world.stats().system_element_amounts[element.index()]
                        - total_before[element.index()])
                    .abs()
                        < 1.0e-4
                );
            }
            world.check_invariants().unwrap();
        }
    }

    #[test]
    fn membrane_polarity_controls_element_permeability() {
        let mut world = World::new(small_config("polarity-transport", 8, 8)).unwrap();
        let id = world.spawn_random_cell().unwrap().unwrap();
        let position = world.cells[id].position;
        world.cells[id].boundary = ElementAmounts::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        for amounts in &mut world.element_fields {
            *amounts = ElementAmounts::new([1.0; 6]);
        }
        assert!((world.cells[id].membrane_polarity() - 0.9).abs() < 1.0e-6);
        world.transport_elements(world.cells.slot(id).unwrap(), position);
        let free = world.cells[id].internal_elements;
        assert!(free[Element::A] > free[Element::B]);
        assert!(free[Element::B] > free[Element::C]);
        assert!(free[Element::C] > free[Element::D]);
        world.cells[id].boundary = ElementAmounts::ZERO;
        assert_eq!(world.cells[id].membrane_polarity(), 0.5);
    }

    #[test]
    fn fission_conserves_pools_and_energy_places_touching_daughters_and_repeats_per_seed() {
        for seed in ["fission-1", "fission-2", "fission-3"] {
            let mut world = World::new(small_config(seed, 8, 8)).unwrap();
            let genome = Genome::random_founder(&mut world.rng, 0.0);
            let id = world
                .spawn_cell_with_genome_at_position(Position::new(0.05, 0.05), genome)
                .unwrap();
            for element in ELEMENT_ORDER {
                world.cells[id].boundary[element] *= 1.6;
            }
            world.cells[id].internal_elements = ElementAmounts::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
            let before = world.cells[id].clone();
            let field_before = world.element_field_totals();
            let mut repeated = world.clone();
            world.fission_cell(id);
            repeated.fission_cell(id);
            assert_eq!(
                bincode::serialize(&world).unwrap(),
                bincode::serialize(&repeated).unwrap()
            );
            assert_eq!(world.cell_count(), 2);
            assert_eq!(world.reaction_counters.divisions, 1);
            assert_eq!(world.element_field_totals(), field_before);
            let child = world.cells.iter().find(|cell| cell.id != id).unwrap();
            let parent = &world.cells[id];
            assert_eq!(parent.energy + child.energy, before.energy);
            assert_eq!(child.lineage_id, before.lineage_id);
            let distance =
                toroidal_distance_squared(parent.position, child.position, 8.0, 8.0).sqrt();
            assert!((distance - parent.radius - child.radius).abs() < 1.0e-5);
            for element in ELEMENT_ORDER {
                for (original, left, right) in [
                    (
                        before.body[element],
                        parent.body[element],
                        child.body[element],
                    ),
                    (
                        before.boundary[element],
                        parent.boundary[element],
                        child.boundary[element],
                    ),
                    (
                        before.internal_elements[element],
                        parent.internal_elements[element],
                        child.internal_elements[element],
                    ),
                ] {
                    assert!((left + right - original).abs() < 1.0e-6);
                }
            }
            world.check_invariants().unwrap();
        }
    }

    #[test]
    fn lysis_releases_all_material_and_records_remaining_energy() {
        let mut config = only_a_config("lysis", 8, 8);
        config.catalyst_upkeep_per_sec = 0.0;
        let mut world = World::new(config).unwrap();
        let mut genome = Genome::random_founder(&mut world.rng, 0.0);
        make_inert(&mut genome);
        genome.maintenance_cost_per_area = 0.0;
        let id = world
            .spawn_cell_with_genome_at_position(Position::new(2.5, 3.5), genome)
            .unwrap();
        for element in ELEMENT_ORDER {
            world.cells[id].boundary[element] *= 0.6;
        }
        world.cells[id].internal_elements = ElementAmounts::new([1.0; 6]);
        let before = world.stats().system_element_amounts;
        let energy = world.cells[id].energy;
        world.step_cell(id);
        assert_eq!(world.cell_count(), 0);
        assert_eq!(world.lysis_count, 1);
        assert_eq!(world.energy_death_count, 0);
        assert_eq!(world.energy_ledger.death_loss, energy);
        for element in ELEMENT_ORDER {
            assert!(
                (world.element_field_totals()[element.index()] - before[element.index()]).abs()
                    < 1.0e-5
            );
        }
        world.check_invariants().unwrap();
    }

    #[test]
    fn fission_requires_membrane_and_minimum_daughter_material() {
        let mut world = World::new(small_config("fission-minimum", 8, 8)).unwrap();
        let id = world.spawn_random_cell().unwrap().unwrap();
        world.set_all_live_cell_energy(1_000_000.0).unwrap();
        world.fission_cell(id);
        assert_eq!(world.cell_count(), 1);
        for element in ELEMENT_ORDER {
            world.cells[id].body[element] *= 0.01;
        }
        world.cells[id].refresh_radius(world.config.cell_body.body_density);
        world.fission_cell(id);
        assert_eq!(world.cell_count(), 1);
        world.check_invariants().unwrap();
    }

    #[test]
    fn assembly_is_ledgered_and_closes_over_two_thousand_ticks() {
        let mut world = World::new(small_config("assembly-ledger", 16, 12)).unwrap();
        world.spawn_founder_cells(8).unwrap();
        for cell in world.cells.iter_mut() {
            for enzyme in &mut cell.genome.enzymes {
                enzyme.body_fraction = 0.2;
                enzyme.boundary_fraction = 0.2;
            }
        }
        for _ in 0..2_000 {
            world.step();
            world.check_invariants().unwrap();
        }
        assert!(world.energy_ledger.assembly_cost > 0.0);
    }

    #[test]
    fn founders_conserve_every_element_and_start_at_reference_geometry() {
        let mut world = World::new(small_config("body-founders", 64, 48)).unwrap();
        let before = world.element_field_totals();
        assert_eq!(world.spawn_founder_cells(32).unwrap(), 32);
        let after = world.stats().system_element_amounts;
        for element in ELEMENT_ORDER {
            assert!((before[element.index()] - after[element.index()]).abs() < 1.0e-4);
        }
        for cell in world.cells.iter() {
            assert!((cell.radius - REFERENCE_CELL_RADIUS).abs() < 1.0e-4);
            assert!((cell.strain(world.config.cell_body.boundary_density) - 1.0).abs() < 1.0e-4);
            assert_eq!(cell.energy, 2.0);
        }
        world.check_invariants().unwrap();
        let ids = live_ids(&world);
        for id in ids {
            world.kill_cell_and_release(id);
        }
        for element in ELEMENT_ORDER {
            assert!(
                (before[element.index()] - world.element_field_totals()[element.index()]).abs()
                    < 1.0e-4
            );
        }
        world.check_invariants().unwrap();
    }

    #[test]
    fn founders_reject_insufficient_material_without_changing_fields() {
        let mut config = only_a_config("poor-founder", 1, 1);
        config.element_fields.initial_amounts = ElementAmounts::new([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let mut world = World::new(config).unwrap();
        let before = world.element_field_totals();
        assert_eq!(world.spawn_founder_cells(3).unwrap(), 0);
        assert_eq!(world.element_field_totals(), before);
        assert_eq!(world.energy_ledger.founder_energy, 0.0);
        world.check_invariants().unwrap();
    }

    #[test]
    fn invariants_reject_invalid_structure_and_inconsistent_radius() {
        let mut world = World::new(small_config("invalid-body", 8, 8)).unwrap();
        let id = world.spawn_random_cell().unwrap().unwrap();
        let cell = world.cells[id].clone();
        world.cells[id].radius *= 2.0;
        assert!(world.check_invariants().is_err());
        world.cells[id] = cell.clone();
        world.cells[id].body = ElementAmounts::ZERO;
        assert!(world.check_invariants().is_err());
        world.cells[id] = cell.clone();
        world.cells[id].boundary[Element::A] = -1.0;
        assert!(world.check_invariants().is_err());
        world.cells[id] = cell;
        world.cells[id].body[Element::A] = f32::NAN;
        assert!(world.check_invariants().is_err());
    }

    #[test]
    fn world_has_width_times_height_tiles() {
        let world = World::new(small_config("tiles", 7, 5)).unwrap();
        assert_eq!(world.width(), 7);
        assert_eq!(world.height(), 5);
        assert_eq!(world.tile_count(), 35);
    }

    #[test]
    fn toroidal_neighbor_indexing_wraps_edges() {
        let world = World::new(only_a_config("neighbors", 3, 2)).unwrap();
        let origin = world.tile_id(0, 0).unwrap();
        let neighbors = world.neighbors(origin).unwrap();
        assert_eq!(neighbors.left, world.tile_id(2, 0).unwrap());
        assert_eq!(neighbors.right, world.tile_id(1, 0).unwrap());
        assert_eq!(neighbors.up, world.tile_id(0, 1).unwrap());
        assert_eq!(neighbors.down, world.tile_id(0, 1).unwrap());
        assert_eq!(neighbors.up_left, world.tile_id(2, 1).unwrap());
        assert_eq!(neighbors.down_right, world.tile_id(1, 1).unwrap());
    }

    #[test]
    fn continuous_element_fields_use_defaults_and_toroidal_tile_layout() {
        let world = World::new(only_a_config("element-field-defaults", 3, 2)).unwrap();
        for x in 0..world.width() {
            for y in 0..world.height() {
                let tile = world.tile_id(x, y).unwrap();
                assert_eq!(world.tile_xy(tile), Some((x, y)));
                assert_eq!(
                    world.tile_element_amounts(tile).unwrap(),
                    world.config.element_fields.initial_amounts
                );
            }
        }
        assert_eq!(world.wrapped_tile_id(-1, -1), world.tile_id(2, 1).unwrap());
    }

    #[test]
    fn continuous_element_field_diffusion_is_per_element_conservative_and_nonnegative() {
        let mut world = World::new(only_a_config("element-field-diffusion", 3, 3)).unwrap();
        for tile_index in 0..world.tile_count() {
            world
                .set_tile_element_amounts(TileId(tile_index), ElementAmounts::ZERO)
                .unwrap();
        }
        let source = world.tile_id(1, 1).unwrap();
        world
            .set_tile_element_amounts(source, ElementAmounts::new([9.0; 6]))
            .unwrap();
        let before = world.element_field_totals();

        world.diffuse_element_fields();

        let after = world.element_field_totals();
        let source_amounts = world.tile_element_amounts(source).unwrap();
        let neighbor_amounts = world
            .tile_element_amounts(world.tile_id(0, 0).unwrap())
            .unwrap();
        for element in ELEMENT_ORDER {
            let alpha =
                world.config.element_fields.diffusivities[element] * world.config.dt_seconds as f32;
            assert!((source_amounts[element] - (9.0 - 8.0 * alpha)).abs() <= 1.0e-6);
            assert!((neighbor_amounts[element] - alpha).abs() <= 1.0e-6);
            assert!((after[element.index()] - before[element.index()]).abs() <= 1.0e-5);
        }
        for tile_index in 0..world.tile_count() {
            let amounts = world.tile_element_amounts(TileId(tile_index)).unwrap();
            for element in ELEMENT_ORDER {
                assert!(amounts[element].is_finite());
                assert!(amounts[element] >= 0.0);
            }
        }
        world.check_invariants().unwrap();
    }

    #[test]
    fn continuous_element_field_diffusion_is_deterministic() {
        let mut first = World::new(only_a_config("element-field-determinism", 5, 4)).unwrap();
        let source = first.tile_id(4, 3).unwrap();
        first
            .set_tile_element_amounts(source, ElementAmounts::new([8.0, 0.0, 4.0, 2.0, 1.0, 0.5]))
            .unwrap();
        let mut second = first.clone();

        for _ in 0..20 {
            first.diffuse_element_fields();
            second.diffuse_element_fields();
        }

        assert_eq!(first.element_fields, second.element_fields);
        assert_eq!(first.element_field_totals(), second.element_field_totals());
    }

    #[test]
    fn zero_heterogeneity_reproduces_the_uniform_fields_bit_for_bit() {
        let mut config = small_config("uniform-fields", 40, 30);
        config.element_fields.heterogeneity = 0.0;
        let world = World::new(config.clone()).unwrap();
        let uniform = vec![config.element_fields.initial_amounts; world.tile_count()];
        assert_eq!(world.element_fields, uniform);
        assert_eq!(world.element_fields_next, uniform);
        let mut expected_rng = crate::rng::Rng::from_seed_str("uniform-fields");
        let expected_landscape =
            crate::environment::enval_landscape(&config.enval_landscape, 40, 30, &mut expected_rng);
        assert_eq!(world.enval_landscape(), expected_landscape.as_slice());
    }

    #[test]
    fn default_heterogeneous_fields_keep_totals_nonnegative_periodic_and_seeded() {
        let config = small_config("heterogeneous-fields", 96, 72);
        let world = World::new(config.clone()).unwrap();
        let totals = world.element_field_totals();
        for element in ELEMENT_ORDER {
            let expected = f64::from(config.element_fields.initial_amounts[element])
                * world.tile_count() as f64;
            assert!(
                (totals[element.index()] - expected).abs() <= 1.0e-5 * expected,
                "{element}: {} vs {expected}",
                totals[element.index()]
            );
            let at = |x: usize, y: usize| world.element_fields[world.index_xy(x, y)][element];
            let mut interior = 0.0_f32;
            for x in 0..world.width - 1 {
                for y in 0..world.height - 1 {
                    interior = interior
                        .max((at(x + 1, y) - at(x, y)).abs())
                        .max((at(x, y + 1) - at(x, y)).abs());
                }
            }
            let mut seam = 0.0_f32;
            for y in 0..world.height {
                seam = seam.max((at(0, y) - at(world.width - 1, y)).abs());
            }
            for x in 0..world.width {
                seam = seam.max((at(x, 0) - at(x, world.height - 1)).abs());
            }
            assert!(
                seam <= interior,
                "{element}: seam {seam} > interior {interior}"
            );
        }
        assert!(world.element_fields.iter().all(|amounts| {
            ELEMENT_ORDER
                .iter()
                .all(|element| amounts[*element].is_finite() && amounts[*element] >= 0.0)
        }));
        let d_values = world
            .element_fields
            .iter()
            .map(|amounts| amounts[Element::D])
            .collect::<Vec<_>>();
        let d_max = d_values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let d_min = d_values.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(d_max > 1.2 * d_min, "fields are not heterogeneous");

        let again = World::new(config).unwrap();
        assert_eq!(world.element_fields, again.element_fields);
        let other = World::new(small_config("heterogeneous-fields-other", 96, 72)).unwrap();
        assert_ne!(world.element_fields, other.element_fields);
        world.check_invariants().unwrap();
    }

    #[test]
    fn seeded_initialization_is_deterministic() {
        let a = World::new(small_config("deterministic", 10, 8))
            .unwrap()
            .stats();
        let b = World::new(small_config("deterministic", 10, 8))
            .unwrap()
            .stats();
        assert_eq!(
            a.extracellular_element_amounts,
            b.extracellular_element_amounts
        );
        assert_eq!(
            a.intracellular_element_amounts,
            b.intracellular_element_amounts
        );
        assert_eq!(a.system_element_amounts, b.system_element_amounts);
        assert_eq!(a.average_enval.to_bits(), b.average_enval.to_bits());
    }

    #[test]
    fn initial_enval_equals_the_landscape_targets_on_every_tile() {
        let world = World::new(small_config("enval-initial", 64, 48)).unwrap();
        assert!(world.average_enval().is_finite());
        assert_eq!(world.enval_landscape().len(), world.tile_count());
        for tile_index in 0..world.tile_count() {
            let tile = TileId(tile_index);
            let target = world.enval_landscape()[tile_index];
            assert_eq!(world.tile_enval(tile).unwrap().to_bits(), target.to_bits());
            assert_eq!(
                world
                    .inspect_tile(tile)
                    .unwrap()
                    .enval_landscape_target
                    .to_bits(),
                target.to_bits()
            );
        }
        let total = world
            .enval
            .iter()
            .map(|value| f64::from(*value))
            .sum::<f64>();
        assert_eq!(world.enval_ledger().initial_total, total);
    }

    #[test]
    fn uniform_enval_field_remains_unchanged_after_diffusion() {
        let mut world = World::new(only_a_config("uniform", 4, 4)).unwrap();
        world.set_all_enval(0.25).unwrap();
        world.diffuse_enval();
        for tile_index in 0..world.tile_count() {
            assert_eq!(
                world.tile_enval(TileId(tile_index)).unwrap().to_bits(),
                0.25_f32.to_bits()
            );
        }
    }

    #[test]
    fn nonuniform_enval_diffuses_and_local_average_wraps() {
        let mut world = World::new(only_a_config("nonuniform", 3, 3)).unwrap();
        world.set_all_enval(0.0).unwrap();
        let corner = world.tile_id(2, 2).unwrap();
        let origin_id = world.tile_id(0, 0).unwrap();
        world.set_tile_enval(corner, 9.0).unwrap();
        let average = world.local_enval_average(origin_id, 1).unwrap();
        assert!((average - 1.0).abs() < 1.0e-6);
        world.diffuse_enval();
        let origin = world.tile_enval(origin_id).unwrap();
        let source = world.tile_enval(corner).unwrap();
        assert!(origin > 0.0);
        assert!(source < 9.0);
        world.check_invariants().unwrap();
    }

    #[test]
    fn enval_values_remain_finite_after_repeated_diffusion() {
        let mut world = World::new(small_config("finite", 8, 6)).unwrap();
        let low = world.tile_id(0, 0).unwrap();
        let high = world.tile_id(7, 5).unwrap();
        world.set_tile_enval(low, 1.0).unwrap();
        world.set_tile_enval(high, -1.0).unwrap();
        for _ in 0..200 {
            world.diffuse_enval();
        }
        for tile_index in 0..world.tile_count() {
            assert!(world.tile_enval(TileId(tile_index)).unwrap().is_finite());
        }
    }

    #[test]
    fn cell_spawn_updates_stats() {
        let mut world = World::new(only_a_config("spawn", 8, 8)).unwrap();
        let spawned = world.spawn_founder_cells(4).unwrap();
        assert_eq!(spawned, 4);
        assert_eq!(world.stats().live_cell_count, 4);
        assert_eq!(world.stats().births, 4);
        world.check_invariants().unwrap();
    }

    #[test]
    fn overlapping_cell_position_is_rejected() {
        let mut world = World::new(only_a_config("occupancy", 4, 4)).unwrap();
        let tile = world.tile_id(0, 0).unwrap();
        let genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        world
            .spawn_cell_with_genome_at(tile, genome.clone())
            .unwrap();
        assert!(world.spawn_cell_with_genome_at(tile, genome).is_err());
    }

    #[test]
    fn continuous_uptake_is_deterministic_and_conservative() {
        let mut world = World::new(only_a_config("uptake", 4, 4)).unwrap();
        let tile = world.tile_id(1, 1).unwrap();
        let genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
        let before_field = world.tile_element_amounts(tile).unwrap().total();
        world.step_cell(cell_id);
        let internal = world.cells[cell_id].internal_elements.total();
        let after_field = world.tile_element_amounts(tile).unwrap().total();
        assert!(internal > 0.0);
        assert!((before_field - (after_field + internal)).abs() <= 1.0e-5);
        assert_valid_reservoir(&world, cell_id);
        world.check_invariants().unwrap();
    }

    fn transport_fraction(world: &World, cell_id: crate::cell::CellId) -> f64 {
        let radius = f64::from(world.cells[cell_id].radius);
        let area = std::f64::consts::PI * radius * radius;
        let perimeter = 2.0 * std::f64::consts::PI * radius;
        (f64::from(world.config.membrane_permeability) * perimeter * world.config.dt_seconds
            / (2.0 * area)
            * 1.75)
            .min(MAX_TRANSPORT_FRACTION)
    }

    fn cell_area(world: &World, cell_id: crate::cell::CellId) -> f64 {
        let radius = f64::from(world.cells[cell_id].radius);
        std::f64::consts::PI * radius * radius
    }

    #[test]
    fn transport_moves_the_gradient_formula_amount_in_one_tick() {
        let mut world = World::new(only_a_config("transport-formula", 4, 4)).unwrap();
        let tile = world.tile_id(1, 1).unwrap();
        let genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
        world.cells[cell_id].internal_elements[Element::A] = 0.1;
        let area = cell_area(&world, cell_id);
        let expected = transport_fraction(&world, cell_id)
            * (f64::from(world.tile_element_amounts(tile).unwrap()[Element::A]) - 0.1 / area)
            * area;

        let (inward, outward) =
            world.transport_elements(world.cells.slot(cell_id).unwrap(), Position::new(1.0, 1.0));

        assert!(
            (inward - expected).abs() <= 1.0e-6,
            "{inward} != {expected}"
        );
        assert_eq!(outward, 0.0);
        assert!(
            (f64::from(world.cells[cell_id].internal_elements[Element::A]) - (0.1 + expected))
                .abs()
                <= 1.0e-6
        );
    }

    #[test]
    fn transport_equilibrates_in_a_uniform_field_without_overshoot() {
        let mut world = World::new(only_a_config("transport-equilibrium", 6, 6)).unwrap();
        let genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        let position = Position::new(2.0, 3.0);
        let cell_id = world
            .spawn_cell_with_genome_at_position(position, genome)
            .unwrap();
        let area = cell_area(&world, cell_id);
        let mut previous_inside = 0.0_f64;
        for _ in 0..400 {
            world.transport_elements(world.cells.slot(cell_id).unwrap(), position);
            let inside = f64::from(world.cells[cell_id].internal_elements[Element::A]) / area;
            let outside = f64::from(world.sample_element_fields_at(position).unwrap()[Element::A]);
            assert!(inside >= previous_inside - 1.0e-9, "concentration fell");
            assert!(
                inside <= outside + 1.0e-6,
                "overshoot: {inside} > {outside}"
            );
            previous_inside = inside;
        }
        let outside = f64::from(world.sample_element_fields_at(position).unwrap()[Element::A]);
        assert!((previous_inside - outside).abs() <= 1.0e-4 * outside);
        for element in [Element::B, Element::C, Element::D, Element::E, Element::F] {
            assert_eq!(world.cells[cell_id].internal_elements[element], 0.0);
        }
    }

    #[test]
    fn transport_leaks_outward_from_a_richer_cell_without_overshoot() {
        let mut world = World::new(only_a_config("transport-leak", 6, 6)).unwrap();
        let genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        let position = Position::new(2.5, 2.5);
        let cell_id = world
            .spawn_cell_with_genome_at_position(position, genome)
            .unwrap();
        world.cells[cell_id].internal_elements =
            ElementAmounts::new([5.0, 3.0, 0.0, 0.0, 0.0, 0.0]);
        let area = cell_area(&world, cell_id);
        let field_before = world.element_field_totals();

        let (inward, outward) =
            world.transport_elements(world.cells.slot(cell_id).unwrap(), position);

        let cell = &world.cells[cell_id];
        assert_eq!(inward, 0.0);
        assert!(outward > 0.0);
        assert!(cell.internal_elements[Element::A] < 5.0);
        assert!(cell.internal_elements[Element::B] < 3.0);
        let field_after = world.element_field_totals();
        for element in [Element::A, Element::B] {
            let inside = f64::from(cell.internal_elements[element]) / area;
            let outside = f64::from(world.sample_element_fields_at(position).unwrap()[element]);
            assert!(inside >= outside, "leak overshot equilibrium");
            assert!(field_after[element.index()] > field_before[element.index()]);
        }
        world.check_invariants().unwrap();
    }

    #[test]
    fn transport_amount_is_the_same_on_a_tile_centre_and_across_a_torus_seam() {
        let moved_at = |position: Position| {
            let mut world = World::new(only_a_config("transport-seam", 8, 6)).unwrap();
            let genome = Genome::random_founder(&mut world.rng, world.avg_enval);
            let cell_id = world
                .spawn_cell_with_genome_at_position(position, genome)
                .unwrap();
            for amounts in &mut world.element_fields {
                *amounts = world.config.element_fields.initial_amounts;
            }
            let before = world.element_field_totals();
            let (inward, _) =
                world.transport_elements(world.cells.slot(cell_id).unwrap(), position);
            let after = world.element_field_totals();
            assert!(
                (before[Element::A.index()] - after[Element::A.index()] - inward).abs() <= 1.0e-5
            );
            inward
        };
        let centre = moved_at(Position::new(3.0, 2.0));
        let seam = moved_at(Position::new(7.5, 5.5));
        let corner_seam = moved_at(Position::new(7.9, 0.2));
        assert!(centre > 0.0);
        assert!((centre - seam).abs() <= 1.0e-6, "{centre} != {seam}");
        assert!(
            (centre - corner_seam).abs() <= 1.0e-6,
            "{centre} != {corner_seam}"
        );
    }

    #[test]
    fn transport_consumes_no_rng() {
        let mut world = World::new(small_config("transport-rng", 6, 6)).unwrap();
        let genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        let position = Position::new(1.3, 4.7);
        let cell_id = world
            .spawn_cell_with_genome_at_position(position, genome)
            .unwrap();
        world.cells[cell_id].internal_elements =
            ElementAmounts::new([0.0, 4.0, 0.0, 0.2, 0.0, 1.0]);
        let mut untouched = world.rng.clone();
        for _ in 0..10 {
            world.transport_elements(world.cells.slot(cell_id).unwrap(), position);
        }
        assert_eq!(
            world.rng.next_f64().to_bits(),
            untouched.next_f64().to_bits()
        );
    }

    #[test]
    fn transport_conserves_total_elements_over_many_ticks_with_many_cells() {
        let mut world = World::new(small_config("transport-conservation", 32, 24)).unwrap();
        world.spawn_founder_cells(40).unwrap();
        let initial = world.stats().total_element_amount;
        for tick in 1..=1_000 {
            world.step();
            if tick % 100 == 0 {
                let total = world.compact_stats().total_element_amount;
                assert!(
                    (total - initial).abs() <= 2.0e-5 * initial,
                    "tick {tick}: {total} vs {initial}"
                );
            }
        }
        assert!(world.stats().reaction_counters.uptake_flux > 0.0);
        assert!(world.stats().reaction_counters.leak_flux > 0.0);
        world.check_invariants().unwrap();
    }

    #[test]
    fn fractional_field_sampling_deposition_and_seam_wrapping_are_conservative() {
        let mut world = World::new(only_a_config("fractional-field-coupling", 4, 3)).unwrap();
        for amounts in &mut world.element_fields {
            *amounts = ElementAmounts::ZERO;
        }
        for (x, y, amount) in [(3, 2, 1.0), (0, 2, 2.0), (3, 0, 3.0), (0, 0, 4.0)] {
            let tile = world.tile_id(x, y).unwrap();
            world.element_fields[tile.index()][Element::A] = amount;
        }
        let position = Position::new(3.5, 2.5);
        let sampled = world.sample_element_fields_at(position).unwrap();
        assert!((sampled[Element::A] - 2.5).abs() <= 1.0e-6);

        let before = world.element_field_totals()[Element::A.index()];
        world
            .deposit_elements_at(
                position,
                ElementAmounts::new([8.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            )
            .unwrap();
        let after = world.element_field_totals()[Element::A.index()];
        assert!((after - before - 8.0).abs() <= 1.0e-6);
        for (x, y, initial) in [(3, 2, 1.0), (0, 2, 2.0), (3, 0, 3.0), (0, 0, 4.0)] {
            let tile = world.tile_id(x, y).unwrap();
            assert!(
                (world.element_fields[tile.index()][Element::A] - (initial + 2.0)).abs() <= 1.0e-6
            );
        }
    }

    #[test]
    fn invalid_environmental_deposition_is_explicit_and_non_mutating() {
        let mut world = World::new(only_a_config("invalid-deposition", 4, 3)).unwrap();
        let before = world.element_fields.clone();
        let invalid_position = Position::new(f32::NAN, 1.0);

        let result = world.deposit_elements_at(
            invalid_position,
            ElementAmounts::new([8.0, 7.0, 6.0, 5.0, 4.0, 3.0]),
        );

        assert!(matches!(
            result,
            Err(super::WorldError::InvalidCellPosition(position)) if position.x.is_nan()
        ));
        assert_eq!(world.element_fields, before);
    }

    #[test]
    fn fractional_uptake_is_nonnegative_and_conservative() {
        let mut world = World::new(only_a_config("fractional-uptake", 4, 4)).unwrap();
        let tile = world.tile_id(1, 1).unwrap();
        let genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
        for amounts in &mut world.element_fields {
            *amounts = ElementAmounts::ZERO;
        }
        for (x, y, amount) in [(1, 1, 0.1), (2, 1, 0.2), (1, 2, 0.3), (2, 2, 0.4)] {
            let tile = world.tile_id(x, y).unwrap();
            world.element_fields[tile.index()][Element::A] = amount;
        }
        let position = Position::new(1.5, 1.5);
        let before = world.element_field_totals()[Element::A.index()];

        let (transferred, _) =
            world.transport_elements(world.cells.slot(cell_id).unwrap(), position);

        let after = world.element_field_totals()[Element::A.index()];
        assert!(transferred > 0.0);
        assert!((before - after - transferred).abs() <= 1.0e-6);
        assert!(world.element_fields.iter().all(|amounts| {
            ELEMENT_ORDER
                .iter()
                .all(|element| amounts[*element].is_finite() && amounts[*element] >= 0.0)
        }));
    }

    #[test]
    fn fractional_enval_sampling_average_and_scatter_preserve_locality() {
        let mut world = World::new(only_a_config("fractional-enval", 6, 6)).unwrap();
        world.set_all_enval(0.0).unwrap();
        world
            .set_tile_enval(world.tile_id(1, 1).unwrap(), 4.0)
            .unwrap();
        assert!((world.sample_enval_at(Position::new(1.5, 1.5)).unwrap() - 1.0).abs() <= 1.0e-6);

        let position = Position::new(1.5, 1.5);
        let manual = (-2..=2)
            .flat_map(|dx| (-2..=2).map(move |dy| (dx, dy)))
            .map(|(dx, dy)| {
                world
                    .sample_enval_at(Position::new(
                        position.x + dx as f32,
                        position.y + dy as f32,
                    ))
                    .unwrap()
            })
            .sum::<f32>()
            / 25.0;
        assert!((world.local_enval_average_at(position, 2).unwrap() - manual).abs() <= 1.0e-6);

        let before_sum = world.enval_sum;
        world.adjust_enval_at(position, 2.0).unwrap();
        assert!((world.enval_sum - before_sum - 2.0).abs() <= 1.0e-6);
        for (x, y) in [(1, 1), (2, 1), (1, 2), (2, 2)] {
            assert!(world.tile_enval(world.tile_id(x, y).unwrap()).unwrap() >= 0.5);
        }
    }

    #[test]
    fn smoothed_enval_grid_sampled_bilinearly_equals_the_twenty_five_sample_average() {
        for (seed, width, height, noise) in [
            ("smoothed-enval-default", 320, 240, 0.5),
            ("smoothed-enval-small", 7, 6, 2.0),
            ("smoothed-enval-narrow", 3, 4, 2.0),
            ("smoothed-enval-line", 1, 9, 2.0),
        ] {
            let mut world = World::new(small_config(seed, width, height)).unwrap();
            for tile in 0..world.tile_count() {
                let value = world.enval_landscape[tile] + world.rng.range(-noise, noise);
                world.set_tile_enval(TileId(tile), value).unwrap();
            }
            world.refresh_enval_smoothed();
            assert!(world.enval_smoothed_current);
            let mut checked = 0;
            for _ in 0..2_000 {
                let position = Position::new(
                    world.rng.range(0.0, width as f32),
                    world.rng.range(0.0, height as f32),
                );
                let smoothed = world.smoothed_local_enval_at(position).unwrap();
                let averaged = world.local_enval_average_at(position, 2).unwrap();
                assert!(
                    (smoothed - averaged).abs() <= 1.0e-6,
                    "{seed} at {position:?}: {smoothed} vs {averaged}"
                );
                checked += 1;
            }
            for x in 0..width {
                for y in 0..height {
                    let position = Position::new(x as f32, y as f32);
                    let averaged = world.local_enval_average_at(position, 2).unwrap();
                    let smoothed = world.smoothed_local_enval_at(position).unwrap();
                    assert!((smoothed - averaged).abs() <= 1.0e-6);
                }
            }
            assert_eq!(checked, 2_000);
        }
    }

    #[test]
    fn tile_stencils_match_bilinear_stencils_bit_for_bit() {
        for (width, height) in [(320, 240), (7, 5), (1, 3)] {
            let mut world = World::new(small_config("tile-stencil", width, height)).unwrap();
            let (w, h) = (width as f32, height as f32);
            let mut positions = vec![
                Position::new(0.0, 0.0),
                Position::new(w - 1.0e-4, h - 1.0e-4),
                Position::new(-1.0e-7, -3.5),
                Position::new(w, h),
                Position::new(2.0 * w + 0.25, -h - 0.75),
            ];
            for _ in 0..5_000 {
                positions.push(Position::new(
                    world.rng.range(-2.0 * w, 3.0 * w),
                    world.rng.range(-2.0 * h, 3.0 * h),
                ));
            }
            for position in positions {
                let expected = BilinearStencil::new(position, width, height).unwrap();
                let actual = world.tile_stencil(position).unwrap();
                for corner in 0..4 {
                    let sample = expected.samples[corner];
                    assert_eq!(actual.tiles[corner], world.index_xy(sample.x, sample.y));
                    assert_eq!(actual.weights[corner].to_bits(), sample.weight.to_bits());
                }
            }
            assert!(world.tile_stencil(Position::new(f32::NAN, 0.0)).is_none());
        }
    }

    #[test]
    fn cells_read_the_smoothed_enval_of_the_start_of_the_cell_phase() {
        let mut world = World::new(small_config("smoothed-enval-phase", 24, 18)).unwrap();
        world.spawn_founder_cells(6).unwrap();
        world.step_many(20);
        let before = world.clone();
        world.step_cells();
        assert!(!world.enval_smoothed_current);
        let mut expected = before.clone();
        expected.refresh_enval_smoothed();
        for cell in before.cells.iter() {
            let reference = before.local_enval_average_at(cell.position, 2).unwrap();
            let smoothed = expected.smoothed_local_enval_at(cell.position).unwrap();
            assert!((reference - smoothed).abs() <= 1.0e-6);
        }
        let last_cell = before.cells.at_slot(before.cells.len() - 1);
        let mut stepped = before.clone();
        stepped.inspect_cell_fluxes(last_cell.id, 64).unwrap();
        stepped.step_cells();
        assert!(stepped.enval != before.enval);
        let log = stepped.inspect_cell_fluxes(last_cell.id, 64).unwrap();
        let reference = expected
            .smoothed_local_enval_at(last_cell.position)
            .unwrap();
        assert!(!log.fluxes.is_empty());
        assert!(log.fluxes.iter().all(|flux| flux.local_enval == reference));
    }

    #[test]
    fn fast_local_enval_average_is_bit_identical_to_averaging_bilinear_samples() {
        let mut world = World::new(small_config("fast-local-enval", 23, 17)).unwrap();
        let mut rng = crate::rng::Rng::from_seed_str("fast-local-enval-values");
        for tile in 0..world.tile_count() {
            world
                .set_tile_enval(TileId(tile), rng.range(-2.0, 2.0))
                .unwrap();
        }
        let reference = |position: Position| {
            let mut sum = 0.0_f64;
            for dx in -2..=2 {
                for dy in -2..=2 {
                    let sample = Position::new(position.x + dx as f32, position.y + dy as f32);
                    sum += f64::from(world.sample_enval_at(sample).unwrap());
                }
            }
            (sum / 25.0) as f32
        };
        let mut positions = (0..2_000)
            .map(|_| Position::new(rng.range(0.0, 23.0), rng.range(0.0, 17.0)))
            .collect::<Vec<_>>();
        positions.extend([
            Position::new(0.0, 0.0),
            Position::new(22.999998, 16.999998),
            Position::new(1.99999, 1.99999),
            Position::new(0.99999, 2.00001),
            Position::new(5.0, 7.0),
        ]);
        for position in positions {
            let fast = world.local_enval_average_at(position, 2).unwrap();
            assert_eq!(
                fast.to_bits(),
                reference(position).to_bits(),
                "{position:?}"
            );
        }
    }

    #[test]
    fn continuous_enval_output_uses_exactly_one_offset_rng_draw() {
        let mut world = World::new(only_a_config("continuous-output-rng", 5, 5)).unwrap();
        world.set_all_enval(0.0).unwrap();
        let mut expected_rng = world.rng.clone();
        let _ = expected_rng.usize(9);

        world
            .add_enval_around(Position::new(1.25, 3.75), 1.0)
            .unwrap();

        assert_eq!(
            world.rng.next_f64().to_bits(),
            expected_rng.next_f64().to_bits()
        );
        assert!((world.enval_sum - 1.0).abs() <= 1.0e-6);
    }

    #[test]
    fn maintenance_can_kill_cell_and_release_continuous_elements() {
        let mut world = World::new(only_a_config("maintenance-death", 4, 4)).unwrap();
        let tile = world.tile_id(1, 1).unwrap();
        let mut genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        make_inert(&mut genome);
        world.config.founder_energy = 0.001;
        genome.maintenance_cost_per_area = 10.0;
        let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
        world.cells[cell_id].internal_elements =
            ElementAmounts::new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        let field_before = world.tile_element_amounts(tile).unwrap()[Element::B];
        world.step_cell(cell_id);
        assert!(world.cell(cell_id).is_none());
        assert_valid_reservoir(&world, cell_id);
        assert!(world.tile_element_amounts(tile).unwrap()[Element::B] >= field_before + 1.0);
        world.check_invariants().unwrap();
    }

    #[test]
    fn mismatched_enval_with_positive_energy_is_never_killed_by_time() {
        let mut world = World::new(only_a_config("energy-only-death-time", 6, 6)).unwrap();
        world.set_all_enval(0.9).unwrap();
        let tile = world.tile_id(2, 2).unwrap();
        let mut genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        genome.enzymes = vec![Enzyme::founder_downhill()];
        set_enval_optimum(&mut genome, -0.9);
        genome.maintenance_cost_per_area = 0.0;
        world.config.founder_energy = 1.0;
        let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
        world.cells[cell_id].birth_sim_time = -10_000.0;

        for _ in 0..5_000 {
            world.step_cell(cell_id);
            world.advance_time();
        }

        let cell = world.cell(cell_id).expect("cell must survive");
        assert!(cell.energy > 0.0);
        world.check_invariants().unwrap();
    }

    #[test]
    fn energy_exhaustion_kills_and_releases_the_reservoir_conservatively() {
        let mut world = World::new(only_a_config("energy-only-death-exhaustion", 4, 4)).unwrap();
        let tile = world.tile_id(1, 1).unwrap();
        let mut genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        make_inert(&mut genome);
        world.config.founder_energy = 0.0105;
        genome.maintenance_cost_per_area = 0.5 / f64::from(crate::spatial::REFERENCE_CELL_AREA);
        let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
        world.cells[cell_id].internal_elements =
            ElementAmounts::new([0.0, 0.4, 0.3, 0.2, 0.1, 0.05]);
        let total_before = world.stats().total_element_amount;

        world.step_cell(cell_id);
        world.step_cell(cell_id);
        assert!(world.cell(cell_id).is_some());
        world.step_cell(cell_id);

        assert!(world.cell(cell_id).is_none());
        let total_after = world.stats().total_element_amount;
        assert!((total_after - total_before).abs() <= 1.0e-5);
        let ledger = world.energy_ledger();
        assert!((ledger.maintenance - 0.0105).abs() <= 1.0e-12);
        assert_eq!(ledger.death_loss, 0.0);
        world.check_invariants().unwrap();
    }

    #[test]
    fn reaction_costs_that_exhaust_energy_kill_the_cell() {
        let mut world = World::new(only_a_config("energy-only-death-reaction", 4, 4)).unwrap();
        let tile = world.tile_id(1, 1).unwrap();
        let mut genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        let mut uphill = Enzyme::metabolic(
            ElementAmounts::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ElementAmounts::new([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            50.0,
            0.5,
            0.0,
        );
        uphill.enval_throughput = 0.0;
        uphill.enval_pump = 0.0;
        uphill.enval_sigma = 1000.0;
        genome.enzymes = vec![uphill];
        world.config.founder_energy = 0.01;
        genome.maintenance_cost_per_area = 0.0;
        let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
        world.cells[cell_id].internal_elements[Element::A] = 5.0;

        world.step_cell(cell_id);

        assert!(world.cell(cell_id).is_none());
        assert!(world.energy_ledger().chemical_cost > 0.0);
        world.check_invariants().unwrap();
    }

    #[test]
    fn metabolic_enval_response_uses_radius_two_local_average() {
        let mut world = World::new(only_a_config("flux-local-enval", 5, 5)).unwrap();
        world.set_all_enval(0.0).unwrap();
        let tile = world.tile_id(2, 2).unwrap();
        world.set_tile_enval(tile, 1.0).unwrap();
        let local = world.default_local_enval_average(tile).unwrap();
        assert!((local - 0.04).abs() <= 1.0e-6);
        let mut genome = Genome::random_founder(&mut world.rng, local);
        let mut catalyst = Enzyme::founder_downhill();
        catalyst.rate = 10.0;
        catalyst.enval_sigma = 0.01;
        catalyst.enval_throughput = 0.0;
        catalyst.enval_pump = 0.0;
        genome.enzymes = vec![catalyst];
        set_enval_optimum(&mut genome, local);
        genome.maintenance_cost_per_area = 0.0;
        let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
        world.cells[cell_id].internal_elements[Element::D] = 1.0;

        world.inspect_cell_fluxes(cell_id, 0).unwrap();
        world.step_cell(cell_id);

        let record = world.flux_log.latest_flux().unwrap();
        assert!((record.local_enval - local).abs() <= 1.0e-6);
        assert!(record.executed_extent > 0.09);
    }

    #[test]
    fn metabolic_enval_input_stays_local_and_output_uses_seeded_neighborhood_release() {
        let mut world = World::new(only_a_config("flux-enval-spatial", 5, 5)).unwrap();
        world.set_all_enval(1.0).unwrap();
        let tile = world.tile_id(2, 2).unwrap();
        let mut genome = Genome::random_founder(&mut world.rng, 1.0);
        let mut catalyst = Enzyme::founder_downhill();
        catalyst.rate = 10.0;
        catalyst.enval_sigma = 1000.0;
        catalyst.secretion_fraction = 0.0;
        genome.enzymes = vec![catalyst];
        set_enval_optimum(&mut genome, 1.0);
        genome.maintenance_cost_per_area = 0.0;
        let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
        world.cells[cell_id].internal_elements[Element::D] = 1.0;
        let before = world.enval.clone();
        let mut expected_rng = world.rng.clone();
        let release_choice = expected_rng.usize(9);
        let release_tile = world.wrapped_tile_id(
            2 + MOORE_WITH_CENTER_DX[release_choice],
            2 + MOORE_WITH_CENTER_DY[release_choice],
        );

        world.inspect_cell_fluxes(cell_id, 0).unwrap();
        world.step_cell(cell_id);

        let record = world.flux_log.latest_flux().unwrap();
        assert!(record.enval_input > 0.0);
        assert!(record.enval_output < 0.0);
        let expected_output_magnitude = record.enval_input.abs()
            * world.cells[cell_id].genome.enzymes[0].enval_release_fraction
            + world.cells[cell_id].genome.enzymes[0].enval_pump * record.executed_extent;
        assert!((record.enval_output.abs() - expected_output_magnitude).abs() <= 1.0e-6);
        for (index, before_value) in before.iter().copied().enumerate() {
            let mut expected = before_value;
            if index == tile.index() {
                expected -= record.enval_input;
            }
            if index == release_tile.index() {
                expected += record.enval_output;
            }
            assert!((world.enval[index] - expected).abs() <= 1.0e-6);
        }
        assert_eq!(
            world.rng.next_f64().to_bits(),
            expected_rng.next_f64().to_bits()
        );
    }

    #[test]
    fn a_cell_step_exchanges_its_net_enval_once_and_draws_one_release_offset() {
        let mut world = World::new(only_a_config("batched-enval-exchange", 7, 7)).unwrap();
        world.set_all_enval(0.8).unwrap();
        let tile = world.tile_id(3, 3).unwrap();
        let mut genome = Genome::random_founder(&mut world.rng, 0.8);
        let mut first = Enzyme::founder_downhill();
        first.rate = 10.0;
        first.enval_sigma = 1000.0;
        first.secretion_fraction = 0.0;
        let mut second = Enzyme::founder_engine_forward();
        second.rate = 10.0;
        second.enval_sigma = 1000.0;
        second.enval_pump = 0.1;
        let mut third = Enzyme::founder_engine_reverse();
        third.rate = 10.0;
        third.enval_sigma = 1000.0;
        genome.enzymes = vec![first, second, third];
        set_enval_optimum(&mut genome, 0.8);
        genome.maintenance_cost_per_area = 0.0;
        let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
        world.cells[cell_id].internal_elements =
            ElementAmounts::new([0.0, 0.5, 0.5, 0.5, 0.0, 0.0]);
        let before = world.enval.clone();
        let ledger_before = world.enval_ledger();
        let mut expected_rng = world.rng.clone();
        let release_choice = expected_rng.usize(9);
        let release_tile = world.wrapped_tile_id(
            3 + MOORE_WITH_CENTER_DX[release_choice],
            3 + MOORE_WITH_CENTER_DY[release_choice],
        );

        world.inspect_cell_fluxes(cell_id, 0).unwrap();
        world.step_cell(cell_id);

        let records = world
            .flux_log
            .fluxes_newest_first()
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(records.len(), 3);
        let input = records.iter().map(|record| record.enval_input).sum::<f32>();
        let output = records
            .iter()
            .map(|record| record.enval_output)
            .sum::<f32>();
        assert!(records.iter().all(|record| record.enval_input > 0.0));
        assert!(records.iter().all(|record| record.enval_output < 0.0));
        for (index, before_value) in before.iter().copied().enumerate() {
            let mut expected = before_value;
            if index == tile.index() {
                expected -= input;
            }
            if index == release_tile.index() {
                expected += output;
            }
            assert!(
                (world.enval[index] - expected).abs() <= 1.0e-6,
                "tile {index}"
            );
        }
        assert_eq!(
            world.rng.next_f64().to_bits(),
            expected_rng.next_f64().to_bits()
        );
        let ledger = world.enval_ledger();
        assert!(
            (ledger.cell_uptake - ledger_before.cell_uptake + f64::from(input)).abs() <= 1.0e-6
        );
        assert!(
            (ledger.cell_emission - ledger_before.cell_emission - f64::from(output)).abs()
                <= 1.0e-6
        );
        assert!(world.enval_ledger_residual().abs() <= 1.0e-5);

        let mut silent = World::new(only_a_config("batched-enval-silent", 7, 7)).unwrap();
        let mut genome = Genome::random_founder(&mut silent.rng, 0.0);
        for enzyme in &mut genome.enzymes {
            enzyme.enval_throughput = 0.0;
            enzyme.enval_pump = 0.0;
        }
        set_enval_optimum(&mut genome, 0.3);
        let cell_id = silent.spawn_cell_with_genome_at(tile, genome).unwrap();
        silent.cells[cell_id].internal_elements =
            ElementAmounts::new([0.0, 0.5, 0.5, 0.5, 0.0, 0.0]);
        let mut untouched_rng = silent.rng.clone();
        let enval_before = silent.enval.clone();
        silent.step_cell(cell_id);
        assert!(silent.reaction_counters.successes > 0);
        assert_eq!(silent.enval, enval_before);
        assert_eq!(
            silent.rng.next_f64().to_bits(),
            untouched_rng.next_f64().to_bits()
        );
    }

    #[test]
    fn active_cell_flux_logs_are_available_even_when_empty() {
        let mut world = World::new(only_a_config("reaction-log-empty", 4, 4)).unwrap();
        let tile = world.tile_id(1, 1).unwrap();
        let genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
        let logs = world.inspect_cell_fluxes(cell_id, 8).unwrap();
        assert!(logs.available);
        assert_eq!(logs.reason, "recorded");
        assert_eq!(logs.limit, 8);
        assert!(!logs.truncated);
        assert_eq!(logs.flux_count, 0);
        assert_eq!(logs.returned_count, 0);
        assert_eq!(logs.order, "newest_first");
        assert!(logs.fluxes.is_empty());
    }

    #[test]
    fn successful_continuous_fluxes_are_logged_and_bounded() {
        let mut world = World::new(only_a_config("reaction-log-bounded", 4, 4)).unwrap();
        let tile = world.tile_id(1, 1).unwrap();
        let mut genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        let mut catalyst = Enzyme::founder_downhill();
        catalyst.enval_sigma = 1000.0;
        catalyst.rate = 10.0;
        catalyst.body_fraction = 0.0;
        catalyst.boundary_fraction = 0.0;
        catalyst.secretion_fraction = 0.25;
        catalyst.enval_optimum = world.tile_enval(tile).unwrap();
        genome.enzymes = vec![catalyst];
        world.config.founder_energy = 10.0;
        genome.maintenance_cost_per_area = 0.0;
        let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
        world.cells[cell_id].internal_elements[Element::D] = 100.0;

        let target_fluxes = CELL_FLUX_LOG_CAPACITY + 5;
        for _ in 0..target_fluxes {
            world.inspect_cell_fluxes(cell_id, 0).unwrap();
            world.step_cell(cell_id);
        }

        assert_eq!(world.flux_log.flux_count(), CELL_FLUX_LOG_CAPACITY);
        let logs = world.inspect_cell_fluxes(cell_id, 2).unwrap();
        assert!(logs.available);
        assert_eq!(logs.reason, "recorded");
        assert_eq!(logs.limit, 2);
        assert_eq!(logs.flux_count, CELL_FLUX_LOG_CAPACITY);
        assert_eq!(logs.returned_count, 2);
        assert!(logs.truncated);
        assert_eq!(logs.order, "newest_first");
        assert_eq!(logs.fluxes.len(), 2);
        let record = &logs.fluxes[0];
        assert_eq!(record.cell_id, cell_id.index());
        assert_eq!(record.x, 1.0);
        assert_eq!(record.y, 1.0);
        assert_eq!(record.catalyst_index, 0);
        assert_eq!(record.site, crate::cell::ReactionSite::Internal);
        assert!(record.executed_extent > 0.0);
        assert!(record.secreted_elements[Element::A.index()] > 0.0);
        assert!(record.energy_after > record.energy_before);
        assert!(
            (record.delta_cell_energy - (record.energy_after - record.energy_before)).abs()
                <= 1.0e-12
        );

        let detail = world.inspect_cell_detail(cell_id, 3).unwrap();
        assert!(detail.recent_fluxes.available);
        assert_eq!(detail.recent_fluxes.returned_count, 3);
        assert_valid_reservoir(&world, cell_id);
        world.check_invariants().unwrap();
    }

    #[test]
    fn metabolic_flux_and_secretion_conserve_total_scalar_elements() {
        let mut world = World::new(only_a_config("flux-secretion-conservation", 4, 4)).unwrap();
        let tile = world.tile_id(1, 1).unwrap();
        let mut genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        let mut catalyst = Enzyme::founder_downhill();
        catalyst.rate = 10.0;
        catalyst.secretion_fraction = 0.5;
        catalyst.enval_sigma = 1000.0;
        genome.enzymes = vec![catalyst];
        genome.maintenance_cost_per_area = 0.0;
        set_enval_optimum(&mut genome, world.default_local_enval_average(tile).unwrap());
        let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
        world.cells[cell_id].internal_elements[Element::D] = 1.0;
        let a_before = world.element_fields[tile.index()][Element::A];
        let before = world.stats().total_element_amount;

        world.inspect_cell_fluxes(cell_id, 0).unwrap();
        world.step_cell(cell_id);

        let after = world.stats().total_element_amount;
        assert!((after - before).abs() <= 1.0e-5);
        let record = world.flux_log.latest_flux().unwrap();
        assert!(record.secreted_elements[Element::A.index()] > 0.0);
        assert!(
            world.element_fields[tile.index()][Element::A]
                + world.cells[cell_id].internal_elements[Element::A]
                > a_before
        );
        assert_valid_reservoir(&world, cell_id);
        world.check_invariants().unwrap();
    }

    #[test]
    fn genome_patch_updates_selected_cell_fields_and_enzymes() {
        let mut world = World::new(only_a_config("genome-patch-cell", 4, 4)).unwrap();
        let tile = world.tile_id(1, 1).unwrap();
        let genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
        let energy_before = world.cells[cell_id].energy;
        let lineage_before = world.cells[cell_id].lineage_id;
        let patch = GenomePatch {
            schema: Some(crate::genome::GENOME_PATCH_SCHEMA.to_owned()),
            genome: Some(GenomeFieldPatch {
                mutation_rate: Some(0.02),
                maintenance_cost_per_area: Some(0.11),
                ..GenomeFieldPatch::default()
            }),
            enzymes: vec![
                EnzymePatchOperation {
                    op: "update".to_owned(),
                    index: Some(0),
                    fields: Some(EnzymeFieldPatch {
                        enval_sigma: Some(0.44),
                        enval_throughput: Some(0.22),
                        rate: Some(0.55),
                        ..EnzymeFieldPatch::default()
                    }),
                    enzyme: None,
                },
                EnzymePatchOperation {
                    op: "append".to_owned(),
                    index: None,
                    fields: None,
                    enzyme: Some(EnzymeFieldPatch {
                        rate: Some(0.77),
                        ..EnzymeFieldPatch::default()
                    }),
                },
            ],
        };

        let result = world.apply_cell_genome_patch(cell_id, &patch).unwrap();
        let cell = &world.cells[cell_id];
        assert_eq!(result.patched_cell_count, 1);
        assert!(
            result
                .changed_fields
                .iter()
                .any(|field| field == "mutation_rate")
        );
        assert_eq!(cell.genome.mutation_rate, 0.02);
        assert_eq!(cell.maintenance_cost_per_area, 0.11);
        assert_eq!(cell.genome.maintenance_cost_per_area, 0.11);
        assert_eq!(cell.energy, energy_before);
        assert_eq!(cell.lineage_id, lineage_before);
        assert_eq!(cell.genome.lineage_id, lineage_before);
        assert_eq!(cell.genome.enzymes.last().unwrap().rate, 0.77);
        assert!(cell.genome.enzymes.len() >= 2);
        world.check_invariants().unwrap();
    }

    #[test]
    fn inspection_reports_each_enzymes_niche_and_rate_weighted_summaries() {
        let mut world = World::new(only_a_config("niche-inspection", 8, 8)).unwrap();
        world.set_all_enval(0.5).unwrap();
        let mut first = Enzyme::founder_downhill();
        first.rate = 1.0;
        first.enval_sigma = 1000.0;
        first.enval_optimum = 0.5;
        let mut second = Enzyme::founder_engine_forward();
        second.rate = 3.0;
        second.enval_sigma = 1000.0;
        second.enval_optimum = -0.5;
        let cell_id = spawn_with(&mut world, Position::new(2.5, 2.5), 7, vec![first, second]);
        let mut third = Enzyme::founder_downhill();
        third.rate = 4.0;
        third.enval_optimum = 1.0;
        spawn_with(&mut world, Position::new(5.5, 5.5), 7, vec![third]);
        world.cells[cell_id].internal_elements =
            ElementAmounts::new([0.0, 0.5, 0.0, 0.5, 0.0, 0.0]);

        let cell = world.inspect_cell(cell_id).unwrap();
        assert_eq!(cell.mean_enval_optimum, (0.5 - 1.5) / 4.0);
        assert_eq!(cell.enval_optimum_spread, 1.0);

        let lineage = world.inspect_lineage(LineageId(7)).unwrap();
        assert!((lineage.mean_enval_optimum - (0.5 - 1.5 + 4.0) / 8.0).abs() <= 1.0e-12);

        world.inspect_cell_fluxes(cell_id, 0).unwrap();
        world.step_cell(cell_id);
        let detail = world.inspect_cell_detail(cell_id, 8).unwrap();
        assert_eq!(detail.genome.enzymes[0].enval_optimum, 0.5);
        assert_eq!(detail.genome.enzymes[1].enval_optimum, -0.5);
        assert_eq!(detail.recent_fluxes.fluxes.len(), 2);
        for record in &detail.recent_fluxes.fluxes {
            let expected = detail.genome.enzymes[record.catalyst_index].enval_optimum;
            assert_eq!(record.enval_optimum, expected);
        }
    }

    #[test]
    fn invalid_genome_patch_is_rejected_atomically() {
        let mut world = World::new(only_a_config("genome-patch-invalid", 4, 4)).unwrap();
        let tile = world.tile_id(1, 1).unwrap();
        let genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
        let before = world.cells[cell_id].genome.clone();
        let patch = GenomePatch {
            schema: Some(crate::genome::GENOME_PATCH_SCHEMA.to_owned()),
            genome: Some(GenomeFieldPatch {
                mutation_rate: Some(1.5),
                ..GenomeFieldPatch::default()
            }),
            enzymes: Vec::new(),
        };

        assert!(world.apply_cell_genome_patch(cell_id, &patch).is_err());
        assert_eq!(world.cells[cell_id].genome, before);
        world.check_invariants().unwrap();
    }

    #[test]
    fn enzyme_patch_respects_min_and_max_bounds() {
        let mut world = World::new(only_a_config("genome-patch-bounds", 4, 4)).unwrap();
        let tile = world.tile_id(1, 1).unwrap();
        let mut genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        genome.enzymes = vec![Enzyme::founder_downhill()];
        let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
        let remove_patch = GenomePatch {
            schema: Some(crate::genome::GENOME_PATCH_SCHEMA.to_owned()),
            genome: None,
            enzymes: vec![EnzymePatchOperation {
                op: "remove".to_owned(),
                index: Some(0),
                fields: None,
                enzyme: None,
            }],
        };
        assert_eq!(world.cells[cell_id].genome.enzymes.len(), MIN_CELL_ENZYMES);
        assert!(
            world
                .apply_cell_genome_patch(cell_id, &remove_patch)
                .is_err()
        );

        while world.cells[cell_id].genome.enzymes.len() < MAX_CELL_ENZYMES {
            world.cells[cell_id]
                .genome
                .enzymes
                .push(Enzyme::founder_downhill());
        }
        let append_patch = GenomePatch {
            schema: Some(crate::genome::GENOME_PATCH_SCHEMA.to_owned()),
            genome: None,
            enzymes: vec![EnzymePatchOperation {
                op: "append".to_owned(),
                index: None,
                fields: None,
                enzyme: Some(EnzymeFieldPatch {
                    rate: Some(0.5),
                    ..EnzymeFieldPatch::default()
                }),
            }],
        };
        assert!(
            world
                .apply_cell_genome_patch(cell_id, &append_patch)
                .is_err()
        );
        assert_eq!(world.cells[cell_id].genome.enzymes.len(), MAX_CELL_ENZYMES);
    }

    #[test]
    fn genome_brush_applies_patch_to_active_cells_in_wrapped_rect() {
        let mut world = World::new(only_a_config("genome-patch-brush", 4, 4)).unwrap();
        let first_tile = world.tile_id(3, 3).unwrap();
        let second_tile = world.tile_id(0, 0).unwrap();
        let first_genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        let second_genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        let first = world
            .spawn_cell_with_genome_at(first_tile, first_genome)
            .unwrap();
        let second = world
            .spawn_cell_with_genome_at(second_tile, second_genome)
            .unwrap();
        let patch = GenomePatch {
            schema: Some(crate::genome::GENOME_PATCH_SCHEMA.to_owned()),
            genome: Some(GenomeFieldPatch {
                mutation_rate: Some(0.33),
                ..GenomeFieldPatch::default()
            }),
            enzymes: Vec::new(),
        };

        let result = world.apply_genome_brush(3, 3, 3, 3, &patch).unwrap();
        assert_eq!(result.visited_tile_count, 9);
        assert_eq!(result.patched_cell_count, 2);
        assert_eq!(world.cells[first].genome.mutation_rate, 0.33);
        assert_eq!(world.cells[second].genome.mutation_rate, 0.33);
        world.check_invariants().unwrap();
    }

    #[test]
    fn genome_brush_patches_every_unique_center_in_the_same_tile() {
        let mut world = World::new(only_a_config("genome-brush-multi-center", 5, 5)).unwrap();
        let first_genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        let first = world
            .spawn_cell_with_genome_at_position(Position::new(1.02, 1.02), first_genome)
            .unwrap();
        let second_genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        let second = world
            .spawn_cell_with_genome_at_position(Position::new(1.75, 1.75), second_genome)
            .unwrap();
        let patch = GenomePatch {
            schema: Some(crate::genome::GENOME_PATCH_SCHEMA.to_owned()),
            genome: Some(GenomeFieldPatch {
                mutation_rate: Some(0.44),
                ..GenomeFieldPatch::default()
            }),
            enzymes: Vec::new(),
        };

        let result = world.apply_genome_brush(1, 1, 1, 1, &patch).unwrap();

        assert_eq!(result.visited_tile_count, 1);
        assert_eq!(result.patched_cell_count, 2);
        assert_eq!(world.cells[first].genome.mutation_rate, 0.44);
        assert_eq!(world.cells[second].genome.mutation_rate, 0.44);
        assert_eq!(world.inspect_tile_xy(1, 1).unwrap().cell_center_count, 2);
    }

    #[test]
    fn geometric_pick_uses_toroidal_containment_nearest_center_and_id_tie_break() {
        let mut world = World::new(only_a_config("geometric-pick", 10, 8)).unwrap();
        let first_genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        let first = world
            .spawn_cell_with_genome_at_position(Position::new(1.0, 2.0), first_genome)
            .unwrap();
        let second_genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        let second = world
            .spawn_cell_with_genome_at_position(Position::new(1.9, 2.0), second_genome)
            .unwrap();

        assert_eq!(world.pick_cell(Position::new(1.1, 2.0)), Some(first));
        assert_eq!(world.pick_cell(Position::new(1.8, 2.0)), Some(second));
        world.cells[second].position = Position::new(1.875, 2.0);
        world.rebuild_spatial_index();
        assert_eq!(world.pick_cell(Position::new(1.4375, 2.0)), Some(first));
        assert_eq!(world.pick_cell(Position::new(5.0, 5.0)), None);

        world.cells[first].position = Position::new(0.1, 4.0);
        world.cells[second].position = Position::new(5.0, 4.0);
        world.rebuild_spatial_index();
        assert_eq!(world.pick_cell(Position::new(9.9, 4.0)), Some(first));
    }

    #[test]
    fn opposite_polarity_enzymes_in_one_cell_exchange_their_net_enval() {
        let mut world = World::new(only_a_config("mixed-polarity-cell", 7, 7)).unwrap();
        world.set_all_enval(0.5).unwrap();
        let tile = world.tile_id(3, 3).unwrap();
        let mut genome = Genome::random_founder(&mut world.rng, 0.5);
        let mut positive = Enzyme::founder_engine_forward();
        positive.rate = 10.0;
        positive.enval_sigma = 1000.0;
        positive.enval_pump = 0.2;
        positive.enval_optimum = 0.5;
        let mut negative = positive;
        negative.enval_optimum = -0.5;
        genome.enzymes = vec![positive, negative];
        genome.maintenance_cost_per_area = 0.0;
        let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
        world.cells[cell_id].internal_elements =
            ElementAmounts::new([0.0, 0.5, 0.0, 0.0, 0.0, 0.0]);
        let before = world.enval.clone();
        let ledger_before = world.enval_ledger();
        let mut expected_rng = world.rng.clone();
        let release_choice = expected_rng.usize(9);
        let release_tile = world.wrapped_tile_id(
            3 + MOORE_WITH_CENTER_DX[release_choice],
            3 + MOORE_WITH_CENTER_DY[release_choice],
        );

        world.inspect_cell_fluxes(cell_id, 0).unwrap();
        world.step_cell(cell_id);

        let records = world
            .flux_log
            .fluxes_newest_first()
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(records.len(), 2);
        let up = records.iter().find(|r| r.catalyst_index == 0).unwrap();
        let down = records.iter().find(|r| r.catalyst_index == 1).unwrap();
        assert_eq!(up.enval_optimum, 0.5);
        assert_eq!(down.enval_optimum, -0.5);
        assert!(up.enval_input > 0.0);
        assert_eq!(down.enval_input, 0.0);
        assert!(up.enval_output < 0.0);
        assert!(down.enval_output > 0.0);

        let input = up.enval_input + down.enval_input;
        let output = up.enval_output + down.enval_output;
        for (index, before_value) in before.iter().copied().enumerate() {
            let mut expected = before_value;
            if index == tile.index() {
                expected -= input;
            }
            if index == release_tile.index() {
                expected += output;
            }
            assert!(
                (world.enval[index] - expected).abs() <= 1.0e-5,
                "tile {index}: {} != {expected}",
                world.enval[index]
            );
        }
        let ledger = world.enval_ledger();
        assert!(
            (ledger.cell_uptake - ledger_before.cell_uptake + f64::from(input)).abs() <= 1.0e-6
        );
        assert!(
            (ledger.cell_emission - ledger_before.cell_emission - f64::from(output)).abs()
                <= 1.0e-6
        );
        assert!(world.enval_ledger_residual().abs() <= 1.0e-5);
        assert_eq!(
            world.rng.next_f64().to_bits(),
            expected_rng.next_f64().to_bits(),
            "the two enzymes share one release draw"
        );
    }

    #[test]
    fn zero_optimum_cell_reacts_without_touching_the_enval_field_or_the_rng() {
        let mut world = World::new(only_a_config("zero-optimum-cell", 7, 7)).unwrap();
        world.set_all_enval(0.5).unwrap();
        let tile = world.tile_id(3, 3).unwrap();
        let mut genome = Genome::random_founder(&mut world.rng, 0.5);
        let mut catalyst = Enzyme::founder_downhill();
        catalyst.rate = 10.0;
        catalyst.enval_sigma = 1000.0;
        catalyst.enval_pump = 0.3;
        catalyst.enval_optimum = 0.0;
        genome.enzymes = vec![catalyst];
        genome.maintenance_cost_per_area = 0.0;
        let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
        world.cells[cell_id].internal_elements[Element::D] = 1.0;
        let before = world.enval.clone();
        let mut untouched_rng = world.rng.clone();

        world.inspect_cell_fluxes(cell_id, 0).unwrap();
        world.step_cell(cell_id);

        let record = world.flux_log.latest_flux().unwrap();
        assert!(record.executed_extent > 0.0);
        assert!(record.raw_chemical_energy > 0.0);
        assert_eq!(record.enval_optimum, 0.0);
        assert_eq!(record.enval_input, 0.0);
        assert_eq!(record.enval_output, 0.0);
        assert_eq!(world.enval, before);
        assert_eq!(world.energy_ledger().pump_cost, 0.0);
        assert_eq!(world.energy_ledger().enval_harvest, 0.0);
        assert_eq!(
            world.rng.next_f64().to_bits(),
            untouched_rng.next_f64().to_bits()
        );
    }

    #[test]
    fn surface_reactions_use_the_enzymes_own_optimum() {
        let mut config = only_a_config("surface-optimum", 8, 8);
        config.catalyst_upkeep_per_sec = 0.0;
        let mut world = World::new(config).unwrap();
        world.set_all_enval(0.5).unwrap();
        let mut catalyst = surface_enzyme(A_ONLY, E_ONLY, 2.0, 1.0);
        catalyst.enval_pump = 0.2;
        catalyst.enval_throughput = 0.2;
        let mut uncoupled = catalyst;
        catalyst.enval_optimum = 0.5;
        uncoupled.enval_optimum = 0.0;
        let coupled_id = spawn_with(&mut world, Position::new(1.5, 1.5), 1, vec![catalyst]);
        let uncoupled_id = spawn_with(&mut world, Position::new(5.5, 5.5), 2, vec![uncoupled]);
        for id in [coupled_id, uncoupled_id] {
            let tile = world.center_tile_id(world.cells[id].position).unwrap();
            world.element_fields[tile.index()][Element::A] = 5.0;
        }

        for (id, coupled) in [(coupled_id, true), (uncoupled_id, false)] {
            let before = world.energy_ledger();
            let context = reaction_context(&world, id);
            world.inspect_cell_fluxes(id, 0).unwrap();
            let enzyme = world.cells[id].genome.enzymes[0];
            world.surface_reaction_on_field(context, 0, &enzyme, 1.0);
            world.apply_enval_exchange(&context);
            let record = world.flux_log.latest_flux().unwrap();
            assert_eq!(record.site, super::ReactionSite::Field);
            assert!(record.executed_extent > 0.0);
            let ledger = world.energy_ledger();
            if coupled {
                assert!(record.enval_input > 0.0);
                assert!(record.enval_output < 0.0);
                assert!(ledger.pump_cost > before.pump_cost);
            } else {
                assert_eq!(record.enval_input, 0.0);
                assert_eq!(record.enval_output, 0.0);
                assert_eq!(ledger.pump_cost, before.pump_cost);
            }
        }
    }

    #[test]
    fn enval_coupling_changes_field() {
        let mut world = World::new(only_a_config("enval-coupling", 4, 4)).unwrap();
        let before = world.average_enval();
        let tile = world.tile_id(1, 1).unwrap();
        let mut genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        let mut catalyst = Enzyme::founder_downhill();
        catalyst.rate = 10.0;
        catalyst.enval_optimum = world.tile_enval(tile).unwrap();
        genome.enzymes = vec![catalyst];
        world.config.founder_energy = 10.0;
        let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
        world.cells[cell_id].internal_elements[Element::D] = 2.0;
        for _ in 0..5 {
            world.step_cell(cell_id);
        }
        assert_ne!(world.average_enval().to_bits(), before.to_bits());
        world.check_invariants().unwrap();
    }

    #[test]
    fn deterministic_runs_produce_identical_stats() {
        fn run_once() -> crate::stats::WorldStats {
            let mut world = World::new(small_config("stats-golden", 16, 12)).unwrap();
            world.spawn_founder_cells(8).unwrap();
            for _ in 0..80 {
                world.step();
                world.check_invariants().unwrap();
            }
            world.stats()
        }
        let a = run_once();
        let b = run_once();
        assert_eq!(a, b);
    }

    #[test]
    fn fixed_seed_cellular_runs_are_deterministic() {
        fn run_once() -> crate::stats::WorldStats {
            let mut world = World::new(small_config("golden-small-001", 16, 16)).unwrap();
            world.spawn_founder_cells(6).unwrap();
            for _ in 0..80 {
                world.step();
                world.check_invariants().unwrap();
            }
            world.stats()
        }
        let a = run_once();
        let b = run_once();
        assert_eq!(a.tick_count, b.tick_count);
        assert_eq!(a.live_cell_count, b.live_cell_count);
        assert_eq!(a.births, b.births);
        assert_eq!(a.deaths, b.deaths);
        assert_eq!(
            a.extracellular_element_amounts,
            b.extracellular_element_amounts
        );
        assert_eq!(
            a.intracellular_element_amounts,
            b.intracellular_element_amounts
        );
        assert_eq!(a.system_element_amounts, b.system_element_amounts);
        assert_eq!(a.average_enval.to_bits(), b.average_enval.to_bits());
    }

    #[test]
    fn invariants_hold_after_many_steps() {
        let mut world = World::new(small_config("many-steps", 12, 10)).unwrap();
        world.spawn_founder_cells(5).unwrap();
        for _ in 0..100 {
            world.step();
            world.check_invariants().unwrap();
        }
    }

    #[test]
    fn fixed_seed_steps_have_stable_summary_stats() {
        let mut world = World::new(small_config("golden-phase12", 16, 12)).unwrap();
        for _ in 0..25 {
            world.step();
        }
        let stats = world.stats();
        assert_eq!(stats.tick_count, 25);
        assert_eq!(stats.width, 16);
        assert_eq!(stats.height, 12);
        assert!(
            world
                .element_field_totals()
                .iter()
                .all(|total| *total > 0.0)
        );
        assert!(stats.average_enval.is_finite());
        world.check_invariants().unwrap();
    }

    #[test]
    fn render_buffers_have_expected_lengths_and_values() {
        let mut world = World::new(small_config("render-buffers", 8, 6)).unwrap();
        world.spawn_founder_cells(5).unwrap();
        let before = world.build_render_buffers();
        assert_eq!(before.tile_enval.len(), 48);
        assert_eq!(before.tile_cell_count.len(), 48);
        assert_eq!(before.tile_mass_density.len(), 48);
        assert_eq!(before.tile_total_elements.len(), 48);
        assert_eq!(before.tile_element_concentrations.len(), 48 * ELEMENT_COUNT);
        assert_eq!(before.cell_count(), world.stats().live_cell_count);
        assert_eq!(before.cell_point_data.len(), before.cell_count() * 4);
        assert_eq!(before.cell_rotation_data.len(), before.cell_count() * 4);
        assert_eq!(before.cell_scale_data.len(), before.cell_count() * 4);
        assert_eq!(before.cell_rgba.len(), before.cell_count() * 4);
        assert_eq!(before.cell_radius.len(), before.cell_count());
        assert!(before.tile_enval.iter().all(|value| value.is_finite()));
        assert!(before.cell_energy.iter().all(|value| value.is_finite()));
        assert_eq!(
            before
                .tile_cell_count
                .iter()
                .map(|count| *count as usize)
                .sum::<usize>(),
            before.cell_count()
        );
        assert!(
            before
                .cell_radius
                .iter()
                .all(|radius| *radius == REFERENCE_CELL_RADIUS)
        );
        assert!(
            before
                .cell_rotation_data
                .chunks_exact(4)
                .all(|rotation| rotation == [0.0, 0.0, 0.0, 1.0])
        );
        assert!(before.cell_scale_data.chunks_exact(4).all(|scale| {
            scale
                == [
                    REFERENCE_CELL_RADIUS,
                    REFERENCE_CELL_RADIUS,
                    REFERENCE_CELL_RADIUS,
                    0.0,
                ]
        }));

        world.step_many(5);
        let after = world.build_render_buffers();
        assert_eq!(after.tile_count(), world.tile_count());
        assert_eq!(after.cell_count(), world.stats().live_cell_count);
        assert!(after.render_epoch >= before.render_epoch);
        world.check_invariants().unwrap();
    }

    #[test]
    fn inspect_helpers_report_tile_and_cell_state() {
        let mut world = World::new(only_a_config("inspect-helpers", 4, 4)).unwrap();
        let tile = world.tile_id(2, 1).unwrap();
        let genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();

        let tile_info = world.inspect_tile(tile).unwrap();
        assert_eq!(tile_info.tile_id, tile);
        assert_eq!(tile_info.x, 2);
        assert_eq!(tile_info.y, 1);
        assert_eq!(tile_info.cell_center_count, 1);
        assert!(tile_info.enval.is_finite());

        let cell_info = world.inspect_cell(cell_id).unwrap();
        assert_eq!(cell_info.cell_id, cell_id);
        assert_eq!(cell_info.x, 2.0);
        assert_eq!(cell_info.y, 1.0);
        assert_eq!(cell_info.radius, REFERENCE_CELL_RADIUS);
        assert!(cell_info.energy.is_finite());
    }

    #[test]
    fn default_local_enval_average_matches_generic_radius_two_average() {
        let mut world = World::new(only_a_config("local-enval-fast-path", 7, 6)).unwrap();
        for x in 0..world.width() {
            for y in 0..world.height() {
                let tile_id = world.tile_id(x, y).unwrap();
                world
                    .set_tile_enval(tile_id, (x as f32 * 0.25) - (y as f32 * 0.10))
                    .unwrap();
            }
        }

        for x in 0..world.width() {
            for y in 0..world.height() {
                let tile_id = world.tile_id(x, y).unwrap();
                let fast = world.default_local_enval_average(tile_id).unwrap();
                let generic = world
                    .local_enval_average(tile_id, super::LOCAL_ENVAL_RADIUS)
                    .unwrap();
                assert_eq!(fast.to_bits(), generic.to_bits());
            }
        }
    }

    #[test]
    fn write_render_buffers_reuses_existing_allocations_and_matches_builder() {
        let mut world = World::new(small_config("render-buffer-reuse", 8, 6)).unwrap();
        world.spawn_founder_cells(4).unwrap();
        world.step_many(5);

        let expected = world.build_render_buffers();
        let mut actual = RenderBuffers::default();
        actual.tile_enval.reserve(128);
        actual.cell_point_data.reserve(128);
        actual.cell_rotation_data.reserve(128);
        actual.cell_scale_data.reserve(128);
        actual.cell_rgba.reserve(128);
        let reserved_tile_capacity = actual.tile_enval.capacity();
        let reserved_point_capacity = actual.cell_point_data.capacity();
        let reserved_rotation_capacity = actual.cell_rotation_data.capacity();
        let reserved_scale_capacity = actual.cell_scale_data.capacity();
        let reserved_color_capacity = actual.cell_rgba.capacity();
        world.write_render_buffers(&mut actual);

        assert_eq!(actual, expected);
        assert!(actual.tile_enval.capacity() >= reserved_tile_capacity);
        assert!(actual.cell_point_data.capacity() >= reserved_point_capacity);
        assert!(actual.cell_rotation_data.capacity() >= reserved_rotation_capacity);
        assert!(actual.cell_scale_data.capacity() >= reserved_scale_capacity);
        assert!(actual.cell_rgba.capacity() >= reserved_color_capacity);
    }

    #[test]
    fn step_many_matches_repeated_single_steps() {
        let mut many = World::new(small_config("step-many", 10, 8)).unwrap();
        many.spawn_founder_cells(4).unwrap();
        let mut repeated = World::new(small_config("step-many", 10, 8)).unwrap();
        repeated.spawn_founder_cells(4).unwrap();

        many.step_many(30);
        for _ in 0..30 {
            repeated.step();
        }

        assert_eq!(many.stats(), repeated.stats());
        many.check_invariants().unwrap();
        repeated.check_invariants().unwrap();
    }

    #[test]
    fn expanded_stats_compartment_counts_and_enzyme_histogram_are_consistent() {
        let mut world = World::new(small_config("expanded-stats", 8, 6)).unwrap();
        world.spawn_founder_cells(4).unwrap();
        world.step_many(10);
        let stats = world.stats();

        assert_eq!(
            stats.occupied_tile_count + stats.empty_tile_count,
            stats.tile_count
        );
        for element in ELEMENT_ORDER {
            let index = element.index();
            assert_eq!(
                stats.system_element_amounts[index],
                stats.extracellular_element_amounts[index]
                    + stats.intracellular_element_amounts[index]
            );
        }
        assert_eq!(
            stats.total_element_amount,
            stats.system_element_amounts.iter().sum::<f64>()
        );
        assert_eq!(stats.live_cell_count, world.cells.len());
        assert_eq!(
            stats.enzyme_count_histogram.iter().sum::<u64>(),
            stats.live_cell_count as u64
        );
        assert_eq!(
            stats.total_enzyme_count,
            live_ids(&world)
                .iter()
                .map(|cell_id| world.cells[*cell_id].genome.enzymes.len() as u64)
                .sum::<u64>()
        );
        assert!(stats.occupancy_fraction >= 0.0 && stats.occupancy_fraction <= 1.0);
        assert!(stats.average_cell_energy.is_finite());
        assert!(stats.enval_std_dev.is_finite());
        world.check_invariants().unwrap();
    }

    #[test]
    fn enval_stddev_reports_uniform_and_nonuniform_fields() {
        let mut world = World::new(only_a_config("enval-stddev", 4, 4)).unwrap();
        world.set_all_enval(0.5).unwrap();
        assert_eq!(world.stats().enval_std_dev.to_bits(), 0.0_f32.to_bits());

        let tile = world.tile_id(0, 0).unwrap();
        world.set_tile_enval(tile, -0.5).unwrap();
        assert!(world.stats().enval_std_dev > 0.0);
    }

    #[test]
    fn reaction_and_operation_counters_are_deterministic() {
        fn run_once() -> crate::stats::WorldStats {
            let mut world = World::new(small_config("counter-determinism", 12, 10)).unwrap();
            world.spawn_founder_cells(5).unwrap();
            for _ in 0..40 {
                world.step();
            }
            world.stats()
        }

        let a = run_once();
        let b = run_once();
        assert_eq!(a, b);
        assert!(a.operation_counters.cell_steps > 0);
        assert!(a.operation_counters.enzyme_entries_seen > 0);
        assert!(a.operation_counters.metabolic_enzyme_attempts > 0);
        assert_eq!(
            a.operation_counters.metabolic_enzyme_attempts,
            a.reaction_counters.total_attempts()
        );
        assert!(a.operation_counters.local_enval_average_calls > 0);
        assert_eq!(a.operation_counters.enzyme_list_clones, 0);
        assert_eq!(a.operation_counters.genome_clones, 0);
    }

    #[test]
    fn compact_stats_do_not_mutate_rng_or_require_percentiles() {
        let mut world = World::new(small_config("compact-stats-stability", 10, 8)).unwrap();
        world.spawn_founder_cells(3).unwrap();
        world.step_many(5);
        let rng_before = format!("{:?}", world.rng());
        let full = world.stats();
        let compact = world.compact_stats();
        let rng_after = format!("{:?}", world.rng());

        assert_eq!(rng_before, rng_after);
        assert_eq!(compact.tick_count, full.tick_count);
        assert_eq!(compact.live_cell_count, full.live_cell_count);
        assert_eq!(compact.system_element_amounts, full.system_element_amounts);
        assert_eq!(compact.total_element_amount, full.total_element_amount);
        assert_eq!(compact.enval_p05.to_bits(), 0.0_f32.to_bits());
        assert_eq!(compact.enval_p50.to_bits(), 0.0_f32.to_bits());
        assert_eq!(compact.enval_p95.to_bits(), 0.0_f32.to_bits());
        assert_eq!(full.enval_p50.to_bits(), world.stats().enval_p50.to_bits());
        world.check_invariants().unwrap();
    }

    fn niche_enzyme(enval_optimum: f32, rate: f32) -> Enzyme {
        let mut enzyme = inert_enzyme();
        enzyme.enval_optimum = enval_optimum;
        enzyme.rate = rate;
        enzyme
    }

    #[test]
    fn niche_stats_match_scripted_optima_and_a_known_enval_field() {
        let mut world = World::new(only_a_config("niche-stats", 16, 16)).unwrap();
        world.set_all_enval(0.2).unwrap();
        let empty = world.stats();
        assert_eq!(empty.mixed_polarity_cells, 0);
        assert_eq!(empty.mean_enval_optimum_spread, 0.0);
        assert_eq!(empty.mean_niche_mismatch, 0.0);

        let scripted = [
            (vec![niche_enzyme(0.5, 1.0), niche_enzyme(-0.5, 3.0)], true, 1.0, 0.6),
            (vec![niche_enzyme(0.04, 1.0), niche_enzyme(-0.06, 1.0)], false, 0.1, 0.21),
            (vec![niche_enzyme(0.3, 2.0)], false, 0.0, 0.1),
            (vec![niche_enzyme(0.05, 1.0), niche_enzyme(-0.05, 1.0)], false, 0.1, 0.2),
            (vec![niche_enzyme(0.0, 1.0), niche_enzyme(-1.0, 1.0)], false, 1.0, 0.7),
        ];
        let count = scripted.len() as f64;
        let mut mixed = 0;
        let mut spread = 0.0_f64;
        let mut mismatch = 0.0_f64;
        for (index, (enzymes, is_mixed, cell_spread, cell_mismatch)) in
            scripted.into_iter().enumerate()
        {
            let position = Position::new(1.5 + index as f32 * 3.0, 8.5);
            spawn_with(&mut world, position, index as u64 + 1, enzymes);
            mixed += usize::from(is_mixed);
            spread += cell_spread;
            mismatch += cell_mismatch;
        }

        let full = world.stats();
        assert_eq!(full.mixed_polarity_cells, mixed);
        assert_eq!(full.mixed_polarity_cells, 1);
        assert!((full.mean_enval_optimum_spread - spread / count).abs() <= 1.0e-6);
        assert!((full.mean_niche_mismatch - mismatch / count).abs() <= 1.0e-6);

        let compact = world.compact_stats();
        assert_eq!(compact.mixed_polarity_cells, full.mixed_polarity_cells);
        assert_eq!(compact.mean_enval_optimum_spread, full.mean_enval_optimum_spread);
        assert_eq!(compact.mean_niche_mismatch, 0.0);
    }

    #[test]
    fn niche_mismatch_reads_the_smoothed_local_enval_not_the_cells_tile() {
        let mut world = World::new(only_a_config("niche-stats-local", 9, 9)).unwrap();
        world.set_all_enval(0.0).unwrap();
        let tile = world.tile_id(4, 4).unwrap();
        world.set_tile_enval(tile, 1.0).unwrap();
        let local = world.default_local_enval_average(tile).unwrap();
        assert!((local - 0.04).abs() <= 1.0e-6);
        let mut genome = Genome::random_founder(&mut world.rng, 0.0);
        genome.enzymes = vec![niche_enzyme(-0.5, 1.0)];
        world.spawn_cell_with_genome_at(tile, genome).unwrap();
        let stats = world.stats();
        assert!((stats.mean_niche_mismatch - f64::from(local + 0.5)).abs() <= 1.0e-6);
    }

    #[test]
    fn niche_stats_read_only_the_world_and_leave_the_rng_alone() {
        let mut world = World::new(small_config("niche-stats-readonly", 10, 8)).unwrap();
        world.spawn_founder_cells(3).unwrap();
        world.step_many(5);
        let rng_before = format!("{:?}", world.rng());
        let first = world.stats();
        let second = world.stats();
        assert_eq!(first, second);
        assert_eq!(rng_before, format!("{:?}", world.rng()));
        assert_eq!(first.mixed_polarity_cells, 0);
        assert_eq!(first.mean_enval_optimum_spread, 0.0);
        assert!(first.mean_niche_mismatch > 0.0);
    }

    #[test]
    fn profiled_step_records_actual_operation_counters() {
        let mut world = World::new(small_config("profile-counters", 10, 8)).unwrap();
        world.spawn_founder_cells(3).unwrap();
        let profile = world.step_profiled();
        assert!(profile.counters.cell_steps > 0);
        assert!(profile.counters.enzyme_entries_seen > 0);
        assert_eq!(
            profile.counters.cell_steps,
            world.operation_counters().cell_steps
        );
        world.check_invariants().unwrap();
    }

    #[test]
    fn continuous_founders_are_canonical_non_overlapping_and_seed_deterministic() {
        let mut first = World::new(only_a_config("continuous-founders", 12, 9)).unwrap();
        let mut second = World::new(only_a_config("continuous-founders", 12, 9)).unwrap();
        assert_eq!(first.spawn_founder_cells(20).unwrap(), 20);
        assert_eq!(second.spawn_founder_cells(20).unwrap(), 20);
        let positions = |world: &World| {
            live_ids(&world)
                .iter()
                .map(|cell_id| world.cells[*cell_id].position)
                .collect::<Vec<_>>()
        };
        let first_positions = positions(&first);
        assert_eq!(first_positions, positions(&second));
        assert!(first_positions.iter().all(|position| {
            position.is_finite()
                && position.x >= 0.0
                && position.x < first.width as f32
                && position.y >= 0.0
                && position.y < first.height as f32
        }));
        for left in 0..first_positions.len() {
            for right in left + 1..first_positions.len() {
                assert!(
                    toroidal_distance_squared(
                        first_positions[left],
                        first_positions[right],
                        first.width as f32,
                        first.height as f32,
                    ) >= (REFERENCE_CELL_RADIUS * 2.0 - GEOMETRY_TOLERANCE).powi(2)
                );
            }
        }
        first.check_invariants().unwrap();
    }

    #[test]
    fn newborns_do_not_receive_a_cell_step_during_their_birth_phase() {
        let mut world = World::new(only_a_config("newborn-waits", 8, 8)).unwrap();
        let mut genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        make_inert(&mut genome);
        world.config.founder_energy = 10.0;
        genome.maintenance_cost_per_area = 0.0;
        let parent = world
            .spawn_cell_with_genome_at_position(Position::new(4.0, 4.0), genome)
            .unwrap();
        let before_steps = world.operation_counters.cell_steps;
        for element in ELEMENT_ORDER {
            world.cells[parent].boundary[element] *= 1.6;
        }
        world.step_cells();
        assert_eq!(live_ids(&world).len(), 2);
        assert_eq!(world.operation_counters.cell_steps - before_steps, 1);
        let child = live_ids(&world)
            .iter()
            .copied()
            .find(|cell_id| *cell_id != parent)
            .unwrap();
        assert_eq!(world.inspect_cell_fluxes(child, 64).unwrap().flux_count, 0);
    }

    #[test]
    fn cell_phase_snapshot_reuses_its_allocation() {
        let mut world = World::new(only_a_config("cell-phase-scratch", 8, 8)).unwrap();
        for position in [
            Position::new(1.0, 1.0),
            Position::new(3.0, 1.0),
            Position::new(1.0, 3.0),
            Position::new(3.0, 3.0),
        ] {
            let mut genome = Genome::random_founder(&mut world.rng, world.avg_enval);
            make_inert(&mut genome);
            world.config.founder_energy = 10.0;
            genome.maintenance_cost_per_area = 0.0;
            world
                .spawn_cell_with_genome_at_position(position, genome)
                .unwrap();
        }

        world.step_cells();
        let scratch_ptr = world.cell_phase_scratch.as_ptr();
        let scratch_capacity = world.cell_phase_scratch.capacity();
        world.step_cells();

        assert_eq!(world.cell_phase_scratch.as_ptr(), scratch_ptr);
        assert_eq!(world.cell_phase_scratch.capacity(), scratch_capacity);
        assert_eq!(world.cell_phase_scratch, live_ids(&world));
        world.check_invariants().unwrap();
    }

    #[test]
    fn fractional_position_death_release_is_conservative() {
        let mut world = World::new(only_a_config("fractional-death", 5, 5)).unwrap();
        let genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        let position = Position::new(1.25, 2.75);
        let cell_id = world
            .spawn_cell_with_genome_at_position(position, genome)
            .unwrap();
        assert_eq!(world.pick_cell(position), Some(cell_id));
        world.cells[cell_id].internal_elements =
            ElementAmounts::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let before = world.element_field_totals();
        let body = world.cells[cell_id].body;
        let boundary = world.cells[cell_id].boundary;
        world.kill_cell_and_release(cell_id);
        assert_eq!(world.pick_cell(position), None);
        assert_eq!(world.spatial_index.len(), 0);
        let after = world.element_field_totals();
        for element in ELEMENT_ORDER {
            assert!(
                (after[element.index()]
                    - before[element.index()]
                    - (element.index() + 1) as f64
                    - f64::from(body[element])
                    - f64::from(boundary[element]))
                .abs()
                    <= 1.0e-5
            );
        }
        world.check_invariants().unwrap();
    }

    #[test]
    fn failed_death_deposition_preserves_the_intracellular_reservoir() {
        let mut world = World::new(only_a_config("invalid-death-release", 5, 5)).unwrap();
        let genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        let cell_id = world
            .spawn_cell_with_genome_at_position(Position::new(1.25, 2.75), genome)
            .unwrap();
        let reservoir = ElementAmounts::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        world.cells[cell_id].internal_elements = reservoir;
        world.cells[cell_id].position = Position::new(f32::NAN, 2.75);

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            world.kill_cell_and_release(cell_id);
        }));

        assert!(result.is_err());
        assert_eq!(world.cells[cell_id].internal_elements, reservoir);
        assert!(world.cell(cell_id).is_some());
    }

    #[test]
    fn spatial_and_tile_caches_update_incrementally_for_cell_lifecycle() {
        let mut world = World::new(only_a_config("incremental-derived-caches", 6, 6)).unwrap();
        let first_genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        let first = world
            .spawn_cell_with_genome_at_position(Position::new(1.02, 1.02), first_genome)
            .unwrap();
        let second_genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        let second = world
            .spawn_cell_with_genome_at_position(Position::new(1.75, 1.75), second_genome)
            .unwrap();

        assert_eq!(world.inspect_tile_xy(1, 1).unwrap().cell_center_count, 2);
        assert_eq!(world.stats().occupied_tile_count, 1);

        world.update_cell_position(second, Position::new(3.25, 3.25));
        assert_eq!(world.inspect_tile_xy(1, 1).unwrap().cell_center_count, 1);
        assert_eq!(world.inspect_tile_xy(3, 3).unwrap().cell_center_count, 1);
        assert_eq!(world.stats().occupied_tile_count, 2);

        world.kill_cell_and_release(first);
        assert_eq!(world.inspect_tile_xy(1, 1).unwrap().cell_center_count, 0);
        assert_eq!(world.inspect_tile_xy(3, 3).unwrap().cell_center_count, 1);
        assert_eq!(world.stats().occupied_tile_count, 1);
        assert_eq!(
            world
                .spatial_index
                .query_radius(Position::new(3.25, 3.25), 0.01),
            vec![second]
        );
        world.check_invariants().unwrap();
    }

    #[test]
    fn energy_ledger_closes_over_long_runs() {
        let mut world = World::new(small_config("energy-ledger-closure", 32, 24)).unwrap();
        world.spawn_founder_cells(12).unwrap();
        world.set_all_live_cell_energy(4.0).unwrap();
        world.set_all_live_cell_energy(3.0).unwrap();
        for tick in 1..=2_400 {
            world.step();
            if tick % 200 == 0 {
                world.check_invariants().unwrap();
            }
        }
        world.check_invariants().unwrap();
        let ledger = world.energy_ledger();
        assert!(world.energy_ledger_residual().abs() <= ledger.closure_tolerance());
        assert!(ledger.founder_energy > 0.0);
        assert!(ledger.injected_energy > 0.0);
        assert!(ledger.extracted_energy > 0.0);
        assert!(ledger.chemical_harvest > 0.0);
        assert!(ledger.maintenance > 0.0);
        assert_eq!(world.stats().energy_ledger, ledger);
    }

    #[test]
    fn energy_ledger_closure_detects_unledgered_energy_changes() {
        let mut world = World::new(small_config("energy-ledger-detects", 12, 10)).unwrap();
        world.spawn_founder_cells(3).unwrap();
        world.step_many(20);
        world.check_invariants().unwrap();
        let cell_id = live_ids(&world)[0];

        world.add_unledgered_energy(cell_id, 0.25);
        assert!(matches!(
            world.check_invariants(),
            Err(super::InvariantError::EnergyLedgerMismatch { .. })
        ));

        world.add_unledgered_energy(cell_id, -0.25);
        world.check_invariants().unwrap();
    }

    #[test]
    fn enval_ledger_attributes_cell_exchange_and_user_edits() {
        let mut world = World::new(small_config("enval-ledger-basic", 24, 18)).unwrap();
        world.spawn_founder_cells(10).unwrap();
        for tick in 1..=2_000 {
            world.step();
            if tick == 500 {
                world.adjust_tile_enval(TileId(7), 0.75).unwrap();
            }
            if tick == 1_000 {
                world.set_tile_enval(TileId(11), -0.5).unwrap();
            }
        }
        let ledger = world.enval_ledger();
        assert!(ledger.edits != 0.0);
        assert!(ledger.cell_uptake != 0.0 || ledger.cell_emission != 0.0);
        let scale = 1.0 + world.tile_count() as f64;
        assert!(
            world.enval_ledger_residual().abs() <= 1.0e-4 * scale,
            "enval ledger residual {}",
            world.enval_ledger_residual()
        );
    }

    fn closed_config(seed: &str, width: usize, height: usize) -> Config {
        let mut config = small_config(seed, width, height);
        config.enval_landscape.relaxation_per_second = 0.0;
        config.enval_recharge.rate_per_second = 0.0;
        config
    }

    fn pump_engine_genome(world: &mut World, lineage: u64, enval_optimum: f32) -> Genome {
        let mut genome = Genome::random_founder(&mut world.rng, 0.0);
        let mut engine = Enzyme::metabolic(
            ElementAmounts::new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            ElementAmounts::new([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            2.0,
            0.0,
            0.0,
        );
        engine.enval_sigma = 1000.0;
        engine.enval_throughput = 0.5;
        engine.enval_pump = 0.4;
        engine.enval_optimum = enval_optimum;
        genome.enzymes = vec![engine];
        genome.lineage_id = LineageId(lineage);
        world.config.founder_energy = 5.0;
        genome.maintenance_cost_per_area = 0.0;
        genome
    }

    #[test]
    fn paid_pumping_prevents_perpetual_motion_between_opposite_polarity_cells() {
        let mut config = closed_config("pump-no-perpetual-motion", 8, 8);
        config.catalyst_upkeep_per_sec = 0.0;
        config.element_fields.initial_amounts = ElementAmounts::new([0.0, 4.0, 0.0, 0.0, 0.0, 0.0]);
        let mut world = World::new(config).unwrap();
        world.set_all_enval(0.0).unwrap();
        let positive = pump_engine_genome(&mut world, 1, 0.3);
        let negative = pump_engine_genome(&mut world, 2, -0.3);
        world
            .spawn_cell_with_genome_at_position(Position::new(3.0, 4.0), positive)
            .unwrap();
        world
            .spawn_cell_with_genome_at_position(Position::new(4.0, 4.0), negative)
            .unwrap();

        let mut previous = world.total_live_cell_energy();
        let start = previous;
        for _ in 0..1_500 {
            world.step();
            let total = world.total_live_cell_energy();
            assert!(
                total <= previous + 1.0e-12,
                "energy rose: {previous} -> {total}"
            );
            previous = total;
        }
        let ledger = world.energy_ledger();
        assert!(
            ledger.enval_harvest > 0.0,
            "the pump/harvest loop never ran"
        );
        assert!(ledger.pump_cost > ledger.enval_harvest);
        assert_eq!(ledger.chemical_harvest, 0.0);
        assert!(previous < start);
        world.check_invariants().unwrap();
    }

    #[test]
    fn perturbed_cell_free_tiles_relax_toward_their_targets_at_kappa_per_tick() {
        let mut config = small_config("landscape-relaxation", 96, 72);
        config.enval_diffusivity_per_second = 0.0;
        config.enval_recharge.rate_per_second = 0.0;
        let mut world = World::new(config).unwrap();
        let targets = world.enval_landscape().to_vec();
        let kappa = 1.0
            - (-f64::from(world.config.enval_landscape.relaxation_per_second)
                * world.config.dt_seconds)
                .exp();
        let edited = TileId(1234);
        world.set_tile_enval(edited, 7.5).unwrap();
        world.set_tile_enval(TileId(77), -3.0).unwrap();
        for _ in 0..300 {
            let before = world.enval.clone();
            let inflow_before = world.enval_ledger().source_inflow;
            world.step();
            let mut applied = 0.0;
            for tile in 0..world.tile_count() {
                let expected = f64::from(before[tile])
                    + kappa * (f64::from(targets[tile]) - f64::from(before[tile]));
                assert!(
                    (f64::from(world.enval[tile]) - expected).abs() <= 1.0e-6,
                    "tile {tile}"
                );
                applied += f64::from(world.enval[tile]) - f64::from(before[tile]);
            }
            let inflow = world.enval_ledger().source_inflow - inflow_before;
            assert!((inflow - applied).abs() <= 1.0e-9);
        }
        let expected_gap = (7.5 - f64::from(targets[edited.index()])) * (1.0 - kappa).powi(300);
        let gap = f64::from(world.enval[edited.index()]) - f64::from(targets[edited.index()]);
        assert!((gap - expected_gap).abs() <= 1.0e-4 * expected_gap.abs());
        assert!(gap < 7.5 - f64::from(targets[edited.index()]));
        assert!(world.enval_ledger_residual().abs() <= 1.0e-6);

        let mut frozen_config = small_config("landscape-relaxation", 96, 72);
        frozen_config.enval_landscape.relaxation_per_second = 0.0;
        frozen_config.enval_recharge.rate_per_second = 0.0;
        frozen_config.enval_diffusivity_per_second = 0.0;
        let mut frozen = World::new(frozen_config).unwrap();
        frozen.set_tile_enval(edited, 7.5).unwrap();
        let before = frozen.enval.clone();
        frozen.step_many(200);
        assert_eq!(frozen.enval_ledger().source_inflow, 0.0);
        assert_eq!(frozen.enval, before);
    }

    #[test]
    fn landscape_is_seed_deterministic_bounded_periodic_and_differs_between_seeds() {
        let first = World::new(small_config("world-landscape", 320, 240)).unwrap();
        let second = World::new(small_config("world-landscape", 320, 240)).unwrap();
        let other = World::new(small_config("world-landscape-other", 320, 240)).unwrap();
        assert_eq!(first.enval_landscape(), second.enval_landscape());
        assert_eq!(first.enval, second.enval);
        assert_ne!(first.enval_landscape(), other.enval_landscape());
        let magnitude = first.config.enval_landscape.magnitude;
        let peak = first
            .enval_landscape()
            .iter()
            .fold(0.0_f32, |peak, value| peak.max(value.abs()));
        assert_eq!(peak, magnitude);
        assert!(
            first
                .enval_landscape()
                .iter()
                .all(|value| value.abs() <= magnitude)
        );
        let at = |x: usize, y: usize| first.enval_landscape()[first.index_xy(x, y)];
        let mut interior = 0.0_f32;
        for x in 0..first.width - 1 {
            for y in 0..first.height - 1 {
                interior = interior
                    .max((at(x + 1, y) - at(x, y)).abs())
                    .max((at(x, y + 1) - at(x, y)).abs());
            }
        }
        let mut seam = 0.0_f32;
        for y in 0..first.height {
            seam = seam.max((at(0, y) - at(first.width - 1, y)).abs());
        }
        for x in 0..first.width {
            seam = seam.max((at(x, 0) - at(x, first.height - 1)).abs());
        }
        assert!(
            interior > 0.0 && seam <= interior,
            "seam {seam} interior {interior}"
        );
        let positive = first
            .enval_landscape()
            .iter()
            .filter(|value| **value > 0.0)
            .count();
        assert!(positive > first.tile_count() / 5 && positive < 4 * first.tile_count() / 5);
    }

    #[test]
    fn enval_ledger_closes_with_landscape_cells_recharge_and_user_edits() {
        let mut config = small_config("enval-ledger-landscape", 64, 48);
        config.initial_founder_count = 12;
        let mut world = World::new(config).unwrap();
        world.spawn_founder_cells(12).unwrap();
        let mut max_residual = 0.0_f64;
        for tick in 1..=2_000 {
            world.step();
            match tick {
                300 => world.adjust_tile_enval(TileId(5), 2.0).unwrap(),
                900 => world.set_tile_enval(TileId(3), -3.0).unwrap(),
                1_400 => world.set_all_enval(0.05).unwrap(),
                _ => {}
            }
            max_residual = max_residual.max(world.enval_ledger_residual().abs());
        }
        let ledger = world.enval_ledger();
        assert!(ledger.source_inflow != 0.0);
        assert!(ledger.recharge != 0.0);
        assert!(ledger.edits != 0.0);
        assert!(ledger.cell_uptake != 0.0 || ledger.cell_emission != 0.0);
        assert!(
            max_residual <= 1.0e-3,
            "enval ledger residual {max_residual}"
        );
        world.check_invariants().unwrap();
    }

    #[test]
    fn cell_free_recharge_conserves_matter_never_flips_enval_and_matches_energy() {
        let mut config = small_config("recharge-cell-free", 64, 48);
        config.element_fields.initial_amounts = ElementAmounts::new([0.2, 0.2, 0.2, 0.0, 0.0, 0.6]);
        config.enval_recharge.rate_per_second = 0.5;
        let mut world = World::new(config).unwrap();
        let initial_total = world.stats().total_element_amount;
        let mut consumed = 0.0_f64;
        for _ in 0..1_500 {
            world.diffuse_element_fields();
            world.diffuse_enval();
            let before = world.enval.clone();
            let energy_before = world.enval_ledger.recharge_energy;
            world.apply_recharge();
            let mut tick_consumed = 0.0_f64;
            for (old, new) in before.iter().zip(&world.enval) {
                assert!(
                    old.signum() == new.signum() || *new == 0.0,
                    "recharge flipped enval"
                );
                assert!(new.abs() <= old.abs());
                tick_consumed += f64::from(old.abs()) - f64::from(new.abs());
            }
            let tick_energy = world.enval_ledger.recharge_energy - energy_before;
            assert!((tick_energy - tick_consumed).abs() <= 1.0e-5 + 1.0e-4 * tick_energy);
            consumed += tick_consumed;
            world.relax_enval_landscape();
            world.refresh_enval_sum();
            world.advance_time();
            assert!(
                world
                    .element_fields
                    .iter()
                    .all(|amounts| amounts[Element::F] >= 0.0)
            );
        }
        let ledger = world.enval_ledger();
        assert!(ledger.recharge_amount > 1.0);
        assert!((ledger.recharge_energy - consumed).abs() <= 1.0e-3 * consumed);
        let total = world.stats().total_element_amount;
        assert!((total - initial_total).abs() <= 1.0e-5 * initial_total);
        world.check_invariants().unwrap();
    }

    #[test]
    fn zero_recharge_rate_is_bit_identical_to_skipping_the_recharge_step() {
        let mut config = small_config("recharge-disabled", 48, 36);
        config.enval_recharge.rate_per_second = 0.0;
        let mut stepped = World::new(config).unwrap();
        stepped.spawn_founder_cells(8).unwrap();
        let mut manual = stepped.clone();
        for _ in 0..400 {
            stepped.step();
            manual.diffuse_element_fields();
            manual.step_cells();
            manual.drift_cells();
            manual.resolve_overlaps();
            manual.diffuse_enval();
            manual.relax_enval_landscape();
            manual.refresh_enval_sum();
            manual.advance_time();
        }
        assert_eq!(stepped.element_fields, manual.element_fields);
        assert_eq!(
            stepped
                .enval
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            manual
                .enval
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(stepped.enval_ledger().recharge_amount, 0.0);
    }

    #[test]
    fn recharge_makes_d_only_where_enval_is_positive_and_e_only_where_negative() {
        let mut config = small_config("recharge-sign", 96, 72);
        config.enval_diffusivity_per_second = 0.0;
        config.element_fields.diffusivities = ElementAmounts::ZERO;
        config.element_fields.initial_amounts = ElementAmounts::new([0.0, 0.0, 0.0, 0.1, 0.1, 1.0]);
        config.element_fields.heterogeneity = 0.0;
        config.enval_recharge.rate_per_second = 0.5;
        let mut world = World::new(config).unwrap();
        world.step_many(300);
        let mut saw_d = false;
        let mut saw_e = false;
        for tile in 0..world.tile_count() {
            let amounts = world.element_fields[tile];
            let d_gain = amounts[Element::D] - 0.1;
            let e_gain = amounts[Element::E] - 0.1;
            let target = world.enval_landscape()[tile];
            if target >= 0.0 {
                assert!(e_gain.abs() <= 1.0e-7);
            }
            if target <= 0.0 {
                assert!(d_gain.abs() <= 1.0e-7);
            }
            if target > 0.05 {
                assert!(d_gain > 0.0);
                saw_d = true;
            } else if target < -0.05 {
                assert!(e_gain > 0.0);
                saw_e = true;
            }
        }
        assert!(saw_d && saw_e);
    }

    #[test]
    fn catalyst_upkeep_charges_every_enzyme_on_top_of_base_maintenance() {
        let drain_with = |enzymes: Vec<Enzyme>| {
            let mut world = World::new(closed_config("catalyst-upkeep", 6, 6)).unwrap();
            let mut genome = Genome::random_founder(&mut world.rng, 0.0);
            genome.enzymes = enzymes;
            world.config.founder_energy = 10.0;
            genome.maintenance_cost_per_area = 0.05;
            let cell_id = world
                .spawn_cell_with_genome_at_position(Position::new(2.0, 2.0), genome)
                .unwrap();
            let before = world.cells[cell_id].energy;
            world.step_cell(cell_id);
            let paid = before - world.cells[cell_id].energy;
            assert!((world.energy_ledger().maintenance - paid).abs() <= 1.0e-15);
            assert_eq!(
                world
                    .inspect_cell_detail(cell_id, 0)
                    .unwrap()
                    .catalyst_upkeep_per_sec,
                0.01 * world.cells[cell_id].genome.enzymes.len() as f64
            );
            paid
        };
        let dt = Config::default().dt_seconds;
        let one = drain_with(vec![inert_enzyme()]);
        assert!(
            (one - (0.05 * f64::from(crate::spatial::REFERENCE_CELL_AREA) + 0.01) * dt).abs()
                <= 1.0e-10
        );
        let four = drain_with(vec![inert_enzyme(); 4]);
        assert!(
            (four - (0.05 * f64::from(crate::spatial::REFERENCE_CELL_AREA) + 0.04) * dt).abs()
                <= 1.0e-10
        );
    }

    #[test]
    fn small_default_world_stays_alive_and_honest() {
        for seed in ["viability-1", "viability-2", "viability-3"] {
            let mut config = small_config(seed, 64, 48);
            config.initial_founder_count = 8;
            let mut world = World::new(config).unwrap();
            assert_eq!(world.spawn_founder_cells(8).unwrap(), 8);
            for tick in 1..=3_000 {
                world.step();
                if tick % 100 == 0 {
                    assert!(
                        world.cell_count() > 0,
                        "seed {seed} went extinct by tick {tick}"
                    );
                }
            }
            let residual = world.energy_ledger_residual();
            assert!(
                residual.abs() <= world.energy_ledger().closure_tolerance(),
                "seed {seed}: ledger residual {residual}"
            );
            world.check_invariants().unwrap();
            assert!(
                world.reaction_counters.divisions >= 1,
                "seed {seed} had no fission"
            );
            let stats = world.stats();
            assert!(stats.energy_ledger.enval_harvest > 0.0);
            assert!(stats.enval_ledger.recharge_energy > 0.0);
        }
    }

    #[test]
    fn dead_cells_are_compacted_with_stable_unreused_ids() {
        let mut world = World::new(closed_config("dead-cell-compaction", 24, 18)).unwrap();
        world.spawn_founder_cells(12).unwrap();
        let ids = live_ids(&world);
        let positions = ids
            .iter()
            .map(|cell_id| (*cell_id, world.cells[*cell_id].position))
            .collect::<Vec<_>>();
        for victim in [ids[0], ids[5], ids[11]] {
            world.kill_cell_and_release(victim);
        }
        world.check_invariants().unwrap();
        assert_eq!(world.cell_count(), 9);
        for (cell_id, position) in positions {
            if [ids[0], ids[5], ids[11]].contains(&cell_id) {
                assert!(world.cell(cell_id).is_none());
                assert!(world.inspect_cell(cell_id).is_none());
                assert!(world.inspect_cell_detail(cell_id, 4).is_none());
                assert!(world.inspect_cell_fluxes(cell_id, 4).is_none());
                assert_ne!(world.pick_cell(position), Some(cell_id));
            } else {
                let cell = world.cell(cell_id).unwrap();
                assert_eq!(cell.id, cell_id);
                assert_eq!(cell.position, position);
                assert_eq!(world.inspect_cell(cell_id).unwrap().cell_id, cell_id);
                assert_eq!(world.pick_cell(position), Some(cell_id));
            }
        }
        let genome = Genome::random_founder(&mut world.rng, 0.0);
        let newborn = world
            .spawn_cell_with_genome_at_position(Position::new(0.5, 0.5), genome)
            .or_else(|_| {
                let genome = Genome::random_founder(&mut world.rng, 0.0);
                world.spawn_cell_with_genome_at_position(Position::new(12.3, 9.1), genome)
            })
            .unwrap();
        assert_eq!(newborn, crate::cell::CellId(12));
        assert!(world.cell(ids[0]).is_none());

        world.step_many(50);
        world.check_invariants().unwrap();
        let reloaded =
            crate::snapshot::from_bytes(&crate::snapshot::to_bytes(&world).unwrap()).unwrap();
        assert_eq!(live_ids(&reloaded), live_ids(&world));
        for cell_id in [ids[0], ids[5], ids[11]] {
            assert!(reloaded.cell(cell_id).is_none());
        }
        reloaded.check_invariants().unwrap();
    }

    #[test]
    fn snapshot_size_does_not_grow_with_cumulative_deaths() {
        let populate = |deaths: usize| {
            let mut world = World::new(closed_config("snapshot-death-size", 40, 30)).unwrap();
            let mut genome = Genome::random_founder(&mut world.rng, 0.0);
            genome.lineage_id = LineageId(7);
            let mut churned = 0;
            while churned < deaths {
                let id = world
                    .spawn_cell_with_genome_at_position(Position::new(20.0, 15.0), genome.clone())
                    .unwrap();
                world.kill_cell_and_release(id);
                churned += 1;
            }
            for index in 0..100 {
                let position = Position::new((index % 20) as f32 * 2.0, (index / 20) as f32 * 2.0);
                world
                    .spawn_cell_with_genome_at_position(position, genome.clone())
                    .unwrap();
            }
            world.check_invariants().unwrap();
            world
        };
        let quiet = populate(0);
        let churned = populate(20_000);
        assert_eq!(quiet.cell_count(), churned.cell_count());
        assert_eq!(churned.stats().deaths, 20_000);
        let quiet_size = crate::snapshot::to_bytes(&quiet).unwrap().len() as f64;
        let churned_size = crate::snapshot::to_bytes(&churned).unwrap().len() as f64;
        assert!(
            (churned_size - quiet_size).abs() <= 0.05 * quiet_size,
            "{churned_size} vs {quiet_size} bytes"
        );
        assert!(churned.cells.issued_ids() >= 20_100);
    }

    fn surface_enzyme(reactants: [f32; 6], products: [f32; 6], rate: f32, exposure: f32) -> Enzyme {
        let mut enzyme = Enzyme::metabolic(
            ElementAmounts::new(reactants),
            ElementAmounts::new(products),
            rate,
            0.5,
            0.0,
        );
        enzyme.enval_sigma = 1000.0;
        enzyme.enval_throughput = 0.0;
        enzyme.enval_pump = 0.0;
        enzyme.exposure = exposure;
        enzyme
    }

    const E_ONLY: [f32; 6] = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    const A_ONLY: [f32; 6] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    fn spawn_with(
        world: &mut World,
        position: Position,
        lineage: u64,
        enzymes: Vec<Enzyme>,
    ) -> crate::cell::CellId {
        let mut genome = Genome::random_founder(&mut world.rng, 0.0);
        genome.enzymes = enzymes;
        genome.lineage_id = LineageId(lineage);
        genome.maintenance_cost_per_area = 0.0;
        world
            .spawn_cell_with_genome_at_position(position, genome)
            .unwrap()
    }

    fn reaction_context(world: &World, cell_id: crate::cell::CellId) -> super::CellReactionContext {
        let slot = world.cells.slot(cell_id).unwrap();
        let cell = world.cells.at_slot(slot);
        super::CellReactionContext {
            cell_id,
            slot,
            position: cell.position,
            stencil: world.tile_stencil(cell.position).unwrap(),
            local_enval: world
                .local_enval_average_at(cell.position, super::LOCAL_ENVAL_RADIUS)
                .unwrap(),
            cell_area: std::f32::consts::PI * cell.radius * cell.radius,
        }
    }

    fn contacts_of(world: &World, cell_id: crate::cell::CellId) -> (Vec<super::Contact>, f64) {
        let cell = &world.cells[cell_id];
        let mut contacts = Vec::new();
        let field = world.collect_contacts(cell_id, cell.position, cell.radius, &mut contacts);
        (contacts, field)
    }

    fn strain_of(world: &World, cell_id: crate::cell::CellId) -> f64 {
        world.cells[cell_id].strain(world.config.cell_body.boundary_density)
    }

    #[test]
    fn exposed_enzyme_digests_a_matching_neighbour_boundary_until_it_lyses() {
        let mut config = only_a_config("surface-digestion", 8, 8);
        config.catalyst_upkeep_per_sec = 0.0;
        config.founder_energy = 50.0;
        let mut world = World::new(config).unwrap();
        let digester = spawn_with(
            &mut world,
            Position::new(3.0, 4.0),
            1,
            vec![surface_enzyme(E_ONLY, A_ONLY, 2.0, 1.0)],
        );
        let prey = spawn_with(&mut world, Position::new(3.9, 4.0), 2, vec![inert_enzyme()]);
        let boundary = world.cells[prey].boundary.total() as f32;
        world.cells[prey].boundary = ElementAmounts::new([0.0, 0.0, 0.0, 0.0, boundary * 0.8, 0.0]);
        let digester_boundary = world.cells[digester].boundary;
        let initial_total = world.stats().total_element_amount;
        world.inspect_cell_fluxes(digester, 64).unwrap();

        let mut previous = strain_of(&world, prey);
        assert!(previous > world.config.cell_body.lysis_strain);
        let mut lysed = false;
        for _ in 0..2_000 {
            world.step_cell(digester);
            let Some(_) = world.cell(prey) else { break };
            let strain = strain_of(&world, prey);
            assert!(strain <= previous);
            previous = strain;
            world.step_cell(prey);
            world.advance_time();
            world.check_invariants().unwrap();
            if world.cell(prey).is_none() {
                lysed = true;
                break;
            }
        }

        assert!(lysed, "prey never lysed; strain {previous}");
        assert!(previous < world.config.cell_body.lysis_strain);
        assert_eq!(world.lysis_count, 1);
        assert_eq!(world.energy_death_count, 0);
        let counters = world.interaction_counters();
        assert_eq!(counters.lyses_after_digestion, 1);
        assert!(counters.boundary_digested > 0.0);
        assert_eq!(counters.cross_lineage_boundary_digested, counters.boundary_digested);
        assert!(counters.cell_surface_extent > 0.0);
        assert_eq!(world.cells[digester].boundary, digester_boundary);
        assert!(
            world
                .flux_log
                .fluxes_newest_first()
                .any(|record| record.site == crate::cell::ReactionSite::Cell(prey))
        );
        let total = world.stats().total_element_amount;
        assert!((total - initial_total).abs() <= 1.0e-5 * initial_total);
        world.check_invariants().unwrap();
    }

    #[test]
    fn lyses_count_as_digestion_only_within_the_attribution_window() {
        for (since, attributed) in [(None, 0), (Some(100), 1), (Some(101), 0)] {
            let mut config = only_a_config("lysis-attribution", 8, 8);
            config.catalyst_upkeep_per_sec = 0.0;
            let mut world = World::new(config).unwrap();
            for _ in 0..200 {
                world.advance_time();
            }
            let id = spawn_with(&mut world, Position::new(2.5, 3.5), 1, vec![inert_enzyme()]);
            for element in ELEMENT_ORDER {
                world.cells[id].boundary[element] *= 0.6;
            }
            world.cells[id].last_digested_tick = since.map(|ago| world.tick_count - ago);
            world.step_cell(id);
            assert_eq!(world.lysis_count, 1);
            assert_eq!(
                world.interaction_counters().lyses_after_digestion,
                attributed,
                "{since:?}"
            );
        }
    }

    #[test]
    fn surface_catalysts_never_digest_their_own_boundary() {
        let mut config = only_a_config("surface-self", 8, 8);
        config.catalyst_upkeep_per_sec = 0.0;
        config.founder_energy = 50.0;
        let mut world = World::new(config).unwrap();
        let cell = spawn_with(
            &mut world,
            Position::new(3.0, 4.0),
            1,
            vec![surface_enzyme(E_ONLY, A_ONLY, 5.0, 1.0)],
        );
        let neighbour = spawn_with(&mut world, Position::new(3.9, 4.0), 1, vec![inert_enzyme()]);
        let total = world.cells[cell].boundary.total() as f32;
        world.cells[cell].boundary = ElementAmounts::new([0.0, 0.0, 0.0, 0.0, total, 0.0]);
        let own_boundary = world.cells[cell].boundary;
        let neighbour_boundary = world.cells[neighbour].boundary;
        let (contacts, _) = contacts_of(&world, cell);
        assert_eq!(contacts.len(), 1);
        assert!(contacts.iter().all(|contact| contact.cell_id != cell));
        world.inspect_cell_fluxes(cell, 64).unwrap();

        for _ in 0..500 {
            world.step_cell(cell);
            world.advance_time();
        }

        assert_eq!(world.cells[cell].boundary, own_boundary);
        assert_eq!(world.cells[neighbour].boundary, neighbour_boundary);
        assert_eq!(world.cells[cell].last_digested_tick, None);
        let counters = world.interaction_counters();
        assert_eq!(counters.boundary_digested, 0.0);
        assert_eq!(counters.cell_surface_extent, 0.0);
        assert_eq!(counters.field_surface_extent, 0.0);
        assert!(world.reaction_counters().no_substrate >= 1_000);
        assert_eq!(world.flux_log.flux_count(), 0);
        world.check_invariants().unwrap();
    }

    #[test]
    fn isolated_surface_catalysis_draws_exactly_on_the_field_and_conserves_matter() {
        let mut config = small_config("surface-field", 12, 10);
        config.founder_energy = 20.0;
        let mut world = World::new(config).unwrap();
        let mut enzyme = Enzyme::founder_downhill();
        enzyme.rate = 5.0;
        enzyme.secretion_fraction = 0.0;
        enzyme.body_fraction = 0.0;
        enzyme.boundary_fraction = 0.0;
        enzyme.enval_sigma = 1000.0;
        enzyme.exposure = 1.0;
        let cell = spawn_with(&mut world, Position::new(5.3, 4.7), 1, vec![enzyme]);
        let (contacts, field_share) = contacts_of(&world, cell);
        assert!(contacts.is_empty());
        assert_eq!(field_share, 1.0);

        let stencil =
            crate::spatial::BilinearStencil::new(Position::new(5.3, 4.7), 12, 10).unwrap();
        let tiles = stencil
            .samples
            .map(|sample| world.index_xy(sample.x, sample.y));
        let tiles_before = tiles.map(|tile| world.element_fields[tile]);
        let field_before = world.element_field_totals();
        let internal_before = world.cells[cell].internal_elements;
        let system_before = world.stats().system_element_amounts;
        world.inspect_cell_fluxes(cell, 4).unwrap();
        let context = reaction_context(&world, cell);
        world.surface_reaction_on_field(context, 0, &enzyme, 1.0);

        let record = world.flux_log.latest_flux().unwrap().clone();
        assert_eq!(record.site, crate::cell::ReactionSite::Field);
        let consumed = f64::from(record.executed_extent);
        assert!(consumed > 0.0);
        let field_after = world.element_field_totals();
        let d = Element::D.index();
        assert!(
            (field_before[d] - field_after[d] - consumed).abs() <= 1.0e-6,
            "{} vs {consumed}",
            field_before[d] - field_after[d]
        );
        for element in ELEMENT_ORDER {
            if element != Element::D {
                assert_eq!(field_after[element.index()], field_before[element.index()]);
            }
        }
        let weighted = (0..4)
            .map(|corner| {
                f64::from(stencil.samples[corner].weight) * f64::from(tiles_before[corner][Element::D])
            })
            .collect::<Vec<_>>();
        let pool = weighted.iter().sum::<f64>();
        for corner in 0..4 {
            let taken = f64::from(tiles_before[corner][Element::D])
                - f64::from(world.element_fields[tiles[corner]][Element::D]);
            assert!((taken - consumed * weighted[corner] / pool).abs() <= 1.0e-6);
        }
        let gained = f64::from(world.cells[cell].internal_elements[Element::A])
            - f64::from(internal_before[Element::A]);
        assert!((gained - consumed).abs() <= 1.0e-6);
        let system_after = world.stats().system_element_amounts;
        for element in ELEMENT_ORDER {
            let index = element.index();
            let expected = if element == Element::D {
                -consumed
            } else if element == Element::A {
                consumed
            } else {
                0.0
            };
            assert!((system_after[index] - system_before[index] - expected).abs() <= 1.0e-5);
        }
        assert!(world.interaction_counters().field_surface_extent > 0.0);
        assert_eq!(world.stats().interaction, world.interaction_counters());
        world.check_invariants().unwrap();

        let initial_total = world.stats().total_element_amount;
        for tick in 1..=1_000 {
            world.step();
            if tick % 100 == 0 {
                world.check_invariants().unwrap();
            }
        }
        let total = world.stats().total_element_amount;
        assert!((total - initial_total).abs() <= 1.0e-6 * initial_total);
        assert!(world.interaction_counters().field_surface_extent > consumed);
        assert_eq!(world.interaction_counters().boundary_digested, 0.0);
        assert!(world.energy_ledger_residual().abs() <= world.energy_ledger().closure_tolerance());
    }

    #[test]
    fn contact_shares_follow_the_formula_sum_to_at_most_one_and_wrap_symmetrically() {
        let mut world = World::new(only_a_config("contact-shares", 20, 16)).unwrap();
        let gap = f64::from(world.config.surface.contact_gap);
        let isolated = spawn_with(&mut world, Position::new(15.0, 12.0), 1, vec![inert_enzyme()]);
        let (contacts, field) = contacts_of(&world, isolated);
        assert!(contacts.is_empty());
        assert_eq!(field, 1.0);

        let mut pair_shares = Vec::new();
        for (a, b) in [(Position::new(5.0, 5.0), Position::new(5.95, 5.0)), (Position::new(0.2, 9.0), Position::new(19.25, 9.0))] {
            let first = spawn_with(&mut world, a, 1, vec![inert_enzyme()]);
            let second = spawn_with(&mut world, b, 2, vec![inert_enzyme()]);
            let (contacts, field) = contacts_of(&world, first);
            assert_eq!(contacts.len(), 1);
            let contact = contacts[0];
            assert_eq!(contact.cell_id, second);
            let ra = f64::from(world.cells[first].radius);
            let rb = f64::from(world.cells[second].radius);
            let closeness = ((ra + rb + gap - contact.distance) / gap).clamp(0.0, 1.0);
            assert!(closeness > 0.0 && closeness < 1.0);
            let expected = closeness * (rb / contact.distance).asin() / std::f64::consts::PI;
            assert!((contact.share - expected).abs() <= 1.0e-12);
            assert!((field + contact.share - 1.0).abs() <= 1.0e-12);
            let (reverse, _) = contacts_of(&world, second);
            assert_eq!(reverse[0].cell_id, first);
            assert!((reverse[0].distance - contact.distance).abs() <= 1.0e-5);
            assert!((reverse[0].share - contact.share).abs() <= 1.0e-5);
            pair_shares.push(contact.share);
        }
        assert!((pair_shares[0] - pair_shares[1]).abs() <= 1.0e-5, "{pair_shares:?}");
        assert!(pair_shares[0] > 0.0 && pair_shares[0] < 1.0 / 6.0);

        let hub = spawn_with(&mut world, Position::new(10.0, 4.0), 1, vec![inert_enzyme()]);
        for index in 0..8 {
            let angle = index as f32 * std::f32::consts::TAU / 8.0;
            let position = Position::new(10.0 + 0.9 * angle.cos(), 4.0 + 0.9 * angle.sin());
            let id = spawn_with(&mut world, Position::new(1.0 + index as f32 * 2.0, 14.0), 3, vec![inert_enzyme()]);
            world.update_cell_position(id, position);
        }
        let (contacts, field) = contacts_of(&world, hub);
        assert_eq!(contacts.len(), 8);
        let total = contacts.iter().map(|contact| contact.share).sum::<f64>();
        assert!((total - 1.0).abs() <= 1.0e-12);
        assert_eq!(field, 0.0);
        assert!(
            contacts
                .windows(2)
                .all(|pair| pair[0].cell_id.index() < pair[1].cell_id.index())
        );
    }

    #[test]
    fn internal_and_surface_reactions_of_one_enzyme_never_exceed_its_capacity() {
        let mut config = small_config("surface-capacity", 12, 10);
        config.founder_energy = 100.0;
        let mut world = World::new(config).unwrap();
        let mut enzyme = surface_enzyme([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], A_ONLY, 3.0, 0.6);
        enzyme.half_saturation = 1.0e-3;
        let actor = spawn_with(&mut world, Position::new(5.0, 5.0), 1, vec![enzyme]);
        for (index, offset) in [(0.0_f32, 0.9_f32), (0.95, 0.0), (0.0, -0.93)].into_iter().enumerate() {
            let id = spawn_with(
                &mut world,
                Position::new(5.0 + offset.0, 5.0 + offset.1),
                2 + index as u64,
                vec![inert_enzyme()],
            );
            world.cells[id].boundary[Element::D] += 2.0;
        }
        world.cells[actor].internal_elements[Element::D] = 5.0;
        world.inspect_cell_fluxes(actor, 64).unwrap();
        let radius = world.cells[actor].radius;
        let area = std::f32::consts::PI * radius * radius;
        world.step_cell(actor);

        let records = world.flux_log.fluxes_newest_first().cloned().collect::<Vec<_>>();
        assert!(records.iter().any(|record| record.site == crate::cell::ReactionSite::Internal));
        assert!(records.iter().any(|record| record.site == crate::cell::ReactionSite::Field));
        assert_eq!(
            records
                .iter()
                .filter(|record| matches!(record.site, crate::cell::ReactionSite::Cell(_)))
                .count(),
            3
        );
        let capacity = f64::from(enzyme.rate)
            * world.config.dt_seconds
            * f64::from(area / crate::spatial::REFERENCE_CELL_AREA);
        let requested = records
            .iter()
            .map(|record| f64::from(record.requested_extent))
            .sum::<f64>();
        assert!(requested <= capacity * (1.0 + 1.0e-6), "{requested} > {capacity}");
        assert!(requested >= 0.9 * capacity, "{requested} vs {capacity}");
        world.check_invariants().unwrap();
    }

    #[test]
    fn lineage_labels_never_change_the_simulation() {
        let run = |distinct_lineages: bool| {
            let mut config = small_config("lineage-blind", 16, 12);
            config.founder_energy = 4.0;
            let mut world = World::new(config).unwrap();
            let mut genome_rng = crate::rng::Rng::from_seed_str("lineage-blind-genomes");
            for index in 0..12_u64 {
                let mut genome = Genome::random_founder(&mut genome_rng, 0.0);
                genome.enzymes[0].exposure = 0.5;
                genome.mutation_rate = 0.5;
                genome.lineage_id = LineageId(if distinct_lineages { 100 + index } else { 7 });
                let position = Position::new(
                    4.0 + (index % 4) as f32 * 0.92,
                    4.0 + (index / 4) as f32 * 0.92,
                );
                let id = world
                    .spawn_cell_with_genome_at_position(position, genome)
                    .unwrap();
                for element in ELEMENT_ORDER {
                    world.cells[id].boundary[element] *= 1.6;
                }
            }
            for _ in 0..600 {
                world.step();
            }
            world.check_invariants().unwrap();
            world
        };
        let same = run(false);
        let distinct = run(true);
        let physics = |world: &World| {
            let mut cells = world.cells.iter().cloned().collect::<Vec<_>>();
            for cell in &mut cells {
                cell.lineage_id = LineageId(0);
                cell.genome.lineage_id = LineageId(0);
            }
            let mut interaction = world.interaction_counters();
            interaction.cross_lineage_boundary_digested = 0.0;
            bincode::serialize(&(
                cells,
                &world.rng,
                &world.element_fields,
                &world.enval,
                world.energy_ledger,
                world.enval_ledger,
                world.reaction_counters,
                interaction,
                world.contact_pair_count,
                world.death_count,
                world.birth_count,
            ))
            .unwrap()
        };
        assert!(same.interaction_counters().boundary_digested > 0.0);
        assert_eq!(same.interaction_counters().cross_lineage_boundary_digested, 0.0);
        assert!(distinct.interaction_counters().cross_lineage_boundary_digested > 0.0);
        let stats = same.stats();
        assert!(stats.fissions > 0, "{} cells, {} deaths, {} lyses", stats.live_cell_count, stats.deaths, stats.lyses);
        assert_eq!(physics(&same), physics(&distinct));
    }

    fn drift_world(seed: &str, columns: usize, rows: usize) -> (World, Vec<crate::cell::CellId>) {
        let mut world = World::new(small_config(seed, columns * 3, rows * 3)).unwrap();
        let mut ids = Vec::new();
        for column in 0..columns {
            for row in 0..rows {
                let position = Position::new(column as f32 * 3.0 + 1.5, row as f32 * 3.0 + 1.5);
                ids.push(spawn_with(&mut world, position, 1, vec![inert_enzyme()]));
            }
        }
        (world, ids)
    }

    fn mean_squared_displacement(
        world: &mut World,
        ids: &[crate::cell::CellId],
        ticks: usize,
    ) -> f64 {
        let mut displacement = vec![(0.0_f64, 0.0_f64); ids.len()];
        for _ in 0..ticks {
            let before = ids.iter().map(|id| world.cells[*id].position).collect::<Vec<_>>();
            world.drift_cells();
            for (index, id) in ids.iter().enumerate() {
                let step = minimum_image_displacement(
                    before[index],
                    world.cells[*id].position,
                    world.width as f32,
                    world.height as f32,
                );
                displacement[index].0 += f64::from(step.x);
                displacement[index].1 += f64::from(step.y);
            }
        }
        displacement
            .iter()
            .map(|(x, y)| x * x + y * y)
            .sum::<f64>()
            / ids.len() as f64
    }

    #[test]
    fn drift_mean_squared_displacement_is_four_d_t_and_scales_inversely_with_radius() {
        let (mut world, ids) = drift_world("drift-msd", 24, 24);
        let (small, reference) = ids.split_at(ids.len() / 2);
        for id in small {
            let cell = &mut world.cells[*id];
            for element in ELEMENT_ORDER {
                cell.body[element] *= 0.25;
            }
            cell.refresh_radius(world.config.cell_body.body_density);
        }
        world.refresh_max_cell_radius();
        let ticks = 1_000;
        let elapsed = ticks as f64 * world.config.dt_seconds;
        let diffusivity = f64::from(world.config.cell_motion.diffusivity);
        let radius_ratio = f64::from(world.cells[reference[0]].radius)
            / f64::from(world.cells[small[0]].radius);
        assert!((radius_ratio - 2.0).abs() < 1.0e-3);

        let mut reference_world = world.clone();
        let reference_msd = mean_squared_displacement(&mut reference_world, reference, ticks);
        let small_msd = mean_squared_displacement(&mut world, small, ticks);
        let expected = 4.0 * diffusivity * elapsed;
        assert!(
            (reference_msd / expected - 1.0).abs() <= 0.15,
            "reference MSD {reference_msd} vs {expected}"
        );
        let ratio = small_msd / reference_msd;
        assert!((ratio - 2.0).abs() <= 0.35, "half-radius MSD ratio {ratio}");
        world.check_invariants().unwrap();
        reference_world.check_invariants().unwrap();
    }

    #[test]
    fn drift_draws_two_normals_per_cell_and_nothing_when_disabled() {
        let (mut world, ids) = drift_world("drift-rng", 4, 3);
        let positions = |world: &World| {
            ids.iter()
                .map(|id| world.cells[*id].position)
                .collect::<Vec<_>>()
        };

        let mut still = world.clone();
        still.config.cell_motion.diffusivity = 0.0;
        let before = positions(&still);
        let mut untouched = still.rng.clone();
        for _ in 0..10 {
            still.drift_cells();
        }
        assert_eq!(positions(&still), before);
        assert_eq!(still.rng.next_f64().to_bits(), untouched.next_f64().to_bits());

        let mut expected = world.rng.clone();
        for _ in 0..2 * ids.len() {
            expected.gaussian(0.0, 1.0);
        }
        world.drift_cells();
        assert_ne!(positions(&world), before);
        assert_eq!(world.rng.next_f64().to_bits(), expected.next_f64().to_bits());
        world.check_invariants().unwrap();
    }

    #[test]
    fn zero_drift_reproduces_the_motionless_world() {
        let mut config = small_config("drift-off", 24, 18);
        config.cell_motion.diffusivity = 0.0;
        let mut stepped = World::new(config).unwrap();
        stepped.spawn_founder_cells(8).unwrap();
        let mut manual = stepped.clone();
        for _ in 0..300 {
            stepped.step();
            manual.diffuse_element_fields();
            manual.step_cells();
            manual.resolve_overlaps();
            manual.enval_phase();
            manual.advance_time();
        }
        assert_eq!(
            bincode::serialize(&stepped).unwrap(),
            bincode::serialize(&manual).unwrap()
        );
    }

    #[test]
    fn drift_wraps_across_seams_and_keeps_spatial_caches_consistent() {
        let (mut world, ids) = drift_world("drift-seam", 2, 2);
        world.config.cell_motion.diffusivity = 1.0;
        for (index, id) in ids.iter().enumerate() {
            let corner = Position::new(
                if index % 2 == 0 { 0.001 } else { 5.999 },
                if index / 2 == 0 { 0.001 } else { 5.999 },
            );
            world.update_cell_position(*id, corner);
        }
        world.check_invariants().unwrap();
        let mut crossings = 0;
        for _ in 0..500 {
            let before = ids.iter().map(|id| world.cells[*id].position).collect::<Vec<_>>();
            world.drift_cells();
            for (index, id) in ids.iter().enumerate() {
                let after = world.cells[*id].position;
                assert!(after.x >= 0.0 && after.x < 6.0 && after.y >= 0.0 && after.y < 6.0);
                if (after.x - before[index].x).abs() > 3.0 || (after.y - before[index].y).abs() > 3.0 {
                    crossings += 1;
                }
            }
            world.check_invariants().unwrap();
        }
        assert!(crossings > 0);
        world.rebuild_spatial_index();
        world.check_invariants().unwrap();
    }

    #[test]
    fn interaction_stats_report_scripted_digestion_exposure_and_contacts() {
        let mut config = only_a_config("interaction-stats", 12, 10);
        config.catalyst_upkeep_per_sec = 0.0;
        config.founder_energy = 50.0;
        config.cell_motion.diffusivity = 0.0;
        let mut world = World::new(config).unwrap();
        let digester = spawn_with(
            &mut world,
            Position::new(3.0, 4.0),
            1,
            vec![surface_enzyme(E_ONLY, A_ONLY, 2.0, 1.0)],
        );
        let prey = spawn_with(&mut world, Position::new(3.9, 4.0), 2, vec![inert_enzyme()]);
        let sibling = spawn_with(&mut world, Position::new(3.0, 4.9), 1, vec![inert_enzyme()]);
        let mut low = inert_enzyme();
        low.exposure = 0.04;
        let mut lower = inert_enzyme();
        lower.exposure = 0.02;
        let loner = spawn_with(&mut world, Position::new(9.0, 8.0), 3, vec![low, lower]);
        let boundary = world.cells[prey].boundary.total() as f32;
        world.cells[prey].boundary = ElementAmounts::new([0.0, 0.0, 0.0, 0.0, boundary * 0.8, 0.0]);
        world.cells[sibling].boundary =
            ElementAmounts::new([boundary * 0.6, 0.0, 0.0, 0.0, boundary * 0.6, 0.0]);

        let fresh = world.stats();
        assert_eq!(fresh.interaction, crate::stats::InteractionCounters::default());
        assert_eq!(fresh.exposed_cell_count, 1);
        assert_eq!(fresh.max_exposure, 1.0);
        assert!((fresh.mean_exposure - 1.06 / 5.0).abs() <= 1.0e-6);
        world.resolve_overlaps();
        let stats = world.compact_stats();
        assert_eq!(stats.contact_pairs, 2);
        assert_eq!(stats.cross_lineage_contact_pairs, 1);

        let inspection = world.inspect_cell(digester).unwrap();
        assert_eq!(
            inspection
                .contacts
                .iter()
                .map(|contact| contact.cell_id)
                .collect::<Vec<_>>(),
            vec![prey.index(), sibling.index()]
        );
        assert_eq!(inspection.contacts[0].lineage_id, LineageId(2));
        assert!(inspection.contacts.iter().all(|contact| contact.share > 0.0));
        let shares = inspection.contacts.iter().map(|contact| contact.share).sum::<f64>();
        assert!((inspection.field_share + shares - 1.0).abs() <= 1.0e-12);
        assert!(world.inspect_cell(loner).unwrap().contacts.is_empty());
        assert_eq!(world.inspect_cell(loner).unwrap().field_share, 1.0);
        assert!((world.inspect_lineage(LineageId(1)).unwrap().mean_exposure - 0.5).abs() <= 1.0e-9);
        assert!((world.inspect_lineage(LineageId(3)).unwrap().mean_exposure - 0.03).abs() <= 1.0e-6);

        for _ in 0..2_000 {
            for id in [digester, prey, sibling] {
                world.step_cell(id);
            }
            world.advance_time();
            if world.cell(prey).is_none() {
                break;
            }
        }
        assert!(world.cell(prey).is_none());
        world.resolve_overlaps();
        let full = world.stats();
        let compact = world.compact_stats();
        let counters = full.interaction;
        assert_eq!(counters, world.interaction_counters());
        assert_eq!(compact.interaction, counters);
        assert_eq!(counters.lyses_after_digestion, 1);
        assert_eq!(counters.field_surface_extent, 0.0);
        assert!(counters.cell_surface_extent > 0.0);
        assert!(counters.cross_lineage_boundary_digested > 0.0);
        assert!(counters.boundary_digested > counters.cross_lineage_boundary_digested);
        assert_eq!(full.contact_pairs, 1);
        assert_eq!(full.cross_lineage_contact_pairs, 0);
        assert_eq!(full.exposed_cell_count, 1);
        assert_eq!(compact.mean_exposure, full.mean_exposure);
        assert!((full.mean_exposure - 1.06 / 4.0).abs() <= 1.0e-6);
        world.check_invariants().unwrap();
    }

    #[test]
    fn cells_without_exposed_enzymes_never_compute_contact_shares() {
        let mut world = World::new(small_config("unexposed-skip", 24, 18)).unwrap();
        world.spawn_founder_cells(12).unwrap();
        world.step_many(200);
        let counters = world.operation_counters();
        assert!(counters.cell_steps > 2_000);
        assert_eq!(counters.surface_contact_queries, 0);
        assert_eq!(world.interaction_counters(), crate::stats::InteractionCounters::default());

        let exposed = live_ids(&world)[0];
        world.cells[exposed].genome.enzymes[0].exposure = 0.2;
        let steps_before = world.operation_counters().cell_steps;
        world.step();
        let counters = world.operation_counters();
        assert!(counters.cell_steps - steps_before > 1);
        assert_eq!(counters.surface_contact_queries, 1);
    }

    mod allocation_counter {
        use std::alloc::{GlobalAlloc, Layout, System};
        use std::cell::Cell;

        pub struct CountingAllocator;

        thread_local! {
            static ARMED: Cell<bool> = const { Cell::new(false) };
            static COUNT: Cell<usize> = const { Cell::new(0) };
        }

        fn record() {
            let _ = ARMED.try_with(|armed| {
                if armed.get() {
                    let _ = COUNT.try_with(|count| count.set(count.get() + 1));
                }
            });
        }

        unsafe impl GlobalAlloc for CountingAllocator {
            unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
                record();
                unsafe { System.alloc(layout) }
            }

            unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
                record();
                unsafe { System.alloc_zeroed(layout) }
            }

            unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
                record();
                unsafe { System.realloc(ptr, layout, new_size) }
            }

            unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
                unsafe { System.dealloc(ptr, layout) }
            }
        }

        pub fn count_allocations(run: impl FnOnce()) -> usize {
            COUNT.with(|count| count.set(0));
            ARMED.with(|armed| armed.set(true));
            run();
            ARMED.with(|armed| armed.set(false));
            COUNT.with(|count| count.get())
        }
    }

    #[global_allocator]
    static ALLOCATOR: allocation_counter::CountingAllocator = allocation_counter::CountingAllocator;

    #[test]
    fn steady_state_steps_allocate_nothing_per_reaction_query_or_pair() {
        let mut config = small_config("steady-state-allocations", 48, 36);
        config.catalyst_upkeep_per_sec = 0.0;
        let mut world = World::new(config).unwrap();
        let slow_growth = |genome: &mut Genome| {
            for engine in &mut genome.enzymes[1..] {
                engine.body_fraction = 0.02;
                engine.boundary_fraction = 0.02;
            }
        };
        for index in 0..80 {
            let mut genome = Genome::random_founder(&mut world.rng, world.avg_enval);
            world.config.founder_energy = 50.0;
            genome.maintenance_cost_per_area = 0.0;
            genome.enzymes[0].exposure = 0.05;
            slow_growth(&mut genome);
            let position = Position::new(
                (index % 10) as f32 * 4.3 + 1.0,
                (index / 10) as f32 * 4.1 + 1.0,
            );
            world
                .spawn_cell_with_genome_at_position(position, genome)
                .unwrap();
            if index % 2 == 0 {
                let mut genome = Genome::random_founder(&mut world.rng, world.avg_enval);
                genome.lineage_id = LineageId(900 + index as u64);
                genome.maintenance_cost_per_area = 0.0;
                slow_growth(&mut genome);
                let neighbour = Position::new(position.x + 0.91, position.y);
                let _ = world.spawn_cell_with_genome_at_position(neighbour, genome);
            }
        }
        world.step_many(150);
        let cells_before = world.cell_count();
        let counters_before = world.operation_counters();
        let interaction_before = world.interaction_counters();

        let allocations = allocation_counter::count_allocations(|| world.step_many(100));

        let counters = world.operation_counters().saturating_delta(counters_before);
        let interaction = world.interaction_counters();
        assert!(counters.reactions_succeeded > 10_000, "{counters:?}");
        assert!(interaction.field_surface_extent > interaction_before.field_surface_extent);
        assert!(interaction.cell_surface_extent > interaction_before.cell_surface_extent);
        assert!(counters.surface_contact_queries > 0);
        assert!(counters.surface_contact_queries < counters.cell_steps);
        assert!(world.contact_pair_count > 0);
        assert!(counters.overlap_candidates > 0 || counters.spatial_candidate_checks > 0);
        assert_eq!(counters.cell_divisions, 0);
        assert_eq!(world.cell_count(), cells_before);
        assert_eq!(
            allocations, 0,
            "steady-state steps allocated {allocations} times"
        );
        world.check_invariants().unwrap();
    }

    fn brute_force_pairs(world: &World, gap: f32) -> Vec<(u32, u32)> {
        let (width, height) = (world.width as f32, world.height as f32);
        let mut pairs = Vec::new();
        for slot_a in 0..world.cells.len() {
            for slot_b in slot_a + 1..world.cells.len() {
                let (a, b) = (world.cells.at_slot(slot_a), world.cells.at_slot(slot_b));
                if toroidal_distance_squared(a.position, b.position, width, height)
                    <= (a.radius + b.radius + gap).powi(2)
                {
                    pairs.push((slot_a as u32, slot_b as u32));
                }
            }
        }
        pairs
    }

    fn brute_force_step(world: &mut World) {
        world.diffuse_element_fields();
        world.contact_list_current = false;
        world.step_cells();
        world.drift_cells();
        world.refresh_max_cell_radius();
        let gap = world.config.surface.contact_gap;
        let contacts = brute_force_pairs(world, gap);
        world.contact_pair_count = contacts.len() as u64;
        world.cross_lineage_contact_pair_count = contacts
            .iter()
            .filter(|(a, b)| {
                world.cells.at_slot(*a as usize).lineage_id
                    != world.cells.at_slot(*b as usize).lineage_id
            })
            .count() as u64;
        if world.cells.len() >= 2 {
            for pass in 0..super::OVERLAP_RELAXATION_PASSES {
                world.contact_pairs = brute_force_pairs(world, if pass == 0 { gap } else { 0.0 });
                world.contact_displacements.clear();
                world.contact_displacements.resize(world.cells.len(), 0.0);
                let corrected = world.overlap_pass();
                world.sync_spatial_index();
                if !corrected {
                    break;
                }
            }
        }
        world.contact_list_current = false;
        world.enval_phase();
        world.advance_time();
    }

    #[test]
    fn shared_contact_list_matches_brute_force_overlap_statistics_and_catalysis() {
        let mut world = World::new(small_config("shared-contact-list", 20, 16)).unwrap();
        world.config.founder_energy = 20.0;
        let mut index = 0;
        for column in 0..7 {
            for row in 0..6 {
                let mut genome = Genome::random_founder(&mut world.rng, world.avg_enval);
                genome.lineage_id = LineageId(1 + (column * 6 + row) as u64 % 3);
                for engine in &mut genome.enzymes[1..] {
                    engine.body_fraction = 0.02;
                    engine.boundary_fraction = 0.02;
                }
                if index % 2 == 0 {
                    genome.enzymes[0].exposure = 0.2;
                    genome.enzymes[0].reactants =
                        ElementAmounts::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
                    genome.enzymes[0].products =
                        ElementAmounts::new([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
                }
                let position = Position::new(
                    3.0 + column as f32 * 0.93 + (row % 2) as f32 * 0.4,
                    3.0 + row as f32 * 0.88,
                );
                world
                    .spawn_cell_with_genome_at_position(position, genome)
                    .unwrap();
                index += 1;
            }
        }
        world.config.cell_motion.diffusivity = 0.5;
        let mut brute = world.clone();
        let mut overlapping_ticks = 0;
        let mut contact_sum = 0;
        let mut loose_ticks = 0;
        for tick in 0..200 {
            if tick % 20 == 10 {
                for target in [&mut world, &mut brute] {
                    let ids = live_ids(target);
                    let (mover, anchor) = (ids[tick % ids.len()], ids[(tick + 7) % ids.len()]);
                    let anchor_position = target.cells[anchor].position;
                    target.cells[mover].position =
                        Position::new(anchor_position.x + 0.05, anchor_position.y)
                            .wrapped(target.width as f32, target.height as f32)
                            .unwrap();
                    target.rebuild_spatial_index();
                }
            }
            world.step();
            brute_force_step(&mut brute);
            assert!(world.contact_list_current);
            assert_eq!(
                world.contact_pair_count, brute.contact_pair_count,
                "tick {tick}"
            );
            assert_eq!(
                world.cross_lineage_contact_pair_count,
                brute.cross_lineage_contact_pair_count
            );
            for (listed, reference) in world.cells.iter().zip(brute.cells.iter()) {
                assert_eq!(listed.id, reference.id);
                assert_eq!(listed.position, reference.position, "tick {tick}");
                assert_eq!(listed.energy.to_bits(), reference.energy.to_bits());
            }
            contact_sum += world.contact_pair_count;
            if !world.contact_loose_extras.is_empty() {
                loose_ticks += 1;
            }
            if world.operation_counters.overlap_corrections > 0 {
                overlapping_ticks += 1;
            }
            world.operation_counters.overlap_corrections = 0;
        }
        assert_eq!(world.reaction_counters.divisions, 0);
        assert!(
            contact_sum > 200 * 20,
            "{contact_sum} contacts over the run"
        );
        assert!(loose_ticks >= 10, "{loose_ticks} ticks with loose cells");
        assert!(
            overlapping_ticks > 20,
            "{overlapping_ticks} ticks with overlaps"
        );
        assert!(world.interaction_counters.cell_surface_extent > 0.0);
        assert_eq!(
            world.interaction_counters.cell_surface_extent.to_bits(),
            brute.interaction_counters.cell_surface_extent.to_bits()
        );
        world.operation_counters = Default::default();
        brute.operation_counters = Default::default();
        assert_eq!(
            crate::snapshot::to_bytes(&world).unwrap(),
            crate::snapshot::to_bytes(&brute).unwrap()
        );
    }

    #[test]
    fn overlap_mechanics_are_symmetric_wrapped_and_do_not_consume_rng() {
        let mut world = World::new(only_a_config("overlap-mechanics", 10, 8)).unwrap();
        let first_genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        let first = world
            .spawn_cell_with_genome_at_position(Position::new(0.1, 2.0), first_genome)
            .unwrap();
        let second_genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        let second = world
            .spawn_cell_with_genome_at_position(Position::new(1.1, 2.0), second_genome)
            .unwrap();
        world.cells[second].position = Position::new(9.8, 2.0);
        world.rebuild_spatial_index();
        let before = toroidal_distance_squared(
            world.cells[first].position,
            world.cells[second].position,
            world.width as f32,
            world.height as f32,
        );
        let first_before = world.cells[first].position;
        let second_before = world.cells[second].position;
        let mut expected_rng = world.rng.clone();
        world.resolve_overlaps();
        let first_movement = minimum_image_displacement(
            first_before,
            world.cells[first].position,
            world.width as f32,
            world.height as f32,
        );
        let second_movement = minimum_image_displacement(
            second_before,
            world.cells[second].position,
            world.width as f32,
            world.height as f32,
        );
        let after = toroidal_distance_squared(
            world.cells[first].position,
            world.cells[second].position,
            world.width as f32,
            world.height as f32,
        );
        assert!(after > before);
        assert!((first_movement.x + second_movement.x).abs() <= 1.0e-6);
        assert!((first_movement.y + second_movement.y).abs() <= 1.0e-6);
        for cell_id in [first, second] {
            let position = world.cells[cell_id].position;
            assert!(position.x >= 0.0 && position.x < world.width as f32);
            assert!(position.y >= 0.0 && position.y < world.height as f32);
        }
        assert_eq!(
            world.rng.next_f64().to_bits(),
            expected_rng.next_f64().to_bits()
        );
        world.cells[first].position = Position::new(4.0, 4.0);
        world.cells[second].position = Position::new(4.0, 4.0);
        world.rebuild_spatial_index();
        let mut replay = world.clone();
        world.resolve_overlaps();
        replay.resolve_overlaps();
        assert_eq!(world.cells[first].position, replay.cells[first].position);
        assert_eq!(world.cells[second].position, replay.cells[second].position);
        assert!(
            toroidal_distance_squared(
                world.cells[first].position,
                world.cells[second].position,
                world.width as f32,
                world.height as f32,
            ) > 0.0
        );
        world.check_invariants().unwrap();
        replay.check_invariants().unwrap();
    }
}
