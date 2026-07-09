use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::error::Error;
use std::fmt;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::bio::{self, GenomeReactionContext, ReactionEnv};
use crate::cell::{
    Cell, CellCountError, CellId, CellState, ReactionMoleculeSummary, ReactionRecord,
};
use crate::chem::{Composition, Element, ELEMENT_COUNT, ELEMENT_ORDER};
use crate::config::{Config, ConfigError};
use crate::genome::{
    Enzyme, EnzymeType, Genome, GenomePatch, LineageId, MAX_CELL_ENZYMES, MIN_CELL_ENZYMES,
};
use crate::molecule::{Molecule, MoleculeError};
use crate::render_buffers::{RenderBuffers, EMPTY_CELL_ID};
use crate::rng::Rng;
use crate::stats::{
    EnzymeTypeCounts, OperationCounters, ReactionCounters, StepProfile, WorldStats,
    ENZYME_COUNT_HISTOGRAM_LEN,
};

const LOCAL_ENVAL_RADIUS: usize = 2;
const LOCAL_ENVAL_WINDOW_DIAMETER: usize = LOCAL_ENVAL_RADIUS * 2 + 1;
const LOCAL_ENVAL_WINDOW_AREA: usize = LOCAL_ENVAL_WINDOW_DIAMETER * LOCAL_ENVAL_WINDOW_DIAMETER;
const MOORE_WITH_CENTER_DX: [isize; 9] = [-1, -1, -1, 0, 0, 1, 1, 1, 0];
const MOORE_WITH_CENTER_DY: [isize; 9] = [-1, 0, 1, -1, 1, -1, 0, 1, 0];
const MAX_REACTION_SUBSTRATES: usize = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TileId(pub usize);

impl TileId {
    pub const fn index(self) -> usize {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MoleculeId(pub usize);

impl MoleculeId {
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

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Tile {
    molecules: Vec<MoleculeId>,
    cell: Option<CellId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MoleculeOwner {
    Tile(TileId),
    Cell(CellId),
    Free,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct MoleculeRecord {
    molecule: Molecule,
    owner: MoleculeOwner,
    owner_slot: usize,
    diffusion_phase: Option<u32>,
    last_diffusion_dir: Option<u8>,
    diffusion_tick: u32,
    wheel_index: Option<usize>,
    wheel_pos: Option<usize>,
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
    pub cell: Option<CellId>,
    pub molecule_count: usize,
    pub mass_count: u32,
    pub element_counts: [u32; ELEMENT_COUNT],
    pub element_mask: u8,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CellInspection {
    pub cell_id: CellId,
    pub tile_id: TileId,
    pub x: usize,
    pub y: usize,
    pub energy: f64,
    pub lineage_id: LineageId,
    pub enzyme_count: usize,
    pub internal_atom_count: u32,
    pub combat_attack_total: u32,
    pub combat_defense_total: u32,
    pub age_seconds: f64,
    pub optimal_enval: f32,
    pub local_enval_average: f32,
    pub repro_threshold: f64,
    pub decay_time: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MoleculeDetailInspection {
    pub list_index: usize,
    pub molecule_id: usize,
    pub composition_counts: [u16; ELEMENT_COUNT],
    pub formula: String,
    pub size: u16,
    pub element_mask: u8,
    pub bond_multiplier: f32,
    pub elemental_energy_sum: f32,
    pub energy: f32,
    pub polarity: f32,
    pub diffusion_rate: f32,
    pub diffusion_period: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MoleculeListInspection {
    pub molecule_count: usize,
    pub atom_count: u32,
    pub element_counts: [u32; ELEMENT_COUNT],
    pub limit: usize,
    pub truncated: bool,
    pub molecules: Vec<MoleculeDetailInspection>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EnzymeDetailInspection {
    pub index: usize,
    pub enzyme_type: &'static str,
    pub is_metabolic: bool,
    pub is_combat: bool,
    pub specificity_mask: u8,
    pub specificity_elements: Vec<&'static str>,
    pub bond_multiplier: f32,
    pub bond_cost_fraction: f32,
    pub bond_harvest_fraction: f32,
    pub downhill_harvest_fraction: f32,
    pub secretion_prob: f32,
    pub enval_sigma: f32,
    pub enval_throughput: f32,
    pub enval_energy_fraction: f32,
    pub enval_release_fraction: f32,
    pub enval_pump: f32,
    pub combat_level: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GenomeDetailInspection {
    pub optimal_enval: f32,
    pub repro_threshold: f64,
    pub initial_energy: f64,
    pub decay_time: f64,
    pub default_secretion_prob: f32,
    pub mutation_rate: f32,
    pub post_divide_mortality: f32,
    pub desired_element_reserve: u32,
    pub enval_stress_factor: f64,
    pub enval_mutation_floor: f32,
    pub maintenance_cost_per_sec: f64,
    pub lineage_id: LineageId,
    pub enzyme_count: usize,
    pub min_cell_enzymes: usize,
    pub max_cell_enzymes: usize,
    pub enzymes: Vec<EnzymeDetailInspection>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ReactionLogInspection {
    pub available: bool,
    pub reason: &'static str,
    pub limit: usize,
    pub truncated: bool,
    pub reaction_count: usize,
    pub returned_count: usize,
    pub order: &'static str,
    pub reactions: Vec<ReactionRecord>,
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
    pub state: &'static str,
    pub time_without_food: f64,
    pub maintenance_cost_per_sec: f64,
    pub death_sim_time: Option<f64>,
    pub genome: GenomeDetailInspection,
    pub internal: MoleculeListInspection,
    pub recent_reactions: ReactionLogInspection,
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
    pub average_attack_total: f64,
    pub average_defense_total: f64,
    pub max_attack_total: u32,
    pub max_defense_total: u32,
    pub cells_with_attackase: u64,
    pub cells_with_defensase: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LineageListInspection {
    pub extant_lineage_count: usize,
    pub total_lineage_records: usize,
    pub limit: usize,
    pub truncated: bool,
    pub lineages: Vec<LineageSummaryInspection>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PredationOutcome {
    winner: CellId,
    loser: CellId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SampledSubstrateIds {
    ids: [MoleculeId; MAX_REACTION_SUBSTRATES],
    len: usize,
}

impl SampledSubstrateIds {
    fn new() -> Self {
        Self {
            ids: [MoleculeId(0); MAX_REACTION_SUBSTRATES],
            len: 0,
        }
    }

    fn is_empty(self) -> bool {
        self.len == 0
    }

    fn len(self) -> usize {
        self.len
    }

    fn push(&mut self, molecule_id: MoleculeId) {
        debug_assert!(self.len < MAX_REACTION_SUBSTRATES);
        self.ids[self.len] = molecule_id;
        self.len += 1;
    }

    fn replace(&mut self, index: usize, molecule_id: MoleculeId) {
        debug_assert!(index < self.len);
        self.ids[index] = molecule_id;
    }

    fn first(self) -> Option<MoleculeId> {
        if self.len == 0 {
            None
        } else {
            Some(self.ids[0])
        }
    }

    fn iter(self) -> impl Iterator<Item = MoleculeId> {
        (0..self.len).map(move |index| self.ids[index])
    }
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
    diffusion_tick: u32,
    base_enval: f32,
    enval_sum: f64,
    avg_enval: f32,
    enval: Vec<f32>,
    enval_next: Vec<f32>,
    tiles: Vec<Tile>,
    molecules: Vec<MoleculeRecord>,
    free_molecule_ids: Vec<MoleculeId>,
    molecule_arena_high_water_mark: usize,
    cells: Vec<Cell>,
    active_cells: Vec<CellId>,
    lineage_counters: BTreeMap<LineageId, LineageCounters>,
    birth_count: u64,
    death_count: u64,
    predation_event_count: u64,
    cells_consumed_count: u64,
    predator_energy_gained: f64,
    predation_enzyme_transfer_count: u64,
    predation_enzyme_replacement_count: u64,
    reaction_counters: ReactionCounters,
    operation_counters: OperationCounters,
    tile_element_counts: Vec<[u32; ELEMENT_COUNT]>,
    #[serde(skip, default)]
    tile_element_masks: Vec<u8>,
    tile_mass_counts: Vec<u32>,
    tile_diffusion_rotor: Vec<u8>,
    neighbors: Vec<NeighborIndices>,
    #[serde(skip, default)]
    predation_pairs: Vec<(TileId, TileId)>,
    #[serde(skip, default)]
    predation_occupied_tiles: Vec<TileId>,
    diffusion_wheel: Vec<Vec<MoleculeId>>,
}

impl World {
    pub fn new(config: Config) -> Result<Self, WorldError> {
        config.validate()?;
        let tile_count = config.tile_count().ok_or(ConfigError::TileCountOverflow)?;
        let mut rng = Rng::from_seed_str(&config.seed);
        let base_enval = rng.range(-1.0, 1.0);
        let enval_sum = f64::from(base_enval) * tile_count as f64;
        let neighbors = build_neighbors(config.width, config.height);
        let predation_pairs = build_predation_pairs(&neighbors);

        let mut world = Self {
            width: config.width,
            height: config.height,
            tile_count,
            tick_count: 0,
            sim_time_seconds: 0.0,
            diffusion_tick: 0,
            base_enval,
            enval_sum,
            avg_enval: (enval_sum / tile_count as f64) as f32,
            enval: vec![base_enval; tile_count],
            enval_next: vec![base_enval; tile_count],
            tiles: vec![Tile::default(); tile_count],
            molecules: Vec::new(),
            free_molecule_ids: Vec::new(),
            molecule_arena_high_water_mark: 0,
            cells: Vec::new(),
            active_cells: Vec::new(),
            lineage_counters: BTreeMap::new(),
            birth_count: 0,
            death_count: 0,
            predation_event_count: 0,
            cells_consumed_count: 0,
            predator_energy_gained: 0.0,
            predation_enzyme_transfer_count: 0,
            predation_enzyme_replacement_count: 0,
            reaction_counters: ReactionCounters::default(),
            operation_counters: OperationCounters::default(),
            tile_element_counts: vec![[0; ELEMENT_COUNT]; tile_count],
            tile_element_masks: vec![0; tile_count],
            tile_mass_counts: vec![0; tile_count],
            tile_diffusion_rotor: (0..tile_count).map(initial_tile_diffusion_rotor).collect(),
            neighbors,
            predation_pairs,
            predation_occupied_tiles: Vec::new(),
            diffusion_wheel: vec![Vec::new(); config.molecule_diffusion_wheel_size],
            rng,
            config,
        };
        world.seed_initial_molecules()?;
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

    pub fn molecule_count(&self) -> usize {
        self.molecules
            .iter()
            .filter(|record| record.owner != MoleculeOwner::Free)
            .count()
    }

    pub fn cell_count(&self) -> usize {
        self.active_cells.len()
    }

    pub fn tick_count(&self) -> u64 {
        self.tick_count
    }

    pub fn sim_time_seconds(&self) -> f64 {
        self.sim_time_seconds
    }

    pub fn base_enval(&self) -> f32 {
        self.base_enval
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

    pub fn predation_enabled(&self) -> bool {
        self.config.predation_enabled
    }

    pub fn set_predation_enabled(&mut self, enabled: bool) {
        self.config.predation_enabled = enabled;
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

    pub fn set_all_live_cell_energy(&mut self, energy: f64) -> Result<(), WorldError> {
        if !energy.is_finite() || energy < 0.0 {
            return Err(WorldError::InvalidEnergyInput(energy));
        }
        for cell_id in self.active_cells.iter().copied() {
            if let Some(cell) = self.cells.get_mut(cell_id.index()) {
                if cell.state == CellState::Active {
                    cell.energy = energy;
                }
            }
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
        self.enval_sum += f64::from(value) - f64::from(old);
        self.avg_enval = (self.enval_sum / self.tile_count as f64) as f32;
        Ok(())
    }

    pub fn adjust_tile_enval(&mut self, tile_id: TileId, delta: f32) -> Result<(), WorldError> {
        if tile_id.index() >= self.tile_count {
            return Err(WorldError::InvalidTile(tile_id));
        }
        if !delta.is_finite() || delta == 0.0 {
            return Ok(());
        }
        let value = self.enval[tile_id.index()] + delta;
        if !value.is_finite() {
            return Err(WorldError::NonFiniteEnvalInput(value));
        }
        self.enval[tile_id.index()] = value;
        self.enval_next[tile_id.index()] = value;
        self.enval_sum += f64::from(delta);
        self.avg_enval = (self.enval_sum / self.tile_count as f64) as f32;
        Ok(())
    }

    pub fn set_all_enval(&mut self, value: f32) -> Result<(), WorldError> {
        if !value.is_finite() {
            return Err(WorldError::NonFiniteEnvalInput(value));
        }
        self.enval.fill(value);
        self.enval_next.fill(value);
        self.enval_sum = f64::from(value) * self.tile_count as f64;
        self.avg_enval = (self.enval_sum / self.tile_count as f64) as f32;
        Ok(())
    }

    pub fn tile_molecules(&self, tile_id: TileId) -> Option<&[MoleculeId]> {
        self.tiles
            .get(tile_id.index())
            .map(|tile| tile.molecules.as_slice())
    }

    pub fn tile_cell(&self, tile_id: TileId) -> Option<CellId> {
        self.tiles.get(tile_id.index()).and_then(|tile| tile.cell)
    }

    pub fn molecule(&self, molecule_id: MoleculeId) -> Option<&Molecule> {
        self.molecules
            .get(molecule_id.index())
            .map(|record| &record.molecule)
    }

    pub fn cell(&self, cell_id: CellId) -> Option<&Cell> {
        self.cells.get(cell_id.index())
    }

    pub fn tile_element_counts(&self, tile_id: TileId) -> Option<[u32; ELEMENT_COUNT]> {
        self.tile_element_counts.get(tile_id.index()).copied()
    }

    pub fn tile_mass_count(&self, tile_id: TileId) -> Option<u32> {
        self.tile_mass_counts.get(tile_id.index()).copied()
    }

    pub fn molecule_tile_ids(&self) -> Vec<TileId> {
        self.molecules
            .iter()
            .filter_map(|record| match record.owner {
                MoleculeOwner::Tile(tile_id) => Some(tile_id),
                MoleculeOwner::Cell(_) | MoleculeOwner::Free => None,
            })
            .collect()
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
        let Some(tile_id) = self.random_empty_tile(50) else {
            return Ok(None);
        };
        let genome = Genome::random_founder(&mut self.rng, self.avg_enval);
        self.spawn_cell_with_genome_at(tile_id, genome).map(Some)
    }

    pub fn spawn_cell_with_genome_at(
        &mut self,
        tile_id: TileId,
        mut genome: Genome,
    ) -> Result<CellId, WorldError> {
        if tile_id.index() >= self.tile_count {
            return Err(WorldError::InvalidTile(tile_id));
        }
        if self.tiles[tile_id.index()].cell.is_some() {
            return Err(WorldError::OccupiedTile(tile_id));
        }
        genome.enforce_enzyme_count_bounds(&mut self.rng);
        let cell_id = CellId(self.cells.len());
        let mut cell = Cell::new(genome, tile_id, self.sim_time_seconds);
        cell.active_slot = Some(self.active_cells.len());
        self.tiles[tile_id.index()].cell = Some(cell_id);
        self.active_cells.push(cell_id);
        self.record_lineage_birth(cell.lineage_id);
        self.birth_count = self.birth_count.saturating_add(1);
        self.cells.push(cell);
        Ok(cell_id)
    }

    pub fn step(&mut self) {
        self.diffuse_molecules();
        self.step_cells();
        self.resolve_predation();
        self.diffuse_enval();
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
        self.diffuse_molecules();
        let molecule_diffusion = start.elapsed();

        let start = Instant::now();
        self.step_cells();
        let cell_step = start.elapsed();

        let start = Instant::now();
        self.resolve_predation();
        let predation = start.elapsed();

        let start = Instant::now();
        self.diffuse_enval();
        let enval_diffusion = start.elapsed();

        self.advance_time();

        StepProfile {
            molecule_diffusion,
            cell_step,
            predation,
            enval_diffusion,
            total: total_start.elapsed(),
            counters: self.operation_counters.saturating_delta(counters_before),
        }
    }

    fn advance_time(&mut self) {
        self.tick_count = self.tick_count.wrapping_add(1);
        self.sim_time_seconds += self.config.dt_seconds;
    }

    pub fn diffuse_enval(&mut self) {
        let alpha = f64::from(self.config.enval_diffusion_alpha);
        let one_minus_alpha = 1.0 - alpha;
        let inv_9 = 1.0 / 9.0;

        for i in 0..self.tile_count {
            let neighbors = self.neighbors[i];
            let center = f64::from(self.enval[i]);
            let sum = center
                + f64::from(self.enval[neighbors.left.index()])
                + f64::from(self.enval[neighbors.right.index()])
                + f64::from(self.enval[neighbors.up.index()])
                + f64::from(self.enval[neighbors.down.index()])
                + f64::from(self.enval[neighbors.up_left.index()])
                + f64::from(self.enval[neighbors.up_right.index()])
                + f64::from(self.enval[neighbors.down_left.index()])
                + f64::from(self.enval[neighbors.down_right.index()]);
            let value = alpha * (sum * inv_9) + one_minus_alpha * center;
            self.enval_next[i] = if value.is_finite() { value as f32 } else { 0.0 };
        }

        let mut sum = 0.0_f64;
        for value in &self.enval_next {
            sum += f64::from(*value);
        }
        std::mem::swap(&mut self.enval, &mut self.enval_next);
        self.enval_sum = sum;
        self.avg_enval = (sum / self.tile_count as f64) as f32;
    }

    pub fn diffuse_molecules(&mut self) {
        self.diffusion_tick = self.diffusion_tick.wrapping_add(1);
        let tick = self.diffusion_tick;
        let wheel_mask = self.config.molecule_diffusion_wheel_size - 1;
        let bucket_index = (tick as usize) & wheel_mask;
        let mut i = 0;
        while i < self.diffusion_wheel[bucket_index].len() {
            let molecule_id = self.diffusion_wheel[bucket_index][i];
            if molecule_id.index() >= self.molecules.len() {
                let _ = self.remove_wheel_at(bucket_index, i);
                continue;
            }
            if self.molecules[molecule_id.index()].diffusion_tick != tick {
                i += 1;
                continue;
            }
            let removed = self.remove_wheel_at(bucket_index, i);
            if let Some(id) = removed {
                self.operation_counters.molecule_diffusion_events = self
                    .operation_counters
                    .molecule_diffusion_events
                    .saturating_add(1);
                self.try_diffuse_molecule(id);
            }
        }
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

        let mut element_counts = [0_u64; ELEMENT_COUNT];
        for counts in &self.tile_element_counts {
            for element in ELEMENT_ORDER {
                element_counts[element.index()] += u64::from(counts[element.index()]);
            }
        }
        for cell in &self.cells {
            if cell.state != CellState::Active {
                continue;
            }
            for element in ELEMENT_ORDER {
                element_counts[element.index()] +=
                    u64::from(cell.internal_element_counts[element.index()]);
            }
        }

        let mut tile_molecule_count = 0_usize;
        let mut cell_molecule_count = 0_usize;
        let mut free_molecule_record_count = 0_usize;
        let mut tile_atom_count = 0_u64;
        let mut cell_atom_count = 0_u64;
        for record in &self.molecules {
            match record.owner {
                MoleculeOwner::Tile(_) => {
                    tile_molecule_count += 1;
                    tile_atom_count += u64::from(record.molecule.size);
                }
                MoleculeOwner::Cell(_) => {
                    cell_molecule_count += 1;
                    cell_atom_count += u64::from(record.molecule.size);
                }
                MoleculeOwner::Free => free_molecule_record_count += 1,
            }
        }
        let molecule_count = tile_molecule_count + cell_molecule_count;

        let occupied_tile_count = self.tiles.iter().filter(|tile| tile.cell.is_some()).count();
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
        let mut time_without_food_sum = 0.0_f64;
        let mut enzyme_sum = 0_usize;
        let mut min_enzyme_count = usize::MAX;
        let mut max_enzyme_count = 0_usize;
        let mut enzyme_count_histogram = [0_u64; ENZYME_COUNT_HISTOGRAM_LEN];
        let mut cells_at_enzyme_cap = 0_usize;
        let mut cells_with_attackase = 0_usize;
        let mut cells_with_defensase = 0_usize;
        let mut attack_sum = 0_u64;
        let mut defense_sum = 0_u64;
        let mut max_attack_total = 0_u32;
        let mut max_defense_total = 0_u32;
        let mut enzyme_type_totals = EnzymeTypeCounts::default();

        for cell_id in &self.active_cells {
            if let Some(cell) = self.cells.get(cell_id.index()) {
                if cell.state != CellState::Active {
                    continue;
                }
                live_cell_count += 1;
                energy_sum += cell.energy;
                min_cell_energy = min_cell_energy.min(cell.energy);
                max_cell_energy = max_cell_energy.max(cell.energy);
                let age = (self.sim_time_seconds - cell.birth_sim_time).max(0.0);
                age_sum += age;
                max_cell_age = max_cell_age.max(age);
                time_without_food_sum += cell.time_without_food;

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

                let mut has_attackase = false;
                let mut has_defensase = false;
                for enzyme in &cell.genome.enzymes {
                    enzyme_type_totals.increment(enzyme.enzyme_type);
                    match enzyme.enzyme_type {
                        EnzymeType::Attackase => has_attackase = true,
                        EnzymeType::Defensase => has_defensase = true,
                        EnzymeType::Anabolase
                        | EnzymeType::Catabolase
                        | EnzymeType::Transmutase => {}
                    }
                }
                if has_attackase {
                    cells_with_attackase += 1;
                }
                if has_defensase {
                    cells_with_defensase += 1;
                }

                attack_sum = attack_sum.saturating_add(u64::from(cell.combat_attack_total));
                defense_sum = defense_sum.saturating_add(u64::from(cell.combat_defense_total));
                max_attack_total = max_attack_total.max(cell.combat_attack_total);
                max_defense_total = max_defense_total.max(cell.combat_defense_total);
            }
        }

        if live_cell_count == 0 {
            min_cell_energy = 0.0;
            max_cell_energy = 0.0;
            min_enzyme_count = 0;
        }

        let dead_cell_count = self
            .cells
            .iter()
            .filter(|cell| cell.state == CellState::Dead)
            .count();
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
            molecule_count,
            tile_molecule_count,
            cell_molecule_count,
            free_molecule_record_count,
            active_molecule_record_count: molecule_count,
            molecule_arena_len: self.molecules.len(),
            molecule_arena_high_water_mark: self
                .molecule_arena_high_water_mark
                .max(self.molecules.len()),
            molecule_slots_reused: self.operation_counters.molecule_slots_reused,
            molecule_slots_newly_allocated: self.operation_counters.molecule_slots_newly_allocated,
            total_atom_count: tile_atom_count + cell_atom_count,
            tile_atom_count,
            cell_atom_count,
            average_molecules_per_tile: if self.tile_count > 0 {
                tile_molecule_count as f64 / self.tile_count as f64
            } else {
                0.0
            },
            average_internal_molecules_per_live_cell: if live_cell_count > 0 {
                cell_molecule_count as f64 / live_cell_count_f64
            } else {
                0.0
            },
            average_atoms_per_live_cell: if live_cell_count > 0 {
                cell_atom_count as f64 / live_cell_count_f64
            } else {
                0.0
            },
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
            element_counts,
            cell_count: live_cell_count,
            live_cell_count,
            cell_record_count: self.cells.len(),
            dead_cell_count,
            births: self.birth_count,
            deaths: self.death_count,
            predation_events: self.predation_event_count,
            cells_consumed: self.cells_consumed_count,
            predator_energy_gained: self.predator_energy_gained,
            average_energy_gained_per_predation: if self.predation_event_count > 0 {
                self.predator_energy_gained / self.predation_event_count as f64
            } else {
                0.0
            },
            predation_enzyme_transfers: self.predation_enzyme_transfer_count,
            predation_enzyme_replacements: self.predation_enzyme_replacement_count,
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
            average_time_without_food: if live_cell_count > 0 {
                time_without_food_sum / live_cell_count_f64
            } else {
                0.0
            },
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
            cells_with_attackase,
            cells_with_defensase,
            average_attack_total: if live_cell_count > 0 {
                attack_sum as f64 / live_cell_count_f64
            } else {
                0.0
            },
            max_attack_total,
            average_defense_total: if live_cell_count > 0 {
                defense_sum as f64 / live_cell_count_f64
            } else {
                0.0
            },
            max_defense_total,
            enzyme_type_totals,
            reaction_counters: self.reaction_counters,
            operation_counters: self.operation_counters,
        }
    }

    pub fn rebuild_derived_caches(&mut self) {
        self.neighbors = build_neighbors(self.width, self.height);
        self.predation_pairs = build_predation_pairs(&self.neighbors);
        self.predation_occupied_tiles.clear();
        self.tile_element_masks = self
            .tile_element_counts
            .iter()
            .copied()
            .map(element_mask_from_counts)
            .collect();
        self.molecule_arena_high_water_mark = self
            .molecule_arena_high_water_mark
            .max(self.molecules.len());
        for cell in &mut self.cells {
            cell.refresh_combat_totals();
            cell.refresh_internal_element_mask();
        }
    }

    pub fn build_render_buffers(&self) -> RenderBuffers {
        let mut buffers = RenderBuffers::default();
        self.write_render_buffers(&mut buffers);
        buffers
    }

    pub fn write_render_buffers(&self, buffers: &mut RenderBuffers) {
        buffers.clear();
        buffers.width = self.width.min(u32::MAX as usize) as u32;
        buffers.height = self.height.min(u32::MAX as usize) as u32;
        buffers.tick_count = self.tick_count;
        buffers.sim_time_seconds = self.sim_time_seconds;
        buffers.render_epoch = (self.tick_count & u64::from(u32::MAX)) as u32;

        buffers.tile_enval.reserve(self.tile_count);
        buffers.tile_occupancy.reserve(self.tile_count);
        buffers.tile_mass.reserve(self.tile_count);
        buffers.tile_molecule_count.reserve(self.tile_count);
        buffers.tile_element_mask.reserve(self.tile_count);
        buffers.cell_id.reserve(self.active_cells.len());
        buffers.cell_x.reserve(self.active_cells.len());
        buffers.cell_y.reserve(self.active_cells.len());
        buffers.cell_energy.reserve(self.active_cells.len());
        buffers.cell_lineage.reserve(self.active_cells.len());
        buffers.cell_flags.reserve(self.active_cells.len());
        buffers.cell_enzyme_count.reserve(self.active_cells.len());
        buffers.cell_age_seconds.reserve(self.active_cells.len());
        buffers.cell_attack.reserve(self.active_cells.len());
        buffers.cell_defense.reserve(self.active_cells.len());

        buffers.tile_enval.extend_from_slice(&self.enval);

        for tile_index in 0..self.tile_count {
            let tile = &self.tiles[tile_index];
            buffers.tile_occupancy.push(
                tile.cell
                    .map(|cell_id| cell_id.index().min(u32::MAX as usize) as u32)
                    .unwrap_or(EMPTY_CELL_ID),
            );
            buffers.tile_mass.push(self.tile_mass_counts[tile_index]);
            buffers
                .tile_molecule_count
                .push(tile.molecules.len().min(u32::MAX as usize) as u32);
            buffers
                .tile_element_mask
                .push(u32::from(self.tile_element_masks[tile_index]));
        }

        for cell_id in self.active_cells.iter().copied() {
            let Some(cell) = self.cells.get(cell_id.index()) else {
                continue;
            };
            if cell.state != CellState::Active {
                continue;
            }
            let Some(tile_id) = cell.tile_id else {
                continue;
            };
            let Some((x, y)) = self.tile_xy(tile_id) else {
                continue;
            };
            buffers
                .cell_id
                .push(cell_id.index().min(u32::MAX as usize) as u32);
            buffers.cell_x.push(x.min(u32::MAX as usize) as u32);
            buffers.cell_y.push(y.min(u32::MAX as usize) as u32);
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
            buffers.cell_attack.push(cell.combat_attack_total);
            buffers.cell_defense.push(cell.combat_defense_total);
        }
    }

    pub fn inspect_tile(&self, tile_id: TileId) -> Option<TileInspection> {
        if tile_id.index() >= self.tile_count {
            return None;
        }
        let (x, y) = self.tile_xy(tile_id)?;
        let element_counts = self.tile_element_counts[tile_id.index()];
        Some(TileInspection {
            tile_id,
            x,
            y,
            enval: self.enval[tile_id.index()],
            cell: self.tiles[tile_id.index()].cell,
            molecule_count: self.tiles[tile_id.index()].molecules.len(),
            mass_count: self.tile_mass_counts[tile_id.index()],
            element_counts,
            element_mask: self.tile_element_masks[tile_id.index()],
        })
    }

    pub fn inspect_tile_xy(&self, x: usize, y: usize) -> Option<TileInspection> {
        self.tile_id(x, y)
            .and_then(|tile_id| self.inspect_tile(tile_id))
    }

    pub fn inspect_cell(&self, cell_id: CellId) -> Option<CellInspection> {
        let cell = self.cells.get(cell_id.index())?;
        if cell.state != CellState::Active {
            return None;
        }
        let tile_id = cell.tile_id?;
        let (x, y) = self.tile_xy(tile_id)?;
        Some(CellInspection {
            cell_id,
            tile_id,
            x,
            y,
            energy: cell.energy,
            lineage_id: cell.lineage_id,
            enzyme_count: cell.genome.enzymes.len(),
            internal_atom_count: cell.internal_atom_count,
            combat_attack_total: cell.combat_attack_total,
            combat_defense_total: cell.combat_defense_total,
            age_seconds: (self.sim_time_seconds - cell.birth_sim_time).max(0.0),
            optimal_enval: cell.genome.optimal_enval,
            local_enval_average: self.default_local_enval_average(tile_id).unwrap_or(0.0),
            repro_threshold: cell.genome.repro_threshold,
            decay_time: cell.genome.decay_time,
        })
    }

    pub fn inspect_cell_detail(
        &self,
        cell_id: CellId,
        molecule_limit: usize,
        reaction_limit: usize,
    ) -> Option<CellDetailInspection> {
        let cell = self.cells.get(cell_id.index())?;
        if cell.state != CellState::Active {
            return None;
        }
        let summary = self.inspect_cell(cell_id)?;
        Some(CellDetailInspection {
            cell: summary,
            state: cell_state_label(cell.state),
            time_without_food: cell.time_without_food,
            maintenance_cost_per_sec: cell.maintenance_cost_per_sec,
            death_sim_time: cell.death_sim_time,
            genome: inspect_genome(&cell.genome),
            internal: self.inspect_cell_molecules(cell_id, molecule_limit)?,
            recent_reactions: self.inspect_cell_reactions(cell_id, reaction_limit)?,
        })
    }

    pub fn inspect_cell_molecules(
        &self,
        cell_id: CellId,
        limit: usize,
    ) -> Option<MoleculeListInspection> {
        let cell = self.cells.get(cell_id.index())?;
        if cell.state != CellState::Active {
            return None;
        }
        let molecules = cell
            .molecules
            .iter()
            .copied()
            .take(limit)
            .enumerate()
            .filter_map(|(list_index, molecule_id)| {
                self.molecules
                    .get(molecule_id.index())
                    .map(|record| inspect_molecule(list_index, molecule_id, record))
            })
            .collect::<Vec<_>>();
        Some(MoleculeListInspection {
            molecule_count: cell.molecules.len(),
            atom_count: cell.internal_atom_count,
            element_counts: cell.internal_element_counts,
            limit,
            truncated: cell.molecules.len() > limit,
            molecules,
        })
    }

    pub fn inspect_cell_reactions(
        &self,
        cell_id: CellId,
        limit: usize,
    ) -> Option<ReactionLogInspection> {
        let cell = self.cells.get(cell_id.index())?;
        if cell.state != CellState::Active {
            return None;
        }
        let reaction_count = cell.recent_reactions.len();
        let reactions = cell
            .recent_reactions
            .iter()
            .rev()
            .take(limit)
            .cloned()
            .collect::<Vec<_>>();
        Some(ReactionLogInspection {
            available: true,
            reason: "recorded",
            limit,
            truncated: reaction_count > reactions.len(),
            reaction_count,
            returned_count: reactions.len(),
            order: "newest_first",
            reactions,
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
        let mut cells = Vec::new();
        for dx in 0..width {
            for dy in 0..height {
                let tile_id = self.wrapped_tile_id(start_x + dx as isize, start_y + dy as isize);
                let Some(cell_id) = self.tiles[tile_id.index()].cell else {
                    continue;
                };
                if self
                    .cells
                    .get(cell_id.index())
                    .map(|cell| cell.state == CellState::Active)
                    .unwrap_or(false)
                {
                    cells.push(cell_id);
                }
            }
        }

        let mut changed = BTreeSet::new();
        for cell_id in cells.iter().copied() {
            let Some(cell) = self.cells.get(cell_id.index()) else {
                return Err(WorldError::InvalidCell(cell_id));
            };
            if cell.state != CellState::Active {
                return Err(WorldError::InvalidCell(cell_id));
            }
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
        let Some(cell) = self.cells.get(cell_id.index()) else {
            return Err(WorldError::InvalidCell(cell_id));
        };
        if cell.state != CellState::Active {
            return Err(WorldError::InvalidCell(cell_id));
        }
        let original_lineage = cell.lineage_id;
        let mut genome = cell.genome.clone();
        let changed_fields = patch
            .apply_to_genome(&mut genome)
            .map_err(|err| WorldError::GenomePatch(err.to_string()))?;
        genome.lineage_id = original_lineage;

        let cell = self
            .cells
            .get_mut(cell_id.index())
            .ok_or(WorldError::InvalidCell(cell_id))?;
        cell.genome = genome;
        cell.lineage_id = original_lineage;
        cell.maintenance_cost_per_sec = cell.genome.maintenance_cost_per_sec;
        cell.refresh_combat_totals();
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
        let mut attack_sum = 0_u64;
        let mut defense_sum = 0_u64;
        let mut max_attack_total = 0_u32;
        let mut max_defense_total = 0_u32;
        let mut cells_with_attackase = 0_u64;
        let mut cells_with_defensase = 0_u64;
        for cell_id in self.active_cells.iter().copied() {
            let Some(cell) = self.cells.get(cell_id.index()) else {
                continue;
            };
            if cell.state != CellState::Active || cell.lineage_id != lineage_id {
                continue;
            }
            live_count = live_count.saturating_add(1);
            energy_sum += cell.energy;
            enzyme_sum = enzyme_sum.saturating_add(cell.genome.enzymes.len() as u64);
            attack_sum = attack_sum.saturating_add(u64::from(cell.combat_attack_total));
            defense_sum = defense_sum.saturating_add(u64::from(cell.combat_defense_total));
            max_attack_total = max_attack_total.max(cell.combat_attack_total);
            max_defense_total = max_defense_total.max(cell.combat_defense_total);
            if cell
                .genome
                .enzymes
                .iter()
                .any(|enzyme| enzyme.enzyme_type == EnzymeType::Attackase)
            {
                cells_with_attackase = cells_with_attackase.saturating_add(1);
            }
            if cell
                .genome
                .enzymes
                .iter()
                .any(|enzyme| enzyme.enzyme_type == EnzymeType::Defensase)
            {
                cells_with_defensase = cells_with_defensase.saturating_add(1);
            }
        }
        let denominator = live_count.max(1) as f64;
        let total_live_cells = self.active_cells.len().max(1) as f64;
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
            average_attack_total: if live_count > 0 {
                attack_sum as f64 / denominator
            } else {
                0.0
            },
            average_defense_total: if live_count > 0 {
                defense_sum as f64 / denominator
            } else {
                0.0
            },
            max_attack_total,
            max_defense_total,
            cells_with_attackase,
            cells_with_defensase,
        }
    }

    pub fn check_invariants(&self) -> Result<(), InvariantError> {
        if self.tiles.len() != self.tile_count
            || self.neighbors.len() != self.tile_count
            || self.enval.len() != self.tile_count
            || self.enval_next.len() != self.tile_count
            || self.tile_element_counts.len() != self.tile_count
            || self.tile_element_masks.len() != self.tile_count
            || self.tile_mass_counts.len() != self.tile_count
        {
            return Err(InvariantError::MismatchedWorldArrayLengths);
        }

        let mut seen_predation_pairs = HashSet::new();
        for (left, right) in self.predation_pairs.iter().copied() {
            if left.index() >= self.tile_count || right.index() >= self.tile_count || left == right
            {
                return Err(InvariantError::InvalidPredationPair { left, right });
            }
            let a = left.index().min(right.index());
            let b = left.index().max(right.index());
            if !seen_predation_pairs.insert((a, b)) {
                return Err(InvariantError::DuplicatePredationPair { left, right });
            }
        }

        let mut seen_molecules = vec![false; self.molecules.len()];
        let mut seen_cells = vec![false; self.cells.len()];
        let mut actual_element_counts = vec![[0_u32; ELEMENT_COUNT]; self.tile_count];
        let mut actual_mass_counts = vec![0_u32; self.tile_count];

        for tile_index in 0..self.tile_count {
            if !self.enval[tile_index].is_finite() {
                return Err(InvariantError::NonFiniteEnval(TileId(tile_index)));
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

            if let Some(cell_id) = self.tiles[tile_index].cell {
                if cell_id.index() >= self.cells.len() {
                    return Err(InvariantError::InvalidCellId {
                        tile: TileId(tile_index),
                        cell: cell_id,
                    });
                }
                if std::mem::replace(&mut seen_cells[cell_id.index()], true) {
                    return Err(InvariantError::DuplicateCellOccupancy(cell_id));
                }
                let cell = &self.cells[cell_id.index()];
                if cell.state != CellState::Active {
                    return Err(InvariantError::DeadCellOnTile(cell_id));
                }
                if cell.tile_id != Some(TileId(tile_index)) {
                    return Err(InvariantError::WrongCellTile {
                        cell: cell_id,
                        expected_tile: TileId(tile_index),
                        actual_tile: cell.tile_id,
                    });
                }
            }

            for (slot, molecule_id) in self.tiles[tile_index].molecules.iter().copied().enumerate()
            {
                if molecule_id.index() >= self.molecules.len() {
                    return Err(InvariantError::InvalidTileMoleculeId {
                        tile: TileId(tile_index),
                        molecule: molecule_id,
                    });
                }
                if std::mem::replace(&mut seen_molecules[molecule_id.index()], true) {
                    return Err(InvariantError::DuplicateMoleculeOwner(molecule_id));
                }
                let record = &self.molecules[molecule_id.index()];
                if record.owner != MoleculeOwner::Tile(TileId(tile_index)) {
                    return Err(InvariantError::WrongTileMoleculeOwner {
                        molecule: molecule_id,
                        expected_tile: TileId(tile_index),
                        actual_owner: record.owner,
                    });
                }
                if record.owner_slot != slot {
                    return Err(InvariantError::WrongMoleculeOwnerSlot {
                        molecule: molecule_id,
                        expected_slot: slot,
                        actual_slot: record.owner_slot,
                    });
                }
                self.validate_molecule_state(molecule_id)?;
                for element in ELEMENT_ORDER {
                    let add = u32::from(record.molecule.composition.count(element));
                    actual_element_counts[tile_index][element.index()] = actual_element_counts
                        [tile_index][element.index()]
                    .checked_add(add)
                    .ok_or(InvariantError::TileCountOverflow(TileId(tile_index)))?;
                }
                actual_mass_counts[tile_index] = actual_mass_counts[tile_index]
                    .checked_add(u32::from(record.molecule.size))
                    .ok_or(InvariantError::TileCountOverflow(TileId(tile_index)))?;
            }
        }

        for (cell_index, cell) in self.cells.iter().enumerate() {
            let cell_id = CellId(cell_index);
            if cell.state == CellState::Dead {
                if cell.tile_id.is_some()
                    || !cell.molecules.is_empty()
                    || cell.internal_atom_count != 0
                    || cell.internal_element_counts != [0; ELEMENT_COUNT]
                    || cell.internal_element_mask != 0
                {
                    return Err(InvariantError::DeadCellOwnsState(cell_id));
                }
                continue;
            }
            if !cell.energy.is_finite() {
                return Err(InvariantError::NonFiniteCellEnergy(cell_id));
            }
            if !seen_cells[cell_index] {
                return Err(InvariantError::LiveCellNotOnTile(cell_id));
            }
            if cell.genome.enzymes.len() < MIN_CELL_ENZYMES
                || cell.genome.enzymes.len() > MAX_CELL_ENZYMES
            {
                return Err(InvariantError::InvalidGenomeEnzymeCount(cell_id));
            }
            for enzyme in &cell.genome.enzymes {
                if enzyme.enzyme_type.is_metabolic()
                    && (enzyme.specificity_mask == 0
                        || enzyme.specificity_mask & !crate::chem::ALL_ELEMENT_MASK != 0)
                {
                    return Err(InvariantError::InvalidEnzymeSpecificity(cell_id));
                }
            }
            if cell.combat_attack_total != cell.genome.attack_total()
                || cell.combat_defense_total != cell.genome.defense_total()
            {
                return Err(InvariantError::CombatTotalsMismatch(cell_id));
            }

            let mut actual_internal_counts = [0_u32; ELEMENT_COUNT];
            let mut actual_internal_atoms = 0_u32;
            for (slot, molecule_id) in cell.molecules.iter().copied().enumerate() {
                if molecule_id.index() >= self.molecules.len() {
                    return Err(InvariantError::InvalidCellMoleculeId {
                        cell: cell_id,
                        molecule: molecule_id,
                    });
                }
                if std::mem::replace(&mut seen_molecules[molecule_id.index()], true) {
                    return Err(InvariantError::DuplicateMoleculeOwner(molecule_id));
                }
                let record = &self.molecules[molecule_id.index()];
                if record.owner != MoleculeOwner::Cell(cell_id) {
                    return Err(InvariantError::WrongCellMoleculeOwner {
                        molecule: molecule_id,
                        expected_cell: cell_id,
                        actual_owner: record.owner,
                    });
                }
                if record.owner_slot != slot {
                    return Err(InvariantError::WrongMoleculeOwnerSlot {
                        molecule: molecule_id,
                        expected_slot: slot,
                        actual_slot: record.owner_slot,
                    });
                }
                self.validate_molecule_state(molecule_id)?;
                for element in ELEMENT_ORDER {
                    actual_internal_counts[element.index()] = actual_internal_counts
                        [element.index()]
                    .checked_add(u32::from(record.molecule.composition.count(element)))
                    .ok_or(InvariantError::CellCountOverflow(cell_id))?;
                }
                actual_internal_atoms = actual_internal_atoms
                    .checked_add(u32::from(record.molecule.size))
                    .ok_or(InvariantError::CellCountOverflow(cell_id))?;
            }
            if actual_internal_counts != cell.internal_element_counts {
                return Err(InvariantError::CellElementCountsMismatch(cell_id));
            }
            if element_mask_from_counts(actual_internal_counts) != cell.internal_element_mask {
                return Err(InvariantError::CellElementMaskMismatch(cell_id));
            }
            if actual_internal_atoms != cell.internal_atom_count {
                return Err(InvariantError::CellMassCountMismatch(cell_id));
            }
        }

        let mut active_seen = HashSet::new();
        for (slot, cell_id) in self.active_cells.iter().copied().enumerate() {
            if cell_id.index() >= self.cells.len() {
                return Err(InvariantError::InvalidActiveCell(cell_id));
            }
            if !active_seen.insert(cell_id) {
                return Err(InvariantError::DuplicateActiveCell(cell_id));
            }
            let cell = &self.cells[cell_id.index()];
            if cell.state != CellState::Active {
                return Err(InvariantError::DeadActiveCell(cell_id));
            }
            if cell.active_slot != Some(slot) {
                return Err(InvariantError::WrongActiveCellSlot {
                    cell: cell_id,
                    expected_slot: slot,
                    actual_slot: cell.active_slot,
                });
            }
        }

        let mut free_list_seen = vec![false; self.molecules.len()];
        for molecule_id in self.free_molecule_ids.iter().copied() {
            if molecule_id.index() >= self.molecules.len() {
                return Err(InvariantError::InvalidFreeListMolecule(molecule_id));
            }
            if std::mem::replace(&mut free_list_seen[molecule_id.index()], true) {
                return Err(InvariantError::DuplicateFreeListMolecule(molecule_id));
            }
            if self.molecules[molecule_id.index()].owner != MoleculeOwner::Free {
                return Err(InvariantError::OwnedMoleculeInFreeList(molecule_id));
            }
        }

        for (index, record) in self.molecules.iter().enumerate() {
            let molecule_id = MoleculeId(index);
            match record.owner {
                MoleculeOwner::Tile(_) | MoleculeOwner::Cell(_) => {
                    if !seen_molecules[index] {
                        return Err(InvariantError::UnlistedOwnedMolecule(molecule_id));
                    }
                }
                MoleculeOwner::Free => {
                    if seen_molecules[index] {
                        return Err(InvariantError::FreeMoleculeListed(molecule_id));
                    }
                    if record.wheel_index.is_some() || record.wheel_pos.is_some() {
                        return Err(InvariantError::FreeMoleculeScheduled(molecule_id));
                    }
                    if !free_list_seen[index] {
                        return Err(InvariantError::FreeMoleculeMissingFromFreeList(molecule_id));
                    }
                }
            }
        }

        for tile_index in 0..self.tile_count {
            if actual_element_counts[tile_index] != self.tile_element_counts[tile_index] {
                return Err(InvariantError::TileElementCountsMismatch(TileId(
                    tile_index,
                )));
            }
            if element_mask_from_counts(actual_element_counts[tile_index])
                != self.tile_element_masks[tile_index]
            {
                return Err(InvariantError::TileElementMaskMismatch(TileId(tile_index)));
            }
            if actual_mass_counts[tile_index] != self.tile_mass_counts[tile_index] {
                return Err(InvariantError::TileMassCountMismatch(TileId(tile_index)));
            }
        }

        let mut wheel_seen = HashSet::new();
        let wheel_mask = self.config.molecule_diffusion_wheel_size - 1;
        for (bucket_index, bucket) in self.diffusion_wheel.iter().enumerate() {
            for (pos, molecule_id) in bucket.iter().copied().enumerate() {
                if molecule_id.index() >= self.molecules.len() {
                    return Err(InvariantError::InvalidWheelMolecule(molecule_id));
                }
                if !wheel_seen.insert(molecule_id) {
                    return Err(InvariantError::DuplicateWheelMolecule(molecule_id));
                }
                let record = &self.molecules[molecule_id.index()];
                if !matches!(record.owner, MoleculeOwner::Tile(_)) {
                    return Err(InvariantError::NonTileMoleculeScheduled(molecule_id));
                }
                if record.wheel_index != Some(bucket_index) || record.wheel_pos != Some(pos) {
                    return Err(InvariantError::WrongWheelPosition {
                        molecule: molecule_id,
                        expected_bucket: bucket_index,
                        expected_pos: pos,
                        actual_bucket: record.wheel_index,
                        actual_pos: record.wheel_pos,
                    });
                }
                if (record.diffusion_tick as usize) & wheel_mask != bucket_index {
                    return Err(InvariantError::WrongWheelBucket {
                        molecule: molecule_id,
                        tick: record.diffusion_tick,
                        bucket: bucket_index,
                    });
                }
            }
        }

        for (index, record) in self.molecules.iter().enumerate() {
            if matches!(record.owner, MoleculeOwner::Tile(_))
                && record.molecule.diffusion_wait() > 0
                && !wheel_seen.contains(&MoleculeId(index))
            {
                return Err(InvariantError::UnscheduledMolecule(MoleculeId(index)));
            }
        }

        let mut actual_lineage_population = BTreeMap::<LineageId, u64>::new();
        for cell_id in &self.active_cells {
            let cell = &self.cells[cell_id.index()];
            if cell.state == CellState::Active {
                *actual_lineage_population
                    .entry(cell.lineage_id)
                    .or_default() += 1;
            }
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

        if !self.predator_energy_gained.is_finite() {
            return Err(InvariantError::NonFinitePredationEnergy);
        }

        Ok(())
    }

    fn index_xy(&self, x: usize, y: usize) -> usize {
        x * self.height + y
    }

    fn seed_initial_molecules(&mut self) -> Result<(), WorldError> {
        let seeding = self.config.molecule_seeding;
        for tile_index in 0..self.tile_count {
            let tile_id = TileId(tile_index);
            self.add_tile_molecule(tile_id, Composition::single(Element::A), 1.0)?;
            if self.rng.chance(seeding.b) {
                self.add_tile_molecule(tile_id, Composition::single(Element::B), 1.0)?;
            }
            if self.rng.chance(seeding.c) {
                self.add_tile_molecule(tile_id, Composition::single(Element::C), 1.0)?;
            }
            if self.rng.chance(seeding.d) {
                self.add_tile_molecule(tile_id, Composition::single(Element::D), 1.0)?;
            }
            if self.rng.chance(seeding.e) {
                self.add_tile_molecule(tile_id, Composition::single(Element::E), 1.0)?;
            }
            if self.rng.chance(seeding.f) {
                self.add_tile_molecule(tile_id, Composition::single(Element::F), 1.0)?;
            }
            if self.rng.chance(seeding.bc) {
                self.add_tile_molecule(tile_id, Composition::bc_dimer(), 1.0)?;
            }
        }
        Ok(())
    }

    fn allocate_molecule_record(
        &mut self,
        molecule: Molecule,
        owner: MoleculeOwner,
        owner_slot: usize,
    ) -> MoleculeId {
        let record = MoleculeRecord {
            molecule,
            owner,
            owner_slot,
            diffusion_phase: None,
            last_diffusion_dir: None,
            diffusion_tick: 0,
            wheel_index: None,
            wheel_pos: None,
        };

        if let Some(molecule_id) = self.free_molecule_ids.pop() {
            debug_assert!(molecule_id.index() < self.molecules.len());
            self.molecules[molecule_id.index()] = record;
            self.operation_counters.molecule_slots_reused = self
                .operation_counters
                .molecule_slots_reused
                .saturating_add(1);
            molecule_id
        } else {
            let molecule_id = MoleculeId(self.molecules.len());
            self.molecules.push(record);
            self.molecule_arena_high_water_mark = self
                .molecule_arena_high_water_mark
                .max(self.molecules.len());
            self.operation_counters.molecule_slots_newly_allocated = self
                .operation_counters
                .molecule_slots_newly_allocated
                .saturating_add(1);
            molecule_id
        }
    }

    fn release_molecule_slot(&mut self, molecule_id: MoleculeId) {
        if molecule_id.index() >= self.molecules.len() {
            return;
        }
        self.unschedule_molecule_diffusion(molecule_id);
        let record = &mut self.molecules[molecule_id.index()];
        record.owner = MoleculeOwner::Free;
        record.owner_slot = 0;
        record.diffusion_phase = None;
        record.last_diffusion_dir = None;
        record.diffusion_tick = 0;
        record.wheel_index = None;
        record.wheel_pos = None;
        self.free_molecule_ids.push(molecule_id);
    }

    fn add_tile_molecule(
        &mut self,
        tile_id: TileId,
        composition: Composition,
        bond_multiplier: f32,
    ) -> Result<MoleculeId, WorldError> {
        let molecule = Molecule::new(composition, bond_multiplier)?;
        self.add_tile_molecule_record(tile_id, molecule)
    }

    fn add_tile_molecule_record(
        &mut self,
        tile_id: TileId,
        molecule: Molecule,
    ) -> Result<MoleculeId, WorldError> {
        if tile_id.index() >= self.tile_count {
            return Err(WorldError::InvalidTile(tile_id));
        }
        let owner_slot = self.tiles[tile_id.index()].molecules.len();
        let molecule_id =
            self.allocate_molecule_record(molecule, MoleculeOwner::Tile(tile_id), owner_slot);
        self.tiles[tile_id.index()].molecules.push(molecule_id);
        self.apply_tile_composition_delta(tile_id, molecule.composition, 1)?;
        self.schedule_molecule_diffusion(molecule_id);
        Ok(molecule_id)
    }

    fn add_cell_molecule_record(
        &mut self,
        cell_id: CellId,
        molecule: Molecule,
    ) -> Result<MoleculeId, WorldError> {
        if cell_id.index() >= self.cells.len()
            || self.cells[cell_id.index()].state != CellState::Active
        {
            return Err(WorldError::InvalidCell(cell_id));
        }
        let owner_slot = self.cells[cell_id.index()].molecules.len();
        let molecule_id =
            self.allocate_molecule_record(molecule, MoleculeOwner::Cell(cell_id), owner_slot);
        self.cells[cell_id.index()].molecules.push(molecule_id);
        self.apply_cell_composition_delta(cell_id, molecule.composition, 1)?;
        Ok(molecule_id)
    }

    fn add_existing_molecule_to_cell(
        &mut self,
        cell_id: CellId,
        molecule_id: MoleculeId,
    ) -> Result<(), WorldError> {
        if cell_id.index() >= self.cells.len()
            || self.cells[cell_id.index()].state != CellState::Active
        {
            return Err(WorldError::InvalidCell(cell_id));
        }
        if molecule_id.index() >= self.molecules.len() {
            return Err(WorldError::InvalidMolecule(molecule_id));
        }
        self.unschedule_molecule_diffusion(molecule_id);
        let composition = self.molecules[molecule_id.index()].molecule.composition;
        let owner_slot = self.cells[cell_id.index()].molecules.len();
        self.cells[cell_id.index()].molecules.push(molecule_id);
        self.apply_cell_composition_delta(cell_id, composition, 1)?;
        let record = &mut self.molecules[molecule_id.index()];
        record.owner = MoleculeOwner::Cell(cell_id);
        record.owner_slot = owner_slot;
        Ok(())
    }

    fn add_existing_molecule_to_tile(
        &mut self,
        tile_id: TileId,
        molecule_id: MoleculeId,
    ) -> Result<(), WorldError> {
        if tile_id.index() >= self.tile_count {
            return Err(WorldError::InvalidTile(tile_id));
        }
        if molecule_id.index() >= self.molecules.len() {
            return Err(WorldError::InvalidMolecule(molecule_id));
        }
        let composition = self.molecules[molecule_id.index()].molecule.composition;
        let owner_slot = self.tiles[tile_id.index()].molecules.len();
        self.tiles[tile_id.index()].molecules.push(molecule_id);
        self.apply_tile_composition_delta(tile_id, composition, 1)?;
        let record = &mut self.molecules[molecule_id.index()];
        record.owner = MoleculeOwner::Tile(tile_id);
        record.owner_slot = owner_slot;
        record.last_diffusion_dir = None;
        self.schedule_molecule_diffusion(molecule_id);
        Ok(())
    }

    fn apply_tile_composition_delta(
        &mut self,
        tile_id: TileId,
        composition: Composition,
        sign: i32,
    ) -> Result<(), WorldError> {
        if tile_id.index() >= self.tile_count {
            return Err(WorldError::InvalidTile(tile_id));
        }
        let counts = &mut self.tile_element_counts[tile_id.index()];
        for element in ELEMENT_ORDER {
            let count = u32::from(composition.count(element));
            if sign >= 0 {
                counts[element.index()] = counts[element.index()]
                    .checked_add(count)
                    .ok_or(WorldError::TileCountOverflow(tile_id))?;
            } else {
                counts[element.index()] = counts[element.index()]
                    .checked_sub(count)
                    .ok_or(WorldError::TileCountUnderflow(tile_id))?;
            }
        }
        self.tile_element_masks[tile_id.index()] = element_mask_from_counts(*counts);
        let atoms = u32::from(composition.size());
        if sign >= 0 {
            self.tile_mass_counts[tile_id.index()] = self.tile_mass_counts[tile_id.index()]
                .checked_add(atoms)
                .ok_or(WorldError::TileCountOverflow(tile_id))?;
        } else {
            self.tile_mass_counts[tile_id.index()] = self.tile_mass_counts[tile_id.index()]
                .checked_sub(atoms)
                .ok_or(WorldError::TileCountUnderflow(tile_id))?;
        }
        Ok(())
    }

    fn apply_cell_composition_delta(
        &mut self,
        cell_id: CellId,
        composition: Composition,
        sign: i32,
    ) -> Result<(), WorldError> {
        self.cells[cell_id.index()]
            .apply_internal_composition_delta(composition, sign)
            .map_err(|err| match err {
                CellCountError::Overflow => WorldError::CellCountOverflow(cell_id),
                CellCountError::Underflow => WorldError::CellCountUnderflow(cell_id),
            })
    }

    fn schedule_molecule_diffusion(&mut self, molecule_id: MoleculeId) {
        self.unschedule_molecule_diffusion(molecule_id);
        if molecule_id.index() >= self.molecules.len() {
            return;
        }
        let MoleculeOwner::Tile(tile_id) = self.molecules[molecule_id.index()].owner else {
            self.molecules[molecule_id.index()].diffusion_tick = 0;
            return;
        };
        let wait = self.molecules[molecule_id.index()]
            .molecule
            .diffusion_wait();
        if wait == 0 {
            self.molecules[molecule_id.index()].diffusion_tick = 0;
            return;
        }

        let phase = match self.molecules[molecule_id.index()].diffusion_phase {
            Some(phase) => phase % wait,
            None => {
                let record = &self.molecules[molecule_id.index()];
                let tile_index = tile_id.index() as u32;
                let tile_offset = record.owner_slot as u32;
                (tile_index
                    + tile_offset
                    + u32::from(record.molecule.element_mask) * 7
                    + u32::from(record.molecule.size) * 13)
                    % wait
            }
        };
        self.molecules[molecule_id.index()].diffusion_phase = Some(phase);

        let current_mod = (self.diffusion_tick.wrapping_add(phase)) % wait;
        let mut offset = wait - current_mod;
        if offset == 0 || offset > wait {
            offset = wait;
        }
        let next_tick = self.diffusion_tick.wrapping_add(offset);
        self.queue_molecule_diffusion_at(molecule_id, next_tick);
    }

    fn queue_molecule_diffusion_at(&mut self, molecule_id: MoleculeId, tick: u32) {
        if molecule_id.index() >= self.molecules.len()
            || !matches!(
                self.molecules[molecule_id.index()].owner,
                MoleculeOwner::Tile(_)
            )
        {
            return;
        }
        let bucket_index = (tick as usize) & (self.config.molecule_diffusion_wheel_size - 1);
        let pos = self.diffusion_wheel[bucket_index].len();
        let record = &mut self.molecules[molecule_id.index()];
        record.diffusion_tick = tick;
        record.wheel_index = Some(bucket_index);
        record.wheel_pos = Some(pos);
        self.diffusion_wheel[bucket_index].push(molecule_id);
    }

    fn unschedule_molecule_diffusion(&mut self, molecule_id: MoleculeId) {
        if molecule_id.index() >= self.molecules.len() {
            return;
        }
        let (bucket_index, pos) = match (
            self.molecules[molecule_id.index()].wheel_index,
            self.molecules[molecule_id.index()].wheel_pos,
        ) {
            (Some(bucket_index), Some(pos)) => (bucket_index, pos),
            _ => return,
        };
        if bucket_index < self.diffusion_wheel.len()
            && pos < self.diffusion_wheel[bucket_index].len()
            && self.diffusion_wheel[bucket_index][pos] == molecule_id
        {
            let _ = self.remove_wheel_at(bucket_index, pos);
        } else if bucket_index < self.diffusion_wheel.len() {
            if let Some(found_pos) = self.diffusion_wheel[bucket_index]
                .iter()
                .position(|candidate| *candidate == molecule_id)
            {
                let _ = self.remove_wheel_at(bucket_index, found_pos);
            } else {
                self.molecules[molecule_id.index()].wheel_index = None;
                self.molecules[molecule_id.index()].wheel_pos = None;
            }
        }
    }

    fn remove_wheel_at(&mut self, bucket_index: usize, pos: usize) -> Option<MoleculeId> {
        if bucket_index >= self.diffusion_wheel.len()
            || pos >= self.diffusion_wheel[bucket_index].len()
        {
            return None;
        }
        let last_pos = self.diffusion_wheel[bucket_index].len() - 1;
        let removed = self.diffusion_wheel[bucket_index][pos];
        if pos != last_pos {
            let swapped = self.diffusion_wheel[bucket_index][last_pos];
            self.diffusion_wheel[bucket_index][pos] = swapped;
            if swapped.index() < self.molecules.len() {
                self.molecules[swapped.index()].wheel_pos = Some(pos);
            }
        }
        self.diffusion_wheel[bucket_index].pop();
        if removed.index() < self.molecules.len() {
            self.molecules[removed.index()].wheel_index = None;
            self.molecules[removed.index()].wheel_pos = None;
        }
        Some(removed)
    }

    fn try_diffuse_molecule(&mut self, molecule_id: MoleculeId) {
        if molecule_id.index() >= self.molecules.len() {
            return;
        }
        let MoleculeOwner::Tile(src_tile) = self.molecules[molecule_id.index()].owner else {
            return;
        };
        let composition = self.molecules[molecule_id.index()].molecule.composition;
        let neighbors = self.neighbors[src_tile.index()];
        let right = neighbors.right;
        let left = neighbors.left;
        let down = neighbors.down;
        let up = neighbors.up;

        let src_density = self.tile_composition_density(src_tile, composition);
        let right_density = self.tile_composition_density(right, composition);
        let left_density = self.tile_composition_density(left, composition);
        let down_density = self.tile_composition_density(down, composition);
        let up_density = self.tile_composition_density(up, composition);

        let mut min_density = src_density;
        let mut tie_mask = 0_u8;
        if right_density < min_density {
            min_density = right_density;
            tie_mask = 1;
        } else if right_density == min_density && right_density < src_density {
            tie_mask |= 1;
        }
        if left_density < min_density {
            min_density = left_density;
            tie_mask = 2;
        } else if left_density == min_density && left_density < src_density {
            tie_mask |= 2;
        }
        if down_density < min_density {
            min_density = down_density;
            tie_mask = 4;
        } else if down_density == min_density && down_density < src_density {
            tie_mask |= 4;
        }
        if up_density < min_density {
            tie_mask = 8;
        } else if up_density == min_density && up_density < src_density {
            tie_mask |= 8;
        }

        if tie_mask != 0 {
            if let Some(last_dir) = self.molecules[molecule_id.index()].last_diffusion_dir {
                let reverse_mask = 1_u8 << ((last_dir ^ 1) & 3);
                let filtered = tie_mask & !reverse_mask;
                if filtered != 0 {
                    tie_mask = filtered;
                }
            }
        }

        let chosen_dir =
            choose_diffusion_direction(&mut self.tile_diffusion_rotor[src_tile.index()], tie_mask);
        if let Some(dir) = chosen_dir {
            let dest_tile = match dir {
                0 => right,
                1 => left,
                2 => down,
                3 => up,
                _ => src_tile,
            };
            if dest_tile != src_tile {
                if self
                    .move_molecule_between_tiles(molecule_id, src_tile, dest_tile, dir)
                    .is_err()
                {
                    // Preserve simulation progress as invariant checks expose structural errors.
                }
            }
        }

        let wait = self.molecules[molecule_id.index()]
            .molecule
            .diffusion_wait();
        if wait > 0
            && matches!(
                self.molecules[molecule_id.index()].owner,
                MoleculeOwner::Tile(_)
            )
        {
            self.queue_molecule_diffusion_at(molecule_id, self.diffusion_tick.wrapping_add(wait));
        }
    }

    fn tile_composition_density(&self, tile_id: TileId, composition: Composition) -> u64 {
        composition.density_against(&self.tile_element_counts[tile_id.index()])
    }

    fn move_molecule_between_tiles(
        &mut self,
        molecule_id: MoleculeId,
        src_tile: TileId,
        dest_tile: TileId,
        chosen_dir: u8,
    ) -> Result<(), WorldError> {
        let src_index = src_tile.index();
        let dest_index = dest_tile.index();
        let src_slot = self.molecules[molecule_id.index()].owner_slot;
        if src_slot >= self.tiles[src_index].molecules.len()
            || self.tiles[src_index].molecules[src_slot] != molecule_id
        {
            return Err(WorldError::MoleculeSlotMismatch(molecule_id));
        }

        self.remove_molecule_from_tile_slot(src_tile, src_slot)?;

        let composition = self.molecules[molecule_id.index()].molecule.composition;
        let dest_slot = self.tiles[dest_index].molecules.len();
        self.tiles[dest_index].molecules.push(molecule_id);
        self.apply_tile_composition_delta(dest_tile, composition, 1)?;
        let record = &mut self.molecules[molecule_id.index()];
        record.owner = MoleculeOwner::Tile(dest_tile);
        record.owner_slot = dest_slot;
        record.last_diffusion_dir = Some(chosen_dir & 3);
        self.operation_counters.molecule_moves =
            self.operation_counters.molecule_moves.saturating_add(1);
        Ok(())
    }

    fn remove_molecule_from_tile_slot(
        &mut self,
        tile_id: TileId,
        slot: usize,
    ) -> Result<MoleculeId, WorldError> {
        if tile_id.index() >= self.tile_count {
            return Err(WorldError::InvalidTile(tile_id));
        }
        let tile_index = tile_id.index();
        let last_slot = self.tiles[tile_index]
            .molecules
            .len()
            .checked_sub(1)
            .ok_or(WorldError::MoleculeSlotMismatch(MoleculeId(usize::MAX)))?;
        if slot > last_slot {
            return Err(WorldError::MoleculeSlotMismatch(MoleculeId(usize::MAX)));
        }
        let removed = self.tiles[tile_index].molecules[slot];
        if slot != last_slot {
            let swapped = self.tiles[tile_index].molecules[last_slot];
            self.tiles[tile_index].molecules[slot] = swapped;
            self.molecules[swapped.index()].owner_slot = slot;
        }
        self.tiles[tile_index].molecules.pop();
        let composition = self.molecules[removed.index()].molecule.composition;
        self.apply_tile_composition_delta(tile_id, composition, -1)?;
        self.unschedule_molecule_diffusion(removed);
        let record = &mut self.molecules[removed.index()];
        record.owner = MoleculeOwner::Free;
        record.owner_slot = 0;
        record.last_diffusion_dir = None;
        Ok(removed)
    }

    fn remove_molecule_from_cell_slot(
        &mut self,
        cell_id: CellId,
        slot: usize,
    ) -> Result<MoleculeId, WorldError> {
        if cell_id.index() >= self.cells.len() {
            return Err(WorldError::InvalidCell(cell_id));
        }
        let last_slot = self.cells[cell_id.index()]
            .molecules
            .len()
            .checked_sub(1)
            .ok_or(WorldError::MoleculeSlotMismatch(MoleculeId(usize::MAX)))?;
        if slot > last_slot {
            return Err(WorldError::MoleculeSlotMismatch(MoleculeId(usize::MAX)));
        }
        let removed = self.cells[cell_id.index()].molecules[slot];
        if slot != last_slot {
            let swapped = self.cells[cell_id.index()].molecules[last_slot];
            self.cells[cell_id.index()].molecules[slot] = swapped;
            self.molecules[swapped.index()].owner_slot = slot;
        }
        self.cells[cell_id.index()].molecules.pop();
        let composition = self.molecules[removed.index()].molecule.composition;
        self.apply_cell_composition_delta(cell_id, composition, -1)?;
        let record = &mut self.molecules[removed.index()];
        record.owner = MoleculeOwner::Free;
        record.owner_slot = 0;
        record.last_diffusion_dir = None;
        Ok(removed)
    }

    fn consume_molecule(&mut self, molecule_id: MoleculeId) -> Result<(), WorldError> {
        if molecule_id.index() >= self.molecules.len() {
            return Err(WorldError::InvalidMolecule(molecule_id));
        }
        match self.molecules[molecule_id.index()].owner {
            MoleculeOwner::Tile(tile_id) => {
                let slot = self.molecules[molecule_id.index()].owner_slot;
                let removed = self.remove_molecule_from_tile_slot(tile_id, slot)?;
                self.release_molecule_slot(removed);
            }
            MoleculeOwner::Cell(cell_id) => {
                let slot = self.molecules[molecule_id.index()].owner_slot;
                let removed = self.remove_molecule_from_cell_slot(cell_id, slot)?;
                self.release_molecule_slot(removed);
            }
            MoleculeOwner::Free => {}
        }
        Ok(())
    }

    fn step_cells(&mut self) {
        for tile_index in 0..self.tile_count {
            let Some(cell_id) = self.tiles[tile_index].cell else {
                continue;
            };
            if cell_id.index() >= self.cells.len()
                || self.cells[cell_id.index()].state != CellState::Active
            {
                continue;
            }
            self.step_cell(cell_id);
        }
    }

    fn step_cell(&mut self, cell_id: CellId) {
        let Some(tile_id) = self
            .cells
            .get(cell_id.index())
            .and_then(|cell| cell.tile_id)
        else {
            return;
        };
        self.operation_counters.cell_steps = self.operation_counters.cell_steps.saturating_add(1);
        self.operation_counters.local_enval_average_calls = self
            .operation_counters
            .local_enval_average_calls
            .saturating_add(1);
        let mut local_enval = self.default_local_enval_average(tile_id).unwrap_or(0.0);
        let mut positive_energy_gain = 0.0_f64;
        let enzyme_count = self.cells[cell_id.index()].genome.enzymes.len();
        let genome_context = GenomeReactionContext::from(&self.cells[cell_id.index()].genome);

        for enzyme_index in 0..enzyme_count {
            if self.cells[cell_id.index()].state != CellState::Active {
                return;
            }
            let Some(enzyme) = self.cells[cell_id.index()]
                .genome
                .enzymes
                .get(enzyme_index)
                .copied()
            else {
                break;
            };
            self.operation_counters.enzyme_entries_seen = self
                .operation_counters
                .enzyme_entries_seen
                .saturating_add(1);
            if !enzyme.enzyme_type.is_metabolic() {
                self.operation_counters.combat_enzyme_skips = self
                    .operation_counters
                    .combat_enzyme_skips
                    .saturating_add(1);
                continue;
            }
            self.operation_counters.metabolic_enzyme_attempts = self
                .operation_counters
                .metabolic_enzyme_attempts
                .saturating_add(1);
            self.reaction_counters
                .attempts_by_type
                .increment(enzyme.enzyme_type);
            if self.cells[cell_id.index()].state != CellState::Active {
                return;
            }
            let max_inputs = bio::max_inputs(&enzyme);
            if max_inputs == 0 {
                let _ = self.rng.next_f64();
                continue;
            }
            if !self.substrate_pool_may_accept(cell_id, tile_id, &enzyme, max_inputs) {
                let _ = self.rng.next_f64();
                continue;
            }
            let env = ReactionEnv {
                tile_enval: self.enval[tile_id.index()],
                local_enval,
                average_enval: self.avg_enval,
            };
            if !bio::reaction_gate_for_context(&enzyme, genome_context, env, &mut self.rng) {
                continue;
            }
            self.operation_counters.reaction_gates_passed = self
                .operation_counters
                .reaction_gates_passed
                .saturating_add(1);
            self.reaction_counters
                .gates_passed_by_type
                .increment(enzyme.enzyme_type);
            let substrate_ids =
                self.sample_accepted_substrates_unchecked(cell_id, tile_id, &enzyme, max_inputs);
            if substrate_ids.is_empty()
                || (enzyme.enzyme_type == EnzymeType::Anabolase && substrate_ids.len() < 2)
            {
                self.reaction_counters
                    .no_substrate_by_type
                    .increment(enzyme.enzyme_type);
                continue;
            }
            let Some(first_substrate_id) = substrate_ids.first() else {
                continue;
            };
            let mut substrate_storage =
                [self.molecules[first_substrate_id.index()].molecule; MAX_REACTION_SUBSTRATES];
            for (index, molecule_id) in substrate_ids.iter().enumerate().skip(1) {
                substrate_storage[index] = self.molecules[molecule_id.index()].molecule;
            }
            let substrates = &substrate_storage[..substrate_ids.len()];
            let cell_energy = self.cells[cell_id.index()].energy;
            let Some(result) = bio::attempt_reaction_for_context(
                &enzyme,
                substrates,
                genome_context,
                cell_energy,
                env,
                &mut self.rng,
            ) else {
                continue;
            };
            let substrate_summaries = substrates
                .iter()
                .copied()
                .map(reaction_molecule_summary)
                .collect::<Vec<_>>();
            let produced_summary = result.produced.map(reaction_molecule_summary);
            let byproduct_summaries = result
                .byproducts
                .iter()
                .copied()
                .map(reaction_molecule_summary)
                .collect::<Vec<_>>();
            let energy_before = cell_energy;

            self.operation_counters.reactions_succeeded = self
                .operation_counters
                .reactions_succeeded
                .saturating_add(1);
            self.reaction_counters
                .successes_by_type
                .increment(enzyme.enzyme_type);
            self.reaction_counters
                .energy_delta_by_type
                .add(enzyme.enzyme_type, result.energy_delta);
            self.reaction_counters
                .enval_input_by_type
                .add(enzyme.enzyme_type, f64::from(result.enval_input));
            self.reaction_counters
                .enval_output_by_type
                .add(enzyme.enzyme_type, f64::from(result.enval_output));

            if result.energy_delta != 0.0 {
                self.cells[cell_id.index()].energy += result.energy_delta;
                if result.energy_delta > 0.0 {
                    positive_energy_gain += result.energy_delta;
                }
            }
            let energy_after = self.cells[cell_id.index()].energy;
            for molecule_id in substrate_ids.iter() {
                let _ = self.consume_molecule(molecule_id);
            }

            if let Some(product) = result.produced {
                if self.should_secrete(cell_id, product, &enzyme) {
                    if self.add_product_around(tile_id, product).is_ok() {
                        self.operation_counters.products_created =
                            self.operation_counters.products_created.saturating_add(1);
                        self.reaction_counters.molecule_outputs =
                            self.reaction_counters.molecule_outputs.saturating_add(1);
                    }
                } else if self.add_cell_molecule_record(cell_id, product).is_ok() {
                    self.operation_counters.products_created =
                        self.operation_counters.products_created.saturating_add(1);
                }
            }
            for byproduct in result.byproducts.iter().copied() {
                if self.add_product_around(tile_id, byproduct).is_ok() {
                    self.operation_counters.byproducts_created =
                        self.operation_counters.byproducts_created.saturating_add(1);
                    self.reaction_counters.molecule_outputs =
                        self.reaction_counters.molecule_outputs.saturating_add(1);
                }
            }

            let mut enval_changed = false;
            if result.enval_input != 0.0 {
                let _ = self.adjust_tile_enval(tile_id, -result.enval_input);
                enval_changed = true;
            }
            if result.enval_output != 0.0 {
                let _ = self.add_enval_around(tile_id, result.enval_output);
                enval_changed = true;
            }
            if enval_changed {
                self.operation_counters.local_enval_average_calls = self
                    .operation_counters
                    .local_enval_average_calls
                    .saturating_add(1);
                local_enval = self.default_local_enval_average(tile_id).unwrap_or(0.0);
            }
            let (x, y) = self.tile_xy(tile_id).unwrap_or((0, 0));
            self.cells[cell_id.index()].push_reaction_record(ReactionRecord {
                tick_count: self.tick_count,
                sim_time_seconds: self.sim_time_seconds,
                cell_id: cell_id.index(),
                tile_id: tile_id.index(),
                x,
                y,
                enzyme_index,
                enzyme_type: enzyme.enzyme_type.as_str().to_owned(),
                status: "success".to_owned(),
                substrate_count: substrate_summaries.len(),
                substrates: substrate_summaries,
                produced: produced_summary,
                byproducts: byproduct_summaries,
                energy_before,
                energy_after,
                delta_cell_energy: result.energy_delta,
                chemical_delta: result.chemical_delta,
                enval_energy: result.enval_energy,
                enval_input: result.enval_input,
                enval_output: result.enval_output,
                delta_enval: result.delta_enval,
                local_enval: env.local_enval,
                optimal_enval: genome_context.optimal_enval,
            });
        }

        if self.cells[cell_id.index()].state != CellState::Active {
            return;
        }
        if positive_energy_gain > 1.0e-6 {
            self.cells[cell_id.index()].time_without_food =
                (self.cells[cell_id.index()].time_without_food - positive_energy_gain * 0.2)
                    .max(0.0);
        } else {
            self.cells[cell_id.index()].time_without_food += self.config.dt_seconds;
        }

        let maintenance_loss =
            self.cells[cell_id.index()].maintenance_cost_per_sec * self.config.dt_seconds;
        if maintenance_loss > 0.0 {
            self.cells[cell_id.index()].energy -= maintenance_loss;
            if self.cells[cell_id.index()].energy <= 0.0 {
                self.cells[cell_id.index()].energy = 0.0;
                self.kill_cell_and_release(cell_id);
                return;
            }
        }

        let reserve = self.cells[cell_id.index()].genome.desired_element_reserve;
        if self.cells[cell_id.index()].internal_atom_count < reserve.saturating_mul(2)
            && !self.tiles[tile_id.index()].molecules.is_empty()
        {
            let slot = self.rng.usize(self.tiles[tile_id.index()].molecules.len());
            if let Ok(molecule_id) = self.remove_molecule_from_tile_slot(tile_id, slot) {
                if self
                    .add_existing_molecule_to_cell(cell_id, molecule_id)
                    .is_ok()
                {
                    self.operation_counters.molecule_uptakes =
                        self.operation_counters.molecule_uptakes.saturating_add(1);
                    self.reaction_counters.molecule_uptakes =
                        self.reaction_counters.molecule_uptakes.saturating_add(1);
                }
            }
        }

        let optimal_enval = self.cells[cell_id.index()].genome.optimal_enval;
        let dist = (local_enval - optimal_enval).abs();
        let enzyme_count = self.cells[cell_id.index()].genome.enzymes.len().max(1) as f64;
        let stress_increment = f64::from(dist).powf(1.6)
            * self.cells[cell_id.index()].genome.enval_stress_factor
            * enzyme_count.max(1.0);
        self.cells[cell_id.index()].time_without_food += stress_increment;
        if self.cells[cell_id.index()].time_without_food
            > self.cells[cell_id.index()].genome.decay_time
        {
            self.kill_cell_and_release(cell_id);
            return;
        }
        if self.cells[cell_id.index()].energy >= self.cells[cell_id.index()].genome.repro_threshold
        {
            self.divide_cell(cell_id, tile_id, local_enval);
        }
    }

    #[cfg(test)]
    fn sample_accepted_substrates(
        &mut self,
        cell_id: CellId,
        tile_id: TileId,
        enzyme: &crate::genome::Enzyme,
        max_count: usize,
    ) -> SampledSubstrateIds {
        debug_assert!(max_count <= MAX_REACTION_SUBSTRATES);
        let max_count = max_count.min(MAX_REACTION_SUBSTRATES);
        if !self.substrate_pool_may_accept(cell_id, tile_id, enzyme, max_count) {
            return SampledSubstrateIds::new();
        }
        self.sample_accepted_substrates_unchecked(cell_id, tile_id, enzyme, max_count)
    }

    fn sample_accepted_substrates_unchecked(
        &mut self,
        cell_id: CellId,
        tile_id: TileId,
        enzyme: &crate::genome::Enzyme,
        max_count: usize,
    ) -> SampledSubstrateIds {
        debug_assert!(max_count <= MAX_REACTION_SUBSTRATES);
        let max_count = max_count.min(MAX_REACTION_SUBSTRATES);
        let mut out = SampledSubstrateIds::new();
        let mut seen = 0_usize;

        let tile_len = self.tiles[tile_id.index()].molecules.len();
        for slot in 0..tile_len {
            let molecule_id = self.tiles[tile_id.index()].molecules[slot];
            self.consider_substrate_candidate(molecule_id, enzyme, max_count, &mut seen, &mut out);
        }

        let cell_len = self.cells[cell_id.index()].molecules.len();
        for slot in 0..cell_len {
            let molecule_id = self.cells[cell_id.index()].molecules[slot];
            self.consider_substrate_candidate(molecule_id, enzyme, max_count, &mut seen, &mut out);
        }

        out
    }

    fn substrate_pool_may_accept(
        &self,
        cell_id: CellId,
        tile_id: TileId,
        enzyme: &crate::genome::Enzyme,
        max_count: usize,
    ) -> bool {
        if max_count == 0
            || cell_id.index() >= self.cells.len()
            || tile_id.index() >= self.tile_count
        {
            return false;
        }
        let tile_molecule_count = self.tiles[tile_id.index()].molecules.len();
        let cell_molecule_count = self.cells[cell_id.index()].molecules.len();
        if tile_molecule_count + cell_molecule_count == 0 {
            return false;
        }
        if enzyme.enzyme_type == EnzymeType::Anabolase
            && tile_molecule_count + cell_molecule_count < 2
        {
            return false;
        }
        let pool_mask = self.tile_element_masks[tile_id.index()]
            | self.cells[cell_id.index()].internal_element_mask;
        match enzyme.enzyme_type {
            EnzymeType::Anabolase | EnzymeType::Catabolase | EnzymeType::Transmutase => {
                (pool_mask & enzyme.specificity_mask) != 0
            }
            EnzymeType::Defensase | EnzymeType::Attackase => false,
        }
    }

    fn consider_substrate_candidate(
        &mut self,
        molecule_id: MoleculeId,
        enzyme: &crate::genome::Enzyme,
        max_count: usize,
        seen: &mut usize,
        out: &mut SampledSubstrateIds,
    ) {
        if max_count == 0 || molecule_id.index() >= self.molecules.len() {
            return;
        }
        let molecule = self.molecules[molecule_id.index()].molecule;
        self.operation_counters.substrate_candidates_scanned = self
            .operation_counters
            .substrate_candidates_scanned
            .saturating_add(1);
        if !bio::enzyme_accepts(enzyme, &molecule) {
            return;
        }
        *seen += 1;
        if out.len() < max_count {
            out.push(molecule_id);
        } else {
            let j = (self.rng.next_f64() * *seen as f64) as usize;
            if j < max_count {
                out.replace(j, molecule_id);
            }
        }
    }

    fn should_secrete(
        &mut self,
        cell_id: CellId,
        product: Molecule,
        enzyme: &crate::genome::Enzyme,
    ) -> bool {
        let reserve = self.cells[cell_id.index()].genome.desired_element_reserve;
        for element in ELEMENT_ORDER {
            if product.composition.count(element) > 0
                && self.cells[cell_id.index()].internal_element_counts[element.index()] < reserve
            {
                return false;
            }
        }
        let probability = if enzyme.secretion_prob.is_finite() {
            enzyme.secretion_prob
        } else {
            self.cells[cell_id.index()].genome.default_secretion_prob
        };
        self.rng.chance(probability)
    }

    fn add_product_around(
        &mut self,
        center_tile: TileId,
        product: Molecule,
    ) -> Result<(), WorldError> {
        let choice = self.rng.usize(9);
        let (x, y) = self
            .tile_xy(center_tile)
            .ok_or(WorldError::InvalidTile(center_tile))?;
        let tile_id = self.wrapped_tile_id(
            x as isize + MOORE_WITH_CENTER_DX[choice],
            y as isize + MOORE_WITH_CENTER_DY[choice],
        );
        self.add_tile_molecule_record(tile_id, product)?;
        Ok(())
    }

    fn add_enval_around(
        &mut self,
        center_tile: TileId,
        enval_delta: f32,
    ) -> Result<(), WorldError> {
        if !enval_delta.is_finite() || enval_delta == 0.0 {
            return Ok(());
        }
        let choice = self.rng.usize(9);
        let (x, y) = self
            .tile_xy(center_tile)
            .ok_or(WorldError::InvalidTile(center_tile))?;
        let tile_id = self.wrapped_tile_id(
            x as isize + MOORE_WITH_CENTER_DX[choice],
            y as isize + MOORE_WITH_CENTER_DY[choice],
        );
        self.adjust_tile_enval(tile_id, enval_delta)
    }

    fn divide_cell(&mut self, cell_id: CellId, tile_id: TileId, local_enval: f32) {
        if self.cells[cell_id.index()].state != CellState::Active {
            return;
        }
        let parent_lineage = self.cells[cell_id.index()].lineage_id;
        let mut child_genome = self.cells[cell_id.index()]
            .genome
            .mutate(&mut self.rng, local_enval);
        child_genome.lineage_id = parent_lineage;

        let child_energy =
            self.cells[cell_id.index()].energy * (0.5 + (self.rng.next_f64() - 0.5) * 0.1);
        self.cells[cell_id.index()].energy =
            (self.cells[cell_id.index()].energy - child_energy).max(0.0);

        let mut child_molecules = Vec::new();
        let mut index = self.cells[cell_id.index()].molecules.len();
        while index > 0 {
            index -= 1;
            if self.rng.chance(0.5) {
                if let Ok(molecule_id) = self.remove_molecule_from_cell_slot(cell_id, index) {
                    child_molecules.push(molecule_id);
                }
            }
        }

        let candidates = self.empty_tiles_within_radius_two(tile_id);
        if candidates.is_empty() {
            self.cells[cell_id.index()].energy += child_energy;
            for molecule_id in child_molecules {
                let _ = self.add_existing_molecule_to_cell(cell_id, molecule_id);
            }
            return;
        }

        let dest_tile = candidates[self.rng.usize(candidates.len())];
        let child_id = match self.spawn_cell_with_genome_at(dest_tile, child_genome) {
            Ok(child_id) => child_id,
            Err(_) => {
                self.cells[cell_id.index()].energy += child_energy;
                for molecule_id in child_molecules {
                    let _ = self.add_existing_molecule_to_cell(cell_id, molecule_id);
                }
                return;
            }
        };
        self.cells[child_id.index()].energy = child_energy;
        self.cells[child_id.index()].lineage_id = parent_lineage;
        self.cells[child_id.index()].genome.lineage_id = parent_lineage;
        for molecule_id in child_molecules {
            let _ = self.add_existing_molecule_to_cell(child_id, molecule_id);
        }
        self.operation_counters.cell_divisions =
            self.operation_counters.cell_divisions.saturating_add(1);
        self.reaction_counters.divisions = self.reaction_counters.divisions.saturating_add(1);
        if self
            .rng
            .chance(self.cells[cell_id.index()].genome.post_divide_mortality)
        {
            self.kill_cell_and_release(cell_id);
        }
    }

    fn empty_tiles_within_radius_two(&self, center_tile: TileId) -> Vec<TileId> {
        let (x, y) = match self.tile_xy(center_tile) {
            Some(xy) => xy,
            None => return Vec::new(),
        };
        let mut candidates = Vec::new();
        for dx in -2..=2 {
            for dy in -2..=2 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let tile_id = self.wrapped_tile_id(x as isize + dx, y as isize + dy);
                if self.tiles[tile_id.index()].cell.is_none() {
                    candidates.push(tile_id);
                }
            }
        }
        candidates
    }

    fn random_empty_tile(&mut self, max_attempts: usize) -> Option<TileId> {
        for _ in 0..max_attempts {
            let x = self.rng.usize(self.width);
            let y = self.rng.usize(self.height);
            let tile_id = TileId(self.index_xy(x, y));
            if self.tiles[tile_id.index()].cell.is_none() {
                return Some(tile_id);
            }
        }

        if self.tile_count == 0 {
            return None;
        }
        let start = self.rng.usize(self.tile_count);
        for offset in 0..self.tile_count {
            let tile_id = TileId((start + offset) % self.tile_count);
            if self.tiles[tile_id.index()].cell.is_none() {
                return Some(tile_id);
            }
        }
        None
    }

    fn kill_cell_and_release(&mut self, cell_id: CellId) {
        if cell_id.index() >= self.cells.len()
            || self.cells[cell_id.index()].state == CellState::Dead
        {
            return;
        }
        let tile_id = self.cells[cell_id.index()].tile_id;
        let molecules = std::mem::take(&mut self.cells[cell_id.index()].molecules);
        self.cells[cell_id.index()].internal_element_counts = [0; ELEMENT_COUNT];
        self.cells[cell_id.index()].internal_element_mask = 0;
        self.cells[cell_id.index()].internal_atom_count = 0;
        if let Some(tile_id) = tile_id {
            for molecule_id in molecules {
                let _ = self.add_existing_molecule_to_tile(tile_id, molecule_id);
            }
            if self.tiles[tile_id.index()].cell == Some(cell_id) {
                self.tiles[tile_id.index()].cell = None;
            }
        }
        self.cells[cell_id.index()].energy = 0.0;
        self.cells[cell_id.index()].state = CellState::Dead;
        self.cells[cell_id.index()].tile_id = None;
        self.cells[cell_id.index()].death_sim_time = Some(self.sim_time_seconds);
        self.remove_active_cell(cell_id);
        self.record_lineage_death(self.cells[cell_id.index()].lineage_id);
        self.death_count = self.death_count.saturating_add(1);
        self.operation_counters.cell_deaths = self.operation_counters.cell_deaths.saturating_add(1);
    }

    fn remove_active_cell(&mut self, cell_id: CellId) {
        let slot = match self.cells[cell_id.index()].active_slot.take() {
            Some(slot) => slot,
            None => return,
        };
        if slot >= self.active_cells.len() {
            return;
        }
        let last = self.active_cells.len() - 1;
        if slot != last {
            let swapped = self.active_cells[last];
            self.active_cells[slot] = swapped;
            if swapped.index() < self.cells.len() {
                self.cells[swapped.index()].active_slot = Some(slot);
            }
        }
        self.active_cells.pop();
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

    fn resolve_predation(&mut self) {
        if !self.config.predation_enabled || self.active_cells.len() < 2 {
            return;
        }

        self.predation_occupied_tiles.clear();
        for cell_id in self.active_cells.iter().copied() {
            if cell_id.index() >= self.cells.len() {
                continue;
            }
            let cell = &self.cells[cell_id.index()];
            if cell.state == CellState::Active {
                if let Some(tile_id) = cell.tile_id {
                    self.predation_occupied_tiles.push(tile_id);
                }
            }
        }
        self.predation_occupied_tiles
            .sort_unstable_by_key(|tile| tile.index());

        let needs_tiny_world_dedupe = self.width <= 2 || self.height <= 2;
        let mut seen_pairs = HashSet::<(usize, usize)>::new();
        for occupied_index in 0..self.predation_occupied_tiles.len() {
            let tile_id = self.predation_occupied_tiles[occupied_index];
            self.operation_counters.predation_occupied_tiles_considered = self
                .operation_counters
                .predation_occupied_tiles_considered
                .saturating_add(1);
            let neighbors = self.neighbors[tile_id.index()];
            for other_tile in [
                neighbors.right,
                neighbors.down,
                neighbors.down_right,
                neighbors.up_right,
            ] {
                if other_tile == tile_id {
                    continue;
                }
                if needs_tiny_world_dedupe {
                    let a = tile_id.index().min(other_tile.index());
                    let b = tile_id.index().max(other_tile.index());
                    if !seen_pairs.insert((a, b)) {
                        continue;
                    }
                }
                self.operation_counters.predation_candidate_neighbor_pairs = self
                    .operation_counters
                    .predation_candidate_neighbor_pairs
                    .saturating_add(1);
                if self.tiles[other_tile.index()].cell.is_none() {
                    continue;
                }
                self.resolve_predation_between_tiles(tile_id, other_tile);
            }
        }
    }

    fn resolve_predation_between_tiles(&mut self, tile_a: TileId, tile_b: TileId) {
        let Some(cell_a) = self.tiles.get(tile_a.index()).and_then(|tile| tile.cell) else {
            return;
        };
        let Some(cell_b) = self.tiles.get(tile_b.index()).and_then(|tile| tile.cell) else {
            return;
        };
        self.operation_counters.predation_pairs_checked = self
            .operation_counters
            .predation_pairs_checked
            .saturating_add(1);
        if self.is_active_cross_lineage_pair(cell_a, cell_b) {
            self.operation_counters.predation_cross_lineage_pairs = self
                .operation_counters
                .predation_cross_lineage_pairs
                .saturating_add(1);
        } else {
            return;
        }
        if self.cells[cell_a.index()].combat_attack_total == 0
            && self.cells[cell_b.index()].combat_attack_total == 0
        {
            return;
        }
        let Some(outcome) = self.resolve_predation_between_cells(cell_a, cell_b) else {
            return;
        };
        self.execute_predation_outcome(outcome);
    }

    fn is_active_cross_lineage_pair(&self, cell_a: CellId, cell_b: CellId) -> bool {
        if cell_a == cell_b
            || cell_a.index() >= self.cells.len()
            || cell_b.index() >= self.cells.len()
        {
            return false;
        }
        let a = &self.cells[cell_a.index()];
        let b = &self.cells[cell_b.index()];
        a.state == CellState::Active && b.state == CellState::Active && a.lineage_id != b.lineage_id
    }

    fn resolve_predation_between_cells(
        &self,
        cell_a: CellId,
        cell_b: CellId,
    ) -> Option<PredationOutcome> {
        if cell_a == cell_b
            || cell_a.index() >= self.cells.len()
            || cell_b.index() >= self.cells.len()
        {
            return None;
        }
        let a = &self.cells[cell_a.index()];
        let b = &self.cells[cell_b.index()];
        if a.state != CellState::Active || b.state != CellState::Active {
            return None;
        }
        if a.lineage_id == b.lineage_id {
            return None;
        }

        let a_attack = a.combat_attack_total;
        let a_defense = a.combat_defense_total;
        let b_attack = b.combat_attack_total;
        let b_defense = b.combat_defense_total;
        let a_can_kill = a_attack > 0 && a_attack > b_defense;
        let b_can_kill = b_attack > 0 && b_attack > a_defense;
        if !a_can_kill && !b_can_kill {
            return None;
        }

        let a_margin = i64::from(a_attack) - i64::from(b_defense);
        let b_margin = i64::from(b_attack) - i64::from(a_defense);
        if a_can_kill && !b_can_kill {
            return Some(PredationOutcome {
                winner: cell_a,
                loser: cell_b,
            });
        }
        if b_can_kill && !a_can_kill {
            return Some(PredationOutcome {
                winner: cell_b,
                loser: cell_a,
            });
        }
        if a_margin == b_margin {
            return None;
        }
        if a_margin > b_margin {
            Some(PredationOutcome {
                winner: cell_a,
                loser: cell_b,
            })
        } else {
            Some(PredationOutcome {
                winner: cell_b,
                loser: cell_a,
            })
        }
    }

    fn execute_predation_outcome(&mut self, outcome: PredationOutcome) {
        let predator_id = outcome.winner;
        let prey_id = outcome.loser;
        if predator_id == prey_id
            || predator_id.index() >= self.cells.len()
            || prey_id.index() >= self.cells.len()
            || self.cells[predator_id.index()].state != CellState::Active
            || self.cells[prey_id.index()].state != CellState::Active
        {
            return;
        }

        let prey_enzymes = self.cells[prey_id.index()].genome.enzymes.clone();
        let transfer_stats = {
            let rng = &mut self.rng;
            let predator = &mut self.cells[predator_id.index()];
            let transfer_stats = predator.genome.absorb_predation_enzymes(&prey_enzymes, rng);
            predator.refresh_combat_totals();
            transfer_stats
        };

        let absorbed_energy = self.cells[prey_id.index()].energy.max(0.0);
        if absorbed_energy > 0.0 {
            self.cells[predator_id.index()].energy += absorbed_energy;
            self.cells[predator_id.index()].time_without_food =
                (self.cells[predator_id.index()].time_without_food - absorbed_energy * 0.2)
                    .max(0.0);
        }

        let prey_molecules = std::mem::take(&mut self.cells[prey_id.index()].molecules);
        self.cells[prey_id.index()].internal_element_counts = [0; ELEMENT_COUNT];
        self.cells[prey_id.index()].internal_element_mask = 0;
        self.cells[prey_id.index()].internal_atom_count = 0;
        for molecule_id in prey_molecules {
            if molecule_id.index() >= self.molecules.len() {
                continue;
            }
            let composition = self.molecules[molecule_id.index()].molecule.composition;
            let owner_slot = self.cells[predator_id.index()].molecules.len();
            self.cells[predator_id.index()].molecules.push(molecule_id);
            let _ = self.apply_cell_composition_delta(predator_id, composition, 1);
            let record = &mut self.molecules[molecule_id.index()];
            record.owner = MoleculeOwner::Cell(predator_id);
            record.owner_slot = owner_slot;
            record.last_diffusion_dir = None;
            record.wheel_index = None;
            record.wheel_pos = None;
            record.diffusion_tick = 0;
        }

        if let Some(tile_id) = self.cells[prey_id.index()].tile_id {
            if self.tiles[tile_id.index()].cell == Some(prey_id) {
                self.tiles[tile_id.index()].cell = None;
            }
        }
        self.cells[prey_id.index()].energy = 0.0;
        self.cells[prey_id.index()].state = CellState::Dead;
        self.cells[prey_id.index()].tile_id = None;
        self.cells[prey_id.index()].death_sim_time = Some(self.sim_time_seconds);
        self.remove_active_cell(prey_id);
        self.record_lineage_death(self.cells[prey_id.index()].lineage_id);
        self.death_count = self.death_count.saturating_add(1);
        self.operation_counters.cell_deaths = self.operation_counters.cell_deaths.saturating_add(1);

        self.predation_event_count = self.predation_event_count.saturating_add(1);
        self.cells_consumed_count = self.cells_consumed_count.saturating_add(1);
        self.operation_counters.predation_events =
            self.operation_counters.predation_events.saturating_add(1);
        self.operation_counters.predation_cells_consumed = self
            .operation_counters
            .predation_cells_consumed
            .saturating_add(1);
        self.predator_energy_gained += absorbed_energy;
        self.predation_enzyme_transfer_count = self
            .predation_enzyme_transfer_count
            .saturating_add((transfer_stats.added + transfer_stats.replacements) as u64);
        self.predation_enzyme_replacement_count = self
            .predation_enzyme_replacement_count
            .saturating_add(transfer_stats.replacements as u64);
    }

    fn validate_molecule_state(&self, molecule_id: MoleculeId) -> Result<(), InvariantError> {
        let record = &self.molecules[molecule_id.index()];
        if !record.molecule.energy.is_finite()
            || !record.molecule.elemental_energy_sum.is_finite()
            || !record.molecule.polarity.is_finite()
            || !record.molecule.diffusion_rate.is_finite()
        {
            return Err(InvariantError::NonFiniteMoleculeState(molecule_id));
        }
        Ok(())
    }
}

fn cell_state_label(state: CellState) -> &'static str {
    match state {
        CellState::Active => "active",
        CellState::Dead => "dead",
    }
}

fn inspect_genome(genome: &Genome) -> GenomeDetailInspection {
    GenomeDetailInspection {
        optimal_enval: genome.optimal_enval,
        repro_threshold: genome.repro_threshold,
        initial_energy: genome.initial_energy,
        decay_time: genome.decay_time,
        default_secretion_prob: genome.default_secretion_prob,
        mutation_rate: genome.mutation_rate,
        post_divide_mortality: genome.post_divide_mortality,
        desired_element_reserve: genome.desired_element_reserve,
        enval_stress_factor: genome.enval_stress_factor,
        enval_mutation_floor: genome.enval_mutation_floor,
        maintenance_cost_per_sec: genome.maintenance_cost_per_sec,
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
        enzyme_type: enzyme.enzyme_type.as_str(),
        is_metabolic: enzyme.enzyme_type.is_metabolic(),
        is_combat: enzyme.enzyme_type.is_combat(),
        specificity_mask: enzyme.specificity_mask,
        specificity_elements: specificity_elements(enzyme.specificity_mask),
        bond_multiplier: enzyme.bond_multiplier,
        bond_cost_fraction: enzyme.bond_cost_fraction,
        bond_harvest_fraction: enzyme.bond_harvest_fraction,
        downhill_harvest_fraction: enzyme.downhill_harvest_fraction,
        secretion_prob: enzyme.secretion_prob,
        enval_sigma: enzyme.enval_sigma,
        enval_throughput: enzyme.enval_throughput,
        enval_energy_fraction: enzyme.enval_energy_fraction,
        enval_release_fraction: enzyme.enval_release_fraction,
        enval_pump: enzyme.enval_pump,
        combat_level: enzyme.combat_level,
    }
}

fn specificity_elements(mask: u8) -> Vec<&'static str> {
    ELEMENT_ORDER
        .iter()
        .copied()
        .filter(|element| mask & element.mask() != 0)
        .map(|element| element.symbol())
        .collect()
}

fn inspect_molecule(
    list_index: usize,
    molecule_id: MoleculeId,
    record: &MoleculeRecord,
) -> MoleculeDetailInspection {
    let molecule = record.molecule;
    MoleculeDetailInspection {
        list_index,
        molecule_id: molecule_id.index(),
        composition_counts: *molecule.composition.counts(),
        formula: composition_formula(molecule.composition),
        size: molecule.size,
        element_mask: molecule.element_mask,
        bond_multiplier: molecule.bond_multiplier,
        elemental_energy_sum: molecule.elemental_energy_sum,
        energy: molecule.energy,
        polarity: molecule.polarity,
        diffusion_rate: molecule.diffusion_rate,
        diffusion_period: molecule.diffusion_period,
    }
}

fn reaction_molecule_summary(molecule: Molecule) -> ReactionMoleculeSummary {
    ReactionMoleculeSummary {
        composition_counts: *molecule.composition.counts(),
        formula: composition_formula(molecule.composition),
        size: molecule.size,
        element_mask: molecule.element_mask,
        bond_multiplier: molecule.bond_multiplier,
        elemental_energy_sum: molecule.elemental_energy_sum,
        energy: molecule.energy,
        polarity: molecule.polarity,
    }
}

fn composition_formula(composition: Composition) -> String {
    let mut formula = String::new();
    for element in ELEMENT_ORDER {
        let count = composition.count(element);
        if count == 0 {
            continue;
        }
        formula.push_str(element.symbol());
        if count > 1 {
            formula.push_str(&count.to_string());
        }
    }
    if formula.is_empty() {
        "empty".to_owned()
    } else {
        formula
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

fn build_predation_pairs(neighbors: &[NeighborIndices]) -> Vec<(TileId, TileId)> {
    let mut pairs = Vec::new();
    let mut seen = HashSet::<(usize, usize)>::new();
    for (tile_index, neighbors) in neighbors.iter().copied().enumerate() {
        let tile_id = TileId(tile_index);
        for other_tile in [
            neighbors.right,
            neighbors.down,
            neighbors.down_right,
            neighbors.up_right,
        ] {
            if other_tile == tile_id {
                continue;
            }
            let a = tile_id.index().min(other_tile.index());
            let b = tile_id.index().max(other_tile.index());
            if seen.insert((a, b)) {
                pairs.push((tile_id, other_tile));
            }
        }
    }
    pairs
}

fn element_mask_from_counts(counts: [u32; ELEMENT_COUNT]) -> u8 {
    let mut mask = 0_u8;
    for element in ELEMENT_ORDER {
        if counts[element.index()] > 0 {
            mask |= element.mask();
        }
    }
    mask
}

fn initial_tile_diffusion_rotor(tile_index: usize) -> u8 {
    let mut hash = (tile_index as u32) ^ 0x9e37_79b9;
    hash = (hash ^ (hash >> 16)).wrapping_mul(0x85eb_ca6b);
    hash = (hash ^ (hash >> 13)).wrapping_mul(0xc2b2_ae35);
    ((hash ^ (hash >> 16)) & 3) as u8
}

fn choose_diffusion_direction(rotor: &mut u8, tie_mask: u8) -> Option<u8> {
    if tie_mask == 0 {
        return None;
    }
    if tie_mask & (tie_mask - 1) == 0 {
        return Some(match tie_mask {
            1 => 0,
            2 => 1,
            4 => 2,
            _ => 3,
        });
    }

    let rotor_start = *rotor & 3;
    for step in 0..4 {
        let dir = (rotor_start + step) & 3;
        if tie_mask & (1 << dir) != 0 {
            *rotor = (dir + 1) & 3;
            return Some(dir);
        }
    }
    None
}

#[derive(Debug)]
pub enum WorldError {
    Config(ConfigError),
    Molecule(MoleculeError),
    InvalidTile(TileId),
    InvalidCell(CellId),
    InvalidMolecule(MoleculeId),
    OccupiedTile(TileId),
    NonFiniteEnvalInput(f32),
    InvalidEnergyInput(f64),
    GenomePatch(String),
    TileCountOverflow(TileId),
    TileCountUnderflow(TileId),
    CellCountOverflow(CellId),
    CellCountUnderflow(CellId),
    MoleculeSlotMismatch(MoleculeId),
}

impl fmt::Display for WorldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Config(err) => write!(f, "invalid config: {err}"),
            Self::Molecule(err) => write!(f, "invalid molecule: {err}"),
            Self::InvalidTile(tile) => write!(f, "invalid tile id {}", tile.index()),
            Self::InvalidCell(cell) => write!(f, "invalid cell id {}", cell.index()),
            Self::InvalidMolecule(molecule) => {
                write!(f, "invalid molecule id {}", molecule.index())
            }
            Self::OccupiedTile(tile) => write!(f, "tile {} is already occupied", tile.index()),
            Self::NonFiniteEnvalInput(value) => write!(f, "non-finite enval value {value}"),
            Self::InvalidEnergyInput(value) => {
                write!(
                    f,
                    "invalid cell energy override: expected finite nonnegative value, got {value}"
                )
            }
            Self::GenomePatch(message) => write!(f, "invalid genome patch: {message}"),
            Self::TileCountOverflow(tile) => {
                write!(f, "tile count overflow at tile {}", tile.index())
            }
            Self::TileCountUnderflow(tile) => {
                write!(f, "tile count underflow at tile {}", tile.index())
            }
            Self::CellCountOverflow(cell) => {
                write!(f, "cell count overflow at cell {}", cell.index())
            }
            Self::CellCountUnderflow(cell) => {
                write!(f, "cell count underflow at cell {}", cell.index())
            }
            Self::MoleculeSlotMismatch(molecule) => write!(
                f,
                "molecule {} is not in its recorded owner slot",
                molecule.index()
            ),
        }
    }
}

impl Error for WorldError {}

impl From<ConfigError> for WorldError {
    fn from(value: ConfigError) -> Self {
        Self::Config(value)
    }
}

impl From<MoleculeError> for WorldError {
    fn from(value: MoleculeError) -> Self {
        Self::Molecule(value)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InvariantError {
    MismatchedWorldArrayLengths,
    NonFiniteEnval(TileId),
    InvalidNeighbor {
        tile: TileId,
        neighbor: TileId,
    },
    InvalidPredationPair {
        left: TileId,
        right: TileId,
    },
    DuplicatePredationPair {
        left: TileId,
        right: TileId,
    },
    InvalidTileMoleculeId {
        tile: TileId,
        molecule: MoleculeId,
    },
    InvalidCellMoleculeId {
        cell: CellId,
        molecule: MoleculeId,
    },
    InvalidCellId {
        tile: TileId,
        cell: CellId,
    },
    InvalidActiveCell(CellId),
    DuplicateCellOccupancy(CellId),
    DuplicateActiveCell(CellId),
    DeadCellOnTile(CellId),
    DeadActiveCell(CellId),
    DeadCellOwnsState(CellId),
    LiveCellNotOnTile(CellId),
    WrongCellTile {
        cell: CellId,
        expected_tile: TileId,
        actual_tile: Option<TileId>,
    },
    WrongActiveCellSlot {
        cell: CellId,
        expected_slot: usize,
        actual_slot: Option<usize>,
    },
    NonFiniteCellEnergy(CellId),
    InvalidGenomeEnzymeCount(CellId),
    InvalidEnzymeSpecificity(CellId),
    CombatTotalsMismatch(CellId),
    DuplicateMoleculeOwner(MoleculeId),
    WrongTileMoleculeOwner {
        molecule: MoleculeId,
        expected_tile: TileId,
        actual_owner: MoleculeOwner,
    },
    WrongCellMoleculeOwner {
        molecule: MoleculeId,
        expected_cell: CellId,
        actual_owner: MoleculeOwner,
    },
    WrongMoleculeOwnerSlot {
        molecule: MoleculeId,
        expected_slot: usize,
        actual_slot: usize,
    },
    NonFiniteMoleculeState(MoleculeId),
    TileCountOverflow(TileId),
    CellCountOverflow(CellId),
    UnlistedOwnedMolecule(MoleculeId),
    FreeMoleculeListed(MoleculeId),
    FreeMoleculeScheduled(MoleculeId),
    InvalidFreeListMolecule(MoleculeId),
    DuplicateFreeListMolecule(MoleculeId),
    OwnedMoleculeInFreeList(MoleculeId),
    FreeMoleculeMissingFromFreeList(MoleculeId),
    TileElementCountsMismatch(TileId),
    TileElementMaskMismatch(TileId),
    TileMassCountMismatch(TileId),
    CellElementCountsMismatch(CellId),
    CellElementMaskMismatch(CellId),
    CellMassCountMismatch(CellId),
    InvalidWheelMolecule(MoleculeId),
    DuplicateWheelMolecule(MoleculeId),
    NonTileMoleculeScheduled(MoleculeId),
    WrongWheelPosition {
        molecule: MoleculeId,
        expected_bucket: usize,
        expected_pos: usize,
        actual_bucket: Option<usize>,
        actual_pos: Option<usize>,
    },
    WrongWheelBucket {
        molecule: MoleculeId,
        tick: u32,
        bucket: usize,
    },
    UnscheduledMolecule(MoleculeId),
    LineagePopulationMismatch {
        lineage: LineageId,
        expected: u64,
        actual: u64,
    },
    NonFinitePredationEnergy,
}

impl fmt::Display for InvariantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MismatchedWorldArrayLengths => {
                f.write_str("world arrays have mismatched lengths")
            }
            Self::NonFiniteEnval(tile) => write!(f, "tile {} has non-finite enval", tile.index()),
            Self::InvalidNeighbor { tile, neighbor } => write!(
                f,
                "tile {} has invalid neighbor {}",
                tile.index(),
                neighbor.index()
            ),
            Self::InvalidPredationPair { left, right } => write!(
                f,
                "invalid predation pair {} <-> {}",
                left.index(),
                right.index()
            ),
            Self::DuplicatePredationPair { left, right } => write!(
                f,
                "duplicate predation pair {} <-> {}",
                left.index(),
                right.index()
            ),
            Self::InvalidTileMoleculeId { tile, molecule } => write!(
                f,
                "tile {} references invalid molecule {}",
                tile.index(),
                molecule.index()
            ),
            Self::InvalidCellMoleculeId { cell, molecule } => write!(
                f,
                "cell {} references invalid molecule {}",
                cell.index(),
                molecule.index()
            ),
            Self::InvalidCellId { tile, cell } => write!(
                f,
                "tile {} references invalid cell {}",
                tile.index(),
                cell.index()
            ),
            Self::InvalidActiveCell(cell) => {
                write!(f, "active list references invalid cell {}", cell.index())
            }
            Self::DuplicateCellOccupancy(cell) => {
                write!(f, "cell {} appears on more than one tile", cell.index())
            }
            Self::DuplicateActiveCell(cell) => write!(
                f,
                "cell {} appears more than once in active list",
                cell.index()
            ),
            Self::DeadCellOnTile(cell) => {
                write!(f, "dead cell {} is present on a tile", cell.index())
            }
            Self::DeadActiveCell(cell) => {
                write!(f, "dead cell {} is present in active list", cell.index())
            }
            Self::DeadCellOwnsState(cell) => {
                write!(f, "dead cell {} still owns tile or molecules", cell.index())
            }
            Self::LiveCellNotOnTile(cell) => {
                write!(f, "live cell {} is not present on a tile", cell.index())
            }
            Self::WrongCellTile {
                cell,
                expected_tile,
                actual_tile,
            } => write!(
                f,
                "cell {} tile mismatch: expected {}, got {:?}",
                cell.index(),
                expected_tile.index(),
                actual_tile
            ),
            Self::WrongActiveCellSlot {
                cell,
                expected_slot,
                actual_slot,
            } => write!(
                f,
                "cell {} active slot mismatch: expected {}, got {:?}",
                cell.index(),
                expected_slot,
                actual_slot
            ),
            Self::NonFiniteCellEnergy(cell) => {
                write!(f, "cell {} has non-finite energy", cell.index())
            }
            Self::InvalidGenomeEnzymeCount(cell) => write!(
                f,
                "cell {} genome enzyme count is outside [1, 10]",
                cell.index()
            ),
            Self::InvalidEnzymeSpecificity(cell) => write!(
                f,
                "cell {} has invalid metabolic specificity mask",
                cell.index()
            ),
            Self::CombatTotalsMismatch(cell) => write!(
                f,
                "cell {} cached combat totals do not match its genome",
                cell.index()
            ),
            Self::DuplicateMoleculeOwner(molecule) => write!(
                f,
                "molecule {} appears in more than one owner list",
                molecule.index()
            ),
            Self::WrongTileMoleculeOwner {
                molecule,
                expected_tile,
                actual_owner,
            } => write!(
                f,
                "molecule {} owner mismatch: expected tile {}, got {:?}",
                molecule.index(),
                expected_tile.index(),
                actual_owner
            ),
            Self::WrongCellMoleculeOwner {
                molecule,
                expected_cell,
                actual_owner,
            } => write!(
                f,
                "molecule {} owner mismatch: expected cell {}, got {:?}",
                molecule.index(),
                expected_cell.index(),
                actual_owner
            ),
            Self::WrongMoleculeOwnerSlot {
                molecule,
                expected_slot,
                actual_slot,
            } => write!(
                f,
                "molecule {} owner slot mismatch: expected {}, got {}",
                molecule.index(),
                expected_slot,
                actual_slot
            ),
            Self::NonFiniteMoleculeState(molecule) => write!(
                f,
                "molecule {} has non-finite derived state",
                molecule.index()
            ),
            Self::TileCountOverflow(tile) => write!(f, "tile {} count overflow", tile.index()),
            Self::CellCountOverflow(cell) => write!(f, "cell {} count overflow", cell.index()),
            Self::UnlistedOwnedMolecule(molecule) => write!(
                f,
                "molecule {} owner record is not listed by that owner",
                molecule.index()
            ),
            Self::FreeMoleculeListed(molecule) => write!(
                f,
                "free molecule {} appears in an owner list",
                molecule.index()
            ),
            Self::FreeMoleculeScheduled(molecule) => write!(
                f,
                "free molecule {} is scheduled for diffusion",
                molecule.index()
            ),
            Self::InvalidFreeListMolecule(molecule) => write!(
                f,
                "free-list references invalid molecule {}",
                molecule.index()
            ),
            Self::DuplicateFreeListMolecule(molecule) => write!(
                f,
                "free-list references molecule {} more than once",
                molecule.index()
            ),
            Self::OwnedMoleculeInFreeList(molecule) => write!(
                f,
                "owned molecule {} appears in the free-list",
                molecule.index()
            ),
            Self::FreeMoleculeMissingFromFreeList(molecule) => write!(
                f,
                "free molecule {} is missing from the free-list",
                molecule.index()
            ),
            Self::TileElementCountsMismatch(tile) => write!(
                f,
                "tile {} element counts do not match molecules",
                tile.index()
            ),
            Self::TileElementMaskMismatch(tile) => write!(
                f,
                "tile {} element mask does not match element counts",
                tile.index()
            ),
            Self::TileMassCountMismatch(tile) => write!(
                f,
                "tile {} mass count does not match molecules",
                tile.index()
            ),
            Self::CellElementCountsMismatch(cell) => write!(
                f,
                "cell {} internal element counts do not match molecules",
                cell.index()
            ),
            Self::CellElementMaskMismatch(cell) => write!(
                f,
                "cell {} internal element mask does not match element counts",
                cell.index()
            ),
            Self::CellMassCountMismatch(cell) => write!(
                f,
                "cell {} internal mass count does not match molecules",
                cell.index()
            ),
            Self::InvalidWheelMolecule(molecule) => write!(
                f,
                "diffusion wheel references invalid molecule {}",
                molecule.index()
            ),
            Self::DuplicateWheelMolecule(molecule) => write!(
                f,
                "diffusion wheel references molecule {} more than once",
                molecule.index()
            ),
            Self::NonTileMoleculeScheduled(molecule) => write!(
                f,
                "non-tile-owned molecule {} is scheduled for diffusion",
                molecule.index()
            ),
            Self::WrongWheelPosition {
                molecule,
                expected_bucket,
                expected_pos,
                actual_bucket,
                actual_pos,
            } => write!(
                f,
                "molecule {} wheel position mismatch: expected bucket {}, pos {}, got {:?}, {:?}",
                molecule.index(),
                expected_bucket,
                expected_pos,
                actual_bucket,
                actual_pos
            ),
            Self::WrongWheelBucket {
                molecule,
                tick,
                bucket,
            } => write!(
                f,
                "molecule {} diffusion tick {} does not map to bucket {}",
                molecule.index(),
                tick,
                bucket
            ),
            Self::UnscheduledMolecule(molecule) => write!(
                f,
                "molecule {} has no diffusion wheel entry",
                molecule.index()
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
            Self::NonFinitePredationEnergy => {
                f.write_str("predation energy-gained counter is non-finite")
            }
        }
    }
}

impl Error for InvariantError {}

#[cfg(test)]
mod tests {
    use super::{InvariantError, MoleculeOwner, RenderBuffers, TileId, World};
    use crate::cell::{CellState, CELL_REACTION_LOG_CAPACITY};
    use crate::chem::{Composition, Element, ELEMENT_ORDER};
    use crate::config::{Config, MoleculeSeedingConfig};
    use crate::genome::{
        Enzyme, EnzymeFieldPatch, EnzymePatchOperation, EnzymeType, Genome, GenomeFieldPatch,
        GenomePatch, LineageId, MAX_CELL_ENZYMES, MIN_CELL_ENZYMES,
    };
    use crate::molecule::Molecule;

    fn small_config(seed: &str, width: usize, height: usize) -> Config {
        Config {
            seed: seed.to_owned(),
            width,
            height,
            ..Config::default()
        }
    }

    fn only_a_config(seed: &str, width: usize, height: usize) -> Config {
        Config {
            seed: seed.to_owned(),
            width,
            height,
            molecule_seeding: MoleculeSeedingConfig {
                b: 0.0,
                c: 0.0,
                d: 0.0,
                e: 0.0,
                f: 0.0,
                bc: 0.0,
            },
            ..Config::default()
        }
    }

    fn combat_genome(
        world: &mut World,
        lineage: u64,
        attack: u32,
        defense: u32,
        energy: f64,
    ) -> Genome {
        let mut genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        genome.lineage_id = LineageId(lineage);
        genome.initial_energy = energy;
        genome.enzymes.clear();
        if attack > 0 {
            genome.enzymes.push(Enzyme::attackase(attack));
        }
        if defense > 0 {
            genome.enzymes.push(Enzyme::defensase(defense));
        }
        if genome.enzymes.is_empty() {
            genome.enzymes.push(Enzyme::anabolase_abc(&mut world.rng));
        }
        genome
    }

    fn spawn_combat_cell(
        world: &mut World,
        tile: TileId,
        lineage: u64,
        attack: u32,
        defense: u32,
        energy: f64,
    ) -> crate::cell::CellId {
        let genome = combat_genome(world, lineage, attack, defense, energy);
        world.spawn_cell_with_genome_at(tile, genome).unwrap()
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
    fn seeded_initialization_is_deterministic() {
        let a = World::new(small_config("deterministic", 10, 8))
            .unwrap()
            .stats();
        let b = World::new(small_config("deterministic", 10, 8))
            .unwrap()
            .stats();
        assert_eq!(a.molecule_count, b.molecule_count);
        assert_eq!(a.total_atom_count, b.total_atom_count);
        assert_eq!(a.element_counts, b.element_counts);
        assert_eq!(a.average_enval.to_bits(), b.average_enval.to_bits());
    }

    #[test]
    fn every_tile_has_at_least_one_a_molecule() {
        let world = World::new(small_config("a-present", 8, 8)).unwrap();
        for tile_index in 0..world.tile_count() {
            let tile_id = TileId(tile_index);
            let has_a = world
                .tile_molecules(tile_id)
                .unwrap()
                .iter()
                .any(|molecule_id| {
                    world
                        .molecule(*molecule_id)
                        .unwrap()
                        .composition
                        .count(Element::A)
                        > 0
                });
            assert!(has_a, "tile {tile_index} lacks seeded A");
        }
        world.check_invariants().unwrap();
    }

    #[test]
    fn tile_element_counts_match_actual_molecules() {
        let world = World::new(small_config("counts", 9, 7)).unwrap();
        for tile_index in 0..world.tile_count() {
            let tile_id = TileId(tile_index);
            let mut actual = [0_u32; crate::chem::ELEMENT_COUNT];
            for molecule_id in world.tile_molecules(tile_id).unwrap() {
                let molecule = world.molecule(*molecule_id).unwrap();
                for element in ELEMENT_ORDER {
                    actual[element.index()] += u32::from(molecule.composition.count(element));
                }
            }
            assert_eq!(world.tile_element_counts(tile_id).unwrap(), actual);
        }
    }

    #[test]
    fn cached_element_masks_track_tile_and_cell_composition() {
        let mut world = World::new(only_a_config("element-mask-cache", 4, 4)).unwrap();
        let tile_id = world.tile_id(0, 0).unwrap();
        let avg_enval = world.avg_enval;
        let genome = Genome::random_founder(&mut world.rng, avg_enval);
        let cell_id = world.spawn_cell_with_genome_at(tile_id, genome).unwrap();

        assert_eq!(
            world.inspect_tile(tile_id).unwrap().element_mask & Element::A.mask(),
            Element::A.mask()
        );
        let molecule = Molecule::new(Composition::single(Element::D), 1.0).unwrap();
        let molecule_id = world.add_cell_molecule_record(cell_id, molecule).unwrap();
        assert_eq!(
            world.cells[cell_id.index()].internal_element_mask & Element::D.mask(),
            Element::D.mask()
        );
        world.consume_molecule(molecule_id).unwrap();
        assert_eq!(
            world.cells[cell_id.index()].internal_element_mask & Element::D.mask(),
            0
        );
        world.check_invariants().unwrap();
    }

    #[test]
    fn base_and_average_enval_are_finite() {
        let world = World::new(small_config("enval", 4, 4)).unwrap();
        assert!(world.base_enval().is_finite());
        assert!(world.average_enval().is_finite());
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
    fn scheduled_molecule_diffusion_preserves_invariants_and_atoms() {
        let mut world = World::new(small_config("diffuse", 12, 10)).unwrap();
        let initial_atoms = world.stats().total_atom_count;
        for _ in 0..200 {
            world.diffuse_molecules();
            world.check_invariants().unwrap();
        }
        assert_eq!(world.stats().total_atom_count, initial_atoms);
    }

    #[test]
    fn molecule_diffusion_moves_some_molecules_in_nontrivial_world() {
        let mut world = World::new(only_a_config("moves", 6, 6)).unwrap();
        let origin = world.tile_id(0, 0).unwrap();
        for _ in 0..12 {
            world
                .add_tile_molecule(origin, Composition::single(Element::B), 1.0)
                .unwrap();
        }
        world.check_invariants().unwrap();
        let before = world.molecule_tile_ids();
        for _ in 0..500 {
            world.diffuse_molecules();
        }
        let after = world.molecule_tile_ids();
        assert_ne!(before, after);
        world.check_invariants().unwrap();
    }

    #[test]
    fn consumed_tile_molecule_slots_are_reused_deterministically() {
        let mut world = World::new(only_a_config("molecule-reuse-tile", 4, 4)).unwrap();
        let tile = world.tile_id(0, 0).unwrap();
        let molecule_id = world
            .add_tile_molecule(tile, Composition::single(Element::B), 1.0)
            .unwrap();
        let arena_len = world.molecules.len();
        let allocations_before = world.operation_counters.molecule_slots_newly_allocated;

        world.consume_molecule(molecule_id).unwrap();
        assert_eq!(world.free_molecule_ids.last().copied(), Some(molecule_id));
        assert_eq!(world.stats().free_molecule_record_count, 1);
        assert_eq!(world.stats().molecule_arena_len, arena_len);
        assert!(!world
            .diffusion_wheel
            .iter()
            .flatten()
            .any(|scheduled| *scheduled == molecule_id));

        let reused = world
            .add_tile_molecule(tile, Composition::single(Element::C), 1.0)
            .unwrap();
        assert_eq!(reused, molecule_id);
        assert_eq!(world.molecules.len(), arena_len);
        assert_eq!(world.operation_counters.molecule_slots_reused, 1);
        assert_eq!(
            world.operation_counters.molecule_slots_newly_allocated,
            allocations_before
        );
        assert_eq!(
            world.molecules[reused.index()].owner,
            MoleculeOwner::Tile(tile)
        );
        assert!(world.tile_molecules(tile).unwrap().contains(&reused));
        world.check_invariants().unwrap();
    }

    #[test]
    fn consumed_cell_molecule_slots_are_reused_deterministically() {
        let mut world = World::new(only_a_config("molecule-reuse-cell", 4, 4)).unwrap();
        let tile = world.tile_id(1, 1).unwrap();
        let genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
        let molecule = Molecule::new(Composition::single(Element::B), 1.0).unwrap();
        let molecule_id = world.add_cell_molecule_record(cell_id, molecule).unwrap();
        let arena_len = world.molecules.len();

        world.consume_molecule(molecule_id).unwrap();
        assert_eq!(world.free_molecule_ids.last().copied(), Some(molecule_id));
        assert_eq!(world.stats().free_molecule_record_count, 1);

        let replacement = Molecule::new(Composition::single(Element::C), 1.0).unwrap();
        let reused = world
            .add_cell_molecule_record(cell_id, replacement)
            .unwrap();
        assert_eq!(reused, molecule_id);
        assert_eq!(world.molecules.len(), arena_len);
        assert_eq!(
            world.molecules[reused.index()].owner,
            MoleculeOwner::Cell(cell_id)
        );
        assert!(world.cells[cell_id.index()].molecules.contains(&reused));
        world.check_invariants().unwrap();
    }

    #[test]
    fn reused_tile_molecules_do_not_keep_stale_diffusion_schedule_entries() {
        let mut world = World::new(only_a_config("molecule-reuse-diffusion", 4, 4)).unwrap();
        let tile = world.tile_id(0, 0).unwrap();
        let molecule_id = world
            .add_tile_molecule(tile, Composition::single(Element::B), 1.0)
            .unwrap();
        assert!(world
            .diffusion_wheel
            .iter()
            .flatten()
            .any(|scheduled| *scheduled == molecule_id));

        world.consume_molecule(molecule_id).unwrap();
        assert!(!world
            .diffusion_wheel
            .iter()
            .flatten()
            .any(|scheduled| *scheduled == molecule_id));

        let reused = world
            .add_tile_molecule(tile, Composition::single(Element::D), 1.0)
            .unwrap();
        assert_eq!(reused, molecule_id);
        let scheduled_count = world
            .diffusion_wheel
            .iter()
            .flatten()
            .filter(|scheduled| **scheduled == molecule_id)
            .count();
        assert_eq!(scheduled_count, 1);
        world.check_invariants().unwrap();
    }

    #[test]
    fn snapshot_roundtrip_preserves_free_list_reuse_order() {
        let mut world = World::new(only_a_config("molecule-reuse-snapshot", 4, 4)).unwrap();
        let tile = world.tile_id(0, 0).unwrap();
        let first = world
            .add_tile_molecule(tile, Composition::single(Element::B), 1.0)
            .unwrap();
        let second = world
            .add_tile_molecule(tile, Composition::single(Element::C), 1.0)
            .unwrap();

        world.consume_molecule(first).unwrap();
        world.consume_molecule(second).unwrap();
        assert_eq!(world.free_molecule_ids, vec![first, second]);

        let bytes = crate::snapshot::to_bytes(&world).unwrap();
        let mut loaded = crate::snapshot::from_bytes(&bytes).unwrap();
        let reused = loaded
            .add_tile_molecule(tile, Composition::single(Element::D), 1.0)
            .unwrap();
        assert_eq!(reused, second);
        loaded.check_invariants().unwrap();
    }

    #[test]
    fn invariant_checker_catches_invalid_molecule_free_list_state() {
        let mut world = World::new(only_a_config("molecule-reuse-invariant", 4, 4)).unwrap();
        let tile = world.tile_id(0, 0).unwrap();
        let molecule_id = world
            .add_tile_molecule(tile, Composition::single(Element::B), 1.0)
            .unwrap();

        world.free_molecule_ids.push(molecule_id);
        assert_eq!(
            world.check_invariants(),
            Err(InvariantError::OwnedMoleculeInFreeList(molecule_id))
        );
    }

    #[test]
    fn render_buffers_ignore_free_molecule_records() {
        let mut world = World::new(only_a_config("molecule-reuse-render", 4, 4)).unwrap();
        let tile = world.tile_id(0, 0).unwrap();
        let molecule_id = world
            .add_tile_molecule(tile, Composition::single(Element::B), 1.0)
            .unwrap();
        world.consume_molecule(molecule_id).unwrap();

        let stats = world.stats();
        let buffers = world.build_render_buffers();
        let rendered_tile_molecules = buffers
            .tile_molecule_count
            .iter()
            .map(|count| *count as usize)
            .sum::<usize>();
        assert_eq!(stats.free_molecule_record_count, 1);
        assert_eq!(rendered_tile_molecules, stats.tile_molecule_count);
        assert!(stats.molecule_arena_len > stats.active_molecule_record_count);
        world.check_invariants().unwrap();
    }

    #[test]
    fn cell_spawn_uses_empty_tile_and_updates_stats() {
        let mut world = World::new(only_a_config("spawn", 8, 8)).unwrap();
        let spawned = world.spawn_founder_cells(4).unwrap();
        assert_eq!(spawned, 4);
        assert_eq!(world.stats().live_cell_count, 4);
        assert_eq!(world.stats().births, 4);
        world.check_invariants().unwrap();
    }

    #[test]
    fn duplicate_cell_occupancy_is_rejected() {
        let mut world = World::new(only_a_config("occupancy", 4, 4)).unwrap();
        let tile = world.tile_id(0, 0).unwrap();
        let genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        world
            .spawn_cell_with_genome_at(tile, genome.clone())
            .unwrap();
        assert!(world.spawn_cell_with_genome_at(tile, genome).is_err());
    }

    #[test]
    fn opportunistic_uptake_transfers_ownership() {
        let mut world = World::new(only_a_config("uptake", 4, 4)).unwrap();
        let tile = world.tile_id(1, 1).unwrap();
        let genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
        world.step_cell(cell_id);
        assert!(world.cells[cell_id.index()].internal_atom_count > 0);
        world.check_invariants().unwrap();
    }

    #[test]
    fn maintenance_can_kill_cell_and_release_molecules() {
        let mut world = World::new(only_a_config("maintenance-death", 4, 4)).unwrap();
        let tile = world.tile_id(1, 1).unwrap();
        let mut genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        genome.initial_energy = 0.001;
        genome.maintenance_cost_per_sec = 10.0;
        let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
        let molecule = Molecule::new(Composition::single(Element::B), 1.0).unwrap();
        world.add_cell_molecule_record(cell_id, molecule).unwrap();
        world.step_cell(cell_id);
        assert_eq!(world.cells[cell_id.index()].state, CellState::Dead);
        assert!(world.tile_molecules(tile).unwrap().len() > 1);
        world.check_invariants().unwrap();
    }

    #[test]
    fn anabolase_basic_reaction_changes_ownership_or_energy() {
        let mut world = World::new(only_a_config("anabolase-world", 4, 4)).unwrap();
        let tile = world.tile_id(1, 1).unwrap();
        let mut genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        genome.optimal_enval = world.tile_enval(tile).unwrap();
        genome.enzymes = vec![Enzyme::anabolase_abc(&mut world.rng)];
        genome.initial_energy = 10.0;
        let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
        world
            .add_tile_molecule(tile, Composition::single(Element::B), 1.0)
            .unwrap();
        world.step_cell(cell_id);
        assert!(world.cells[cell_id.index()].energy.is_finite());
        world.check_invariants().unwrap();
    }

    #[test]
    fn active_cell_reaction_logs_are_available_even_when_empty() {
        let mut world = World::new(only_a_config("reaction-log-empty", 4, 4)).unwrap();
        let tile = world.tile_id(1, 1).unwrap();
        let genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
        let logs = world.inspect_cell_reactions(cell_id, 8).unwrap();
        assert!(logs.available);
        assert_eq!(logs.reason, "recorded");
        assert_eq!(logs.limit, 8);
        assert!(!logs.truncated);
        assert_eq!(logs.reaction_count, 0);
        assert_eq!(logs.returned_count, 0);
        assert_eq!(logs.order, "newest_first");
        assert!(logs.reactions.is_empty());
    }

    #[test]
    fn successful_reactions_are_logged_and_bounded() {
        let mut world = World::new(only_a_config("reaction-log-bounded", 4, 4)).unwrap();
        let tile = world.tile_id(1, 1).unwrap();
        let mut genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        genome.optimal_enval = world.tile_enval(tile).unwrap();
        let mut catabolase = Enzyme::catabolase_abc(&mut world.rng);
        catabolase.enval_sigma = 1000.0;
        genome.enzymes = vec![catabolase];
        genome.initial_energy = 10.0;
        genome.repro_threshold = 1_000_000.0;
        genome.decay_time = 1_000_000.0;
        genome.maintenance_cost_per_sec = 0.0;
        let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();

        let target_reactions = CELL_REACTION_LOG_CAPACITY + 5;
        for _ in 0..target_reactions {
            let molecule = Molecule::new(Composition::bc_dimer(), 1.5).unwrap();
            world.add_cell_molecule_record(cell_id, molecule).unwrap();
        }
        for _ in 0..target_reactions {
            world.step_cell(cell_id);
        }

        assert_eq!(
            world.cells[cell_id.index()].recent_reactions.len(),
            CELL_REACTION_LOG_CAPACITY
        );
        let logs = world.inspect_cell_reactions(cell_id, 2).unwrap();
        assert!(logs.available);
        assert_eq!(logs.reason, "recorded");
        assert_eq!(logs.limit, 2);
        assert_eq!(logs.reaction_count, CELL_REACTION_LOG_CAPACITY);
        assert_eq!(logs.returned_count, 2);
        assert!(logs.truncated);
        assert_eq!(logs.order, "newest_first");
        assert_eq!(logs.reactions.len(), 2);
        let record = &logs.reactions[0];
        assert_eq!(record.cell_id, cell_id.index());
        assert_eq!(record.tile_id, tile.index());
        assert_eq!(record.x, 1);
        assert_eq!(record.y, 1);
        assert_eq!(record.enzyme_index, 0);
        assert_eq!(record.enzyme_type, "catabolase");
        assert_eq!(record.status, "success");
        assert_eq!(record.substrate_count, 1);
        assert_eq!(record.substrates.len(), 1);
        assert_eq!(record.substrates[0].formula, "BC");
        assert!(!record.byproducts.is_empty());
        assert!(record.energy_after > record.energy_before);
        assert!(
            (record.delta_cell_energy - (record.energy_after - record.energy_before)).abs()
                <= 1.0e-12
        );

        let detail = world.inspect_cell_detail(cell_id, 4, 3).unwrap();
        assert!(detail.recent_reactions.available);
        assert_eq!(detail.recent_reactions.returned_count, 3);
    }

    #[test]
    fn genome_patch_updates_selected_cell_fields_and_enzymes() {
        let mut world = World::new(only_a_config("genome-patch-cell", 4, 4)).unwrap();
        let tile = world.tile_id(1, 1).unwrap();
        let genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
        let energy_before = world.cells[cell_id.index()].energy;
        let lineage_before = world.cells[cell_id.index()].lineage_id;
        let patch = GenomePatch {
            schema: Some(crate::genome::GENOME_PATCH_SCHEMA.to_owned()),
            genome: Some(GenomeFieldPatch {
                optimal_enval: Some(0.25),
                repro_threshold: Some(8.5),
                mutation_rate: Some(0.02),
                maintenance_cost_per_sec: Some(0.11),
                ..GenomeFieldPatch::default()
            }),
            enzymes: vec![
                EnzymePatchOperation {
                    op: "update".to_owned(),
                    index: Some(0),
                    fields: Some(EnzymeFieldPatch {
                        enval_sigma: Some(0.44),
                        enval_throughput: Some(0.22),
                        specificity_mask: Some(0b0000_0111),
                        ..EnzymeFieldPatch::default()
                    }),
                    enzyme: None,
                },
                EnzymePatchOperation {
                    op: "append".to_owned(),
                    index: None,
                    fields: None,
                    enzyme: Some(EnzymeFieldPatch {
                        enzyme_type: Some("attackase".to_owned()),
                        combat_level: Some(123),
                        ..EnzymeFieldPatch::default()
                    }),
                },
            ],
        };

        let result = world.apply_cell_genome_patch(cell_id, &patch).unwrap();
        let cell = &world.cells[cell_id.index()];
        assert_eq!(result.patched_cell_count, 1);
        assert!(result
            .changed_fields
            .iter()
            .any(|field| field == "optimal_enval"));
        assert!((cell.genome.optimal_enval - 0.25).abs() <= 1.0e-6);
        assert_eq!(cell.genome.repro_threshold, 8.5);
        assert_eq!(cell.maintenance_cost_per_sec, 0.11);
        assert_eq!(cell.genome.maintenance_cost_per_sec, 0.11);
        assert_eq!(cell.energy, energy_before);
        assert_eq!(cell.lineage_id, lineage_before);
        assert_eq!(cell.genome.lineage_id, lineage_before);
        assert_eq!(cell.combat_attack_total, 123);
        assert!(cell.genome.enzymes.len() >= 2);
        world.check_invariants().unwrap();
    }

    #[test]
    fn invalid_genome_patch_is_rejected_atomically() {
        let mut world = World::new(only_a_config("genome-patch-invalid", 4, 4)).unwrap();
        let tile = world.tile_id(1, 1).unwrap();
        let genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
        let before = world.cells[cell_id.index()].genome.clone();
        let patch = GenomePatch {
            schema: Some(crate::genome::GENOME_PATCH_SCHEMA.to_owned()),
            genome: Some(GenomeFieldPatch {
                mutation_rate: Some(1.5),
                ..GenomeFieldPatch::default()
            }),
            enzymes: Vec::new(),
        };

        assert!(world.apply_cell_genome_patch(cell_id, &patch).is_err());
        assert_eq!(world.cells[cell_id.index()].genome, before);
        world.check_invariants().unwrap();
    }

    #[test]
    fn enzyme_patch_respects_min_and_max_bounds() {
        let mut world = World::new(only_a_config("genome-patch-bounds", 4, 4)).unwrap();
        let tile = world.tile_id(1, 1).unwrap();
        let mut genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        genome.enzymes = vec![Enzyme::anabolase_abc(&mut world.rng)];
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
        assert_eq!(
            world.cells[cell_id.index()].genome.enzymes.len(),
            MIN_CELL_ENZYMES
        );
        assert!(world
            .apply_cell_genome_patch(cell_id, &remove_patch)
            .is_err());

        while world.cells[cell_id.index()].genome.enzymes.len() < MAX_CELL_ENZYMES {
            world.cells[cell_id.index()]
                .genome
                .enzymes
                .push(Enzyme::defensase(1));
        }
        let append_patch = GenomePatch {
            schema: Some(crate::genome::GENOME_PATCH_SCHEMA.to_owned()),
            genome: None,
            enzymes: vec![EnzymePatchOperation {
                op: "append".to_owned(),
                index: None,
                fields: None,
                enzyme: Some(EnzymeFieldPatch {
                    enzyme_type: Some("defensase".to_owned()),
                    combat_level: Some(1),
                    ..EnzymeFieldPatch::default()
                }),
            }],
        };
        assert!(world
            .apply_cell_genome_patch(cell_id, &append_patch)
            .is_err());
        assert_eq!(
            world.cells[cell_id.index()].genome.enzymes.len(),
            MAX_CELL_ENZYMES
        );
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
                decay_time: Some(3333.0),
                ..GenomeFieldPatch::default()
            }),
            enzymes: Vec::new(),
        };

        let result = world.apply_genome_brush(3, 3, 3, 3, &patch).unwrap();
        assert_eq!(result.visited_tile_count, 9);
        assert_eq!(result.patched_cell_count, 2);
        assert_eq!(world.cells[first.index()].genome.decay_time, 3333.0);
        assert_eq!(world.cells[second.index()].genome.decay_time, 3333.0);
        world.check_invariants().unwrap();
    }

    #[test]
    fn catabolase_basic_reaction_can_harvest_bond_energy() {
        let mut world = World::new(only_a_config("catabolase-world", 4, 4)).unwrap();
        let tile = world.tile_id(1, 1).unwrap();
        let mut genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        genome.optimal_enval = world.tile_enval(tile).unwrap();
        genome.enzymes = vec![Enzyme::catabolase_abc(&mut world.rng)];
        genome.initial_energy = 1.0;
        let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
        world
            .add_tile_molecule(tile, Composition::bc_dimer(), 1.5)
            .unwrap();
        for _ in 0..5 {
            world.step_cell(cell_id);
        }
        assert!(world.cells[cell_id.index()].energy.is_finite());
        world.check_invariants().unwrap();
    }

    #[test]
    fn transmutase_basic_reaction_runs() {
        let mut world = World::new(only_a_config("transmutase-world", 4, 4)).unwrap();
        let tile = world.tile_id(1, 1).unwrap();
        let mut genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        genome.optimal_enval = world.tile_enval(tile).unwrap();
        genome.enzymes = vec![Enzyme::random(EnzymeType::Transmutase, &mut world.rng)];
        genome.enzymes[0].specificity_mask = crate::chem::ALL_ELEMENT_MASK;
        genome.initial_energy = 10.0;
        let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
        world
            .add_tile_molecule(tile, Composition::single(Element::F), 1.0)
            .unwrap();
        for _ in 0..10 {
            world.step_cell(cell_id);
        }
        assert!(world.cells[cell_id.index()].energy.is_finite());
        world.check_invariants().unwrap();
    }

    #[test]
    fn enval_coupling_changes_field() {
        let mut world = World::new(only_a_config("enval-coupling", 4, 4)).unwrap();
        let before = world.average_enval();
        let tile = world.tile_id(1, 1).unwrap();
        let mut genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        genome.optimal_enval = world.tile_enval(tile).unwrap();
        genome.enzymes = vec![Enzyme::anabolase_abc(&mut world.rng)];
        genome.initial_energy = 10.0;
        let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
        world
            .add_tile_molecule(tile, Composition::single(Element::B), 1.0)
            .unwrap();
        for _ in 0..5 {
            world.step_cell(cell_id);
        }
        assert_ne!(world.average_enval().to_bits(), before.to_bits());
        world.check_invariants().unwrap();
    }

    #[test]
    fn division_partitions_energy_and_ownership() {
        let mut world = World::new(only_a_config("division", 6, 6)).unwrap();
        let tile = world.tile_id(3, 3).unwrap();
        let mut genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        genome.repro_threshold = 1.0;
        genome.initial_energy = 10.0;
        let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
        let molecule = Molecule::new(Composition::single(Element::B), 1.0).unwrap();
        world.add_cell_molecule_record(cell_id, molecule).unwrap();
        world.divide_cell(cell_id, tile, world.average_enval());
        assert!(world.stats().live_cell_count >= 2);
        assert!(world.cells[cell_id.index()].energy < 10.0);
        world.check_invariants().unwrap();
    }

    #[test]
    fn predation_without_attack_does_nothing() {
        let mut world = World::new(only_a_config("predation-no-attack", 4, 4)).unwrap();
        let a_tile = world.tile_id(0, 0).unwrap();
        let b_tile = world.tile_id(1, 0).unwrap();
        let a = spawn_combat_cell(&mut world, a_tile, 1, 0, 0, 1.0);
        let b = spawn_combat_cell(&mut world, b_tile, 2, 0, 0, 1.0);
        world.resolve_predation();
        assert_eq!(world.cells[a.index()].state, CellState::Active);
        assert_eq!(world.cells[b.index()].state, CellState::Active);
        assert_eq!(world.stats().predation_events, 0);
    }

    #[test]
    fn predation_requires_attack_to_exceed_defense() {
        let mut world = World::new(only_a_config("predation-threshold", 4, 4)).unwrap();
        let a_tile = world.tile_id(0, 0).unwrap();
        let b_tile = world.tile_id(1, 0).unwrap();
        let a = spawn_combat_cell(&mut world, a_tile, 1, 5, 0, 1.0);
        let b = spawn_combat_cell(&mut world, b_tile, 2, 0, 5, 1.0);
        world.resolve_predation();
        assert_eq!(world.cells[a.index()].state, CellState::Active);
        assert_eq!(world.cells[b.index()].state, CellState::Active);
        assert_eq!(world.stats().predation_events, 0);
    }

    #[test]
    fn one_sided_predation_wins_and_assimilates_energy_and_molecules() {
        let mut world = World::new(only_a_config("predation-assimilate", 4, 4)).unwrap();
        let predator_tile = world.tile_id(0, 0).unwrap();
        let prey_tile = world.tile_id(1, 0).unwrap();
        let predator = spawn_combat_cell(&mut world, predator_tile, 1, 20, 0, 1.0);
        let prey = spawn_combat_cell(&mut world, prey_tile, 2, 0, 1, 3.0);
        world
            .add_cell_molecule_record(
                prey,
                Molecule::new(Composition::single(Element::B), 1.0).unwrap(),
            )
            .unwrap();
        world.resolve_predation();

        assert_eq!(world.cells[predator.index()].state, CellState::Active);
        assert_eq!(world.cells[prey.index()].state, CellState::Dead);
        assert!(world.cells[predator.index()].energy >= 4.0);
        assert!(world.cells[predator.index()].internal_atom_count >= 1);
        assert_eq!(world.stats().predation_events, 1);
        assert_eq!(world.stats().cells_consumed, 1);
        assert_eq!(world.stats().deaths, 1);
        world.check_invariants().unwrap();
    }

    #[test]
    fn stats_count_only_extant_lineages_after_predation() {
        let mut world = World::new(only_a_config("lineage-extant-count", 4, 4)).unwrap();
        let predator_tile = world.tile_id(0, 0).unwrap();
        let prey_tile = world.tile_id(1, 0).unwrap();
        let predator = spawn_combat_cell(&mut world, predator_tile, 1, 20, 0, 1.0);
        let prey = spawn_combat_cell(&mut world, prey_tile, 2, 0, 1, 1.0);

        assert_eq!(world.stats().lineage_count, 2);
        assert_eq!(world.lineage_counters().len(), 2);

        world.resolve_predation();

        assert_eq!(world.cells[predator.index()].state, CellState::Active);
        assert_eq!(world.cells[prey.index()].state, CellState::Dead);
        assert_eq!(world.stats().lineage_count, 1);
        assert_eq!(
            world
                .lineage_counters()
                .get(&LineageId(2))
                .unwrap()
                .population,
            0
        );
        assert!(!world
            .top_lineages(10)
            .iter()
            .any(|(lineage, _)| *lineage == LineageId(2)));
        world.check_invariants().unwrap();
    }

    #[test]
    fn mutual_predation_larger_margin_wins_and_equal_margin_does_not() {
        let mut world = World::new(only_a_config("predation-margin", 4, 4)).unwrap();
        let a_tile = world.tile_id(0, 0).unwrap();
        let b_tile = world.tile_id(1, 0).unwrap();
        let a = spawn_combat_cell(&mut world, a_tile, 1, 30, 20, 1.0);
        let b = spawn_combat_cell(&mut world, b_tile, 2, 25, 5, 1.0);
        world.resolve_predation();
        assert_eq!(world.cells[a.index()].state, CellState::Active);
        assert_eq!(world.cells[b.index()].state, CellState::Dead);

        let mut world = World::new(only_a_config("predation-margin-equal", 4, 4)).unwrap();
        let a_tile = world.tile_id(0, 0).unwrap();
        let b_tile = world.tile_id(1, 0).unwrap();
        let a = spawn_combat_cell(&mut world, a_tile, 1, 20, 5, 1.0);
        let b = spawn_combat_cell(&mut world, b_tile, 2, 20, 5, 1.0);
        world.resolve_predation();
        assert_eq!(world.cells[a.index()].state, CellState::Active);
        assert_eq!(world.cells[b.index()].state, CellState::Active);
        assert_eq!(world.stats().predation_events, 0);
    }

    #[test]
    fn same_lineage_and_non_neighbors_do_not_predate() {
        let mut world = World::new(only_a_config("predation-same-lineage", 5, 5)).unwrap();
        let a_tile = world.tile_id(0, 0).unwrap();
        let b_tile = world.tile_id(1, 0).unwrap();
        let a = spawn_combat_cell(&mut world, a_tile, 1, 50, 0, 1.0);
        let b = spawn_combat_cell(&mut world, b_tile, 1, 0, 1, 1.0);
        world.resolve_predation();
        assert_eq!(world.cells[a.index()].state, CellState::Active);
        assert_eq!(world.cells[b.index()].state, CellState::Active);

        let mut world = World::new(only_a_config("predation-nonneighbor", 5, 5)).unwrap();
        let a_tile = world.tile_id(0, 0).unwrap();
        let b_tile = world.tile_id(2, 2).unwrap();
        let a = spawn_combat_cell(&mut world, a_tile, 1, 50, 0, 1.0);
        let b = spawn_combat_cell(&mut world, b_tile, 2, 0, 1, 1.0);
        world.resolve_predation();
        assert_eq!(world.cells[a.index()].state, CellState::Active);
        assert_eq!(world.cells[b.index()].state, CellState::Active);
    }

    #[test]
    fn diagonal_moore_neighbor_predation_works() {
        let mut world = World::new(only_a_config("predation-diagonal", 4, 4)).unwrap();
        let a_tile = world.tile_id(0, 0).unwrap();
        let b_tile = world.tile_id(1, 1).unwrap();
        let predator = spawn_combat_cell(&mut world, a_tile, 1, 50, 0, 1.0);
        let prey = spawn_combat_cell(&mut world, b_tile, 2, 0, 1, 1.0);
        world.resolve_predation();
        assert_eq!(world.cells[predator.index()].state, CellState::Active);
        assert_eq!(world.cells[prey.index()].state, CellState::Dead);
        world.check_invariants().unwrap();
    }

    #[test]
    fn occupied_neighbor_predation_checks_only_occupied_pairs_in_sparse_world() {
        let mut world = World::new(only_a_config("predation-occupied-scan", 10, 10)).unwrap();
        let predator_tile = world.tile_id(0, 0).unwrap();
        let prey_tile = world.tile_id(1, 0).unwrap();
        let predator = spawn_combat_cell(&mut world, predator_tile, 1, 50, 0, 1.0);
        let prey = spawn_combat_cell(&mut world, prey_tile, 2, 0, 1, 1.0);

        world.resolve_predation();
        let counters = world.operation_counters();
        assert_eq!(world.cells[predator.index()].state, CellState::Active);
        assert_eq!(world.cells[prey.index()].state, CellState::Dead);
        assert_eq!(counters.predation_occupied_tiles_considered, 2);
        assert_eq!(counters.predation_candidate_neighbor_pairs, 8);
        assert_eq!(counters.predation_pairs_checked, 1);
        assert_eq!(counters.predation_cross_lineage_pairs, 1);
        assert_eq!(counters.predation_events, 1);
        world.check_invariants().unwrap();
    }

    #[test]
    fn same_lineage_predation_pair_is_checked_but_not_cross_lineage() {
        let mut world = World::new(only_a_config("predation-same-lineage-counters", 5, 5)).unwrap();
        let a_tile = world.tile_id(0, 0).unwrap();
        let b_tile = world.tile_id(1, 0).unwrap();
        let a = spawn_combat_cell(&mut world, a_tile, 1, 50, 0, 1.0);
        let b = spawn_combat_cell(&mut world, b_tile, 1, 0, 1, 1.0);

        world.resolve_predation();
        let counters = world.operation_counters();
        assert_eq!(world.cells[a.index()].state, CellState::Active);
        assert_eq!(world.cells[b.index()].state, CellState::Active);
        assert_eq!(counters.predation_pairs_checked, 1);
        assert_eq!(counters.predation_cross_lineage_pairs, 0);
        assert_eq!(counters.predation_events, 0);
        world.check_invariants().unwrap();
    }

    #[test]
    fn tiny_toroidal_world_predation_does_not_double_count_wrapped_pairs() {
        let mut world = World::new(only_a_config("predation-tiny-dedupe", 1, 2)).unwrap();
        let predator_tile = world.tile_id(0, 0).unwrap();
        let prey_tile = world.tile_id(0, 1).unwrap();
        let predator = spawn_combat_cell(&mut world, predator_tile, 1, 50, 0, 1.0);
        let prey = spawn_combat_cell(&mut world, prey_tile, 2, 0, 1, 1.0);

        world.resolve_predation();
        let counters = world.operation_counters();
        assert_eq!(world.cells[predator.index()].state, CellState::Active);
        assert_eq!(world.cells[prey.index()].state, CellState::Dead);
        assert_eq!(counters.predation_candidate_neighbor_pairs, 1);
        assert_eq!(counters.predation_pairs_checked, 1);
        assert_eq!(counters.predation_cross_lineage_pairs, 1);
        assert_eq!(counters.predation_events, 1);
        world.check_invariants().unwrap();
    }

    #[test]
    fn transferred_attackase_refreshes_combat_totals() {
        let mut world = World::new(only_a_config("predation-transfer-combat", 4, 4)).unwrap();
        let predator_tile = world.tile_id(0, 0).unwrap();
        let prey_tile = world.tile_id(1, 0).unwrap();
        let predator = spawn_combat_cell(&mut world, predator_tile, 1, 200, 200, 1.0);
        let mut prey_genome = combat_genome(&mut world, 2, 0, 1, 1.0);
        prey_genome.enzymes.push(Enzyme::attackase(99));
        let prey = world
            .spawn_cell_with_genome_at(prey_tile, prey_genome)
            .unwrap();
        world.resolve_predation();
        assert_eq!(world.cells[prey.index()].state, CellState::Dead);
        assert_eq!(
            world.cells[predator.index()].combat_attack_total,
            world.cells[predator.index()].genome.attack_total()
        );
        world.check_invariants().unwrap();
    }

    #[test]
    fn deterministic_predation_runs_match() {
        fn run_once() -> crate::stats::WorldStats {
            let mut world = World::new(small_config("predation-golden", 16, 12)).unwrap();
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
        assert_eq!(a.molecule_count, b.molecule_count);
        assert_eq!(a.total_atom_count, b.total_atom_count);
        assert_eq!(a.element_counts, b.element_counts);
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
        assert!(stats.molecule_count >= stats.tile_count);
        assert!(stats.total_atom_count >= stats.tile_count as u64);
        assert!(stats.average_enval.is_finite());
        world.check_invariants().unwrap();
    }

    #[test]
    fn render_buffers_have_expected_lengths_and_values() {
        let mut world = World::new(small_config("render-buffers", 8, 6)).unwrap();
        world.spawn_founder_cells(5).unwrap();
        let before = world.build_render_buffers();
        assert_eq!(before.tile_enval.len(), 48);
        assert_eq!(before.tile_occupancy.len(), 48);
        assert_eq!(before.tile_mass.len(), 48);
        assert_eq!(before.tile_molecule_count.len(), 48);
        assert_eq!(before.tile_element_mask.len(), 48);
        assert_eq!(before.cell_count(), world.stats().live_cell_count);
        assert!(before.tile_enval.iter().all(|value| value.is_finite()));
        assert!(before.cell_energy.iter().all(|value| value.is_finite()));
        assert!(before
            .tile_occupancy
            .iter()
            .any(|value| *value != crate::EMPTY_CELL_ID));

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
        assert_eq!(tile_info.cell, Some(cell_id));
        assert!(tile_info.enval.is_finite());

        let cell_info = world.inspect_cell(cell_id).unwrap();
        assert_eq!(cell_info.cell_id, cell_id);
        assert_eq!(cell_info.tile_id, tile);
        assert_eq!(cell_info.x, 2);
        assert_eq!(cell_info.y, 1);
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
        let reserved_tile_capacity = actual.tile_enval.capacity();
        world.write_render_buffers(&mut actual);

        assert_eq!(actual, expected);
        assert!(actual.tile_enval.capacity() >= reserved_tile_capacity);
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
        assert_eq!(
            stats.tile_molecule_count + stats.cell_molecule_count,
            stats.molecule_count
        );
        assert_eq!(
            stats.molecule_count + stats.free_molecule_record_count,
            world.molecules.len()
        );
        assert_eq!(stats.active_molecule_record_count, stats.molecule_count);
        assert_eq!(stats.molecule_arena_len, world.molecules.len());
        assert!(stats.molecule_arena_high_water_mark >= stats.molecule_arena_len);
        assert_eq!(
            stats.tile_atom_count + stats.cell_atom_count,
            stats.total_atom_count
        );
        assert_eq!(stats.cell_record_count, world.cells.len());
        assert_eq!(
            stats.dead_cell_count,
            world
                .cells
                .iter()
                .filter(|cell| cell.state == CellState::Dead)
                .count()
        );
        assert_eq!(
            stats.enzyme_count_histogram.iter().sum::<u64>(),
            stats.live_cell_count as u64
        );
        assert_eq!(
            stats.enzyme_type_totals.total(),
            world
                .active_cells
                .iter()
                .map(|cell_id| world.cells[cell_id.index()].genome.enzymes.len() as u64)
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
    fn substrate_sampling_stack_capacity_covers_all_metabolic_enzyme_classes() {
        let mut rng = crate::rng::Rng::from_seed_str("substrate-capacity");
        for enzyme_type in [
            EnzymeType::Anabolase,
            EnzymeType::Catabolase,
            EnzymeType::Transmutase,
        ] {
            let enzyme = Enzyme::random(enzyme_type, &mut rng);
            assert!(crate::bio::max_inputs(&enzyme) <= super::MAX_REACTION_SUBSTRATES);
        }
    }

    #[test]
    fn substrate_sampling_prefilter_skips_impossible_pools_without_scanning() {
        let mut world = World::new(only_a_config("substrate-prefilter-skip", 4, 4)).unwrap();
        let tile_id = world.tile_id(0, 0).unwrap();
        let avg_enval = world.avg_enval;
        let genome = Genome::random_founder(&mut world.rng, avg_enval);
        let cell_id = world.spawn_cell_with_genome_at(tile_id, genome).unwrap();
        let mut enzyme = Enzyme::anabolase_abc(&mut world.rng);
        enzyme.specificity_mask = Element::D.mask();

        let before = world.operation_counters.substrate_candidates_scanned;
        let substrates = world.sample_accepted_substrates(cell_id, tile_id, &enzyme, 3);

        assert!(substrates.is_empty());
        assert_eq!(
            world.operation_counters.substrate_candidates_scanned,
            before
        );
    }

    #[test]
    fn impossible_substrate_pool_skips_reaction_gate_without_scanning() {
        let mut world = World::new(only_a_config("substrate-pregate-skip", 4, 4)).unwrap();
        let tile_id = world.tile_id(0, 0).unwrap();
        let mut genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        let mut enzyme = Enzyme::anabolase_abc(&mut world.rng);
        enzyme.specificity_mask = Element::D.mask();
        genome.enzymes = vec![enzyme];
        genome.initial_energy = 10.0;
        genome.repro_threshold = 1_000_000.0;
        genome.decay_time = 1_000_000.0;
        genome.desired_element_reserve = 0;
        genome.maintenance_cost_per_sec = 0.0;
        let cell_id = world.spawn_cell_with_genome_at(tile_id, genome).unwrap();

        let attempts_before = world.reaction_counters.total_attempts();
        let gates_before = world.operation_counters.reaction_gates_passed;
        let scans_before = world.operation_counters.substrate_candidates_scanned;

        world.step_cell(cell_id);

        assert_eq!(
            world.reaction_counters.total_attempts(),
            attempts_before + 1
        );
        assert_eq!(world.operation_counters.reaction_gates_passed, gates_before);
        assert_eq!(
            world.operation_counters.substrate_candidates_scanned,
            scans_before
        );
        world.check_invariants().unwrap();
    }

    #[test]
    fn substrate_sampling_still_scans_and_samples_possible_pools() {
        let mut world = World::new(only_a_config("substrate-prefilter-hit", 4, 4)).unwrap();
        let tile_id = world.tile_id(0, 0).unwrap();
        let avg_enval = world.avg_enval;
        let genome = Genome::random_founder(&mut world.rng, avg_enval);
        let cell_id = world.spawn_cell_with_genome_at(tile_id, genome).unwrap();
        let mut enzyme = Enzyme::random(EnzymeType::Transmutase, &mut world.rng);
        enzyme.specificity_mask = Element::A.mask();

        let before = world.operation_counters.substrate_candidates_scanned;
        let substrates = world.sample_accepted_substrates(cell_id, tile_id, &enzyme, 1);

        assert_eq!(substrates.len(), 1);
        assert!(world.operation_counters.substrate_candidates_scanned > before);
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
        assert_eq!(compact.molecule_count, full.molecule_count);
        assert_eq!(compact.total_atom_count, full.total_atom_count);
        assert_eq!(compact.enval_p05.to_bits(), 0.0_f32.to_bits());
        assert_eq!(compact.enval_p50.to_bits(), 0.0_f32.to_bits());
        assert_eq!(compact.enval_p95.to_bits(), 0.0_f32.to_bits());
        assert_eq!(full.enval_p50.to_bits(), world.stats().enval_p50.to_bits());
        world.check_invariants().unwrap();
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
}
