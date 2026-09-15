use serde::{Deserialize, Serialize};

use crate::chem::{ELEMENT_COUNT, ElementAmounts};
use crate::genome::{Genome, LineageId};
use crate::world::TileId;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CellId(pub usize);

impl CellId {
    pub const fn index(self) -> usize {
        self.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CellState {
    Active,
    Dead,
}

pub const CELL_FLUX_LOG_CAPACITY: usize = 64;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FluxRecord {
    pub tick_count: u64,
    pub sim_time_seconds: f64,
    pub cell_id: usize,
    pub tile_id: usize,
    pub x: usize,
    pub y: usize,
    pub catalyst_index: usize,
    pub catalyst_type: String,
    pub reactants: [f32; ELEMENT_COUNT],
    pub products: [f32; ELEMENT_COUNT],
    pub requested_extent: f32,
    pub executed_extent: f32,
    pub element_deltas: [f32; ELEMENT_COUNT],
    pub secreted_elements: [f32; ELEMENT_COUNT],
    pub energy_before: f64,
    pub energy_after: f64,
    pub delta_cell_energy: f64,
    pub raw_chemical_energy: f64,
    pub enval_energy: f64,
    pub enval_input: f32,
    pub enval_output: f32,
    pub local_enval: f32,
    pub optimal_enval: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Cell {
    pub state: CellState,
    pub tile_id: Option<TileId>,
    pub energy: f64,
    pub genome: Genome,
    pub lineage_id: LineageId,
    pub internal_elements: ElementAmounts,
    pub time_without_food: f64,
    pub birth_sim_time: f64,
    pub death_sim_time: Option<f64>,
    pub maintenance_cost_per_sec: f64,
    pub combat_attack_total: u32,
    pub combat_defense_total: u32,
    pub active_slot: Option<usize>,
    #[serde(default)]
    pub recent_fluxes: Vec<FluxRecord>,
}

impl Cell {
    pub fn new(genome: Genome, tile_id: TileId, birth_sim_time: f64) -> Self {
        let energy = genome.initial_energy;
        let lineage_id = genome.lineage_id;
        let maintenance_cost_per_sec = genome.maintenance_cost_per_sec;
        let combat_attack_total = genome.attack_total();
        let combat_defense_total = genome.defense_total();
        Self {
            state: CellState::Active,
            tile_id: Some(tile_id),
            energy,
            genome,
            lineage_id,
            internal_elements: ElementAmounts::ZERO,
            time_without_food: 0.0,
            birth_sim_time,
            death_sim_time: None,
            maintenance_cost_per_sec,
            combat_attack_total,
            combat_defense_total,
            active_slot: None,
            recent_fluxes: Vec::new(),
        }
    }

    pub fn push_flux_record(&mut self, record: FluxRecord) {
        self.recent_fluxes.push(record);
        if self.recent_fluxes.len() > CELL_FLUX_LOG_CAPACITY {
            let overflow = self.recent_fluxes.len() - CELL_FLUX_LOG_CAPACITY;
            self.recent_fluxes.drain(0..overflow);
        }
    }

    pub fn is_alive(&self) -> bool {
        self.state == CellState::Active
    }

    pub fn refresh_combat_totals(&mut self) {
        self.combat_attack_total = self.genome.attack_total();
        self.combat_defense_total = self.genome.defense_total();
    }
}
