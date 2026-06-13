use serde::{Deserialize, Serialize};

use crate::chem::{Composition, ELEMENT_COUNT, ELEMENT_ORDER};
use crate::genome::{Genome, LineageId};
use crate::world::{MoleculeId, TileId};

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

pub const CELL_REACTION_LOG_CAPACITY: usize = 64;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ReactionMoleculeSummary {
    pub composition_counts: [u16; ELEMENT_COUNT],
    pub formula: String,
    pub size: u16,
    pub element_mask: u8,
    pub bond_multiplier: f32,
    pub elemental_energy_sum: f32,
    pub energy: f32,
    pub polarity: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ReactionRecord {
    pub tick_count: u64,
    pub sim_time_seconds: f64,
    pub cell_id: usize,
    pub tile_id: usize,
    pub x: usize,
    pub y: usize,
    pub enzyme_index: usize,
    pub enzyme_type: String,
    pub status: String,
    pub substrate_count: usize,
    pub substrates: Vec<ReactionMoleculeSummary>,
    pub produced: Option<ReactionMoleculeSummary>,
    pub byproducts: Vec<ReactionMoleculeSummary>,
    pub energy_before: f64,
    pub energy_after: f64,
    pub delta_cell_energy: f64,
    pub chemical_delta: f64,
    pub enval_energy: f64,
    pub enval_input: f32,
    pub enval_output: f32,
    pub delta_enval: f32,
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
    pub molecules: Vec<MoleculeId>,
    pub internal_element_counts: [u32; ELEMENT_COUNT],
    pub internal_atom_count: u32,
    pub time_without_food: f64,
    pub birth_sim_time: f64,
    pub death_sim_time: Option<f64>,
    pub maintenance_cost_per_sec: f64,
    pub combat_attack_total: u32,
    pub combat_defense_total: u32,
    pub active_slot: Option<usize>,
    #[serde(default)]
    pub recent_reactions: Vec<ReactionRecord>,
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
            molecules: Vec::new(),
            internal_element_counts: [0; ELEMENT_COUNT],
            internal_atom_count: 0,
            time_without_food: 0.0,
            birth_sim_time,
            death_sim_time: None,
            maintenance_cost_per_sec,
            combat_attack_total,
            combat_defense_total,
            active_slot: None,
            recent_reactions: Vec::new(),
        }
    }

    pub fn push_reaction_record(&mut self, record: ReactionRecord) {
        self.recent_reactions.push(record);
        if self.recent_reactions.len() > CELL_REACTION_LOG_CAPACITY {
            let overflow = self.recent_reactions.len() - CELL_REACTION_LOG_CAPACITY;
            self.recent_reactions.drain(0..overflow);
        }
    }

    pub fn is_alive(&self) -> bool {
        self.state == CellState::Active
    }

    pub fn refresh_combat_totals(&mut self) {
        self.combat_attack_total = self.genome.attack_total();
        self.combat_defense_total = self.genome.defense_total();
    }

    pub fn can_hold_molecule(&self, molecule_id: MoleculeId) -> bool {
        !self.molecules.contains(&molecule_id)
    }

    pub fn apply_internal_composition_delta(
        &mut self,
        composition: Composition,
        sign: i32,
    ) -> Result<(), CellCountError> {
        for element in ELEMENT_ORDER {
            let count = u32::from(composition.count(element));
            if sign >= 0 {
                self.internal_element_counts[element.index()] = self.internal_element_counts
                    [element.index()]
                .checked_add(count)
                .ok_or(CellCountError::Overflow)?;
            } else {
                self.internal_element_counts[element.index()] = self.internal_element_counts
                    [element.index()]
                .checked_sub(count)
                .ok_or(CellCountError::Underflow)?;
            }
        }
        let atoms = u32::from(composition.size());
        if sign >= 0 {
            self.internal_atom_count = self
                .internal_atom_count
                .checked_add(atoms)
                .ok_or(CellCountError::Overflow)?;
        } else {
            self.internal_atom_count = self
                .internal_atom_count
                .checked_sub(atoms)
                .ok_or(CellCountError::Underflow)?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CellCountError {
    Overflow,
    Underflow,
}
