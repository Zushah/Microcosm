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
