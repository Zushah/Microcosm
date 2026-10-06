use std::ops::{Index, IndexMut};

use serde::{Deserialize, Serialize};

use crate::chem::{ELEMENT_COUNT, ElementAmounts};
use crate::genome::{EnzymeType, Genome, LineageId};
use crate::spatial::{DEFAULT_CELL_RADIUS, Position};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CellId(pub usize);

impl CellId {
    pub const fn index(self) -> usize {
        self.0
    }
}

pub const CELL_FLUX_LOG_CAPACITY: usize = 64;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FluxRecord {
    pub tick_count: u64,
    pub sim_time_seconds: f64,
    pub cell_id: usize,
    pub x: f32,
    pub y: f32,
    pub catalyst_index: usize,
    #[serde(with = "enzyme_type_text")]
    pub catalyst_type: EnzymeType,
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

mod enzyme_type_text {
    use serde::{Deserialize, Deserializer, Serializer};

    use crate::genome::EnzymeType;

    pub fn serialize<S: Serializer>(value: &EnzymeType, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(value.as_str())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<EnzymeType, D::Error> {
        let text = String::deserialize(deserializer)?;
        text.parse().map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Cell {
    pub id: CellId,
    pub position: Position,
    pub radius: f32,
    pub energy: f64,
    pub genome: Genome,
    pub lineage_id: LineageId,
    pub internal_elements: ElementAmounts,
    pub birth_sim_time: f64,
    pub maintenance_cost_per_sec: f64,
    pub combat_attack_total: u32,
    pub combat_defense_total: u32,
    recent_fluxes: Vec<FluxRecord>,
    flux_head: usize,
}

impl Cell {
    pub fn new(genome: Genome, position: Position, birth_sim_time: f64) -> Self {
        let energy = genome.initial_energy;
        let lineage_id = genome.lineage_id;
        let maintenance_cost_per_sec = genome.maintenance_cost_per_sec;
        let combat_attack_total = genome.attack_total();
        let combat_defense_total = genome.defense_total();
        Self {
            id: CellId(usize::MAX),
            position,
            radius: DEFAULT_CELL_RADIUS,
            energy,
            genome,
            lineage_id,
            internal_elements: ElementAmounts::ZERO,
            birth_sim_time,
            maintenance_cost_per_sec,
            combat_attack_total,
            combat_defense_total,
            recent_fluxes: Vec::new(),
            flux_head: 0,
        }
    }

    pub fn push_flux_record(&mut self, record: FluxRecord) {
        if self.recent_fluxes.len() < CELL_FLUX_LOG_CAPACITY {
            if self.recent_fluxes.capacity() == 0 {
                self.recent_fluxes.reserve_exact(CELL_FLUX_LOG_CAPACITY);
            }
            self.recent_fluxes.push(record);
        } else {
            self.recent_fluxes[self.flux_head] = record;
            self.flux_head = (self.flux_head + 1) % CELL_FLUX_LOG_CAPACITY;
        }
    }

    pub fn flux_count(&self) -> usize {
        self.recent_fluxes.len()
    }

    pub fn fluxes_newest_first(&self) -> impl Iterator<Item = &FluxRecord> + '_ {
        let (older, newer) = self.recent_fluxes.split_at(self.flux_head);
        older.iter().rev().chain(newer.iter().rev())
    }

    pub fn latest_flux(&self) -> Option<&FluxRecord> {
        self.fluxes_newest_first().next()
    }

    pub fn refresh_combat_totals(&mut self) {
        self.combat_attack_total = self.genome.attack_total();
        self.combat_defense_total = self.genome.defense_total();
    }
}

const NO_SLOT: u32 = u32::MAX;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CellStore {
    cells: Vec<Cell>,
    next_id: usize,
    #[serde(skip, default)]
    slots: Vec<u32>,
}

impl CellStore {
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    pub fn issued_ids(&self) -> usize {
        self.next_id
    }

    pub fn slot(&self, id: CellId) -> Option<usize> {
        match self.slots.get(id.index()) {
            Some(slot) if *slot != NO_SLOT => Some(*slot as usize),
            _ => None,
        }
    }

    pub fn contains(&self, id: CellId) -> bool {
        self.slot(id).is_some()
    }

    pub fn get(&self, id: CellId) -> Option<&Cell> {
        self.slot(id).map(|slot| &self.cells[slot])
    }

    pub fn get_mut(&mut self, id: CellId) -> Option<&mut Cell> {
        self.slot(id).map(|slot| &mut self.cells[slot])
    }

    pub fn at_slot(&self, slot: usize) -> &Cell {
        &self.cells[slot]
    }

    pub fn at_slot_mut(&mut self, slot: usize) -> &mut Cell {
        &mut self.cells[slot]
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Cell> {
        self.cells.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, Cell> {
        self.cells.iter_mut()
    }

    pub fn ids(&self) -> impl Iterator<Item = CellId> + '_ {
        self.cells.iter().map(|cell| cell.id)
    }

    pub fn insert(&mut self, mut cell: Cell) -> CellId {
        let id = CellId(self.next_id);
        self.next_id += 1;
        cell.id = id;
        let slot = u32::try_from(self.cells.len()).expect("live cell count fits in u32");
        if self.slots.len() <= id.index() {
            self.slots.resize(id.index() + 1, NO_SLOT);
        }
        self.slots[id.index()] = slot;
        self.cells.push(cell);
        id
    }

    pub fn remove(&mut self, id: CellId) -> Option<Cell> {
        let slot = self.slot(id)?;
        self.slots[id.index()] = NO_SLOT;
        let removed = self.cells.swap_remove(slot);
        if let Some(moved) = self.cells.get(slot) {
            self.slots[moved.id.index()] = slot as u32;
        }
        Some(removed)
    }

    pub fn rebuild_index(&mut self) {
        self.slots.clear();
        self.slots.resize(self.next_id, NO_SLOT);
        for (slot, cell) in self.cells.iter().enumerate() {
            if let Some(entry) = self.slots.get_mut(cell.id.index()) {
                *entry = slot as u32;
            }
        }
    }

    pub fn is_consistent(&self) -> bool {
        if self.slots.len() > self.next_id {
            return false;
        }
        let indexed = self.slots.iter().filter(|slot| **slot != NO_SLOT).count();
        indexed == self.cells.len()
            && self.cells.iter().enumerate().all(|(slot, cell)| {
                cell.id.index() < self.next_id && self.slot(cell.id) == Some(slot)
            })
    }
}

impl Index<CellId> for CellStore {
    type Output = Cell;

    fn index(&self, id: CellId) -> &Cell {
        let slot = self.slot(id).expect("cell id must refer to a live cell");
        &self.cells[slot]
    }
}

impl IndexMut<CellId> for CellStore {
    fn index_mut(&mut self, id: CellId) -> &mut Cell {
        let slot = self.slot(id).expect("cell id must refer to a live cell");
        &mut self.cells[slot]
    }
}

#[cfg(test)]
mod tests {
    use super::{CELL_FLUX_LOG_CAPACITY, Cell, CellId, CellStore, FluxRecord};
    use crate::genome::{EnzymeType, Genome};
    use crate::rng::Rng;
    use crate::spatial::Position;

    fn cell(seed: &str) -> Cell {
        let mut rng = Rng::from_seed_str(seed);
        Cell::new(
            Genome::random_founder(&mut rng, 0.0),
            Position::new(1.0, 1.0),
            0.0,
        )
    }

    fn record(tick: u64) -> FluxRecord {
        FluxRecord {
            tick_count: tick,
            sim_time_seconds: 0.0,
            cell_id: 0,
            x: 0.0,
            y: 0.0,
            catalyst_index: 0,
            catalyst_type: EnzymeType::Metabolic,
            reactants: [0.0; 6],
            products: [0.0; 6],
            requested_extent: 0.0,
            executed_extent: 0.0,
            element_deltas: [0.0; 6],
            secreted_elements: [0.0; 6],
            energy_before: 0.0,
            energy_after: 0.0,
            delta_cell_energy: 0.0,
            raw_chemical_energy: 0.0,
            enval_energy: 0.0,
            enval_input: 0.0,
            enval_output: 0.0,
            local_enval: 0.0,
            optimal_enval: 0.0,
        }
    }

    #[test]
    fn flux_ring_buffer_keeps_the_newest_records_in_order() {
        let mut cell = cell("flux-ring");
        assert!(cell.latest_flux().is_none());
        for tick in 0..10 {
            cell.push_flux_record(record(tick));
        }
        let ticks = cell
            .fluxes_newest_first()
            .map(|record| record.tick_count)
            .collect::<Vec<_>>();
        assert_eq!(ticks, (0..10).rev().collect::<Vec<_>>());

        let total = CELL_FLUX_LOG_CAPACITY as u64 * 2 + 7;
        for tick in 10..total {
            cell.push_flux_record(record(tick));
        }
        assert_eq!(cell.flux_count(), CELL_FLUX_LOG_CAPACITY);
        let ticks = cell
            .fluxes_newest_first()
            .map(|record| record.tick_count)
            .collect::<Vec<_>>();
        let expected = (total - CELL_FLUX_LOG_CAPACITY as u64..total)
            .rev()
            .collect::<Vec<_>>();
        assert_eq!(ticks, expected);
        assert_eq!(cell.latest_flux().unwrap().tick_count, total - 1);
    }

    #[test]
    fn store_ids_are_stable_never_reused_and_dead_ids_resolve_to_none() {
        let mut store = CellStore::default();
        let ids = (0..5)
            .map(|index| store.insert(cell(&format!("store-{index}"))))
            .collect::<Vec<_>>();
        assert_eq!(ids, (0..5).map(CellId).collect::<Vec<_>>());

        let removed = store.remove(ids[1]).unwrap();
        assert_eq!(removed.id, ids[1]);
        assert!(store.get(ids[1]).is_none());
        assert!(store.remove(ids[1]).is_none());
        assert_eq!(
            store.ids().collect::<Vec<_>>(),
            vec![ids[0], ids[4], ids[2], ids[3]]
        );
        for id in [ids[0], ids[2], ids[3], ids[4]] {
            assert_eq!(store[id].id, id);
        }

        let fresh = store.insert(cell("store-fresh"));
        assert_eq!(fresh, CellId(5));
        assert!(store.is_consistent());

        let mut reloaded = store.clone();
        reloaded.slots.clear();
        assert!(reloaded.get(ids[0]).is_none());
        reloaded.rebuild_index();
        assert!(reloaded.is_consistent());
        assert_eq!(
            reloaded.ids().collect::<Vec<_>>(),
            store.ids().collect::<Vec<_>>()
        );
        assert!(reloaded.get(ids[1]).is_none());
    }
}
