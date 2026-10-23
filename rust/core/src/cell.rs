use std::fmt;
use std::ops::{Index, IndexMut};

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::chem::{ELEMENT_COUNT, ElementAmounts};
use crate::genome::{Genome, LineageId};
use crate::spatial::Position;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CellId(pub usize);

impl CellId {
    pub const fn index(self) -> usize {
        self.0
    }
}

pub const CELL_FLUX_LOG_CAPACITY: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReactionSite {
    Internal,
    Field,
    Cell(CellId),
}

impl fmt::Display for ReactionSite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Internal => f.write_str("internal"),
            Self::Field => f.write_str("field"),
            Self::Cell(cell_id) => write!(f, "cell:{}", cell_id.index()),
        }
    }
}

impl std::str::FromStr for ReactionSite {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "internal" => Ok(Self::Internal),
            "field" => Ok(Self::Field),
            other => other
                .strip_prefix("cell:")
                .and_then(|id| id.parse().ok())
                .map(|id| Self::Cell(CellId(id)))
                .ok_or_else(|| format!("unsupported reaction site '{other}'")),
        }
    }
}

impl Serialize for ReactionSite {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for ReactionSite {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let text = String::deserialize(deserializer)?;
        text.parse().map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FluxRecord {
    pub tick_count: u64,
    pub sim_time_seconds: f64,
    pub cell_id: usize,
    pub x: f32,
    pub y: f32,
    pub catalyst_index: usize,
    pub site: ReactionSite,
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
    pub id: CellId,
    pub position: Position,
    pub radius: f32,
    pub energy: f64,
    pub genome: Genome,
    pub lineage_id: LineageId,
    pub internal_elements: ElementAmounts,
    pub body: ElementAmounts,
    pub boundary: ElementAmounts,
    pub birth_sim_time: f64,
    pub maintenance_cost_per_area: f64,
    pub last_digested_tick: Option<u64>,
}

impl Cell {
    pub fn new(
        genome: Genome,
        position: Position,
        birth_sim_time: f64,
        energy: f64,
        body: ElementAmounts,
        boundary: ElementAmounts,
        body_density: f32,
    ) -> Self {
        let lineage_id = genome.lineage_id;
        let maintenance_cost_per_area = genome.maintenance_cost_per_area;
        Self {
            id: CellId(usize::MAX),
            position,
            radius: (body.total() / f64::from(body_density) / std::f64::consts::PI).sqrt() as f32,
            body,
            boundary,
            energy,
            genome,
            lineage_id,
            internal_elements: ElementAmounts::ZERO,
            birth_sim_time,
            maintenance_cost_per_area,
            last_digested_tick: None,
        }
    }

    pub fn area(&self, body_density: f32) -> f64 {
        self.body.total() / f64::from(body_density)
    }

    pub fn refresh_radius(&mut self, body_density: f32) {
        self.radius = (self.area(body_density) / std::f64::consts::PI).sqrt() as f32;
    }

    pub fn membrane_polarity(&self) -> f64 {
        let total = self.boundary.total();
        if total <= 0.0 {
            return 0.5;
        }
        crate::chem::ELEMENT_ORDER
            .iter()
            .map(|element| {
                f64::from(self.boundary[*element]) * f64::from(element.properties().polarity)
            })
            .sum::<f64>()
            / total
    }

    pub fn strain(&self, boundary_density: f32) -> f64 {
        self.boundary.total()
            / f64::from(boundary_density)
            / (std::f64::consts::TAU * f64::from(self.radius))
    }
}

#[derive(Clone, Debug, Default)]
pub struct CellFluxLog {
    recent_fluxes: Vec<FluxRecord>,
    flux_head: usize,
}

impl CellFluxLog {
    pub fn clear(&mut self) {
        self.recent_fluxes.clear();
        self.flux_head = 0;
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
    use super::{CELL_FLUX_LOG_CAPACITY, Cell, CellId, CellStore, FluxRecord, ReactionSite};
    use crate::genome::Genome;
    use crate::rng::Rng;
    use crate::spatial::Position;

    fn cell(seed: &str) -> Cell {
        let mut rng = Rng::from_seed_str(seed);
        Cell::new(
            Genome::random_founder(&mut rng, 0.0),
            Position::new(1.0, 1.0),
            0.0,
            2.0,
            crate::chem::ElementAmounts::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            crate::chem::ElementAmounts::ZERO,
            1.0,
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
            site: ReactionSite::Internal,
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
        let mut cell = super::CellFluxLog::default();
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
    fn reaction_sites_render_and_parse_as_text() {
        for (site, text) in [
            (ReactionSite::Internal, "internal"),
            (ReactionSite::Field, "field"),
            (ReactionSite::Cell(CellId(42)), "cell:42"),
        ] {
            assert_eq!(site.to_string(), text);
            assert_eq!(text.parse::<ReactionSite>().unwrap(), site);
        }
        assert!("cell:".parse::<ReactionSite>().is_err());
        assert!("surface".parse::<ReactionSite>().is_err());
        let mut flux = record(3);
        flux.site = ReactionSite::Cell(CellId(7));
        let decoded: FluxRecord = bincode::deserialize(&bincode::serialize(&flux).unwrap()).unwrap();
        assert_eq!(decoded, flux);
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
