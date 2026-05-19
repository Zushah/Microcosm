use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::chem::{Element, ELEMENT_COUNT};
use crate::genome::EnzymeType;

pub const ENZYME_COUNT_HISTOGRAM_LEN: usize = 11;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnzymeTypeCounts {
    pub anabolase: u64,
    pub catabolase: u64,
    pub transmutase: u64,
    pub defensase: u64,
    pub attackase: u64,
}

impl EnzymeTypeCounts {
    pub fn add(&mut self, enzyme_type: EnzymeType, value: u64) {
        match enzyme_type {
            EnzymeType::Anabolase => self.anabolase = self.anabolase.saturating_add(value),
            EnzymeType::Catabolase => self.catabolase = self.catabolase.saturating_add(value),
            EnzymeType::Transmutase => self.transmutase = self.transmutase.saturating_add(value),
            EnzymeType::Defensase => self.defensase = self.defensase.saturating_add(value),
            EnzymeType::Attackase => self.attackase = self.attackase.saturating_add(value),
        }
    }

    pub fn increment(&mut self, enzyme_type: EnzymeType) {
        self.add(enzyme_type, 1);
    }

    pub fn get(self, enzyme_type: EnzymeType) -> u64 {
        match enzyme_type {
            EnzymeType::Anabolase => self.anabolase,
            EnzymeType::Catabolase => self.catabolase,
            EnzymeType::Transmutase => self.transmutase,
            EnzymeType::Defensase => self.defensase,
            EnzymeType::Attackase => self.attackase,
        }
    }

    pub fn total(self) -> u64 {
        self.anabolase
            .saturating_add(self.catabolase)
            .saturating_add(self.transmutase)
            .saturating_add(self.defensase)
            .saturating_add(self.attackase)
    }

    pub fn saturating_delta(self, previous: Self) -> Self {
        Self {
            anabolase: self.anabolase.saturating_sub(previous.anabolase),
            catabolase: self.catabolase.saturating_sub(previous.catabolase),
            transmutase: self.transmutase.saturating_sub(previous.transmutase),
            defensase: self.defensase.saturating_sub(previous.defensase),
            attackase: self.attackase.saturating_sub(previous.attackase),
        }
    }

    pub fn add_assign(&mut self, other: Self) {
        self.anabolase = self.anabolase.saturating_add(other.anabolase);
        self.catabolase = self.catabolase.saturating_add(other.catabolase);
        self.transmutase = self.transmutase.saturating_add(other.transmutase);
        self.defensase = self.defensase.saturating_add(other.defensase);
        self.attackase = self.attackase.saturating_add(other.attackase);
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EnzymeTypeAmounts {
    pub anabolase: f64,
    pub catabolase: f64,
    pub transmutase: f64,
    pub defensase: f64,
    pub attackase: f64,
}

impl EnzymeTypeAmounts {
    pub fn add(&mut self, enzyme_type: EnzymeType, value: f64) {
        match enzyme_type {
            EnzymeType::Anabolase => self.anabolase += value,
            EnzymeType::Catabolase => self.catabolase += value,
            EnzymeType::Transmutase => self.transmutase += value,
            EnzymeType::Defensase => self.defensase += value,
            EnzymeType::Attackase => self.attackase += value,
        }
    }

    pub fn total(self) -> f64 {
        self.anabolase + self.catabolase + self.transmutase + self.defensase + self.attackase
    }

    pub fn delta(self, previous: Self) -> Self {
        Self {
            anabolase: self.anabolase - previous.anabolase,
            catabolase: self.catabolase - previous.catabolase,
            transmutase: self.transmutase - previous.transmutase,
            defensase: self.defensase - previous.defensase,
            attackase: self.attackase - previous.attackase,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperationCounters {
    pub cell_steps: u64,
    pub enzyme_entries_seen: u64,
    pub metabolic_enzyme_attempts: u64,
    pub reaction_gates_passed: u64,
    pub reactions_succeeded: u64,
    pub substrate_candidates_scanned: u64,
    pub molecule_diffusion_events: u64,
    pub molecule_moves: u64,
    pub molecule_slots_reused: u64,
    pub molecule_slots_newly_allocated: u64,
    pub molecule_uptakes: u64,
    pub products_created: u64,
    pub byproducts_created: u64,
    pub cell_divisions: u64,
    pub cell_deaths: u64,
    pub predation_pairs_checked: u64,
    pub predation_occupied_tiles_considered: u64,
    pub predation_candidate_neighbor_pairs: u64,
    pub predation_cross_lineage_pairs: u64,
    pub predation_events: u64,
    pub predation_cells_consumed: u64,
    pub local_enval_average_calls: u64,
    pub combat_enzyme_skips: u64,
    pub enzyme_list_clones: u64,
    pub genome_clones: u64,
}

impl OperationCounters {
    pub fn saturating_delta(self, previous: Self) -> Self {
        Self {
            cell_steps: self.cell_steps.saturating_sub(previous.cell_steps),
            enzyme_entries_seen: self
                .enzyme_entries_seen
                .saturating_sub(previous.enzyme_entries_seen),
            metabolic_enzyme_attempts: self
                .metabolic_enzyme_attempts
                .saturating_sub(previous.metabolic_enzyme_attempts),
            reaction_gates_passed: self
                .reaction_gates_passed
                .saturating_sub(previous.reaction_gates_passed),
            reactions_succeeded: self
                .reactions_succeeded
                .saturating_sub(previous.reactions_succeeded),
            substrate_candidates_scanned: self
                .substrate_candidates_scanned
                .saturating_sub(previous.substrate_candidates_scanned),
            molecule_diffusion_events: self
                .molecule_diffusion_events
                .saturating_sub(previous.molecule_diffusion_events),
            molecule_moves: self.molecule_moves.saturating_sub(previous.molecule_moves),
            molecule_slots_reused: self
                .molecule_slots_reused
                .saturating_sub(previous.molecule_slots_reused),
            molecule_slots_newly_allocated: self
                .molecule_slots_newly_allocated
                .saturating_sub(previous.molecule_slots_newly_allocated),
            molecule_uptakes: self
                .molecule_uptakes
                .saturating_sub(previous.molecule_uptakes),
            products_created: self
                .products_created
                .saturating_sub(previous.products_created),
            byproducts_created: self
                .byproducts_created
                .saturating_sub(previous.byproducts_created),
            cell_divisions: self.cell_divisions.saturating_sub(previous.cell_divisions),
            cell_deaths: self.cell_deaths.saturating_sub(previous.cell_deaths),
            predation_pairs_checked: self
                .predation_pairs_checked
                .saturating_sub(previous.predation_pairs_checked),
            predation_occupied_tiles_considered: self
                .predation_occupied_tiles_considered
                .saturating_sub(previous.predation_occupied_tiles_considered),
            predation_candidate_neighbor_pairs: self
                .predation_candidate_neighbor_pairs
                .saturating_sub(previous.predation_candidate_neighbor_pairs),
            predation_cross_lineage_pairs: self
                .predation_cross_lineage_pairs
                .saturating_sub(previous.predation_cross_lineage_pairs),
            predation_events: self
                .predation_events
                .saturating_sub(previous.predation_events),
            predation_cells_consumed: self
                .predation_cells_consumed
                .saturating_sub(previous.predation_cells_consumed),
            local_enval_average_calls: self
                .local_enval_average_calls
                .saturating_sub(previous.local_enval_average_calls),
            combat_enzyme_skips: self
                .combat_enzyme_skips
                .saturating_sub(previous.combat_enzyme_skips),
            enzyme_list_clones: self
                .enzyme_list_clones
                .saturating_sub(previous.enzyme_list_clones),
            genome_clones: self.genome_clones.saturating_sub(previous.genome_clones),
        }
    }

    pub fn add_assign(&mut self, other: Self) {
        self.cell_steps = self.cell_steps.saturating_add(other.cell_steps);
        self.enzyme_entries_seen = self
            .enzyme_entries_seen
            .saturating_add(other.enzyme_entries_seen);
        self.metabolic_enzyme_attempts = self
            .metabolic_enzyme_attempts
            .saturating_add(other.metabolic_enzyme_attempts);
        self.reaction_gates_passed = self
            .reaction_gates_passed
            .saturating_add(other.reaction_gates_passed);
        self.reactions_succeeded = self
            .reactions_succeeded
            .saturating_add(other.reactions_succeeded);
        self.substrate_candidates_scanned = self
            .substrate_candidates_scanned
            .saturating_add(other.substrate_candidates_scanned);
        self.molecule_diffusion_events = self
            .molecule_diffusion_events
            .saturating_add(other.molecule_diffusion_events);
        self.molecule_moves = self.molecule_moves.saturating_add(other.molecule_moves);
        self.molecule_slots_reused = self
            .molecule_slots_reused
            .saturating_add(other.molecule_slots_reused);
        self.molecule_slots_newly_allocated = self
            .molecule_slots_newly_allocated
            .saturating_add(other.molecule_slots_newly_allocated);
        self.molecule_uptakes = self.molecule_uptakes.saturating_add(other.molecule_uptakes);
        self.products_created = self.products_created.saturating_add(other.products_created);
        self.byproducts_created = self
            .byproducts_created
            .saturating_add(other.byproducts_created);
        self.cell_divisions = self.cell_divisions.saturating_add(other.cell_divisions);
        self.cell_deaths = self.cell_deaths.saturating_add(other.cell_deaths);
        self.predation_pairs_checked = self
            .predation_pairs_checked
            .saturating_add(other.predation_pairs_checked);
        self.predation_occupied_tiles_considered = self
            .predation_occupied_tiles_considered
            .saturating_add(other.predation_occupied_tiles_considered);
        self.predation_candidate_neighbor_pairs = self
            .predation_candidate_neighbor_pairs
            .saturating_add(other.predation_candidate_neighbor_pairs);
        self.predation_cross_lineage_pairs = self
            .predation_cross_lineage_pairs
            .saturating_add(other.predation_cross_lineage_pairs);
        self.predation_events = self.predation_events.saturating_add(other.predation_events);
        self.predation_cells_consumed = self
            .predation_cells_consumed
            .saturating_add(other.predation_cells_consumed);
        self.local_enval_average_calls = self
            .local_enval_average_calls
            .saturating_add(other.local_enval_average_calls);
        self.combat_enzyme_skips = self
            .combat_enzyme_skips
            .saturating_add(other.combat_enzyme_skips);
        self.enzyme_list_clones = self
            .enzyme_list_clones
            .saturating_add(other.enzyme_list_clones);
        self.genome_clones = self.genome_clones.saturating_add(other.genome_clones);
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ReactionCounters {
    pub attempts_by_type: EnzymeTypeCounts,
    pub gates_passed_by_type: EnzymeTypeCounts,
    pub successes_by_type: EnzymeTypeCounts,
    pub no_substrate_by_type: EnzymeTypeCounts,
    pub energy_delta_by_type: EnzymeTypeAmounts,
    pub enval_input_by_type: EnzymeTypeAmounts,
    pub enval_output_by_type: EnzymeTypeAmounts,
    pub molecule_uptakes: u64,
    pub molecule_outputs: u64,
    pub divisions: u64,
}

impl ReactionCounters {
    pub fn saturating_delta(self, previous: Self) -> Self {
        Self {
            attempts_by_type: self
                .attempts_by_type
                .saturating_delta(previous.attempts_by_type),
            gates_passed_by_type: self
                .gates_passed_by_type
                .saturating_delta(previous.gates_passed_by_type),
            successes_by_type: self
                .successes_by_type
                .saturating_delta(previous.successes_by_type),
            no_substrate_by_type: self
                .no_substrate_by_type
                .saturating_delta(previous.no_substrate_by_type),
            energy_delta_by_type: self
                .energy_delta_by_type
                .delta(previous.energy_delta_by_type),
            enval_input_by_type: self.enval_input_by_type.delta(previous.enval_input_by_type),
            enval_output_by_type: self
                .enval_output_by_type
                .delta(previous.enval_output_by_type),
            molecule_uptakes: self
                .molecule_uptakes
                .saturating_sub(previous.molecule_uptakes),
            molecule_outputs: self
                .molecule_outputs
                .saturating_sub(previous.molecule_outputs),
            divisions: self.divisions.saturating_sub(previous.divisions),
        }
    }

    pub fn total_attempts(self) -> u64 {
        self.attempts_by_type.total()
    }

    pub fn total_successes(self) -> u64 {
        self.successes_by_type.total()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorldStats {
    pub tick_count: u64,
    pub sim_time_seconds: f64,
    pub width: usize,
    pub height: usize,
    pub tile_count: usize,
    pub occupied_tile_count: usize,
    pub empty_tile_count: usize,
    pub occupancy_fraction: f64,
    pub molecule_count: usize,
    pub tile_molecule_count: usize,
    pub cell_molecule_count: usize,
    pub free_molecule_record_count: usize,
    pub active_molecule_record_count: usize,
    pub molecule_arena_len: usize,
    pub molecule_arena_high_water_mark: usize,
    pub molecule_slots_reused: u64,
    pub molecule_slots_newly_allocated: u64,
    pub total_atom_count: u64,
    pub tile_atom_count: u64,
    pub cell_atom_count: u64,
    pub average_molecules_per_tile: f64,
    pub average_internal_molecules_per_live_cell: f64,
    pub average_atoms_per_live_cell: f64,
    pub average_enval: f32,
    pub min_enval: f32,
    pub max_enval: f32,
    pub enval_std_dev: f32,
    pub enval_p05: f32,
    pub enval_p50: f32,
    pub enval_p95: f32,
    pub positive_enval_tile_count: usize,
    pub negative_enval_tile_count: usize,
    pub near_zero_enval_tile_count: usize,
    pub element_counts: [u64; ELEMENT_COUNT],
    pub cell_count: usize,
    pub live_cell_count: usize,
    pub cell_record_count: usize,
    pub dead_cell_count: usize,
    pub births: u64,
    pub deaths: u64,
    pub predation_events: u64,
    pub cells_consumed: u64,
    pub predator_energy_gained: f64,
    pub average_energy_gained_per_predation: f64,
    pub predation_enzyme_transfers: u64,
    pub predation_enzyme_replacements: u64,
    pub lineage_count: usize,
    pub extant_lineage_count: usize,
    pub total_lineage_records: usize,
    pub extinct_lineage_count: usize,
    pub dominant_lineage_id: u64,
    pub dominant_lineage_population: u64,
    pub dominant_lineage_share: f64,
    pub lineage_entropy: f64,
    pub average_cell_energy: f64,
    pub min_cell_energy: f64,
    pub max_cell_energy: f64,
    pub total_cell_energy: f64,
    pub average_cell_age: f64,
    pub max_cell_age: f64,
    pub average_time_without_food: f64,
    pub average_enzyme_count: f64,
    pub min_enzyme_count: usize,
    pub max_enzyme_count: usize,
    pub enzyme_count_histogram: [u64; ENZYME_COUNT_HISTOGRAM_LEN],
    pub cells_at_enzyme_cap: usize,
    pub fraction_cells_at_enzyme_cap: f64,
    pub cells_with_attackase: usize,
    pub cells_with_defensase: usize,
    pub average_attack_total: f64,
    pub max_attack_total: u32,
    pub average_defense_total: f64,
    pub max_defense_total: u32,
    pub enzyme_type_totals: EnzymeTypeCounts,
    pub reaction_counters: ReactionCounters,
    pub operation_counters: OperationCounters,
}

impl WorldStats {
    pub fn element_count(&self, element: Element) -> u64 {
        self.element_counts[element.index()]
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StepProfile {
    pub molecule_diffusion: Duration,
    pub cell_step: Duration,
    pub predation: Duration,
    pub enval_diffusion: Duration,
    pub total: Duration,
    pub counters: OperationCounters,
}

impl StepProfile {
    pub fn add_assign(&mut self, other: Self) {
        self.molecule_diffusion += other.molecule_diffusion;
        self.cell_step += other.cell_step;
        self.predation += other.predation;
        self.enval_diffusion += other.enval_diffusion;
        self.total += other.total;
        self.counters.add_assign(other.counters);
    }
}
