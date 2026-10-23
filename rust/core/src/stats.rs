use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::chem::{ELEMENT_COUNT, Element};

pub const ENZYME_COUNT_HISTOGRAM_LEN: usize = 11;
pub const EXPOSED_CELL_THRESHOLD: f32 = 0.05;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperationCounters {
    pub element_field_diffusion_tiles: u64,
    pub cell_steps: u64,
    pub enzyme_entries_seen: u64,
    pub metabolic_enzyme_attempts: u64,
    pub reactions_succeeded: u64,
    pub element_uptake_events: u64,
    pub cell_divisions: u64,
    pub cell_deaths: u64,
    pub local_enval_average_calls: u64,
    pub enzyme_list_clones: u64,
    pub genome_clones: u64,
    pub spatial_candidate_checks: u64,
    pub overlap_candidates: u64,
    pub overlap_corrections: u64,
    pub surface_contact_queries: u64,
}

impl OperationCounters {
    pub fn saturating_delta(self, previous: Self) -> Self {
        Self {
            element_field_diffusion_tiles: self
                .element_field_diffusion_tiles
                .saturating_sub(previous.element_field_diffusion_tiles),
            cell_steps: self.cell_steps.saturating_sub(previous.cell_steps),
            enzyme_entries_seen: self
                .enzyme_entries_seen
                .saturating_sub(previous.enzyme_entries_seen),
            metabolic_enzyme_attempts: self
                .metabolic_enzyme_attempts
                .saturating_sub(previous.metabolic_enzyme_attempts),
            reactions_succeeded: self
                .reactions_succeeded
                .saturating_sub(previous.reactions_succeeded),
            element_uptake_events: self
                .element_uptake_events
                .saturating_sub(previous.element_uptake_events),
            cell_divisions: self.cell_divisions.saturating_sub(previous.cell_divisions),
            cell_deaths: self.cell_deaths.saturating_sub(previous.cell_deaths),
            local_enval_average_calls: self
                .local_enval_average_calls
                .saturating_sub(previous.local_enval_average_calls),
            enzyme_list_clones: self
                .enzyme_list_clones
                .saturating_sub(previous.enzyme_list_clones),
            genome_clones: self.genome_clones.saturating_sub(previous.genome_clones),
            spatial_candidate_checks: self
                .spatial_candidate_checks
                .saturating_sub(previous.spatial_candidate_checks),
            overlap_candidates: self
                .overlap_candidates
                .saturating_sub(previous.overlap_candidates),
            overlap_corrections: self
                .overlap_corrections
                .saturating_sub(previous.overlap_corrections),
            surface_contact_queries: self
                .surface_contact_queries
                .saturating_sub(previous.surface_contact_queries),
        }
    }

    pub fn add_assign(&mut self, other: Self) {
        self.element_field_diffusion_tiles = self
            .element_field_diffusion_tiles
            .saturating_add(other.element_field_diffusion_tiles);
        self.cell_steps = self.cell_steps.saturating_add(other.cell_steps);
        self.enzyme_entries_seen = self
            .enzyme_entries_seen
            .saturating_add(other.enzyme_entries_seen);
        self.metabolic_enzyme_attempts = self
            .metabolic_enzyme_attempts
            .saturating_add(other.metabolic_enzyme_attempts);
        self.reactions_succeeded = self
            .reactions_succeeded
            .saturating_add(other.reactions_succeeded);
        self.element_uptake_events = self
            .element_uptake_events
            .saturating_add(other.element_uptake_events);
        self.cell_divisions = self.cell_divisions.saturating_add(other.cell_divisions);
        self.cell_deaths = self.cell_deaths.saturating_add(other.cell_deaths);
        self.local_enval_average_calls = self
            .local_enval_average_calls
            .saturating_add(other.local_enval_average_calls);
        self.enzyme_list_clones = self
            .enzyme_list_clones
            .saturating_add(other.enzyme_list_clones);
        self.genome_clones = self.genome_clones.saturating_add(other.genome_clones);
        self.spatial_candidate_checks = self
            .spatial_candidate_checks
            .saturating_add(other.spatial_candidate_checks);
        self.overlap_candidates = self
            .overlap_candidates
            .saturating_add(other.overlap_candidates);
        self.overlap_corrections = self
            .overlap_corrections
            .saturating_add(other.overlap_corrections);
        self.surface_contact_queries = self
            .surface_contact_queries
            .saturating_add(other.surface_contact_queries);
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ReactionCounters {
    pub attempts: u64,
    pub successes: u64,
    pub no_substrate: u64,
    pub energy_delta: f64,
    pub enval_input: f64,
    pub enval_output: f64,
    pub executed_metabolic_flux: f64,
    pub uptake_flux: f64,
    pub leak_flux: f64,
    pub secretion_flux: f64,
    pub divisions: u64,
}

impl ReactionCounters {
    pub fn saturating_delta(self, previous: Self) -> Self {
        Self {
            attempts: self.attempts.saturating_sub(previous.attempts),
            successes: self.successes.saturating_sub(previous.successes),
            no_substrate: self.no_substrate.saturating_sub(previous.no_substrate),
            energy_delta: self.energy_delta - previous.energy_delta,
            enval_input: self.enval_input - previous.enval_input,
            enval_output: self.enval_output - previous.enval_output,
            executed_metabolic_flux: self.executed_metabolic_flux
                - previous.executed_metabolic_flux,
            uptake_flux: self.uptake_flux - previous.uptake_flux,
            leak_flux: self.leak_flux - previous.leak_flux,
            secretion_flux: self.secretion_flux - previous.secretion_flux,
            divisions: self.divisions.saturating_sub(previous.divisions),
        }
    }

    pub fn total_attempts(self) -> u64 {
        self.attempts
    }

    pub fn total_successes(self) -> u64 {
        self.successes
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct InteractionCounters {
    pub field_surface_extent: f64,
    pub cell_surface_extent: f64,
    pub boundary_digested: f64,
    pub cross_lineage_boundary_digested: f64,
    pub lyses_after_digestion: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EnergyLedger {
    pub founder_energy: f64,
    pub injected_energy: f64,
    pub extracted_energy: f64,
    pub chemical_harvest: f64,
    pub chemical_cost: f64,
    pub enval_harvest: f64,
    pub pump_cost: f64,
    pub assembly_cost: f64,
    pub maintenance: f64,
    pub death_loss: f64,
}

impl EnergyLedger {
    pub fn expected_cell_energy(&self) -> f64 {
        self.founder_energy + self.injected_energy + self.chemical_harvest + self.enval_harvest
            - self.extracted_energy
            - self.chemical_cost
            - self.assembly_cost
            - self.pump_cost
            - self.maintenance
            - self.death_loss
    }

    pub fn magnitude(&self) -> f64 {
        self.founder_energy
            + self.injected_energy
            + self.extracted_energy
            + self.chemical_harvest
            + self.chemical_cost
            + self.enval_harvest
            + self.assembly_cost
            + self.pump_cost
            + self.maintenance
            + self.death_loss
    }

    pub fn closure_tolerance(&self) -> f64 {
        1.0e-9 * self.magnitude() + 1.0e-6
    }

    pub fn terms(&self) -> [(&'static str, f64); 10] {
        [
            ("founder_energy", self.founder_energy),
            ("injected_energy", self.injected_energy),
            ("extracted_energy", self.extracted_energy),
            ("chemical_harvest", self.chemical_harvest),
            ("chemical_cost", self.chemical_cost),
            ("enval_harvest", self.enval_harvest),
            ("pump_cost", self.pump_cost),
            ("assembly_cost", self.assembly_cost),
            ("maintenance", self.maintenance),
            ("death_loss", self.death_loss),
        ]
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EnvalLedger {
    pub initial_total: f64,
    pub source_inflow: f64,
    pub cell_uptake: f64,
    pub cell_emission: f64,
    pub recharge: f64,
    pub edits: f64,
    pub recharge_amount: f64,
    pub recharge_energy: f64,
}

impl EnvalLedger {
    pub fn net_change(&self) -> f64 {
        self.source_inflow + self.cell_uptake + self.cell_emission + self.recharge + self.edits
    }
}

pub fn renewable_coverage(energy: &EnergyLedger, enval: &EnvalLedger) -> f64 {
    let denominator = energy.enval_harvest + energy.chemical_harvest;
    if denominator > 0.0 {
        (energy.enval_harvest + enval.recharge_energy) / denominator
    } else {
        0.0
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
    pub extracellular_element_amounts: [f64; ELEMENT_COUNT],
    pub intracellular_element_amounts: [f64; ELEMENT_COUNT],
    pub system_element_amounts: [f64; ELEMENT_COUNT],
    pub total_element_amount: f64,
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
    pub cell_count: usize,
    pub live_cell_count: usize,
    pub births: u64,
    pub deaths: u64,
    pub fissions: u64,
    pub energy_deaths: u64,
    pub lyses: u64,
    pub radius_median: f32,
    pub radius_p95: f32,
    pub radius_max: f32,
    pub mean_strain: f64,
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
    pub average_enzyme_count: f64,
    pub min_enzyme_count: usize,
    pub max_enzyme_count: usize,
    pub enzyme_count_histogram: [u64; ENZYME_COUNT_HISTOGRAM_LEN],
    pub cells_at_enzyme_cap: usize,
    pub fraction_cells_at_enzyme_cap: f64,
    pub total_enzyme_count: u64,
    pub mean_exposure: f64,
    pub max_exposure: f32,
    pub exposed_cell_count: usize,
    pub contact_pairs: u64,
    pub cross_lineage_contact_pairs: u64,
    pub interaction: InteractionCounters,
    pub reaction_counters: ReactionCounters,
    pub operation_counters: OperationCounters,
    pub energy_ledger: EnergyLedger,
    pub enval_ledger: EnvalLedger,
    pub energy_ledger_residual: f64,
    pub renewable_coverage: f64,
}

impl WorldStats {
    pub fn extracellular_element_amount(&self, element: Element) -> f64 {
        self.extracellular_element_amounts[element.index()]
    }

    pub fn intracellular_element_amount(&self, element: Element) -> f64 {
        self.intracellular_element_amounts[element.index()]
    }

    pub fn system_element_amount(&self, element: Element) -> f64 {
        self.system_element_amounts[element.index()]
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StepProfile {
    pub element_field_diffusion: Duration,
    pub cell_step: Duration,
    pub cell_mechanics: Duration,
    pub enval_diffusion: Duration,
    pub total: Duration,
    pub counters: OperationCounters,
}

impl StepProfile {
    pub fn add_assign(&mut self, other: Self) {
        self.element_field_diffusion += other.element_field_diffusion;
        self.cell_step += other.cell_step;
        self.cell_mechanics += other.cell_mechanics;
        self.enval_diffusion += other.enval_diffusion;
        self.total += other.total;
        self.counters.add_assign(other.counters);
    }
}
