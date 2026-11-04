//! The one table of world statistics. Each row names a stat, says how to read it,
//! and states where it appears: rows with a CSV format are columns in table order,
//! rows with a JSON path are keys of the JSON record, and the text lines below are
//! templates that refer to rows by name. Adding a stat means adding one row.

use microcosmcore::WorldStats;
use serde_json::{Map, Value as Json};

use crate::output::StatsInterval;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Display,
    Fixed(usize),
    Exp(usize),
    Signed,
    Debug,
}

#[derive(Clone, Copy, Debug)]
pub enum Value<'a> {
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Amounts(&'a [f64; 6]),
    Counts(&'a [u64]),
}

impl From<u64> for Value<'_> {
    fn from(value: u64) -> Self {
        Self::U64(value)
    }
}

impl From<usize> for Value<'_> {
    fn from(value: usize) -> Self {
        Self::U64(value as u64)
    }
}

impl From<i64> for Value<'_> {
    fn from(value: i64) -> Self {
        Self::I64(value)
    }
}

impl From<f32> for Value<'_> {
    fn from(value: f32) -> Self {
        Self::F32(value)
    }
}

impl From<f64> for Value<'_> {
    fn from(value: f64) -> Self {
        Self::F64(value)
    }
}

impl Value<'_> {
    pub fn render(self, format: Format) -> String {
        match (self, format) {
            (Self::U64(value), Format::Signed) => format!("{value:+}"),
            (Self::U64(value), _) => value.to_string(),
            (Self::I64(value), Format::Signed) => format!("{value:+}"),
            (Self::I64(value), _) => value.to_string(),
            (Self::F32(value), Format::Fixed(digits)) => format!("{value:.digits$}"),
            (Self::F32(value), Format::Exp(digits)) => format!("{value:.digits$e}"),
            (Self::F32(value), _) => value.to_string(),
            (Self::F64(value), Format::Fixed(digits)) => format!("{value:.digits$}"),
            (Self::F64(value), Format::Exp(digits)) => format!("{value:.digits$e}"),
            (Self::F64(value), _) => value.to_string(),
            (Self::Amounts(values), _) => format!("{values:?}"),
            (Self::Counts(values), _) => format!("{values:?}"),
        }
    }

    fn json(self) -> Json {
        match self {
            Self::U64(value) => value.into(),
            Self::I64(value) => value.into(),
            Self::F32(value) => value.into(),
            Self::F64(value) => value.into(),
            Self::Amounts(values) => values.as_slice().into(),
            Self::Counts(values) => values.into(),
        }
    }
}

pub struct StatField {
    pub name: &'static str,
    pub csv: Option<Format>,
    pub json: Option<&'static str>,
    pub get: for<'a> fn(&'a WorldStats, &'a StatsInterval) -> Value<'a>,
}

const D: Option<Format> = Some(Format::Display);
const F3: Option<Format> = Some(Format::Fixed(3));
const F6: Option<Format> = Some(Format::Fixed(6));
const F8: Option<Format> = Some(Format::Fixed(8));
const F9: Option<Format> = Some(Format::Fixed(9));
const E6: Option<Format> = Some(Format::Exp(6));
const NO_CSV: Option<Format> = None;

macro_rules! stat {
    ($name:literal, $csv:expr, $get:expr) => {
        StatField {
            name: $name,
            csv: $csv,
            json: None,
            get: $get,
        }
    };
    ($name:literal, $csv:expr, json $json:literal, $get:expr) => {
        StatField {
            name: $name,
            csv: $csv,
            json: Some($json),
            get: $get,
        }
    };
}

pub static STAT_FIELDS: &[StatField] = &[
    stat!("tick", D, json "tick", |s, _| s.tick_count.into()),
    stat!("sim_time", F6, json "sim_time", |s, _| s.sim_time_seconds.into()),
    stat!("population", D, |s, _| s.cell_count.into()),
    stat!("live_cells", D, json "population", |s, _| s.live_cell_count.into()),
    stat!("occupancy_fraction", F8, json "occupancy_fraction", |s, _| s.occupancy_fraction.into()),
    stat!("occupied_tiles", D, |s, _| s.occupied_tile_count.into()),
    stat!("empty_tiles", D, |s, _| s.empty_tile_count.into()),
    stat!("births", D, json "births", |s, _| s.births.into()),
    stat!("deaths", D, json "deaths", |s, _| s.deaths.into()),
    stat!("interval_births", D, json "interval.births", |_, i| i.births.into()),
    stat!("interval_deaths", D, json "interval.deaths", |_, i| i.deaths.into()),
    stat!("interval_pop_delta", D, json "interval.population_delta", |_, i| i.population_delta.into()),
    stat!("lineages", D, json "lineages", |s, _| s.lineage_count.into()),
    stat!("total_lineage_records", D, json "total_lineage_records", |s, _| s.total_lineage_records.into()),
    stat!("extinct_lineages", D, json "extinct_lineages", |s, _| s.extinct_lineage_count.into()),
    stat!("dominant_lineage", D, json "dominant_lineage", |s, _| s.dominant_lineage_id.into()),
    stat!("dominant_lineage_population", D, |s, _| s
        .dominant_lineage_population
        .into()),
    stat!("dominant_lineage_share", F8, json "dominant_lineage_share", |s, _| s.dominant_lineage_share.into()),
    stat!("lineage_entropy", F8, json "lineage_entropy", |s, _| s.lineage_entropy.into()),
    stat!("extracellular_a", F8, json "chemistry.extracellular.A", |s, _| s.extracellular_element_amounts[0].into()),
    stat!("extracellular_b", F8, json "chemistry.extracellular.B", |s, _| s.extracellular_element_amounts[1].into()),
    stat!("extracellular_c", F8, json "chemistry.extracellular.C", |s, _| s.extracellular_element_amounts[2].into()),
    stat!("extracellular_d", F8, json "chemistry.extracellular.D", |s, _| s.extracellular_element_amounts[3].into()),
    stat!("extracellular_e", F8, json "chemistry.extracellular.E", |s, _| s.extracellular_element_amounts[4].into()),
    stat!("extracellular_f", F8, json "chemistry.extracellular.F", |s, _| s.extracellular_element_amounts[5].into()),
    stat!("intracellular_a", F8, json "chemistry.intracellular.A", |s, _| s.intracellular_element_amounts[0].into()),
    stat!("intracellular_b", F8, json "chemistry.intracellular.B", |s, _| s.intracellular_element_amounts[1].into()),
    stat!("intracellular_c", F8, json "chemistry.intracellular.C", |s, _| s.intracellular_element_amounts[2].into()),
    stat!("intracellular_d", F8, json "chemistry.intracellular.D", |s, _| s.intracellular_element_amounts[3].into()),
    stat!("intracellular_e", F8, json "chemistry.intracellular.E", |s, _| s.intracellular_element_amounts[4].into()),
    stat!("intracellular_f", F8, json "chemistry.intracellular.F", |s, _| s.intracellular_element_amounts[5].into()),
    stat!("system_a", F8, json "chemistry.system.A", |s, _| s.system_element_amounts[0].into()),
    stat!("system_b", F8, json "chemistry.system.B", |s, _| s.system_element_amounts[1].into()),
    stat!("system_c", F8, json "chemistry.system.C", |s, _| s.system_element_amounts[2].into()),
    stat!("system_d", F8, json "chemistry.system.D", |s, _| s.system_element_amounts[3].into()),
    stat!("system_e", F8, json "chemistry.system.E", |s, _| s.system_element_amounts[4].into()),
    stat!("system_f", F8, json "chemistry.system.F", |s, _| s.system_element_amounts[5].into()),
    stat!("total_element_amount", F8, json "chemistry.total_element_amount", |s, _| s.total_element_amount.into()),
    stat!("avg_energy", F6, json "avg_energy", |s, _| s.average_cell_energy.into()),
    stat!("min_energy", F6, json "min_energy", |s, _| s.min_cell_energy.into()),
    stat!("max_energy", F6, json "max_energy", |s, _| s.max_cell_energy.into()),
    stat!("total_energy", F6, json "total_energy", |s, _| s.total_cell_energy.into()),
    stat!("avg_age", F6, |s, _| s.average_cell_age.into()),
    stat!("max_age", F6, |s, _| s.max_cell_age.into()),
    stat!("avg_enzyme_count", F6, json "avg_enzyme_count", |s, _| s.average_enzyme_count.into()),
    stat!("min_enzyme_count", D, |s, _| s.min_enzyme_count.into()),
    stat!("max_enzyme_count", D, |s, _| s.max_enzyme_count.into()),
    stat!("cells_at_enzyme_cap", D, json "cells_at_enzyme_cap", |s, _| s.cells_at_enzyme_cap.into()),
    stat!("fraction_at_enzyme_cap", F8, |s, _| s
        .fraction_cells_at_enzyme_cap
        .into()),
    stat!("total_enzymes", D, json "total_enzyme_count", |s, _| s.total_enzyme_count.into()),
    stat!("rx_attempts", D, json "metabolism.attempts", |s, _| s.reaction_counters.total_attempts().into()),
    stat!("rx_successes", D, json "metabolism.successes", |s, _| s.reaction_counters.total_successes().into()),
    stat!("rx_no_substrate", D, json "metabolism.no_substrate", |s, _| s.reaction_counters.no_substrate.into()),
    stat!("rx_energy_delta_total", F6, json "metabolism.energy_delta", |s, _| s.reaction_counters.energy_delta.into()),
    stat!("rx_enval_input_total", F6, json "metabolism.enval_input", |s, _| s.reaction_counters.enval_input.into()),
    stat!("rx_enval_output_total", F6, json "metabolism.enval_output", |s, _| s.reaction_counters.enval_output.into()),
    stat!("executed_metabolic_flux", F8, json "metabolism.executed_metabolic_flux", |s, _| s.reaction_counters.executed_metabolic_flux.into()),
    stat!("uptake_flux", F8, json "metabolism.uptake_flux", |s, _| s.reaction_counters.uptake_flux.into()),
    stat!("leak_flux", F8, json "metabolism.leak_flux", |s, _| s.reaction_counters.leak_flux.into()),
    stat!("secretion_flux", F8, json "metabolism.secretion_flux", |s, _| s.reaction_counters.secretion_flux.into()),
    stat!("divisions", D, json "metabolism.divisions", |s, _| s.reaction_counters.divisions.into()),
    stat!("interval_divisions", D, |_, i| i.divisions.into()),
    stat!("interval_reaction_attempts", D, json "interval.reaction_attempts", |_, i| i.reaction_attempts.into()),
    stat!("interval_reaction_successes", D, json "interval.reaction_successes", |_, i| i.reaction_successes.into()),
    stat!("interval_executed_metabolic_flux", F8, json "interval.executed_metabolic_flux", |_, i| i.executed_metabolic_flux.into()),
    stat!("interval_uptake_flux", F8, json "interval.uptake_flux", |_, i| i.uptake_flux.into()),
    stat!("interval_leak_flux", F8, json "interval.leak_flux", |_, i| i.leak_flux.into()),
    stat!("interval_secretion_flux", F8, json "interval.secretion_flux", |_, i| i.secretion_flux.into()),
    stat!("cell_steps", D, |s, _| s
        .operation_counters
        .cell_steps
        .into()),
    stat!("interval_cell_steps", D, json "interval.cell_steps", |_, i| i.cell_steps.into()),
    stat!("interval_cell_steps_per_sec", F3, json "interval.cell_steps_per_sec", |_, i| i.cell_steps_per_sec.into()),
    stat!("element_field_diffusion_tiles", D, |s, _| s
        .operation_counters
        .element_field_diffusion_tiles
        .into()),
    stat!("element_uptake_events", D, |s, _| s
        .operation_counters
        .element_uptake_events
        .into()),
    stat!("enval_avg", F6, json "enval.average", |s, _| s.average_enval.into()),
    stat!("enval_min", F6, json "enval.min", |s, _| s.min_enval.into()),
    stat!("enval_max", F6, json "enval.max", |s, _| s.max_enval.into()),
    stat!("enval_stddev", F6, json "enval.stddev", |s, _| s.enval_std_dev.into()),
    stat!("enval_p05", F6, json "enval.p05", |s, _| s.enval_p05.into()),
    stat!("enval_p50", F6, json "enval.p50", |s, _| s.enval_p50.into()),
    stat!("enval_p95", F6, json "enval.p95", |s, _| s.enval_p95.into()),
    stat!("enval_positive_tiles", D, |s, _| s
        .positive_enval_tile_count
        .into()),
    stat!("enval_negative_tiles", D, |s, _| s
        .negative_enval_tile_count
        .into()),
    stat!("enval_near_zero_tiles", D, |s, _| s
        .near_zero_enval_tile_count
        .into()),
    stat!("energy_founder", F9, json "energy_ledger.founder_energy", |s, _| s.energy_ledger.founder_energy.into()),
    stat!("energy_injected", F9, json "energy_ledger.injected_energy", |s, _| s.energy_ledger.injected_energy.into()),
    stat!("energy_extracted", F9, json "energy_ledger.extracted_energy", |s, _| s.energy_ledger.extracted_energy.into()),
    stat!("energy_chemical_harvest", F9, json "energy_ledger.chemical_harvest", |s, _| s.energy_ledger.chemical_harvest.into()),
    stat!("energy_chemical_cost", F9, json "energy_ledger.chemical_cost", |s, _| s.energy_ledger.chemical_cost.into()),
    stat!("energy_enval_harvest", F9, json "energy_ledger.enval_harvest", |s, _| s.energy_ledger.enval_harvest.into()),
    stat!("energy_pump_cost", F9, json "energy_ledger.pump_cost", |s, _| s.energy_ledger.pump_cost.into()),
    stat!("energy_maintenance", F9, json "energy_ledger.maintenance", |s, _| s.energy_ledger.maintenance.into()),
    stat!("energy_death_loss", F9, json "energy_ledger.death_loss", |s, _| s.energy_ledger.death_loss.into()),
    stat!("energy_ledger_residual", E6, json "energy_ledger.residual", |s, _| s.energy_ledger_residual.into()),
    stat!("renewable_coverage", F8, json "energy_ledger.renewable_coverage", |s, _| s.renewable_coverage.into()),
    stat!("enval_source_inflow", F9, json "enval_ledger.source_inflow", |s, _| s.enval_ledger.source_inflow.into()),
    stat!("enval_cell_uptake", F9, json "enval_ledger.cell_uptake", |s, _| s.enval_ledger.cell_uptake.into()),
    stat!("enval_cell_emission", F9, json "enval_ledger.cell_emission", |s, _| s.enval_ledger.cell_emission.into()),
    stat!("enval_recharge", F9, json "enval_ledger.recharge", |s, _| s.enval_ledger.recharge.into()),
    stat!("enval_edits", F9, json "enval_ledger.edits", |s, _| s.enval_ledger.edits.into()),
    stat!("recharge_amount", F9, json "enval_ledger.recharge_amount", |s, _| s.enval_ledger.recharge_amount.into()),
    stat!("recharge_energy", F9, json "enval_ledger.recharge_energy", |s, _| s.enval_ledger.recharge_energy.into()),
    stat!("fissions", D, json "fissions", |s, _| s.fissions.into()),
    stat!("energy_deaths", D, json "energy_deaths", |s, _| s.energy_deaths.into()),
    stat!("lyses", D, json "lyses", |s, _| s.lyses.into()),
    stat!("radius_median", D, json "radius_median", |s, _| s.radius_median.into()),
    stat!("radius_p95", D, json "radius_p95", |s, _| s.radius_p95.into()),
    stat!("radius_max", D, json "radius_max", |s, _| s.radius_max.into()),
    stat!("mean_strain", D, json "mean_strain", |s, _| s.mean_strain.into()),
    stat!("assembly_cost", D, json "energy_ledger.assembly_cost", |s, _| s.energy_ledger.assembly_cost.into()),
    stat!("mean_exposure", F8, json "interaction.mean_exposure", |s, _| s.mean_exposure.into()),
    stat!("max_exposure", F6, json "interaction.max_exposure", |s, _| s.max_exposure.into()),
    stat!("exposed_cells", D, json "interaction.exposed_cells", |s, _| s.exposed_cell_count.into()),
    stat!("contact_pairs", D, json "interaction.contact_pairs", |s, _| s.contact_pairs.into()),
    stat!("cross_lineage_contact_pairs", D, json "interaction.cross_lineage_contact_pairs", |s, _| s.cross_lineage_contact_pairs.into()),
    stat!("field_surface_extent", F9, json "interaction.field_surface_extent", |s, _| s.interaction.field_surface_extent.into()),
    stat!("cell_surface_extent", F9, json "interaction.cell_surface_extent", |s, _| s.interaction.cell_surface_extent.into()),
    stat!("boundary_digested", F9, json "interaction.boundary_digested", |s, _| s.interaction.boundary_digested.into()),
    stat!("cross_lineage_boundary_digested", F9, json "interaction.cross_lineage_boundary_digested", |s, _| s.interaction.cross_lineage_boundary_digested.into()),
    stat!("lyses_after_digestion", D, json "interaction.lyses_after_digestion", |s, _| s.interaction.lyses_after_digestion.into()),
    stat!("mixed_polarity_cells", D, json "niche.mixed_polarity_cells", |s, _| s.mixed_polarity_cells.into()),
    stat!("mean_enval_optimum_spread", F8, json "niche.mean_enval_optimum_spread", |s, _| s.mean_enval_optimum_spread.into()),
    stat!("mean_niche_mismatch", F8, json "niche.mean_niche_mismatch", |s, _| s.mean_niche_mismatch.into()),
    stat!("width", NO_CSV, json "width", |s, _| s.width.into()),
    stat!("height", NO_CSV, json "height", |s, _| s.height.into()),
    stat!("tile_count", NO_CSV, |s, _| s.tile_count.into()),
    stat!("extracellular_total", NO_CSV, |s, _| s
        .extracellular_element_amounts
        .iter()
        .sum::<f64>()
        .into()),
    stat!("intracellular_total", NO_CSV, |s, _| s
        .intracellular_element_amounts
        .iter()
        .sum::<f64>()
        .into()),
    stat!("extracellular_amounts", NO_CSV, |s, _| Value::Amounts(
        &s.extracellular_element_amounts
    )),
    stat!("intracellular_amounts", NO_CSV, |s, _| Value::Amounts(
        &s.intracellular_element_amounts
    )),
    stat!("system_amounts", NO_CSV, |s, _| Value::Amounts(
        &s.system_element_amounts
    )),
    stat!("enzyme_histogram_1_10", NO_CSV, |s, _| Value::Counts(
        &s.enzyme_count_histogram[1..]
    )),
    stat!("interval_ticks", NO_CSV, json "interval.ticks", |_, i| i.tick_delta.into()),
    stat!("interval_sim_seconds", NO_CSV, json "interval.sim_seconds", |_, i| i.sim_seconds_delta.into()),
    stat!("interval_wall_seconds", NO_CSV, json "interval.wall_seconds", |_, i| i.wall_seconds.into()),
    stat!("interval_steps_per_sec", NO_CSV, |_, i| i
        .steps_per_sec
        .into()),
    stat!("interval_enzyme_attempts", NO_CSV, |_, i| i
        .enzyme_attempts
        .into()),
    stat!("interval_enzyme_attempts_per_sec", NO_CSV, json "interval.enzyme_attempts_per_sec", |_, i| i.enzyme_attempts_per_sec.into()),
    stat!("interval_reactions_per_sec", NO_CSV, |_, i| i
        .reactions_per_sec
        .into()),
];

pub const COMPACT_TEXT: &[&str] = &[
    "tick={tick} time={sim_time:.3}s size={width}x{height} tiles={tile_count} occ={occupancy_fraction:.3} cells={live_cells} d_cells={interval_pop_delta:+} births={births} d_births={interval_births} deaths={deaths} d_deaths={interval_deaths} lineages={lineages} total_elements={total_element_amount:.6} extracellular={extracellular_total:.6} intracellular={intracellular_total:.6} avg_energy={avg_energy:.3} avg_enzymes={avg_enzyme_count:.2} cap={fraction_at_enzyme_cap:.3} rx_success={rx_successes} d_rx_success={interval_reaction_successes} flux={executed_metabolic_flux:.6} d_flux={interval_executed_metabolic_flux:.6} uptake={uptake_flux:.6} secretion={secretion_flux:.6} cell_steps={cell_steps} d_cell_steps={interval_cell_steps} enval_avg={enval_avg:.6} enval_min={enval_min:.6} enval_max={enval_max:.6} enval_sd={enval_stddev:.6} system=A:{system_a:.4} B:{system_b:.4} C:{system_c:.4} D:{system_d:.4} E:{system_e:.4} F:{system_f:.4} energy=founder:{energy_founder:.3} injected:{energy_injected:.3} extracted:{energy_extracted:.3} chem_harvest:{energy_chemical_harvest:.3} chem_cost:{energy_chemical_cost:.3} enval_harvest:{energy_enval_harvest:.3} pump:{energy_pump_cost:.3} maintenance:{energy_maintenance:.3} death_loss:{energy_death_loss:.3} residual:{energy_ledger_residual:.3e} coverage={renewable_coverage:.4} fissions={fissions} energy_deaths={energy_deaths} lyses={lyses} radius_median={radius_median:.4} radius_p95={radius_p95:.4} radius_max={radius_max:.4} mean_strain={mean_strain:.4} assembly_cost={assembly_cost:.6} exposure_mean={mean_exposure:.4} exposure_max={max_exposure:.3} exposed_cells={exposed_cells} contacts={contact_pairs} cross_contacts={cross_lineage_contact_pairs} field_extent={field_surface_extent:.6} cell_extent={cell_surface_extent:.6} digested={boundary_digested:.6} cross_digested={cross_lineage_boundary_digested:.6} lyses_after_digestion={lyses_after_digestion} mixed_polarity={mixed_polarity_cells} optimum_spread={mean_enval_optimum_spread:.4} niche_mismatch={mean_niche_mismatch:.4}",
];

pub const FULL_TEXT: &[&str] = &[
    "  fissions={fissions} energy_deaths={energy_deaths} lyses={lyses} radius_median={radius_median:.4} radius_p95={radius_p95:.4} radius_max={radius_max:.4} mean_strain={mean_strain:.4} assembly_cost={assembly_cost:.6}",
    "tick={tick} time={sim_time:.3}s size={width}x{height} sim_interval={interval_sim_seconds:.3}s wall_interval={interval_wall_seconds:.3}s steps_per_sec={interval_steps_per_sec:.3}",
    "  grid occupied={occupied_tiles} empty={empty_tiles} occupancy={occupancy_fraction:.6}",
    "  chemistry total={total_element_amount:.6} extracellular={extracellular_amounts:?} intracellular={intracellular_amounts:?} system={system_amounts:?}",
    "  cells live={live_cells} births={births} (+{interval_births}) deaths={deaths} (+{interval_deaths}) divisions={divisions} (+{interval_divisions}) avg_energy={avg_energy:.3} min_energy={min_energy:.3} max_energy={max_energy:.3} avg_age={avg_age:.3}s max_age={max_age:.3}s",
    "  enzymes avg={avg_enzyme_count:.2} min={min_enzyme_count} max={max_enzyme_count} cap={cells_at_enzyme_cap} cap_frac={fraction_at_enzyme_cap:.3} hist_1_10={enzyme_histogram_1_10:?} total={total_enzymes}",
    "  interaction exposure_mean={mean_exposure:.4} exposure_max={max_exposure:.3} exposed_cells={exposed_cells} contacts={contact_pairs} cross_contacts={cross_lineage_contact_pairs} field_extent={field_surface_extent:.6} cell_extent={cell_surface_extent:.6} digested={boundary_digested:.6} cross_digested={cross_lineage_boundary_digested:.6} lyses_after_digestion={lyses_after_digestion}",
    "  niche mixed_polarity={mixed_polarity_cells} optimum_spread={mean_enval_optimum_spread:.4} mismatch={mean_niche_mismatch:.4}",
    "  lineages extant={lineages} total_records={total_lineage_records} extinct={extinct_lineages} dominant={dominant_lineage} dominant_pop={dominant_lineage_population} dominant_share={dominant_lineage_share:.3} entropy={lineage_entropy:.3}",
    "  metabolism attempts={rx_attempts} (+{interval_reaction_attempts}) successes={rx_successes} (+{interval_reaction_successes}) no_substrate={rx_no_substrate} executed_flux={executed_metabolic_flux:.6} (+{interval_executed_metabolic_flux:.6}) uptake_flux={uptake_flux:.6} (+{interval_uptake_flux:.6}) secretion_flux={secretion_flux:.6} (+{interval_secretion_flux:.6}) energy_delta={rx_energy_delta_total:.3} enval_in={rx_enval_input_total:.3} enval_out={rx_enval_output_total:.3}",
    "  enval avg={enval_avg:.6} min={enval_min:.6} p05={enval_p05:.6} p50={enval_p50:.6} p95={enval_p95:.6} max={enval_max:.6} sd={enval_stddev:.6} pos={enval_positive_tiles} neg={enval_negative_tiles} near_zero={enval_near_zero_tiles}",
    "  energy_ledger founder={energy_founder:.6} injected={energy_injected:.6} extracted={energy_extracted:.6} chemical_harvest={energy_chemical_harvest:.6} chemical_cost={energy_chemical_cost:.6} enval_harvest={energy_enval_harvest:.6} pump_cost={energy_pump_cost:.6} maintenance={energy_maintenance:.6} death_loss={energy_death_loss:.6} residual={energy_ledger_residual:.3e} renewable_coverage={renewable_coverage:.6}",
    "  enval_ledger source_inflow={enval_source_inflow:.6} cell_uptake={enval_cell_uptake:.6} cell_emission={enval_cell_emission:.6} recharge={enval_recharge:.6} edits={enval_edits:.6} recharge_amount={recharge_amount:.6} recharge_energy={recharge_energy:.6}",
    "  interval ticks={interval_ticks} pop_delta={interval_pop_delta:+} cell_steps={interval_cell_steps} cell_steps_per_sec={interval_cell_steps_per_sec:.3} enzyme_attempts={interval_enzyme_attempts} enzyme_attempts_per_sec={interval_enzyme_attempts_per_sec:.3} reactions_per_sec={interval_reactions_per_sec:.3}",
];

fn field_named(name: &str) -> Option<&'static StatField> {
    STAT_FIELDS.iter().find(|field| field.name == name)
}

fn parse_format(spec: &str) -> Option<Format> {
    match spec {
        "" => Some(Format::Display),
        "+" => Some(Format::Signed),
        "?" => Some(Format::Debug),
        _ => {
            let digits = spec.strip_prefix('.')?;
            match digits.strip_suffix('e') {
                Some(digits) => digits.parse().ok().map(Format::Exp),
                None => digits.parse().ok().map(Format::Fixed),
            }
        }
    }
}

pub fn csv_header() -> String {
    let columns = STAT_FIELDS
        .iter()
        .filter(|field| field.csv.is_some())
        .map(|field| field.name)
        .collect::<Vec<_>>();
    columns.join(",")
}

pub fn csv_row(stats: &WorldStats, interval: &StatsInterval) -> String {
    let values = STAT_FIELDS
        .iter()
        .filter_map(|field| Some((field.get)(stats, interval).render(field.csv?)))
        .collect::<Vec<_>>();
    values.join(",")
}

pub fn json_record(stats: &WorldStats, interval: &StatsInterval) -> Json {
    let mut root = Map::new();
    for field in STAT_FIELDS {
        let Some(path) = field.json else {
            continue;
        };
        let mut node = &mut root;
        let mut keys = path.split('.').peekable();
        while let Some(key) = keys.next() {
            if keys.peek().is_none() {
                node.insert(key.to_owned(), (field.get)(stats, interval).json());
                break;
            }
            let child = node
                .entry(key.to_owned())
                .or_insert_with(|| Json::Object(Map::new()));
            node = child.as_object_mut().expect("JSON paths nest consistently");
        }
    }
    Json::Object(root)
}

pub fn render_text(template: &str, stats: &WorldStats, interval: &StatsInterval) -> String {
    let mut line = String::with_capacity(template.len() * 2);
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        line.push_str(&rest[..open]);
        let close = open + rest[open..].find('}').expect("closed text placeholder");
        let (name, spec) = rest[open + 1..close]
            .split_once(':')
            .unwrap_or((&rest[open + 1..close], ""));
        let field = field_named(name).expect("text placeholder names a stat field");
        let format = parse_format(spec).expect("valid text placeholder format");
        line.push_str(&(field.get)(stats, interval).render(format));
        rest = &rest[close + 1..];
    }
    line.push_str(rest);
    line
}

#[cfg(test)]
pub fn template_placeholders(template: &str) -> Vec<(&str, &str)> {
    template
        .split('{')
        .skip(1)
        .map(|part| {
            let inner = &part[..part.find('}').expect("closed placeholder")];
            inner.split_once(':').unwrap_or((inner, ""))
        })
        .collect()
}

#[cfg(test)]
pub fn format_is_valid(spec: &str) -> bool {
    parse_format(spec).is_some()
}

#[cfg(test)]
pub fn has_field(name: &str) -> bool {
    field_named(name).is_some()
}
