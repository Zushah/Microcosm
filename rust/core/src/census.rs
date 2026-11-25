use serde::Serialize;

use crate::chem::{ELEMENT_COUNT, Element};
use crate::stats::{EnergyLedger, EnvalLedger};

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Spread {
    pub count: usize,
    pub min: f64,
    pub p05: f64,
    pub p50: f64,
    pub p95: f64,
    pub max: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GeneSpread {
    pub name: &'static str,
    pub spread: Option<Spread>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ReactionKind {
    pub reactant: Option<Element>,
    pub product: Option<Element>,
    pub coupling: i8,
    pub enzymes: usize,
    pub cells: usize,
    pub gene_medians: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Census {
    pub tick: u64,
    pub sim_time_seconds: f64,

    pub live_cells: usize,
    pub births: u64,
    pub deaths: u64,
    pub fissions: u64,
    pub energy_deaths: u64,
    pub lyses: u64,
    pub extant_lineages: usize,
    pub dominant_lineage_share: f64,
    pub generation: Option<Spread>,

    pub enzyme_genes: Vec<GeneSpread>,
    pub genome_genes: Vec<GeneSpread>,

    pub enzymes_per_genome: Option<Spread>,
    pub stoichiometric_entries_per_enzyme: Option<Spread>,
    pub zero_exposure_enzyme_fraction: f64,
    pub exposed_cell_fraction: f64,

    pub reaction_kinds: Vec<ReactionKind>,

    pub field_mean_per_tile: [f64; ELEMENT_COUNT],
    pub in_cells: [f64; ELEMENT_COUNT],
    pub total_matter: f64,
    pub initial_matter: f64,
    pub matter_drift: f64,

    pub enval_mean: f64,
    pub enval_std_dev: f64,
    pub enval_p05: f64,
    pub enval_p50: f64,
    pub enval_p95: f64,
    pub positive_enval_fraction: f64,
    pub negative_enval_fraction: f64,
    pub landscape_std_dev: f64,
    pub enval_flattening: f64,

    pub energy_ledger: EnergyLedger,
    pub enval_ledger: EnvalLedger,
    pub energy_residual: f64,
    pub enval_residual: f64,
    pub renewable_coverage: f64,
}
