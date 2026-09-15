use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::chem::{ELEMENT_COUNT, ELEMENT_ORDER, Element, ElementAmounts};
use crate::rng::Rng;

pub const MIN_CELL_ENZYMES: usize = 1;
pub const MAX_CELL_ENZYMES: usize = 10;
pub const DEFAULT_COMBAT_LEVEL_MEAN: u32 = 100;
pub const DEFAULT_COMBAT_LEVEL_SIGMA: f64 = 10.0;
pub const COMBAT_LEVEL_MUTATION_STEP_MIN: u32 = 1;
pub const COMBAT_LEVEL_MUTATION_STEP_MAX: u32 = 4;
pub const GENOME_PATCH_SCHEMA: &str = "microcosm.genome_patch.v2";

const STOICHIOMETRY_EPSILON: f64 = 1.0e-6;
const EVOLVABLE_ENZYME_TYPES: [EnzymeType; 3] = [
    EnzymeType::Metabolic,
    EnzymeType::Attackase,
    EnzymeType::Defensase,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct LineageId(pub u64);

impl LineageId {
    pub const fn raw(self) -> u64 {
        self.0
    }
}

impl fmt::Display for LineageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EnzymeType {
    Metabolic,
    Defensase,
    Attackase,
}

impl EnzymeType {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Metabolic => "metabolic",
            Self::Defensase => "defensase",
            Self::Attackase => "attackase",
        }
    }

    pub const fn is_metabolic(self) -> bool {
        matches!(self, Self::Metabolic)
    }

    pub const fn is_combat(self) -> bool {
        matches!(self, Self::Defensase | Self::Attackase)
    }
}

impl std::str::FromStr for EnzymeType {
    type Err = GenomePatchError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "metabolic" => Ok(Self::Metabolic),
            "defensase" => Ok(Self::Defensase),
            "attackase" => Ok(Self::Attackase),
            other => Err(GenomePatchError::new(format!(
                "unsupported enzyme_type '{other}'"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Enzyme {
    pub enzyme_type: EnzymeType,
    pub reactants: ElementAmounts,
    pub products: ElementAmounts,
    pub rate: f32,
    pub energy_harvest_fraction: f32,
    pub secretion_fraction: f32,
    pub enval_sigma: f32,
    pub enval_throughput: f32,
    pub enval_energy_fraction: f32,
    pub enval_release_fraction: f32,
    pub enval_pump: f32,
    pub combat_level: u32,
}

impl Enzyme {
    pub fn metabolic(
        reactants: ElementAmounts,
        products: ElementAmounts,
        rate: f32,
        energy_harvest_fraction: f32,
        secretion_fraction: f32,
    ) -> Self {
        Self {
            enzyme_type: EnzymeType::Metabolic,
            reactants,
            products,
            rate,
            energy_harvest_fraction,
            secretion_fraction,
            enval_sigma: 0.18,
            enval_throughput: 0.12,
            enval_energy_fraction: 2.0 / 3.0,
            enval_release_fraction: 1.0 / 3.0,
            enval_pump: 0.3,
            combat_level: 0,
        }
    }

    pub fn founder_downhill() -> Self {
        Self::metabolic(
            ElementAmounts::new([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            ElementAmounts::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            0.85,
            0.65,
            0.08,
        )
    }

    pub fn founder_reshape() -> Self {
        Self::metabolic(
            ElementAmounts::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ElementAmounts::new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            0.32,
            0.25,
            0.12,
        )
    }

    pub fn defensase(level: u32) -> Self {
        Self::combat(EnzymeType::Defensase, level)
    }

    pub fn attackase(level: u32) -> Self {
        Self::combat(EnzymeType::Attackase, level)
    }

    pub fn random(enzyme_type: EnzymeType, rng: &mut Rng) -> Self {
        match enzyme_type {
            EnzymeType::Metabolic => Self::random_metabolic(rng),
            EnzymeType::Defensase | EnzymeType::Attackase => {
                let level = normalize_combat_level(rng.gaussian(
                    f64::from(DEFAULT_COMBAT_LEVEL_MEAN),
                    DEFAULT_COMBAT_LEVEL_SIGMA,
                ));
                Self::combat(enzyme_type, level)
            }
        }
    }

    pub fn edit_default(enzyme_type: EnzymeType) -> Self {
        match enzyme_type {
            EnzymeType::Metabolic => Self::founder_downhill(),
            EnzymeType::Defensase => Self::defensase(DEFAULT_COMBAT_LEVEL_MEAN),
            EnzymeType::Attackase => Self::attackase(DEFAULT_COMBAT_LEVEL_MEAN),
        }
    }

    pub fn validate(&self) -> Result<(), CatalystError> {
        if self.enzyme_type.is_combat() {
            if self.combat_level == 0 {
                return Err(CatalystError::new("combat_level must be >= 1"));
            }
            return Ok(());
        }

        self.reactants
            .validate_nonnegative("reactants")
            .map_err(|error| CatalystError::new(error.to_string()))?;
        self.products
            .validate_nonnegative("products")
            .map_err(|error| CatalystError::new(error.to_string()))?;
        for (name, value) in [
            ("rate", self.rate),
            ("energy_harvest_fraction", self.energy_harvest_fraction),
            ("secretion_fraction", self.secretion_fraction),
            ("enval_sigma", self.enval_sigma),
            ("enval_throughput", self.enval_throughput),
            ("enval_energy_fraction", self.enval_energy_fraction),
            ("enval_release_fraction", self.enval_release_fraction),
            ("enval_pump", self.enval_pump),
        ] {
            if !value.is_finite() {
                return Err(CatalystError::new(format!("{name} must be finite")));
            }
        }
        if self.rate <= 0.0 {
            return Err(CatalystError::new("rate must be > 0"));
        }
        if self.enval_sigma <= 0.0 {
            return Err(CatalystError::new("enval_sigma must be > 0"));
        }
        for (name, value) in [
            ("energy_harvest_fraction", self.energy_harvest_fraction),
            ("secretion_fraction", self.secretion_fraction),
            ("enval_energy_fraction", self.enval_energy_fraction),
            ("enval_release_fraction", self.enval_release_fraction),
        ] {
            if !(0.0..=1.0).contains(&value) {
                return Err(CatalystError::new(format!("{name} must be in [0, 1]")));
            }
        }
        if self.enval_throughput < 0.0 || self.enval_pump < 0.0 {
            return Err(CatalystError::new(
                "enval_throughput and enval_pump must be nonnegative",
            ));
        }
        if self.enval_energy_fraction + self.enval_release_fraction > 1.0 + 1.0e-6 {
            return Err(CatalystError::new(
                "enval energy and release fractions must sum to at most 1",
            ));
        }

        let reactant_total = self.reactants.total();
        let product_total = self.products.total();
        if reactant_total <= STOICHIOMETRY_EPSILON || product_total <= STOICHIOMETRY_EPSILON {
            return Err(CatalystError::new(
                "reactants and products must each have positive total stoichiometry",
            ));
        }
        let tolerance = STOICHIOMETRY_EPSILON * reactant_total.max(product_total).max(1.0);
        if (reactant_total - product_total).abs() > tolerance {
            return Err(CatalystError::new(
                "reactant and product scalar totals must be equal",
            ));
        }
        let distinct = ELEMENT_ORDER
            .iter()
            .any(|element| (self.reactants[*element] - self.products[*element]).abs() > 1.0e-6);
        if !distinct {
            return Err(CatalystError::new(
                "reactant and product stoichiometry must be distinct",
            ));
        }
        Ok(())
    }

    fn random_metabolic(rng: &mut Rng) -> Self {
        let source = ELEMENT_ORDER[rng.usize(ELEMENT_COUNT)];
        let mut target = ELEMENT_ORDER[rng.usize(ELEMENT_COUNT - 1)];
        if target.index() >= source.index() {
            target = ELEMENT_ORDER[target.index() + 1];
        }
        let mut reactants = ElementAmounts::ZERO;
        let mut products = ElementAmounts::ZERO;
        reactants[source] = 1.0;
        products[target] = 1.0;
        let mut enzyme = Self::metabolic(
            reactants,
            products,
            0.15 + rng.next_f32() * 0.85,
            rng.next_f32(),
            rng.next_f32() * 0.35,
        );
        enzyme.enval_sigma = 0.12 + rng.next_f32() * 0.18;
        enzyme.enval_throughput = 0.04 + rng.next_f32() * 0.18;
        enzyme.enval_pump = rng.next_f32() * 0.08;
        enzyme
    }

    fn combat(enzyme_type: EnzymeType, level: u32) -> Self {
        Self {
            enzyme_type,
            reactants: ElementAmounts::ZERO,
            products: ElementAmounts::ZERO,
            rate: 0.0,
            energy_harvest_fraction: 0.0,
            secretion_fraction: 0.0,
            enval_sigma: 0.0,
            enval_throughput: 0.0,
            enval_energy_fraction: 0.0,
            enval_release_fraction: 0.0,
            enval_pump: 0.0,
            combat_level: level.max(1),
        }
    }

    fn normalize_after_mutation(&mut self) {
        if self.enzyme_type.is_combat() {
            *self = Self::combat(self.enzyme_type, self.combat_level);
            return;
        }

        for element in ELEMENT_ORDER {
            if !self.reactants[element].is_finite() || self.reactants[element] < 0.0 {
                self.reactants[element] = 0.0;
            }
            if !self.products[element].is_finite() || self.products[element] < 0.0 {
                self.products[element] = 0.0;
            }
        }
        if self.reactants.total() <= STOICHIOMETRY_EPSILON {
            self.reactants[Element::D] = 1.0;
        }
        if self.products.total() <= STOICHIOMETRY_EPSILON {
            self.products[Element::A] = 1.0;
        }
        let reactant_total = self.reactants.total();
        let product_total = self.products.total();
        let scale = (reactant_total / product_total) as f32;
        for element in ELEMENT_ORDER {
            self.products[element] *= scale;
        }
        if ELEMENT_ORDER
            .iter()
            .all(|element| (self.reactants[*element] - self.products[*element]).abs() <= 1.0e-6)
        {
            let source = ELEMENT_ORDER
                .iter()
                .copied()
                .find(|element| self.products[*element] > 1.0e-4)
                .unwrap_or(Element::A);
            let target = ELEMENT_ORDER[(source.index() + 1) % ELEMENT_COUNT];
            let shift = self.products[source].min((reactant_total as f32 * 0.1).max(1.0e-4));
            self.products[source] -= shift;
            self.products[target] += shift;
        }
        self.rate = finite_or(self.rate, 0.4).max(1.0e-4);
        self.energy_harvest_fraction = finite_or(self.energy_harvest_fraction, 0.5).clamp(0.0, 1.0);
        self.secretion_fraction = finite_or(self.secretion_fraction, 0.1).clamp(0.0, 1.0);
        self.enval_sigma = finite_or(self.enval_sigma, 0.18).max(1.0e-4);
        self.enval_throughput = finite_or(self.enval_throughput, 0.12).max(0.0);
        self.enval_energy_fraction =
            finite_or(self.enval_energy_fraction, 2.0 / 3.0).clamp(0.0, 1.0);
        self.enval_release_fraction = finite_or(self.enval_release_fraction, 1.0 / 3.0)
            .clamp(0.0, 1.0 - self.enval_energy_fraction);
        self.enval_pump = finite_or(self.enval_pump, 0.3).max(0.0);
        self.combat_level = 0;
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CatalystError {
    message: String,
}

impl CatalystError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for CatalystError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for CatalystError {}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PredationEnzymeTransferStats {
    pub added: usize,
    pub replacements: usize,
    pub replacement_slots: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Genome {
    pub optimal_enval: f32,
    pub enzymes: Vec<Enzyme>,
    pub repro_threshold: f64,
    pub initial_energy: f64,
    pub decay_time: f64,
    pub mutation_rate: f32,
    pub post_divide_mortality: f32,
    pub desired_element_reserve: f32,
    pub enval_stress_factor: f64,
    pub enval_mutation_floor: f32,
    pub maintenance_cost_per_sec: f64,
    pub lineage_id: LineageId,
}

impl Genome {
    pub fn random_founder(rng: &mut Rng, average_enval: f32) -> Self {
        let optimal_enval = average_enval + (rng.next_f32() - 0.5) * 0.20;
        let defensase_level = normalize_combat_level(rng.gaussian(
            f64::from(DEFAULT_COMBAT_LEVEL_MEAN),
            DEFAULT_COMBAT_LEVEL_SIGMA,
        ));
        let mut genome = Self {
            optimal_enval,
            enzymes: vec![
                Enzyme::founder_downhill(),
                Enzyme::founder_reshape(),
                Enzyme::defensase(defensase_level),
            ],
            repro_threshold: 2.0 + rng.next_f64() * 6.0,
            initial_energy: 1.2 + rng.next_f64() * 1.8,
            decay_time: 700.0 + rng.next_f64() * 2000.0,
            mutation_rate: 0.06,
            post_divide_mortality: 0.0,
            desired_element_reserve: 2.0,
            enval_stress_factor: 0.02,
            enval_mutation_floor: 0.03,
            maintenance_cost_per_sec: 0.05,
            lineage_id: LineageId(rng.usize(1_000_000_000) as u64),
        };
        genome.enforce_enzyme_count_bounds(rng);
        genome
    }

    pub fn mutate(&self, rng: &mut Rng, reference_enval: f32) -> Self {
        let mut genome = self.clone();
        let mutation_rate = genome.mutation_rate;
        let reference_enval = if reference_enval.is_finite() {
            reference_enval
        } else {
            0.0
        };
        genome.enforce_enzyme_count_bounds(rng);

        if rng.chance(mutation_rate) {
            genome.repro_threshold =
                (genome.repro_threshold * (1.0 + (rng.next_f64() - 0.5) * 0.2)).max(0.1);
        }
        if rng.chance(mutation_rate) {
            genome.decay_time = (genome.decay_time * (1.0 + (rng.next_f64() - 0.5) * 0.2))
                .round()
                .max(50.0);
        }
        if rng.chance(mutation_rate) {
            genome.desired_element_reserve =
                (genome.desired_element_reserve * (1.0 + (rng.next_f32() - 0.5) * 0.25)).max(0.0);
        }
        if rng.chance(mutation_rate * 0.5) {
            genome.enval_stress_factor =
                (genome.enval_stress_factor * (1.0 + (rng.next_f64() - 0.5) * 0.3)).max(0.001);
        }
        genome.optimal_enval = mutate_optimal_enval(
            genome.optimal_enval,
            reference_enval,
            genome.enval_mutation_floor,
            rng,
        );

        for enzyme in &mut genome.enzymes {
            if rng.chance(mutation_rate * 0.2) {
                let next_type = EVOLVABLE_ENZYME_TYPES[rng.usize(EVOLVABLE_ENZYME_TYPES.len())];
                *enzyme = Enzyme::random(next_type, rng);
            } else if enzyme.enzyme_type.is_combat() {
                if rng.chance(mutation_rate) {
                    enzyme.combat_level = mutate_combat_level(enzyme.combat_level, rng);
                }
            } else {
                if rng.chance(mutation_rate) {
                    let element = ELEMENT_ORDER[rng.usize(ELEMENT_COUNT)];
                    enzyme.reactants[element] = (enzyme.reactants[element]
                        * (1.0 + (rng.next_f32() - 0.5) * 0.5)
                        + rng.next_f32() * 0.05)
                        .max(0.0);
                }
                if rng.chance(mutation_rate) {
                    let element = ELEMENT_ORDER[rng.usize(ELEMENT_COUNT)];
                    enzyme.products[element] = (enzyme.products[element]
                        * (1.0 + (rng.next_f32() - 0.5) * 0.5)
                        + rng.next_f32() * 0.05)
                        .max(0.0);
                }
                if rng.chance(mutation_rate) {
                    enzyme.rate = (enzyme.rate * (1.0 + (rng.next_f32() - 0.5) * 0.4)).max(1.0e-4);
                }
                if rng.chance(mutation_rate) {
                    enzyme.energy_harvest_fraction = (enzyme.energy_harvest_fraction
                        + (rng.next_f32() - 0.5) * 0.12)
                        .clamp(0.0, 1.0);
                }
                if rng.chance(mutation_rate) {
                    enzyme.secretion_fraction =
                        (enzyme.secretion_fraction + (rng.next_f32() - 0.5) * 0.12).clamp(0.0, 1.0);
                }
                if rng.chance(mutation_rate) {
                    enzyme.enval_sigma =
                        (enzyme.enval_sigma * (1.0 + (rng.next_f32() - 0.5) * 0.35)).max(0.02);
                }
                if rng.chance(mutation_rate) {
                    enzyme.enval_throughput =
                        (enzyme.enval_throughput * (1.0 + (rng.next_f32() - 0.5) * 0.4)).max(0.0);
                }
                if rng.chance(mutation_rate) {
                    enzyme.enval_energy_fraction = (enzyme.enval_energy_fraction
                        + (rng.next_f32() - 0.5) * 0.1)
                        .clamp(0.0, 1.0);
                }
                if rng.chance(mutation_rate) {
                    enzyme.enval_release_fraction = (enzyme.enval_release_fraction
                        + (rng.next_f32() - 0.5) * 0.1)
                        .clamp(0.0, 1.0);
                }
                if rng.chance(mutation_rate) {
                    enzyme.enval_pump =
                        (enzyme.enval_pump * (1.0 + (rng.next_f32() - 0.5) * 0.4)).max(0.0);
                }
                enzyme.normalize_after_mutation();
            }
        }

        if rng.chance(mutation_rate * 0.3) && genome.enzymes.len() > MIN_CELL_ENZYMES {
            let index = rng.usize(genome.enzymes.len());
            genome.enzymes.swap_remove(index);
        } else if rng.chance(mutation_rate * 0.3) && genome.enzymes.len() < MAX_CELL_ENZYMES {
            let enzyme_type = EVOLVABLE_ENZYME_TYPES[rng.usize(EVOLVABLE_ENZYME_TYPES.len())];
            genome.enzymes.push(Enzyme::random(enzyme_type, rng));
        }

        genome.enforce_enzyme_count_bounds(rng);
        genome
    }

    pub fn absorb_predation_enzymes(
        &mut self,
        prey_enzymes: &[Enzyme],
        rng: &mut Rng,
    ) -> PredationEnzymeTransferStats {
        let mut stats = PredationEnzymeTransferStats::default();
        if prey_enzymes.is_empty() {
            self.enforce_enzyme_count_bounds(rng);
            return stats;
        }
        self.enforce_enzyme_count_bounds(rng);
        let free_slots = MAX_CELL_ENZYMES.saturating_sub(self.enzymes.len());
        if free_slots > 0 {
            if prey_enzymes.len() <= free_slots {
                self.enzymes.extend_from_slice(prey_enzymes);
                stats.added = prey_enzymes.len();
            } else {
                let selected = select_random_indices(prey_enzymes.len(), free_slots, rng);
                for index in selected {
                    self.enzymes.push(prey_enzymes[index]);
                    stats.added += 1;
                }
            }
            self.enforce_enzyme_count_bounds(rng);
            return stats;
        }

        let replacement_order = select_random_indices(self.enzymes.len(), self.enzymes.len(), rng);
        let mut replacement_index = 0;
        for enzyme in prey_enzymes {
            if replacement_index >= replacement_order.len() {
                break;
            }
            if !rng.chance(0.25) {
                continue;
            }
            let slot = replacement_order[replacement_index];
            replacement_index += 1;
            self.enzymes[slot] = *enzyme;
            stats.replacements += 1;
            stats.replacement_slots.push(slot);
        }
        self.enforce_enzyme_count_bounds(rng);
        stats
    }

    pub fn enforce_enzyme_count_bounds(&mut self, rng: &mut Rng) {
        if self.enzymes.len() > MAX_CELL_ENZYMES {
            self.enzymes.truncate(MAX_CELL_ENZYMES);
        }
        if self.enzymes.is_empty() {
            self.enzymes
                .push(Enzyme::random(EnzymeType::Metabolic, rng));
        }
        for enzyme in &mut self.enzymes {
            enzyme.normalize_after_mutation();
        }
    }

    pub fn validate(&self) -> Result<(), CatalystError> {
        if self.enzymes.len() < MIN_CELL_ENZYMES || self.enzymes.len() > MAX_CELL_ENZYMES {
            return Err(CatalystError::new("enzyme count is outside allowed bounds"));
        }
        for enzyme in &self.enzymes {
            enzyme.validate()?;
        }
        Ok(())
    }

    pub fn attack_total(&self) -> u32 {
        self.enzymes
            .iter()
            .filter(|enzyme| enzyme.enzyme_type == EnzymeType::Attackase)
            .map(|enzyme| enzyme.combat_level.max(1))
            .sum()
    }

    pub fn defense_total(&self) -> u32 {
        self.enzymes
            .iter()
            .filter(|enzyme| enzyme.enzyme_type == EnzymeType::Defensase)
            .map(|enzyme| enzyme.combat_level.max(1))
            .sum()
    }

    pub fn refresh_combat_totals(&mut self) {
        let _ = self.attack_total();
        let _ = self.defense_total();
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenomePatchError {
    message: String,
}

impl GenomePatchError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for GenomePatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for GenomePatchError {}

#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenomePatch {
    pub schema: Option<String>,
    pub genome: Option<GenomeFieldPatch>,
    #[serde(default)]
    pub enzymes: Vec<EnzymePatchOperation>,
}

impl GenomePatch {
    pub fn apply_to_genome(&self, genome: &mut Genome) -> Result<Vec<String>, GenomePatchError> {
        if let Some(schema) = &self.schema
            && schema != GENOME_PATCH_SCHEMA
        {
            return Err(GenomePatchError::new(format!(
                "unsupported genome patch schema '{schema}'"
            )));
        }
        let mut changed_fields = Vec::new();
        if let Some(fields) = &self.genome {
            fields.apply_to_genome(genome, &mut changed_fields)?;
        }
        for operation in &self.enzymes {
            operation.apply_to_enzymes(&mut genome.enzymes, &mut changed_fields)?;
        }
        genome
            .validate()
            .map_err(|error| GenomePatchError::new(error.to_string()))?;
        Ok(changed_fields)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenomeFieldPatch {
    pub optimal_enval: Option<f32>,
    pub repro_threshold: Option<f64>,
    pub decay_time: Option<f64>,
    pub mutation_rate: Option<f32>,
    pub post_divide_mortality: Option<f32>,
    pub desired_element_reserve: Option<f32>,
    pub enval_stress_factor: Option<f64>,
    pub enval_mutation_floor: Option<f32>,
    pub maintenance_cost_per_sec: Option<f64>,
}

impl GenomeFieldPatch {
    fn apply_to_genome(
        &self,
        genome: &mut Genome,
        changed_fields: &mut Vec<String>,
    ) -> Result<(), GenomePatchError> {
        macro_rules! set_field {
            ($field:ident, $validator:ident) => {
                if let Some(value) = self.$field {
                    genome.$field = $validator(stringify!($field), value)?;
                    changed_fields.push(stringify!($field).to_owned());
                }
            };
        }
        set_field!(optimal_enval, finite_f32);
        set_field!(repro_threshold, positive_f64);
        set_field!(decay_time, positive_f64);
        set_field!(mutation_rate, probability_f32);
        set_field!(post_divide_mortality, probability_f32);
        set_field!(desired_element_reserve, nonnegative_f32);
        set_field!(enval_stress_factor, nonnegative_f64);
        set_field!(enval_mutation_floor, nonnegative_f32);
        set_field!(maintenance_cost_per_sec, nonnegative_f64);
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnzymePatchOperation {
    pub op: String,
    pub index: Option<usize>,
    pub fields: Option<EnzymeFieldPatch>,
    pub enzyme: Option<EnzymeFieldPatch>,
}

impl EnzymePatchOperation {
    fn apply_to_enzymes(
        &self,
        enzymes: &mut Vec<Enzyme>,
        changed_fields: &mut Vec<String>,
    ) -> Result<(), GenomePatchError> {
        match self.op.as_str() {
            "update" => {
                let index = self.required_index()?;
                let fields = self.fields.as_ref().ok_or_else(|| {
                    GenomePatchError::new("enzyme update operation requires fields")
                })?;
                let enzyme = enzymes
                    .get_mut(index)
                    .ok_or_else(|| GenomePatchError::new(format!("no enzyme at index {index}")))?;
                fields.apply_to_enzyme(enzyme, &format!("enzyme[{index}]"), changed_fields)
            }
            "append" => {
                if enzymes.len() >= MAX_CELL_ENZYMES {
                    return Err(GenomePatchError::new(format!(
                        "cannot append enzyme: max enzyme count is {MAX_CELL_ENZYMES}"
                    )));
                }
                let fields = self
                    .enzyme
                    .as_ref()
                    .or(self.fields.as_ref())
                    .ok_or_else(|| {
                        GenomePatchError::new("enzyme append operation requires enzyme fields")
                    })?;
                let index = enzymes.len();
                let mut enzyme = fields.build_new_enzyme()?;
                fields.apply_to_enzyme(&mut enzyme, &format!("enzyme[{index}]"), changed_fields)?;
                enzymes.push(enzyme);
                changed_fields.push(format!("enzyme[{index}].append"));
                Ok(())
            }
            "replace" => {
                let index = self.required_index()?;
                let fields = self
                    .enzyme
                    .as_ref()
                    .or(self.fields.as_ref())
                    .ok_or_else(|| {
                        GenomePatchError::new("enzyme replace operation requires enzyme fields")
                    })?;
                if index >= enzymes.len() {
                    return Err(GenomePatchError::new(format!("no enzyme at index {index}")));
                }
                let mut enzyme = fields.build_new_enzyme()?;
                fields.apply_to_enzyme(&mut enzyme, &format!("enzyme[{index}]"), changed_fields)?;
                enzymes[index] = enzyme;
                changed_fields.push(format!("enzyme[{index}].replace"));
                Ok(())
            }
            "remove" => {
                let index = self.required_index()?;
                if enzymes.len() <= MIN_CELL_ENZYMES {
                    return Err(GenomePatchError::new(format!(
                        "cannot remove enzyme: min enzyme count is {MIN_CELL_ENZYMES}"
                    )));
                }
                if index >= enzymes.len() {
                    return Err(GenomePatchError::new(format!("no enzyme at index {index}")));
                }
                enzymes.remove(index);
                changed_fields.push(format!("enzyme[{index}].remove"));
                Ok(())
            }
            other => Err(GenomePatchError::new(format!(
                "unsupported enzyme patch op '{other}'"
            ))),
        }
    }

    fn required_index(&self) -> Result<usize, GenomePatchError> {
        self.index.ok_or_else(|| {
            GenomePatchError::new(format!("enzyme {} operation requires index", self.op))
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnzymeFieldPatch {
    pub enzyme_type: Option<String>,
    pub reactants: Option<[f32; ELEMENT_COUNT]>,
    pub products: Option<[f32; ELEMENT_COUNT]>,
    pub rate: Option<f32>,
    pub energy_harvest_fraction: Option<f32>,
    pub secretion_fraction: Option<f32>,
    pub enval_sigma: Option<f32>,
    pub enval_throughput: Option<f32>,
    pub enval_energy_fraction: Option<f32>,
    pub enval_release_fraction: Option<f32>,
    pub enval_pump: Option<f32>,
    pub combat_level: Option<u32>,
}

impl EnzymeFieldPatch {
    fn build_new_enzyme(&self) -> Result<Enzyme, GenomePatchError> {
        let enzyme_type = self
            .enzyme_type
            .as_deref()
            .ok_or_else(|| GenomePatchError::new("new enzyme operations require enzyme_type"))?
            .parse::<EnzymeType>()?;
        Ok(Enzyme::edit_default(enzyme_type))
    }

    fn apply_to_enzyme(
        &self,
        enzyme: &mut Enzyme,
        label: &str,
        changed_fields: &mut Vec<String>,
    ) -> Result<(), GenomePatchError> {
        if let Some(value) = &self.enzyme_type {
            let enzyme_type = value.parse::<EnzymeType>()?;
            if enzyme.enzyme_type != enzyme_type {
                *enzyme = Enzyme::edit_default(enzyme_type);
            }
            changed_fields.push(format!("{label}.enzyme_type"));
        }
        if let Some(values) = self.reactants {
            enzyme.reactants = patch_element_amounts(&format!("{label}.reactants"), values)?;
            changed_fields.push(format!("{label}.reactants"));
        }
        if let Some(values) = self.products {
            enzyme.products = patch_element_amounts(&format!("{label}.products"), values)?;
            changed_fields.push(format!("{label}.products"));
        }
        macro_rules! set_enzyme_field {
            ($field:ident, $validator:ident) => {
                if let Some(value) = self.$field {
                    enzyme.$field = $validator(&format!("{label}.{}", stringify!($field)), value)?;
                    changed_fields.push(format!("{label}.{}", stringify!($field)));
                }
            };
        }
        set_enzyme_field!(rate, positive_f32);
        set_enzyme_field!(energy_harvest_fraction, probability_f32);
        set_enzyme_field!(secretion_fraction, probability_f32);
        set_enzyme_field!(enval_sigma, positive_f32);
        set_enzyme_field!(enval_throughput, nonnegative_f32);
        set_enzyme_field!(enval_energy_fraction, probability_f32);
        set_enzyme_field!(enval_release_fraction, probability_f32);
        set_enzyme_field!(enval_pump, nonnegative_f32);
        if let Some(value) = self.combat_level {
            if value == 0 {
                return Err(GenomePatchError::new(format!(
                    "{label}.combat_level must be >= 1"
                )));
            }
            enzyme.combat_level = value;
            changed_fields.push(format!("{label}.combat_level"));
        }
        enzyme
            .validate()
            .map_err(|error| GenomePatchError::new(format!("{label}: {error}")))
    }
}

fn patch_element_amounts(
    name: &str,
    values: [f32; ELEMENT_COUNT],
) -> Result<ElementAmounts, GenomePatchError> {
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() || *value < 0.0 {
            return Err(GenomePatchError::new(format!(
                "{name}[{index}] must be finite and nonnegative"
            )));
        }
    }
    Ok(ElementAmounts::new(values))
}

fn finite_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() { value } else { fallback }
}

fn finite_f32(name: &str, value: f32) -> Result<f32, GenomePatchError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(GenomePatchError::new(format!("{name} must be finite")))
    }
}

fn finite_f64(name: &str, value: f64) -> Result<f64, GenomePatchError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(GenomePatchError::new(format!("{name} must be finite")))
    }
}

fn positive_f32(name: &str, value: f32) -> Result<f32, GenomePatchError> {
    let value = finite_f32(name, value)?;
    if value > 0.0 {
        Ok(value)
    } else {
        Err(GenomePatchError::new(format!("{name} must be > 0")))
    }
}

fn positive_f64(name: &str, value: f64) -> Result<f64, GenomePatchError> {
    let value = finite_f64(name, value)?;
    if value > 0.0 {
        Ok(value)
    } else {
        Err(GenomePatchError::new(format!("{name} must be > 0")))
    }
}

fn nonnegative_f32(name: &str, value: f32) -> Result<f32, GenomePatchError> {
    let value = finite_f32(name, value)?;
    if value >= 0.0 {
        Ok(value)
    } else {
        Err(GenomePatchError::new(format!("{name} must be >= 0")))
    }
}

fn nonnegative_f64(name: &str, value: f64) -> Result<f64, GenomePatchError> {
    let value = finite_f64(name, value)?;
    if value >= 0.0 {
        Ok(value)
    } else {
        Err(GenomePatchError::new(format!("{name} must be >= 0")))
    }
}

fn probability_f32(name: &str, value: f32) -> Result<f32, GenomePatchError> {
    let value = finite_f32(name, value)?;
    if (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err(GenomePatchError::new(format!("{name} must be in [0, 1]")))
    }
}

fn select_random_indices(count: usize, limit: usize, rng: &mut Rng) -> Vec<usize> {
    let capped = limit.min(count);
    let mut indices = (0..count).collect::<Vec<_>>();
    for i in 0..capped {
        let swap_index = i + rng.usize(count - i);
        indices.swap(i, swap_index);
    }
    indices.truncate(capped);
    indices
}

pub fn normalize_combat_level(value: f64) -> u32 {
    let numeric = if value.is_finite() {
        value
    } else {
        f64::from(DEFAULT_COMBAT_LEVEL_MEAN)
    };
    numeric.round().max(1.0) as u32
}

pub fn mutate_combat_level(level: u32, rng: &mut Rng) -> u32 {
    let current = level.max(1);
    let step = COMBAT_LEVEL_MUTATION_STEP_MIN
        + rng.usize((COMBAT_LEVEL_MUTATION_STEP_MAX - COMBAT_LEVEL_MUTATION_STEP_MIN + 1) as usize)
            as u32;
    let branch = if rng.chance(0.5) {
        0_i32
    } else if rng.chance(0.5) {
        -1
    } else {
        1
    };
    if branch == 0 {
        current
    } else {
        ((i64::from(current) + i64::from(branch) * i64::from(step)).max(1)) as u32
    }
}

pub fn mutate_optimal_enval(parent: f32, reference: f32, floor: f32, rng: &mut Rng) -> f32 {
    let midpoint = (parent + reference) * 0.5;
    let mut step = (parent - midpoint).abs();
    let mut toward_sign = (reference - parent).signum();
    if step < floor {
        step = floor;
        if toward_sign == 0.0 {
            toward_sign = if rng.chance(0.5) { -1.0 } else { 1.0 };
        }
    }
    let r = rng.next_f32();
    if r < 0.5 {
        parent
    } else if r < 0.75 {
        parent + toward_sign * step
    } else {
        parent - toward_sign * step
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Enzyme, EnzymeFieldPatch, EnzymePatchOperation, EnzymeType, GENOME_PATCH_SCHEMA, Genome,
        GenomePatch, MAX_CELL_ENZYMES, MIN_CELL_ENZYMES,
    };
    use crate::rng::Rng;

    #[test]
    fn founder_genome_has_valid_distinct_metabolic_pathways_and_combat_scaffold() {
        let mut rng = Rng::from_seed_str("founder-genome");
        let genome = Genome::random_founder(&mut rng, 0.1);
        assert_eq!(genome.enzymes.len(), 3);
        assert_eq!(genome.enzymes[0].enzyme_type, EnzymeType::Metabolic);
        assert_eq!(genome.enzymes[1].enzyme_type, EnzymeType::Metabolic);
        assert_eq!(genome.enzymes[2].enzyme_type, EnzymeType::Defensase);
        assert!(
            genome.enzymes[0].reactants.intrinsic_energy()
                > genome.enzymes[0].products.intrinsic_energy()
        );
        assert_ne!(genome.enzymes[0].reactants, genome.enzymes[1].reactants);
        genome.validate().unwrap();
        assert_eq!(genome.attack_total(), 0);
        assert!(genome.defense_total() > 0);
    }

    #[test]
    fn catalyst_validation_rejects_unbalanced_empty_and_noop_stoichiometry() {
        let mut enzyme = Enzyme::founder_downhill();
        enzyme.products[crate::chem::Element::A] = 2.0;
        assert!(enzyme.validate().is_err());
        enzyme.reactants = crate::chem::ElementAmounts::ZERO;
        enzyme.products = crate::chem::ElementAmounts::ZERO;
        assert!(enzyme.validate().is_err());
        enzyme.reactants = crate::chem::ElementAmounts::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        enzyme.products = enzyme.reactants;
        assert!(enzyme.validate().is_err());
    }

    #[test]
    fn mutation_is_deterministic_and_preserves_validity_and_count_bounds() {
        let mut founder_rng = Rng::from_seed_str("mutation-parent");
        let mut parent = Genome::random_founder(&mut founder_rng, 0.0);
        parent.mutation_rate = 1.0;
        let mut first_rng = Rng::from_seed_str("mutation-child");
        let mut second_rng = Rng::from_seed_str("mutation-child");
        for _ in 0..100 {
            let first = parent.mutate(&mut first_rng, 0.25);
            let second = parent.mutate(&mut second_rng, 0.25);
            assert_eq!(first, second);
            assert!((MIN_CELL_ENZYMES..=MAX_CELL_ENZYMES).contains(&first.enzymes.len()));
            first.validate().unwrap();
            parent = first;
        }
    }

    fn genome_with_enzymes(count: usize, start_level: u32) -> Genome {
        let mut rng = Rng::from_seed_str("enzyme-transfer-genome");
        let mut genome = Genome::random_founder(&mut rng, 0.0);
        genome.enzymes.clear();
        for index in 0..count {
            let level = start_level + index as u32;
            genome.enzymes.push(if index % 2 == 0 {
                Enzyme::attackase(level)
            } else {
                Enzyme::defensase(level)
            });
        }
        genome
    }

    #[test]
    fn predation_enzyme_transfer_adds_all_in_order_without_consuming_rng_when_space_allows() {
        let mut predator = genome_with_enzymes(4, 10);
        let prey = genome_with_enzymes(3, 100);
        let original_len = predator.enzymes.len();
        let mut rng = Rng::from_seed_str("transfer-all");
        let mut untouched_rng = rng.clone();

        let stats = predator.absorb_predation_enzymes(&prey.enzymes, &mut rng);

        assert_eq!(predator.enzymes.len(), 7);
        assert_eq!(stats.added, 3);
        assert_eq!(stats.replacements, 0);
        assert_eq!(&predator.enzymes[original_len..], prey.enzymes.as_slice());
        assert_eq!(rng.next_f64().to_bits(), untouched_rng.next_f64().to_bits());
    }

    #[test]
    fn predation_enzyme_transfer_randomly_fills_partial_free_slots() {
        fn run_once(predator_count: usize, prey_count: usize, seed: &str) -> (Genome, usize) {
            let mut predator = genome_with_enzymes(predator_count, 10);
            let prey = genome_with_enzymes(prey_count, 100);
            let mut rng = Rng::from_seed_str(seed);
            let stats = predator.absorb_predation_enzymes(&prey.enzymes, &mut rng);
            (predator, stats.added)
        }

        let (first, first_added) = run_once(7, 4, "transfer-partial-a");
        let (first_repeat, first_repeat_added) = run_once(7, 4, "transfer-partial-a");
        assert_eq!(first, first_repeat);
        assert_eq!(first_added, 3);
        assert_eq!(first_repeat_added, 3);
        assert_eq!(first.enzymes.len(), MAX_CELL_ENZYMES);

        let (second, second_added) = run_once(4, 7, "transfer-partial-b");
        assert_eq!(second_added, 6);
        assert_eq!(second.enzymes.len(), MAX_CELL_ENZYMES);
    }

    #[test]
    fn predation_enzyme_replacement_is_deterministic_and_uses_unique_slots() {
        fn run_once() -> (Genome, Vec<usize>, usize) {
            let mut predator = genome_with_enzymes(MAX_CELL_ENZYMES, 10);
            let prey = genome_with_enzymes(7, 200);
            let mut rng = Rng::from_seed_str("replacement-determinism");
            let stats = predator.absorb_predation_enzymes(&prey.enzymes, &mut rng);
            (predator, stats.replacement_slots, stats.replacements)
        }

        let (first, first_slots, first_replacements) = run_once();
        let (second, second_slots, second_replacements) = run_once();
        assert_eq!(first, second);
        assert_eq!(first_slots, second_slots);
        assert_eq!(first_replacements, second_replacements);
        assert!(first_replacements > 0);
        assert_eq!(first.enzymes.len(), MAX_CELL_ENZYMES);

        let mut unique_slots = first_slots.clone();
        unique_slots.sort_unstable();
        unique_slots.dedup();
        assert_eq!(unique_slots.len(), first_slots.len());
    }

    #[test]
    fn metabolic_enval_pump_uses_the_previous_default_and_fallback() {
        let mut enzyme = Enzyme::founder_downhill();
        assert_eq!(enzyme.enval_pump, 0.3);
        enzyme.enval_pump = f32::NAN;
        enzyme.normalize_after_mutation();
        assert_eq!(enzyme.enval_pump, 0.3);
    }

    #[test]
    fn v2_patch_accepts_generic_metabolic_fields_and_rejects_v1() {
        let mut rng = Rng::from_seed_str("patch-v2");
        let mut genome = Genome::random_founder(&mut rng, 0.0);
        let patch = GenomePatch {
            schema: Some(GENOME_PATCH_SCHEMA.to_owned()),
            genome: None,
            enzymes: vec![EnzymePatchOperation {
                op: "update".to_owned(),
                index: Some(0),
                fields: Some(EnzymeFieldPatch {
                    reactants: Some([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
                    products: Some([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
                    rate: Some(0.5),
                    energy_harvest_fraction: Some(0.4),
                    secretion_fraction: Some(0.2),
                    ..EnzymeFieldPatch::default()
                }),
                enzyme: None,
            }],
        };
        patch.apply_to_genome(&mut genome).unwrap();
        genome.validate().unwrap();

        let old = GenomePatch {
            schema: Some("microcosm.genome_patch.v1".to_owned()),
            genome: None,
            enzymes: Vec::new(),
        };
        assert!(old.apply_to_genome(&mut genome).is_err());
    }
}
