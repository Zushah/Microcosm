use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::chem::{ELEMENT_COUNT, ELEMENT_ORDER, Element, ElementAmounts};
use crate::rng::Rng;

pub const MIN_CELL_ENZYMES: usize = 1;
pub const MAX_CELL_ENZYMES: usize = 10;
pub const GENOME_PATCH_SCHEMA: &str = "microcosm.genome_patch.v5";
pub const DEFAULT_HALF_SATURATION: f32 = 0.1;
pub const FOUNDER_ENGINE_RATE: f32 = 0.8;
pub const FOUNDER_ENGINE_ENVAL_THROUGHPUT: f32 = 0.18;
const FOUNDER_ENGINE_HARVEST_FRACTION: f32 = 0.65;
const MIN_HALF_SATURATION: f32 = 1.0e-3;

const STOICHIOMETRY_EPSILON: f64 = 1.0e-6;

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

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Enzyme {
    pub reactants: ElementAmounts,
    pub products: ElementAmounts,
    pub rate: f32,
    pub half_saturation: f32,
    pub energy_harvest_fraction: f32,
    pub secretion_fraction: f32,
    pub body_fraction: f32,
    pub boundary_fraction: f32,
    pub enval_sigma: f32,
    pub enval_throughput: f32,
    pub enval_energy_fraction: f32,
    pub enval_release_fraction: f32,
    pub enval_pump: f32,
    pub exposure: f32,
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
            reactants,
            products,
            rate,
            half_saturation: DEFAULT_HALF_SATURATION,
            energy_harvest_fraction,
            secretion_fraction,
            body_fraction: 0.0,
            boundary_fraction: 0.0,
            enval_sigma: 0.18,
            enval_throughput: 0.12,
            enval_energy_fraction: 2.0 / 3.0,
            enval_release_fraction: 1.0 / 3.0,
            enval_pump: 0.3,
            exposure: 0.0,
        }
    }

    pub fn founder_downhill() -> Self {
        let mut enzyme = Self::metabolic(
            ElementAmounts::new([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            ElementAmounts::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            0.85,
            0.65,
            0.08,
        );
        enzyme.body_fraction = 0.2;
        enzyme.boundary_fraction = 0.2;
        enzyme
    }

    pub fn founder_engine_forward() -> Self {
        Self::founder_engine(
            ElementAmounts::new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            ElementAmounts::new([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        )
    }

    pub fn founder_engine_reverse() -> Self {
        Self::founder_engine(
            ElementAmounts::new([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            ElementAmounts::new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        )
    }

    fn founder_engine(reactants: ElementAmounts, products: ElementAmounts) -> Self {
        let mut enzyme = Self::metabolic(
            reactants,
            products,
            FOUNDER_ENGINE_RATE,
            FOUNDER_ENGINE_HARVEST_FRACTION,
            0.0,
        );
        enzyme.body_fraction = 0.02;
        enzyme.boundary_fraction = 0.02;
        enzyme.enval_throughput = FOUNDER_ENGINE_ENVAL_THROUGHPUT;
        enzyme.enval_pump = 0.0;
        enzyme
    }

    pub fn random(rng: &mut Rng) -> Self {
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

    pub fn validate(&self) -> Result<(), CatalystError> {
        self.reactants
            .validate_nonnegative("reactants")
            .map_err(|error| CatalystError::new(error.to_string()))?;
        self.products
            .validate_nonnegative("products")
            .map_err(|error| CatalystError::new(error.to_string()))?;
        for (name, value) in [
            ("rate", self.rate),
            ("half_saturation", self.half_saturation),
            ("energy_harvest_fraction", self.energy_harvest_fraction),
            ("secretion_fraction", self.secretion_fraction),
            ("body_fraction", self.body_fraction),
            ("boundary_fraction", self.boundary_fraction),
            ("enval_sigma", self.enval_sigma),
            ("enval_throughput", self.enval_throughput),
            ("enval_energy_fraction", self.enval_energy_fraction),
            ("enval_release_fraction", self.enval_release_fraction),
            ("enval_pump", self.enval_pump),
            ("exposure", self.exposure),
        ] {
            if !value.is_finite() {
                return Err(CatalystError::new(format!("{name} must be finite")));
            }
        }
        if self.rate <= 0.0 {
            return Err(CatalystError::new("rate must be > 0"));
        }
        if self.half_saturation <= 0.0 {
            return Err(CatalystError::new("half_saturation must be > 0"));
        }
        if self.enval_sigma <= 0.0 {
            return Err(CatalystError::new("enval_sigma must be > 0"));
        }
        for (name, value) in [
            ("energy_harvest_fraction", self.energy_harvest_fraction),
            ("secretion_fraction", self.secretion_fraction),
            ("body_fraction", self.body_fraction),
            ("boundary_fraction", self.boundary_fraction),
            ("enval_energy_fraction", self.enval_energy_fraction),
            ("enval_release_fraction", self.enval_release_fraction),
            ("exposure", self.exposure),
        ] {
            if !(0.0..=1.0).contains(&value) {
                return Err(CatalystError::new(format!("{name} must be in [0, 1]")));
            }
        }
        if self.secretion_fraction + self.body_fraction + self.boundary_fraction > 1.0 + 1.0e-6 {
            return Err(CatalystError::new(
                "routing fractions must sum to at most 1",
            ));
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

    fn normalize_after_mutation(&mut self) {
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
        self.half_saturation =
            finite_or(self.half_saturation, DEFAULT_HALF_SATURATION).max(MIN_HALF_SATURATION);
        self.energy_harvest_fraction = finite_or(self.energy_harvest_fraction, 0.5).clamp(0.0, 1.0);
        self.secretion_fraction = finite_or(self.secretion_fraction, 0.1).clamp(0.0, 1.0);
        self.body_fraction = finite_or(self.body_fraction, 0.0).clamp(0.0, 1.0);
        self.boundary_fraction = finite_or(self.boundary_fraction, 0.0).clamp(0.0, 1.0);
        let routing_sum = self.secretion_fraction + self.body_fraction + self.boundary_fraction;
        if routing_sum > 1.0 {
            self.secretion_fraction /= routing_sum;
            self.body_fraction /= routing_sum;
            self.boundary_fraction /= routing_sum;
        }
        self.enval_sigma = finite_or(self.enval_sigma, 0.18).max(1.0e-4);
        self.enval_throughput = finite_or(self.enval_throughput, 0.12).max(0.0);
        self.enval_energy_fraction =
            finite_or(self.enval_energy_fraction, 2.0 / 3.0).clamp(0.0, 1.0);
        self.enval_release_fraction = finite_or(self.enval_release_fraction, 1.0 / 3.0)
            .clamp(0.0, 1.0 - self.enval_energy_fraction);
        self.enval_pump = finite_or(self.enval_pump, 0.3).max(0.0);
        self.exposure = finite_or(self.exposure, 0.0).clamp(0.0, 1.0);
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Genome {
    pub optimal_enval: f32,
    pub enzymes: Vec<Enzyme>,
    pub mutation_rate: f32,
    pub enval_mutation_floor: f32,
    pub maintenance_cost_per_area: f64,
    pub lineage_id: LineageId,
}

impl Genome {
    pub fn random_founder(rng: &mut Rng, average_enval: f32) -> Self {
        let optimal_enval = average_enval + (rng.next_f32() - 0.5) * 0.20;
        let mut genome = Self {
            optimal_enval,
            enzymes: vec![
                Enzyme::founder_downhill(),
                Enzyme::founder_engine_forward(),
                Enzyme::founder_engine_reverse(),
            ],
            mutation_rate: 0.06,
            enval_mutation_floor: 0.03,
            maintenance_cost_per_area: 0.0786,
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

        genome.optimal_enval = mutate_optimal_enval(
            genome.optimal_enval,
            reference_enval,
            genome.enval_mutation_floor,
            rng,
        );

        for enzyme in &mut genome.enzymes {
            if rng.chance(mutation_rate * 0.2) {
                *enzyme = Enzyme::random(rng);
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
                    enzyme.half_saturation = (enzyme.half_saturation
                        * (1.0 + (rng.next_f32() - 0.5) * 0.4))
                        .max(MIN_HALF_SATURATION);
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
                    enzyme.body_fraction =
                        (enzyme.body_fraction + (rng.next_f32() - 0.5) * 0.12).clamp(0.0, 1.0);
                }
                if rng.chance(mutation_rate) {
                    enzyme.boundary_fraction =
                        (enzyme.boundary_fraction + (rng.next_f32() - 0.5) * 0.12).clamp(0.0, 1.0);
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
                if rng.chance(mutation_rate) {
                    enzyme.exposure =
                        (enzyme.exposure + (rng.next_f32() - 0.5) * 0.12).clamp(0.0, 1.0);
                }
                enzyme.normalize_after_mutation();
            }
        }

        if rng.chance(mutation_rate * 0.3) && genome.enzymes.len() > MIN_CELL_ENZYMES {
            let index = rng.usize(genome.enzymes.len());
            genome.enzymes.swap_remove(index);
        } else if rng.chance(mutation_rate * 0.3) && genome.enzymes.len() < MAX_CELL_ENZYMES {
            genome.enzymes.push(Enzyme::random(rng));
        }

        genome.enforce_enzyme_count_bounds(rng);
        genome
    }

    pub fn enforce_enzyme_count_bounds(&mut self, rng: &mut Rng) {
        if self.enzymes.len() > MAX_CELL_ENZYMES {
            self.enzymes.truncate(MAX_CELL_ENZYMES);
        }
        if self.enzymes.is_empty() {
            self.enzymes.push(Enzyme::random(rng));
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
    pub mutation_rate: Option<f32>,
    pub enval_mutation_floor: Option<f32>,
    pub maintenance_cost_per_area: Option<f64>,
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
        set_field!(mutation_rate, probability_f32);
        set_field!(enval_mutation_floor, nonnegative_f32);
        set_field!(maintenance_cost_per_area, nonnegative_f64);
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
                let mut enzyme = fields.build_new_enzyme();
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
                let mut enzyme = fields.build_new_enzyme();
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
    pub reactants: Option<[f32; ELEMENT_COUNT]>,
    pub products: Option<[f32; ELEMENT_COUNT]>,
    pub rate: Option<f32>,
    pub half_saturation: Option<f32>,
    pub energy_harvest_fraction: Option<f32>,
    pub secretion_fraction: Option<f32>,
    pub body_fraction: Option<f32>,
    pub boundary_fraction: Option<f32>,
    pub enval_sigma: Option<f32>,
    pub enval_throughput: Option<f32>,
    pub enval_energy_fraction: Option<f32>,
    pub enval_release_fraction: Option<f32>,
    pub enval_pump: Option<f32>,
    pub exposure: Option<f32>,
}

impl EnzymeFieldPatch {
    fn build_new_enzyme(&self) -> Enzyme {
        Enzyme::founder_downhill()
    }

    fn apply_to_enzyme(
        &self,
        enzyme: &mut Enzyme,
        label: &str,
        changed_fields: &mut Vec<String>,
    ) -> Result<(), GenomePatchError> {
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
        set_enzyme_field!(half_saturation, positive_f32);
        set_enzyme_field!(energy_harvest_fraction, probability_f32);
        set_enzyme_field!(secretion_fraction, probability_f32);
        set_enzyme_field!(body_fraction, probability_f32);
        set_enzyme_field!(boundary_fraction, probability_f32);
        set_enzyme_field!(enval_sigma, positive_f32);
        set_enzyme_field!(enval_throughput, nonnegative_f32);
        set_enzyme_field!(enval_energy_fraction, probability_f32);
        set_enzyme_field!(enval_release_fraction, probability_f32);
        set_enzyme_field!(enval_pump, nonnegative_f32);
        set_enzyme_field!(exposure, probability_f32);
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
        Enzyme, EnzymeFieldPatch, EnzymePatchOperation, GENOME_PATCH_SCHEMA, Genome,
        GenomePatch, MAX_CELL_ENZYMES, MIN_CELL_ENZYMES,
    };
    use crate::rng::Rng;

    #[test]
    fn routing_patch_round_trips_and_rejects_v3_and_invalid_routes() {
        let mut rng = Rng::from_seed_str("v4-routing");
        let original = Genome::random_founder(&mut rng, 0.0);
        let patch = GenomePatch {
            schema: Some(GENOME_PATCH_SCHEMA.to_owned()),
            genome: Some(super::GenomeFieldPatch {
                maintenance_cost_per_area: Some(0.12),
                ..Default::default()
            }),
            enzymes: vec![EnzymePatchOperation {
                op: "update".to_owned(),
                index: Some(0),
                enzyme: None,
                fields: Some(EnzymeFieldPatch {
                    body_fraction: Some(0.3),
                    boundary_fraction: Some(0.4),
                    ..Default::default()
                }),
            }],
        };
        let mut genome = original.clone();
        patch.apply_to_genome(&mut genome).unwrap();
        let decoded: Genome = bincode::deserialize(&bincode::serialize(&genome).unwrap()).unwrap();
        assert_eq!(decoded, genome);
        assert_eq!(decoded.enzymes[0].body_fraction, 0.3);
        assert_eq!(decoded.enzymes[0].boundary_fraction, 0.4);
        assert_eq!(decoded.maintenance_cost_per_area, 0.12);
        let mut old = patch.clone();
        old.schema = Some("microcosm.genome_patch.v3".to_owned());
        assert!(old.apply_to_genome(&mut original.clone()).is_err());
        for (body, boundary) in [(-0.1, 0.0), (1.1, 0.0), (0.5, 0.5)] {
            let mut invalid = patch.clone();
            let fields = invalid.enzymes[0].fields.as_mut().unwrap();
            fields.body_fraction = Some(body);
            fields.boundary_fraction = Some(boundary);
            assert!(invalid.apply_to_genome(&mut original.clone()).is_err());
        }
    }

    #[test]
    fn routing_fractions_remain_valid_over_one_hundred_generations() {
        let mut rng = crate::rng::Rng::from_seed_str("routing-mutations");
        let mut genome = super::Genome::random_founder(&mut rng, 0.0);
        genome.mutation_rate = 1.0;
        for enzyme in &mut genome.enzymes {
            enzyme.body_fraction = 0.45;
            enzyme.boundary_fraction = 0.45;
        }
        for _ in 0..100 {
            genome = genome.mutate(&mut rng, 0.0);
            genome.validate().unwrap();
            for enzyme in &genome.enzymes {
                assert!(
                    enzyme.secretion_fraction + enzyme.body_fraction + enzyme.boundary_fraction
                        <= 1.0 + 1.0e-6
                );
            }
        }
    }

    #[test]
    fn founder_genome_is_downhill_then_enval_engine_pair() {
        use crate::chem::{Element, ElementAmounts};

        let mut rng = Rng::from_seed_str("founder-genome");
        let genome = Genome::random_founder(&mut rng, 0.1);
        genome.validate().unwrap();
        assert_eq!(genome.enzymes.len(), 3);
        assert!((genome.optimal_enval - 0.1).abs() <= 0.1);

        let downhill = genome.enzymes[0];
        let reference = Enzyme::founder_downhill();
        assert_eq!(downhill.reactants, reference.reactants);
        assert_eq!(downhill.products, reference.products);
        assert_eq!(downhill.reactants[Element::D], 1.0);
        assert_eq!(downhill.products[Element::A], 1.0);
        assert_eq!(downhill.body_fraction, 0.2);
        assert_eq!(downhill.boundary_fraction, 0.2);
        assert_eq!(downhill.rate, reference.rate);
        assert_eq!(downhill.enval_pump, reference.enval_pump);
        assert_eq!(
            downhill.energy_harvest_fraction,
            reference.energy_harvest_fraction
        );

        let b = ElementAmounts::new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
        let c = ElementAmounts::new([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
        let defaults = Enzyme::metabolic(b, c, 1.0, 0.5, 0.5);
        for (index, (reactants, products)) in [(b, c), (c, b)].into_iter().enumerate() {
            let engine = genome.enzymes[index + 1];
            assert_eq!(engine.reactants, reactants);
            assert_eq!(engine.products, products);
            assert_eq!(engine.body_fraction, 0.02);
            assert_eq!(engine.boundary_fraction, 0.02);
            assert_eq!(engine.rate, 0.8);
            assert_eq!(engine.energy_harvest_fraction, 0.65);
            assert_eq!(engine.secretion_fraction, 0.0);
            assert_eq!(engine.enval_throughput, 0.18);
            assert_eq!(engine.enval_pump, 0.0);
            assert_eq!(engine.half_saturation, defaults.half_saturation);
            assert_eq!(engine.enval_sigma, defaults.enval_sigma);
            assert_eq!(engine.enval_energy_fraction, defaults.enval_energy_fraction);
            assert!(
                (engine.enval_release_fraction - defaults.enval_release_fraction).abs() <= 1.0e-6
            );
            engine.validate().unwrap();
        }
        let forward_release = genome.enzymes[1].reactants.intrinsic_energy()
            - genome.enzymes[1].products.intrinsic_energy();
        let reverse_release = genome.enzymes[2].reactants.intrinsic_energy()
            - genome.enzymes[2].products.intrinsic_energy();
        assert!((forward_release - 0.1).abs() <= 1.0e-6);
        assert!((reverse_release + 0.1).abs() <= 1.0e-6);
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

    #[test]
    fn metabolic_enval_pump_uses_the_previous_default_and_fallback() {
        let mut enzyme = Enzyme::founder_downhill();
        assert_eq!(enzyme.enval_pump, 0.3);
        enzyme.enval_pump = f32::NAN;
        enzyme.normalize_after_mutation();
        assert_eq!(enzyme.enval_pump, 0.3);
    }

    #[test]
    fn patch_round_trips_half_saturation_and_rejects_v2() {
        let mut rng = Rng::from_seed_str("patch-v3-half-saturation");
        let mut genome = Genome::random_founder(&mut rng, 0.0);
        let patch = GenomePatch {
            schema: Some("microcosm.genome_patch.v5".to_owned()),
            genome: None,
            enzymes: vec![EnzymePatchOperation {
                op: "update".to_owned(),
                index: Some(0),
                fields: Some(EnzymeFieldPatch {
                    half_saturation: Some(0.37),
                    ..EnzymeFieldPatch::default()
                }),
                enzyme: None,
            }],
        };
        let changed = patch.apply_to_genome(&mut genome).unwrap();
        assert_eq!(changed, vec!["enzyme[0].half_saturation".to_owned()]);
        assert_eq!(genome.enzymes[0].half_saturation, 0.37);

        let mut appended = genome.clone();
        let append = GenomePatch {
            schema: Some(GENOME_PATCH_SCHEMA.to_owned()),
            genome: None,
            enzymes: vec![EnzymePatchOperation {
                op: "append".to_owned(),
                index: None,
                fields: None,
                enzyme: Some(EnzymeFieldPatch {
                    half_saturation: Some(0.05),
                    ..EnzymeFieldPatch::default()
                }),
            }],
        };
        append.apply_to_genome(&mut appended).unwrap();
        assert_eq!(appended.enzymes.last().unwrap().half_saturation, 0.05);

        for invalid in [0.0, -0.1] {
            let mut target = genome.clone();
            let patch = GenomePatch {
                schema: Some(GENOME_PATCH_SCHEMA.to_owned()),
                genome: None,
                enzymes: vec![EnzymePatchOperation {
                    op: "update".to_owned(),
                    index: Some(0),
                    fields: Some(EnzymeFieldPatch {
                        half_saturation: Some(invalid),
                        ..EnzymeFieldPatch::default()
                    }),
                    enzyme: None,
                }],
            };
            assert!(patch.apply_to_genome(&mut target).is_err());
        }

        let v2 = GenomePatch {
            schema: Some(format!("microcosm.genome_patch.v{}", 2)),
            genome: None,
            enzymes: Vec::new(),
        };
        assert!(v2.apply_to_genome(&mut genome).is_err());
    }

    #[test]
    fn half_saturation_stays_valid_over_one_hundred_mutating_generations() {
        let mut rng = Rng::from_seed_str("half-saturation-mutation");
        let mut genome = Genome::random_founder(&mut rng, 0.0);
        genome.mutation_rate = 1.0;
        let mut saw_change = false;
        for _ in 0..100 {
            let child = genome.mutate(&mut rng, 0.0);
            child.validate().unwrap();
            for enzyme in &child.enzymes {
                assert!(enzyme.half_saturation.is_finite());
                assert!(enzyme.half_saturation >= 1.0e-3);
                saw_change |= enzyme.half_saturation != super::DEFAULT_HALF_SATURATION;
            }
            genome = child;
        }
        assert!(saw_change);

        let mut enzyme = Enzyme::founder_downhill();
        enzyme.half_saturation = f32::NAN;
        enzyme.normalize_after_mutation();
        assert_eq!(enzyme.half_saturation, super::DEFAULT_HALF_SATURATION);
        enzyme.half_saturation = 1.0e-9;
        enzyme.normalize_after_mutation();
        assert_eq!(enzyme.half_saturation, 1.0e-3);
    }

    #[test]
    fn patch_accepts_generic_catalyst_fields_and_rejects_v1() {
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

    fn exposure_patch(schema: &str, exposure: f32) -> GenomePatch {
        GenomePatch {
            schema: Some(schema.to_owned()),
            genome: None,
            enzymes: vec![EnzymePatchOperation {
                op: "update".to_owned(),
                index: Some(1),
                fields: Some(EnzymeFieldPatch {
                    exposure: Some(exposure),
                    ..EnzymeFieldPatch::default()
                }),
                enzyme: None,
            }],
        }
    }

    #[test]
    fn founders_start_with_every_enzyme_fully_internal() {
        let mut rng = Rng::from_seed_str("founder-exposure");
        for _ in 0..20 {
            let genome = Genome::random_founder(&mut rng, 0.0);
            assert!(genome.enzymes.iter().all(|enzyme| enzyme.exposure == 0.0));
        }
        assert_eq!(Enzyme::random(&mut rng).exposure, 0.0);
    }

    #[test]
    fn v5_patch_round_trips_exposure_and_rejects_v4_and_out_of_range_values() {
        assert_eq!(GENOME_PATCH_SCHEMA, "microcosm.genome_patch.v5");
        let mut rng = Rng::from_seed_str("exposure-patch");
        let original = Genome::random_founder(&mut rng, 0.0);

        let mut genome = original.clone();
        let changed = exposure_patch(GENOME_PATCH_SCHEMA, 0.35)
            .apply_to_genome(&mut genome)
            .unwrap();
        assert_eq!(changed, vec!["enzyme[1].exposure".to_owned()]);
        assert_eq!(genome.enzymes[1].exposure, 0.35);
        let decoded: Genome = bincode::deserialize(&bincode::serialize(&genome).unwrap()).unwrap();
        assert_eq!(decoded, genome);

        let append = GenomePatch {
            schema: Some(GENOME_PATCH_SCHEMA.to_owned()),
            genome: None,
            enzymes: vec![EnzymePatchOperation {
                op: "append".to_owned(),
                index: None,
                fields: None,
                enzyme: Some(EnzymeFieldPatch {
                    exposure: Some(1.0),
                    ..EnzymeFieldPatch::default()
                }),
            }],
        };
        let mut appended = original.clone();
        append.apply_to_genome(&mut appended).unwrap();
        assert_eq!(appended.enzymes.last().unwrap().exposure, 1.0);

        for schema in ["microcosm.genome_patch.v4", "microcosm.genome_patch.v3"] {
            let mut target = original.clone();
            assert!(exposure_patch(schema, 0.35).apply_to_genome(&mut target).is_err());
            assert_eq!(target, original);
        }
        for invalid in [-0.01, 1.01, f32::NAN, f32::INFINITY] {
            let mut target = original.clone();
            assert!(
                exposure_patch(GENOME_PATCH_SCHEMA, invalid)
                    .apply_to_genome(&mut target)
                    .is_err()
            );
        }
    }

    #[test]
    fn exposure_stays_valid_and_drifts_over_one_hundred_mutating_generations() {
        let mut rng = Rng::from_seed_str("exposure-mutation");
        let mut genome = Genome::random_founder(&mut rng, 0.0);
        genome.mutation_rate = 1.0;
        genome.enzymes[0].exposure = 0.98;
        genome.enzymes[1].exposure = 0.02;
        let mut saw_interior = false;
        let mut saw_bounds = false;
        for _ in 0..100 {
            genome = genome.mutate(&mut rng, 0.0);
            genome.validate().unwrap();
            for enzyme in &genome.enzymes {
                assert!(enzyme.exposure.is_finite());
                assert!((0.0..=1.0).contains(&enzyme.exposure));
                saw_interior |= enzyme.exposure > 0.0 && enzyme.exposure < 1.0;
                saw_bounds |= enzyme.exposure == 0.0 || enzyme.exposure == 1.0;
            }
        }
        assert!(saw_interior && saw_bounds);

        let mut enzyme = Enzyme::founder_downhill();
        for (raw, normalized) in [(f32::NAN, 0.0), (-0.5, 0.0), (1.5, 1.0), (0.25, 0.25)] {
            enzyme.exposure = raw;
            enzyme.normalize_after_mutation();
            assert_eq!(enzyme.exposure, normalized);
        }
        enzyme.exposure = 1.5;
        assert!(enzyme.validate().is_err());
    }
}
