use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::chem::{normalize_specificity_mask, Element, ALL_ELEMENT_MASK, ELEMENT_ORDER};
use crate::rng::Rng;

pub const MIN_CELL_ENZYMES: usize = 1;
pub const MAX_CELL_ENZYMES: usize = 10;
pub const DEFAULT_COMBAT_LEVEL_MEAN: u32 = 100;
pub const DEFAULT_COMBAT_LEVEL_SIGMA: f64 = 10.0;
pub const COMBAT_LEVEL_MUTATION_STEP_MIN: u32 = 1;
pub const COMBAT_LEVEL_MUTATION_STEP_MAX: u32 = 4;

const ABC_MASK: u8 = Element::A.mask() | Element::B.mask() | Element::C.mask();
const EVOLVABLE_ENZYME_TYPES: [EnzymeType; 5] = [
    EnzymeType::Anabolase,
    EnzymeType::Catabolase,
    EnzymeType::Transmutase,
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
    Anabolase,
    Catabolase,
    Transmutase,
    Defensase,
    Attackase,
}

impl EnzymeType {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Anabolase => "anabolase",
            Self::Catabolase => "catabolase",
            Self::Transmutase => "transmutase",
            Self::Defensase => "defensase",
            Self::Attackase => "attackase",
        }
    }

    pub const fn is_metabolic(self) -> bool {
        matches!(self, Self::Anabolase | Self::Catabolase | Self::Transmutase)
    }

    pub const fn is_combat(self) -> bool {
        matches!(self, Self::Defensase | Self::Attackase)
    }
}

impl std::str::FromStr for EnzymeType {
    type Err = GenomePatchError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "anabolase" => Ok(Self::Anabolase),
            "catabolase" => Ok(Self::Catabolase),
            "transmutase" => Ok(Self::Transmutase),
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
    pub specificity_mask: u8,
    pub bond_multiplier: f32,
    pub bond_cost_fraction: f32,
    pub bond_harvest_fraction: f32,
    pub downhill_harvest_fraction: f32,
    pub secretion_prob: f32,
    pub enval_sigma: f32,
    pub enval_throughput: f32,
    pub enval_energy_fraction: f32,
    pub enval_release_fraction: f32,
    pub enval_pump: f32,
    pub combat_level: u32,
}

impl Enzyme {
    pub fn anabolase_abc(rng: &mut Rng) -> Self {
        let mut enzyme = Self::metabolic_base(EnzymeType::Anabolase, ABC_MASK, rng);
        enzyme.bond_multiplier = 1.18;
        enzyme.bond_cost_fraction = 0.70;
        enzyme.secretion_prob = 0.12;
        enzyme.enval_throughput = 0.18;
        enzyme
    }

    pub fn catabolase_abc(rng: &mut Rng) -> Self {
        let mut enzyme = Self::metabolic_base(EnzymeType::Catabolase, ABC_MASK, rng);
        enzyme.bond_harvest_fraction = 1.0;
        enzyme.enval_throughput = 0.14;
        enzyme
    }

    pub fn defensase(level: u32) -> Self {
        Self::combat(EnzymeType::Defensase, level)
    }

    pub fn attackase(level: u32) -> Self {
        Self::combat(EnzymeType::Attackase, level)
    }

    pub fn random(enzyme_type: EnzymeType, rng: &mut Rng) -> Self {
        if enzyme_type.is_combat() {
            let level = normalize_combat_level(rng.gaussian(
                f64::from(DEFAULT_COMBAT_LEVEL_MEAN),
                DEFAULT_COMBAT_LEVEL_SIGMA,
            ));
            return Self::combat(enzyme_type, level);
        }

        let mut enzyme = Self::metabolic_base(enzyme_type, make_random_specificity_mask(rng), rng);
        enzyme.secretion_prob = rng.next_f32();
        enzyme.apply_class_defaults(rng);
        enzyme
    }

    pub fn apply_class_defaults(&mut self, rng: &mut Rng) {
        if self.enzyme_type.is_combat() {
            if self.combat_level == 0 {
                self.combat_level = normalize_combat_level(rng.gaussian(
                    f64::from(DEFAULT_COMBAT_LEVEL_MEAN),
                    DEFAULT_COMBAT_LEVEL_SIGMA,
                ));
            } else {
                self.combat_level = self.combat_level.max(1);
            }
            self.specificity_mask = 0;
            self.bond_multiplier = 1.0;
            self.bond_cost_fraction = 0.0;
            self.bond_harvest_fraction = 0.0;
            self.downhill_harvest_fraction = 0.0;
            self.secretion_prob = 0.0;
            self.enval_sigma = 0.0;
            self.enval_throughput = 0.0;
            self.enval_energy_fraction = 0.0;
            self.enval_release_fraction = 0.0;
            self.enval_pump = 0.0;
            return;
        }

        if self.enval_sigma <= 0.0 || !self.enval_sigma.is_finite() {
            self.enval_sigma = 0.16 + rng.next_f32() * 0.08;
        }
        if self.secretion_prob < 0.0 || !self.secretion_prob.is_finite() {
            self.secretion_prob = 0.15;
        }
        self.secretion_prob = self.secretion_prob.clamp(0.0, 1.0);
        self.specificity_mask = normalize_specificity_mask(
            self.specificity_mask,
            default_specificity_mask_for_type(self.enzyme_type),
        );
        if self.enval_energy_fraction <= 0.0 || !self.enval_energy_fraction.is_finite() {
            self.enval_energy_fraction = 2.0 / 3.0;
        }
        self.enval_energy_fraction = self.enval_energy_fraction.clamp(0.0, 1.0);
        if self.enval_release_fraction <= 0.0 || !self.enval_release_fraction.is_finite() {
            self.enval_release_fraction = 1.0 / 3.0;
        }
        self.enval_release_fraction = self
            .enval_release_fraction
            .clamp(0.0, 1.0 - self.enval_energy_fraction);
        self.enval_pump = if self.enval_pump.is_finite() && self.enval_pump >= 0.0 {
            self.enval_pump
        } else {
            0.3
        };
        self.combat_level = 0;

        match self.enzyme_type {
            EnzymeType::Anabolase => {
                if self.enval_throughput <= 0.0 || !self.enval_throughput.is_finite() {
                    self.enval_throughput = 0.18;
                }
                if self.bond_multiplier <= 0.0 || !self.bond_multiplier.is_finite() {
                    self.bond_multiplier = 1.18;
                }
                if self.bond_cost_fraction < 0.0 || !self.bond_cost_fraction.is_finite() {
                    self.bond_cost_fraction = 0.70;
                }
                self.bond_harvest_fraction = 0.0;
                self.downhill_harvest_fraction = 0.0;
            }
            EnzymeType::Catabolase => {
                if self.enval_throughput <= 0.0 || !self.enval_throughput.is_finite() {
                    self.enval_throughput = 0.14;
                }
                self.bond_harvest_fraction = 1.0;
                self.bond_multiplier = 1.0;
                self.bond_cost_fraction = 0.0;
                self.downhill_harvest_fraction = 0.0;
            }
            EnzymeType::Transmutase => {
                if self.enval_throughput <= 0.0 || !self.enval_throughput.is_finite() {
                    self.enval_throughput = 0.04;
                }
                if self.downhill_harvest_fraction < 0.0
                    || !self.downhill_harvest_fraction.is_finite()
                {
                    self.downhill_harvest_fraction = 0.18;
                }
                self.downhill_harvest_fraction = self.downhill_harvest_fraction.clamp(0.0, 1.0);
                self.bond_multiplier = 1.0;
                self.bond_cost_fraction = 0.0;
                self.bond_harvest_fraction = 0.0;
            }
            EnzymeType::Defensase | EnzymeType::Attackase => unreachable!(),
        }
    }

    fn metabolic_base(enzyme_type: EnzymeType, specificity_mask: u8, rng: &mut Rng) -> Self {
        let mut enzyme = Self {
            enzyme_type,
            specificity_mask,
            bond_multiplier: 1.0,
            bond_cost_fraction: 0.0,
            bond_harvest_fraction: 0.0,
            downhill_harvest_fraction: 0.0,
            secretion_prob: 0.15,
            enval_sigma: 0.16 + rng.next_f32() * 0.08,
            enval_throughput: 0.10,
            enval_energy_fraction: 2.0 / 3.0,
            enval_release_fraction: 1.0 / 3.0,
            enval_pump: 0.3,
            combat_level: 0,
        };
        enzyme.apply_class_defaults(rng);
        enzyme
    }

    fn combat(enzyme_type: EnzymeType, level: u32) -> Self {
        Self {
            enzyme_type,
            specificity_mask: 0,
            bond_multiplier: 1.0,
            bond_cost_fraction: 0.0,
            bond_harvest_fraction: 0.0,
            downhill_harvest_fraction: 0.0,
            secretion_prob: 0.0,
            enval_sigma: 0.0,
            enval_throughput: 0.0,
            enval_energy_fraction: 0.0,
            enval_release_fraction: 0.0,
            enval_pump: 0.0,
            combat_level: level.max(1),
        }
    }

    pub fn edit_default(enzyme_type: EnzymeType) -> Self {
        let mut enzyme = Self {
            enzyme_type,
            specificity_mask: default_specificity_mask_for_type(enzyme_type),
            bond_multiplier: if enzyme_type == EnzymeType::Anabolase {
                1.18
            } else {
                1.0
            },
            bond_cost_fraction: if enzyme_type == EnzymeType::Anabolase {
                0.70
            } else {
                0.0
            },
            bond_harvest_fraction: if enzyme_type == EnzymeType::Catabolase {
                1.0
            } else {
                0.0
            },
            downhill_harvest_fraction: if enzyme_type == EnzymeType::Transmutase {
                0.18
            } else {
                0.0
            },
            secretion_prob: if enzyme_type.is_metabolic() {
                0.15
            } else {
                0.0
            },
            enval_sigma: if enzyme_type.is_metabolic() {
                0.18
            } else {
                0.0
            },
            enval_throughput: match enzyme_type {
                EnzymeType::Anabolase => 0.18,
                EnzymeType::Catabolase => 0.14,
                EnzymeType::Transmutase => 0.04,
                EnzymeType::Defensase | EnzymeType::Attackase => 0.0,
            },
            enval_energy_fraction: if enzyme_type.is_metabolic() {
                2.0 / 3.0
            } else {
                0.0
            },
            enval_release_fraction: if enzyme_type.is_metabolic() {
                1.0 / 3.0
            } else {
                0.0
            },
            enval_pump: if enzyme_type.is_metabolic() { 0.3 } else { 0.0 },
            combat_level: if enzyme_type.is_combat() {
                DEFAULT_COMBAT_LEVEL_MEAN
            } else {
                0
            },
        };
        enzyme.normalize_for_edit();
        enzyme
    }

    pub fn normalize_for_edit(&mut self) {
        if self.enzyme_type.is_combat() {
            self.specificity_mask = 0;
            self.bond_multiplier = 1.0;
            self.bond_cost_fraction = 0.0;
            self.bond_harvest_fraction = 0.0;
            self.downhill_harvest_fraction = 0.0;
            self.secretion_prob = 0.0;
            self.enval_sigma = 0.0;
            self.enval_throughput = 0.0;
            self.enval_energy_fraction = 0.0;
            self.enval_release_fraction = 0.0;
            self.enval_pump = 0.0;
            self.combat_level = self.combat_level.max(1);
            return;
        }

        self.specificity_mask = normalize_specificity_mask(
            self.specificity_mask,
            default_specificity_mask_for_type(self.enzyme_type),
        );
        if self.enval_sigma <= 0.0 || !self.enval_sigma.is_finite() {
            self.enval_sigma = 0.18;
        }
        if self.secretion_prob < 0.0 || !self.secretion_prob.is_finite() {
            self.secretion_prob = 0.15;
        }
        self.secretion_prob = self.secretion_prob.clamp(0.0, 1.0);
        if self.enval_energy_fraction <= 0.0 || !self.enval_energy_fraction.is_finite() {
            self.enval_energy_fraction = 2.0 / 3.0;
        }
        self.enval_energy_fraction = self.enval_energy_fraction.clamp(0.0, 1.0);
        if self.enval_release_fraction <= 0.0 || !self.enval_release_fraction.is_finite() {
            self.enval_release_fraction = 1.0 / 3.0;
        }
        self.enval_release_fraction = self
            .enval_release_fraction
            .clamp(0.0, 1.0 - self.enval_energy_fraction);
        if self.enval_pump < 0.0 || !self.enval_pump.is_finite() {
            self.enval_pump = 0.3;
        }
        self.combat_level = 0;
        match self.enzyme_type {
            EnzymeType::Anabolase => {
                if self.enval_throughput <= 0.0 || !self.enval_throughput.is_finite() {
                    self.enval_throughput = 0.18;
                }
                if self.bond_multiplier <= 0.0 || !self.bond_multiplier.is_finite() {
                    self.bond_multiplier = 1.18;
                }
                if self.bond_cost_fraction < 0.0 || !self.bond_cost_fraction.is_finite() {
                    self.bond_cost_fraction = 0.70;
                }
                self.bond_cost_fraction = self.bond_cost_fraction.clamp(0.0, 1.0);
                self.bond_harvest_fraction = 0.0;
                self.downhill_harvest_fraction = 0.0;
            }
            EnzymeType::Catabolase => {
                if self.enval_throughput <= 0.0 || !self.enval_throughput.is_finite() {
                    self.enval_throughput = 0.14;
                }
                self.bond_multiplier = 1.0;
                self.bond_cost_fraction = 0.0;
                self.bond_harvest_fraction = 1.0;
                self.downhill_harvest_fraction = 0.0;
            }
            EnzymeType::Transmutase => {
                if self.enval_throughput <= 0.0 || !self.enval_throughput.is_finite() {
                    self.enval_throughput = 0.04;
                }
                if self.downhill_harvest_fraction < 0.0
                    || !self.downhill_harvest_fraction.is_finite()
                {
                    self.downhill_harvest_fraction = 0.18;
                }
                self.downhill_harvest_fraction = self.downhill_harvest_fraction.clamp(0.0, 1.0);
                self.bond_multiplier = 1.0;
                self.bond_cost_fraction = 0.0;
                self.bond_harvest_fraction = 0.0;
            }
            EnzymeType::Defensase | EnzymeType::Attackase => unreachable!(),
        }
    }
}

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
    pub default_secretion_prob: f32,
    pub mutation_rate: f32,
    pub post_divide_mortality: f32,
    pub desired_element_reserve: u32,
    pub enval_stress_factor: f64,
    pub enval_mutation_floor: f32,
    pub maintenance_cost_per_sec: f64,
    pub lineage_id: LineageId,
}

impl Genome {
    pub fn random_founder(rng: &mut Rng, average_enval: f32) -> Self {
        let optimal_enval = average_enval + (rng.next_f32() - 0.5) * 0.20;
        let anabolase = Enzyme::anabolase_abc(rng);
        let catabolase = Enzyme::catabolase_abc(rng);
        let defensase_level = normalize_combat_level(rng.gaussian(
            f64::from(DEFAULT_COMBAT_LEVEL_MEAN),
            DEFAULT_COMBAT_LEVEL_SIGMA,
        ));
        let mut genome = Self {
            optimal_enval,
            enzymes: vec![anabolase, catabolase, Enzyme::defensase(defensase_level)],
            repro_threshold: 2.0 + rng.next_f64() * 6.0,
            initial_energy: 1.2 + rng.next_f64() * 1.8,
            decay_time: 700.0 + rng.next_f64() * 2000.0,
            default_secretion_prob: 0.15,
            mutation_rate: 0.06,
            post_divide_mortality: 0.0,
            desired_element_reserve: 2,
            enval_stress_factor: 0.02,
            enval_mutation_floor: 0.03,
            maintenance_cost_per_sec: 0.05,
            lineage_id: LineageId(rng.usize(1_000_000_000) as u64),
        };
        genome.enforce_enzyme_count_bounds(rng);
        genome.refresh_combat_totals();
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
            genome.default_secretion_prob =
                (genome.default_secretion_prob + (rng.next_f32() - 0.5) * 0.2).clamp(0.0, 1.0);
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
            enzyme.apply_class_defaults(rng);
            if rng.chance(mutation_rate) {
                if !enzyme.enzyme_type.is_combat() && rng.chance(mutation_rate) {
                    enzyme.specificity_mask = mutate_specificity_mask(enzyme.specificity_mask, rng);
                }
                if rng.chance(mutation_rate * 0.2) {
                    let next_type = EVOLVABLE_ENZYME_TYPES[rng.usize(EVOLVABLE_ENZYME_TYPES.len())];
                    enzyme.enzyme_type = next_type;
                    if next_type.is_combat() {
                        enzyme.combat_level = 0;
                    } else if enzyme.specificity_mask == 0 {
                        enzyme.specificity_mask = default_specificity_mask_for_type(next_type);
                    }
                    enzyme.apply_class_defaults(rng);
                }
                if enzyme.enzyme_type.is_combat() {
                    enzyme.combat_level = mutate_combat_level(enzyme.combat_level, rng);
                } else {
                    if rng.chance(mutation_rate) {
                        enzyme.enval_sigma =
                            (enzyme.enval_sigma * (1.0 + (rng.next_f32() - 0.5) * 0.35)).max(0.02);
                    }
                    if rng.chance(mutation_rate) {
                        enzyme.secretion_prob =
                            (enzyme.secretion_prob + (rng.next_f32() - 0.5) * 0.2).clamp(0.0, 1.0);
                    }
                    if rng.chance(mutation_rate) {
                        enzyme.enval_throughput = (enzyme.enval_throughput
                            * (1.0 + (rng.next_f32() - 0.5) * 0.4))
                            .max(0.01);
                    }
                    if enzyme.enzyme_type == EnzymeType::Anabolase && rng.chance(mutation_rate) {
                        enzyme.bond_multiplier = (enzyme.bond_multiplier
                            * (1.0 + (rng.next_f32() - 0.5) * 0.18))
                            .max(1.01);
                    }
                    if enzyme.enzyme_type == EnzymeType::Transmutase && rng.chance(mutation_rate) {
                        enzyme.downhill_harvest_fraction = (enzyme.downhill_harvest_fraction
                            + (rng.next_f32() - 0.5) * 0.08)
                            .clamp(0.0, 1.0);
                    }
                }
            }
            enzyme.apply_class_defaults(rng);
        }

        if rng.chance(mutation_rate * 0.3) && genome.enzymes.len() > MIN_CELL_ENZYMES {
            let idx = rng.usize(genome.enzymes.len());
            genome.enzymes.swap_remove(idx);
        } else if rng.chance(mutation_rate * 0.3) && genome.enzymes.len() < MAX_CELL_ENZYMES {
            let enzyme_type = EVOLVABLE_ENZYME_TYPES[rng.usize(EVOLVABLE_ENZYME_TYPES.len())];
            genome.enzymes.push(Enzyme::random(enzyme_type, rng));
        }

        genome.enforce_enzyme_count_bounds(rng);
        genome.repro_threshold = genome.repro_threshold.max(0.01);
        genome.decay_time = genome.decay_time.max(50.0);
        genome.refresh_combat_totals();
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
                for enzyme in prey_enzymes {
                    self.push_absorbed_enzyme(*enzyme, rng);
                    stats.added += 1;
                }
            } else {
                let selected = select_random_indices(prey_enzymes.len(), free_slots, rng);
                for index in selected {
                    self.push_absorbed_enzyme(prey_enzymes[index], rng);
                    stats.added += 1;
                }
            }
            self.enforce_enzyme_count_bounds(rng);
            return stats;
        }

        let replacement_order = select_random_indices(self.enzymes.len(), self.enzymes.len(), rng);
        let mut replacement_index = 0_usize;
        for enzyme in prey_enzymes {
            if replacement_index >= replacement_order.len() {
                break;
            }
            if !rng.chance(0.25) {
                continue;
            }
            let slot = replacement_order[replacement_index];
            replacement_index += 1;
            let mut absorbed = *enzyme;
            absorbed.apply_class_defaults(rng);
            self.enzymes[slot] = absorbed;
            stats.replacements += 1;
            stats.replacement_slots.push(slot);
        }
        self.enforce_enzyme_count_bounds(rng);
        stats
    }

    fn push_absorbed_enzyme(&mut self, enzyme: Enzyme, rng: &mut Rng) {
        if self.enzymes.len() >= MAX_CELL_ENZYMES {
            return;
        }
        let mut absorbed = enzyme;
        absorbed.apply_class_defaults(rng);
        self.enzymes.push(absorbed);
    }

    pub fn enforce_enzyme_count_bounds(&mut self, rng: &mut Rng) {
        if self.enzymes.len() > MAX_CELL_ENZYMES {
            self.enzymes.truncate(MAX_CELL_ENZYMES);
        }
        if self.enzymes.is_empty() {
            self.enzymes
                .push(Enzyme::random(EnzymeType::Anabolase, rng));
        }
        for enzyme in &mut self.enzymes {
            enzyme.apply_class_defaults(rng);
        }
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

pub const GENOME_PATCH_SCHEMA: &str = "microcosm.genome_patch.v1";

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
        if let Some(schema) = &self.schema {
            if schema != GENOME_PATCH_SCHEMA {
                return Err(GenomePatchError::new(format!(
                    "unsupported genome patch schema '{schema}'"
                )));
            }
        }
        let mut changed_fields = Vec::new();
        if let Some(fields) = &self.genome {
            fields.apply_to_genome(genome, &mut changed_fields)?;
        }
        for operation in &self.enzymes {
            operation.apply_to_enzymes(&mut genome.enzymes, &mut changed_fields)?;
        }
        if genome.enzymes.len() < MIN_CELL_ENZYMES || genome.enzymes.len() > MAX_CELL_ENZYMES {
            return Err(GenomePatchError::new(format!(
                "genome enzyme count must be in [{MIN_CELL_ENZYMES}, {MAX_CELL_ENZYMES}], got {}",
                genome.enzymes.len()
            )));
        }
        Ok(changed_fields)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenomeFieldPatch {
    pub optimal_enval: Option<f32>,
    pub repro_threshold: Option<f64>,
    pub decay_time: Option<f64>,
    pub default_secretion_prob: Option<f32>,
    pub mutation_rate: Option<f32>,
    pub post_divide_mortality: Option<f32>,
    pub desired_element_reserve: Option<u32>,
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
        if let Some(value) = self.optimal_enval {
            genome.optimal_enval = finite_f32("optimal_enval", value)?;
            changed_fields.push("optimal_enval".to_owned());
        }
        if let Some(value) = self.repro_threshold {
            genome.repro_threshold = positive_f64("repro_threshold", value)?;
            changed_fields.push("repro_threshold".to_owned());
        }
        if let Some(value) = self.decay_time {
            genome.decay_time = positive_f64("decay_time", value)?;
            changed_fields.push("decay_time".to_owned());
        }
        if let Some(value) = self.default_secretion_prob {
            genome.default_secretion_prob = probability_f32("default_secretion_prob", value)?;
            changed_fields.push("default_secretion_prob".to_owned());
        }
        if let Some(value) = self.mutation_rate {
            genome.mutation_rate = probability_f32("mutation_rate", value)?;
            changed_fields.push("mutation_rate".to_owned());
        }
        if let Some(value) = self.post_divide_mortality {
            genome.post_divide_mortality = probability_f32("post_divide_mortality", value)?;
            changed_fields.push("post_divide_mortality".to_owned());
        }
        if let Some(value) = self.desired_element_reserve {
            genome.desired_element_reserve = value;
            changed_fields.push("desired_element_reserve".to_owned());
        }
        if let Some(value) = self.enval_stress_factor {
            genome.enval_stress_factor = nonnegative_f64("enval_stress_factor", value)?;
            changed_fields.push("enval_stress_factor".to_owned());
        }
        if let Some(value) = self.enval_mutation_floor {
            genome.enval_mutation_floor = nonnegative_f32("enval_mutation_floor", value)?;
            changed_fields.push("enval_mutation_floor".to_owned());
        }
        if let Some(value) = self.maintenance_cost_per_sec {
            genome.maintenance_cost_per_sec = nonnegative_f64("maintenance_cost_per_sec", value)?;
            changed_fields.push("maintenance_cost_per_sec".to_owned());
        }
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
                let mut enzyme = fields.build_new_enzyme()?;
                let index = enzymes.len();
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
    pub specificity_mask: Option<u8>,
    pub specificity_elements: Option<Vec<String>>,
    pub bond_multiplier: Option<f32>,
    pub bond_cost_fraction: Option<f32>,
    pub bond_harvest_fraction: Option<f32>,
    pub downhill_harvest_fraction: Option<f32>,
    pub secretion_prob: Option<f32>,
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
            enzyme.enzyme_type = value.parse::<EnzymeType>()?;
            changed_fields.push(format!("{label}.enzyme_type"));
        }
        if self.specificity_mask.is_some() && self.specificity_elements.is_some() {
            return Err(GenomePatchError::new(format!(
                "{label} cannot specify both specificity_mask and specificity_elements"
            )));
        }
        if let Some(mask) = self.specificity_mask {
            enzyme.specificity_mask = normalize_specificity_mask(
                mask,
                default_specificity_mask_for_type(enzyme.enzyme_type),
            );
            changed_fields.push(format!("{label}.specificity_mask"));
        }
        if let Some(elements) = &self.specificity_elements {
            enzyme.specificity_mask = specificity_mask_from_strings(elements)?;
            changed_fields.push(format!("{label}.specificity_mask"));
        }
        if let Some(value) = self.bond_multiplier {
            enzyme.bond_multiplier = positive_f32(&format!("{label}.bond_multiplier"), value)?;
            changed_fields.push(format!("{label}.bond_multiplier"));
        }
        if let Some(value) = self.bond_cost_fraction {
            enzyme.bond_cost_fraction =
                probability_f32(&format!("{label}.bond_cost_fraction"), value)?;
            changed_fields.push(format!("{label}.bond_cost_fraction"));
        }
        if let Some(value) = self.bond_harvest_fraction {
            enzyme.bond_harvest_fraction =
                probability_f32(&format!("{label}.bond_harvest_fraction"), value)?;
            changed_fields.push(format!("{label}.bond_harvest_fraction"));
        }
        if let Some(value) = self.downhill_harvest_fraction {
            enzyme.downhill_harvest_fraction =
                probability_f32(&format!("{label}.downhill_harvest_fraction"), value)?;
            changed_fields.push(format!("{label}.downhill_harvest_fraction"));
        }
        if let Some(value) = self.secretion_prob {
            enzyme.secretion_prob = probability_f32(&format!("{label}.secretion_prob"), value)?;
            changed_fields.push(format!("{label}.secretion_prob"));
        }
        if let Some(value) = self.enval_sigma {
            enzyme.enval_sigma = positive_f32(&format!("{label}.enval_sigma"), value)?;
            changed_fields.push(format!("{label}.enval_sigma"));
        }
        if let Some(value) = self.enval_throughput {
            enzyme.enval_throughput = nonnegative_f32(&format!("{label}.enval_throughput"), value)?;
            changed_fields.push(format!("{label}.enval_throughput"));
        }
        if let Some(value) = self.enval_energy_fraction {
            enzyme.enval_energy_fraction =
                probability_f32(&format!("{label}.enval_energy_fraction"), value)?;
            changed_fields.push(format!("{label}.enval_energy_fraction"));
        }
        if let Some(value) = self.enval_release_fraction {
            enzyme.enval_release_fraction =
                probability_f32(&format!("{label}.enval_release_fraction"), value)?;
            changed_fields.push(format!("{label}.enval_release_fraction"));
        }
        if let Some(value) = self.enval_pump {
            enzyme.enval_pump = nonnegative_f32(&format!("{label}.enval_pump"), value)?;
            changed_fields.push(format!("{label}.enval_pump"));
        }
        if let Some(value) = self.combat_level {
            if value == 0 {
                return Err(GenomePatchError::new(format!(
                    "{label}.combat_level must be >= 1"
                )));
            }
            enzyme.combat_level = value;
            changed_fields.push(format!("{label}.combat_level"));
        }
        enzyme.normalize_for_edit();
        Ok(())
    }
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

fn specificity_mask_from_strings(elements: &[String]) -> Result<u8, GenomePatchError> {
    if elements.is_empty() {
        return Err(GenomePatchError::new(
            "specificity_elements must not be empty",
        ));
    }
    let mut mask = 0_u8;
    for element in elements {
        let bit = match element.as_str() {
            "A" | "a" => Element::A.mask(),
            "B" | "b" => Element::B.mask(),
            "C" | "c" => Element::C.mask(),
            "D" | "d" => Element::D.mask(),
            "E" | "e" => Element::E.mask(),
            "F" | "f" => Element::F.mask(),
            other => {
                return Err(GenomePatchError::new(format!(
                    "invalid specificity element '{other}'"
                )))
            }
        };
        mask |= bit;
    }
    Ok(normalize_specificity_mask(mask, ALL_ELEMENT_MASK))
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

pub fn default_specificity_mask_for_type(enzyme_type: EnzymeType) -> u8 {
    match enzyme_type {
        EnzymeType::Anabolase | EnzymeType::Catabolase => ABC_MASK,
        EnzymeType::Transmutase | EnzymeType::Defensase | EnzymeType::Attackase => ALL_ELEMENT_MASK,
    }
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

pub fn make_random_specificity_mask(rng: &mut Rng) -> u8 {
    let mut elements = ELEMENT_ORDER;
    rng.shuffle_in_place(&mut elements);
    let target_count = 1 + rng.usize(3);
    let mut mask = 0_u8;
    for element in elements.iter().take(target_count) {
        mask |= element.mask();
    }
    normalize_specificity_mask(mask, ALL_ELEMENT_MASK)
}

pub fn mutate_specificity_mask(mask: u8, rng: &mut Rng) -> u8 {
    let mut next = normalize_specificity_mask(mask, ALL_ELEMENT_MASK);
    let mut present = Vec::new();
    let mut absent = Vec::new();
    for element in ELEMENT_ORDER {
        let bit = element.mask();
        if next & bit != 0 {
            present.push(bit);
        } else {
            absent.push(bit);
        }
    }

    if present.len() <= 1 && absent.is_empty() {
        return next;
    }
    if present.len() <= 1 {
        next |= absent[rng.usize(absent.len())];
        return normalize_specificity_mask(next, ALL_ELEMENT_MASK);
    }
    if absent.is_empty() {
        next &= !present[rng.usize(present.len())];
        return normalize_specificity_mask(next, ALL_ELEMENT_MASK);
    }

    if rng.chance(0.5) {
        next |= absent[rng.usize(absent.len())];
    } else {
        next &= !present[rng.usize(present.len())];
    }
    normalize_specificity_mask(next, ALL_ELEMENT_MASK)
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
    use super::{Enzyme, EnzymeType, Genome, MAX_CELL_ENZYMES, MIN_CELL_ENZYMES};
    use crate::rng::Rng;

    #[test]
    fn founder_genome_has_expected_enzyme_classes() {
        let mut rng = Rng::from_seed_str("founder-genome");
        let genome = Genome::random_founder(&mut rng, 0.1);
        assert_eq!(genome.enzymes.len(), 3);
        assert_eq!(genome.enzymes[0].enzyme_type, EnzymeType::Anabolase);
        assert_eq!(genome.enzymes[1].enzyme_type, EnzymeType::Catabolase);
        assert_eq!(genome.enzymes[2].enzyme_type, EnzymeType::Defensase);
        assert_eq!(genome.attack_total(), 0);
        assert!(genome.defense_total() > 0);
    }

    fn genome_with_enzymes(count: usize, start_level: u32) -> Genome {
        let mut rng = Rng::from_seed_str("enzyme-transfer-genome");
        let mut genome = Genome::random_founder(&mut rng, 0.0);
        genome.enzymes.clear();
        for i in 0..count {
            let level = start_level + i as u32;
            if i % 2 == 0 {
                genome.enzymes.push(Enzyme::attackase(level));
            } else {
                genome.enzymes.push(Enzyme::defensase(level));
            }
        }
        genome
    }

    #[test]
    fn predation_enzyme_transfer_adds_all_when_space_allows() {
        let mut rng = Rng::from_seed_str("transfer-all");
        let mut predator = genome_with_enzymes(4, 10);
        let prey = genome_with_enzymes(3, 100);
        let stats = predator.absorb_predation_enzymes(&prey.enzymes, &mut rng);
        assert_eq!(predator.enzymes.len(), 7);
        assert_eq!(stats.added, 3);
        assert_eq!(stats.replacements, 0);
    }

    #[test]
    fn predation_enzyme_transfer_fills_partial_slots() {
        let mut rng = Rng::from_seed_str("transfer-partial-a");
        let mut predator = genome_with_enzymes(7, 10);
        let prey = genome_with_enzymes(4, 100);
        let stats = predator.absorb_predation_enzymes(&prey.enzymes, &mut rng);
        assert_eq!(predator.enzymes.len(), MAX_CELL_ENZYMES);
        assert_eq!(stats.added, 3);

        let mut rng = Rng::from_seed_str("transfer-partial-b");
        let mut predator = genome_with_enzymes(4, 10);
        let prey = genome_with_enzymes(7, 100);
        let stats = predator.absorb_predation_enzymes(&prey.enzymes, &mut rng);
        assert_eq!(predator.enzymes.len(), MAX_CELL_ENZYMES);
        assert_eq!(stats.added, 6);
    }

    #[test]
    fn predation_enzyme_replacement_is_deterministic_and_unique() {
        fn run_once() -> (Genome, Vec<usize>, usize) {
            let mut rng = Rng::from_seed_str("replacement-determinism");
            let mut predator = genome_with_enzymes(MAX_CELL_ENZYMES, 10);
            let prey = genome_with_enzymes(7, 200);
            let stats = predator.absorb_predation_enzymes(&prey.enzymes, &mut rng);
            (predator, stats.replacement_slots, stats.replacements)
        }

        let (a_genome, a_slots, a_replacements) = run_once();
        let (b_genome, b_slots, b_replacements) = run_once();
        assert_eq!(a_genome, b_genome);
        assert_eq!(a_slots, b_slots);
        assert_eq!(a_replacements, b_replacements);
        assert_eq!(a_genome.enzymes.len(), MAX_CELL_ENZYMES);

        let mut sorted = a_slots.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), a_slots.len());
    }

    #[test]
    fn catabolase_class_defaults_match_js_harvest_fraction() {
        let mut rng = Rng::from_seed_str("catabolase-defaults");
        let mut enzyme = Enzyme::catabolase_abc(&mut rng);
        enzyme.bond_harvest_fraction = 2.5;
        enzyme.apply_class_defaults(&mut rng);
        assert_eq!(enzyme.bond_harvest_fraction, 1.0);
    }

    #[test]
    fn mutation_preserves_enzyme_count_bounds() {
        let mut rng = Rng::from_seed_str("mutation-bounds");
        let mut genome = Genome::random_founder(&mut rng, 0.0);
        genome.enzymes.truncate(0);
        for _ in 0..100 {
            genome = genome.mutate(&mut rng, 0.0);
            assert!(genome.enzymes.len() >= MIN_CELL_ENZYMES);
            assert!(genome.enzymes.len() <= MAX_CELL_ENZYMES);
        }
    }
}
