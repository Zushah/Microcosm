use crate::chem::{Composition, Element, ELEMENT_COUNT, ELEMENT_ORDER};
use crate::genome::{Enzyme, EnzymeType, Genome};
use crate::molecule::Molecule;
use crate::rng::Rng;

pub const DEFAULT_ENVAL_SIGMA: f32 = 0.18;
pub const DEFAULT_ENVAL_ENERGY_FRACTION: f32 = 2.0 / 3.0;
pub const DEFAULT_ENVAL_RELEASE_FRACTION: f32 = 1.0 / 3.0;
pub const DEFAULT_ENVAL_PUMP: f32 = 0.3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnzymeClass {
    pub max_inputs: usize,
    pub base_rate: f32,
    pub energy_cost: f64,
    pub enval_throughput: f32,
    pub enval_pump: f32,
    pub bond_multiplier: f32,
    pub bond_cost_fraction: f32,
    pub bond_harvest_fraction: f32,
    pub downhill_harvest_fraction: f32,
}

pub const ANABOLASE_CLASS: EnzymeClass = EnzymeClass {
    max_inputs: 3,
    base_rate: 0.85,
    energy_cost: 0.005,
    enval_throughput: 0.18,
    enval_pump: 0.3,
    bond_multiplier: 1.18,
    bond_cost_fraction: 0.70,
    bond_harvest_fraction: 0.0,
    downhill_harvest_fraction: 0.0,
};

pub const CATABOLASE_CLASS: EnzymeClass = EnzymeClass {
    max_inputs: 1,
    base_rate: 0.98,
    energy_cost: 0.006,
    enval_throughput: 0.14,
    enval_pump: 0.3,
    bond_multiplier: 1.0,
    bond_cost_fraction: 0.0,
    bond_harvest_fraction: 1.0,
    downhill_harvest_fraction: 0.0,
};

pub const TRANSMUTASE_CLASS: EnzymeClass = EnzymeClass {
    max_inputs: 1,
    base_rate: 0.30,
    energy_cost: 0.12,
    enval_throughput: 0.04,
    enval_pump: 0.3,
    bond_multiplier: 1.0,
    bond_cost_fraction: 0.0,
    bond_harvest_fraction: 0.0,
    downhill_harvest_fraction: 0.18,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReactionEnv {
    pub tile_enval: f32,
    pub local_enval: f32,
    pub average_enval: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GenomeReactionContext {
    pub optimal_enval: f32,
    pub repro_threshold: f64,
}

impl From<&Genome> for GenomeReactionContext {
    fn from(genome: &Genome) -> Self {
        Self {
            optimal_enval: genome.optimal_enval,
            repro_threshold: genome.repro_threshold,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnvalExchange {
    pub energy_bonus: f64,
    pub enval_input: f32,
    pub enval_output: f32,
    pub delta_enval: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReactionOutcome {
    pub produced: Option<Molecule>,
    pub byproducts: Vec<Molecule>,
    pub energy_delta: f64,
    pub chemical_delta: f64,
    pub enval_energy: f64,
    pub enval_input: f32,
    pub enval_output: f32,
    pub delta_enval: f32,
    pub raw_delta: f64,
}

pub fn enzyme_class(enzyme_type: EnzymeType) -> Option<EnzymeClass> {
    match enzyme_type {
        EnzymeType::Anabolase => Some(ANABOLASE_CLASS),
        EnzymeType::Catabolase => Some(CATABOLASE_CLASS),
        EnzymeType::Transmutase => Some(TRANSMUTASE_CLASS),
        EnzymeType::Defensase | EnzymeType::Attackase => None,
    }
}

pub fn reaction_rate(enzyme: &Enzyme, genome: &Genome, env: ReactionEnv) -> f32 {
    reaction_rate_for_context(enzyme, GenomeReactionContext::from(genome), env)
}

pub fn reaction_rate_for_context(
    enzyme: &Enzyme,
    genome: GenomeReactionContext,
    env: ReactionEnv,
) -> f32 {
    let Some(class) = enzyme_class(enzyme.enzyme_type) else {
        return 0.0;
    };
    (class.base_rate * 1.2).min(1.0) * compute_enval_factor(enzyme, genome, env)
}

pub fn reaction_gate(enzyme: &Enzyme, genome: &Genome, env: ReactionEnv, rng: &mut Rng) -> bool {
    reaction_gate_for_context(enzyme, GenomeReactionContext::from(genome), env, rng)
}

pub fn reaction_gate_for_context(
    enzyme: &Enzyme,
    genome: GenomeReactionContext,
    env: ReactionEnv,
    rng: &mut Rng,
) -> bool {
    let rate = reaction_rate_for_context(enzyme, genome, env);
    rng.next_f64() <= f64::from(rate)
}

pub fn max_inputs(enzyme: &Enzyme) -> usize {
    enzyme_class(enzyme.enzyme_type)
        .map(|class| class.max_inputs)
        .unwrap_or(0)
}

pub fn enzyme_accepts(enzyme: &Enzyme, molecule: &Molecule) -> bool {
    let specificity_mask = enzyme.specificity_mask;
    let molecule_mask = molecule.element_mask;
    if molecule_mask == 0 {
        return false;
    }
    match enzyme.enzyme_type {
        EnzymeType::Catabolase => molecule.size >= 2 && (molecule_mask & !specificity_mask) == 0,
        EnzymeType::Transmutase => (molecule_mask & specificity_mask) != 0,
        EnzymeType::Anabolase => (molecule_mask & !specificity_mask) == 0,
        EnzymeType::Defensase | EnzymeType::Attackase => false,
    }
}

pub fn attempt_reaction(
    enzyme: &Enzyme,
    substrates: &[Molecule],
    genome: &Genome,
    cell_energy: f64,
    env: ReactionEnv,
    rng: &mut Rng,
) -> Option<ReactionOutcome> {
    attempt_reaction_for_context(
        enzyme,
        substrates,
        GenomeReactionContext::from(genome),
        cell_energy,
        env,
        rng,
    )
}

pub fn attempt_reaction_for_context(
    enzyme: &Enzyme,
    substrates: &[Molecule],
    genome: GenomeReactionContext,
    cell_energy: f64,
    env: ReactionEnv,
    rng: &mut Rng,
) -> Option<ReactionOutcome> {
    let class = enzyme_class(enzyme.enzyme_type)?;
    let enval_exchange = compute_enval_exchange(enzyme, class, genome, env, rng);
    let result = match enzyme.enzyme_type {
        EnzymeType::Anabolase => {
            do_anabolase(enzyme, substrates, class, cell_energy, enval_exchange)
        }
        EnzymeType::Catabolase => do_catabolase(enzyme, substrates, class, enval_exchange, rng),
        EnzymeType::Transmutase => do_transmutase(
            enzyme,
            substrates,
            class,
            genome,
            cell_energy,
            env,
            enval_exchange,
            rng,
        ),
        EnzymeType::Defensase | EnzymeType::Attackase => None,
    }?;

    let unchanged_composition =
        composition_delta_is_zero(substrates, &result.produced, &result.byproducts);
    const ENERGY_NOISE: f64 = 1.0e-2;
    if unchanged_composition
        && result.raw_delta.abs() < ENERGY_NOISE
        && f64::from(result.delta_enval).abs() < ENERGY_NOISE
    {
        return None;
    }
    if enzyme.enzyme_type == EnzymeType::Catabolase && result.energy_delta <= 0.0 {
        return None;
    }
    Some(result)
}

fn compute_enval_factor(enzyme: &Enzyme, genome: GenomeReactionContext, env: ReactionEnv) -> f32 {
    let sigma_base = if enzyme.enval_sigma.is_finite() {
        enzyme.enval_sigma
    } else {
        DEFAULT_ENVAL_SIGMA
    };
    let sigma = sigma_base.max(1.0e-6);
    let d = env.local_enval - genome.optimal_enval;
    let denom = 2.0 * sigma * sigma;
    (-(d * d) / denom).exp()
}

fn resolve_enval_polarity(genome: GenomeReactionContext, rng: &mut Rng) -> f32 {
    if genome.optimal_enval > 0.0 {
        1.0
    } else if genome.optimal_enval < 0.0 {
        -1.0
    } else if rng.chance(0.5) {
        -1.0
    } else {
        1.0
    }
}

fn compute_enval_exchange(
    enzyme: &Enzyme,
    class: EnzymeClass,
    genome: GenomeReactionContext,
    env: ReactionEnv,
    rng: &mut Rng,
) -> EnvalExchange {
    let polarity = resolve_enval_polarity(genome, rng);
    let available_aligned = (polarity * env.local_enval).max(0.0);
    let max_input = if enzyme.enval_throughput.is_finite() {
        enzyme.enval_throughput.max(0.0)
    } else {
        class.enval_throughput.max(0.0)
    };
    let input_magnitude = available_aligned.min(max_input);
    let energy_fraction = enzyme.enval_energy_fraction.clamp(0.0, 1.0);
    let release_fraction = enzyme
        .enval_release_fraction
        .min(1.0 - energy_fraction)
        .clamp(0.0, 1.0);
    let pump_magnitude = if enzyme.enval_pump.is_finite() {
        enzyme.enval_pump.max(0.0)
    } else {
        class.enval_pump.max(DEFAULT_ENVAL_PUMP)
    };
    let energy_bonus = f64::from(input_magnitude * energy_fraction);
    let recycled_output_magnitude = input_magnitude * release_fraction;
    let output_magnitude = recycled_output_magnitude + pump_magnitude;
    let enval_input = polarity * input_magnitude;
    let enval_output = -polarity * output_magnitude;
    EnvalExchange {
        energy_bonus,
        enval_input,
        enval_output,
        delta_enval: enval_output - enval_input,
    }
}

fn do_anabolase(
    enzyme: &Enzyme,
    substrates: &[Molecule],
    class: EnzymeClass,
    cell_energy: f64,
    enval: EnvalExchange,
) -> Option<ReactionOutcome> {
    if substrates.len() < 2 {
        return None;
    }

    let mut counts = [0_u16; ELEMENT_COUNT];
    let mut substrate_bond_energy = 0.0_f64;
    for molecule in substrates {
        substrate_bond_energy += f64::from(bond_energy(*molecule));
        for element in ELEMENT_ORDER {
            counts[element.index()] =
                counts[element.index()].checked_add(molecule.composition.count(element))?;
        }
    }

    let composition = Composition::try_new(counts).ok()?;
    let bond_multiplier = if enzyme.bond_multiplier > 0.0 {
        enzyme.bond_multiplier
    } else {
        class.bond_multiplier
    };
    let product = Molecule::new(composition, bond_multiplier).ok()?;
    let product_bond_energy = f64::from(bond_energy(product));
    let additional_bond_storage = (product_bond_energy - substrate_bond_energy).max(0.0);
    let bond_cost_fraction = f64::from(if enzyme.bond_cost_fraction.is_finite() {
        enzyme.bond_cost_fraction.max(0.0)
    } else {
        class.bond_cost_fraction.max(0.0)
    });
    let bond_energy_cost = additional_bond_storage * bond_cost_fraction;
    let catalytic_cost = class.energy_cost;
    let required_energy = bond_energy_cost + catalytic_cost;

    if cell_energy + enval.energy_bonus < required_energy {
        return None;
    }

    let chemical_delta = -bond_energy_cost - catalytic_cost;
    let energy_delta = chemical_delta + enval.energy_bonus;
    Some(ReactionOutcome {
        produced: Some(product),
        byproducts: Vec::new(),
        energy_delta,
        chemical_delta,
        enval_energy: enval.energy_bonus,
        enval_input: enval.enval_input,
        enval_output: enval.enval_output,
        delta_enval: enval.delta_enval,
        raw_delta: chemical_delta,
    })
}

fn do_catabolase(
    enzyme: &Enzyme,
    substrates: &[Molecule],
    class: EnzymeClass,
    enval: EnvalExchange,
    rng: &mut Rng,
) -> Option<ReactionOutcome> {
    let molecule = *substrates.first()?;
    if molecule.size < 2 {
        return None;
    }

    let substrate_bond_energy = f64::from(bond_energy(molecule));
    let byproducts = fragment_composition(molecule.composition, rng)?;
    if byproducts.len() == 1 && byproducts[0].composition == molecule.composition {
        return None;
    }
    let product_bond_energy = byproducts
        .iter()
        .map(|molecule| f64::from(bond_energy(*molecule)))
        .sum::<f64>();
    let bond_energy_released_raw = (substrate_bond_energy - product_bond_energy).max(0.0);
    if bond_energy_released_raw <= 0.0 {
        return None;
    }

    let harvest_fraction = f64::from(
        enzyme
            .bond_harvest_fraction
            .max(class.bond_harvest_fraction)
            .max(1.0),
    );
    let bond_energy_delta = bond_energy_released_raw * harvest_fraction;
    let chemical_delta = bond_energy_delta - class.energy_cost;
    let energy_delta = chemical_delta + enval.energy_bonus;
    Some(ReactionOutcome {
        produced: None,
        byproducts,
        energy_delta,
        chemical_delta,
        enval_energy: enval.energy_bonus,
        enval_input: enval.enval_input,
        enval_output: enval.enval_output,
        delta_enval: enval.delta_enval,
        raw_delta: chemical_delta,
    })
}

fn do_transmutase(
    enzyme: &Enzyme,
    substrates: &[Molecule],
    class: EnzymeClass,
    genome: GenomeReactionContext,
    cell_energy: f64,
    env: ReactionEnv,
    enval: EnvalExchange,
    rng: &mut Rng,
) -> Option<ReactionOutcome> {
    let molecule = *substrates.first()?;
    let mut uphill = Vec::new();
    let mut downhill = Vec::new();

    for element in ELEMENT_ORDER {
        let count = molecule.composition.count(element);
        if count == 0 || enzyme.specificity_mask & element.mask() == 0 {
            continue;
        }
        if let Some(target) = transmutation_up(element) {
            let product = transmuted_molecule(molecule, element, target)?;
            uphill.push(TransmutationCandidate {
                product,
                raw: f64::from(element.properties().energy - target.properties().energy),
                weight: u32::from(count),
            });
        }
        if let Some(target) = transmutation_down(element) {
            let product = transmuted_molecule(molecule, element, target)?;
            downhill.push(TransmutationCandidate {
                product,
                raw: f64::from(element.properties().energy - target.properties().energy),
                weight: u32::from(count),
            });
        }
    }

    if uphill.is_empty() && downhill.is_empty() {
        return None;
    }

    let available_energy = cell_energy + enval.energy_bonus;
    let affordable_uphill: Vec<_> = uphill
        .into_iter()
        .filter(|candidate| available_energy >= class.energy_cost + (-candidate.raw).max(0.0))
        .collect();

    let chosen = if !affordable_uphill.is_empty() && !downhill.is_empty() {
        let uphill_bias =
            compute_transmutase_uphill_bias(enzyme, class, genome, cell_energy, env, rng);
        if rng.chance(uphill_bias) {
            choose_weighted(&affordable_uphill, rng)
        } else {
            choose_weighted(&downhill, rng)
        }
    } else if !affordable_uphill.is_empty() {
        choose_weighted(&affordable_uphill, rng)
    } else {
        choose_weighted(&downhill, rng)
    }?;

    let raw = chosen.raw;
    let downhill_harvest_fraction = if enzyme.downhill_harvest_fraction.is_finite() {
        enzyme.downhill_harvest_fraction
    } else {
        class.downhill_harvest_fraction
    }
    .clamp(0.0, 1.0);
    let transmutation_energy_delta = if raw >= 0.0 {
        raw * f64::from(downhill_harvest_fraction)
    } else {
        raw
    };
    let required_energy = class.energy_cost + (-raw).max(0.0);
    if raw < 0.0 && available_energy < required_energy {
        return None;
    }

    let chemical_delta = transmutation_energy_delta - class.energy_cost;
    let energy_delta = chemical_delta + enval.energy_bonus;
    Some(ReactionOutcome {
        produced: Some(chosen.product),
        byproducts: Vec::new(),
        energy_delta,
        chemical_delta,
        enval_energy: enval.energy_bonus,
        enval_input: enval.enval_input,
        enval_output: enval.enval_output,
        delta_enval: enval.delta_enval,
        raw_delta: chemical_delta,
    })
}

#[derive(Clone, Copy, Debug)]
struct TransmutationCandidate {
    product: Molecule,
    raw: f64,
    weight: u32,
}

fn choose_weighted(
    options: &[TransmutationCandidate],
    rng: &mut Rng,
) -> Option<TransmutationCandidate> {
    if options.is_empty() {
        return None;
    }
    let total = options.iter().map(|option| option.weight).sum::<u32>();
    if total == 0 {
        return Some(options[rng.usize(options.len())]);
    }
    let mut r = rng.next_f64() * f64::from(total);
    for option in options {
        r -= f64::from(option.weight);
        if r <= 0.0 {
            return Some(*option);
        }
    }
    options.last().copied()
}

fn compute_transmutase_uphill_bias(
    enzyme: &Enzyme,
    class: EnzymeClass,
    genome: GenomeReactionContext,
    cell_energy: f64,
    env: ReactionEnv,
    rng: &mut Rng,
) -> f32 {
    let polarity = resolve_enval_polarity(genome, rng);
    let aligned_enval = (polarity * env.local_enval).max(0.0);
    let throughput = if enzyme.enval_throughput.is_finite() {
        enzyme.enval_throughput
    } else {
        class.enval_throughput
    }
    .max(1.0e-6);
    let aligned_frac = (aligned_enval / throughput).clamp(0.0, 1.0);
    let reserve_target = (genome.repro_threshold * 0.5).max(0.25);
    let energy_frac = (cell_energy / reserve_target).clamp(0.0, 1.0) as f32;
    (0.04 + 0.35 * aligned_frac + 0.20 * energy_frac).clamp(0.0, 1.0)
}

fn fragment_composition(composition: Composition, rng: &mut Rng) -> Option<Vec<Molecule>> {
    let mut atoms = Vec::new();
    for element in ELEMENT_ORDER {
        for _ in 0..composition.count(element) {
            atoms.push(element);
        }
    }
    if atoms.len() < 2 {
        return None;
    }
    rng.shuffle_in_place(&mut atoms);
    let first_count = ((atoms.len() as f64) * 0.6).round() as usize;
    let first_count = first_count.max(1).min(atoms.len() - 1);
    let mut first = [0_u16; ELEMENT_COUNT];
    let mut second = [0_u16; ELEMENT_COUNT];
    for element in atoms.iter().take(first_count) {
        first[element.index()] += 1;
    }
    for element in atoms.iter().skip(first_count) {
        second[element.index()] += 1;
    }
    let mut out = Vec::with_capacity(2);
    if let Ok(composition) = Composition::try_new(first) {
        out.push(Molecule::new(composition, 1.0).ok()?);
    }
    if let Ok(composition) = Composition::try_new(second) {
        out.push(Molecule::new(composition, 1.0).ok()?);
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

fn transmuted_molecule(molecule: Molecule, source: Element, target: Element) -> Option<Molecule> {
    let mut counts = *molecule.composition.counts();
    if counts[source.index()] == 0 {
        return None;
    }
    counts[source.index()] -= 1;
    counts[target.index()] = counts[target.index()].checked_add(1)?;
    let composition = Composition::try_new(counts).ok()?;
    Molecule::new(composition, molecule.bond_multiplier).ok()
}

fn transmutation_up(element: Element) -> Option<Element> {
    match element {
        Element::F => Some(Element::A),
        Element::A => Some(Element::C),
        Element::C => Some(Element::B),
        Element::B => Some(Element::E),
        Element::E => Some(Element::D),
        Element::D => None,
    }
}

fn transmutation_down(element: Element) -> Option<Element> {
    match element {
        Element::A | Element::B | Element::C | Element::D | Element::E => Some(Element::F),
        Element::F => None,
    }
}

pub fn bond_energy(molecule: Molecule) -> f32 {
    (molecule.energy - molecule.elemental_energy_sum).max(0.0)
}

fn composition_delta_is_zero(
    consumed: &[Molecule],
    produced: &Option<Molecule>,
    byproducts: &[Molecule],
) -> bool {
    let mut counts = [0_i32; ELEMENT_COUNT];
    for molecule in consumed {
        for element in ELEMENT_ORDER {
            counts[element.index()] -= i32::from(molecule.composition.count(element));
        }
    }
    if let Some(molecule) = produced {
        for element in ELEMENT_ORDER {
            counts[element.index()] += i32::from(molecule.composition.count(element));
        }
    }
    for molecule in byproducts {
        for element in ELEMENT_ORDER {
            counts[element.index()] += i32::from(molecule.composition.count(element));
        }
    }
    counts.iter().all(|count| *count == 0)
}

#[cfg(test)]
mod tests {
    use super::{attempt_reaction, enzyme_accepts, ReactionEnv};
    use crate::chem::{Composition, Element};
    use crate::genome::{Enzyme, Genome};
    use crate::molecule::Molecule;
    use crate::rng::Rng;

    #[test]
    fn metabolic_specificity_accepts_expected_molecules() {
        let mut rng = Rng::from_seed_str("accepts");
        let enzyme = Enzyme::anabolase_abc(&mut rng);
        let a = Molecule::new(Composition::single(Element::A), 1.0).unwrap();
        let d = Molecule::new(Composition::single(Element::D), 1.0).unwrap();
        assert!(enzyme_accepts(&enzyme, &a));
        assert!(!enzyme_accepts(&enzyme, &d));
    }

    #[test]
    fn anabolase_reaction_builds_product() {
        let mut rng = Rng::from_seed_str("anabolase-basic");
        let genome = Genome::random_founder(&mut rng, 0.1);
        let enzyme = genome.enzymes[0];
        let substrates = [
            Molecule::new(Composition::single(Element::A), 1.0).unwrap(),
            Molecule::new(Composition::single(Element::B), 1.0).unwrap(),
        ];
        let result = attempt_reaction(
            &enzyme,
            &substrates,
            &genome,
            10.0,
            ReactionEnv {
                tile_enval: 0.1,
                local_enval: 0.1,
                average_enval: 0.1,
            },
            &mut rng,
        )
        .unwrap();
        let product = result.produced.unwrap();
        assert_eq!(product.composition.count(Element::A), 1);
        assert_eq!(product.composition.count(Element::B), 1);
    }
}
