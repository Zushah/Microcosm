use crate::chem::{ELEMENT_COUNT, ELEMENT_ORDER, ElementAmounts};
use crate::genome::{Enzyme, EnzymeType, Genome};
use crate::rng::Rng;

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
pub struct FluxOutcome {
    pub requested_extent: f32,
    pub executed_extent: f32,
    pub consumed: ElementAmounts,
    pub retained_products: ElementAmounts,
    pub secreted_products: ElementAmounts,
    pub element_deltas: [f32; ELEMENT_COUNT],
    pub energy_delta: f64,
    pub raw_chemical_energy: f64,
    pub enval_energy: f64,
    pub enval_input: f32,
    pub enval_output: f32,
}

pub fn compute_enval_factor(
    enzyme: &Enzyme,
    genome: GenomeReactionContext,
    env: ReactionEnv,
) -> f32 {
    if enzyme.enzyme_type != EnzymeType::Metabolic {
        return 0.0;
    }
    let sigma = enzyme.enval_sigma.max(1.0e-6);
    let distance = env.local_enval - genome.optimal_enval;
    (-(distance * distance) / (2.0 * sigma * sigma)).exp()
}

pub fn compute_flux(
    enzyme: &Enzyme,
    reservoir: ElementAmounts,
    cell_energy: f64,
    dt_seconds: f64,
    genome: GenomeReactionContext,
    env: ReactionEnv,
    rng: &mut Rng,
) -> Option<FluxOutcome> {
    if enzyme.enzyme_type != EnzymeType::Metabolic
        || enzyme.validate().is_err()
        || !cell_energy.is_finite()
        || cell_energy < 0.0
        || !dt_seconds.is_finite()
        || dt_seconds <= 0.0
    {
        return None;
    }

    let mut substrate_limit = f64::INFINITY;
    for element in ELEMENT_ORDER {
        let coefficient = f64::from(enzyme.reactants[element]);
        if coefficient > 0.0 {
            substrate_limit = substrate_limit.min(f64::from(reservoir[element]) / coefficient);
        }
    }
    if !substrate_limit.is_finite() || substrate_limit <= 0.0 {
        return None;
    }

    let enval_factor = f64::from(compute_enval_factor(enzyme, genome, env));
    let nominal_extent = f64::from(enzyme.rate) * dt_seconds * enval_factor;
    let requested_extent = nominal_extent.min(substrate_limit).max(0.0);
    if requested_extent <= 1.0e-12 {
        return None;
    }

    let reactant_energy = enzyme.reactants.intrinsic_energy();
    let product_energy = enzyme.products.intrinsic_energy();
    let energy_deficit_per_extent = (product_energy - reactant_energy).max(0.0);
    let polarity = resolve_enval_polarity(genome.optimal_enval, rng);
    let aligned_available = f64::from((polarity * env.local_enval).max(0.0));

    let supported = |extent: f64| {
        let input_magnitude = aligned_available.min(f64::from(enzyme.enval_throughput) * extent);
        let enval_energy = input_magnitude * f64::from(enzyme.enval_energy_fraction);
        energy_deficit_per_extent * extent <= cell_energy + enval_energy + 1.0e-12
    };

    let executed_extent = if energy_deficit_per_extent > 0.0 && !supported(requested_extent) {
        let mut low = 0.0;
        let mut high = requested_extent;
        for _ in 0..48 {
            let midpoint = (low + high) * 0.5;
            if supported(midpoint) {
                low = midpoint;
            } else {
                high = midpoint;
            }
        }
        low
    } else {
        requested_extent
    };
    if executed_extent <= 1.0e-12 {
        return None;
    }

    let input_magnitude =
        aligned_available.min(f64::from(enzyme.enval_throughput) * executed_extent);
    let enval_energy = input_magnitude * f64::from(enzyme.enval_energy_fraction);
    let output_magnitude = input_magnitude * f64::from(enzyme.enval_release_fraction)
        + f64::from(enzyme.enval_pump) * executed_extent;
    let raw_chemical_energy = (reactant_energy - product_energy) * executed_extent;
    let chemical_energy_delta = if raw_chemical_energy >= 0.0 {
        raw_chemical_energy * f64::from(enzyme.energy_harvest_fraction)
    } else {
        raw_chemical_energy
    };
    let energy_delta = chemical_energy_delta + enval_energy;

    let mut consumed = ElementAmounts::ZERO;
    let mut retained_products = ElementAmounts::ZERO;
    let mut secreted_products = ElementAmounts::ZERO;
    let mut element_deltas = [0.0_f32; ELEMENT_COUNT];
    for element in ELEMENT_ORDER {
        let consumed_value = f64::from(enzyme.reactants[element]) * executed_extent;
        let produced_value = f64::from(enzyme.products[element]) * executed_extent;
        let secreted_value = produced_value * f64::from(enzyme.secretion_fraction);
        consumed[element] = consumed_value as f32;
        secreted_products[element] = secreted_value as f32;
        retained_products[element] = (produced_value - secreted_value) as f32;
        element_deltas[element.index()] = (produced_value - consumed_value) as f32;
    }

    Some(FluxOutcome {
        requested_extent: requested_extent as f32,
        executed_extent: executed_extent as f32,
        consumed,
        retained_products,
        secreted_products,
        element_deltas,
        energy_delta,
        raw_chemical_energy,
        enval_energy,
        enval_input: (f64::from(polarity) * input_magnitude) as f32,
        enval_output: (-f64::from(polarity) * output_magnitude) as f32,
    })
}

fn resolve_enval_polarity(optimal_enval: f32, rng: &mut Rng) -> f32 {
    if optimal_enval > 0.0 {
        1.0
    } else if optimal_enval < 0.0 {
        -1.0
    } else if rng.chance(0.5) {
        -1.0
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::{GenomeReactionContext, ReactionEnv, compute_enval_factor, compute_flux};
    use crate::chem::{Element, ElementAmounts};
    use crate::genome::Enzyme;
    use crate::rng::Rng;

    fn context() -> GenomeReactionContext {
        GenomeReactionContext {
            optimal_enval: 0.25,
            repro_threshold: 10.0,
        }
    }

    fn environment(local_enval: f32) -> ReactionEnv {
        ReactionEnv {
            tile_enval: -0.8,
            local_enval,
            average_enval: 0.0,
        }
    }

    #[test]
    fn enval_factor_uses_local_average_not_tile_value() {
        let enzyme = Enzyme::founder_downhill();
        let matched = compute_enval_factor(&enzyme, context(), environment(0.25));
        let mismatched = compute_enval_factor(&enzyme, context(), environment(-0.25));
        assert!((matched - 1.0).abs() <= 1.0e-6);
        assert!(mismatched < matched);
    }

    #[test]
    fn flux_is_deterministic_substrate_limited_and_scalar_conservative() {
        let enzyme = Enzyme::metabolic(
            ElementAmounts::new([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ElementAmounts::new([0.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
            100.0,
            0.5,
            0.25,
        );
        let reservoir = ElementAmounts::new([0.5, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let mut first_rng = Rng::from_seed_str("flux-determinism");
        let mut second_rng = Rng::from_seed_str("flux-determinism");
        let first = compute_flux(
            &enzyme,
            reservoir,
            10.0,
            1.0,
            context(),
            environment(0.25),
            &mut first_rng,
        )
        .unwrap();
        let second = compute_flux(
            &enzyme,
            reservoir,
            10.0,
            1.0,
            context(),
            environment(0.25),
            &mut second_rng,
        )
        .unwrap();
        assert_eq!(first, second);
        assert!((first.executed_extent - 0.25).abs() <= 1.0e-6);
        assert!((first.consumed.total() - 0.5).abs() <= 1.0e-6);
        assert!(
            (first.retained_products.total() + first.secreted_products.total()
                - first.consumed.total())
            .abs()
                <= 1.0e-6
        );
    }

    #[test]
    fn downhill_flux_harvests_intrinsic_energy() {
        let enzyme = Enzyme::founder_downhill();
        let mut reservoir = ElementAmounts::ZERO;
        reservoir[Element::D] = 1.0;
        let mut rng = Rng::from_seed_str("downhill-flux");
        let outcome = compute_flux(
            &enzyme,
            reservoir,
            0.0,
            0.1,
            context(),
            environment(0.0),
            &mut rng,
        )
        .unwrap();
        assert!(outcome.raw_chemical_energy > 0.0);
        assert!(outcome.energy_delta > 0.0);
    }

    #[test]
    fn uphill_flux_is_energy_limited_and_never_overdraws_energy() {
        let mut enzyme = Enzyme::founder_reshape();
        enzyme.rate = 10.0;
        enzyme.enval_throughput = 0.0;
        enzyme.enval_pump = 0.0;
        let reservoir = ElementAmounts::new([10.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let mut rng = Rng::from_seed_str("uphill-flux");
        let outcome = compute_flux(
            &enzyme,
            reservoir,
            0.1,
            1.0,
            context(),
            environment(0.0),
            &mut rng,
        )
        .unwrap();
        assert!(outcome.executed_extent < outcome.requested_extent);
        assert!(outcome.energy_delta < 0.0);
        assert!(0.1 + outcome.energy_delta >= -1.0e-10);
    }

    #[test]
    fn zero_optimum_uses_seeded_rng_polarity_and_nonzero_optima_do_not_consume_rng() {
        let enzyme = Enzyme::founder_downhill();
        let mut reservoir = ElementAmounts::ZERO;
        reservoir[Element::D] = 1.0;
        let zero_context = GenomeReactionContext {
            optimal_enval: 0.0,
            repro_threshold: 10.0,
        };
        let mut expected_rng = Rng::from_seed_str("zero-optimum-polarity");
        let expected_polarity = if expected_rng.chance(0.5) { -1.0 } else { 1.0 };
        let expected_next = expected_rng.next_f64();
        let mut actual_rng = Rng::from_seed_str("zero-optimum-polarity");

        let outcome = compute_flux(
            &enzyme,
            reservoir,
            0.0,
            0.1,
            zero_context,
            environment(0.25),
            &mut actual_rng,
        )
        .unwrap();

        assert_eq!(outcome.enval_output.signum(), -expected_polarity);
        assert_eq!(actual_rng.next_f64().to_bits(), expected_next.to_bits());

        for (optimal_enval, local_enval) in [(0.25, 0.25), (-0.25, -0.25)] {
            let mut actual_rng = Rng::from_seed_str("nonzero-optimum-polarity");
            let mut untouched_rng = actual_rng.clone();
            compute_flux(
                &enzyme,
                reservoir,
                0.0,
                0.1,
                GenomeReactionContext {
                    optimal_enval,
                    repro_threshold: 10.0,
                },
                environment(local_enval),
                &mut actual_rng,
            )
            .unwrap();
            assert_eq!(
                actual_rng.next_f64().to_bits(),
                untouched_rng.next_f64().to_bits()
            );
        }
    }
}
