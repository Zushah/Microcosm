use crate::chem::{ELEMENT_COUNT, ELEMENT_ORDER, ElementAmounts};
use crate::genome::{Enzyme, EnzymeType, Genome};
use crate::rng::Rng;

pub const ENVAL_PUMP_ENERGY_COST: f64 = 1.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReactionEnv {
    pub local_enval: f32,
    pub cell_area: f32,
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
    pub chemical_energy_delta: f64,
    pub enval_energy: f64,
    pub pump_cost: f64,
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

pub fn compute_saturation(enzyme: &Enzyme, reservoir: ElementAmounts, cell_area: f32) -> f64 {
    let area = f64::from(cell_area);
    if enzyme.enzyme_type != EnzymeType::Metabolic || !area.is_finite() || area <= 0.0 {
        return 0.0;
    }
    let mut limiting = f64::INFINITY;
    for element in ELEMENT_ORDER {
        let coefficient = f64::from(enzyme.reactants[element]);
        if coefficient > 0.0 {
            limiting = limiting.min(f64::from(reservoir[element]) / area / coefficient);
        }
    }
    if !limiting.is_finite() || limiting <= 0.0 {
        return 0.0;
    }
    limiting / (f64::from(enzyme.half_saturation) + limiting)
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
    let saturation = compute_saturation(enzyme, reservoir, env.cell_area);
    let nominal_extent = f64::from(enzyme.rate) * dt_seconds * enval_factor * saturation;
    let requested_extent = nominal_extent.min(substrate_limit).max(0.0);
    if requested_extent <= 1.0e-12 {
        return None;
    }

    let reactant_energy = enzyme.reactants.intrinsic_energy();
    let product_energy = enzyme.products.intrinsic_energy();
    let energy_deficit_per_extent = (product_energy - reactant_energy).max(0.0);
    let pump_cost_per_extent = ENVAL_PUMP_ENERGY_COST * f64::from(enzyme.enval_pump);
    let cost_per_extent = energy_deficit_per_extent + pump_cost_per_extent;
    let polarity = resolve_enval_polarity(genome.optimal_enval, rng);
    let aligned_available = f64::from((polarity * env.local_enval).max(0.0));

    let supported = |extent: f64| {
        let input_magnitude = aligned_available.min(f64::from(enzyme.enval_throughput) * extent);
        let enval_energy = input_magnitude * f64::from(enzyme.enval_energy_fraction);
        cost_per_extent * extent <= cell_energy + enval_energy
    };

    let executed_extent = if cost_per_extent > 0.0 && !supported(requested_extent) {
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
    let pump_cost = pump_cost_per_extent * executed_extent;
    let energy_delta = chemical_energy_delta + enval_energy - pump_cost;

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
        chemical_energy_delta,
        enval_energy,
        pump_cost,
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
    use super::{
        GenomeReactionContext, ReactionEnv, compute_enval_factor, compute_flux, compute_saturation,
    };
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
            local_enval,
            cell_area: 1.0,
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
    fn requested_extent_saturates_monotonically_and_is_half_at_the_half_saturation_point() {
        let mut enzyme = Enzyme::founder_downhill();
        enzyme.enval_pump = 0.0;
        enzyme.half_saturation = 0.2;
        let area = 0.5_f32;
        let dt = 0.01;
        let ceiling = f64::from(enzyme.rate) * dt;
        let requested_at = |concentration: f32| {
            let mut reservoir = ElementAmounts::ZERO;
            reservoir[Element::D] = concentration * area;
            let mut rng = Rng::from_seed_str("saturation");
            compute_flux(
                &enzyme,
                reservoir,
                10.0,
                dt,
                context(),
                ReactionEnv {
                    local_enval: 0.25,
                    cell_area: area,
                },
                &mut rng,
            )
            .map_or(0.0, |outcome| f64::from(outcome.requested_extent))
        };

        let mut previous = 0.0;
        for step in 1..=60 {
            let requested = requested_at(step as f32 * 0.05);
            assert!(requested > previous, "not monotonic at step {step}");
            assert!(requested < ceiling);
            previous = requested;
        }
        let at_half = requested_at(0.2);
        assert!((at_half - 0.5 * ceiling).abs() <= 1.0e-6 * ceiling.max(1.0));

        let mut reservoir = ElementAmounts::ZERO;
        reservoir[Element::D] = 0.2 * area;
        assert!((compute_saturation(&enzyme, reservoir, area) - 0.5).abs() <= 1.0e-6);
    }

    #[test]
    fn saturation_uses_the_limiting_stoichiometric_reactant() {
        let mut enzyme = Enzyme::metabolic(
            ElementAmounts::new([2.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            ElementAmounts::new([0.0, 0.0, 3.0, 0.0, 0.0, 0.0]),
            1.0,
            0.5,
            0.0,
        );
        enzyme.half_saturation = 1.0;
        let reservoir = ElementAmounts::new([4.0, 4.0, 0.0, 0.0, 0.0, 0.0]);
        let saturation = compute_saturation(&enzyme, reservoir, 2.0);
        assert!((saturation - 0.5).abs() <= 1.0e-9);
    }

    #[test]
    fn enzyme_with_an_absent_reactant_requests_nothing() {
        let enzyme = Enzyme::founder_downhill();
        let reservoir = ElementAmounts::new([5.0, 5.0, 5.0, 0.0, 5.0, 5.0]);
        assert_eq!(compute_saturation(&enzyme, reservoir, 0.6), 0.0);
        let mut rng = Rng::from_seed_str("absent-reactant");
        assert!(
            compute_flux(
                &enzyme,
                reservoir,
                1.0,
                0.01,
                context(),
                environment(0.25),
                &mut rng,
            )
            .is_none()
        );
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
            1.0,
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
        let mut enzyme = Enzyme::founder_engine_reverse();
        enzyme.rate = 100.0;
        enzyme.enval_throughput = 0.0;
        enzyme.enval_pump = 0.0;
        let reservoir = ElementAmounts::new([0.0, 0.0, 10.0, 0.0, 0.0, 0.0]);
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
    fn pumping_costs_one_energy_per_unit_and_is_limited_by_what_the_cell_can_pay() {
        let mut enzyme = Enzyme::metabolic(
            ElementAmounts::new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            ElementAmounts::new([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            10.0,
            0.0,
            0.0,
        );
        enzyme.enval_pump = 0.5;
        enzyme.enval_throughput = 0.0;
        enzyme.enval_sigma = 1000.0;
        let reservoir = ElementAmounts::new([0.0, 100.0, 0.0, 0.0, 0.0, 0.0]);

        let mut rng = Rng::from_seed_str("pump-cost-rich");
        let rich = compute_flux(
            &enzyme,
            reservoir,
            100.0,
            0.01,
            context(),
            environment(0.25),
            &mut rng,
        )
        .unwrap();
        let extent = f64::from(rich.executed_extent);
        assert!((rich.pump_cost - 0.5 * extent).abs() <= 1.0e-7 * rich.pump_cost);
        assert!((rich.energy_delta + rich.pump_cost).abs() <= 1.0e-12);
        assert!(
            (f64::from(rich.enval_output.abs()) - 0.5 * extent).abs() <= 1.0e-6,
            "pumped enval must match the paid amount"
        );

        let mut rng = Rng::from_seed_str("pump-cost-poor");
        let poor = compute_flux(
            &enzyme,
            reservoir,
            0.01,
            0.01,
            context(),
            environment(0.25),
            &mut rng,
        )
        .unwrap();
        assert!(poor.executed_extent < poor.requested_extent);
        assert!(poor.pump_cost <= 0.01 + 1.0e-12);
        assert!(0.01 + poor.energy_delta >= -1.0e-12);

        let mut rng = Rng::from_seed_str("pump-cost-broke");
        assert!(
            compute_flux(
                &enzyme,
                reservoir,
                0.0,
                0.01,
                context(),
                environment(0.25),
                &mut rng,
            )
            .is_none()
        );
    }

    #[test]
    fn pump_and_deficit_can_be_paid_from_this_reactions_enval_harvest() {
        let mut enzyme = Enzyme::metabolic(
            ElementAmounts::new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            ElementAmounts::new([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            1.0,
            0.0,
            0.0,
        );
        enzyme.enval_pump = 0.1;
        enzyme.enval_throughput = 1.0;
        enzyme.enval_energy_fraction = 0.5;
        enzyme.enval_release_fraction = 0.0;
        enzyme.enval_sigma = 1000.0;
        let reservoir = ElementAmounts::new([0.0, 100.0, 0.0, 0.0, 0.0, 0.0]);
        let mut rng = Rng::from_seed_str("pump-paid-by-harvest");
        let outcome = compute_flux(
            &enzyme,
            reservoir,
            0.0,
            0.01,
            context(),
            environment(5.0),
            &mut rng,
        )
        .unwrap();
        assert!(outcome.pump_cost > 0.0);
        assert!(outcome.enval_energy > outcome.pump_cost);
        assert!(outcome.energy_delta > 0.0);
        assert!(
            (outcome.energy_delta
                - (outcome.chemical_energy_delta + outcome.enval_energy - outcome.pump_cost))
                .abs()
                <= 1.0e-12
        );
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
            1.0,
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
                1.0,
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
