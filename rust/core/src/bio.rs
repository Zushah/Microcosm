use crate::chem::{ELEMENT_COUNT, ELEMENT_ORDER, ElementAmounts};
use crate::config::CellBodyConfig;
use crate::genome::Enzyme;
use crate::spatial::REFERENCE_CELL_AREA;

pub const ENVAL_PUMP_ENERGY_COST: f64 = 1.0;

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReactionEnv {
    pub local_enval: f32,
    pub cell_area: f32,
    pub cell_body: CellBodyConfig,
    pub capacity_fraction: f64,
    pub pool_extent: Option<f32>,
}

#[cfg(test)]
impl ReactionEnv {
    pub fn internal(local_enval: f32, cell_area: f32, cell_body: CellBodyConfig) -> Self {
        Self {
            local_enval,
            cell_area,
            cell_body,
            capacity_fraction: 1.0,
            pool_extent: None,
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
    pub body_products: ElementAmounts,
    pub boundary_products: ElementAmounts,
    pub assembly_cost: f64,
    pub element_deltas: [f32; ELEMENT_COUNT],
    pub energy_delta: f64,
    pub raw_chemical_energy: f64,
    pub chemical_energy_delta: f64,
    pub enval_energy: f64,
    pub pump_cost: f64,
    pub enval_input: f32,
    pub enval_output: f32,
}

#[cfg(test)]
pub fn compute_enval_factor(enzyme: &Enzyme, env: ReactionEnv) -> f32 {
    compute_enval_factor_at(enzyme, env.local_enval)
}

#[inline]
fn compute_enval_factor_at(enzyme: &Enzyme, local_enval: f32) -> f32 {
    let sigma = enzyme.enval_sigma.max(1.0e-6);
    let distance = local_enval - enzyme.enval_optimum;
    (-(distance * distance) / (2.0 * sigma * sigma)).exp()
}

#[cfg(test)]
fn compute_saturation(enzyme: &Enzyme, reservoir: ElementAmounts, pool_extent: f32) -> f64 {
    let extent = f64::from(pool_extent);
    if !extent.is_finite() || extent <= 0.0 {
        return 0.0;
    }
    let mut limiting = f64::INFINITY;
    for element in ELEMENT_ORDER {
        let coefficient = f64::from(enzyme.reactants[element]);
        if coefficient > 0.0 {
            limiting = limiting.min(f64::from(reservoir[element]) / extent / coefficient);
        }
    }
    if !limiting.is_finite() || limiting <= 0.0 {
        return 0.0;
    }
    limiting / (f64::from(enzyme.half_saturation) + limiting)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FluxExtent {
    pub requested_extent: f64,
    pub executed_extent: f64,
    pub secretion_fraction: f64,
    pub body_fraction: f64,
    pub boundary_fraction: f64,
    pub retained_fraction: f64,
    pub assembly_cost: f64,
    pub energy_delta: f64,
    pub raw_chemical_energy: f64,
    pub chemical_energy_delta: f64,
    pub enval_energy: f64,
    pub pump_cost: f64,
    pub enval_input: f32,
    pub enval_output: f32,
}

impl FluxExtent {
    pub fn executed_extent_f32(&self) -> f32 {
        self.executed_extent as f32
    }

    pub fn consumed(&self, coefficient: f32) -> f32 {
        (f64::from(coefficient) * self.executed_extent) as f32
    }

    pub fn outcome(&self, enzyme: &Enzyme) -> FluxOutcome {
        let mut consumed = ElementAmounts::ZERO;
        let mut retained_products = ElementAmounts::ZERO;
        let mut secreted_products = ElementAmounts::ZERO;
        let mut body_products = ElementAmounts::ZERO;
        let mut boundary_products = ElementAmounts::ZERO;
        let mut element_deltas = [0.0_f32; ELEMENT_COUNT];
        for element in ELEMENT_ORDER {
            let consumed_value = f64::from(enzyme.reactants[element]) * self.executed_extent;
            let produced_value = f64::from(enzyme.products[element]) * self.executed_extent;
            consumed[element] = consumed_value as f32;
            secreted_products[element] = (produced_value * self.secretion_fraction) as f32;
            body_products[element] = (produced_value * self.body_fraction) as f32;
            boundary_products[element] = (produced_value * self.boundary_fraction) as f32;
            retained_products[element] = (produced_value * self.retained_fraction) as f32;
            element_deltas[element.index()] = (produced_value - consumed_value) as f32;
        }
        FluxOutcome {
            requested_extent: self.requested_extent as f32,
            executed_extent: self.executed_extent as f32,
            consumed,
            retained_products,
            secreted_products,
            body_products,
            boundary_products,
            assembly_cost: self.assembly_cost,
            element_deltas,
            energy_delta: self.energy_delta,
            raw_chemical_energy: self.raw_chemical_energy,
            chemical_energy_delta: self.chemical_energy_delta,
            enval_energy: self.enval_energy,
            pump_cost: self.pump_cost,
            enval_input: self.enval_input,
            enval_output: self.enval_output,
        }
    }
}

#[cfg(test)]
pub fn compute_flux(
    enzyme: &Enzyme,
    reservoir: ElementAmounts,
    cell_energy: f64,
    dt_seconds: f64,
    env: ReactionEnv,
) -> Option<FluxOutcome> {
    if !dt_seconds.is_finite() || dt_seconds <= 0.0 {
        return None;
    }
    let cell = FluxCell::new(env.local_enval, env.cell_area, env.cell_body);
    let terms = EnzymeTerms::new(enzyme, &cell, dt_seconds);
    let pool_extent = f64::from(env.pool_extent.unwrap_or(env.cell_area));
    compute_extent(
        &terms,
        enzyme,
        &reservoir,
        cell_energy,
        env.capacity_fraction,
        pool_extent,
        &cell,
    )
    .map(|flux| flux.outcome(enzyme))
}

#[derive(Clone, Copy, Debug)]
pub struct FluxCell {
    pub local_enval: f32,
    pub cell_area: f64,
    area_scale: f64,
    assembly_half_saturation: f64,
    assembly_energy_per_unit: f64,
}

impl FluxCell {
    pub fn new(local_enval: f32, cell_area: f32, cell_body: CellBodyConfig) -> Self {
        Self {
            local_enval,
            cell_area: f64::from(cell_area),
            area_scale: f64::from(cell_area / REFERENCE_CELL_AREA),
            assembly_half_saturation: f64::from(cell_body.assembly_half_saturation),
            assembly_energy_per_unit: cell_body.assembly_energy_per_unit,
        }
    }
}

#[inline]
pub fn mask_elements(mut mask: u8) -> impl Iterator<Item = usize> {
    std::iter::from_fn(move || {
        if mask == 0 {
            return None;
        }
        let index = mask.trailing_zeros() as usize;
        mask &= mask - 1;
        Some(index)
    })
}

#[inline]
fn element_mask(values: &[f32; ELEMENT_COUNT], present: impl Fn(f32) -> bool) -> u8 {
    let mut mask = 0_u8;
    for (index, value) in values.iter().enumerate() {
        mask |= u8::from(present(*value)) << index;
    }
    mask
}

#[derive(Clone, Copy, Debug)]
pub struct EnzymeTerms {
    pub reactant_mask: u8,
    pub product_mask: u8,
    rate_extent: f64,
    half_saturation: f64,
    reactant_energy: f64,
    product_energy: f64,
    product_total: f64,
    polarity: f64,
    pump_per_extent: f64,
    aligned_available: f64,
    throughput: f64,
    energy_fraction: f64,
    release_fraction: f64,
    harvest_fraction: f64,
    secretion_fraction: f64,
    body_fraction: f64,
    boundary_fraction: f64,
}

impl EnzymeTerms {
    #[inline]
    pub fn new(enzyme: &Enzyme, cell: &FluxCell, dt_seconds: f64) -> Self {
        let enval_factor = f64::from(compute_enval_factor_at(enzyme, cell.local_enval));
        let polarity = enval_polarity(enzyme.enval_optimum);
        Self {
            reactant_mask: element_mask(enzyme.reactants.as_array(), |value| value > 0.0),
            product_mask: element_mask(enzyme.products.as_array(), |value| value != 0.0),
            rate_extent: f64::from(enzyme.rate) * dt_seconds * enval_factor,
            half_saturation: f64::from(enzyme.half_saturation),
            reactant_energy: enzyme.reactants.intrinsic_energy(),
            product_energy: enzyme.products.intrinsic_energy(),
            product_total: enzyme.products.total(),
            polarity: f64::from(polarity),
            pump_per_extent: f64::from(enzyme.enval_pump) * f64::from(polarity.abs()),
            aligned_available: f64::from((polarity * cell.local_enval).max(0.0)),
            throughput: f64::from(enzyme.enval_throughput),
            energy_fraction: f64::from(enzyme.enval_energy_fraction),
            release_fraction: f64::from(enzyme.enval_release_fraction),
            harvest_fraction: f64::from(enzyme.energy_harvest_fraction),
            secretion_fraction: f64::from(enzyme.secretion_fraction),
            body_fraction: f64::from(enzyme.body_fraction),
            boundary_fraction: f64::from(enzyme.boundary_fraction),
        }
    }
}

#[inline(always)]
pub fn compute_extent(
    terms: &EnzymeTerms,
    enzyme: &Enzyme,
    reservoir: &ElementAmounts,
    cell_energy: f64,
    capacity_fraction: f64,
    pool_extent: f64,
    cell: &FluxCell,
) -> Option<FluxExtent> {
    let reactive = (0.0..f64::INFINITY).contains(&cell_energy) && capacity_fraction > 0.0;
    if !reactive {
        return None;
    }

    let reactants = enzyme.reactants.as_array();
    let available_amounts = reservoir.as_array();
    let mut substrate_limit = f64::INFINITY;
    let mut limiting = f64::INFINITY;
    for index in mask_elements(terms.reactant_mask) {
        let coefficient = reactants[index];
        let available = f64::from(available_amounts[index]);
        let per_extent = available / pool_extent;
        let (by_amount, by_concentration) = if coefficient == 1.0 {
            (available, per_extent)
        } else {
            let coefficient = f64::from(coefficient);
            (available / coefficient, per_extent / coefficient)
        };
        if by_amount < substrate_limit {
            substrate_limit = by_amount;
        }
        if by_concentration < limiting {
            limiting = by_concentration;
        }
    }
    if !(substrate_limit > 0.0 && substrate_limit < f64::INFINITY) {
        return None;
    }

    let saturates = pool_extent > 0.0
        && pool_extent < f64::INFINITY
        && limiting > 0.0
        && limiting < f64::INFINITY;
    let saturation = if saturates {
        limiting / (terms.half_saturation + limiting)
    } else {
        0.0
    };
    let nominal_extent = terms.rate_extent * saturation * cell.area_scale * capacity_fraction;
    let requested_extent = nominal_extent.min(substrate_limit).max(0.0);
    if requested_extent <= 1.0e-12 {
        return None;
    }

    let energy_deficit_per_extent = (terms.product_energy - terms.reactant_energy).max(0.0);
    let pump_cost_per_extent = ENVAL_PUMP_ENERGY_COST * terms.pump_per_extent;
    let energy_density = cell_energy / cell.cell_area;
    let assembly_saturation = energy_density / (cell.assembly_half_saturation + energy_density);
    let body_fraction = terms.body_fraction * assembly_saturation;
    let boundary_fraction = terms.boundary_fraction * assembly_saturation;
    let assembly_cost_per_extent =
        cell.assembly_energy_per_unit * terms.product_total * (body_fraction + boundary_fraction);
    let cost_per_extent =
        energy_deficit_per_extent + pump_cost_per_extent + assembly_cost_per_extent;
    let aligned_available = terms.aligned_available;

    let executed_extent = affordable_extent(
        requested_extent,
        cost_per_extent,
        cell_energy,
        aligned_available,
        terms.throughput,
        terms.energy_fraction,
    );
    if executed_extent <= 1.0e-12 {
        return None;
    }

    let input_magnitude = aligned_available.min(terms.throughput * executed_extent);
    let enval_energy = input_magnitude * terms.energy_fraction;
    let output_magnitude =
        input_magnitude * terms.release_fraction + terms.pump_per_extent * executed_extent;
    let raw_chemical_energy = (terms.reactant_energy - terms.product_energy) * executed_extent;
    let chemical_energy_delta = if raw_chemical_energy >= 0.0 {
        raw_chemical_energy * terms.harvest_fraction
    } else {
        raw_chemical_energy
    };
    let pump_cost = pump_cost_per_extent * executed_extent;
    let assembly_cost = assembly_cost_per_extent * executed_extent;
    let energy_delta = chemical_energy_delta + enval_energy - pump_cost - assembly_cost;
    let secretion_fraction = terms.secretion_fraction;

    Some(FluxExtent {
        requested_extent,
        executed_extent,
        secretion_fraction,
        body_fraction,
        boundary_fraction,
        retained_fraction: (1.0 - secretion_fraction - body_fraction - boundary_fraction).max(0.0),
        assembly_cost,
        energy_delta,
        raw_chemical_energy,
        chemical_energy_delta,
        enval_energy,
        pump_cost,
        enval_input: (terms.polarity * input_magnitude) as f32,
        enval_output: (-terms.polarity * output_magnitude) as f32,
    })
}

#[inline]
fn affordable_extent(
    requested_extent: f64,
    cost_per_extent: f64,
    cell_energy: f64,
    aligned_available: f64,
    enval_throughput: f64,
    enval_energy_fraction: f64,
) -> f64 {
    let supported = |extent: f64| {
        let input_magnitude = aligned_available.min(enval_throughput * extent);
        let enval_energy = input_magnitude * enval_energy_fraction;
        cost_per_extent * extent <= cell_energy + enval_energy
    };
    if cost_per_extent > 0.0 && !supported(requested_extent) {
        let mut extent =
            (cell_energy + enval_energy_fraction * aligned_available) / cost_per_extent;
        let net_cost_per_extent = cost_per_extent - enval_energy_fraction * enval_throughput;
        if net_cost_per_extent > 0.0 {
            extent = extent.min(cell_energy / net_cost_per_extent);
        }
        let mut extent = extent.clamp(0.0, requested_extent);
        while extent > 0.0 && !supported(extent) {
            extent = extent.next_down();
        }
        extent
    } else {
        requested_extent
    }
}

fn enval_polarity(enval_optimum: f32) -> f32 {
    if enval_optimum > 0.0 {
        1.0
    } else if enval_optimum < 0.0 {
        -1.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ENVAL_PUMP_ENERGY_COST, ReactionEnv, affordable_extent, compute_enval_factor, compute_flux,
        compute_saturation,
    };
    use crate::chem::{Element, ElementAmounts};
    use crate::genome::Enzyme;
    use crate::rng::Rng;

    const TEST_OPTIMUM: f32 = 0.25;

    fn tuned(mut enzyme: Enzyme) -> Enzyme {
        enzyme.enval_optimum = TEST_OPTIMUM;
        enzyme
    }

    fn environment(local_enval: f32) -> ReactionEnv {
        ReactionEnv::internal(local_enval, 1.0, crate::config::CellBodyConfig::default())
    }

    #[test]
    fn assembly_routes_saturated_products_and_charges_their_cost() {
        let mut enzyme = tuned(Enzyme::founder_downhill());
        enzyme.body_fraction = 0.2;
        enzyme.boundary_fraction = 0.3;
        enzyme.enval_pump = 0.0;
        let env = environment(0.25);
        let reservoir = ElementAmounts::new([0.0, 0.0, 0.0, 10.0, 0.0, 0.0]);
        for energy in [0.0, 2.0, 20.0] {
            let outcome = compute_flux(&enzyme, reservoir, energy, 0.01, env).unwrap();
            let saturation = energy / (2.0 + energy);
            let extent = f64::from(outcome.executed_extent);
            assert!((outcome.body_products.total() - extent * 0.2 * saturation).abs() < 1.0e-8);
            assert!((outcome.boundary_products.total() - extent * 0.3 * saturation).abs() < 1.0e-8);
            assert!(
                (outcome.assembly_cost
                    - 0.5 * (outcome.body_products.total() + outcome.boundary_products.total()))
                .abs()
                    < 1.0e-8
            );
            assert!(
                (outcome.energy_delta
                    - (outcome.chemical_energy_delta + outcome.enval_energy
                        - outcome.pump_cost
                        - outcome.assembly_cost))
                    .abs()
                    < 1.0e-12
            );
            assert!(
                (outcome.consumed.total()
                    - outcome.retained_products.total()
                    - outcome.secreted_products.total()
                    - outcome.body_products.total()
                    - outcome.boundary_products.total())
                .abs()
                    < 1.0e-8
            );
            if energy == 0.0 {
                assert_eq!(
                    outcome.body_products.total() + outcome.boundary_products.total(),
                    0.0
                );
            }
        }
    }

    #[test]
    fn enzyme_extent_scales_with_area_at_equal_concentration() {
        let enzyme = tuned(Enzyme::founder_downhill());
        let at_area = |area: f32| {
            let mut env = environment(0.25);
            env.cell_area = area;
            let reservoir = ElementAmounts::new([0.0, 0.0, 0.0, 10.0 * area, 0.0, 0.0]);
            compute_flux(&enzyme, reservoir, 20.0, 0.01, env)
                .unwrap()
                .requested_extent
        };
        assert_eq!(at_area(2.0), at_area(1.0) * 2.0);
    }

    #[test]
    fn internal_capacity_is_the_unexposed_share_and_exact_at_zero_exposure() {
        let reservoir = ElementAmounts::new([0.0, 0.0, 0.0, 10.0, 0.0, 0.0]);
        let run = |exposure: f32| {
            let mut enzyme = tuned(Enzyme::founder_downhill());
            enzyme.exposure = exposure;
            let env = ReactionEnv {
                capacity_fraction: 1.0 - f64::from(enzyme.exposure),
                ..environment(0.25)
            };
            compute_flux(&enzyme, reservoir, 20.0, 0.01, env)
        };
        let reference = compute_flux(
            &tuned(Enzyme::founder_downhill()),
            reservoir,
            20.0,
            0.01,
            environment(0.25),
        )
        .unwrap();
        assert_eq!(run(0.0).unwrap(), reference);
        let quarter = run(0.75).unwrap();
        assert!(
            (f64::from(quarter.requested_extent) - 0.25 * f64::from(reference.requested_extent))
                .abs()
                <= 1.0e-6 * f64::from(reference.requested_extent)
        );
        assert!(run(1.0).is_none());
    }

    #[test]
    fn enval_factor_uses_local_average_not_tile_value() {
        let enzyme = tuned(Enzyme::founder_downhill());
        let matched = compute_enval_factor(&enzyme, environment(0.25));
        let mismatched = compute_enval_factor(&enzyme, environment(-0.25));
        assert!((matched - 1.0).abs() <= 1.0e-6);
        assert!(mismatched < matched);
    }

    #[test]
    fn requested_extent_saturates_monotonically_and_is_half_at_the_half_saturation_point() {
        let mut enzyme = tuned(Enzyme::founder_downhill());
        enzyme.enval_pump = 0.0;
        enzyme.half_saturation = 0.2;
        let area = 0.5_f32;
        let dt = 0.01;
        let ceiling =
            f64::from(enzyme.rate) * dt * f64::from(area / crate::spatial::REFERENCE_CELL_AREA);
        let requested_at = |concentration: f32| {
            let mut reservoir = ElementAmounts::ZERO;
            reservoir[Element::D] = concentration * area;
            compute_flux(
                &enzyme,
                reservoir,
                10.0,
                dt,
                ReactionEnv::internal(0.25, area, crate::config::CellBodyConfig::default()),
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
        assert!(compute_flux(&enzyme, reservoir, 1.0, 0.01, environment(0.25),).is_none());
    }

    #[test]
    fn flux_is_deterministic_substrate_limited_and_scalar_conservative() {
        let enzyme = tuned(Enzyme::metabolic(
            ElementAmounts::new([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ElementAmounts::new([0.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
            100.0,
            0.5,
            0.25,
        ));
        let reservoir = ElementAmounts::new([0.5, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let first = compute_flux(&enzyme, reservoir, 10.0, 1.0, environment(0.25)).unwrap();
        let second = compute_flux(&enzyme, reservoir, 10.0, 1.0, environment(0.25)).unwrap();
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
        let enzyme = tuned(Enzyme::founder_downhill());
        let mut reservoir = ElementAmounts::ZERO;
        reservoir[Element::D] = 1.0;
        let outcome = compute_flux(&enzyme, reservoir, 1.0, 0.1, environment(0.0)).unwrap();
        assert!(outcome.raw_chemical_energy > 0.0);
        assert!(outcome.energy_delta > 0.0);
    }

    #[test]
    fn uphill_flux_is_energy_limited_and_never_overdraws_energy() {
        let mut enzyme = tuned(Enzyme::founder_engine_reverse());
        enzyme.rate = 100.0;
        enzyme.enval_throughput = 0.0;
        enzyme.enval_pump = 0.0;
        let reservoir = ElementAmounts::new([0.0, 0.0, 10.0, 0.0, 0.0, 0.0]);
        let outcome = compute_flux(&enzyme, reservoir, 0.1, 1.0, environment(0.0)).unwrap();
        assert!(outcome.executed_extent < outcome.requested_extent);
        assert!(outcome.energy_delta < 0.0);
        assert!(0.1 + outcome.energy_delta >= -1.0e-10);
    }

    fn bisected_extent(
        requested_extent: f64,
        cost_per_extent: f64,
        cell_energy: f64,
        aligned_available: f64,
        enval_throughput: f64,
        enval_energy_fraction: f64,
    ) -> f64 {
        let supported = |extent: f64| {
            let input_magnitude = aligned_available.min(enval_throughput * extent);
            let enval_energy = input_magnitude * enval_energy_fraction;
            cost_per_extent * extent <= cell_energy + enval_energy
        };
        if cost_per_extent > 0.0 && !supported(requested_extent) {
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
        }
    }

    #[test]
    fn closed_form_extent_matches_the_bisection_and_never_overdraws_energy() {
        let mut rng = Rng::from_seed_str("closed-form-extent");
        let body = crate::config::CellBodyConfig::default();
        let mut limited = 0;
        for sample in 0..20_000 {
            let mut enzyme = match sample % 4 {
                0 => Enzyme::founder_engine_forward(),
                1 => Enzyme::founder_engine_reverse(),
                2 => Enzyme::founder_downhill(),
                _ => Enzyme::random(&mut rng),
            };
            enzyme.rate *= rng.range(0.5, 40.0);
            enzyme.enval_throughput = rng.range(0.0, 0.5);
            enzyme.enval_energy_fraction = rng.range(0.0, 1.0);
            enzyme.enval_pump = if rng.chance(0.5) {
                rng.range(0.0, 0.5)
            } else {
                0.0
            };
            enzyme.body_fraction = rng.range(0.0, 0.3);
            enzyme.boundary_fraction = rng.range(0.0, 0.3);
            let energy = if rng.chance(0.2) {
                0.0
            } else {
                10.0_f64.powf(f64::from(rng.range(-6.0, 1.5)))
            };
            let local_enval = rng.range(-2.0, 2.0);
            enzyme.enval_optimum = if sample % 8 == 0 {
                0.0
            } else {
                rng.range(-1.0, 1.0)
            };
            let area = rng.range(0.2, 2.0);
            let mut reservoir = ElementAmounts::ZERO;
            for element in crate::chem::ELEMENT_ORDER {
                reservoir[element] = rng.range(0.0, 3.0) * area;
            }
            let env = ReactionEnv::internal(local_enval, area, body);
            let Some(outcome) = compute_flux(&enzyme, reservoir, energy, 0.01, env) else {
                continue;
            };

            let requested = f64::from(outcome.requested_extent);
            let deficit =
                (enzyme.products.intrinsic_energy() - enzyme.reactants.intrinsic_energy()).max(0.0);
            let density = energy / f64::from(area);
            let saturation = density / (f64::from(body.assembly_half_saturation) + density);
            let assembly = body.assembly_energy_per_unit
                * enzyme.products.total()
                * (f64::from(enzyme.body_fraction) + f64::from(enzyme.boundary_fraction))
                * saturation;
            let polarity = if enzyme.enval_optimum > 0.0 {
                1.0
            } else if enzyme.enval_optimum < 0.0 {
                -1.0
            } else {
                0.0
            };
            let pump_cost = if polarity == 0.0 {
                0.0
            } else {
                ENVAL_PUMP_ENERGY_COST * f64::from(enzyme.enval_pump)
            };
            let cost = deficit + pump_cost + assembly;
            let available = f64::from((polarity * local_enval).max(0.0));
            let arguments = (
                requested,
                cost,
                energy,
                available,
                f64::from(enzyme.enval_throughput),
                f64::from(enzyme.enval_energy_fraction),
            );
            let closed = affordable_extent(
                arguments.0,
                arguments.1,
                arguments.2,
                arguments.3,
                arguments.4,
                arguments.5,
            );
            let bisected = bisected_extent(
                arguments.0,
                arguments.1,
                arguments.2,
                arguments.3,
                arguments.4,
                arguments.5,
            );
            let bisection_resolution = requested * 2.0_f64.powi(-48);
            assert!(
                (closed - bisected).abs() <= 1.0e-9 * bisected + bisection_resolution,
                "sample {sample}: closed {closed} bisected {bisected} requested {requested}"
            );
            assert!(closed <= requested);
            let supported = |extent: f64| {
                cost * extent <= energy + available.min(arguments.4 * extent) * arguments.5
            };
            assert!(supported(closed));
            if closed < requested {
                limited += 1;
                assert!(!supported(closed * (1.0 + 1.0e-12)));
            }
            assert!(
                energy + outcome.energy_delta >= -1.0e-12,
                "sample {sample} overdrew {}",
                energy + outcome.energy_delta
            );
        }
        assert!(limited > 1_000, "only {limited} energy-limited samples");
    }

    #[test]
    fn pumping_costs_one_energy_per_unit_and_is_limited_by_what_the_cell_can_pay() {
        let mut enzyme = tuned(Enzyme::metabolic(
            ElementAmounts::new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            ElementAmounts::new([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            10.0,
            0.0,
            0.0,
        ));
        enzyme.enval_pump = 0.5;
        enzyme.enval_throughput = 0.0;
        enzyme.enval_sigma = 1000.0;
        let reservoir = ElementAmounts::new([0.0, 100.0, 0.0, 0.0, 0.0, 0.0]);
        let rich = compute_flux(&enzyme, reservoir, 100.0, 0.01, environment(0.25)).unwrap();
        let extent = f64::from(rich.executed_extent);
        assert!((rich.pump_cost - 0.5 * extent).abs() <= 1.0e-7 * rich.pump_cost);
        assert!((rich.energy_delta + rich.pump_cost).abs() <= 1.0e-12);
        assert!(
            (f64::from(rich.enval_output.abs()) - 0.5 * extent).abs() <= 1.0e-6,
            "pumped enval must match the paid amount"
        );
        let poor = compute_flux(&enzyme, reservoir, 0.01, 0.01, environment(0.25)).unwrap();
        assert!(poor.executed_extent < poor.requested_extent);
        assert!(poor.pump_cost <= 0.01 + 1.0e-12);
        assert!(0.01 + poor.energy_delta >= -1.0e-12);
        assert!(compute_flux(&enzyme, reservoir, 0.0, 0.01, environment(0.25),).is_none());
    }

    #[test]
    fn pump_and_deficit_can_be_paid_from_this_reactions_enval_harvest() {
        let mut enzyme = tuned(Enzyme::metabolic(
            ElementAmounts::new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            ElementAmounts::new([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            1.0,
            0.0,
            0.0,
        ));
        enzyme.enval_pump = 0.1;
        enzyme.enval_throughput = 1.0;
        enzyme.enval_energy_fraction = 0.5;
        enzyme.enval_release_fraction = 0.0;
        enzyme.enval_sigma = 1000.0;
        let reservoir = ElementAmounts::new([0.0, 100.0, 0.0, 0.0, 0.0, 0.0]);
        let outcome = compute_flux(&enzyme, reservoir, 0.0, 0.01, environment(5.0)).unwrap();
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
    fn zero_optimum_has_no_enval_coupling_and_unchanged_chemistry() {
        let mut coupled = Enzyme::founder_downhill();
        coupled.enval_pump = 0.3;
        coupled.enval_throughput = 0.2;
        assert_eq!(coupled.enval_optimum, 0.0);
        let mut uncoupled = coupled;
        uncoupled.enval_pump = 0.0;
        uncoupled.enval_throughput = 0.0;
        uncoupled.enval_optimum = 1.0e-30;
        let reservoir = ElementAmounts::new([0.0, 0.0, 0.0, 10.0, 0.0, 0.0]);
        for local in [-0.25, 0.0, 0.25] {
            for energy in [0.0, 1.0e-4, 20.0] {
                let env = environment(local);
                let Some(outcome) = compute_flux(&coupled, reservoir, energy, 0.01, env) else {
                    panic!("zero-optimum enzyme must still react at energy {energy}");
                };
                assert_eq!(outcome.enval_input, 0.0);
                assert_eq!(outcome.enval_output, 0.0);
                assert_eq!(outcome.enval_energy, 0.0);
                assert_eq!(outcome.pump_cost, 0.0);
                assert!(outcome.executed_extent > 0.0);
                assert_eq!(
                    Some(outcome),
                    compute_flux(&uncoupled, reservoir, energy, 0.01, env),
                );
            }
        }
    }

    #[test]
    fn negative_zero_optimum_is_exactly_zero() {
        let mut enzyme = Enzyme::founder_downhill();
        enzyme.enval_optimum = -0.0;
        let reservoir = ElementAmounts::new([0.0, 0.0, 0.0, 10.0, 0.0, 0.0]);
        let outcome = compute_flux(&enzyme, reservoir, 20.0, 0.01, environment(-0.25)).unwrap();
        assert_eq!(outcome.enval_input, 0.0);
        assert_eq!(outcome.enval_output, 0.0);
        assert_eq!(outcome.pump_cost, 0.0);
    }

    #[test]
    fn enval_factor_and_polarity_read_the_enzymes_own_optimum() {
        let reservoir = ElementAmounts::new([0.0, 0.0, 0.0, 10.0, 0.0, 0.0]);
        let mut positive = Enzyme::founder_downhill();
        positive.enval_optimum = 0.5;
        positive.enval_sigma = 0.25;
        let mut negative = positive;
        negative.enval_optimum = -0.5;
        let env = environment(0.5);
        assert_eq!(compute_enval_factor(&positive, env), 1.0);
        let expected = (-(1.0_f32 * 1.0) / (2.0 * 0.25 * 0.25)).exp();
        assert!((compute_enval_factor(&negative, env) - expected).abs() <= 1.0e-6);

        let up = compute_flux(&positive, reservoir, 20.0, 0.01, env).unwrap();
        let down = compute_flux(&negative, reservoir, 20.0, 0.01, env).unwrap();
        assert!(up.enval_input > 0.0);
        assert_eq!(down.enval_input, 0.0);
        assert!(up.enval_output < 0.0);
        assert!(down.enval_output > 0.0);

        let mirrored = compute_flux(&negative, reservoir, 20.0, 0.01, environment(-0.5)).unwrap();
        assert_eq!(mirrored.enval_input, -up.enval_input);
        assert_eq!(mirrored.enval_output, -up.enval_output);
    }
}
