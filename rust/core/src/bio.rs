use crate::chem::{ELEMENT_COUNT, ELEMENT_ORDER, ElementAmounts};
use crate::config::CellBodyConfig;
use crate::genome::Enzyme;
use crate::spatial::REFERENCE_CELL_AREA;

pub const ENVAL_PUMP_ENERGY_COST: f64 = 1.0;

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

#[inline]
fn compute_enval_factor_at(enzyme: &Enzyme, local_enval: f32) -> f32 {
    let sigma = enzyme.enval_sigma.max(1.0e-6);
    let distance = local_enval - enzyme.enval_optimum;
    (-(distance * distance) / (2.0 * sigma * sigma)).exp()
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
