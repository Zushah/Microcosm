use crate::chem::{ELEMENT_ORDER, Element, ElementAmounts};
use crate::config::{ElementFieldConfig, EnvalLandscapeConfig};
use crate::rng::Rng;

pub const ENVAL_PER_RECHARGE_ENERGY: f64 = 1.0;

pub fn enval_landscape(
    config: &EnvalLandscapeConfig,
    width: usize,
    height: usize,
    rng: &mut Rng,
) -> Vec<f32> {
    let mut targets = periodic_noise(width, height, config.scale, rng);
    let peak = targets
        .iter()
        .fold(0.0_f32, |peak, value| peak.max(value.abs()));
    if peak > 0.0 {
        for value in &mut targets {
            *value = *value / peak * config.magnitude;
        }
    }
    targets
}

pub fn relax_enval(enval: &mut [f32], targets: &[f32], kappa: f64) -> f64 {
    let mut applied = 0.0;
    for (value, target) in enval.iter_mut().zip(targets) {
        applied += relax_value(value, *target, kappa);
    }
    applied
}

const NOISE_OCTAVES: [(f32, usize); 2] = [(1.0, 1), (0.5, 2)];

pub fn periodic_noise(width: usize, height: usize, scale: f32, rng: &mut Rng) -> Vec<f32> {
    let mut noise = vec![0.0_f32; width * height];
    let base_columns = ((width as f32 / scale).round() as usize).max(1);
    let base_rows = ((height as f32 / scale).round() as usize).max(1);
    let total_amplitude = NOISE_OCTAVES
        .iter()
        .map(|(amplitude, _)| amplitude)
        .sum::<f32>();
    for (amplitude, multiple) in NOISE_OCTAVES {
        let columns = base_columns * multiple;
        let rows = base_rows * multiple;
        let lattice = (0..columns * rows)
            .map(|_| rng.range(-1.0, 1.0))
            .collect::<Vec<_>>();
        let lattice_at = |column: usize, row: usize| lattice[column * rows + row];
        for x in 0..width {
            let (x0, x1, tx) = lattice_coordinate(x, width, columns);
            for y in 0..height {
                let (y0, y1, ty) = lattice_coordinate(y, height, rows);
                let top = lattice_at(x0, y0) + (lattice_at(x1, y0) - lattice_at(x0, y0)) * tx;
                let bottom = lattice_at(x0, y1) + (lattice_at(x1, y1) - lattice_at(x0, y1)) * tx;
                noise[x * height + y] += amplitude * (top + (bottom - top) * ty);
            }
        }
    }
    for value in &mut noise {
        *value = (*value / total_amplitude).clamp(-1.0, 1.0);
    }
    noise
}

fn lattice_coordinate(index: usize, extent: usize, cells: usize) -> (usize, usize, f32) {
    let position = index as f64 * cells as f64 / extent as f64;
    let lower = position.floor();
    let fraction = (position - lower) as f32;
    let lower = lower as usize % cells;
    let smooth = fraction * fraction * (3.0 - 2.0 * fraction);
    (lower, (lower + 1) % cells, smooth)
}

pub fn initial_element_fields(
    config: &ElementFieldConfig,
    width: usize,
    height: usize,
    rng: &mut Rng,
) -> Vec<ElementAmounts> {
    let tile_count = width * height;
    let mut fields = vec![config.initial_amounts; tile_count];
    if config.heterogeneity <= 0.0 {
        return fields;
    }
    for element in ELEMENT_ORDER {
        let noise = periodic_noise(width, height, config.heterogeneity_scale, rng);
        let base = config.initial_amounts[element];
        if base <= 0.0 {
            continue;
        }
        let weights = noise
            .iter()
            .map(|value| f64::from((1.0 + config.heterogeneity * value).max(0.0)))
            .collect::<Vec<_>>();
        let weight_total = weights.iter().sum::<f64>();
        if weight_total <= 0.0 {
            continue;
        }
        let scale = f64::from(base) * tile_count as f64 / weight_total;
        for (amounts, weight) in fields.iter_mut().zip(&weights) {
            amounts[element] = (weight * scale) as f32;
        }
    }
    fields
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RechargeOutcome {
    pub enval_delta: f64,
    pub amount: f64,
    pub energy: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EnvalSweepOutcome {
    pub recharge: RechargeOutcome,
    pub relaxation: f64,
    pub enval_sum: f64,
}

#[derive(Clone, Copy, Debug)]
struct Recharger {
    rate_per_second: f64,
    dt_seconds: f64,
    lift_to_d: f64,
    lift_to_e: f64,
}

impl Recharger {
    fn new(rate_per_second: f64, dt_seconds: f64) -> Self {
        let spent_energy = f64::from(Element::F.properties().energy);
        Self {
            rate_per_second,
            dt_seconds,
            lift_to_d: f64::from(Element::D.properties().energy) - spent_energy,
            lift_to_e: f64::from(Element::E.properties().energy) - spent_energy,
        }
    }

    #[inline]
    fn apply(&self, value: &mut f32, amounts: &mut ElementAmounts, outcome: &mut RechargeOutcome) {
        let field = f64::from(*value);
        let levels = amounts.as_mut_array();
        let spent = f64::from(levels[Element::F.index()]);
        if field == 0.0 || spent <= 0.0 {
            return;
        }
        let positive = field > 0.0;
        let (target, lift) = if positive {
            (Element::D.index(), self.lift_to_d)
        } else {
            (Element::E.index(), self.lift_to_e)
        };
        let magnitude = field.abs();
        let wanted = self.rate_per_second * magnitude * spent * self.dt_seconds;
        let capped = magnitude / (lift * ENVAL_PER_RECHARGE_ENERGY);
        let amount = if wanted < spent { wanted } else { spent };
        let amount = if capped < amount { capped } else { amount } as f32;
        if amount <= 0.0 {
            return;
        }
        let available = levels[Element::F.index()];
        let amount = if available < amount {
            available
        } else {
            amount
        };
        levels[Element::F.index()] -= amount;
        levels[target] += amount;
        let consumed = f64::from(amount) * lift * ENVAL_PER_RECHARGE_ENERGY;
        let remaining = magnitude - consumed;
        let remaining = if remaining > 0.0 { remaining } else { 0.0 };
        let next = if positive { remaining } else { -remaining } as f32;
        outcome.enval_delta += f64::from(next) - field;
        outcome.amount += f64::from(amount);
        outcome.energy += f64::from(amount) * lift;
        *value = next;
    }
}

#[inline]
fn relax_value(value: &mut f32, target: f32, kappa: f64) -> f64 {
    let old = *value;
    let next = (f64::from(old) + kappa * (f64::from(target) - f64::from(old))) as f32;
    *value = next;
    f64::from(next) - f64::from(old)
}

pub fn recharge(
    enval: &mut [f32],
    fields: &mut [ElementAmounts],
    rate_per_second: f64,
    dt_seconds: f64,
) -> RechargeOutcome {
    let mut outcome = RechargeOutcome::default();
    if rate_per_second <= 0.0 {
        return outcome;
    }
    let recharger = Recharger::new(rate_per_second, dt_seconds);
    for (value, amounts) in enval.iter_mut().zip(fields.iter_mut()) {
        recharger.apply(value, amounts, &mut outcome);
    }
    outcome
}

pub fn recharge_and_relax(
    enval: &mut [f32],
    fields: &mut [ElementAmounts],
    targets: &[f32],
    rate_per_second: f64,
    dt_seconds: f64,
    kappa: f64,
) -> EnvalSweepOutcome {
    let mut outcome = EnvalSweepOutcome::default();
    let recharger = Recharger::new(rate_per_second, dt_seconds);
    let recharging = rate_per_second > 0.0;
    let relaxing = kappa > 0.0;
    for ((value, amounts), target) in enval.iter_mut().zip(fields.iter_mut()).zip(targets) {
        if recharging {
            recharger.apply(value, amounts, &mut outcome.recharge);
        }
        if relaxing {
            outcome.relaxation += relax_value(value, *target, kappa);
        }
        outcome.enval_sum += f64::from(*value);
    }
    outcome
}
