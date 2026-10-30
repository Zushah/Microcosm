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

#[cfg(test)]
mod tests {
    use super::{enval_landscape, initial_element_fields, periodic_noise, recharge, relax_enval};
    use crate::chem::{Element, ElementAmounts};
    use crate::config::{ElementFieldConfig, EnvalLandscapeConfig};
    use crate::rng::Rng;

    #[test]
    fn landscape_is_seeded_bounded_peaks_at_the_magnitude_and_is_periodic() {
        let (width, height) = (160, 120);
        let config = EnvalLandscapeConfig {
            magnitude: 1.3,
            scale: 40.0,
            relaxation_per_second: 0.2,
        };
        let mut rng = Rng::from_seed_str("landscape");
        let targets = enval_landscape(&config, width, height, &mut rng);
        assert_eq!(targets.len(), width * height);
        assert!(targets.iter().all(|value| value.abs() <= 1.3));
        let peak = targets
            .iter()
            .fold(0.0_f32, |peak, value| peak.max(value.abs()));
        assert_eq!(peak, 1.3);
        assert!(targets.iter().any(|value| *value > 0.3));
        assert!(targets.iter().any(|value| *value < -0.3));
        let (interior, seam) = max_neighbour_step(&targets, width, height);
        assert!(interior < 0.25, "landscape is not smooth: {interior}");
        assert!(
            seam <= interior,
            "seam step {seam} exceeds interior step {interior}"
        );

        let mut same_rng = Rng::from_seed_str("landscape");
        assert_eq!(
            targets,
            enval_landscape(&config, width, height, &mut same_rng)
        );
        assert_eq!(rng.next_f64().to_bits(), same_rng.next_f64().to_bits());
        let mut other_rng = Rng::from_seed_str("landscape-other");
        assert_ne!(
            targets,
            enval_landscape(&config, width, height, &mut other_rng)
        );

        let mut rng = Rng::from_seed_str("landscape");
        let mut unit = config;
        unit.magnitude = 1.0;
        let unit_targets = enval_landscape(&unit, width, height, &mut rng);
        for (scaled, unit) in targets.iter().zip(&unit_targets) {
            assert!((scaled - 1.3 * unit).abs() <= 1.0e-6);
        }
    }

    #[test]
    fn relaxation_moves_every_tile_by_kappa_and_reports_the_applied_change() {
        let targets = vec![1.0_f32, -0.5, 0.25, 0.0];
        let mut enval = vec![0.0_f32, 0.5, 0.25, -2.0];
        let kappa = 1.0 - (-0.2_f64 * 0.01).exp();
        let mut total = 0.0;
        for _ in 0..10_000 {
            let before = enval.clone();
            let applied = relax_enval(&mut enval, &targets, kappa);
            let mut expected_applied = 0.0;
            for tile in 0..enval.len() {
                let expected = f64::from(before[tile])
                    + kappa * (f64::from(targets[tile]) - f64::from(before[tile]));
                assert!((f64::from(enval[tile]) - expected).abs() <= 1.0e-7);
                expected_applied += f64::from(enval[tile]) - f64::from(before[tile]);
            }
            assert!((applied - expected_applied).abs() <= 1.0e-12);
            total += applied;
        }
        assert_eq!(enval[2], 0.25);
        let decay = (1.0 - kappa).powi(10_000);
        assert!((f64::from(enval[0]) - (1.0 - decay)).abs() <= 1.0e-4);
        let final_sum = enval.iter().map(|value| f64::from(*value)).sum::<f64>();
        assert!((total - (final_sum + 1.25)).abs() <= 1.0e-5);
        let mut frozen = enval.clone();
        assert_eq!(relax_enval(&mut frozen, &targets, 0.0), 0.0);
        assert_eq!(frozen, enval);
    }

    #[test]
    fn recharge_lifts_f_by_field_sign_conserves_matter_and_never_flips_enval() {
        let mut enval = vec![0.8, -0.6, 0.0, 1.0e-4, -2.0, 0.3];
        let base = ElementAmounts::new([0.1, 0.1, 0.1, 0.1, 0.1, 0.5]);
        let mut fields = vec![base; enval.len()];
        fields[5][Element::F] = 0.0;
        let total_before = fields.iter().map(|amounts| amounts.total()).sum::<f64>();
        let enval_before = enval.clone();

        let outcome = recharge(&mut enval, &mut fields, 0.5, 0.01);

        let total_after = fields.iter().map(|amounts| amounts.total()).sum::<f64>();
        assert!((total_after - total_before).abs() <= 1.0e-6);
        for tile in 0..enval.len() {
            let before = enval_before[tile];
            assert!(fields[tile][Element::F] >= 0.0);
            assert!(before.signum() == enval[tile].signum() || enval[tile] == 0.0);
            assert!(enval[tile].abs() <= before.abs());
            let d_gain = fields[tile][Element::D] - base[Element::D];
            let e_gain = fields[tile][Element::E] - base[Element::E];
            if before > 0.0 && tile != 5 {
                assert!(d_gain > 0.0);
                assert_eq!(e_gain, 0.0);
            } else if before < 0.0 {
                assert!(e_gain > 0.0);
                assert_eq!(d_gain, 0.0);
            } else {
                assert_eq!(d_gain, 0.0);
                assert_eq!(e_gain, 0.0);
            }
        }
        assert_eq!(enval[2], 0.0);
        assert_eq!(enval[5], 0.3);
        let consumed = enval_before
            .iter()
            .zip(&enval)
            .map(|(before, after)| f64::from(before.abs()) - f64::from(after.abs()))
            .sum::<f64>();
        let signed_change = enval_before
            .iter()
            .zip(&enval)
            .map(|(before, after)| f64::from(*after) - f64::from(*before))
            .sum::<f64>();
        assert!((outcome.energy - consumed).abs() <= 1.0e-6);
        assert!((outcome.enval_delta - signed_change).abs() <= 1.0e-9);
        assert!(outcome.amount > 0.0);
    }

    #[test]
    fn recharge_is_capped_by_available_enval_so_the_field_reaches_zero_exactly() {
        let mut enval = vec![1.0e-3_f32];
        let mut fields = vec![ElementAmounts::new([0.0, 0.0, 0.0, 0.0, 0.0, 10.0])];
        let outcome = recharge(&mut enval, &mut fields, 50.0, 1.0);
        assert!(enval[0].abs() <= 1.0e-9);
        assert!(enval[0] >= 0.0);
        assert!((outcome.energy - 1.0e-3).abs() <= 1.0e-8);
        assert!((fields[0][Element::D] - 1.0e-3 / 4.2).abs() <= 1.0e-8);
    }

    fn max_neighbour_step(noise: &[f32], width: usize, height: usize) -> (f32, f32) {
        let at = |x: usize, y: usize| noise[x * height + y];
        let mut interior = 0.0_f32;
        for x in 0..width - 1 {
            for y in 0..height - 1 {
                interior = interior
                    .max((at(x + 1, y) - at(x, y)).abs())
                    .max((at(x, y + 1) - at(x, y)).abs());
            }
        }
        let mut seam = 0.0_f32;
        for y in 0..height {
            seam = seam.max((at(0, y) - at(width - 1, y)).abs());
        }
        for x in 0..width {
            seam = seam.max((at(x, 0) - at(x, height - 1)).abs());
        }
        (interior, seam)
    }

    #[test]
    fn periodic_noise_is_bounded_smooth_seamless_and_seeded() {
        let (width, height) = (96, 72);
        let mut rng = Rng::from_seed_str("periodic-noise");
        let noise = periodic_noise(width, height, 24.0, &mut rng);
        assert!(noise.iter().all(|value| (-1.0..=1.0).contains(value)));
        let spread = noise.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            - noise.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(spread > 0.5, "noise is nearly flat: {spread}");
        let (interior, seam) = max_neighbour_step(&noise, width, height);
        assert!(interior < 0.25, "noise is not smooth: {interior}");
        assert!(
            seam <= interior,
            "seam step {seam} exceeds interior step {interior}"
        );

        let mut same_rng = Rng::from_seed_str("periodic-noise");
        assert_eq!(noise, periodic_noise(width, height, 24.0, &mut same_rng));
        let mut other_rng = Rng::from_seed_str("periodic-noise-other");
        assert_ne!(noise, periodic_noise(width, height, 24.0, &mut other_rng));
    }

    #[test]
    fn heterogeneous_initial_fields_keep_exact_totals_and_zero_heterogeneity_is_uniform() {
        let (width, height) = (80, 60);
        let config = ElementFieldConfig::default();
        let mut rng = Rng::from_seed_str("initial-fields");
        let fields = initial_element_fields(&config, width, height, &mut rng);
        for element in crate::chem::ELEMENT_ORDER {
            let total = fields
                .iter()
                .map(|amounts| f64::from(amounts[element]))
                .sum::<f64>();
            let expected = f64::from(config.initial_amounts[element]) * (width * height) as f64;
            assert!((total - expected).abs() <= 1.0e-5 * expected);
            assert!(fields.iter().all(|amounts| amounts[element] >= 0.0));
            let values = fields
                .iter()
                .map(|amounts| amounts[element])
                .collect::<Vec<_>>();
            let (interior, seam) = max_neighbour_step(&values, width, height);
            assert!(seam <= interior);
        }
        assert!(
            fields
                .iter()
                .any(|amounts| amounts != &config.initial_amounts)
        );

        let uniform = ElementFieldConfig {
            heterogeneity: 0.0,
            ..config
        };
        let mut rng = Rng::from_seed_str("initial-fields");
        let mut untouched = rng.clone();
        let fields = initial_element_fields(&uniform, width, height, &mut rng);
        assert!(
            fields
                .iter()
                .all(|amounts| *amounts == config.initial_amounts)
        );
        assert_eq!(rng.next_f64().to_bits(), untouched.next_f64().to_bits());
    }

    #[test]
    fn zero_recharge_rate_is_a_no_op() {
        let mut enval = vec![0.5_f32, -0.5];
        let mut fields = vec![ElementAmounts::new([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]); 2];
        let enval_before = enval.clone();
        let fields_before = fields.clone();
        let outcome = recharge(&mut enval, &mut fields, 0.0, 0.01);
        assert_eq!(enval, enval_before);
        assert_eq!(fields, fields_before);
        assert_eq!(outcome.amount, 0.0);
    }
}
