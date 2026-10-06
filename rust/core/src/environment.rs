use serde::{Deserialize, Serialize};

use crate::chem::{ELEMENT_ORDER, Element, ElementAmounts};
use crate::config::{ElementFieldConfig, EnvalSourceConfig};
use crate::rng::Rng;
use crate::spatial::{Position, toroidal_distance_squared};

pub const ENVAL_PER_RECHARGE_ENERGY: f64 = 1.0;
const SOURCE_PLACEMENT_ATTEMPTS: usize = 64;
const SOURCE_SEPARATION_RADII: f32 = 4.0;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct EnvalSource {
    pub center: Position,
    pub target: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EnvalSources {
    pub sources: Vec<EnvalSource>,
    tiles: Vec<u32>,
    targets: Vec<f32>,
}

impl EnvalSources {
    pub fn place(config: &EnvalSourceConfig, width: usize, height: usize, rng: &mut Rng) -> Self {
        let mut placed = Self::default();
        if config.pairs == 0 || width == 0 || height == 0 {
            return placed;
        }
        let (world_width, world_height) = (width as f32, height as f32);
        let min_separation_squared = (SOURCE_SEPARATION_RADII * config.radius).powi(2);
        let mut claimed = vec![false; width * height];
        for index in 0..config.pairs.saturating_mul(2) {
            let mut center = Position::default();
            for _ in 0..SOURCE_PLACEMENT_ATTEMPTS {
                center = Position::new(rng.range(0.0, world_width), rng.range(0.0, world_height));
                let separated = placed.sources.iter().all(|source| {
                    toroidal_distance_squared(center, source.center, world_width, world_height)
                        >= min_separation_squared
                });
                if separated {
                    break;
                }
            }
            let target = if index % 2 == 0 {
                config.magnitude
            } else {
                -config.magnitude
            };
            placed.sources.push(EnvalSource { center, target });
            placed.claim_disc(center, config.radius, target, width, height, &mut claimed);
        }
        placed
    }

    fn claim_disc(
        &mut self,
        center: Position,
        radius: f32,
        target: f32,
        width: usize,
        height: usize,
        claimed: &mut [bool],
    ) {
        let (world_width, world_height) = (width as f32, height as f32);
        let reach = radius.ceil() as isize + 1;
        let radius_squared = radius * radius;
        let center_x = center.x.floor() as isize;
        let center_y = center.y.floor() as isize;
        for dx in -reach..=reach {
            for dy in -reach..=reach {
                let x = (center_x + dx).rem_euclid(width as isize) as usize;
                let y = (center_y + dy).rem_euclid(height as isize) as usize;
                let tile = x * height + y;
                if claimed[tile] {
                    continue;
                }
                let tile_center = Position::new(x as f32, y as f32);
                if toroidal_distance_squared(center, tile_center, world_width, world_height)
                    <= radius_squared
                {
                    claimed[tile] = true;
                    self.tiles.push(tile as u32);
                    self.targets.push(target);
                }
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.tiles.is_empty()
    }

    pub fn tile_count(&self) -> usize {
        self.tiles.len()
    }

    pub fn tiles(&self) -> impl Iterator<Item = (usize, f32)> + '_ {
        self.tiles
            .iter()
            .zip(&self.targets)
            .map(|(tile, target)| (*tile as usize, *target))
    }

    pub fn target_for_tile(&self, tile: usize) -> Option<f32> {
        self.tiles()
            .find(|(source_tile, _)| *source_tile == tile)
            .map(|(_, target)| target)
    }

    pub fn write_targets(&self, enval: &mut [f32]) {
        for (tile, target) in self.tiles() {
            enval[tile] = target;
        }
    }

    pub fn relax(&self, enval: &mut [f32], kappa: f64) -> f64 {
        let mut applied = 0.0;
        for (tile, target) in self.tiles() {
            let old = enval[tile];
            let value = (f64::from(old) + kappa * (f64::from(target) - f64::from(old))) as f32;
            enval[tile] = value;
            applied += f64::from(value) - f64::from(old);
        }
        applied
    }
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
    let spent_energy = f64::from(Element::F.properties().energy);
    let lift_to_d = f64::from(Element::D.properties().energy) - spent_energy;
    let lift_to_e = f64::from(Element::E.properties().energy) - spent_energy;
    for (value, amounts) in enval.iter_mut().zip(fields.iter_mut()) {
        let field = f64::from(*value);
        let spent = f64::from(amounts[Element::F]);
        if field == 0.0 || spent <= 0.0 {
            continue;
        }
        let (target, lift) = if field > 0.0 {
            (Element::D, lift_to_d)
        } else {
            (Element::E, lift_to_e)
        };
        let magnitude = field.abs();
        let amount = spent
            .min(rate_per_second * magnitude * spent * dt_seconds)
            .min(magnitude / (lift * ENVAL_PER_RECHARGE_ENERGY)) as f32;
        if amount <= 0.0 {
            continue;
        }
        let amount = amount.min(amounts[Element::F]);
        amounts[Element::F] -= amount;
        amounts[target] += amount;
        let consumed = f64::from(amount) * lift * ENVAL_PER_RECHARGE_ENERGY;
        let remaining = (magnitude - consumed).max(0.0);
        let next = (field.signum() * remaining) as f32;
        let next = if next.signum() != value.signum() {
            0.0
        } else {
            next
        };
        outcome.enval_delta += f64::from(next) - field;
        outcome.amount += f64::from(amount);
        outcome.energy += f64::from(amount) * lift;
        *value = next;
    }
    outcome
}

#[cfg(test)]
mod tests {
    use super::{EnvalSources, initial_element_fields, periodic_noise, recharge};
    use crate::chem::{Element, ElementAmounts};
    use crate::config::ElementFieldConfig;
    use crate::config::EnvalSourceConfig;
    use crate::rng::Rng;
    use crate::spatial::toroidal_distance;

    #[test]
    fn placement_is_seed_deterministic_alternates_sign_and_honours_separation() {
        let config = EnvalSourceConfig::default();
        let mut first_rng = Rng::from_seed_str("source-placement");
        let mut second_rng = Rng::from_seed_str("source-placement");
        let first = EnvalSources::place(&config, 320, 240, &mut first_rng);
        let second = EnvalSources::place(&config, 320, 240, &mut second_rng);
        assert_eq!(first, second);
        assert_eq!(
            first_rng.next_f64().to_bits(),
            second_rng.next_f64().to_bits()
        );
        assert_eq!(first.sources.len(), 6);
        for (index, source) in first.sources.iter().enumerate() {
            let expected = if index % 2 == 0 { 1.0 } else { -1.0 };
            assert_eq!(source.target, expected);
        }
        for (left_index, left) in first.sources.iter().enumerate() {
            for right in &first.sources[left_index + 1..] {
                assert!(
                    toroidal_distance(left.center, right.center, 320.0, 240.0) >= 24.0,
                    "sources too close"
                );
            }
        }
        let disc_tiles = first.tile_count() / first.sources.len();
        assert!(
            (100..=125).contains(&disc_tiles),
            "{disc_tiles} tiles per disc"
        );

        let mut other_rng = Rng::from_seed_str("source-placement-other");
        assert_ne!(
            EnvalSources::place(&config, 320, 240, &mut other_rng).sources,
            first.sources
        );
    }

    #[test]
    fn infeasible_separation_still_places_every_source_without_duplicate_tiles() {
        let config = EnvalSourceConfig {
            pairs: 4,
            ..EnvalSourceConfig::default()
        };
        let mut rng = Rng::from_seed_str("source-crowded");
        let sources = EnvalSources::place(&config, 10, 8, &mut rng);
        assert_eq!(sources.sources.len(), 8);
        let mut tiles = sources.tiles().map(|(tile, _)| tile).collect::<Vec<_>>();
        let count = tiles.len();
        tiles.sort_unstable();
        tiles.dedup();
        assert_eq!(tiles.len(), count);
        assert!(count > 70 && count <= 80);
        assert!(sources.tiles().all(|(_, target)| target.abs() == 1.0));
        assert!(sources.tiles().filter(|(_, target)| *target == 1.0).count() > count / 2);
    }

    #[test]
    fn zero_pairs_place_nothing_and_consume_no_rng() {
        let config = EnvalSourceConfig {
            pairs: 0,
            ..EnvalSourceConfig::default()
        };
        let mut rng = Rng::from_seed_str("source-none");
        let mut untouched = rng.clone();
        let sources = EnvalSources::place(&config, 32, 24, &mut rng);
        assert!(sources.is_empty());
        assert_eq!(rng.next_f64().to_bits(), untouched.next_f64().to_bits());
    }

    #[test]
    fn relaxation_converges_to_target_and_reports_the_applied_change() {
        let config = EnvalSourceConfig {
            pairs: 1,
            radius: 2.0,
            ..EnvalSourceConfig::default()
        };
        let mut rng = Rng::from_seed_str("source-relax");
        let sources = EnvalSources::place(&config, 16, 12, &mut rng);
        let mut enval = vec![0.0_f32; 16 * 12];
        let kappa = 1.0 - (-0.05_f64).exp();
        let mut total = 0.0;
        for _ in 0..600 {
            let before = enval.iter().map(|value| f64::from(*value)).sum::<f64>();
            let applied = sources.relax(&mut enval, kappa);
            let after = enval.iter().map(|value| f64::from(*value)).sum::<f64>();
            assert!((after - before - applied).abs() <= 1.0e-9);
            total += applied;
        }
        for (tile, target) in sources.tiles() {
            assert!((enval[tile] - target).abs() <= 1.0e-6);
            assert_eq!(sources.target_for_tile(tile), Some(target));
        }
        let expected_total = sources
            .tiles()
            .map(|(_, target)| f64::from(target))
            .sum::<f64>();
        assert!((total - expected_total).abs() <= 1.0e-4);
        let off_source = (0..enval.len())
            .find(|tile| sources.target_for_tile(*tile).is_none())
            .unwrap();
        assert_eq!(enval[off_source], 0.0);
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
