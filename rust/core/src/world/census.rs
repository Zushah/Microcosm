use super::World;
use super::fields::sample_smoothed_enval;
use crate::cell::Cell;
use crate::chem::{ELEMENT_COUNT, ELEMENT_ORDER};
use crate::genome::MAX_CELL_ENZYMES;
use crate::render_buffers::{RenderBuffers, RenderVisualState};
use crate::stats::{
    ENZYME_COUNT_HISTOGRAM_LEN, EXPOSED_CELL_THRESHOLD, MIXED_POLARITY_THRESHOLD, WorldStats,
    renewable_coverage,
};

impl World {
    pub fn stats(&self) -> WorldStats {
        self.collect_stats(true)
    }

    pub fn compact_stats(&self) -> WorldStats {
        self.collect_stats(false)
    }

    pub(super) fn collect_stats(&self, include_distribution_stats: bool) -> WorldStats {
        let mut min_enval = f32::INFINITY;
        let mut max_enval = f32::NEG_INFINITY;
        let mut enval_sum = 0.0_f64;
        let mut enval_sum_sq = 0.0_f64;
        let mut positive_enval_tile_count = 0_usize;
        let mut negative_enval_tile_count = 0_usize;
        let mut near_zero_enval_tile_count = 0_usize;
        for value in &self.enval {
            min_enval = min_enval.min(*value);
            max_enval = max_enval.max(*value);
            let as_f64 = f64::from(*value);
            enval_sum += as_f64;
            enval_sum_sq += as_f64 * as_f64;
            if *value > 1.0e-4 {
                positive_enval_tile_count += 1;
            } else if *value < -1.0e-4 {
                negative_enval_tile_count += 1;
            } else {
                near_zero_enval_tile_count += 1;
            }
        }
        let (enval_p05, enval_p50, enval_p95) = if include_distribution_stats {
            let mut sorted_enval = self.enval.clone();
            sorted_enval.sort_by(|a, b| a.total_cmp(b));
            (
                percentile_sorted_f32(&sorted_enval, 0.05),
                percentile_sorted_f32(&sorted_enval, 0.50),
                percentile_sorted_f32(&sorted_enval, 0.95),
            )
        } else {
            (0.0, 0.0, 0.0)
        };
        let enval_average = if self.tile_count > 0 {
            enval_sum / self.tile_count as f64
        } else {
            0.0
        };
        let enval_variance = if self.tile_count > 0 {
            (enval_sum_sq / self.tile_count as f64 - enval_average * enval_average).max(0.0)
        } else {
            0.0
        };
        let enval_std_dev = enval_variance.sqrt() as f32;
        if self.tile_count == 0 {
            min_enval = 0.0;
            max_enval = 0.0;
        }

        let mut extracellular_element_amounts = [0.0_f64; ELEMENT_COUNT];
        for amounts in &self.element_fields {
            for element in ELEMENT_ORDER {
                extracellular_element_amounts[element.index()] += f64::from(amounts[element]);
            }
        }
        let mut intracellular_element_amounts = [0.0_f64; ELEMENT_COUNT];
        for cell in self.cells.iter() {
            for element in ELEMENT_ORDER {
                intracellular_element_amounts[element.index()] +=
                    f64::from(cell.internal_elements[element])
                        + f64::from(cell.body[element])
                        + f64::from(cell.boundary[element]);
            }
        }
        let mut system_element_amounts = [0.0_f64; ELEMENT_COUNT];
        for element in ELEMENT_ORDER {
            system_element_amounts[element.index()] = extracellular_element_amounts
                [element.index()]
                + intracellular_element_amounts[element.index()];
        }
        let total_element_amount = system_element_amounts.iter().sum();

        let occupied_tile_count = self
            .tile_cell_counts
            .iter()
            .filter(|count| **count > 0)
            .count();
        let empty_tile_count = self.tile_count.saturating_sub(occupied_tile_count);
        let occupancy_fraction = if self.tile_count > 0 {
            occupied_tile_count as f64 / self.tile_count as f64
        } else {
            0.0
        };

        let mut live_cell_count = 0_usize;
        let mut energy_sum = 0.0_f64;
        let mut min_cell_energy = f64::INFINITY;
        let mut max_cell_energy = f64::NEG_INFINITY;
        let mut age_sum = 0.0_f64;
        let mut max_cell_age = 0.0_f64;
        let mut enzyme_sum = 0_usize;
        let mut min_enzyme_count = usize::MAX;
        let mut max_enzyme_count = 0_usize;
        let mut enzyme_count_histogram = [0_u64; ENZYME_COUNT_HISTOGRAM_LEN];
        let mut cells_at_enzyme_cap = 0_usize;
        let mut exposure_sum = 0.0_f64;
        let mut max_exposure = 0.0_f32;
        let mut exposed_cell_count = 0_usize;
        let mut mixed_polarity_cells = 0_usize;
        let mut enval_optimum_spread_sum = 0.0_f64;
        let mut niche_mismatch_sum = 0.0_f64;
        let smoothed_enval = if include_distribution_stats {
            self.fresh_smoothed_enval()
        } else {
            Vec::new()
        };

        for cell in self.cells.iter() {
            live_cell_count += 1;
            energy_sum += cell.energy;
            min_cell_energy = min_cell_energy.min(cell.energy);
            max_cell_energy = max_cell_energy.max(cell.energy);
            let age = (self.sim_time_seconds - cell.birth_sim_time).max(0.0);
            age_sum += age;
            max_cell_age = max_cell_age.max(age);

            let enzyme_count = cell.genome.enzymes.len();
            enzyme_sum += enzyme_count;
            min_enzyme_count = min_enzyme_count.min(enzyme_count);
            max_enzyme_count = max_enzyme_count.max(enzyme_count);
            let histogram_index = enzyme_count.min(ENZYME_COUNT_HISTOGRAM_LEN - 1);
            enzyme_count_histogram[histogram_index] =
                enzyme_count_histogram[histogram_index].saturating_add(1);
            if enzyme_count >= MAX_CELL_ENZYMES {
                cells_at_enzyme_cap += 1;
            }
            let mut cell_max_exposure = 0.0_f32;
            let mut cell_min_optimum = f32::INFINITY;
            let mut cell_max_optimum = f32::NEG_INFINITY;
            for enzyme in &cell.genome.enzymes {
                exposure_sum += f64::from(enzyme.exposure);
                cell_max_exposure = cell_max_exposure.max(enzyme.exposure);
                cell_min_optimum = cell_min_optimum.min(enzyme.enval_optimum);
                cell_max_optimum = cell_max_optimum.max(enzyme.enval_optimum);
            }
            max_exposure = max_exposure.max(cell_max_exposure);
            if cell_max_exposure >= EXPOSED_CELL_THRESHOLD {
                exposed_cell_count += 1;
            }
            if cell_max_optimum > MIXED_POLARITY_THRESHOLD
                && cell_min_optimum < -MIXED_POLARITY_THRESHOLD
            {
                mixed_polarity_cells += 1;
            }
            if enzyme_count > 0 {
                enval_optimum_spread_sum += f64::from(cell_max_optimum - cell_min_optimum);
            }
            if include_distribution_stats {
                niche_mismatch_sum += self.cell_niche_mismatch(cell, &smoothed_enval);
            }
        }

        if live_cell_count == 0 {
            min_cell_energy = 0.0;
            max_cell_energy = 0.0;
            min_enzyme_count = 0;
        }

        let total_lineage_records = self.lineage_counters.len();
        let extant_lineage_count = self.extant_lineage_count();
        let extinct_lineage_count = self
            .lineage_counters
            .values()
            .filter(|counters| counters.population == 0)
            .count();
        let mut dominant_lineage_id = 0_u64;
        let mut dominant_lineage_population = 0_u64;
        let mut lineage_entropy = 0.0_f64;
        for (lineage_id, counters) in &self.lineage_counters {
            if counters.population > dominant_lineage_population {
                dominant_lineage_id = lineage_id.raw();
                dominant_lineage_population = counters.population;
            }
            if live_cell_count > 0 && counters.population > 0 {
                let p = counters.population as f64 / live_cell_count as f64;
                lineage_entropy -= p * p.ln();
            }
        }
        let dominant_lineage_share = if live_cell_count > 0 {
            dominant_lineage_population as f64 / live_cell_count as f64
        } else {
            0.0
        };

        let mut radii = self
            .cells
            .iter()
            .map(|cell| cell.radius)
            .collect::<Vec<_>>();
        radii.sort_unstable_by(f32::total_cmp);
        let radius_median = percentile_sorted_f32(&radii, 0.5);
        let radius_p95 = percentile_sorted_f32(&radii, 0.95);
        let radius_max = radii.last().copied().unwrap_or(0.0);
        let mean_strain = self
            .cells
            .iter()
            .map(|cell| cell.strain(self.config.cell_body.boundary_density))
            .sum::<f64>()
            / live_cell_count.max(1) as f64;
        let live_cell_count_f64 = live_cell_count as f64;
        WorldStats {
            tick_count: self.tick_count,
            sim_time_seconds: self.sim_time_seconds,
            width: self.width,
            height: self.height,
            tile_count: self.tile_count,
            occupied_tile_count,
            empty_tile_count,
            occupancy_fraction,
            extracellular_element_amounts,
            intracellular_element_amounts,
            system_element_amounts,
            total_element_amount,
            average_enval: self.avg_enval,
            min_enval,
            max_enval,
            enval_std_dev,
            enval_p05,
            enval_p50,
            enval_p95,
            positive_enval_tile_count,
            negative_enval_tile_count,
            near_zero_enval_tile_count,
            cell_count: live_cell_count,
            live_cell_count,
            births: self.birth_count,
            deaths: self.death_count,
            fissions: self.reaction_counters.divisions,
            energy_deaths: self.energy_death_count,
            lyses: self.lysis_count,
            radius_median,
            radius_p95,
            radius_max,
            mean_strain,
            lineage_count: extant_lineage_count,
            extant_lineage_count,
            total_lineage_records,
            extinct_lineage_count,
            dominant_lineage_id,
            dominant_lineage_population,
            dominant_lineage_share,
            lineage_entropy,
            average_cell_energy: if live_cell_count > 0 {
                energy_sum / live_cell_count_f64
            } else {
                0.0
            },
            min_cell_energy,
            max_cell_energy,
            total_cell_energy: energy_sum,
            average_cell_age: if live_cell_count > 0 {
                age_sum / live_cell_count_f64
            } else {
                0.0
            },
            max_cell_age,
            average_enzyme_count: if live_cell_count > 0 {
                enzyme_sum as f64 / live_cell_count_f64
            } else {
                0.0
            },
            min_enzyme_count,
            max_enzyme_count,
            enzyme_count_histogram,
            cells_at_enzyme_cap,
            fraction_cells_at_enzyme_cap: if live_cell_count > 0 {
                cells_at_enzyme_cap as f64 / live_cell_count_f64
            } else {
                0.0
            },
            total_enzyme_count: enzyme_sum as u64,
            mean_exposure: if enzyme_sum > 0 {
                exposure_sum / enzyme_sum as f64
            } else {
                0.0
            },
            max_exposure,
            exposed_cell_count,
            mixed_polarity_cells,
            mean_enval_optimum_spread: enval_optimum_spread_sum / live_cell_count.max(1) as f64,
            mean_niche_mismatch: niche_mismatch_sum / live_cell_count.max(1) as f64,
            contact_pairs: self.contact_pair_count,
            cross_lineage_contact_pairs: self.cross_lineage_contact_pair_count,
            interaction: self.interaction_counters,
            reaction_counters: self.reaction_counters,
            operation_counters: self.operation_counters,
            energy_ledger: self.energy_ledger,
            enval_ledger: self.enval_ledger,
            energy_ledger_residual: energy_sum - self.energy_ledger.expected_cell_energy(),
            renewable_coverage: renewable_coverage(&self.energy_ledger, &self.enval_ledger),
        }
    }

    pub(super) fn cell_niche_mismatch(&self, cell: &Cell, smoothed_enval: &[f32]) -> f64 {
        let local = self.tile_stencil(cell.position).map_or(0.0, |stencil| {
            f64::from(sample_smoothed_enval(smoothed_enval, &stencil))
        });
        let mut rate_total = 0.0_f64;
        let mut weighted_mismatch = 0.0_f64;
        for enzyme in &cell.genome.enzymes {
            let rate = f64::from(enzyme.rate);
            rate_total += rate;
            weighted_mismatch += rate * (local - f64::from(enzyme.enval_optimum)).abs();
        }
        if rate_total > 0.0 {
            weighted_mismatch / rate_total
        } else {
            0.0
        }
    }

    #[cfg(test)]
    pub(crate) fn build_render_buffers(&self) -> RenderBuffers {
        let mut buffers = RenderBuffers::default();
        self.write_render_buffers(&mut buffers);
        buffers
    }

    pub fn write_render_buffers(&self, buffers: &mut RenderBuffers) {
        self.write_render_buffers_with_visual(buffers, &RenderVisualState::default());
    }

    pub fn write_render_buffers_with_visual(
        &self,
        buffers: &mut RenderBuffers,
        visual: &RenderVisualState,
    ) {
        buffers.clear_for_refresh();
        buffers.width = self.width.min(u32::MAX as usize) as u32;
        buffers.height = self.height.min(u32::MAX as usize) as u32;
        buffers.tick_count = self.tick_count;
        buffers.sim_time_seconds = self.sim_time_seconds;
        buffers.render_epoch = (self.tick_count & u64::from(u32::MAX)) as u32;

        buffers.tile_enval.reserve(self.tile_count);
        buffers.tile_cell_count.reserve(self.tile_count);
        buffers.tile_mass_density.reserve(self.tile_count);
        buffers.tile_total_elements.reserve(self.tile_count);
        buffers
            .tile_element_concentrations
            .reserve(self.tile_count.saturating_mul(ELEMENT_COUNT));
        buffers.cell_id.reserve(self.cells.len());
        buffers
            .cell_point_data
            .reserve(self.cells.len().saturating_mul(4));
        buffers
            .cell_rotation_data
            .reserve(self.cells.len().saturating_mul(4));
        buffers
            .cell_scale_data
            .reserve(self.cells.len().saturating_mul(4));
        buffers
            .cell_rgba
            .reserve(self.cells.len().saturating_mul(4));
        buffers.cell_radius.reserve(self.cells.len());
        buffers.cell_energy.reserve(self.cells.len());
        buffers.cell_lineage.reserve(self.cells.len());
        buffers.cell_flags.reserve(self.cells.len());
        buffers.cell_enzyme_count.reserve(self.cells.len());
        buffers.cell_age_seconds.reserve(self.cells.len());

        buffers.tile_enval.extend_from_slice(&self.enval);

        buffers
            .tile_cell_count
            .extend_from_slice(&self.tile_cell_counts);

        for tile_index in 0..self.tile_count {
            let amounts = self.element_fields[tile_index];
            buffers.tile_mass_density.push(amounts.mass() as f32);
            buffers.tile_total_elements.push(amounts.total() as f32);
            buffers
                .tile_element_concentrations
                .extend_from_slice(amounts.as_array());
        }

        for cell in self.cells.iter() {
            buffers
                .cell_id
                .push(cell.id.index().min(u32::MAX as usize) as u32);
            buffers.cell_point_data.extend_from_slice(&[
                cell.position.x + 0.5 - self.width as f32 * 0.5,
                self.height as f32 * 0.5 - cell.position.y - 0.5,
                0.5,
                0.0,
            ]);
            buffers
                .cell_rotation_data
                .extend_from_slice(&[0.0, 0.0, 0.0, 1.0]);
            buffers.cell_scale_data.extend_from_slice(&[
                cell.radius,
                cell.radius,
                cell.radius,
                0.0,
            ]);
            buffers.cell_radius.push(cell.radius);
            buffers.cell_energy.push(cell.energy as f32);
            buffers
                .cell_lineage
                .push(cell.lineage_id.raw().min(u64::from(u32::MAX)) as u32);
            buffers.cell_flags.push(1);
            buffers
                .cell_enzyme_count
                .push(cell.genome.enzymes.len().min(u32::MAX as usize) as u32);
            buffers
                .cell_age_seconds
                .push((self.sim_time_seconds - cell.birth_sim_time).max(0.0) as f32);
        }
        buffers.refresh_lattice_rgba(visual);
    }
}

fn percentile_sorted_f32(sorted_values: &[f32], fraction: f64) -> f32 {
    if sorted_values.is_empty() {
        return 0.0;
    }
    let max_index = sorted_values.len() - 1;
    let index = ((max_index as f64) * fraction.clamp(0.0, 1.0)).round() as usize;
    sorted_values[index.min(max_index)]
}
