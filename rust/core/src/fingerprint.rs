//! A stable 64-bit digest of the simulation state, used to prove that refactors
//! and optimizations leave a trajectory bit-identical. Integers are hashed as
//! little-endian `u64`s and floats by their bit patterns, so the digest does not
//! depend on the platform or on how structs are laid out in memory. Structs are
//! destructured exhaustively so a new field cannot be left out by accident.

use crate::cell::Cell;
use crate::chem::ElementAmounts;
use crate::config::{
    CellBodyConfig, CellMotionConfig, Config, ElementFieldConfig, EnvalLandscapeConfig,
    EnvalRechargeConfig, SurfaceConfig,
};
use crate::genome::{Enzyme, Genome};
use crate::stats::{EnergyLedger, EnvalLedger, InteractionCounters, ReactionCounters};

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

pub(crate) struct Fingerprint(u64);

impl Fingerprint {
    pub(crate) fn new() -> Self {
        Self(FNV_OFFSET_BASIS)
    }

    pub(crate) fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        let mut hash = self.0;
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        self.0 = hash;
    }

    pub(crate) fn u32(&mut self, value: u32) {
        self.write(&value.to_le_bytes());
    }

    pub(crate) fn u64(&mut self, value: u64) {
        self.write(&value.to_le_bytes());
    }

    pub(crate) fn usize(&mut self, value: usize) {
        self.u64(value as u64);
    }

    pub(crate) fn f32(&mut self, value: f32) {
        self.u32(value.to_bits());
    }

    pub(crate) fn f64(&mut self, value: f64) {
        self.u64(value.to_bits());
    }

    pub(crate) fn optional_f64(&mut self, value: Option<f64>) {
        match value {
            Some(value) => {
                self.u32(1);
                self.f64(value);
            }
            None => self.u32(0),
        }
    }

    pub(crate) fn optional_u64(&mut self, value: Option<u64>) {
        match value {
            Some(value) => {
                self.u32(1);
                self.u64(value);
            }
            None => self.u32(0),
        }
    }

    pub(crate) fn text(&mut self, value: &str) {
        self.usize(value.len());
        self.write(value.as_bytes());
    }

    pub(crate) fn amounts(&mut self, value: &ElementAmounts) {
        for amount in value.as_array() {
            self.f32(*amount);
        }
    }

    pub(crate) fn f32_slice(&mut self, values: &[f32]) {
        self.usize(values.len());
        for value in values {
            self.f32(*value);
        }
    }

    pub(crate) fn amounts_slice(&mut self, values: &[ElementAmounts]) {
        self.usize(values.len());
        for value in values {
            self.amounts(value);
        }
    }

    pub(crate) fn config(&mut self, config: &Config) {
        let Config {
            cell_body,
            founder_energy,
            width,
            height,
            seed,
            initial_founder_count,
            dt_seconds,
            enval_diffusivity_per_second,
            element_fields,
            membrane_permeability,
            catalyst_upkeep_per_sec,
            enval_landscape,
            enval_recharge,
            cell_motion,
            surface,
        } = config;
        let CellBodyConfig {
            body_density,
            boundary_density,
            permeability_sigma,
            fission_strain,
            lysis_strain,
            assembly_half_saturation,
            assembly_energy_per_unit,
        } = cell_body;
        self.f32(*body_density);
        self.f32(*boundary_density);
        self.f32(*permeability_sigma);
        self.f64(*fission_strain);
        self.f64(*lysis_strain);
        self.f32(*assembly_half_saturation);
        self.f64(*assembly_energy_per_unit);
        self.f64(*founder_energy);
        self.usize(*width);
        self.usize(*height);
        self.text(seed);
        self.usize(*initial_founder_count);
        self.f64(*dt_seconds);
        self.f32(*enval_diffusivity_per_second);
        let ElementFieldConfig {
            initial_amounts,
            diffusivities,
            heterogeneity,
            heterogeneity_scale,
        } = element_fields;
        self.amounts(initial_amounts);
        self.amounts(diffusivities);
        self.f32(*heterogeneity);
        self.f32(*heterogeneity_scale);
        self.f32(*membrane_permeability);
        self.f64(*catalyst_upkeep_per_sec);
        let EnvalLandscapeConfig {
            magnitude,
            scale,
            relaxation_per_second,
        } = enval_landscape;
        self.f32(*magnitude);
        self.f32(*scale);
        self.f32(*relaxation_per_second);
        let EnvalRechargeConfig { rate_per_second } = enval_recharge;
        self.f32(*rate_per_second);
        let CellMotionConfig { diffusivity } = cell_motion;
        self.f32(*diffusivity);
        let SurfaceConfig { contact_gap } = surface;
        self.f32(*contact_gap);
    }

    pub(crate) fn cell(&mut self, cell: &Cell) {
        let Cell {
            id,
            position,
            radius,
            energy,
            genome,
            lineage_id,
            internal_elements,
            body,
            boundary,
            birth_sim_time,
            maintenance_cost_per_area,
            last_digested_tick,
        } = cell;
        self.usize(id.index());
        self.f32(position.x);
        self.f32(position.y);
        self.f32(*radius);
        self.f64(*energy);
        self.genome(genome);
        self.u64(lineage_id.raw());
        self.amounts(internal_elements);
        self.amounts(body);
        self.amounts(boundary);
        self.f64(*birth_sim_time);
        self.f64(*maintenance_cost_per_area);
        self.optional_u64(*last_digested_tick);
    }

    fn genome(&mut self, genome: &Genome) {
        let Genome {
            enzymes,
            mutation_rate,
            maintenance_cost_per_area,
            lineage_id,
        } = genome;
        self.usize(enzymes.len());
        for enzyme in enzymes {
            self.enzyme(enzyme);
        }
        self.f32(*mutation_rate);
        self.f64(*maintenance_cost_per_area);
        self.u64(lineage_id.raw());
    }

    fn enzyme(&mut self, enzyme: &Enzyme) {
        let Enzyme {
            reactants,
            products,
            rate,
            half_saturation,
            energy_harvest_fraction,
            secretion_fraction,
            body_fraction,
            boundary_fraction,
            enval_optimum,
            enval_sigma,
            enval_throughput,
            enval_energy_fraction,
            enval_release_fraction,
            enval_pump,
            exposure,
        } = enzyme;
        self.amounts(reactants);
        self.amounts(products);
        for value in [
            rate,
            half_saturation,
            energy_harvest_fraction,
            secretion_fraction,
            body_fraction,
            boundary_fraction,
            enval_optimum,
            enval_sigma,
            enval_throughput,
            enval_energy_fraction,
            enval_release_fraction,
            enval_pump,
            exposure,
        ] {
            self.f32(*value);
        }
    }

    pub(crate) fn reaction_counters(&mut self, counters: &ReactionCounters) {
        let ReactionCounters {
            attempts,
            successes,
            no_substrate,
            energy_delta,
            enval_input,
            enval_output,
            executed_metabolic_flux,
            uptake_flux,
            leak_flux,
            secretion_flux,
            divisions,
        } = counters;
        self.u64(*attempts);
        self.u64(*successes);
        self.u64(*no_substrate);
        for value in [
            energy_delta,
            enval_input,
            enval_output,
            executed_metabolic_flux,
            uptake_flux,
            leak_flux,
            secretion_flux,
        ] {
            self.f64(*value);
        }
        self.u64(*divisions);
    }

    pub(crate) fn interaction_counters(&mut self, counters: &InteractionCounters) {
        let InteractionCounters {
            field_surface_extent,
            cell_surface_extent,
            boundary_digested,
            cross_lineage_boundary_digested,
            lyses_after_digestion,
        } = counters;
        for value in [
            field_surface_extent,
            cell_surface_extent,
            boundary_digested,
            cross_lineage_boundary_digested,
        ] {
            self.f64(*value);
        }
        self.u64(*lyses_after_digestion);
    }

    pub(crate) fn energy_ledger(&mut self, ledger: &EnergyLedger) {
        let EnergyLedger {
            founder_energy,
            injected_energy,
            extracted_energy,
            chemical_harvest,
            chemical_cost,
            enval_harvest,
            pump_cost,
            assembly_cost,
            maintenance,
            death_loss,
        } = ledger;
        for value in [
            founder_energy,
            injected_energy,
            extracted_energy,
            chemical_harvest,
            chemical_cost,
            enval_harvest,
            pump_cost,
            assembly_cost,
            maintenance,
            death_loss,
        ] {
            self.f64(*value);
        }
    }

    pub(crate) fn enval_ledger(&mut self, ledger: &EnvalLedger) {
        let EnvalLedger {
            initial_total,
            source_inflow,
            cell_uptake,
            cell_emission,
            recharge,
            edits,
            recharge_amount,
            recharge_energy,
        } = ledger;
        for value in [
            initial_total,
            source_inflow,
            cell_uptake,
            cell_emission,
            recharge,
            edits,
            recharge_amount,
            recharge_energy,
        ] {
            self.f64(*value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Fingerprint;

    #[test]
    fn fnv1a_matches_the_published_test_vectors() {
        let digest = |bytes: &[u8]| {
            let mut fingerprint = Fingerprint::new();
            fingerprint.write(bytes);
            fingerprint.finish()
        };
        assert_eq!(digest(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(digest(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(digest(b"foobar"), 0x8594_4171_f739_67e8);
    }

    #[test]
    fn floats_hash_by_bit_pattern() {
        let digest = |value: f32| {
            let mut fingerprint = Fingerprint::new();
            fingerprint.f32(value);
            fingerprint.finish()
        };
        assert_ne!(digest(0.0), digest(-0.0));
        assert_eq!(digest(1.5), digest(1.5));
    }
}
