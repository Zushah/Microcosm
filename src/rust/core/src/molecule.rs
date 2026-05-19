use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::chem::Composition;

const UINT32_RANGE: f64 = 4_294_967_296.0;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Molecule {
    pub composition: Composition,
    pub bond_multiplier: f32,
    pub size: u16,
    pub element_mask: u8,
    pub elemental_energy_sum: f32,
    pub energy: f32,
    pub polarity: f32,
    pub diffusion_rate: f32,
    pub diffusion_period: u32,
    pub diffusion_threshold: u32,
    pub diffusion_inv_log1m_p: f32,
}

impl Molecule {
    pub fn new(composition: Composition, bond_multiplier: f32) -> Result<Self, MoleculeError> {
        if composition.is_empty() {
            return Err(MoleculeError::EmptyComposition);
        }
        if !bond_multiplier.is_finite() {
            return Err(MoleculeError::NonFiniteBondMultiplier(bond_multiplier));
        }

        let multiplier = if bond_multiplier == 0.0 {
            1.0
        } else {
            bond_multiplier
        };
        let size = composition.size();
        let element_mask = composition.element_mask();
        let elemental_energy_sum = composition.elemental_energy_sum();
        let energy = elemental_energy_sum * multiplier;
        let polarity = composition.mean_polarity();
        let diffusion_rate = (0.01 + (1.0 / (f32::from(size) + 1.0)) * polarity * 0.04).min(0.04);
        let diffusion_period = if diffusion_rate > 0.0 {
            (1.0 / diffusion_rate).round().max(1.0) as u32
        } else {
            0
        };
        let threshold = (f64::from(diffusion_rate) * UINT32_RANGE)
            .floor()
            .clamp(0.0, f64::from(u32::MAX)) as u32;
        let diffusion_inv_log1m_p = if diffusion_rate > 0.0 && diffusion_rate < 1.0 {
            1.0 / (-diffusion_rate).ln_1p()
        } else {
            0.0
        };

        Ok(Self {
            composition,
            bond_multiplier: multiplier,
            size,
            element_mask,
            elemental_energy_sum,
            energy,
            polarity,
            diffusion_rate,
            diffusion_period,
            diffusion_threshold: threshold,
            diffusion_inv_log1m_p,
        })
    }

    pub fn diffusion_wait(self) -> u32 {
        self.diffusion_period
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MoleculeError {
    EmptyComposition,
    NonFiniteBondMultiplier(f32),
}

impl fmt::Display for MoleculeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyComposition => {
                f.write_str("molecule composition must contain at least one atom")
            }
            Self::NonFiniteBondMultiplier(value) => {
                write!(f, "molecule bond multiplier must be finite, got {value}")
            }
        }
    }
}

impl Error for MoleculeError {}

#[cfg(test)]
mod tests {
    use super::Molecule;
    use crate::chem::{Composition, Element};

    #[test]
    fn bond_multiplier_scales_energy() {
        let molecule = Molecule::new(Composition::single(Element::B), 2.5).unwrap();
        assert_eq!(molecule.elemental_energy_sum, 1.0);
        assert_eq!(molecule.energy, 2.5);
    }

    #[test]
    fn zero_bond_multiplier_uses_js_refresh_fallback() {
        let molecule = Molecule::new(Composition::single(Element::B), 0.0).unwrap();
        assert_eq!(molecule.bond_multiplier, 1.0);
        assert_eq!(molecule.energy, 1.0);
    }

    #[test]
    fn nonfinite_bond_multiplier_is_rejected() {
        assert!(Molecule::new(Composition::single(Element::A), f32::NAN).is_err());
    }
}
