use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

pub const ELEMENT_COUNT: usize = 6;
pub const ALL_ELEMENT_MASK: u8 = 0b0011_1111;
pub const ELEMENT_ORDER: [Element; ELEMENT_COUNT] = [
    Element::A,
    Element::B,
    Element::C,
    Element::D,
    Element::E,
    Element::F,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum Element {
    A = 0,
    B = 1,
    C = 2,
    D = 3,
    E = 4,
    F = 5,
}

impl Element {
    pub const fn index(self) -> usize {
        self as usize
    }

    pub const fn mask(self) -> u8 {
        1_u8 << self.index()
    }

    pub const fn symbol(self) -> &'static str {
        match self {
            Element::A => "A",
            Element::B => "B",
            Element::C => "C",
            Element::D => "D",
            Element::E => "E",
            Element::F => "F",
        }
    }

    pub fn properties(self) -> ElementProperties {
        ELEMENT_PROPERTIES[self.index()]
    }
}

impl fmt::Display for Element {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.symbol())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ElementProperties {
    pub mass: f32,
    pub polarity: f32,
    pub energy: f32,
}

pub const ELEMENT_PROPERTIES: [ElementProperties; ELEMENT_COUNT] = [
    ElementProperties {
        mass: 1.0,
        polarity: 0.9,
        energy: 0.5,
    },
    ElementProperties {
        mass: 1.2,
        polarity: 0.4,
        energy: 1.0,
    },
    ElementProperties {
        mass: 1.4,
        polarity: 0.2,
        energy: 0.9,
    },
    ElementProperties {
        mass: 1.8,
        polarity: 0.1,
        energy: 4.0,
    },
    ElementProperties {
        mass: 0.8,
        polarity: 1.0,
        energy: 3.0,
    },
    ElementProperties {
        mass: 1.0,
        polarity: 0.6,
        energy: -0.2,
    },
];

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Composition {
    counts: [u16; ELEMENT_COUNT],
}

impl Composition {
    pub fn try_new(counts: [u16; ELEMENT_COUNT]) -> Result<Self, CompositionError> {
        let composition = Self { counts };
        if composition.is_empty() {
            Err(CompositionError::Empty)
        } else {
            Ok(composition)
        }
    }

    pub const fn from_counts_unchecked(counts: [u16; ELEMENT_COUNT]) -> Self {
        Self { counts }
    }

    pub fn single(element: Element) -> Self {
        let mut counts = [0_u16; ELEMENT_COUNT];
        counts[element.index()] = 1;
        Self { counts }
    }

    pub fn bc_dimer() -> Self {
        let mut counts = [0_u16; ELEMENT_COUNT];
        counts[Element::B.index()] = 1;
        counts[Element::C.index()] = 1;
        Self { counts }
    }

    pub const fn counts(&self) -> &[u16; ELEMENT_COUNT] {
        &self.counts
    }

    pub fn count(&self, element: Element) -> u16 {
        self.counts[element.index()]
    }

    pub fn is_empty(&self) -> bool {
        self.counts.iter().all(|count| *count == 0)
    }

    pub fn size(&self) -> u16 {
        self.counts.iter().copied().sum()
    }

    pub fn atom_count(&self) -> u64 {
        u64::from(self.size())
    }

    pub fn element_mask(&self) -> u8 {
        let mut mask = 0_u8;
        for element in ELEMENT_ORDER {
            if self.count(element) > 0 {
                mask |= element.mask();
            }
        }
        mask
    }

    pub fn elemental_energy_sum(&self) -> f32 {
        let mut energy = 0.0;
        for element in ELEMENT_ORDER {
            energy += element.properties().energy * f32::from(self.count(element));
        }
        energy
    }

    pub fn mean_polarity(&self) -> f32 {
        let size = self.size();
        if size == 0 {
            return 0.0;
        }
        let mut polarity = 0.0;
        for element in ELEMENT_ORDER {
            polarity += element.properties().polarity * f32::from(self.count(element));
        }
        polarity / f32::from(size)
    }

    pub fn density_against(&self, element_counts: &[u32; ELEMENT_COUNT]) -> u64 {
        let mut density = 0_u64;
        for element in ELEMENT_ORDER {
            density += u64::from(self.count(element)) * u64::from(element_counts[element.index()]);
        }
        density
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompositionError {
    Empty,
}

impl fmt::Display for CompositionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => f.write_str("composition must contain at least one atom"),
        }
    }
}

impl Error for CompositionError {}

pub fn normalize_specificity_mask(mask: u8, fallback_mask: u8) -> u8 {
    let normalized = mask & ALL_ELEMENT_MASK;
    let fallback = fallback_mask & ALL_ELEMENT_MASK;
    if normalized != 0 {
        normalized
    } else if fallback != 0 {
        fallback
    } else {
        ALL_ELEMENT_MASK
    }
}

#[cfg(test)]
mod tests {
    use super::{Composition, Element};
    use crate::molecule::Molecule;

    fn approx_eq(a: f32, b: f32) {
        assert!((a - b).abs() <= 1.0e-6, "{a} != {b}");
    }

    #[test]
    fn creates_a_molecule_with_expected_derived_state() {
        let molecule = Molecule::new(Composition::single(Element::A), 1.0).unwrap();
        assert_eq!(molecule.size, 1);
        assert_eq!(molecule.element_mask, Element::A.mask());
        approx_eq(molecule.elemental_energy_sum, 0.5);
        approx_eq(molecule.energy, 0.5);
        approx_eq(molecule.polarity, 0.9);
        approx_eq(molecule.diffusion_rate, 0.028);
        assert_eq!(molecule.diffusion_period, 36);
    }

    #[test]
    fn creates_bc_dimer_with_expected_derived_state() {
        let molecule = Molecule::new(Composition::bc_dimer(), 1.0).unwrap();
        assert_eq!(molecule.size, 2);
        assert_eq!(molecule.element_mask, Element::B.mask() | Element::C.mask());
        approx_eq(molecule.elemental_energy_sum, 1.9);
        approx_eq(molecule.energy, 1.9);
        approx_eq(molecule.polarity, 0.3);
        approx_eq(molecule.diffusion_rate, 0.014);
        assert_eq!(molecule.diffusion_period, 71);
    }

    #[test]
    fn empty_composition_is_rejected() {
        assert_eq!(
            Composition::try_new([0; 6]).unwrap_err(),
            super::CompositionError::Empty
        );
        assert!(Molecule::new(Composition::default(), 1.0).is_err());
    }

    #[test]
    fn composition_size_and_mask_are_deterministic() {
        let composition = Composition::try_new([2, 0, 1, 0, 3, 0]).unwrap();
        assert_eq!(composition.size(), 6);
        assert_eq!(
            composition.element_mask(),
            Element::A.mask() | Element::C.mask() | Element::E.mask()
        );
        approx_eq(composition.elemental_energy_sum(), 10.9);
        approx_eq(
            composition.mean_polarity(),
            (2.0 * 0.9 + 0.2 + 3.0 * 1.0) / 6.0,
        );
    }
}
