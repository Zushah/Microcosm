use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

pub const ELEMENT_COUNT: usize = 6;
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

/// A compact, fixed-size vector of continuous A-F quantities.
///
/// This type deliberately does not clamp values: callers must preserve their
/// accounting invariants, while [`ElementAmounts::validate_nonnegative`] can
/// be used at configuration and runtime boundaries.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[repr(transparent)]
pub struct ElementAmounts([f32; ELEMENT_COUNT]);

impl ElementAmounts {
    pub const ZERO: Self = Self([0.0; ELEMENT_COUNT]);

    pub const fn new(values: [f32; ELEMENT_COUNT]) -> Self {
        Self(values)
    }

    pub const fn as_array(&self) -> &[f32; ELEMENT_COUNT] {
        &self.0
    }

    pub fn as_mut_array(&mut self) -> &mut [f32; ELEMENT_COUNT] {
        &mut self.0
    }

    pub fn get(&self, element: Element) -> f32 {
        self.0[element.index()]
    }

    pub fn set(&mut self, element: Element, value: f32) {
        self.0[element.index()] = value;
    }

    pub fn total(self) -> f64 {
        self.0.iter().map(|value| f64::from(*value)).sum()
    }

    pub fn intrinsic_energy(self) -> f64 {
        ELEMENT_ORDER
            .iter()
            .map(|element| f64::from(self.get(*element)) * f64::from(element.properties().energy))
            .sum()
    }

    pub fn mass(self) -> f64 {
        ELEMENT_ORDER
            .iter()
            .map(|element| f64::from(self.get(*element)) * f64::from(element.properties().mass))
            .sum()
    }

    pub fn validate_nonnegative(self, name: &'static str) -> Result<(), ElementAmountsError> {
        for element in ELEMENT_ORDER {
            let value = self.get(element);
            if !value.is_finite() || value < 0.0 {
                return Err(ElementAmountsError {
                    name,
                    element,
                    value,
                });
            }
        }
        Ok(())
    }
}

impl std::ops::Index<Element> for ElementAmounts {
    type Output = f32;

    fn index(&self, element: Element) -> &Self::Output {
        &self.0[element.index()]
    }
}

impl std::ops::IndexMut<Element> for ElementAmounts {
    fn index_mut(&mut self, element: Element) -> &mut Self::Output {
        &mut self.0[element.index()]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ElementAmountsError {
    pub name: &'static str,
    pub element: Element,
    pub value: f32,
}

impl fmt::Display for ElementAmountsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid {}.{}: expected a finite nonnegative value, got {}",
            self.name, self.element, self.value
        )
    }
}

impl Error for ElementAmountsError {}

#[cfg(test)]
mod tests {
    use super::{Element, ElementAmounts};

    fn approx_eq(a: f32, b: f32) {
        assert!((a - b).abs() <= 1.0e-6, "{a} != {b}");
    }

    #[test]
    fn continuous_element_amounts_are_indexed_and_accumulated_in_element_order() {
        let mut amounts = ElementAmounts::new([1.0, 0.65, 0.5, 0.12, 0.08, 0.05]);
        approx_eq(amounts[Element::C], 0.5);
        amounts[Element::C] = 0.75;
        approx_eq(amounts.get(Element::C), 0.75);
        assert!((amounts.total() - 2.65).abs() <= 1.0e-6);
        assert!((amounts.mass() - 3.16).abs() <= 1.0e-6);
        assert!((amounts.intrinsic_energy() - 2.535).abs() <= 1.0e-6);
        amounts.validate_nonnegative("amounts").unwrap();
    }

    #[test]
    fn continuous_element_amounts_reject_negative_and_nonfinite_values() {
        let negative = ElementAmounts::new([0.0, 0.0, -0.01, 0.0, 0.0, 0.0]);
        let error = negative.validate_nonnegative("amounts").unwrap_err();
        assert_eq!(error.element, Element::C);

        let nonfinite = ElementAmounts::new([0.0, 0.0, 0.0, f32::INFINITY, 0.0, 0.0]);
        let error = nonfinite.validate_nonnegative("amounts").unwrap_err();
        assert_eq!(error.element, Element::D);
    }
}
