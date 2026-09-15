use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::chem::{ELEMENT_ORDER, ElementAmounts, ElementAmountsError};

pub const DEFAULT_WIDTH: usize = 320;
pub const DEFAULT_HEIGHT: usize = 240;
pub const DEFAULT_SEED: &str = "42";
pub const DEFAULT_INITIAL_FOUNDER_COUNT: usize = 32;
pub const DEFAULT_DT_SECONDS: f64 = 0.010;
pub const DEFAULT_ENVAL_DIFFUSION_ALPHA: f32 = 0.18;

pub const DEFAULT_ELEMENT_FIELD_AMOUNTS: ElementAmounts =
    ElementAmounts::new([1.00, 0.65, 0.50, 0.12, 0.08, 0.05]);
pub const DEFAULT_ELEMENT_FIELD_DIFFUSIVITIES: ElementAmounts =
    ElementAmounts::new([0.028, 0.018, 0.014, 0.012, 0.030, 0.022]);

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ElementFieldConfig {
    pub initial_amounts: ElementAmounts,
    pub diffusivities: ElementAmounts,
}

impl Default for ElementFieldConfig {
    fn default() -> Self {
        Self {
            initial_amounts: DEFAULT_ELEMENT_FIELD_AMOUNTS,
            diffusivities: DEFAULT_ELEMENT_FIELD_DIFFUSIVITIES,
        }
    }
}

impl ElementFieldConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        self.initial_amounts
            .validate_nonnegative("element_fields.initial_amounts")?;
        self.diffusivities
            .validate_nonnegative("element_fields.diffusivities")?;
        for element in ELEMENT_ORDER {
            let diffusivity = self.diffusivities[element];
            if diffusivity > 1.0 {
                return Err(ConfigError::InvalidElementDiffusivity {
                    element: element.symbol(),
                    value: diffusivity,
                });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Config {
    pub width: usize,
    pub height: usize,
    pub seed: String,
    pub initial_founder_count: usize,
    pub dt_seconds: f64,
    pub enval_diffusion_alpha: f32,
    pub element_fields: ElementFieldConfig,
    pub predation_enabled: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            width: DEFAULT_WIDTH,
            height: DEFAULT_HEIGHT,
            seed: DEFAULT_SEED.to_owned(),
            initial_founder_count: DEFAULT_INITIAL_FOUNDER_COUNT,
            dt_seconds: DEFAULT_DT_SECONDS,
            enval_diffusion_alpha: DEFAULT_ENVAL_DIFFUSION_ALPHA,
            element_fields: ElementFieldConfig::default(),
            predation_enabled: true,
        }
    }
}

impl Config {
    pub fn tile_count(&self) -> Option<usize> {
        self.width.checked_mul(self.height)
    }

    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.width == 0 {
            return Err(ConfigError::InvalidDimension("width", self.width));
        }
        if self.height == 0 {
            return Err(ConfigError::InvalidDimension("height", self.height));
        }
        if self.tile_count().is_none() {
            return Err(ConfigError::TileCountOverflow);
        }
        if !self.dt_seconds.is_finite() || self.dt_seconds <= 0.0 {
            return Err(ConfigError::InvalidDtSeconds(self.dt_seconds));
        }
        if !self.enval_diffusion_alpha.is_finite()
            || !(0.0..=1.0).contains(&self.enval_diffusion_alpha)
        {
            return Err(ConfigError::InvalidProbability(
                "enval_diffusion_alpha",
                self.enval_diffusion_alpha,
            ));
        }
        self.element_fields.validate()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ConfigError {
    InvalidDimension(&'static str, usize),
    TileCountOverflow,
    InvalidDtSeconds(f64),
    InvalidProbability(&'static str, f32),
    InvalidElementAmount(ElementAmountsError),
    InvalidElementDiffusivity { element: &'static str, value: f32 },
}

impl From<ElementAmountsError> for ConfigError {
    fn from(value: ElementAmountsError) -> Self {
        Self::InvalidElementAmount(value)
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimension(name, value) => {
                write!(f, "invalid {name}: expected at least 1, got {value}")
            }
            Self::TileCountOverflow => write!(f, "world tile count overflows usize"),
            Self::InvalidDtSeconds(value) => {
                write!(
                    f,
                    "invalid dt_seconds: expected a positive finite value, got {value}"
                )
            }
            Self::InvalidProbability(name, value) => write!(
                f,
                "invalid {name}: expected a finite probability in [0, 1], got {value}"
            ),
            Self::InvalidElementAmount(error) => error.fmt(f),
            Self::InvalidElementDiffusivity { element, value } => write!(
                f,
                "invalid element_fields.diffusivities.{element}: expected a finite mixing strength in [0, 1], got {value}"
            ),
        }
    }
}

impl Error for ConfigError {}

#[cfg(test)]
mod tests {
    use super::{
        Config, ConfigError, DEFAULT_ELEMENT_FIELD_AMOUNTS, DEFAULT_ELEMENT_FIELD_DIFFUSIVITIES,
        ElementFieldConfig,
    };
    use crate::chem::Element;

    fn approx_eq(a: f32, b: f32) {
        assert!((a - b).abs() <= 1.0e-6, "{a} != {b}");
    }

    #[test]
    fn continuous_field_defaults_preserve_expected_elemental_abundance() {
        assert_eq!(
            *DEFAULT_ELEMENT_FIELD_AMOUNTS.as_array(),
            [1.00, 0.65, 0.50, 0.12, 0.08, 0.05]
        );
        assert_eq!(
            *DEFAULT_ELEMENT_FIELD_DIFFUSIVITIES.as_array(),
            [0.028, 0.018, 0.014, 0.012, 0.030, 0.022]
        );

        for element in crate::chem::ELEMENT_ORDER {
            approx_eq(
                DEFAULT_ELEMENT_FIELD_DIFFUSIVITIES[element],
                0.01 + 0.02 * element.properties().polarity,
            );
        }
        Config::default().validate().unwrap();
    }

    #[test]
    fn continuous_field_config_rejects_invalid_amounts_and_diffusivities() {
        let mut config = ElementFieldConfig::default();
        config.initial_amounts[Element::A] = -0.01;
        assert!(matches!(
            config.validate(),
            Err(ConfigError::InvalidElementAmount(_))
        ));

        let mut config = ElementFieldConfig::default();
        config.diffusivities[Element::F] = 1.01;
        assert!(matches!(
            config.validate(),
            Err(ConfigError::InvalidElementDiffusivity { element: "F", .. })
        ));
    }
}
