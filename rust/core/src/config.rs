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
pub const DEFAULT_MEMBRANE_PERMEABILITY: f32 = 4.0;
pub const DEFAULT_CATALYST_UPKEEP_PER_SEC: f64 = 0.01;
pub const DEFAULT_ENVAL_SOURCE_PAIRS: usize = 3;
pub const DEFAULT_ENVAL_SOURCE_RADIUS: f32 = 6.0;
pub const DEFAULT_ENVAL_SOURCE_MAGNITUDE: f32 = 1.0;
pub const DEFAULT_ENVAL_SOURCE_RELAXATION_PER_SECOND: f32 = 5.0;
pub const DEFAULT_ENVAL_RECHARGE_RATE_PER_SECOND: f32 = 0.1;

pub const DEFAULT_ELEMENT_FIELD_AMOUNTS: ElementAmounts =
    ElementAmounts::new([1.00, 0.65, 0.50, 0.12, 0.08, 0.05]);
pub const DEFAULT_ELEMENT_FIELD_DIFFUSIVITIES: ElementAmounts =
    ElementAmounts::new([0.028, 0.018, 0.014, 0.012, 0.030, 0.022]);
pub const DEFAULT_ELEMENT_FIELD_HETEROGENEITY: f32 = 0.5;
pub const DEFAULT_ELEMENT_FIELD_HETEROGENEITY_SCALE: f32 = 24.0;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ElementFieldConfig {
    pub initial_amounts: ElementAmounts,
    pub diffusivities: ElementAmounts,
    #[serde(default = "default_heterogeneity")]
    pub heterogeneity: f32,
    #[serde(default = "default_heterogeneity_scale")]
    pub heterogeneity_scale: f32,
}

fn default_heterogeneity() -> f32 {
    DEFAULT_ELEMENT_FIELD_HETEROGENEITY
}

fn default_heterogeneity_scale() -> f32 {
    DEFAULT_ELEMENT_FIELD_HETEROGENEITY_SCALE
}

impl Default for ElementFieldConfig {
    fn default() -> Self {
        Self {
            initial_amounts: DEFAULT_ELEMENT_FIELD_AMOUNTS,
            diffusivities: DEFAULT_ELEMENT_FIELD_DIFFUSIVITIES,
            heterogeneity: DEFAULT_ELEMENT_FIELD_HETEROGENEITY,
            heterogeneity_scale: DEFAULT_ELEMENT_FIELD_HETEROGENEITY_SCALE,
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
        if !self.heterogeneity.is_finite() || !(0.0..1.0).contains(&self.heterogeneity) {
            return Err(ConfigError::InvalidHeterogeneity(self.heterogeneity));
        }
        if !self.heterogeneity_scale.is_finite() || self.heterogeneity_scale < 2.0 {
            return Err(ConfigError::InvalidHeterogeneityScale(
                self.heterogeneity_scale,
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct EnvalSourceConfig {
    pub pairs: usize,
    pub radius: f32,
    pub magnitude: f32,
    pub relaxation_per_second: f32,
}

impl Default for EnvalSourceConfig {
    fn default() -> Self {
        Self {
            pairs: DEFAULT_ENVAL_SOURCE_PAIRS,
            radius: DEFAULT_ENVAL_SOURCE_RADIUS,
            magnitude: DEFAULT_ENVAL_SOURCE_MAGNITUDE,
            relaxation_per_second: DEFAULT_ENVAL_SOURCE_RELAXATION_PER_SECOND,
        }
    }
}

impl EnvalSourceConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        for (name, value) in [
            ("enval_sources.radius", self.radius),
            ("enval_sources.magnitude", self.magnitude),
            (
                "enval_sources.relaxation_per_second",
                self.relaxation_per_second,
            ),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(ConfigError::InvalidPositive(name, f64::from(value)));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct EnvalRechargeConfig {
    pub rate_per_second: f32,
}

impl Default for EnvalRechargeConfig {
    fn default() -> Self {
        Self {
            rate_per_second: DEFAULT_ENVAL_RECHARGE_RATE_PER_SECOND,
        }
    }
}

impl EnvalRechargeConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if !self.rate_per_second.is_finite() || self.rate_per_second < 0.0 {
            return Err(ConfigError::InvalidNonnegative(
                "enval_recharge.rate_per_second",
                f64::from(self.rate_per_second),
            ));
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
    pub membrane_permeability: f32,
    pub catalyst_upkeep_per_sec: f64,
    pub enval_sources: EnvalSourceConfig,
    pub enval_recharge: EnvalRechargeConfig,
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
            membrane_permeability: DEFAULT_MEMBRANE_PERMEABILITY,
            catalyst_upkeep_per_sec: DEFAULT_CATALYST_UPKEEP_PER_SEC,
            enval_sources: EnvalSourceConfig::default(),
            enval_recharge: EnvalRechargeConfig::default(),
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
        if !self.membrane_permeability.is_finite() || self.membrane_permeability <= 0.0 {
            return Err(ConfigError::InvalidPositive(
                "membrane_permeability",
                f64::from(self.membrane_permeability),
            ));
        }
        if !self.catalyst_upkeep_per_sec.is_finite() || self.catalyst_upkeep_per_sec < 0.0 {
            return Err(ConfigError::InvalidNonnegative(
                "catalyst_upkeep_per_sec",
                self.catalyst_upkeep_per_sec,
            ));
        }
        self.enval_sources.validate()?;
        self.enval_recharge.validate()?;
        self.element_fields.validate()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ConfigError {
    InvalidDimension(&'static str, usize),
    TileCountOverflow,
    InvalidDtSeconds(f64),
    InvalidProbability(&'static str, f32),
    InvalidPositive(&'static str, f64),
    InvalidNonnegative(&'static str, f64),
    InvalidElementAmount(ElementAmountsError),
    InvalidElementDiffusivity { element: &'static str, value: f32 },
    InvalidHeterogeneity(f32),
    InvalidHeterogeneityScale(f32),
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
            Self::InvalidPositive(name, value) => write!(
                f,
                "invalid {name}: expected a positive finite value, got {value}"
            ),
            Self::InvalidNonnegative(name, value) => write!(
                f,
                "invalid {name}: expected a nonnegative finite value, got {value}"
            ),
            Self::InvalidElementAmount(error) => error.fmt(f),
            Self::InvalidElementDiffusivity { element, value } => write!(
                f,
                "invalid element_fields.diffusivities.{element}: expected a finite mixing strength in [0, 1], got {value}"
            ),
            Self::InvalidHeterogeneity(value) => write!(
                f,
                "invalid element_fields.heterogeneity: expected a finite value in [0, 1), got {value}"
            ),
            Self::InvalidHeterogeneityScale(value) => write!(
                f,
                "invalid element_fields.heterogeneity_scale: expected a finite value >= 2, got {value}"
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
    fn source_recharge_and_upkeep_config_is_validated() {
        let config = Config::default();
        assert_eq!(config.enval_sources.pairs, 3);
        assert_eq!(config.enval_sources.radius, 6.0);
        assert_eq!(config.enval_sources.magnitude, 1.0);
        assert_eq!(config.enval_sources.relaxation_per_second, 5.0);
        assert_eq!(config.enval_recharge.rate_per_second, 0.1);
        assert_eq!(config.catalyst_upkeep_per_sec, 0.01);

        let mut closed = Config::default();
        closed.enval_sources.pairs = 0;
        closed.enval_recharge.rate_per_second = 0.0;
        closed.catalyst_upkeep_per_sec = 0.0;
        closed.validate().unwrap();

        let mut config = Config::default();
        config.enval_sources.radius = 0.0;
        assert!(matches!(
            config.validate(),
            Err(ConfigError::InvalidPositive("enval_sources.radius", _))
        ));
        let mut config = Config::default();
        config.enval_sources.relaxation_per_second = f32::NAN;
        assert!(config.validate().is_err());
        let mut config = Config::default();
        config.enval_recharge.rate_per_second = -0.1;
        assert!(matches!(
            config.validate(),
            Err(ConfigError::InvalidNonnegative(
                "enval_recharge.rate_per_second",
                _
            ))
        ));
        let mut config = Config::default();
        config.catalyst_upkeep_per_sec = -1.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn membrane_permeability_must_be_positive_and_finite() {
        for value in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let config = Config {
                membrane_permeability: value,
                ..Config::default()
            };
            assert!(matches!(
                config.validate(),
                Err(ConfigError::InvalidPositive("membrane_permeability", _))
            ));
        }
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

        for heterogeneity in [-0.1, 1.0, f32::NAN] {
            let config = ElementFieldConfig {
                heterogeneity,
                ..ElementFieldConfig::default()
            };
            assert!(matches!(
                config.validate(),
                Err(ConfigError::InvalidHeterogeneity(_))
            ));
        }
        for heterogeneity_scale in [1.9, f32::INFINITY] {
            let config = ElementFieldConfig {
                heterogeneity_scale,
                ..ElementFieldConfig::default()
            };
            assert!(matches!(
                config.validate(),
                Err(ConfigError::InvalidHeterogeneityScale(_))
            ));
        }
        let uniform = ElementFieldConfig {
            heterogeneity: 0.0,
            ..ElementFieldConfig::default()
        };
        uniform.validate().unwrap();
    }
}
