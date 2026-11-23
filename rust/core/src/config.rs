use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::chem::{ELEMENT_ORDER, ElementAmounts, ElementAmountsError};

pub const DEFAULT_WIDTH: usize = 320;
pub const DEFAULT_HEIGHT: usize = 240;
pub const DEFAULT_SEED: &str = "42";
pub const DEFAULT_INITIAL_FOUNDER_COUNT: usize = 32;
pub const DEFAULT_DT_SECONDS: f64 = 0.010;
pub const DEFAULT_ENVAL_DIFFUSIVITY_PER_SECOND: f32 = 18.0;
pub const DEFAULT_MEMBRANE_PERMEABILITY: f32 = 4.0;
pub const DEFAULT_CATALYST_UPKEEP_PER_SEC: f64 = 0.01;
pub const DEFAULT_ENVAL_LANDSCAPE_MAGNITUDE: f32 = 1.0;
pub const DEFAULT_ENVAL_LANDSCAPE_SCALE: f32 = 40.0;
pub const DEFAULT_ENVAL_LANDSCAPE_RELAXATION_PER_SECOND: f32 = 0.2;
pub const DEFAULT_ENVAL_RECHARGE_RATE_PER_SECOND: f32 = 0.1;
pub const DEFAULT_SURFACE_CONTACT_GAP: f32 = 0.1;
pub const DEFAULT_CELL_DIFFUSIVITY: f32 = 0.2;

pub const DEFAULT_ELEMENT_FIELD_AMOUNTS: ElementAmounts =
    ElementAmounts::new([1.00, 0.65, 0.50, 0.12, 0.08, 0.05]);
pub const DEFAULT_ELEMENT_FIELD_DIFFUSIVITIES: ElementAmounts =
    ElementAmounts::new([2.8, 1.8, 1.4, 1.2, 3.0, 2.2]);
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
    pub fn validate(&self, dt_seconds: f64) -> Result<(), ConfigError> {
        self.initial_amounts
            .validate_nonnegative("element_fields.initial_amounts")?;
        for element in ELEMENT_ORDER {
            let diffusivity = self.diffusivities[element];
            if !diffusivity_is_valid(diffusivity, dt_seconds) {
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

pub fn diffusivity_is_valid(rate_per_second: f32, dt_seconds: f64) -> bool {
    rate_per_second.is_finite()
        && rate_per_second >= 0.0
        && f64::from(rate_per_second) * dt_seconds <= 1.0
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct EnvalLandscapeConfig {
    pub magnitude: f32,
    pub scale: f32,
    pub relaxation_per_second: f32,
}

impl Default for EnvalLandscapeConfig {
    fn default() -> Self {
        Self {
            magnitude: DEFAULT_ENVAL_LANDSCAPE_MAGNITUDE,
            scale: DEFAULT_ENVAL_LANDSCAPE_SCALE,
            relaxation_per_second: DEFAULT_ENVAL_LANDSCAPE_RELAXATION_PER_SECOND,
        }
    }
}

impl EnvalLandscapeConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if !self.magnitude.is_finite() || self.magnitude <= 0.0 {
            return Err(ConfigError::InvalidPositive(
                "enval_landscape.magnitude",
                f64::from(self.magnitude),
            ));
        }
        if !self.scale.is_finite() || self.scale < 2.0 {
            return Err(ConfigError::InvalidLandscapeScale(self.scale));
        }
        if !self.relaxation_per_second.is_finite() || self.relaxation_per_second < 0.0 {
            return Err(ConfigError::InvalidNonnegative(
                "enval_landscape.relaxation_per_second",
                f64::from(self.relaxation_per_second),
            ));
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

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CellMotionConfig {
    pub diffusivity: f32,
}

impl Default for CellMotionConfig {
    fn default() -> Self {
        Self {
            diffusivity: DEFAULT_CELL_DIFFUSIVITY,
        }
    }
}

impl CellMotionConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if !self.diffusivity.is_finite() || self.diffusivity < 0.0 {
            return Err(ConfigError::InvalidNonnegative(
                "cell_motion.diffusivity",
                f64::from(self.diffusivity),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SurfaceConfig {
    pub contact_gap: f32,
}

impl Default for SurfaceConfig {
    fn default() -> Self {
        Self {
            contact_gap: DEFAULT_SURFACE_CONTACT_GAP,
        }
    }
}

impl SurfaceConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        if !self.contact_gap.is_finite() || self.contact_gap <= 0.0 {
            return Err(ConfigError::InvalidPositive(
                "surface.contact_gap",
                f64::from(self.contact_gap),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CellBodyConfig {
    pub body_density: f32,
    pub boundary_density: f32,
    pub permeability_sigma: f32,
    pub fission_strain: f64,
    pub lysis_strain: f64,
    pub assembly_half_saturation: f32,
    pub assembly_energy_per_unit: f64,
}

impl Default for CellBodyConfig {
    fn default() -> Self {
        Self {
            body_density: 1.0,
            boundary_density: 0.2,
            permeability_sigma: 0.3,
            fission_strain: 1.485,
            lysis_strain: 0.7,
            assembly_half_saturation: 2.0,
            assembly_energy_per_unit: 0.5,
        }
    }
}

impl CellBodyConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        for (name, value) in [
            ("cell_body.body_density", self.body_density),
            ("cell_body.permeability_sigma", self.permeability_sigma),
            ("cell_body.boundary_density", self.boundary_density),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(ConfigError::InvalidPositive(name, f64::from(value)));
            }
        }
        for (name, value) in [
            (
                "cell_body.assembly_half_saturation",
                f64::from(self.assembly_half_saturation),
            ),
            (
                "cell_body.assembly_energy_per_unit",
                self.assembly_energy_per_unit,
            ),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(ConfigError::InvalidPositive(name, value));
            }
        }
        if !self.fission_strain.is_finite() || self.fission_strain <= std::f64::consts::SQRT_2 {
            return Err(ConfigError::InvalidPositive(
                "cell_body.fission_strain > sqrt(2)",
                self.fission_strain,
            ));
        }
        if !self.lysis_strain.is_finite()
            || !(0.0..1.0).contains(&self.lysis_strain)
            || self.lysis_strain == 0.0
        {
            return Err(ConfigError::InvalidPositive(
                "cell_body.lysis_strain in (0, 1)",
                self.lysis_strain,
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Config {
    pub cell_body: CellBodyConfig,
    pub founder_energy: f64,
    pub width: usize,
    pub height: usize,
    pub seed: String,
    pub initial_founder_count: usize,
    pub dt_seconds: f64,
    pub enval_diffusivity_per_second: f32,
    pub element_fields: ElementFieldConfig,
    pub membrane_permeability: f32,
    pub catalyst_upkeep_per_sec: f64,
    pub enval_landscape: EnvalLandscapeConfig,
    pub enval_recharge: EnvalRechargeConfig,
    pub cell_motion: CellMotionConfig,
    pub surface: SurfaceConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            cell_body: CellBodyConfig::default(),
            founder_energy: 2.0,
            width: DEFAULT_WIDTH,
            height: DEFAULT_HEIGHT,
            seed: DEFAULT_SEED.to_owned(),
            initial_founder_count: DEFAULT_INITIAL_FOUNDER_COUNT,
            dt_seconds: DEFAULT_DT_SECONDS,
            enval_diffusivity_per_second: DEFAULT_ENVAL_DIFFUSIVITY_PER_SECOND,
            element_fields: ElementFieldConfig::default(),
            membrane_permeability: DEFAULT_MEMBRANE_PERMEABILITY,
            catalyst_upkeep_per_sec: DEFAULT_CATALYST_UPKEEP_PER_SEC,
            enval_landscape: EnvalLandscapeConfig::default(),
            enval_recharge: EnvalRechargeConfig::default(),
            cell_motion: CellMotionConfig::default(),
            surface: SurfaceConfig::default(),
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
        if !diffusivity_is_valid(self.enval_diffusivity_per_second, self.dt_seconds) {
            return Err(ConfigError::InvalidEnvalDiffusivity(
                self.enval_diffusivity_per_second,
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
        self.cell_body.validate()?;
        if !self.founder_energy.is_finite() || self.founder_energy < 0.0 {
            return Err(ConfigError::InvalidNonnegative(
                "founder_energy",
                self.founder_energy,
            ));
        }
        self.enval_landscape.validate()?;
        self.enval_recharge.validate()?;
        self.cell_motion.validate()?;
        self.surface.validate()?;
        self.element_fields.validate(self.dt_seconds)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ConfigError {
    InvalidDimension(&'static str, usize),
    TileCountOverflow,
    InvalidDtSeconds(f64),
    InvalidPositive(&'static str, f64),
    InvalidNonnegative(&'static str, f64),
    InvalidElementAmount(ElementAmountsError),
    InvalidEnvalDiffusivity(f32),
    InvalidElementDiffusivity { element: &'static str, value: f32 },
    InvalidHeterogeneity(f32),
    InvalidHeterogeneityScale(f32),
    InvalidLandscapeScale(f32),
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
            Self::InvalidPositive(name, value) => write!(
                f,
                "invalid {name}: expected a positive finite value, got {value}"
            ),
            Self::InvalidNonnegative(name, value) => write!(
                f,
                "invalid {name}: expected a nonnegative finite value, got {value}"
            ),
            Self::InvalidElementAmount(error) => error.fmt(f),
            Self::InvalidEnvalDiffusivity(value) => write!(
                f,
                "invalid enval_diffusivity_per_second: expected a finite nonnegative rate with rate * dt_seconds <= 1, got {value}"
            ),
            Self::InvalidElementDiffusivity { element, value } => write!(
                f,
                "invalid element_fields.diffusivities.{element}: expected a finite nonnegative rate per second with rate * dt_seconds <= 1, got {value}"
            ),
            Self::InvalidHeterogeneity(value) => write!(
                f,
                "invalid element_fields.heterogeneity: expected a finite value in [0, 1), got {value}"
            ),
            Self::InvalidHeterogeneityScale(value) => write!(
                f,
                "invalid element_fields.heterogeneity_scale: expected a finite value >= 2, got {value}"
            ),
            Self::InvalidLandscapeScale(value) => write!(
                f,
                "invalid enval_landscape.scale: expected a finite value >= 2, got {value}"
            ),
        }
    }
}

impl Error for ConfigError {}
