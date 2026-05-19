use std::error::Error;
use std::fmt;

use serde::{Deserialize, Serialize};

pub const DEFAULT_WIDTH: usize = 320;
pub const DEFAULT_HEIGHT: usize = 240;
pub const DEFAULT_SEED: &str = "demo";
pub const DEFAULT_INITIAL_FOUNDER_COUNT: usize = 32;
pub const DEFAULT_DT_SECONDS: f64 = 0.010;
pub const DEFAULT_MOLECULE_DIFFUSION_WHEEL_SIZE: usize = 4096;
pub const DEFAULT_ENVAL_DIFFUSION_ALPHA: f32 = 0.18;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MoleculeSeedingConfig {
    pub b: f32,
    pub c: f32,
    pub d: f32,
    pub e: f32,
    pub f: f32,
    pub bc: f32,
}

impl Default for MoleculeSeedingConfig {
    fn default() -> Self {
        Self {
            b: 0.60,
            c: 0.45,
            d: 0.12,
            e: 0.08,
            f: 0.05,
            bc: 0.05,
        }
    }
}

impl MoleculeSeedingConfig {
    pub fn validate(&self) -> Result<(), ConfigError> {
        for (name, value) in [
            ("molecule_seeding.b", self.b),
            ("molecule_seeding.c", self.c),
            ("molecule_seeding.d", self.d),
            ("molecule_seeding.e", self.e),
            ("molecule_seeding.f", self.f),
            ("molecule_seeding.bc", self.bc),
        ] {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(ConfigError::InvalidProbability(name, value));
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
    pub molecule_diffusion_wheel_size: usize,
    pub enval_diffusion_alpha: f32,
    pub molecule_seeding: MoleculeSeedingConfig,
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
            molecule_diffusion_wheel_size: DEFAULT_MOLECULE_DIFFUSION_WHEEL_SIZE,
            enval_diffusion_alpha: DEFAULT_ENVAL_DIFFUSION_ALPHA,
            molecule_seeding: MoleculeSeedingConfig::default(),
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
        if self.molecule_diffusion_wheel_size == 0
            || !self.molecule_diffusion_wheel_size.is_power_of_two()
        {
            return Err(ConfigError::InvalidDiffusionWheelSize(
                self.molecule_diffusion_wheel_size,
            ));
        }
        if !self.enval_diffusion_alpha.is_finite()
            || !(0.0..=1.0).contains(&self.enval_diffusion_alpha)
        {
            return Err(ConfigError::InvalidProbability(
                "enval_diffusion_alpha",
                self.enval_diffusion_alpha,
            ));
        }
        self.molecule_seeding.validate()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ConfigError {
    InvalidDimension(&'static str, usize),
    TileCountOverflow,
    InvalidDtSeconds(f64),
    InvalidDiffusionWheelSize(usize),
    InvalidProbability(&'static str, f32),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimension(name, value) => {
                write!(f, "invalid {name}: expected at least 1, got {value}")
            }
            Self::TileCountOverflow => write!(f, "world tile count overflows usize"),
            Self::InvalidDtSeconds(value) => {
                write!(f, "invalid dt_seconds: expected a positive finite value, got {value}")
            }
            Self::InvalidDiffusionWheelSize(value) => write!(
                f,
                "invalid molecule_diffusion_wheel_size: expected a nonzero power of two, got {value}"
            ),
            Self::InvalidProbability(name, value) => write!(
                f,
                "invalid {name}: expected a finite probability in [0, 1], got {value}"
            ),
        }
    }
}

impl Error for ConfigError {}
