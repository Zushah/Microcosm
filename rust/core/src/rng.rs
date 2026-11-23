use serde::{Deserialize, Serialize};

use crate::fingerprint::Fingerprint;

const UINT32_RANGE: f64 = 4_294_967_296.0;
const MAX_F32_BELOW_ONE: f32 = f32::from_bits(0x3f7f_ffff);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Rng {
    a: u32,
    b: u32,
    c: u32,
    d: u32,
    gaussian_spare: Option<f64>,
}

impl Rng {
    pub fn from_seed_str(seed: &str) -> Self {
        let mut seed_factory = Xmur3::new(seed);
        let mut rng = Self {
            a: seed_factory.next_u32(),
            b: seed_factory.next_u32(),
            c: seed_factory.next_u32(),
            d: seed_factory.next_u32(),
            gaussian_spare: None,
        };
        for _ in 0..12 {
            let _ = rng.next_f64();
        }
        rng
    }

    pub fn next_f64(&mut self) -> f64 {
        let t = self.a.wrapping_add(self.b).wrapping_add(self.d);
        self.d = self.d.wrapping_add(1);
        self.a = self.b ^ (self.b >> 9);
        self.b = self.c.wrapping_add(self.c << 3);
        self.c = self.c.rotate_left(21).wrapping_add(t);
        f64::from(t) / UINT32_RANGE
    }

    pub fn next_f32(&mut self) -> f32 {
        let value = self.next_f64() as f32;
        if value >= 1.0 {
            MAX_F32_BELOW_ONE
        } else {
            value
        }
    }

    pub fn chance(&mut self, probability: f32) -> bool {
        if probability.is_nan() || probability <= 0.0 {
            return false;
        }
        if probability >= 1.0 {
            return true;
        }
        self.next_f64() < f64::from(probability)
    }

    pub fn range(&mut self, min: f32, max: f32) -> f32 {
        min + (max - min) * self.next_f32()
    }

    pub fn usize(&mut self, upper: usize) -> usize {
        if upper == 0 {
            0
        } else {
            (self.next_f64() * upper as f64) as usize
        }
    }

    pub fn gaussian(&mut self, mean: f64, stddev: f64) -> f64 {
        let center = if mean.is_finite() { mean } else { 0.0 };
        let sigma = if stddev.is_finite() { stddev } else { 0.0 };
        if sigma <= 0.0 {
            return center;
        }

        if let Some(spare) = self.gaussian_spare.take() {
            return center + spare * sigma;
        }

        let mut u = 0.0;
        let mut v = 0.0;
        while u <= f64::EPSILON {
            u = self.next_f64();
        }
        while v <= f64::EPSILON {
            v = self.next_f64();
        }

        let magnitude = (-2.0 * u.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * v;
        self.gaussian_spare = Some(magnitude * theta.sin());
        center + magnitude * theta.cos() * sigma
    }

    pub(crate) fn hash_state(&self, fingerprint: &mut Fingerprint) {
        let Self {
            a,
            b,
            c,
            d,
            gaussian_spare,
        } = self;
        for word in [a, b, c, d] {
            fingerprint.u32(*word);
        }
        fingerprint.optional_f64(*gaussian_spare);
    }
}

#[derive(Clone, Debug)]
struct Xmur3 {
    h: u32,
}

impl Xmur3 {
    fn new(text: &str) -> Self {
        let units: Vec<u16> = text.encode_utf16().collect();
        let mut h = 1_779_033_703_u32 ^ units.len() as u32;
        for unit in units {
            h = (h ^ u32::from(unit)).wrapping_mul(3_432_918_353);
            h = h.rotate_left(13);
        }
        Self { h }
    }

    fn next_u32(&mut self) -> u32 {
        self.h = (self.h ^ (self.h >> 16)).wrapping_mul(2_246_822_507);
        self.h = (self.h ^ (self.h >> 13)).wrapping_mul(3_266_489_909);
        self.h ^ (self.h >> 16)
    }
}
