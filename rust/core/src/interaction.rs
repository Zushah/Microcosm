use crate::cell::CellId;

const COINCIDENT_DISTANCE: f64 = 1.0e-6;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Contact {
    pub cell_id: CellId,
    pub slot: usize,
    pub distance: f64,
    pub share: f64,
}

pub fn contact_closeness(radius_a: f64, radius_b: f64, distance: f64, gap: f64) -> f64 {
    if gap.is_nan() || gap <= 0.0 {
        return 0.0;
    }
    ((radius_a + radius_b + gap - distance) / gap).clamp(0.0, 1.0)
}

pub fn perimeter_share(radius_a: f64, radius_b: f64, distance: f64, gap: f64) -> f64 {
    let closeness = contact_closeness(radius_a, radius_b, distance, gap);
    if closeness <= 0.0 {
        return 0.0;
    }
    if distance <= COINCIDENT_DISTANCE {
        return 0.5 * closeness;
    }
    closeness * (radius_b / distance).min(1.0).asin() / std::f64::consts::PI
}

pub fn normalize_shares(contacts: &mut [Contact]) -> f64 {
    let total = contacts.iter().map(|contact| contact.share).sum::<f64>();
    if total > 1.0 {
        for contact in contacts.iter_mut() {
            contact.share /= total;
        }
        return 0.0;
    }
    (1.0 - total).max(0.0)
}
