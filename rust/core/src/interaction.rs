use crate::cell::CellId;

const COINCIDENT_DISTANCE: f64 = 1.0e-6;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Contact {
    pub cell_id: CellId,
    pub distance: f64,
    pub share: f64,
}

pub fn contact_closeness(radius_a: f64, radius_b: f64, distance: f64, gap: f64) -> f64 {
    if !(gap > 0.0) {
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

#[cfg(test)]
mod tests {
    use super::{Contact, contact_closeness, normalize_shares, perimeter_share};
    use crate::cell::CellId;

    #[test]
    fn shares_follow_closeness_times_subtended_angle() {
        let (ra, rb, gap) = (0.45, 0.3, 0.1);
        let touching = ra + rb;
        assert!((contact_closeness(ra, rb, touching, gap) - 1.0).abs() < 1.0e-12);
        assert_eq!(contact_closeness(ra, rb, touching - 0.05, gap), 1.0);
        assert_eq!(contact_closeness(ra, rb, touching + gap, gap), 0.0);
        assert!((contact_closeness(ra, rb, touching + 0.25 * gap, gap) - 0.75).abs() < 1.0e-12);
        let expected = (rb / touching).asin() / std::f64::consts::PI;
        assert!((perimeter_share(ra, rb, touching, gap) - expected).abs() < 1.0e-12);
        let halfway = touching + 0.5 * gap;
        let expected = 0.5 * (rb / halfway).asin() / std::f64::consts::PI;
        assert!((perimeter_share(ra, rb, halfway, gap) - expected).abs() < 1.0e-12);
        assert_eq!(perimeter_share(ra, rb, touching + gap + 1.0e-9, gap), 0.0);
        assert_eq!(perimeter_share(ra, rb, 0.2, gap), 0.5);
        assert_eq!(perimeter_share(ra, rb, 0.0, gap), 0.5);
        assert_eq!(perimeter_share(ra, rb, touching, 0.0), 0.0);
    }

    #[test]
    fn crowded_shares_are_scaled_to_one_and_leave_no_field() {
        let contact = |index: usize, share: f64| Contact {
            cell_id: CellId(index),
            distance: 0.9,
            share,
        };
        let mut sparse = vec![contact(1, 0.2), contact(2, 0.3)];
        assert!((normalize_shares(&mut sparse) - 0.5).abs() < 1.0e-15);
        assert_eq!(sparse[0].share, 0.2);
        let mut crowded = vec![contact(1, 0.5), contact(2, 0.5), contact(3, 1.0)];
        assert_eq!(normalize_shares(&mut crowded), 0.0);
        let total = crowded.iter().map(|contact| contact.share).sum::<f64>();
        assert!(total <= 1.0 + 1.0e-15 && total >= 1.0 - 1.0e-15);
        assert_eq!(crowded[2].share, 0.5);
        assert_eq!(normalize_shares(&mut []), 1.0);
    }
}
