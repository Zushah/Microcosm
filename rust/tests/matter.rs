mod common;

use common::*;
use microcosmcore::{
    ELEMENT_COUNT, ELEMENT_ORDER, Element, InvariantError, MATTER_CONSERVATION_TOLERANCE, Position,
    World, check_matter_conservation,
};

/// Founders take their material from the field in f32 arithmetic, so total matter moves by rounding only.
const FOUNDER_MATTER_REL_TOL: f64 = 1.0e-9;

#[test]
fn matter_is_conserved_in_a_living_world() {
    for seed in ["matter-4", "matter-27"] {
        let mut world = founded_world(seed, 64, 48, 8);
        assert_eq!(world.cell_count(), 8);
        let initial = total_matter(&world);
        for tick in 1..=2_500 {
            world.step();
            if tick % 100 == 0 {
                let drift = relative_change(initial, total_matter(&world));
                assert!(
                    drift <= MATTER_REL_TOL,
                    "seed {seed} tick {tick}: matter drifted by {drift:e}"
                );
            }
            if tick % 500 == 0 {
                world.check_invariants().unwrap();
            }
        }
        let stats = world.compact_stats();
        assert!(stats.births > 8, "seed {seed}: no births");
        assert!(stats.deaths > 0, "seed {seed}: no deaths");
        assert!(stats.fissions > 0, "seed {seed}: no divisions");
        assert!(world.reaction_counters().successes > 0);
    }
}

#[test]
fn cell_free_diffusion_conserves_each_element_and_never_goes_negative() {
    let mut config = small_config("cell-free-diffusion", 64, 48);
    config.enval_recharge.rate_per_second = 0.0;
    let mut world = World::new(config).unwrap();
    let initial_field = element_field(&world);
    let initial = world.compact_stats().extracellular_element_amounts;
    world.step_many(500);
    let after = world.compact_stats().extracellular_element_amounts;
    for element in ELEMENT_ORDER {
        let index = element.index();
        let drift = relative_change(initial[index], after[index]);
        assert!(drift <= MATTER_REL_TOL, "{element} drifted by {drift:e}");
    }
    world.check_invariants().unwrap();
    assert_ne!(element_field(&world), initial_field);
}

#[test]
fn founders_take_their_material_from_the_field() {
    let mut world = World::new(small_config("founder-material", 32, 24)).unwrap();
    let before = world.compact_stats();
    assert_eq!(world.spawn_founder_cells(12).unwrap(), 12);
    let after = world.compact_stats();
    assert!(
        relative_change(before.total_element_amount, after.total_element_amount) <= MATTER_REL_TOL
    );
    let extracellular =
        |stats: &microcosmcore::WorldStats| stats.extracellular_element_amounts.iter().sum::<f64>();
    assert!(extracellular(&after) < extracellular(&before));
    assert!(after.intracellular_element_amounts.iter().sum::<f64>() > 0.0);
    world.check_invariants().unwrap();

    let occupied = world.cells().next().unwrap().position;
    let fingerprint = world.state_fingerprint();
    assert!(
        world
            .spawn_cell_with_genome_at_position(occupied, founder_genome("founder-overlap"))
            .is_err()
    );
    assert_eq!(world.state_fingerprint(), fingerprint);

    let mut config = small_config("founder-scarcity", 1, 1);
    uniform_fields(&mut config, [0.1, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let mut barren = World::new(config).unwrap();
    let tile = barren.tile_id(0, 0).unwrap();
    let amounts = barren.tile_element_amounts(tile).unwrap();
    assert_eq!(barren.spawn_founder_cells(3).unwrap(), 0);
    assert_eq!(barren.cell_count(), 0);
    assert_eq!(barren.tile_element_amounts(tile).unwrap(), amounts);
    assert_eq!(barren.energy_ledger().founder_energy, 0.0);
}

#[test]
fn recharge_conserves_matter_and_lifts_f_by_the_sign_of_enval() {
    let mut config = small_config("recharge-sign", 64, 48);
    config.enval_landscape.relaxation_per_second = 0.0;
    config.enval_diffusivity_per_second = 0.0;
    config.element_fields.diffusivities = microcosmcore::ElementAmounts::ZERO;
    let mut world = World::new(config).unwrap();
    let initial = total_matter(&world);
    let enval = enval_field(&world);
    let before = element_field(&world);
    world.step_many(500);
    let after = element_field(&world);
    assert!(relative_change(initial, total_matter(&world)) <= MATTER_REL_TOL);
    assert!(world.enval_ledger().recharge_amount > 0.0);

    let (mut made_d, mut made_e) = (false, false);
    for tile in 0..world.tile_count() {
        let (old, new) = (before[tile], after[tile]);
        for element in [Element::A, Element::B, Element::C] {
            assert_eq!(new[element], old[element], "tile {tile} {element}");
        }
        assert!(new[Element::F] <= old[Element::F], "tile {tile}: F rose");
        if enval[tile] <= 0.0 {
            assert_eq!(new[Element::D], old[Element::D], "tile {tile}: D rose");
        }
        if enval[tile] >= 0.0 {
            assert_eq!(new[Element::E], old[Element::E], "tile {tile}: E rose");
        }
        made_d |= new[Element::D] > old[Element::D];
        made_e |= new[Element::E] > old[Element::E];
    }
    assert!(made_d && made_e);
    world.check_invariants().unwrap();
}

#[test]
fn a_cell_in_a_uniform_field_equilibrates_without_overshoot() {
    let mut config = small_config("equilibrium", 6, 6);
    uniform_fields(&mut config, [2.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    config.cell_motion.diffusivity = 0.0;
    let mut world = World::new(config).unwrap();
    let id = world
        .spawn_cell_with_genome_at_position(Position::new(2.0, 3.0), founder_genome("equilibrium"))
        .unwrap();
    let body_density = world.config().cell_body.body_density;
    let concentrations = |world: &World| {
        let cell = world.cell(id).unwrap();
        let inside = f64::from(cell.internal_elements[Element::A]) / cell.area(body_density);
        let outside = f64::from(world.sample_element_fields_at(cell.position).unwrap()[Element::A]);
        (inside, outside)
    };
    let mut previous = concentrations(&world).0;
    for tick in 1..=1_500 {
        world.step();
        let (inside, outside) = concentrations(&world);
        assert!(inside >= previous, "tick {tick}: inside fell");
        assert!(
            inside <= outside + 1.0e-6,
            "tick {tick}: inside {inside} overshot outside {outside}"
        );
        previous = inside;
    }
    let (inside, outside) = concentrations(&world);
    assert!(
        relative_change(outside, inside) <= 1.0e-4,
        "inside {inside} outside {outside}"
    );
    let cell = world.cell(id).unwrap();
    for element in &ELEMENT_ORDER[1..] {
        assert_eq!(cell.internal_elements[*element], 0.0);
    }
    assert_eq!(world.reaction_counters().successes, 0);
}

#[test]
fn elements_cross_faster_the_closer_their_polarity_is_to_the_membrane() {
    let mut config = small_config("polarity-transport", 8, 8);
    uniform_fields(&mut config, [1.0; ELEMENT_COUNT]);
    config.cell_motion.diffusivity = 0.0;
    config.enval_recharge.rate_per_second = 0.0;
    let mut world = World::new(config).unwrap();
    let id = world
        .spawn_cell_with_genome_at_position(Position::new(4.0, 4.0), inert_genome("polarity"))
        .unwrap();
    world.step();
    let cell = world.cell(id).unwrap();
    let membrane = cell.membrane_polarity();
    let mut order = ELEMENT_ORDER.to_vec();
    order.sort_by(|a, b| {
        let distance =
            |element: &Element| (f64::from(element.properties().polarity) - membrane).abs();
        distance(a).total_cmp(&distance(b))
    });
    for pair in order.windows(2) {
        assert!(
            cell.internal_elements[pair[0]] > cell.internal_elements[pair[1]],
            "{} should cross faster than {} (membrane polarity {membrane})",
            pair[0],
            pair[1]
        );
    }
    assert_eq!(world.reaction_counters().successes, 0);
}

#[test]
fn initial_matter_is_recorded_at_creation() {
    let mut world = World::new(small_config("initial-matter", 48, 36)).unwrap();
    let relative = |a: f64, b: f64| (a - b).abs() / a.abs().max(1.0);
    assert!(relative(world.initial_matter(), world.total_matter()) <= 1.0e-12);
    world.spawn_founder_cells(8).unwrap();
    assert_eq!(world.cell_count(), 8);
    let drift = relative(world.initial_matter(), world.total_matter());
    assert!(
        drift <= FOUNDER_MATTER_REL_TOL,
        "founders moved total matter by {drift:e}"
    );
    let system = world.compact_stats().system_element_amounts.iter().sum();
    assert!(relative(world.total_matter(), system) <= 1.0e-12);
}

#[test]
fn matter_law_helper_accepts_and_rejects() {
    assert!(check_matter_conservation(1000.0, 1000.5).is_ok());
    assert!(matches!(
        check_matter_conservation(1000.0, 1002.0),
        Err(InvariantError::MatterNotConserved { .. })
    ));
    assert!(check_matter_conservation(0.0, 0.0005).is_ok());
    assert!(check_matter_conservation(0.0, 0.002).is_err());
    assert!(check_matter_conservation(1000.0, f64::NAN).is_err());
    assert!(check_matter_conservation(1000.0, f64::INFINITY).is_err());
    const { assert!(MATTER_CONSERVATION_TOLERANCE > MATTER_REL_TOL) };
}

#[test]
fn matter_law_holds_over_a_living_run() {
    let mut world = founded_world("matter-law", 64, 48, 8);
    for tick in 1..=2_000 {
        world.step();
        if tick % 250 == 0 {
            world.check_invariants().unwrap();
        }
    }
    assert!(world.compact_stats().fissions > 0, "no divisions");
    let drift = world.matter_drift();
    assert!(
        drift.abs() <= MATTER_REL_TOL,
        "matter drifted by {drift:e} over 2,000 ticks"
    );
}
