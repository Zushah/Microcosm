mod common;

use common::*;
use microcosmcore::{ElementAmounts, Enzyme, Position, World};

#[test]
fn the_energy_ledger_closes_over_long_runs() {
    let mut world = founded_world("energy-ledger-closure", 32, 24, 12);
    world.set_all_live_cell_energy(4.0).unwrap();
    world.set_all_live_cell_energy(3.0).unwrap();
    for tick in 1..=2_400 {
        world.step();
        if tick % 200 == 0 {
            world.check_invariants().unwrap();
        }
    }
    world.check_invariants().unwrap();
    let ledger = world.energy_ledger();
    let residual = world.energy_ledger_residual();
    assert!(
        residual.abs() <= ledger.closure_tolerance(),
        "energy ledger residual {residual}"
    );
    for (term, value) in [
        ("founder_energy", ledger.founder_energy),
        ("injected_energy", ledger.injected_energy),
        ("extracted_energy", ledger.extracted_energy),
        ("chemical_harvest", ledger.chemical_harvest),
        ("maintenance", ledger.maintenance),
    ] {
        assert!(value > 0.0, "{term} never moved");
    }
    assert!(world.compact_stats().energy_deaths > 0);
    assert_eq!(world.stats().energy_ledger, ledger);
}

#[test]
fn no_cell_ever_holds_negative_energy() {
    let mut config = small_config("energy-overdraw", 24, 24);
    config.founder_energy = 0.5;
    let mut world = World::new(config).unwrap();
    for column in 0..4 {
        for row in 0..4 {
            let position = Position::new(column as f32 * 6.0 + 3.5, row as f32 * 6.0 + 3.5);
            let tile = world.wrapped_tile_id(position.x as isize, position.y as isize);
            let local = world.tile_enval(tile).unwrap();
            let mut genome = founder_genome(&format!("energy-overdraw-{column}-{row}"));
            genome.enzymes.push(costly_uphill_enzyme(local));
            world
                .spawn_cell_with_genome_at_position(position, genome)
                .unwrap();
        }
    }
    for tick in 1..=500 {
        world.step();
        world
            .check_invariants()
            .unwrap_or_else(|error| panic!("tick {tick}: {error}"));
        assert!(world.cells().all(|cell| cell.energy >= 0.0));
    }
    assert_ledgers_close(&world);
    let ledger = world.energy_ledger();
    assert!(ledger.chemical_cost > 0.0);
    assert!(world.compact_stats().energy_deaths > 0);
}

#[test]
fn energy_edits_are_ledgered() {
    let mut world = founded_world("energy-edits", 24, 18, 6);
    world.step_many(10);

    let ledger = world.energy_ledger();
    let rise = world.cells().map(|cell| 5.0 - cell.energy).sum::<f64>();
    world.set_all_live_cell_energy(5.0).unwrap();
    assert_eq!(
        world.energy_ledger().injected_energy - ledger.injected_energy,
        rise
    );
    assert_eq!(
        world.energy_ledger().extracted_energy,
        ledger.extracted_energy
    );
    assert!(world.cells().all(|cell| cell.energy == 5.0));
    assert_ledgers_close(&world);

    let ledger = world.energy_ledger();
    let fall = world.cells().map(|cell| cell.energy - 0.25).sum::<f64>();
    world.set_all_live_cell_energy(0.25).unwrap();
    assert_eq!(
        world.energy_ledger().extracted_energy - ledger.extracted_energy,
        fall
    );
    assert_eq!(
        world.energy_ledger().injected_energy,
        ledger.injected_energy
    );
    assert_ledgers_close(&world);

    let ledger = world.energy_ledger();
    let fingerprint = world.state_fingerprint();
    for invalid in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(world.set_all_live_cell_energy(invalid).is_err());
    }
    assert_eq!(world.energy_ledger(), ledger);
    assert_eq!(world.state_fingerprint(), fingerprint);
    assert!(world.cells().all(|cell| cell.energy == 0.25));
}

#[test]
fn catalysts_must_balance_their_matter() {
    let catalyst = |reactants: [f32; 6], products: [f32; 6]| {
        Enzyme::metabolic(
            ElementAmounts::new(reactants),
            ElementAmounts::new(products),
            1.0,
            0.5,
            0.1,
        )
    };
    let balanced = catalyst(
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
    );
    assert!(balanced.validate().is_ok());
    let unbalanced = catalyst(
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
    );
    assert!(unbalanced.validate().is_err());
    let empty = catalyst([0.0; 6], [0.0; 6]);
    assert!(empty.validate().is_err());
    let no_reactants = catalyst([0.0; 6], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    assert!(no_reactants.validate().is_err());
    let no_op = catalyst(
        [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
    );
    assert!(no_op.validate().is_err());
    for founder in [
        Enzyme::founder_downhill(),
        Enzyme::founder_engine_forward(),
        Enzyme::founder_engine_reverse(),
    ] {
        assert!(founder.validate().is_ok());
        assert_eq!(founder.reactants.total(), founder.products.total());
    }
}

fn niche_world(seed: &str, field: f32, enval_optimum: f32) -> World {
    let mut config = small_config(seed, 8, 8);
    config.element_fields.heterogeneity = 0.0;
    config.enval_landscape.relaxation_per_second = 0.0;
    config.enval_recharge.rate_per_second = 0.0;
    config.cell_motion.diffusivity = 0.0;
    let mut world = World::new(config).unwrap();
    for tile in tiles(&world) {
        world.set_tile_enval(tile, field).unwrap();
    }
    let genome = with_enval_optimum(founder_genome(seed), enval_optimum);
    world
        .spawn_cell_with_genome_at_position(Position::new(4.5, 4.5), genome)
        .unwrap();
    world
}

#[test]
fn an_enzyme_with_zero_optimum_never_touches_the_enval_field() {
    let mut world = niche_world("zero-optimum", 0.3, 0.0);
    world.step_many(200);
    assert_eq!(world.cell_count(), 1);
    assert!(
        enval_field(&world)
            .iter()
            .all(|value| value.to_bits() == 0.3_f32.to_bits())
    );
    let enval = world.enval_ledger();
    assert_eq!(enval.cell_uptake, 0.0);
    assert_eq!(enval.cell_emission, 0.0);
    let energy = world.energy_ledger();
    assert_eq!(energy.enval_harvest, 0.0);
    assert!(energy.chemical_harvest > 0.0);
    assert!(world.reaction_counters().successes > 0);
    assert_ledgers_close(&world);
}

#[test]
fn enzymes_draw_enval_of_their_optimums_sign() {
    let mut changes = Vec::new();
    for sign in [1.0_f32, -1.0] {
        let mut world = niche_world("optimum-sign", 0.3 * sign, 0.3 * sign);
        let before = total_enval(&world);
        world.step_many(200);
        assert_eq!(world.cell_count(), 1);
        let change = total_enval(&world) - before;
        assert!(
            change * f64::from(sign) < 0.0,
            "enval of sign {sign} changed by {change}"
        );
        let ledger = world.enval_ledger();
        assert!(ledger.cell_uptake * f64::from(sign) < 0.0);
        assert!(world.energy_ledger().enval_harvest > 0.0);
        assert_ledgers_close(&world);
        changes.push(change);
    }
    assert!(
        (changes[0] + changes[1]).abs() <= 1.0e-6 * changes[0].abs(),
        "opposite signs are not mirror images: {changes:?}"
    );
}
