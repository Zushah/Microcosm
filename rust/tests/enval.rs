mod common;

use common::*;
use microcosmcore::{Config, TileId, World};

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn the_enval_ledger_closes_with_landscape_cells_recharge_and_edits() {
    let mut world = founded_world("enval-ledger-landscape", 64, 48, 12);
    for tick in 1..=2_000 {
        world.step();
        match tick {
            300 => world.adjust_tile_enval(TileId(5), 2.0).unwrap(),
            900 => world.set_tile_enval(TileId(3), -3.0).unwrap(),
            1_400 => {
                for x in 8..24 {
                    for y in 8..24 {
                        let tile = world.tile_id(x, y).unwrap();
                        world.set_tile_enval(tile, 0.05).unwrap();
                    }
                }
            }
            _ => {}
        }
        let residual = world.enval_ledger_residual().abs();
        let allowed = enval_tolerance(&world);
        assert!(
            residual <= allowed,
            "tick {tick}: enval ledger residual {residual} exceeds {allowed}"
        );
    }
    let ledger = world.enval_ledger();
    assert!(ledger.source_inflow != 0.0);
    assert!(ledger.recharge != 0.0);
    assert!(ledger.edits != 0.0);
    assert!(ledger.cell_uptake != 0.0 || ledger.cell_emission != 0.0);
    assert!(world.reaction_counters().successes > 0);
    assert!(world.cell_count() > 0);
    world.check_invariants().unwrap();
}

fn cell_free_config(seed: &str) -> Config {
    let mut config = small_config(seed, 64, 48);
    config.enval_landscape.relaxation_per_second = 0.0;
    config.enval_recharge.rate_per_second = 0.0;
    config
}

#[test]
fn diffusion_conserves_enval_and_leaves_a_uniform_field_unchanged() {
    let mut uniform = World::new(cell_free_config("enval-diffusion-uniform")).unwrap();
    for tile in tiles(&uniform) {
        uniform.set_tile_enval(tile, 0.25).unwrap();
    }
    let before = enval_field(&uniform);
    uniform.step_many(200);
    assert_eq!(bits(&enval_field(&uniform)), bits(&before));

    let mut landscape = World::new(cell_free_config("enval-diffusion-landscape")).unwrap();
    let before = enval_field(&landscape);
    let initial = total_enval(&landscape);
    landscape.step_many(200);
    assert_ne!(enval_field(&landscape), before);
    let drift = (total_enval(&landscape) - initial).abs();
    let allowed = enval_tolerance(&landscape);
    assert!(
        drift <= allowed,
        "enval drifted by {drift}, allowed {allowed}"
    );
    assert_ledgers_close(&landscape);
    landscape.check_invariants().unwrap();
}

#[test]
fn recharge_spends_enval_equal_to_the_energy_it_lifts_and_never_flips_a_sign() {
    for (seed, saturating, ticks) in [
        ("recharge-first-law", false, 500),
        ("recharge-saturated", true, 50),
    ] {
        let mut config = small_config(seed, 64, 48);
        config.enval_landscape.relaxation_per_second = 0.0;
        config.enval_diffusivity_per_second = 0.0;
        if saturating {
            uniform_fields(&mut config, [0.0, 0.0, 0.0, 0.0, 0.0, 2.0]);
            config.enval_recharge.rate_per_second = 1_000.0;
        }
        let mut world = World::new(config).unwrap();
        let initial = enval_field(&world);
        let intrinsic_before = field_intrinsic_energy(&world);
        let magnitude_before = total_abs_enval(&world);
        let matter_before = total_matter(&world);
        for tick in 1..=ticks {
            let before = enval_field(&world);
            world.step();
            for (tile, (old, new)) in before.iter().zip(enval_field(&world)).enumerate() {
                assert!(
                    new == 0.0 || new.signum() == old.signum(),
                    "{seed} tick {tick} tile {tile}: recharge flipped {old} to {new}"
                );
                assert!(
                    new.abs() <= old.abs(),
                    "{seed} tick {tick} tile {tile}: |enval| grew"
                );
            }
        }
        if saturating {
            let exhausted = initial
                .iter()
                .zip(enval_field(&world))
                .filter(|(old, new)| new.abs() <= 1.0e-6 * old.abs())
                .count();
            assert!(
                exhausted > world.tile_count() / 2,
                "only {exhausted} tiles exhausted"
            );
        }
        let ledger = world.enval_ledger();
        let lifted = field_intrinsic_energy(&world) - intrinsic_before;
        let spent = magnitude_before - total_abs_enval(&world);
        assert!(ledger.recharge_energy > 0.0);
        assert!(ledger.recharge_amount > 0.0);
        for (name, value) in [("lifted", lifted), ("spent", spent)] {
            let disagreement = relative_change(ledger.recharge_energy, value);
            assert!(
                disagreement <= RECHARGE_REL_TOL,
                "{seed}: {name} {value} vs recharge_energy {} ({disagreement:e})",
                ledger.recharge_energy
            );
        }
        assert!(relative_change(matter_before, total_matter(&world)) <= MATTER_REL_TOL);
        assert_ledgers_close(&world);
    }
}

#[test]
fn enval_edits_are_ledgered() {
    let mut config = cell_free_config("enval-edits");
    config.enval_diffusivity_per_second = 0.0;
    let mut world = World::new(config).unwrap();
    let tile = world.tile_id(10, 20).unwrap();

    let old = world.tile_enval(tile).unwrap();
    let edits = world.enval_ledger().edits;
    world.set_tile_enval(tile, 1.75).unwrap();
    assert_eq!(world.tile_enval(tile), Some(1.75));
    assert_eq!(world.enval_ledger().edits - edits, 1.75 - f64::from(old));

    let edits = world.enval_ledger().edits;
    world.adjust_tile_enval(tile, -0.5).unwrap();
    assert_eq!(world.tile_enval(tile), Some(1.25));
    assert_eq!(world.enval_ledger().edits - edits, -0.5);
    assert!(world.enval_ledger_residual().abs() <= 1.0e-12 * total_abs_enval(&world));

    world.step_many(10);
    assert_eq!(world.tile_enval(tile), Some(1.25));
    assert!(world.enval_ledger_residual().abs() <= 1.0e-12 * total_abs_enval(&world));

    let field = enval_field(&world);
    let ledger = world.enval_ledger();
    let outside = TileId(world.tile_count());
    assert!(world.set_tile_enval(outside, 1.0).is_err());
    assert!(world.adjust_tile_enval(outside, 1.0).is_err());
    assert!(world.set_tile_enval(tile, f32::NAN).is_err());
    assert!(world.set_tile_enval(tile, f32::INFINITY).is_err());
    let _ = world.adjust_tile_enval(tile, f32::NAN);
    assert_eq!(bits(&enval_field(&world)), bits(&field));
    assert_eq!(world.enval_ledger(), ledger);

    world.set_tile_enval(tile, f32::MAX).unwrap();
    let field = enval_field(&world);
    let ledger = world.enval_ledger();
    assert!(world.adjust_tile_enval(tile, f32::MAX).is_err());
    assert_eq!(bits(&enval_field(&world)), bits(&field));
    assert_eq!(world.enval_ledger(), ledger);
}

#[test]
fn enval_ledger_closes_within_its_tolerance() {
    let mut world = founded_world("enval-ledger-tolerance", 64, 48, 8);
    world.step_many(1_500);
    let ledger = world.enval_ledger();
    assert!(ledger.magnitude() > 0.0);
    let residual = world.enval_ledger_residual();
    assert!(
        residual.abs() <= ledger.closure_tolerance(),
        "enval ledger residual {residual:e} exceeds {:e}",
        ledger.closure_tolerance()
    );
}
