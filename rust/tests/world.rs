mod common;

use common::*;
use microcosmcore::{
    CellId, Position, TileId, World, minimum_image_displacement, toroidal_distance, wrap_coordinate,
};

#[test]
fn space_is_a_torus() {
    let world = World::new(small_config("torus", 7, 5)).unwrap();
    assert_eq!(world.tile_count(), 35);
    for x in 0..7 {
        for y in 0..5 {
            let tile = world.tile_id(x, y).unwrap();
            assert_eq!(world.tile_xy(tile), Some((x, y)));
            assert_eq!(world.wrapped_tile_id(x as isize, y as isize), tile);
            assert_eq!(world.wrapped_tile_id(x as isize - 7, y as isize + 5), tile);
        }
    }
    assert_eq!(world.tile_id(7, 0), None);
    assert_eq!(world.tile_id(0, 5), None);
    assert_eq!(world.tile_xy(TileId(35)), None);
    assert_eq!(world.wrapped_tile_id(-1, -1), world.tile_id(6, 4).unwrap());
    assert_eq!(world.wrapped_tile_id(7, 5), world.tile_id(0, 0).unwrap());

    assert_eq!(wrap_coordinate(-0.25, 7.0), 6.75);
    assert_eq!(wrap_coordinate(7.25, 7.0), 0.25);
    assert_eq!(wrap_coordinate(7.0, 7.0), 0.0);
    assert_eq!(wrap_coordinate(3.5, 7.0), 3.5);
    assert!(wrap_coordinate(-1.0e-7, 7.0) < 7.0);
    assert!(wrap_coordinate(-1.0e-30, 7.0) < 7.0);

    let near_origin = Position::new(0.1, 0.1);
    let near_far_corner = Position::new(6.9, 4.9);
    let across_both_seams = toroidal_distance(near_origin, near_far_corner, 7.0, 5.0);
    assert!((across_both_seams - 0.2_f32.hypot(0.2)).abs() < 1.0e-5);
    let across_x = toroidal_distance(Position::new(0.2, 2.0), Position::new(6.8, 2.0), 7.0, 5.0);
    assert!((across_x - 0.4).abs() < 1.0e-5);
    let across_y = toroidal_distance(Position::new(3.0, 4.75), Position::new(3.0, 0.25), 7.0, 5.0);
    assert!((across_y - 0.5).abs() < 1.0e-5);
    let interior = toroidal_distance(Position::new(1.0, 1.0), Position::new(4.0, 3.0), 7.0, 5.0);
    assert!((interior - 13.0_f32.sqrt()).abs() < 1.0e-5);
}

#[test]
fn the_landscape_is_seeded_bounded_and_seamless() {
    let first = World::new(small_config("world-landscape", 320, 240)).unwrap();
    let second = World::new(small_config("world-landscape", 320, 240)).unwrap();
    let other = World::new(small_config("world-landscape-other", 320, 240)).unwrap();
    assert_eq!(first.enval_landscape(), second.enval_landscape());
    assert_ne!(first.enval_landscape(), other.enval_landscape());

    let landscape = first.enval_landscape();
    let magnitude = first.config().enval_landscape.magnitude;
    assert!(
        landscape
            .iter()
            .all(|value| value.is_finite() && value.abs() <= magnitude + 1.0e-6)
    );
    assert_eq!(enval_field(&first), landscape);

    let (width, height) = first.dimensions();
    let at = |x: usize, y: usize| landscape[first.tile_id(x, y).unwrap().index()];
    let mut interior = 0.0_f32;
    for x in 0..width - 1 {
        for y in 0..height - 1 {
            interior = interior
                .max((at(x + 1, y) - at(x, y)).abs())
                .max((at(x, y + 1) - at(x, y)).abs());
        }
    }
    let mut seam = 0.0_f32;
    for y in 0..height {
        seam = seam.max((at(0, y) - at(width - 1, y)).abs());
    }
    for x in 0..width {
        seam = seam.max((at(x, 0) - at(x, height - 1)).abs());
    }
    assert!(
        interior > 0.0 && seam <= interior,
        "seam {seam} interior {interior}"
    );
    let positive = landscape.iter().filter(|value| **value > 0.0).count();
    assert!(positive > landscape.len() / 5 && positive < 4 * landscape.len() / 5);
}

#[test]
fn cell_free_enval_relaxes_toward_the_landscape_at_kappa_per_tick() {
    let mut config = small_config("landscape-relaxation", 96, 72);
    config.enval_diffusivity_per_second = 0.0;
    config.enval_recharge.rate_per_second = 0.0;
    let mut world = World::new(config.clone()).unwrap();
    let targets = world.enval_landscape().to_vec();
    let kappa =
        1.0 - (-f64::from(config.enval_landscape.relaxation_per_second) * config.dt_seconds).exp();
    let edited = TileId(1234);
    world.set_tile_enval(edited, 7.5).unwrap();
    world.set_tile_enval(TileId(77), -3.0).unwrap();
    for _ in 0..300 {
        let before = enval_field(&world);
        let inflow_before = world.enval_ledger().source_inflow;
        world.step();
        let after = enval_field(&world);
        let mut applied = 0.0;
        for tile in 0..world.tile_count() {
            let old = f64::from(before[tile]);
            let expected = old + kappa * (f64::from(targets[tile]) - old);
            assert!(
                (f64::from(after[tile]) - expected).abs() <= 1.0e-6,
                "tile {tile}"
            );
            applied += f64::from(after[tile]) - old;
        }
        let inflow = world.enval_ledger().source_inflow - inflow_before;
        assert!((inflow - applied).abs() <= 1.0e-9);
    }
    let initial_gap = 7.5 - f64::from(targets[edited.index()]);
    let expected_gap = initial_gap * (1.0 - kappa).powi(300);
    let gap = f64::from(world.tile_enval(edited).unwrap()) - f64::from(targets[edited.index()]);
    assert!((gap - expected_gap).abs() <= 1.0e-4 * expected_gap.abs());
    assert!(gap < initial_gap);
    assert_ledgers_close(&world);

    let mut frozen_config = config;
    frozen_config.enval_landscape.relaxation_per_second = 0.0;
    let mut frozen = World::new(frozen_config).unwrap();
    frozen.set_tile_enval(edited, 7.5).unwrap();
    let before = enval_field(&frozen);
    frozen.step_many(200);
    assert_eq!(frozen.enval_ledger().source_inflow, 0.0);
    assert_eq!(
        enval_field(&frozen)
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        before
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
}

fn inert_grid(
    seed: &str,
    configure: impl FnOnce(&mut microcosmcore::Config),
) -> (World, Vec<CellId>) {
    let mut config = small_config(seed, 96, 96);
    uniform_fields(&mut config, [2.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    configure(&mut config);
    let mut world = World::new(config).unwrap();
    let genome = inert_genome(seed);
    let mut ids = Vec::new();
    for column in 0..24 {
        for row in 0..24 {
            let position = Position::new(column as f32 * 4.0 + 2.5, row as f32 * 4.0 + 2.5);
            ids.push(
                world
                    .spawn_cell_with_genome_at_position(position, genome.clone())
                    .unwrap(),
            );
        }
    }
    (world, ids)
}

fn positions(world: &World, ids: &[CellId]) -> Vec<Position> {
    ids.iter()
        .map(|id| world.cell(*id).unwrap().position)
        .collect()
}

#[test]
fn cells_drift_with_mean_squared_displacement_4dt() {
    let (mut world, ids) = inert_grid("drift-msd", |_| {});
    let ticks = 250;
    let (width, height) = (world.width() as f32, world.height() as f32);
    let mut displacement = vec![(0.0_f64, 0.0_f64); ids.len()];
    for _ in 0..ticks {
        let before = positions(&world, &ids);
        world.step();
        for (index, after) in positions(&world, &ids).into_iter().enumerate() {
            let step = minimum_image_displacement(before[index], after, width, height);
            displacement[index].0 += f64::from(step.x);
            displacement[index].1 += f64::from(step.y);
        }
    }
    assert_eq!(world.cell_count(), ids.len());
    let msd = displacement.iter().map(|(x, y)| x * x + y * y).sum::<f64>() / ids.len() as f64;
    let elapsed = ticks as f64 * world.config().dt_seconds;
    let expected = 4.0 * f64::from(world.config().cell_motion.diffusivity) * elapsed;
    assert!(
        (msd / expected - 1.0).abs() <= 0.15,
        "mean squared displacement {msd} vs 4Dt {expected}"
    );
    world.check_invariants().unwrap();
}

#[test]
fn without_drift_cells_stay_put() {
    let (mut world, ids) = inert_grid("drift-off", |config| {
        config.cell_motion.diffusivity = 0.0;
    });
    let before = positions(&world, &ids);
    world.step_many(300);
    assert_eq!(world.cell_count(), ids.len());
    assert_eq!(positions(&world, &ids), before);
    world.check_invariants().unwrap();
}
