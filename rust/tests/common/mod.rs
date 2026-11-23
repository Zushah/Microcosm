#![allow(dead_code)]

use microcosmcore::{
    Config, ELEMENT_COUNT, ElementAmounts, Enzyme, Genome, LineageId, Rng, TileId, World,
};

pub const MATTER_REL_TOL: f64 = 5.0e-5;
pub const ENVAL_REL_TOL: f64 = 4.0e-5;
pub const RECHARGE_REL_TOL: f64 = 1.0e-4;

pub fn small_config(seed: &str, width: usize, height: usize) -> Config {
    Config {
        seed: seed.to_owned(),
        width,
        height,
        ..Config::default()
    }
}

pub fn uniform_fields(config: &mut Config, amounts: [f32; ELEMENT_COUNT]) {
    config.element_fields.initial_amounts = ElementAmounts::new(amounts);
    config.element_fields.heterogeneity = 0.0;
}

pub fn founded_world(seed: &str, width: usize, height: usize, founders: usize) -> World {
    let mut world = World::new(small_config(seed, width, height)).unwrap();
    world.spawn_founder_cells(founders).unwrap();
    world
}

pub fn founder_genome(seed: &str) -> Genome {
    Genome::random_founder(&mut Rng::from_seed_str(seed), 0.0)
}

pub fn inert_enzyme() -> Enzyme {
    let mut enzyme = Enzyme::founder_downhill();
    enzyme.enval_optimum = 50.0;
    enzyme.enval_sigma = 1.0e-4;
    enzyme
}

pub fn inert_genome(seed: &str) -> Genome {
    let mut genome = founder_genome(seed);
    genome.enzymes = vec![inert_enzyme()];
    genome
}

pub fn body_grower() -> Enzyme {
    let mut enzyme = Enzyme::founder_downhill();
    enzyme.secretion_fraction = 0.0;
    enzyme.body_fraction = 1.0;
    enzyme.boundary_fraction = 0.0;
    enzyme.enval_optimum = 0.0;
    enzyme.enval_sigma = 1000.0;
    enzyme
}

pub fn costly_uphill_enzyme(enval_optimum: f32) -> Enzyme {
    let mut enzyme = Enzyme::metabolic(
        ElementAmounts::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ElementAmounts::new([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        10.0,
        0.65,
        0.5,
    );
    enzyme.enval_optimum = enval_optimum;
    enzyme.enval_sigma = 10.0;
    enzyme
}

pub fn with_enval_optimum(mut genome: Genome, enval_optimum: f32) -> Genome {
    for enzyme in &mut genome.enzymes {
        enzyme.enval_optimum = enval_optimum;
    }
    genome
}

pub fn with_lineage(mut genome: Genome, lineage: u64) -> Genome {
    genome.lineage_id = LineageId(lineage);
    genome
}

pub fn tiles(world: &World) -> impl Iterator<Item = TileId> + use<> {
    (0..world.tile_count()).map(TileId)
}

pub fn enval_field(world: &World) -> Vec<f32> {
    tiles(world)
        .map(|tile| world.tile_enval(tile).unwrap())
        .collect()
}

pub fn element_field(world: &World) -> Vec<ElementAmounts> {
    tiles(world)
        .map(|tile| world.tile_element_amounts(tile).unwrap())
        .collect()
}

pub fn total_matter(world: &World) -> f64 {
    world.compact_stats().total_element_amount
}

pub fn total_enval(world: &World) -> f64 {
    enval_field(world)
        .iter()
        .map(|value| f64::from(*value))
        .sum()
}

pub fn total_abs_enval(world: &World) -> f64 {
    enval_field(world)
        .iter()
        .map(|value| f64::from(value.abs()))
        .sum()
}

pub fn field_intrinsic_energy(world: &World) -> f64 {
    element_field(world)
        .iter()
        .map(|amounts| amounts.intrinsic_energy())
        .sum()
}

pub fn relative_change(before: f64, after: f64) -> f64 {
    (after - before).abs() / before.abs().max(f64::MIN_POSITIVE)
}

pub fn enval_tolerance(world: &World) -> f64 {
    ENVAL_REL_TOL * total_abs_enval(world).max(1.0)
}

pub fn assert_ledgers_close(world: &World) {
    let energy = world.energy_ledger_residual();
    let tolerance = world.energy_ledger().closure_tolerance();
    assert!(
        energy.abs() <= tolerance,
        "energy ledger residual {energy} exceeds {tolerance}"
    );
    let enval = world.enval_ledger_residual();
    let tolerance = enval_tolerance(world);
    assert!(
        enval.abs() <= tolerance,
        "enval ledger residual {enval} exceeds {tolerance}"
    );
}
