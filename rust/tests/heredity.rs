mod common;

use std::collections::BTreeSet;

use common::*;
use microcosmcore::{
    Enzyme, GENOME_PATCH_SCHEMA, Genome, GenomePatch, LineageId, MAX_CELL_ENZYMES,
    MIN_CELL_ENZYMES, Position, Rng, World,
};

fn genes(enzyme: &Enzyme) -> Vec<f32> {
    let mut genes = enzyme.reactants.as_array().to_vec();
    genes.extend_from_slice(enzyme.products.as_array());
    genes.extend([
        enzyme.rate,
        enzyme.half_saturation,
        enzyme.energy_harvest_fraction,
        enzyme.secretion_fraction,
        enzyme.body_fraction,
        enzyme.boundary_fraction,
        enzyme.enval_optimum,
        enzyme.enval_sigma,
        enzyme.enval_throughput,
        enzyme.enval_energy_fraction,
        enzyme.enval_release_fraction,
        enzyme.enval_pump,
        enzyme.exposure,
    ]);
    genes
}

#[test]
fn mutation_keeps_genomes_valid_over_many_generations() {
    let mut rng = Rng::from_seed_str("mutation-validity");
    let mut count_changed = false;
    let mut gene_changed = false;
    for lineage in 0..64 {
        let mut genome = founder_genome(&format!("mutation-validity-{lineage}"));
        for generation in 0..200 {
            let child = genome.mutate(&mut rng);
            child.validate().unwrap_or_else(|error| {
                panic!("lineage {lineage} generation {generation}: {error}")
            });
            assert!((MIN_CELL_ENZYMES..=MAX_CELL_ENZYMES).contains(&child.enzymes.len()));
            assert_eq!(child.lineage_id, genome.lineage_id);
            for enzyme in &child.enzymes {
                assert!(genes(enzyme).iter().all(|gene| gene.is_finite()));
                let routed =
                    enzyme.secretion_fraction + enzyme.body_fraction + enzyme.boundary_fraction;
                assert!(routed <= 1.0 + 1.0e-6, "routing fractions sum to {routed}");
            }
            count_changed |= child.enzymes.len() != genome.enzymes.len();
            gene_changed |= child.enzymes.len() == genome.enzymes.len()
                && child
                    .enzymes
                    .iter()
                    .zip(&genome.enzymes)
                    .any(|(new, old)| genes(new) != genes(old));
            genome = child;
        }
    }
    assert!(count_changed, "the enzyme count never changed");
    assert!(gene_changed, "no gene ever changed");
}

#[test]
fn mutation_is_deterministic_given_the_rng() {
    let mut first_rng = Rng::from_seed_str("mutation-determinism");
    let mut second_rng = Rng::from_seed_str("mutation-determinism");
    let mut first = founder_genome("mutation-determinism");
    let mut second = first.clone();
    let founder = first.clone();
    for _ in 0..100 {
        first = first.mutate(&mut first_rng);
        second = second.mutate(&mut second_rng);
        assert_eq!(first, second);
    }
    assert_ne!(first, founder);
}

#[test]
fn daughters_inherit_their_lineage() {
    let mut world = founded_world("inheritance", 32, 24, 8);
    let founders = world
        .cells()
        .map(|cell| cell.lineage_id)
        .collect::<BTreeSet<_>>();
    assert_eq!(founders.len(), 8);
    world.step_many(1_500);
    assert!(world.compact_stats().births > 8 + 8, "too few births");
    assert!(world.cell_count() > 0);
    for cell in world.cells() {
        assert!(founders.contains(&cell.lineage_id));
        assert_eq!(cell.genome.lineage_id, cell.lineage_id);
    }
}

#[test]
fn lineage_labels_never_change_the_trajectory() {
    let run = |distinct_lineages: bool| {
        let mut config = small_config("lineage-blind", 16, 12);
        config.founder_energy = 4.0;
        let mut world = World::new(config).unwrap();
        let mut genome_rng = Rng::from_seed_str("lineage-blind-genomes");
        for index in 0..12_u64 {
            let genome = Genome::random_founder(&mut genome_rng, world.average_enval());
            let lineage = if distinct_lineages { 100 + index } else { 7 };
            let position = Position::new(
                2.5 + (index % 4) as f32 * 3.0,
                2.5 + (index / 4) as f32 * 3.0,
            );
            world
                .spawn_cell_with_genome_at_position(position, with_lineage(genome, lineage))
                .unwrap();
        }
        world.step_many(1_000);
        world.check_invariants().unwrap();
        world
    };
    let physics = |world: &World| {
        let cells = world
            .cells()
            .cloned()
            .map(|mut cell| {
                cell.lineage_id = LineageId(0);
                cell.genome.lineage_id = LineageId(0);
                cell
            })
            .collect::<Vec<_>>();
        serde_json::to_string(&(
            cells,
            world.rng(),
            element_field(world),
            enval_field(world),
            world.energy_ledger(),
            world.enval_ledger(),
            world.reaction_counters(),
            world.compact_stats().deaths,
        ))
        .unwrap()
    };
    let same = run(false);
    let distinct = run(true);
    assert_eq!(same.extant_lineage_count(), 1);
    assert!(distinct.extant_lineage_count() > 1);
    let stats = same.compact_stats();
    assert!(stats.fissions > 0, "no cell divided");
    assert!(stats.deaths > 0, "no cell died");
    assert_eq!(physics(&same), physics(&distinct));
}

fn parse(json: &str) -> GenomePatch {
    serde_json::from_str(json).unwrap()
}

#[test]
fn genome_patches_round_trip_and_reject_old_schemas() {
    let mut world = founded_world("genome-patch", 16, 12, 4);
    let id = world.cells().next().unwrap().id;
    let enzymes_before = world.cell(id).unwrap().genome.enzymes.len();
    let patch = parse(&format!(
        r#"{{"schema":"{GENOME_PATCH_SCHEMA}","enzymes":[
            {{"op":"update","index":0,"fields":{{"rate":1.25,"exposure":0.375,"enval_optimum":-0.5}}}},
            {{"op":"append","enzyme":{{"rate":0.75,"reactants":[0,1,0,0,0,0],"products":[0,0,1,0,0,0]}}}}
        ]}}"#
    ));
    let result = world.apply_cell_genome_patch(id, &patch).unwrap();
    assert_eq!(result.patched_cell_count, 1);
    for field in [
        "enzyme[0].rate",
        "enzyme[0].exposure",
        "enzyme[0].enval_optimum",
    ] {
        assert!(result.changed_fields.iter().any(|changed| changed == field));
    }
    let detail = world.inspect_cell_detail(id, 4).unwrap();
    let enzymes = &detail.genome.enzymes;
    assert_eq!(enzymes.len(), enzymes_before + 1);
    assert_eq!(enzymes[0].rate, 1.25);
    assert_eq!(enzymes[0].exposure, 0.375);
    assert_eq!(enzymes[0].enval_optimum, -0.5);
    let appended = enzymes.last().unwrap();
    assert_eq!(appended.rate, 0.75);
    assert_eq!(appended.reactants, [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
    assert_eq!(appended.products, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
    world.step_many(10);
    world.check_invariants().unwrap();

    let genome = world.cell(id).unwrap().genome.clone();
    let stale = parse(
        r#"{"schema":"microcosm.genome_patch.v5","enzymes":[{"op":"update","index":0,"fields":{"rate":2.0}}]}"#,
    );
    assert!(world.apply_cell_genome_patch(id, &stale).is_err());
    assert_eq!(world.cell(id).unwrap().genome, genome);
    for unknown in [
        r#"{"schema":"microcosm.genome_patch.v6","colour":"red"}"#,
        r#"{"enzymes":[{"op":"update","index":0,"fields":{"level":7}}]}"#,
        r#"{"enzymes":[{"op":"update","index":0,"fields":{"rate":1.0},"extra":1}]}"#,
    ] {
        assert!(
            serde_json::from_str::<GenomePatch>(unknown).is_err(),
            "{unknown}"
        );
    }
}

#[test]
fn an_invalid_genome_patch_changes_nothing() {
    let mut world = founded_world("genome-patch-atomic", 16, 12, 4);
    world.step_many(20);
    let id = world.cells().next().unwrap().id;
    let genome = world.cell(id).unwrap().genome.clone();
    let fingerprint = world.state_fingerprint();
    let patch = parse(
        r#"{"enzymes":[
            {"op":"update","index":0,"fields":{"rate":3.0}},
            {"op":"update","index":1,"fields":{"exposure":1.5}}
        ]}"#,
    );
    assert!(world.apply_cell_genome_patch(id, &patch).is_err());
    assert!(world.apply_genome_brush(0, 0, 16, 12, &patch).is_err());

    let removals = vec![r#"{"op":"remove","index":0}"#; genome.enzymes.len()].join(",");
    let too_few = parse(&format!(r#"{{"enzymes":[{removals}]}}"#));
    assert!(world.apply_cell_genome_patch(id, &too_few).is_err());
    let appends = [r#"{"op":"append","enzyme":{"rate":0.5}}"#; MAX_CELL_ENZYMES].join(",");
    let too_many = parse(&format!(r#"{{"enzymes":[{appends}]}}"#));
    assert!(world.apply_cell_genome_patch(id, &too_many).is_err());

    assert_eq!(world.cell(id).unwrap().genome, genome);
    assert_eq!(world.state_fingerprint(), fingerprint);
}

#[test]
fn the_genome_brush_patches_every_cell_in_a_wrapped_rectangle() {
    let mut world = founded_world("genome-brush", 12, 10, 30);
    let (width, height) = world.dimensions();
    let covered =
        |x: usize, y: usize| [width - 1, 0, 1].contains(&x) && [height - 1, 0, 1].contains(&y);
    let inside = world
        .cells()
        .filter(|cell| {
            covered(
                cell.position.x.floor() as usize,
                cell.position.y.floor() as usize,
            )
        })
        .map(|cell| cell.id.index())
        .collect::<BTreeSet<_>>();
    assert!(!inside.is_empty(), "no cell sits under the brush");
    assert!(
        inside.len() < world.cell_count(),
        "every cell sits under the brush"
    );
    let before = world
        .cells()
        .map(|cell| (cell.id.index(), cell.genome.clone()))
        .collect::<Vec<_>>();
    let patch = parse(r#"{"enzymes":[{"op":"update","index":0,"fields":{"rate":2.5}}]}"#);
    let result = world.apply_genome_brush(0, 0, 3, 3, &patch).unwrap();
    assert_eq!(result.patched_cell_count, inside.len());
    assert_eq!(result.visited_tile_count, 9);
    for (index, genome) in before {
        let cell = world.cells().find(|cell| cell.id.index() == index).unwrap();
        if inside.contains(&index) {
            assert_eq!(cell.genome.enzymes[0].rate, 2.5);
            assert_eq!(cell.genome.enzymes[1..], genome.enzymes[1..]);
        } else {
            assert_eq!(cell.genome, genome);
        }
    }
}
