mod common;

use std::collections::BTreeSet;

use common::*;
use microcosmcore::{CellId, Position, World, toroidal_distance};

fn live_ids(world: &World) -> BTreeSet<usize> {
    world.cells().map(|cell| cell.id.index()).collect()
}

#[test]
fn fission_splits_a_cell_into_touching_daughters_of_one_lineage() {
    let mut config = small_config("fission", 24, 24);
    config.cell_motion.diffusivity = 0.0;
    let mut world = World::new(config).unwrap();
    assert_eq!(world.spawn_founder_cells(4).unwrap(), 4);
    let (width, height) = (world.width() as f32, world.height() as f32);
    for tick in 1..=2_000 {
        let seen = live_ids(&world);
        let matter = total_matter(&world);
        let before = world.compact_stats();
        world.step();
        let after = world.compact_stats();
        if after.fissions != before.fissions + 1 || after.deaths != before.deaths {
            continue;
        }
        let newborns = world
            .cells()
            .filter(|cell| !seen.contains(&cell.id.index()))
            .collect::<Vec<_>>();
        assert_eq!(newborns.len(), 1, "tick {tick}");
        let newborn = newborns[0];
        let gap = world
            .cells()
            .filter(|cell| cell.id != newborn.id && cell.lineage_id == newborn.lineage_id)
            .map(|sibling| {
                (toroidal_distance(newborn.position, sibling.position, width, height)
                    - newborn.radius
                    - sibling.radius)
                    .abs()
            })
            .fold(f32::INFINITY, f32::min);
        assert!(
            gap <= 1.0e-4,
            "tick {tick}: newborn is {gap} from its sibling"
        );
        assert!(relative_change(matter, total_matter(&world)) <= MATTER_REL_TOL);
        assert_ledgers_close(&world);
        world.check_invariants().unwrap();
        return;
    }
    panic!("no clean division within 2,000 ticks");
}

#[test]
fn a_starving_cell_returns_all_its_matter() {
    let mut config = small_config("starvation", 16, 16);
    config.founder_energy = 0.05;
    let mut world = World::new(config).unwrap();
    let genome = inert_genome("starvation");
    for column in 0..4 {
        for row in 0..4 {
            let position = Position::new(column as f32 * 4.0 + 2.5, row as f32 * 4.0 + 2.5);
            world
                .spawn_cell_with_genome_at_position(position, genome.clone())
                .unwrap();
        }
    }
    let matter = total_matter(&world);
    for _ in 0..1_000 {
        if world.cell_count() == 0 {
            break;
        }
        world.step();
    }
    assert_eq!(world.cell_count(), 0, "cells outlived their energy");
    let stats = world.compact_stats();
    assert_eq!(stats.energy_deaths, 16);
    assert_eq!(stats.lyses, 0);
    assert_eq!(stats.intracellular_element_amounts, [0.0; 6]);
    assert!(relative_change(matter, stats.total_element_amount) <= MATTER_REL_TOL);
    assert!(world.energy_ledger().maintenance > 0.0);
    assert_ledgers_close(&world);
    world.check_invariants().unwrap();
}

#[test]
fn a_ruptured_cell_returns_all_its_matter() {
    let mut config = small_config("rupture", 8, 8);
    uniform_fields(&mut config, [0.5, 0.0, 0.0, 3.0, 0.0, 0.0]);
    config.cell_motion.diffusivity = 0.0;
    config.founder_energy = 5.0;
    let mut world = World::new(config).unwrap();
    let mut genome = founder_genome("rupture");
    genome.enzymes = vec![body_grower()];
    world
        .spawn_cell_with_genome_at_position(Position::new(4.5, 4.5), genome)
        .unwrap();
    let matter = total_matter(&world);
    for _ in 0..1_000 {
        if world.cell_count() == 0 {
            break;
        }
        world.step();
        world.check_invariants().unwrap();
    }
    assert_eq!(world.cell_count(), 0, "the cell never ruptured");
    let stats = world.compact_stats();
    assert_eq!(stats.lyses, 1);
    assert_eq!(stats.energy_deaths, 0);
    assert_eq!(stats.intracellular_element_amounts, [0.0; 6]);
    assert!(relative_change(matter, stats.total_element_amount) <= MATTER_REL_TOL);
    let ledger = world.energy_ledger();
    assert!(ledger.assembly_cost > 0.0);
    assert!(
        ledger.death_loss > 0.0,
        "a ruptured cell takes its energy with it"
    );
    assert_ledgers_close(&world);
}

#[test]
fn cell_ids_are_stable_and_never_reused() {
    let mut world = founded_world("cell-ids", 32, 24, 12);
    let mut previous = live_ids(&world);
    let mut retired = BTreeSet::new();
    let mut highest = previous.iter().max().copied().unwrap();
    for _ in 0..2_500 {
        world.step();
        let current = live_ids(&world);
        assert_eq!(current.len(), world.cell_count(), "ids are not unique");
        for id in previous.difference(&current) {
            assert!(world.cell(CellId(*id)).is_none());
            retired.insert(*id);
        }
        for id in current.difference(&previous) {
            assert!(*id > highest, "id {id} was issued after {highest}");
        }
        assert!(current.is_disjoint(&retired), "a retired id came back");
        for id in &current {
            assert_eq!(world.cell(CellId(*id)).unwrap().id, CellId(*id));
        }
        highest = highest.max(current.iter().max().copied().unwrap_or(highest));
        previous = current;
    }
    assert!(!retired.is_empty(), "no cell died");
    assert!(highest >= 12 + retired.len(), "too few births");
}

#[test]
fn invariants_hold_every_tick() {
    for seed in ["invariants-1", "invariants-2", "invariants-3"] {
        let mut world = founded_world(seed, 12, 10, 5);
        for tick in 1..=300 {
            world.step();
            world
                .check_invariants()
                .unwrap_or_else(|error| panic!("seed {seed} tick {tick}: {error}"));
            let population = world
                .lineage_counters()
                .values()
                .map(|counters| counters.population)
                .sum::<u64>();
            assert_eq!(population as usize, world.cell_count());
        }
        assert!(world.reaction_counters().successes > 0);
    }
}

#[test]
fn fission_increments_generation_of_both_daughters() {
    let mut world = founded_world("generation", 32, 32, 6);
    assert!(world.cells().all(|cell| cell.generation == 0));
    let mut fission_tick = None;
    for tick in 1..=3_000 {
        let deaths_before = world.compact_stats().deaths;
        world.step();
        let stats = world.compact_stats();
        if stats.fissions > 0 {
            assert_eq!(stats.fissions, 1, "two divisions in tick {tick}");
            fission_tick = Some((tick, stats.deaths == deaths_before));
            break;
        }
    }
    let (tick, no_deaths) = fission_tick.expect("no division within 3,000 ticks");
    let generations = world
        .cells()
        .map(|cell| cell.generation)
        .collect::<Vec<_>>();
    let daughters = generations.iter().filter(|g| **g == 1).count();
    assert!(
        generations.iter().all(|g| *g <= 1),
        "tick {tick}: {generations:?}"
    );
    if no_deaths {
        assert_eq!(daughters, 2, "tick {tick}: {generations:?}");
    } else {
        assert!((1..=2).contains(&daughters), "tick {tick}: {generations:?}");
    }
}
