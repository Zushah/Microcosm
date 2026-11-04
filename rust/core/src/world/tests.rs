use super::cells::CellReactionContext;
use super::fields::sample_smoothed_enval;
use super::{GEOMETRY_TOLERANCE, MAX_TRANSPORT_FRACTION, TileId, World};
use crate::cell::CELL_FLUX_LOG_CAPACITY;
use crate::chem::{ELEMENT_COUNT, ELEMENT_ORDER, Element, ElementAmounts};
use crate::config::Config;
use crate::genome::{
    Enzyme, EnzymeFieldPatch, EnzymePatchOperation, Genome, GenomeFieldPatch, GenomePatch,
    LineageId, MAX_CELL_ENZYMES, MIN_CELL_ENZYMES,
};
use crate::interaction::Contact;
use crate::render_buffers::RenderBuffers;
use crate::spatial::{
    BilinearStencil, Position, REFERENCE_CELL_RADIUS, minimum_image_displacement,
    toroidal_distance_squared,
};

impl World {
    fn step_cell_fresh(&mut self, cell_id: crate::cell::CellId) {
        self.refresh_enval_smoothed();
        if let Some(slot) = self.cells.slot(cell_id) {
            self.step_cell(cell_id, slot);
        }
        self.refresh_average_enval();
    }
}

fn tile_local_enval(world: &World, tile: TileId) -> f32 {
    let (x, y) = world.tile_xy(tile).unwrap();
    world
        .local_enval_at(Position::new(x as f32, y as f32))
        .unwrap()
}

const GOLDEN_FINGERPRINT: &str = "893a85bdd472ea31";

fn live_ids(world: &World) -> Vec<crate::cell::CellId> {
    world.cells.ids().collect()
}

fn small_config(seed: &str, width: usize, height: usize) -> Config {
    Config {
        seed: seed.to_owned(),
        width,
        height,
        ..Config::default()
    }
}

fn only_a_config(seed: &str, width: usize, height: usize) -> Config {
    let mut config = small_config(seed, width, height);
    config.element_fields.initial_amounts = ElementAmounts::new([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    config.element_fields.heterogeneity = 0.0;
    config
}

fn inert_enzyme() -> Enzyme {
    let mut enzyme = Enzyme::founder_downhill();
    enzyme.enval_optimum = 50.0;
    enzyme.enval_sigma = 1.0e-4;
    enzyme
}

fn make_inert(genome: &mut Genome) {
    genome.enzymes = vec![inert_enzyme()];
}

fn set_enval_optimum(genome: &mut Genome, optimum: f32) {
    for enzyme in &mut genome.enzymes {
        enzyme.enval_optimum = optimum;
    }
}

fn assert_valid_reservoir(world: &World, cell_id: crate::cell::CellId) {
    let Some(cell) = world.cell(cell_id) else {
        return;
    };
    for element in ELEMENT_ORDER {
        assert!(cell.internal_elements[element].is_finite());
        assert!(cell.internal_elements[element] >= 0.0);
    }
}

fn golden_world() -> World {
    let config = small_config("fingerprint-golden", 64, 48);
    let founders = config.initial_founder_count;
    let mut world = World::new(config).unwrap();
    world.spawn_founder_cells(founders).unwrap();
    world
}

#[test]
#[cfg_attr(
    not(target_os = "linux"),
    ignore = "pinned fingerprint is glibc-specific"
)]
fn golden_fingerprint_pins_the_trajectory() {
    let mut world = golden_world();
    world.step_many(2_000);
    assert_eq!(
        format!("{:016x}", world.state_fingerprint()),
        GOLDEN_FINGERPRINT,
    );
}

#[test]
fn fingerprint_repeats_and_survives_snapshot_resume() {
    let mut uninterrupted = golden_world();
    uninterrupted.step_many(300);
    let mut repeated = golden_world();
    repeated.step_many(150);
    let bytes = crate::snapshot::to_bytes(&repeated).unwrap();
    let mut resumed = crate::snapshot::from_bytes(&bytes).unwrap();
    assert_eq!(resumed.state_fingerprint(), repeated.state_fingerprint());
    resumed.step_many(150);
    repeated.step_many(150);
    assert_eq!(
        repeated.state_fingerprint(),
        uninterrupted.state_fingerprint()
    );
    assert_eq!(
        resumed.state_fingerprint(),
        uninterrupted.state_fingerprint()
    );
}

#[test]
fn fingerprint_sees_state_changes_but_not_operation_counters() {
    let mut world = golden_world();
    world.step_many(10);
    let before = world.state_fingerprint();

    let mut counted = world.clone();
    counted.operation_counters.cell_steps += 1;
    assert_eq!(counted.state_fingerprint(), before);

    let mut stepped = world.clone();
    stepped.step();
    assert_ne!(stepped.state_fingerprint(), before);

    let mut nudged = world.clone();
    nudged.rng.next_f64();
    assert_ne!(nudged.state_fingerprint(), before);

    let mut nudged = world.clone();
    let id = live_ids(&nudged)[0];
    nudged.cells[id].genome.enzymes[0].exposure += 0.25;
    assert_ne!(nudged.state_fingerprint(), before);

    let mut nudged = world.clone();
    nudged.energy_ledger.pump_cost += 1.0;
    assert_ne!(nudged.state_fingerprint(), before);
}

#[test]
fn watching_is_exclusive_resets_on_change_and_never_changes_trajectory() {
    let mut plain = World::new(small_config("watch-determinism", 16, 12)).unwrap();
    plain.spawn_founder_cells(8).unwrap();
    let mut watched = plain.clone();
    let first = live_ids(&watched)[0];
    let second = live_ids(&watched)[1];
    assert_eq!(
        watched
            .inspect_cell_detail(first, 64)
            .unwrap()
            .recent_fluxes
            .flux_count,
        0
    );
    for tick in 0..500 {
        plain.step();
        watched.step();
        if tick % 10 == 0 {
            let id = if tick < 250 { first } else { second };
            if watched.cell(id).is_some() {
                watched.inspect_cell_detail(id, 64).unwrap();
            }
        }
        assert_eq!(
            bincode::serialize(&plain).unwrap(),
            bincode::serialize(&watched).unwrap()
        );
    }
    assert_eq!(plain.flux_log.flux_count(), 0);
    assert!(watched.flux_log.flux_count() > 0);
    assert!(
        watched
            .flux_log
            .fluxes_newest_first()
            .all(|record| record.cell_id == second.index())
    );
    assert_eq!(
        watched.inspect_cell_fluxes(first, 64).unwrap().flux_count,
        0
    );
    let bytes = crate::snapshot::to_bytes(&watched).unwrap();
    let mut restored = crate::snapshot::from_bytes(&bytes).unwrap();
    assert_eq!(restored.watched_cell_id, None);
    assert_eq!(restored.flux_log.flux_count(), 0);
    watched.step_many(100);
    restored.step_many(100);
    assert_eq!(
        bincode::serialize(&watched).unwrap(),
        bincode::serialize(&restored).unwrap()
    );
}

#[test]
fn snapshot_cell_storage_is_under_one_point_five_kilobytes_per_live_cell() {
    let mut world = if let Ok(path) = std::env::var("MICROCOSM_REFERENCE_SNAPSHOT") {
        crate::snapshot::load_from_path(path).unwrap()
    } else {
        let mut world = World::new(small_config("body-snapshot-size", 64, 48)).unwrap();
        world.spawn_founder_cells(100).unwrap();
        world.inspect_cell_fluxes(live_ids(&world)[0], 64).unwrap();
        world.step_many(100);
        world
    };
    let populated = crate::snapshot::to_bytes(&world).unwrap().len();
    let count = world.cell_count();
    world.cells = crate::cell::CellStore::default();
    let empty = crate::snapshot::to_bytes(&world).unwrap().len();
    println!(
        "snapshot populated={populated} empty={empty} cells={count} bytes_per_cell={:.3}",
        (populated - empty) as f64 / count as f64
    );
    assert!(
        populated - empty <= 1_500 * count,
        "{} bytes / {count} cells",
        populated - empty
    );
}

#[test]
fn mixed_radius_pairs_picking_and_placement_match_brute_force() {
    let mut world = World::new(small_config("mixed-radius", 32, 24)).unwrap();
    world.spawn_founder_cells(80).unwrap();
    for (index, cell) in world.cells.iter_mut().enumerate() {
        let scale = if index % 13 == 0 {
            36.0
        } else {
            0.25 + (index % 5) as f32 * 0.5
        };
        for element in ELEMENT_ORDER {
            cell.body[element] *= scale;
        }
        cell.refresh_radius(world.config.cell_body.body_density);
    }
    world.rebuild_derived_caches();
    let brute_force = |world: &World, reach: f32| {
        let mut expected = Vec::new();
        for a in world.cells.iter() {
            for b in world.cells.iter() {
                if a.id.index() < b.id.index()
                    && toroidal_distance_squared(a.position, b.position, 32.0, 24.0)
                        <= (a.radius + b.radius + reach).powi(2)
                {
                    expected.push((a.id, b.id));
                }
            }
        }
        expected.sort_unstable_by_key(|(a, b)| (a.index(), b.index()));
        expected
    };
    for reach in [0.0, 0.1, 0.5] {
        let (contacts, _) = world.collect_contact_list(reach);
        let mut listed = world
            .contact_pairs
            .iter()
            .map(|&(a, b)| {
                let (a, b) = (
                    world.cells.at_slot(a as usize).id,
                    world.cells.at_slot(b as usize).id,
                );
                if a.index() < b.index() {
                    (a, b)
                } else {
                    (b, a)
                }
            })
            .collect::<Vec<_>>();
        listed.sort_unstable_by_key(|(a, b)| (a.index(), b.index()));
        assert_eq!(listed, brute_force(&world, reach));
        assert!(world.contact_pairs.windows(2).all(|pair| pair[0] < pair[1]));
        if reach >= world.config.surface.contact_gap {
            assert_eq!(
                contacts as usize,
                brute_force(&world, world.config.surface.contact_gap).len()
            );
        }
        for (slot, owner) in world.contact_row_owners.iter().enumerate() {
            assert_eq!(world.cells.at_slot(slot).id, *owner);
            let start = world.contact_row_offsets[slot] as usize;
            let end = world.contact_row_offsets[slot + 1] as usize;
            let mut row = world.contact_row_neighbors[start..end]
                .iter()
                .map(|slot| world.contact_row_owners[*slot as usize])
                .collect::<Vec<_>>();
            row.sort_unstable_by_key(|id| id.index());
            let mut expected = world
                .contact_pairs
                .iter()
                .filter_map(|&(a, b)| {
                    let (a, b) = (a as usize, b as usize);
                    (a == slot).then_some(b).or((b == slot).then_some(a))
                })
                .map(|other| world.contact_row_owners[other])
                .collect::<Vec<_>>();
            expected.sort_unstable_by_key(|id| id.index());
            assert_eq!(row, expected);
        }
        let capacity = world.contact_pairs.capacity();
        let pointer = world.contact_pairs.as_ptr();
        world.collect_contact_list(reach);
        assert_eq!(world.contact_pairs.capacity(), capacity);
        assert_eq!(world.contact_pairs.as_ptr(), pointer);
    }
    for index in 0..500 {
        let position = Position::new(
            (index * 17 % 320) as f32 * 0.1,
            (index * 13 % 240) as f32 * 0.1,
        );
        let expected = world
            .cells
            .iter()
            .filter_map(|cell| {
                let distance = toroidal_distance_squared(position, cell.position, 32.0, 24.0);
                (distance <= (cell.radius + GEOMETRY_TOLERANCE).powi(2))
                    .then_some((cell.id, distance))
            })
            .min_by(|(a, da), (b, db)| da.total_cmp(db).then_with(|| a.index().cmp(&b.index())))
            .map(|(id, _)| id);
        assert_eq!(world.pick_cell(position), expected);
        for radius in [0.1, 0.45, 2.5] {
            let clear = world.cells.iter().all(|cell| {
                toroidal_distance_squared(position, cell.position, 32.0, 24.0)
                    >= (radius + cell.radius - GEOMETRY_TOLERANCE).max(0.0).powi(2)
            });
            assert_eq!(world.position_is_clear(position, radius, None), clear);
        }
    }
}

#[test]
fn overlap_resolution_separates_disturbed_fission_daughters() {
    let mut world = World::new(small_config("daughter-overlap", 16, 12)).unwrap();
    let id = world.spawn_random_cell().unwrap().unwrap();
    for element in ELEMENT_ORDER {
        world.cells[id].boundary[element] *= 1.6;
    }
    world.fission_cell(id);
    let child_id = world.cells.ids().find(|other| *other != id).unwrap();
    let center = world.cells[id].position;
    world.update_cell_position(child_id, center);
    let rng_before = bincode::serialize(&world.rng).unwrap();
    for _ in 0..4 {
        world.resolve_overlaps();
    }
    let parent = &world.cells[id];
    let child = &world.cells[child_id];
    assert!(
        toroidal_distance_squared(parent.position, child.position, 16.0, 12.0).sqrt()
            >= parent.radius + child.radius - GEOMETRY_TOLERANCE
    );
    assert_eq!(bincode::serialize(&world.rng).unwrap(), rng_before);
    world.check_invariants().unwrap();
}

#[test]
fn transport_equilibrium_preserves_body_size_for_mixed_areas() {
    for scale in [0.25, 1.0, 4.0, 16.0] {
        let mut world = World::new(only_a_config("body-transport", 16, 12)).unwrap();
        let genome = Genome::random_founder(&mut world.rng, 0.0);
        let position = Position::new(3.5, 4.5);
        let id = world
            .spawn_cell_with_genome_at_position(position, genome)
            .unwrap();
        for element in ELEMENT_ORDER {
            world.cells[id].body[element] *= scale;
        }
        world.cells[id].refresh_radius(world.config.cell_body.body_density);
        let radius = world.cells[id].radius;
        let body = world.cells[id].body;
        let boundary = world.cells[id].boundary;
        let total_before = world.stats().system_element_amounts;
        let rng_before = bincode::serialize(&world.rng).unwrap();
        for _ in 0..2_000 {
            world.transport_elements(world.cells.slot(id).unwrap(), position);
        }
        let inside =
            f64::from(world.cells[id].internal_elements[Element::A]) / cell_area(&world, id);
        let outside = f64::from(world.sample_element_fields_at(position).unwrap()[Element::A]);
        assert!(
            (inside - outside).abs() < 1.0e-5,
            "scale={scale} inside={inside} outside={outside}"
        );
        assert_eq!(world.cells[id].radius, radius);
        assert_eq!(world.cells[id].body, body);
        assert_eq!(world.cells[id].boundary, boundary);
        assert_eq!(bincode::serialize(&world.rng).unwrap(), rng_before);
        for element in ELEMENT_ORDER {
            assert!(
                (world.stats().system_element_amounts[element.index()]
                    - total_before[element.index()])
                .abs()
                    < 1.0e-4
            );
        }
        world.check_invariants().unwrap();
    }
}

#[test]
fn membrane_polarity_controls_element_permeability() {
    let mut world = World::new(small_config("polarity-transport", 8, 8)).unwrap();
    let id = world.spawn_random_cell().unwrap().unwrap();
    let position = world.cells[id].position;
    world.cells[id].boundary = ElementAmounts::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    for amounts in &mut world.element_fields {
        *amounts = ElementAmounts::new([1.0; 6]);
    }
    assert!((world.cells[id].membrane_polarity() - 0.9).abs() < 1.0e-6);
    world.transport_elements(world.cells.slot(id).unwrap(), position);
    let free = world.cells[id].internal_elements;
    assert!(free[Element::A] > free[Element::B]);
    assert!(free[Element::B] > free[Element::C]);
    assert!(free[Element::C] > free[Element::D]);
    world.cells[id].boundary = ElementAmounts::ZERO;
    assert_eq!(world.cells[id].membrane_polarity(), 0.5);
}

#[test]
fn fission_conserves_pools_and_energy_places_touching_daughters_and_repeats_per_seed() {
    for seed in ["fission-1", "fission-2", "fission-3"] {
        let mut world = World::new(small_config(seed, 8, 8)).unwrap();
        let genome = Genome::random_founder(&mut world.rng, 0.0);
        let id = world
            .spawn_cell_with_genome_at_position(Position::new(0.05, 0.05), genome)
            .unwrap();
        for element in ELEMENT_ORDER {
            world.cells[id].boundary[element] *= 1.6;
        }
        world.cells[id].internal_elements = ElementAmounts::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let before = world.cells[id].clone();
        let field_before = world.element_field_totals();
        let mut repeated = world.clone();
        world.fission_cell(id);
        repeated.fission_cell(id);
        assert_eq!(
            bincode::serialize(&world).unwrap(),
            bincode::serialize(&repeated).unwrap()
        );
        assert_eq!(world.cell_count(), 2);
        assert_eq!(world.reaction_counters.divisions, 1);
        assert_eq!(world.element_field_totals(), field_before);
        let child = world.cells.iter().find(|cell| cell.id != id).unwrap();
        let parent = &world.cells[id];
        assert_eq!(parent.energy + child.energy, before.energy);
        assert_eq!(child.lineage_id, before.lineage_id);
        let distance = toroidal_distance_squared(parent.position, child.position, 8.0, 8.0).sqrt();
        assert!((distance - parent.radius - child.radius).abs() < 1.0e-5);
        for element in ELEMENT_ORDER {
            for (original, left, right) in [
                (
                    before.body[element],
                    parent.body[element],
                    child.body[element],
                ),
                (
                    before.boundary[element],
                    parent.boundary[element],
                    child.boundary[element],
                ),
                (
                    before.internal_elements[element],
                    parent.internal_elements[element],
                    child.internal_elements[element],
                ),
            ] {
                assert!((left + right - original).abs() < 1.0e-6);
            }
        }
        world.check_invariants().unwrap();
    }
}

#[test]
fn lysis_releases_all_material_and_records_remaining_energy() {
    let mut config = only_a_config("lysis", 8, 8);
    config.catalyst_upkeep_per_sec = 0.0;
    let mut world = World::new(config).unwrap();
    let mut genome = Genome::random_founder(&mut world.rng, 0.0);
    make_inert(&mut genome);
    genome.maintenance_cost_per_area = 0.0;
    let id = world
        .spawn_cell_with_genome_at_position(Position::new(2.5, 3.5), genome)
        .unwrap();
    for element in ELEMENT_ORDER {
        world.cells[id].boundary[element] *= 0.6;
    }
    world.cells[id].internal_elements = ElementAmounts::new([1.0; 6]);
    let before = world.stats().system_element_amounts;
    let energy = world.cells[id].energy;
    world.step_cell_fresh(id);
    assert_eq!(world.cell_count(), 0);
    assert_eq!(world.lysis_count, 1);
    assert_eq!(world.energy_death_count, 0);
    assert_eq!(world.energy_ledger.death_loss, energy);
    for element in ELEMENT_ORDER {
        assert!(
            (world.element_field_totals()[element.index()] - before[element.index()]).abs()
                < 1.0e-5
        );
    }
    world.check_invariants().unwrap();
}

#[test]
fn fission_requires_membrane_and_minimum_daughter_material() {
    let mut world = World::new(small_config("fission-minimum", 8, 8)).unwrap();
    let id = world.spawn_random_cell().unwrap().unwrap();
    world.set_all_live_cell_energy(1_000_000.0).unwrap();
    world.fission_cell(id);
    assert_eq!(world.cell_count(), 1);
    for element in ELEMENT_ORDER {
        world.cells[id].body[element] *= 0.01;
    }
    world.cells[id].refresh_radius(world.config.cell_body.body_density);
    world.fission_cell(id);
    assert_eq!(world.cell_count(), 1);
    world.check_invariants().unwrap();
}

#[test]
fn assembly_is_ledgered_and_closes_over_two_thousand_ticks() {
    let mut world = World::new(small_config("assembly-ledger", 16, 12)).unwrap();
    world.spawn_founder_cells(8).unwrap();
    for cell in world.cells.iter_mut() {
        for enzyme in &mut cell.genome.enzymes {
            enzyme.body_fraction = 0.2;
            enzyme.boundary_fraction = 0.2;
        }
    }
    for _ in 0..2_000 {
        world.step();
        world.check_invariants().unwrap();
    }
    assert!(world.energy_ledger.assembly_cost > 0.0);
}

#[test]
fn founders_conserve_every_element_and_start_at_reference_geometry() {
    let mut world = World::new(small_config("body-founders", 64, 48)).unwrap();
    let before = world.element_field_totals();
    assert_eq!(world.spawn_founder_cells(32).unwrap(), 32);
    let after = world.stats().system_element_amounts;
    for element in ELEMENT_ORDER {
        assert!((before[element.index()] - after[element.index()]).abs() < 1.0e-4);
    }
    for cell in world.cells.iter() {
        assert!((cell.radius - REFERENCE_CELL_RADIUS).abs() < 1.0e-4);
        assert!((cell.strain(world.config.cell_body.boundary_density) - 1.0).abs() < 1.0e-4);
        assert_eq!(cell.energy, 2.0);
    }
    world.check_invariants().unwrap();
    let ids = live_ids(&world);
    for id in ids {
        world.kill_cell_and_release(id);
    }
    for element in ELEMENT_ORDER {
        assert!(
            (before[element.index()] - world.element_field_totals()[element.index()]).abs()
                < 1.0e-4
        );
    }
    world.check_invariants().unwrap();
}

#[test]
fn founders_reject_insufficient_material_without_changing_fields() {
    let mut config = only_a_config("poor-founder", 1, 1);
    config.element_fields.initial_amounts = ElementAmounts::new([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let mut world = World::new(config).unwrap();
    let before = world.element_field_totals();
    assert_eq!(world.spawn_founder_cells(3).unwrap(), 0);
    assert_eq!(world.element_field_totals(), before);
    assert_eq!(world.energy_ledger.founder_energy, 0.0);
    world.check_invariants().unwrap();
}

#[test]
fn invariants_reject_invalid_structure_and_inconsistent_radius() {
    let mut world = World::new(small_config("invalid-body", 8, 8)).unwrap();
    let id = world.spawn_random_cell().unwrap().unwrap();
    let cell = world.cells[id].clone();
    world.cells[id].radius *= 2.0;
    assert!(world.check_invariants().is_err());
    world.cells[id] = cell.clone();
    world.cells[id].body = ElementAmounts::ZERO;
    assert!(world.check_invariants().is_err());
    world.cells[id] = cell.clone();
    world.cells[id].boundary[Element::A] = -1.0;
    assert!(world.check_invariants().is_err());
    world.cells[id] = cell;
    world.cells[id].body[Element::A] = f32::NAN;
    assert!(world.check_invariants().is_err());
}

#[test]
fn world_has_width_times_height_tiles() {
    let world = World::new(small_config("tiles", 7, 5)).unwrap();
    assert_eq!(world.width(), 7);
    assert_eq!(world.height(), 5);
    assert_eq!(world.tile_count(), 35);
}

#[test]
fn toroidal_tile_addressing_wraps_edges() {
    let world = World::new(only_a_config("wrapped-tiles", 3, 2)).unwrap();
    assert_eq!(world.wrapped_tile_id(-1, 0), world.tile_id(2, 0).unwrap());
    assert_eq!(world.wrapped_tile_id(1, 0), world.tile_id(1, 0).unwrap());
    assert_eq!(world.wrapped_tile_id(0, -1), world.tile_id(0, 1).unwrap());
    assert_eq!(world.wrapped_tile_id(0, 1), world.tile_id(0, 1).unwrap());
    assert_eq!(world.wrapped_tile_id(-1, -1), world.tile_id(2, 1).unwrap());
    assert_eq!(world.wrapped_tile_id(1, 1), world.tile_id(1, 1).unwrap());
}

#[test]
fn continuous_element_fields_use_defaults_and_toroidal_tile_layout() {
    let world = World::new(only_a_config("element-field-defaults", 3, 2)).unwrap();
    for x in 0..world.width() {
        for y in 0..world.height() {
            let tile = world.tile_id(x, y).unwrap();
            assert_eq!(world.tile_xy(tile), Some((x, y)));
            assert_eq!(
                world.tile_element_amounts(tile).unwrap(),
                world.config.element_fields.initial_amounts
            );
        }
    }
    assert_eq!(world.wrapped_tile_id(-1, -1), world.tile_id(2, 1).unwrap());
}

#[test]
fn continuous_element_field_diffusion_is_per_element_conservative_and_nonnegative() {
    let mut world = World::new(only_a_config("element-field-diffusion", 3, 3)).unwrap();
    for tile_index in 0..world.tile_count() {
        world
            .set_tile_element_amounts(TileId(tile_index), ElementAmounts::ZERO)
            .unwrap();
    }
    let source = world.tile_id(1, 1).unwrap();
    world
        .set_tile_element_amounts(source, ElementAmounts::new([9.0; 6]))
        .unwrap();
    let before = world.element_field_totals();

    world.diffuse_element_fields();

    let after = world.element_field_totals();
    let source_amounts = world.tile_element_amounts(source).unwrap();
    let neighbor_amounts = world
        .tile_element_amounts(world.tile_id(0, 0).unwrap())
        .unwrap();
    for element in ELEMENT_ORDER {
        let alpha =
            world.config.element_fields.diffusivities[element] * world.config.dt_seconds as f32;
        assert!((source_amounts[element] - (9.0 - 8.0 * alpha)).abs() <= 1.0e-6);
        assert!((neighbor_amounts[element] - alpha).abs() <= 1.0e-6);
        assert!((after[element.index()] - before[element.index()]).abs() <= 1.0e-5);
    }
    for tile_index in 0..world.tile_count() {
        let amounts = world.tile_element_amounts(TileId(tile_index)).unwrap();
        for element in ELEMENT_ORDER {
            assert!(amounts[element].is_finite());
            assert!(amounts[element] >= 0.0);
        }
    }
    world.check_invariants().unwrap();
}

#[test]
fn continuous_element_field_diffusion_is_deterministic() {
    let mut first = World::new(only_a_config("element-field-determinism", 5, 4)).unwrap();
    let source = first.tile_id(4, 3).unwrap();
    first
        .set_tile_element_amounts(source, ElementAmounts::new([8.0, 0.0, 4.0, 2.0, 1.0, 0.5]))
        .unwrap();
    let mut second = first.clone();

    for _ in 0..20 {
        first.diffuse_element_fields();
        second.diffuse_element_fields();
    }

    assert_eq!(first.element_fields, second.element_fields);
    assert_eq!(first.element_field_totals(), second.element_field_totals());
}

#[test]
fn zero_heterogeneity_reproduces_the_uniform_fields_bit_for_bit() {
    let mut config = small_config("uniform-fields", 40, 30);
    config.element_fields.heterogeneity = 0.0;
    let world = World::new(config.clone()).unwrap();
    let uniform = vec![config.element_fields.initial_amounts; world.tile_count()];
    assert_eq!(world.element_fields, uniform);
    let mut expected_rng = crate::rng::Rng::from_seed_str("uniform-fields");
    let expected_landscape =
        crate::environment::enval_landscape(&config.enval_landscape, 40, 30, &mut expected_rng);
    assert_eq!(world.enval_landscape(), expected_landscape.as_slice());
}

#[test]
fn default_heterogeneous_fields_keep_totals_nonnegative_periodic_and_seeded() {
    let config = small_config("heterogeneous-fields", 96, 72);
    let world = World::new(config.clone()).unwrap();
    let totals = world.element_field_totals();
    for element in ELEMENT_ORDER {
        let expected =
            f64::from(config.element_fields.initial_amounts[element]) * world.tile_count() as f64;
        assert!(
            (totals[element.index()] - expected).abs() <= 1.0e-5 * expected,
            "{element}: {} vs {expected}",
            totals[element.index()]
        );
        let at = |x: usize, y: usize| world.element_fields[world.index_xy(x, y)][element];
        let mut interior = 0.0_f32;
        for x in 0..world.width - 1 {
            for y in 0..world.height - 1 {
                interior = interior
                    .max((at(x + 1, y) - at(x, y)).abs())
                    .max((at(x, y + 1) - at(x, y)).abs());
            }
        }
        let mut seam = 0.0_f32;
        for y in 0..world.height {
            seam = seam.max((at(0, y) - at(world.width - 1, y)).abs());
        }
        for x in 0..world.width {
            seam = seam.max((at(x, 0) - at(x, world.height - 1)).abs());
        }
        assert!(
            seam <= interior,
            "{element}: seam {seam} > interior {interior}"
        );
    }
    assert!(world.element_fields.iter().all(|amounts| {
        ELEMENT_ORDER
            .iter()
            .all(|element| amounts[*element].is_finite() && amounts[*element] >= 0.0)
    }));
    let d_values = world
        .element_fields
        .iter()
        .map(|amounts| amounts[Element::D])
        .collect::<Vec<_>>();
    let d_max = d_values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let d_min = d_values.iter().copied().fold(f32::INFINITY, f32::min);
    assert!(d_max > 1.2 * d_min, "fields are not heterogeneous");

    let again = World::new(config).unwrap();
    assert_eq!(world.element_fields, again.element_fields);
    let other = World::new(small_config("heterogeneous-fields-other", 96, 72)).unwrap();
    assert_ne!(world.element_fields, other.element_fields);
    world.check_invariants().unwrap();
}

#[test]
fn seeded_initialization_is_deterministic() {
    let a = World::new(small_config("deterministic", 10, 8))
        .unwrap()
        .stats();
    let b = World::new(small_config("deterministic", 10, 8))
        .unwrap()
        .stats();
    assert_eq!(
        a.extracellular_element_amounts,
        b.extracellular_element_amounts
    );
    assert_eq!(
        a.intracellular_element_amounts,
        b.intracellular_element_amounts
    );
    assert_eq!(a.system_element_amounts, b.system_element_amounts);
    assert_eq!(a.average_enval.to_bits(), b.average_enval.to_bits());
}

#[test]
fn initial_enval_equals_the_landscape_targets_on_every_tile() {
    let world = World::new(small_config("enval-initial", 64, 48)).unwrap();
    assert!(world.average_enval().is_finite());
    assert_eq!(world.enval_landscape().len(), world.tile_count());
    for tile_index in 0..world.tile_count() {
        let tile = TileId(tile_index);
        let target = world.enval_landscape()[tile_index];
        assert_eq!(world.tile_enval(tile).unwrap().to_bits(), target.to_bits());
        assert_eq!(
            world
                .inspect_tile(tile)
                .unwrap()
                .enval_landscape_target
                .to_bits(),
            target.to_bits()
        );
    }
    let total = world
        .enval
        .iter()
        .map(|value| f64::from(*value))
        .sum::<f64>();
    assert_eq!(world.enval_ledger().initial_total, total);
}

#[test]
fn uniform_enval_field_remains_unchanged_after_diffusion() {
    let mut world = World::new(only_a_config("uniform", 4, 4)).unwrap();
    world.set_all_enval(0.25).unwrap();
    world.diffuse_enval();
    for tile_index in 0..world.tile_count() {
        assert_eq!(
            world.tile_enval(TileId(tile_index)).unwrap().to_bits(),
            0.25_f32.to_bits()
        );
    }
}

#[test]
fn nonuniform_enval_diffuses_and_local_average_wraps() {
    let mut world = World::new(only_a_config("nonuniform", 3, 3)).unwrap();
    world.set_all_enval(0.0).unwrap();
    let corner = world.tile_id(2, 2).unwrap();
    let origin_id = world.tile_id(0, 0).unwrap();
    world.set_tile_enval(corner, 9.0).unwrap();
    let average = tile_local_enval(&world, origin_id);
    assert!((average - 9.0 * 4.0 / 25.0).abs() < 1.0e-6);
    world.diffuse_enval();
    let origin = world.tile_enval(origin_id).unwrap();
    let source = world.tile_enval(corner).unwrap();
    assert!(origin > 0.0);
    assert!(source < 9.0);
    world.check_invariants().unwrap();
}

#[test]
fn enval_values_remain_finite_after_repeated_diffusion() {
    let mut world = World::new(small_config("finite", 8, 6)).unwrap();
    let low = world.tile_id(0, 0).unwrap();
    let high = world.tile_id(7, 5).unwrap();
    world.set_tile_enval(low, 1.0).unwrap();
    world.set_tile_enval(high, -1.0).unwrap();
    for _ in 0..200 {
        world.diffuse_enval();
    }
    for tile_index in 0..world.tile_count() {
        assert!(world.tile_enval(TileId(tile_index)).unwrap().is_finite());
    }
}

#[test]
fn cell_spawn_updates_stats() {
    let mut world = World::new(only_a_config("spawn", 8, 8)).unwrap();
    let spawned = world.spawn_founder_cells(4).unwrap();
    assert_eq!(spawned, 4);
    assert_eq!(world.stats().live_cell_count, 4);
    assert_eq!(world.stats().births, 4);
    world.check_invariants().unwrap();
}

#[test]
fn overlapping_cell_position_is_rejected() {
    let mut world = World::new(only_a_config("occupancy", 4, 4)).unwrap();
    let tile = world.tile_id(0, 0).unwrap();
    let genome = Genome::random_founder(&mut world.rng, world.avg_enval);
    world
        .spawn_cell_with_genome_at(tile, genome.clone())
        .unwrap();
    assert!(world.spawn_cell_with_genome_at(tile, genome).is_err());
}

#[test]
fn continuous_uptake_is_deterministic_and_conservative() {
    let mut world = World::new(only_a_config("uptake", 4, 4)).unwrap();
    let tile = world.tile_id(1, 1).unwrap();
    let genome = Genome::random_founder(&mut world.rng, world.avg_enval);
    let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
    let before_field = world.tile_element_amounts(tile).unwrap().total();
    world.step_cell_fresh(cell_id);
    let internal = world.cells[cell_id].internal_elements.total();
    let after_field = world.tile_element_amounts(tile).unwrap().total();
    assert!(internal > 0.0);
    assert!((before_field - (after_field + internal)).abs() <= 1.0e-5);
    assert_valid_reservoir(&world, cell_id);
    world.check_invariants().unwrap();
}

fn transport_fraction(world: &World, cell_id: crate::cell::CellId) -> f64 {
    let radius = f64::from(world.cells[cell_id].radius);
    let area = std::f64::consts::PI * radius * radius;
    let perimeter = 2.0 * std::f64::consts::PI * radius;
    (f64::from(world.config.membrane_permeability) * perimeter * world.config.dt_seconds
        / (2.0 * area)
        * 1.75)
        .min(MAX_TRANSPORT_FRACTION)
}

fn cell_area(world: &World, cell_id: crate::cell::CellId) -> f64 {
    let radius = f64::from(world.cells[cell_id].radius);
    std::f64::consts::PI * radius * radius
}

#[test]
fn transport_moves_the_gradient_formula_amount_in_one_tick() {
    let mut world = World::new(only_a_config("transport-formula", 4, 4)).unwrap();
    let tile = world.tile_id(1, 1).unwrap();
    let genome = Genome::random_founder(&mut world.rng, world.avg_enval);
    let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
    world.cells[cell_id].internal_elements[Element::A] = 0.1;
    let area = cell_area(&world, cell_id);
    let expected = transport_fraction(&world, cell_id)
        * (f64::from(world.tile_element_amounts(tile).unwrap()[Element::A]) - 0.1 / area)
        * area;

    let (inward, outward) =
        world.transport_elements(world.cells.slot(cell_id).unwrap(), Position::new(1.0, 1.0));

    assert!(
        (inward - expected).abs() <= 1.0e-6,
        "{inward} != {expected}"
    );
    assert_eq!(outward, 0.0);
    assert!(
        (f64::from(world.cells[cell_id].internal_elements[Element::A]) - (0.1 + expected)).abs()
            <= 1.0e-6
    );
}

#[test]
fn transport_equilibrates_in_a_uniform_field_without_overshoot() {
    let mut world = World::new(only_a_config("transport-equilibrium", 6, 6)).unwrap();
    let genome = Genome::random_founder(&mut world.rng, world.avg_enval);
    let position = Position::new(2.0, 3.0);
    let cell_id = world
        .spawn_cell_with_genome_at_position(position, genome)
        .unwrap();
    let area = cell_area(&world, cell_id);
    let mut previous_inside = 0.0_f64;
    for _ in 0..400 {
        world.transport_elements(world.cells.slot(cell_id).unwrap(), position);
        let inside = f64::from(world.cells[cell_id].internal_elements[Element::A]) / area;
        let outside = f64::from(world.sample_element_fields_at(position).unwrap()[Element::A]);
        assert!(inside >= previous_inside - 1.0e-9, "concentration fell");
        assert!(
            inside <= outside + 1.0e-6,
            "overshoot: {inside} > {outside}"
        );
        previous_inside = inside;
    }
    let outside = f64::from(world.sample_element_fields_at(position).unwrap()[Element::A]);
    assert!((previous_inside - outside).abs() <= 1.0e-4 * outside);
    for element in [Element::B, Element::C, Element::D, Element::E, Element::F] {
        assert_eq!(world.cells[cell_id].internal_elements[element], 0.0);
    }
}

#[test]
fn transport_leaks_outward_from_a_richer_cell_without_overshoot() {
    let mut world = World::new(only_a_config("transport-leak", 6, 6)).unwrap();
    let genome = Genome::random_founder(&mut world.rng, world.avg_enval);
    let position = Position::new(2.5, 2.5);
    let cell_id = world
        .spawn_cell_with_genome_at_position(position, genome)
        .unwrap();
    world.cells[cell_id].internal_elements = ElementAmounts::new([5.0, 3.0, 0.0, 0.0, 0.0, 0.0]);
    let area = cell_area(&world, cell_id);
    let field_before = world.element_field_totals();

    let (inward, outward) = world.transport_elements(world.cells.slot(cell_id).unwrap(), position);

    let cell = &world.cells[cell_id];
    assert_eq!(inward, 0.0);
    assert!(outward > 0.0);
    assert!(cell.internal_elements[Element::A] < 5.0);
    assert!(cell.internal_elements[Element::B] < 3.0);
    let field_after = world.element_field_totals();
    for element in [Element::A, Element::B] {
        let inside = f64::from(cell.internal_elements[element]) / area;
        let outside = f64::from(world.sample_element_fields_at(position).unwrap()[element]);
        assert!(inside >= outside, "leak overshot equilibrium");
        assert!(field_after[element.index()] > field_before[element.index()]);
    }
    world.check_invariants().unwrap();
}

#[test]
fn transport_amount_is_the_same_on_a_tile_centre_and_across_a_torus_seam() {
    let moved_at = |position: Position| {
        let mut world = World::new(only_a_config("transport-seam", 8, 6)).unwrap();
        let genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        let cell_id = world
            .spawn_cell_with_genome_at_position(position, genome)
            .unwrap();
        for amounts in &mut world.element_fields {
            *amounts = world.config.element_fields.initial_amounts;
        }
        let before = world.element_field_totals();
        let (inward, _) = world.transport_elements(world.cells.slot(cell_id).unwrap(), position);
        let after = world.element_field_totals();
        assert!((before[Element::A.index()] - after[Element::A.index()] - inward).abs() <= 1.0e-5);
        inward
    };
    let centre = moved_at(Position::new(3.0, 2.0));
    let seam = moved_at(Position::new(7.5, 5.5));
    let corner_seam = moved_at(Position::new(7.9, 0.2));
    assert!(centre > 0.0);
    assert!((centre - seam).abs() <= 1.0e-6, "{centre} != {seam}");
    assert!(
        (centre - corner_seam).abs() <= 1.0e-6,
        "{centre} != {corner_seam}"
    );
}

#[test]
fn transport_consumes_no_rng() {
    let mut world = World::new(small_config("transport-rng", 6, 6)).unwrap();
    let genome = Genome::random_founder(&mut world.rng, world.avg_enval);
    let position = Position::new(1.3, 4.7);
    let cell_id = world
        .spawn_cell_with_genome_at_position(position, genome)
        .unwrap();
    world.cells[cell_id].internal_elements = ElementAmounts::new([0.0, 4.0, 0.0, 0.2, 0.0, 1.0]);
    let mut untouched = world.rng.clone();
    for _ in 0..10 {
        world.transport_elements(world.cells.slot(cell_id).unwrap(), position);
    }
    assert_eq!(
        world.rng.next_f64().to_bits(),
        untouched.next_f64().to_bits()
    );
}

#[test]
fn transport_conserves_total_elements_over_many_ticks_with_many_cells() {
    let mut world = World::new(small_config("transport-conservation", 32, 24)).unwrap();
    world.spawn_founder_cells(40).unwrap();
    let initial = world.stats().total_element_amount;
    for tick in 1..=1_000 {
        world.step();
        if tick % 100 == 0 {
            let total = world.compact_stats().total_element_amount;
            assert!(
                (total - initial).abs() <= 2.0e-5 * initial,
                "tick {tick}: {total} vs {initial}"
            );
        }
    }
    assert!(world.stats().reaction_counters.uptake_flux > 0.0);
    assert!(world.stats().reaction_counters.leak_flux > 0.0);
    world.check_invariants().unwrap();
}

#[test]
fn fractional_field_sampling_deposition_and_seam_wrapping_are_conservative() {
    let mut world = World::new(only_a_config("fractional-field-coupling", 4, 3)).unwrap();
    for amounts in &mut world.element_fields {
        *amounts = ElementAmounts::ZERO;
    }
    for (x, y, amount) in [(3, 2, 1.0), (0, 2, 2.0), (3, 0, 3.0), (0, 0, 4.0)] {
        let tile = world.tile_id(x, y).unwrap();
        world.element_fields[tile.index()][Element::A] = amount;
    }
    let position = Position::new(3.5, 2.5);
    let sampled = world.sample_element_fields_at(position).unwrap();
    assert!((sampled[Element::A] - 2.5).abs() <= 1.0e-6);

    let before = world.element_field_totals()[Element::A.index()];
    world
        .deposit_elements_at(
            position,
            ElementAmounts::new([8.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
        .unwrap();
    let after = world.element_field_totals()[Element::A.index()];
    assert!((after - before - 8.0).abs() <= 1.0e-6);
    for (x, y, initial) in [(3, 2, 1.0), (0, 2, 2.0), (3, 0, 3.0), (0, 0, 4.0)] {
        let tile = world.tile_id(x, y).unwrap();
        assert!((world.element_fields[tile.index()][Element::A] - (initial + 2.0)).abs() <= 1.0e-6);
    }
}

#[test]
fn invalid_environmental_deposition_is_explicit_and_non_mutating() {
    let mut world = World::new(only_a_config("invalid-deposition", 4, 3)).unwrap();
    let before = world.element_fields.clone();
    let invalid_position = Position::new(f32::NAN, 1.0);

    let result = world.deposit_elements_at(
        invalid_position,
        ElementAmounts::new([8.0, 7.0, 6.0, 5.0, 4.0, 3.0]),
    );

    assert!(matches!(
        result,
        Err(super::WorldError::InvalidCellPosition(position)) if position.x.is_nan()
    ));
    assert_eq!(world.element_fields, before);
}

#[test]
fn fractional_uptake_is_nonnegative_and_conservative() {
    let mut world = World::new(only_a_config("fractional-uptake", 4, 4)).unwrap();
    let tile = world.tile_id(1, 1).unwrap();
    let genome = Genome::random_founder(&mut world.rng, world.avg_enval);
    let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
    for amounts in &mut world.element_fields {
        *amounts = ElementAmounts::ZERO;
    }
    for (x, y, amount) in [(1, 1, 0.1), (2, 1, 0.2), (1, 2, 0.3), (2, 2, 0.4)] {
        let tile = world.tile_id(x, y).unwrap();
        world.element_fields[tile.index()][Element::A] = amount;
    }
    let position = Position::new(1.5, 1.5);
    let before = world.element_field_totals()[Element::A.index()];

    let (transferred, _) = world.transport_elements(world.cells.slot(cell_id).unwrap(), position);

    let after = world.element_field_totals()[Element::A.index()];
    assert!(transferred > 0.0);
    assert!((before - after - transferred).abs() <= 1.0e-6);
    assert!(world.element_fields.iter().all(|amounts| {
        ELEMENT_ORDER
            .iter()
            .all(|element| amounts[*element].is_finite() && amounts[*element] >= 0.0)
    }));
}

#[test]
fn fractional_enval_sampling_average_and_scatter_preserve_locality() {
    let mut world = World::new(only_a_config("fractional-enval", 6, 6)).unwrap();
    world.set_all_enval(0.0).unwrap();
    world
        .set_tile_enval(world.tile_id(1, 1).unwrap(), 4.0)
        .unwrap();
    assert!((world.sample_enval_at(Position::new(1.5, 1.5)).unwrap() - 1.0).abs() <= 1.0e-6);

    let position = Position::new(1.5, 1.5);
    let manual = (-2..=2)
        .flat_map(|dx| (-2..=2).map(move |dy| (dx, dy)))
        .map(|(dx, dy)| {
            world
                .sample_enval_at(Position::new(
                    position.x + dx as f32,
                    position.y + dy as f32,
                ))
                .unwrap()
        })
        .sum::<f32>()
        / 25.0;
    assert!((world.local_enval_at(position).unwrap() - manual).abs() <= 1.0e-6);

    let before_sum = world.enval_sum;
    world.adjust_enval_at(position, 2.0).unwrap();
    assert!((world.enval_sum - before_sum - 2.0).abs() <= 1.0e-6);
    for (x, y) in [(1, 1), (2, 1), (1, 2), (2, 2)] {
        assert!(world.tile_enval(world.tile_id(x, y).unwrap()).unwrap() >= 0.5);
    }
}

#[test]
fn smoothed_enval_grid_sampled_bilinearly_equals_a_brute_force_twenty_five_sample_average() {
    for (seed, width, height, noise) in [
        ("smoothed-enval-default", 320, 240, 0.5),
        ("smoothed-enval-small", 7, 6, 2.0),
        ("smoothed-enval-narrow", 3, 4, 2.0),
        ("smoothed-enval-line", 1, 9, 2.0),
    ] {
        let mut world = World::new(small_config(seed, width, height)).unwrap();
        for tile in 0..world.tile_count() {
            let value = world.enval_landscape[tile] + world.rng.range(-noise, noise);
            world.set_tile_enval(TileId(tile), value).unwrap();
        }
        world.refresh_enval_smoothed();
        assert!(world.enval_smoothed_current);
        let brute_force = |position: Position| {
            let mut sum = 0.0_f64;
            for dx in -2..=2 {
                for dy in -2..=2 {
                    let sample = Position::new(position.x + dx as f32, position.y + dy as f32);
                    sum += f64::from(world.sample_enval_at(sample).unwrap());
                }
            }
            (sum / 25.0) as f32
        };
        let mut positions = (0..2_000)
            .map(|index| {
                Position::new(
                    (index as f32 * 0.618_034).fract() * width as f32,
                    (index as f32 * 0.754_877).fract() * height as f32,
                )
            })
            .collect::<Vec<_>>();
        for x in 0..width {
            for y in 0..height {
                positions.push(Position::new(x as f32, y as f32));
            }
        }
        positions.push(Position::new(width as f32 - 1.0e-4, height as f32 - 1.0e-4));
        for position in positions {
            let stencil = world.tile_stencil(position).unwrap();
            let smoothed = sample_smoothed_enval(&world.enval_smoothed, &stencil);
            let reference = brute_force(position);
            assert!(
                (smoothed - reference).abs() <= 1.0e-6,
                "{seed} at {position:?}: {smoothed} vs {reference}"
            );
        }
        let position = Position::new(0.25 * width as f32, 0.75 * height as f32);
        let stencil = world.tile_stencil(position).unwrap();
        assert_eq!(
            world.local_enval_at(position),
            Some(sample_smoothed_enval(&world.enval_smoothed, &stencil))
        );
    }
}

#[test]
fn tile_stencils_match_bilinear_stencils_bit_for_bit() {
    for (width, height) in [(320, 240), (7, 5), (1, 3)] {
        let mut world = World::new(small_config("tile-stencil", width, height)).unwrap();
        let (w, h) = (width as f32, height as f32);
        let mut positions = vec![
            Position::new(0.0, 0.0),
            Position::new(w - 1.0e-4, h - 1.0e-4),
            Position::new(-1.0e-7, -3.5),
            Position::new(w, h),
            Position::new(2.0 * w + 0.25, -h - 0.75),
        ];
        for _ in 0..5_000 {
            positions.push(Position::new(
                world.rng.range(-2.0 * w, 3.0 * w),
                world.rng.range(-2.0 * h, 3.0 * h),
            ));
        }
        for position in positions {
            let expected = BilinearStencil::new(position, width, height).unwrap();
            let actual = world.tile_stencil(position).unwrap();
            for corner in 0..4 {
                let sample = expected.samples[corner];
                assert_eq!(actual.tiles[corner], world.index_xy(sample.x, sample.y));
                assert_eq!(actual.weights[corner].to_bits(), sample.weight.to_bits());
            }
        }
        assert!(world.tile_stencil(Position::new(f32::NAN, 0.0)).is_none());
    }
}

#[test]
fn cells_read_the_smoothed_enval_of_the_start_of_the_cell_phase() {
    let mut world = World::new(small_config("smoothed-enval-phase", 24, 18)).unwrap();
    world.spawn_founder_cells(6).unwrap();
    world.step_many(20);
    let before = world.clone();
    world.step_cells();
    assert!(!world.enval_smoothed_current);
    let mut expected = before.clone();
    expected.refresh_enval_smoothed();
    for cell in before.cells.iter() {
        let reference = before.local_enval_at(cell.position).unwrap();
        let smoothed = expected.local_enval_at(cell.position).unwrap();
        assert!((reference - smoothed).abs() <= 1.0e-6);
    }
    let last_cell = before.cells.at_slot(before.cells.len() - 1);
    let mut stepped = before.clone();
    stepped.inspect_cell_fluxes(last_cell.id, 64).unwrap();
    stepped.step_cells();
    assert!(stepped.enval != before.enval);
    let log = stepped.inspect_cell_fluxes(last_cell.id, 64).unwrap();
    let reference = expected.local_enval_at(last_cell.position).unwrap();
    assert!(!log.fluxes.is_empty());
    assert!(log.fluxes.iter().all(|flux| flux.local_enval == reference));
}

#[test]
fn one_cell_steps_emission_lands_on_its_bilinear_stencil_and_draws_no_rng() {
    let mut world = World::new(only_a_config("emission-stencil", 9, 9)).unwrap();
    world.set_all_enval(0.5).unwrap();
    let position = Position::new(3.25, 5.75);
    let mut genome = Genome::random_founder(&mut world.rng, 0.5);
    let mut catalyst = Enzyme::founder_downhill();
    catalyst.rate = 10.0;
    catalyst.enval_sigma = 1000.0;
    catalyst.enval_throughput = 0.0;
    catalyst.enval_pump = 0.5;
    catalyst.secretion_fraction = 0.0;
    genome.enzymes = vec![catalyst];
    set_enval_optimum(&mut genome, 0.5);
    genome.maintenance_cost_per_area = 0.0;
    let cell_id = world
        .spawn_cell_with_genome_at_position(position, genome)
        .unwrap();
    world.cells[cell_id].internal_elements[Element::D] = 1.0;
    let stencil = world.tile_stencil(position).unwrap();
    let before = world.enval.clone();
    let ledger_before = world.enval_ledger();
    let rng_before = format!("{:?}", world.rng);
    world.inspect_cell_fluxes(cell_id, 0).unwrap();

    world.step_cell_fresh(cell_id);

    assert_eq!(
        format!("{:?}", world.rng),
        rng_before,
        "emission draws no RNG"
    );
    let record = world.flux_log.latest_flux().unwrap();
    assert_eq!(record.enval_input, 0.0);
    let emission = record.enval_output;
    assert!(
        emission < 0.0,
        "the pump must emit for this test to mean anything"
    );
    for (index, before_value) in before.iter().copied().enumerate() {
        let corner = stencil.tiles.iter().position(|tile| *tile == index);
        match corner {
            Some(corner) => assert_eq!(
                world.enval[index],
                before_value + emission * stencil.weights[corner],
                "stencil tile {index}"
            ),
            None => assert_eq!(
                world.enval[index], before_value,
                "tile {index} is off the stencil"
            ),
        }
    }
    let ledger = world.enval_ledger();
    let field_change = world
        .enval
        .iter()
        .zip(&before)
        .map(|(after, before)| f64::from(*after) - f64::from(*before))
        .sum::<f64>();
    assert_eq!(ledger.cell_uptake, ledger_before.cell_uptake);
    assert!((ledger.cell_emission - ledger_before.cell_emission - field_change).abs() <= 1.0e-12);
    assert!(world.enval_ledger_residual().abs() <= 1.0e-5);
}

#[test]
fn maintenance_can_kill_cell_and_release_continuous_elements() {
    let mut world = World::new(only_a_config("maintenance-death", 4, 4)).unwrap();
    let tile = world.tile_id(1, 1).unwrap();
    let mut genome = Genome::random_founder(&mut world.rng, world.avg_enval);
    make_inert(&mut genome);
    world.config.founder_energy = 0.001;
    genome.maintenance_cost_per_area = 10.0;
    let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
    world.cells[cell_id].internal_elements = ElementAmounts::new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
    let field_before = world.tile_element_amounts(tile).unwrap()[Element::B];
    world.step_cell_fresh(cell_id);
    assert!(world.cell(cell_id).is_none());
    assert_valid_reservoir(&world, cell_id);
    assert!(world.tile_element_amounts(tile).unwrap()[Element::B] >= field_before + 1.0);
    world.check_invariants().unwrap();
}

#[test]
fn mismatched_enval_with_positive_energy_is_never_killed_by_time() {
    let mut world = World::new(only_a_config("energy-only-death-time", 6, 6)).unwrap();
    world.set_all_enval(0.9).unwrap();
    let tile = world.tile_id(2, 2).unwrap();
    let mut genome = Genome::random_founder(&mut world.rng, world.avg_enval);
    genome.enzymes = vec![Enzyme::founder_downhill()];
    set_enval_optimum(&mut genome, -0.9);
    genome.maintenance_cost_per_area = 0.0;
    world.config.founder_energy = 1.0;
    let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
    world.cells[cell_id].birth_sim_time = -10_000.0;

    for _ in 0..5_000 {
        world.step_cell_fresh(cell_id);
        world.advance_time();
    }

    let cell = world.cell(cell_id).expect("cell must survive");
    assert!(cell.energy > 0.0);
    world.check_invariants().unwrap();
}

#[test]
fn energy_exhaustion_kills_and_releases_the_reservoir_conservatively() {
    let mut world = World::new(only_a_config("energy-only-death-exhaustion", 4, 4)).unwrap();
    let tile = world.tile_id(1, 1).unwrap();
    let mut genome = Genome::random_founder(&mut world.rng, world.avg_enval);
    make_inert(&mut genome);
    world.config.founder_energy = 0.0105;
    genome.maintenance_cost_per_area = 0.5 / f64::from(crate::spatial::REFERENCE_CELL_AREA);
    let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
    world.cells[cell_id].internal_elements = ElementAmounts::new([0.0, 0.4, 0.3, 0.2, 0.1, 0.05]);
    let total_before = world.stats().total_element_amount;

    world.step_cell_fresh(cell_id);
    world.step_cell_fresh(cell_id);
    assert!(world.cell(cell_id).is_some());
    world.step_cell_fresh(cell_id);

    assert!(world.cell(cell_id).is_none());
    let total_after = world.stats().total_element_amount;
    assert!((total_after - total_before).abs() <= 1.0e-5);
    let ledger = world.energy_ledger();
    assert!((ledger.maintenance - 0.0105).abs() <= 1.0e-12);
    assert_eq!(ledger.death_loss, 0.0);
    world.check_invariants().unwrap();
}

#[test]
fn reaction_costs_that_exhaust_energy_kill_the_cell() {
    let mut world = World::new(only_a_config("energy-only-death-reaction", 4, 4)).unwrap();
    let tile = world.tile_id(1, 1).unwrap();
    let mut genome = Genome::random_founder(&mut world.rng, world.avg_enval);
    let mut uphill = Enzyme::metabolic(
        ElementAmounts::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ElementAmounts::new([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        50.0,
        0.5,
        0.0,
    );
    uphill.enval_throughput = 0.0;
    uphill.enval_pump = 0.0;
    uphill.enval_sigma = 1000.0;
    genome.enzymes = vec![uphill];
    world.config.founder_energy = 0.01;
    genome.maintenance_cost_per_area = 0.0;
    let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
    world.cells[cell_id].internal_elements[Element::A] = 5.0;

    world.step_cell_fresh(cell_id);

    assert!(world.cell(cell_id).is_none());
    assert!(world.energy_ledger().chemical_cost > 0.0);
    world.check_invariants().unwrap();
}

#[test]
fn metabolic_enval_response_uses_radius_two_local_average() {
    let mut world = World::new(only_a_config("flux-local-enval", 5, 5)).unwrap();
    world.set_all_enval(0.0).unwrap();
    let tile = world.tile_id(2, 2).unwrap();
    world.set_tile_enval(tile, 1.0).unwrap();
    let local = tile_local_enval(&world, tile);
    assert!((local - 0.04).abs() <= 1.0e-6);
    let mut genome = Genome::random_founder(&mut world.rng, local);
    let mut catalyst = Enzyme::founder_downhill();
    catalyst.rate = 10.0;
    catalyst.enval_sigma = 0.01;
    catalyst.enval_throughput = 0.0;
    catalyst.enval_pump = 0.0;
    genome.enzymes = vec![catalyst];
    set_enval_optimum(&mut genome, local);
    genome.maintenance_cost_per_area = 0.0;
    let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
    world.cells[cell_id].internal_elements[Element::D] = 1.0;

    world.inspect_cell_fluxes(cell_id, 0).unwrap();
    world.step_cell_fresh(cell_id);

    let record = world.flux_log.latest_flux().unwrap();
    assert!((record.local_enval - local).abs() <= 1.0e-6);
    assert!(record.executed_extent > 0.09);
}

#[test]
fn metabolic_enval_input_and_output_both_stay_on_the_cells_own_stencil() {
    let mut world = World::new(only_a_config("flux-enval-spatial", 5, 5)).unwrap();
    world.set_all_enval(1.0).unwrap();
    let tile = world.tile_id(2, 2).unwrap();
    let mut genome = Genome::random_founder(&mut world.rng, 1.0);
    let mut catalyst = Enzyme::founder_downhill();
    catalyst.rate = 10.0;
    catalyst.enval_sigma = 1000.0;
    catalyst.secretion_fraction = 0.0;
    genome.enzymes = vec![catalyst];
    set_enval_optimum(&mut genome, 1.0);
    genome.maintenance_cost_per_area = 0.0;
    let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
    world.cells[cell_id].internal_elements[Element::D] = 1.0;
    let before = world.enval.clone();
    let mut expected_rng = world.rng.clone();

    world.inspect_cell_fluxes(cell_id, 0).unwrap();
    world.step_cell_fresh(cell_id);

    let record = world.flux_log.latest_flux().unwrap();
    assert!(record.enval_input > 0.0);
    assert!(record.enval_output < 0.0);
    let expected_output_magnitude = record.enval_input.abs()
        * world.cells[cell_id].genome.enzymes[0].enval_release_fraction
        + world.cells[cell_id].genome.enzymes[0].enval_pump * record.executed_extent;
    assert!((record.enval_output.abs() - expected_output_magnitude).abs() <= 1.0e-6);
    for (index, before_value) in before.iter().copied().enumerate() {
        let mut expected = before_value;
        if index == tile.index() {
            expected = expected - record.enval_input + record.enval_output;
        }
        assert!((world.enval[index] - expected).abs() <= 1.0e-6);
    }
    assert_eq!(
        world.rng.next_f64().to_bits(),
        expected_rng.next_f64().to_bits()
    );
}

#[test]
fn a_cell_step_exchanges_its_net_enval_once_and_draws_no_rng() {
    let mut world = World::new(only_a_config("batched-enval-exchange", 7, 7)).unwrap();
    world.set_all_enval(0.8).unwrap();
    let tile = world.tile_id(3, 3).unwrap();
    let mut genome = Genome::random_founder(&mut world.rng, 0.8);
    let mut first = Enzyme::founder_downhill();
    first.rate = 10.0;
    first.enval_sigma = 1000.0;
    first.secretion_fraction = 0.0;
    let mut second = Enzyme::founder_engine_forward();
    second.rate = 10.0;
    second.enval_sigma = 1000.0;
    second.enval_pump = 0.1;
    let mut third = Enzyme::founder_engine_reverse();
    third.rate = 10.0;
    third.enval_sigma = 1000.0;
    genome.enzymes = vec![first, second, third];
    set_enval_optimum(&mut genome, 0.8);
    genome.maintenance_cost_per_area = 0.0;
    let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
    world.cells[cell_id].internal_elements = ElementAmounts::new([0.0, 0.5, 0.5, 0.5, 0.0, 0.0]);
    let before = world.enval.clone();
    let ledger_before = world.enval_ledger();
    let mut expected_rng = world.rng.clone();

    world.inspect_cell_fluxes(cell_id, 0).unwrap();
    world.step_cell_fresh(cell_id);

    let records = world
        .flux_log
        .fluxes_newest_first()
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(records.len(), 3);
    let input = records.iter().map(|record| record.enval_input).sum::<f32>();
    let output = records
        .iter()
        .map(|record| record.enval_output)
        .sum::<f32>();
    assert!(records.iter().all(|record| record.enval_input > 0.0));
    assert!(records.iter().all(|record| record.enval_output < 0.0));
    for (index, before_value) in before.iter().copied().enumerate() {
        let mut expected = before_value;
        if index == tile.index() {
            expected = expected - input + output;
        }
        assert!(
            (world.enval[index] - expected).abs() <= 1.0e-6,
            "tile {index}"
        );
    }
    assert_eq!(
        world.rng.next_f64().to_bits(),
        expected_rng.next_f64().to_bits()
    );
    let ledger = world.enval_ledger();
    assert!((ledger.cell_uptake - ledger_before.cell_uptake + f64::from(input)).abs() <= 1.0e-6);
    assert!(
        (ledger.cell_emission - ledger_before.cell_emission - f64::from(output)).abs() <= 1.0e-6
    );
    assert!(world.enval_ledger_residual().abs() <= 1.0e-5);

    let mut silent = World::new(only_a_config("batched-enval-silent", 7, 7)).unwrap();
    let mut genome = Genome::random_founder(&mut silent.rng, 0.0);
    for enzyme in &mut genome.enzymes {
        enzyme.enval_throughput = 0.0;
        enzyme.enval_pump = 0.0;
    }
    set_enval_optimum(&mut genome, 0.3);
    let cell_id = silent.spawn_cell_with_genome_at(tile, genome).unwrap();
    silent.cells[cell_id].internal_elements = ElementAmounts::new([0.0, 0.5, 0.5, 0.5, 0.0, 0.0]);
    let mut untouched_rng = silent.rng.clone();
    let enval_before = silent.enval.clone();
    silent.step_cell_fresh(cell_id);
    assert!(silent.reaction_counters.successes > 0);
    assert_eq!(silent.enval, enval_before);
    assert_eq!(
        silent.rng.next_f64().to_bits(),
        untouched_rng.next_f64().to_bits()
    );
}

#[test]
fn active_cell_flux_logs_are_available_even_when_empty() {
    let mut world = World::new(only_a_config("reaction-log-empty", 4, 4)).unwrap();
    let tile = world.tile_id(1, 1).unwrap();
    let genome = Genome::random_founder(&mut world.rng, world.avg_enval);
    let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
    let logs = world.inspect_cell_fluxes(cell_id, 8).unwrap();
    assert!(logs.available);
    assert_eq!(logs.reason, "recorded");
    assert_eq!(logs.limit, 8);
    assert!(!logs.truncated);
    assert_eq!(logs.flux_count, 0);
    assert_eq!(logs.returned_count, 0);
    assert_eq!(logs.order, "newest_first");
    assert!(logs.fluxes.is_empty());
}

#[test]
fn successful_continuous_fluxes_are_logged_and_bounded() {
    let mut world = World::new(only_a_config("reaction-log-bounded", 4, 4)).unwrap();
    let tile = world.tile_id(1, 1).unwrap();
    let mut genome = Genome::random_founder(&mut world.rng, world.avg_enval);
    let mut catalyst = Enzyme::founder_downhill();
    catalyst.enval_sigma = 1000.0;
    catalyst.rate = 10.0;
    catalyst.body_fraction = 0.0;
    catalyst.boundary_fraction = 0.0;
    catalyst.secretion_fraction = 0.25;
    catalyst.enval_optimum = world.tile_enval(tile).unwrap();
    genome.enzymes = vec![catalyst];
    world.config.founder_energy = 10.0;
    genome.maintenance_cost_per_area = 0.0;
    let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
    world.cells[cell_id].internal_elements[Element::D] = 100.0;

    let target_fluxes = CELL_FLUX_LOG_CAPACITY + 5;
    for _ in 0..target_fluxes {
        world.inspect_cell_fluxes(cell_id, 0).unwrap();
        world.step_cell_fresh(cell_id);
    }

    assert_eq!(world.flux_log.flux_count(), CELL_FLUX_LOG_CAPACITY);
    let logs = world.inspect_cell_fluxes(cell_id, 2).unwrap();
    assert!(logs.available);
    assert_eq!(logs.reason, "recorded");
    assert_eq!(logs.limit, 2);
    assert_eq!(logs.flux_count, CELL_FLUX_LOG_CAPACITY);
    assert_eq!(logs.returned_count, 2);
    assert!(logs.truncated);
    assert_eq!(logs.order, "newest_first");
    assert_eq!(logs.fluxes.len(), 2);
    let record = &logs.fluxes[0];
    assert_eq!(record.cell_id, cell_id.index());
    assert_eq!(record.x, 1.0);
    assert_eq!(record.y, 1.0);
    assert_eq!(record.catalyst_index, 0);
    assert_eq!(record.site, crate::cell::ReactionSite::Internal);
    assert!(record.executed_extent > 0.0);
    assert!(record.secreted_elements[Element::A.index()] > 0.0);
    assert!(record.energy_after > record.energy_before);
    assert!(
        (record.delta_cell_energy - (record.energy_after - record.energy_before)).abs() <= 1.0e-12
    );

    let detail = world.inspect_cell_detail(cell_id, 3).unwrap();
    assert!(detail.recent_fluxes.available);
    assert_eq!(detail.recent_fluxes.returned_count, 3);
    assert_valid_reservoir(&world, cell_id);
    world.check_invariants().unwrap();
}

#[test]
fn metabolic_flux_and_secretion_conserve_total_scalar_elements() {
    let mut world = World::new(only_a_config("flux-secretion-conservation", 4, 4)).unwrap();
    let tile = world.tile_id(1, 1).unwrap();
    let mut genome = Genome::random_founder(&mut world.rng, world.avg_enval);
    let mut catalyst = Enzyme::founder_downhill();
    catalyst.rate = 10.0;
    catalyst.secretion_fraction = 0.5;
    catalyst.enval_sigma = 1000.0;
    genome.enzymes = vec![catalyst];
    genome.maintenance_cost_per_area = 0.0;
    set_enval_optimum(&mut genome, tile_local_enval(&world, tile));
    let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
    world.cells[cell_id].internal_elements[Element::D] = 1.0;
    let a_before = world.element_fields[tile.index()][Element::A];
    let before = world.stats().total_element_amount;

    world.inspect_cell_fluxes(cell_id, 0).unwrap();
    world.step_cell_fresh(cell_id);

    let after = world.stats().total_element_amount;
    assert!((after - before).abs() <= 1.0e-5);
    let record = world.flux_log.latest_flux().unwrap();
    assert!(record.secreted_elements[Element::A.index()] > 0.0);
    assert!(
        world.element_fields[tile.index()][Element::A]
            + world.cells[cell_id].internal_elements[Element::A]
            > a_before
    );
    assert_valid_reservoir(&world, cell_id);
    world.check_invariants().unwrap();
}

#[test]
fn genome_patch_updates_selected_cell_fields_and_enzymes() {
    let mut world = World::new(only_a_config("genome-patch-cell", 4, 4)).unwrap();
    let tile = world.tile_id(1, 1).unwrap();
    let genome = Genome::random_founder(&mut world.rng, world.avg_enval);
    let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
    let energy_before = world.cells[cell_id].energy;
    let lineage_before = world.cells[cell_id].lineage_id;
    let patch = GenomePatch {
        schema: Some(crate::genome::GENOME_PATCH_SCHEMA.to_owned()),
        genome: Some(GenomeFieldPatch {
            mutation_rate: Some(0.02),
            maintenance_cost_per_area: Some(0.11),
        }),
        enzymes: vec![
            EnzymePatchOperation {
                op: "update".to_owned(),
                index: Some(0),
                fields: Some(EnzymeFieldPatch {
                    enval_sigma: Some(0.44),
                    enval_throughput: Some(0.22),
                    rate: Some(0.55),
                    ..EnzymeFieldPatch::default()
                }),
                enzyme: None,
            },
            EnzymePatchOperation {
                op: "append".to_owned(),
                index: None,
                fields: None,
                enzyme: Some(EnzymeFieldPatch {
                    rate: Some(0.77),
                    ..EnzymeFieldPatch::default()
                }),
            },
        ],
    };

    let result = world.apply_cell_genome_patch(cell_id, &patch).unwrap();
    let cell = &world.cells[cell_id];
    assert_eq!(result.patched_cell_count, 1);
    assert!(
        result
            .changed_fields
            .iter()
            .any(|field| field == "mutation_rate")
    );
    assert_eq!(cell.genome.mutation_rate, 0.02);
    assert_eq!(cell.maintenance_cost_per_area, 0.11);
    assert_eq!(cell.genome.maintenance_cost_per_area, 0.11);
    assert_eq!(cell.energy, energy_before);
    assert_eq!(cell.lineage_id, lineage_before);
    assert_eq!(cell.genome.lineage_id, lineage_before);
    assert_eq!(cell.genome.enzymes.last().unwrap().rate, 0.77);
    assert!(cell.genome.enzymes.len() >= 2);
    world.check_invariants().unwrap();
}

#[test]
fn inspection_reports_each_enzymes_niche_and_rate_weighted_summaries() {
    let mut world = World::new(only_a_config("niche-inspection", 8, 8)).unwrap();
    world.set_all_enval(0.5).unwrap();
    let mut first = Enzyme::founder_downhill();
    first.rate = 1.0;
    first.enval_sigma = 1000.0;
    first.enval_optimum = 0.5;
    let mut second = Enzyme::founder_engine_forward();
    second.rate = 3.0;
    second.enval_sigma = 1000.0;
    second.enval_optimum = -0.5;
    let cell_id = spawn_with(&mut world, Position::new(2.5, 2.5), 7, vec![first, second]);
    let mut third = Enzyme::founder_downhill();
    third.rate = 4.0;
    third.enval_optimum = 1.0;
    spawn_with(&mut world, Position::new(5.5, 5.5), 7, vec![third]);
    world.cells[cell_id].internal_elements = ElementAmounts::new([0.0, 0.5, 0.0, 0.5, 0.0, 0.0]);

    let cell = world.inspect_cell(cell_id).unwrap();
    assert_eq!(cell.mean_enval_optimum, (0.5 - 1.5) / 4.0);
    assert_eq!(cell.enval_optimum_spread, 1.0);

    let lineage = world.inspect_lineage(LineageId(7)).unwrap();
    assert!((lineage.mean_enval_optimum - (0.5 - 1.5 + 4.0) / 8.0).abs() <= 1.0e-12);

    world.inspect_cell_fluxes(cell_id, 0).unwrap();
    world.step_cell_fresh(cell_id);
    let detail = world.inspect_cell_detail(cell_id, 8).unwrap();
    assert_eq!(detail.genome.enzymes[0].enval_optimum, 0.5);
    assert_eq!(detail.genome.enzymes[1].enval_optimum, -0.5);
    assert_eq!(detail.recent_fluxes.fluxes.len(), 2);
    for record in &detail.recent_fluxes.fluxes {
        let expected = detail.genome.enzymes[record.catalyst_index].enval_optimum;
        assert_eq!(record.enval_optimum, expected);
    }
}

#[test]
fn invalid_genome_patch_is_rejected_atomically() {
    let mut world = World::new(only_a_config("genome-patch-invalid", 4, 4)).unwrap();
    let tile = world.tile_id(1, 1).unwrap();
    let genome = Genome::random_founder(&mut world.rng, world.avg_enval);
    let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
    let before = world.cells[cell_id].genome.clone();
    let patch = GenomePatch {
        schema: Some(crate::genome::GENOME_PATCH_SCHEMA.to_owned()),
        genome: Some(GenomeFieldPatch {
            mutation_rate: Some(1.5),
            ..GenomeFieldPatch::default()
        }),
        enzymes: Vec::new(),
    };

    assert!(world.apply_cell_genome_patch(cell_id, &patch).is_err());
    assert_eq!(world.cells[cell_id].genome, before);
    world.check_invariants().unwrap();
}

#[test]
fn enzyme_patch_respects_min_and_max_bounds() {
    let mut world = World::new(only_a_config("genome-patch-bounds", 4, 4)).unwrap();
    let tile = world.tile_id(1, 1).unwrap();
    let mut genome = Genome::random_founder(&mut world.rng, world.avg_enval);
    genome.enzymes = vec![Enzyme::founder_downhill()];
    let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
    let remove_patch = GenomePatch {
        schema: Some(crate::genome::GENOME_PATCH_SCHEMA.to_owned()),
        genome: None,
        enzymes: vec![EnzymePatchOperation {
            op: "remove".to_owned(),
            index: Some(0),
            fields: None,
            enzyme: None,
        }],
    };
    assert_eq!(world.cells[cell_id].genome.enzymes.len(), MIN_CELL_ENZYMES);
    assert!(
        world
            .apply_cell_genome_patch(cell_id, &remove_patch)
            .is_err()
    );

    while world.cells[cell_id].genome.enzymes.len() < MAX_CELL_ENZYMES {
        world.cells[cell_id]
            .genome
            .enzymes
            .push(Enzyme::founder_downhill());
    }
    let append_patch = GenomePatch {
        schema: Some(crate::genome::GENOME_PATCH_SCHEMA.to_owned()),
        genome: None,
        enzymes: vec![EnzymePatchOperation {
            op: "append".to_owned(),
            index: None,
            fields: None,
            enzyme: Some(EnzymeFieldPatch {
                rate: Some(0.5),
                ..EnzymeFieldPatch::default()
            }),
        }],
    };
    assert!(
        world
            .apply_cell_genome_patch(cell_id, &append_patch)
            .is_err()
    );
    assert_eq!(world.cells[cell_id].genome.enzymes.len(), MAX_CELL_ENZYMES);
}

#[test]
fn genome_brush_applies_patch_to_active_cells_in_wrapped_rect() {
    let mut world = World::new(only_a_config("genome-patch-brush", 4, 4)).unwrap();
    let first_tile = world.tile_id(3, 3).unwrap();
    let second_tile = world.tile_id(0, 0).unwrap();
    let first_genome = Genome::random_founder(&mut world.rng, world.avg_enval);
    let second_genome = Genome::random_founder(&mut world.rng, world.avg_enval);
    let first = world
        .spawn_cell_with_genome_at(first_tile, first_genome)
        .unwrap();
    let second = world
        .spawn_cell_with_genome_at(second_tile, second_genome)
        .unwrap();
    let patch = GenomePatch {
        schema: Some(crate::genome::GENOME_PATCH_SCHEMA.to_owned()),
        genome: Some(GenomeFieldPatch {
            mutation_rate: Some(0.33),
            ..GenomeFieldPatch::default()
        }),
        enzymes: Vec::new(),
    };

    let result = world.apply_genome_brush(3, 3, 3, 3, &patch).unwrap();
    assert_eq!(result.visited_tile_count, 9);
    assert_eq!(result.patched_cell_count, 2);
    assert_eq!(world.cells[first].genome.mutation_rate, 0.33);
    assert_eq!(world.cells[second].genome.mutation_rate, 0.33);
    world.check_invariants().unwrap();
}

#[test]
fn genome_brush_patches_every_unique_center_in_the_same_tile() {
    let mut world = World::new(only_a_config("genome-brush-multi-center", 5, 5)).unwrap();
    let first_genome = Genome::random_founder(&mut world.rng, world.avg_enval);
    let first = world
        .spawn_cell_with_genome_at_position(Position::new(1.02, 1.02), first_genome)
        .unwrap();
    let second_genome = Genome::random_founder(&mut world.rng, world.avg_enval);
    let second = world
        .spawn_cell_with_genome_at_position(Position::new(1.75, 1.75), second_genome)
        .unwrap();
    let patch = GenomePatch {
        schema: Some(crate::genome::GENOME_PATCH_SCHEMA.to_owned()),
        genome: Some(GenomeFieldPatch {
            mutation_rate: Some(0.44),
            ..GenomeFieldPatch::default()
        }),
        enzymes: Vec::new(),
    };

    let result = world.apply_genome_brush(1, 1, 1, 1, &patch).unwrap();

    assert_eq!(result.visited_tile_count, 1);
    assert_eq!(result.patched_cell_count, 2);
    assert_eq!(world.cells[first].genome.mutation_rate, 0.44);
    assert_eq!(world.cells[second].genome.mutation_rate, 0.44);
    assert_eq!(world.inspect_tile_xy(1, 1).unwrap().cell_center_count, 2);
}

#[test]
fn geometric_pick_uses_toroidal_containment_nearest_center_and_id_tie_break() {
    let mut world = World::new(only_a_config("geometric-pick", 10, 8)).unwrap();
    let first_genome = Genome::random_founder(&mut world.rng, world.avg_enval);
    let first = world
        .spawn_cell_with_genome_at_position(Position::new(1.0, 2.0), first_genome)
        .unwrap();
    let second_genome = Genome::random_founder(&mut world.rng, world.avg_enval);
    let second = world
        .spawn_cell_with_genome_at_position(Position::new(1.9, 2.0), second_genome)
        .unwrap();

    assert_eq!(world.pick_cell(Position::new(1.1, 2.0)), Some(first));
    assert_eq!(world.pick_cell(Position::new(1.8, 2.0)), Some(second));
    world.cells[second].position = Position::new(1.875, 2.0);
    world.rebuild_spatial_index();
    assert_eq!(world.pick_cell(Position::new(1.4375, 2.0)), Some(first));
    assert_eq!(world.pick_cell(Position::new(5.0, 5.0)), None);

    world.cells[first].position = Position::new(0.1, 4.0);
    world.cells[second].position = Position::new(5.0, 4.0);
    world.rebuild_spatial_index();
    assert_eq!(world.pick_cell(Position::new(9.9, 4.0)), Some(first));
}

#[test]
fn opposite_polarity_enzymes_in_one_cell_exchange_their_net_enval() {
    let mut world = World::new(only_a_config("mixed-polarity-cell", 7, 7)).unwrap();
    world.set_all_enval(0.5).unwrap();
    let tile = world.tile_id(3, 3).unwrap();
    let mut genome = Genome::random_founder(&mut world.rng, 0.5);
    let mut positive = Enzyme::founder_engine_forward();
    positive.rate = 10.0;
    positive.enval_sigma = 1000.0;
    positive.enval_pump = 0.2;
    positive.enval_optimum = 0.5;
    let mut negative = positive;
    negative.enval_optimum = -0.5;
    genome.enzymes = vec![positive, negative];
    genome.maintenance_cost_per_area = 0.0;
    let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
    world.cells[cell_id].internal_elements = ElementAmounts::new([0.0, 0.5, 0.0, 0.0, 0.0, 0.0]);
    let before = world.enval.clone();
    let ledger_before = world.enval_ledger();
    let mut expected_rng = world.rng.clone();

    world.inspect_cell_fluxes(cell_id, 0).unwrap();
    world.step_cell_fresh(cell_id);

    let records = world
        .flux_log
        .fluxes_newest_first()
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(records.len(), 2);
    let up = records.iter().find(|r| r.catalyst_index == 0).unwrap();
    let down = records.iter().find(|r| r.catalyst_index == 1).unwrap();
    assert_eq!(up.enval_optimum, 0.5);
    assert_eq!(down.enval_optimum, -0.5);
    assert!(up.enval_input > 0.0);
    assert_eq!(down.enval_input, 0.0);
    assert!(up.enval_output < 0.0);
    assert!(down.enval_output > 0.0);

    let input = up.enval_input + down.enval_input;
    let output = up.enval_output + down.enval_output;
    for (index, before_value) in before.iter().copied().enumerate() {
        let mut expected = before_value;
        if index == tile.index() {
            expected = expected - input + output;
        }
        assert!(
            (world.enval[index] - expected).abs() <= 1.0e-5,
            "tile {index}: {} != {expected}",
            world.enval[index]
        );
    }
    let ledger = world.enval_ledger();
    assert!((ledger.cell_uptake - ledger_before.cell_uptake + f64::from(input)).abs() <= 1.0e-6);
    assert!(
        (ledger.cell_emission - ledger_before.cell_emission - f64::from(output)).abs() <= 1.0e-6
    );
    assert!(world.enval_ledger_residual().abs() <= 1.0e-5);
    assert_eq!(
        world.rng.next_f64().to_bits(),
        expected_rng.next_f64().to_bits(),
        "enval exchange draws nothing"
    );
}

#[test]
fn zero_optimum_cell_reacts_without_touching_the_enval_field_or_the_rng() {
    let mut world = World::new(only_a_config("zero-optimum-cell", 7, 7)).unwrap();
    world.set_all_enval(0.5).unwrap();
    let tile = world.tile_id(3, 3).unwrap();
    let mut genome = Genome::random_founder(&mut world.rng, 0.5);
    let mut catalyst = Enzyme::founder_downhill();
    catalyst.rate = 10.0;
    catalyst.enval_sigma = 1000.0;
    catalyst.enval_pump = 0.3;
    catalyst.enval_optimum = 0.0;
    genome.enzymes = vec![catalyst];
    genome.maintenance_cost_per_area = 0.0;
    let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
    world.cells[cell_id].internal_elements[Element::D] = 1.0;
    let before = world.enval.clone();
    let mut untouched_rng = world.rng.clone();

    world.inspect_cell_fluxes(cell_id, 0).unwrap();
    world.step_cell_fresh(cell_id);

    let record = world.flux_log.latest_flux().unwrap();
    assert!(record.executed_extent > 0.0);
    assert!(record.raw_chemical_energy > 0.0);
    assert_eq!(record.enval_optimum, 0.0);
    assert_eq!(record.enval_input, 0.0);
    assert_eq!(record.enval_output, 0.0);
    assert_eq!(world.enval, before);
    assert_eq!(world.energy_ledger().pump_cost, 0.0);
    assert_eq!(world.energy_ledger().enval_harvest, 0.0);
    assert_eq!(
        world.rng.next_f64().to_bits(),
        untouched_rng.next_f64().to_bits()
    );
}

#[test]
fn surface_reactions_use_the_enzymes_own_optimum() {
    let mut config = only_a_config("surface-optimum", 8, 8);
    config.catalyst_upkeep_per_sec = 0.0;
    let mut world = World::new(config).unwrap();
    world.set_all_enval(0.5).unwrap();
    let mut catalyst = surface_enzyme(A_ONLY, E_ONLY, 2.0, 1.0);
    catalyst.enval_pump = 0.2;
    catalyst.enval_throughput = 0.2;
    let mut uncoupled = catalyst;
    catalyst.enval_optimum = 0.5;
    uncoupled.enval_optimum = 0.0;
    let coupled_id = spawn_with(&mut world, Position::new(1.5, 1.5), 1, vec![catalyst]);
    let uncoupled_id = spawn_with(&mut world, Position::new(5.5, 5.5), 2, vec![uncoupled]);
    for id in [coupled_id, uncoupled_id] {
        let tile = world.center_tile_id(world.cells[id].position).unwrap();
        world.element_fields[tile.index()][Element::A] = 5.0;
    }

    for (id, coupled) in [(coupled_id, true), (uncoupled_id, false)] {
        let before = world.energy_ledger();
        world.inspect_cell_fluxes(id, 0).unwrap();
        let context = reaction_context(&world, id);
        let enzyme = world.cells[id].genome.enzymes[0];
        let terms =
            crate::bio::EnzymeTerms::new(&enzyme, &context.flux_cell, world.config.dt_seconds);
        let reacting = super::cells::ReactingEnzyme {
            index: 0,
            enzyme: &enzyme,
            terms: &terms,
        };
        world.surface_reaction_on_field(&context, reacting, 1.0);
        world.apply_enval_exchange(&context);
        let record = world.flux_log.latest_flux().unwrap();
        assert_eq!(record.site, crate::cell::ReactionSite::Field);
        assert!(record.executed_extent > 0.0);
        let ledger = world.energy_ledger();
        if coupled {
            assert!(record.enval_input > 0.0);
            assert!(record.enval_output < 0.0);
            assert!(ledger.pump_cost > before.pump_cost);
        } else {
            assert_eq!(record.enval_input, 0.0);
            assert_eq!(record.enval_output, 0.0);
            assert_eq!(ledger.pump_cost, before.pump_cost);
        }
    }
}

#[test]
fn enval_coupling_changes_field() {
    let mut world = World::new(only_a_config("enval-coupling", 4, 4)).unwrap();
    let before = world.average_enval();
    let tile = world.tile_id(1, 1).unwrap();
    let mut genome = Genome::random_founder(&mut world.rng, world.avg_enval);
    let mut catalyst = Enzyme::founder_downhill();
    catalyst.rate = 10.0;
    catalyst.enval_optimum = world.tile_enval(tile).unwrap();
    genome.enzymes = vec![catalyst];
    world.config.founder_energy = 10.0;
    let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();
    world.cells[cell_id].internal_elements[Element::D] = 2.0;
    for _ in 0..5 {
        world.step_cell_fresh(cell_id);
    }
    assert_ne!(world.average_enval().to_bits(), before.to_bits());
    world.check_invariants().unwrap();
}

#[test]
fn deterministic_runs_produce_identical_stats() {
    fn run_once() -> crate::stats::WorldStats {
        let mut world = World::new(small_config("stats-golden", 16, 12)).unwrap();
        world.spawn_founder_cells(8).unwrap();
        for _ in 0..80 {
            world.step();
            world.check_invariants().unwrap();
        }
        world.stats()
    }
    let a = run_once();
    let b = run_once();
    assert_eq!(a, b);
}

#[test]
fn fixed_seed_cellular_runs_are_deterministic() {
    fn run_once() -> crate::stats::WorldStats {
        let mut world = World::new(small_config("golden-small-001", 16, 16)).unwrap();
        world.spawn_founder_cells(6).unwrap();
        for _ in 0..80 {
            world.step();
            world.check_invariants().unwrap();
        }
        world.stats()
    }
    let a = run_once();
    let b = run_once();
    assert_eq!(a.tick_count, b.tick_count);
    assert_eq!(a.live_cell_count, b.live_cell_count);
    assert_eq!(a.births, b.births);
    assert_eq!(a.deaths, b.deaths);
    assert_eq!(
        a.extracellular_element_amounts,
        b.extracellular_element_amounts
    );
    assert_eq!(
        a.intracellular_element_amounts,
        b.intracellular_element_amounts
    );
    assert_eq!(a.system_element_amounts, b.system_element_amounts);
    assert_eq!(a.average_enval.to_bits(), b.average_enval.to_bits());
}

#[test]
fn invariants_hold_after_many_steps() {
    let mut world = World::new(small_config("many-steps", 12, 10)).unwrap();
    world.spawn_founder_cells(5).unwrap();
    for _ in 0..100 {
        world.step();
        world.check_invariants().unwrap();
    }
}

#[test]
fn fixed_seed_steps_have_stable_summary_stats() {
    let mut world = World::new(small_config("golden-phase12", 16, 12)).unwrap();
    for _ in 0..25 {
        world.step();
    }
    let stats = world.stats();
    assert_eq!(stats.tick_count, 25);
    assert_eq!(stats.width, 16);
    assert_eq!(stats.height, 12);
    assert!(
        world
            .element_field_totals()
            .iter()
            .all(|total| *total > 0.0)
    );
    assert!(stats.average_enval.is_finite());
    world.check_invariants().unwrap();
}

#[test]
fn render_buffers_have_expected_lengths_and_values() {
    let mut world = World::new(small_config("render-buffers", 8, 6)).unwrap();
    world.spawn_founder_cells(5).unwrap();
    let before = world.build_render_buffers();
    assert_eq!(before.tile_enval.len(), 48);
    assert_eq!(before.tile_cell_count.len(), 48);
    assert_eq!(before.tile_mass_density.len(), 48);
    assert_eq!(before.tile_total_elements.len(), 48);
    assert_eq!(before.tile_element_concentrations.len(), 48 * ELEMENT_COUNT);
    assert_eq!(before.cell_count(), world.stats().live_cell_count);
    assert_eq!(before.cell_point_data.len(), before.cell_count() * 4);
    assert_eq!(before.cell_rotation_data.len(), before.cell_count() * 4);
    assert_eq!(before.cell_scale_data.len(), before.cell_count() * 4);
    assert_eq!(before.cell_rgba.len(), before.cell_count() * 4);
    assert_eq!(before.cell_radius.len(), before.cell_count());
    assert!(before.tile_enval.iter().all(|value| value.is_finite()));
    assert!(before.cell_energy.iter().all(|value| value.is_finite()));
    assert_eq!(
        before
            .tile_cell_count
            .iter()
            .map(|count| *count as usize)
            .sum::<usize>(),
        before.cell_count()
    );
    assert!(
        before
            .cell_radius
            .iter()
            .all(|radius| *radius == REFERENCE_CELL_RADIUS)
    );
    assert!(
        before
            .cell_rotation_data
            .chunks_exact(4)
            .all(|rotation| rotation == [0.0, 0.0, 0.0, 1.0])
    );
    assert!(before.cell_scale_data.chunks_exact(4).all(|scale| {
        scale
            == [
                REFERENCE_CELL_RADIUS,
                REFERENCE_CELL_RADIUS,
                REFERENCE_CELL_RADIUS,
                0.0,
            ]
    }));

    world.step_many(5);
    let after = world.build_render_buffers();
    assert_eq!(after.tile_count(), world.tile_count());
    assert_eq!(after.cell_count(), world.stats().live_cell_count);
    assert!(after.render_epoch >= before.render_epoch);
    world.check_invariants().unwrap();
}

#[test]
fn inspect_helpers_report_tile_and_cell_state() {
    let mut world = World::new(only_a_config("inspect-helpers", 4, 4)).unwrap();
    let tile = world.tile_id(2, 1).unwrap();
    let genome = Genome::random_founder(&mut world.rng, world.avg_enval);
    let cell_id = world.spawn_cell_with_genome_at(tile, genome).unwrap();

    let tile_info = world.inspect_tile(tile).unwrap();
    assert_eq!(tile_info.tile_id, tile);
    assert_eq!(tile_info.x, 2);
    assert_eq!(tile_info.y, 1);
    assert_eq!(tile_info.cell_center_count, 1);
    assert!(tile_info.enval.is_finite());

    let cell_info = world.inspect_cell(cell_id).unwrap();
    assert_eq!(cell_info.cell_id, cell_id);
    assert_eq!(cell_info.x, 2.0);
    assert_eq!(cell_info.y, 1.0);
    assert_eq!(cell_info.radius, REFERENCE_CELL_RADIUS);
    assert!(cell_info.energy.is_finite());
}

#[test]
fn write_render_buffers_reuses_existing_allocations_and_matches_builder() {
    let mut world = World::new(small_config("render-buffer-reuse", 8, 6)).unwrap();
    world.spawn_founder_cells(4).unwrap();
    world.step_many(5);

    let expected = world.build_render_buffers();
    let mut actual = RenderBuffers::default();
    actual.tile_enval.reserve(128);
    actual.cell_point_data.reserve(128);
    actual.cell_rotation_data.reserve(128);
    actual.cell_scale_data.reserve(128);
    actual.cell_rgba.reserve(128);
    let reserved_tile_capacity = actual.tile_enval.capacity();
    let reserved_point_capacity = actual.cell_point_data.capacity();
    let reserved_rotation_capacity = actual.cell_rotation_data.capacity();
    let reserved_scale_capacity = actual.cell_scale_data.capacity();
    let reserved_color_capacity = actual.cell_rgba.capacity();
    world.write_render_buffers(&mut actual);

    assert_eq!(actual, expected);
    assert!(actual.tile_enval.capacity() >= reserved_tile_capacity);
    assert!(actual.cell_point_data.capacity() >= reserved_point_capacity);
    assert!(actual.cell_rotation_data.capacity() >= reserved_rotation_capacity);
    assert!(actual.cell_scale_data.capacity() >= reserved_scale_capacity);
    assert!(actual.cell_rgba.capacity() >= reserved_color_capacity);
}

#[test]
fn step_many_matches_repeated_single_steps() {
    let mut many = World::new(small_config("step-many", 10, 8)).unwrap();
    many.spawn_founder_cells(4).unwrap();
    let mut repeated = World::new(small_config("step-many", 10, 8)).unwrap();
    repeated.spawn_founder_cells(4).unwrap();

    many.step_many(30);
    for _ in 0..30 {
        repeated.step();
    }

    assert_eq!(many.stats(), repeated.stats());
    many.check_invariants().unwrap();
    repeated.check_invariants().unwrap();
}

#[test]
fn expanded_stats_compartment_counts_and_enzyme_histogram_are_consistent() {
    let mut world = World::new(small_config("expanded-stats", 8, 6)).unwrap();
    world.spawn_founder_cells(4).unwrap();
    world.step_many(10);
    let stats = world.stats();

    assert_eq!(
        stats.occupied_tile_count + stats.empty_tile_count,
        stats.tile_count
    );
    for element in ELEMENT_ORDER {
        let index = element.index();
        assert_eq!(
            stats.system_element_amounts[index],
            stats.extracellular_element_amounts[index] + stats.intracellular_element_amounts[index]
        );
    }
    assert_eq!(
        stats.total_element_amount,
        stats.system_element_amounts.iter().sum::<f64>()
    );
    assert_eq!(stats.live_cell_count, world.cells.len());
    assert_eq!(
        stats.enzyme_count_histogram.iter().sum::<u64>(),
        stats.live_cell_count as u64
    );
    assert_eq!(
        stats.total_enzyme_count,
        live_ids(&world)
            .iter()
            .map(|cell_id| world.cells[*cell_id].genome.enzymes.len() as u64)
            .sum::<u64>()
    );
    assert!(stats.occupancy_fraction >= 0.0 && stats.occupancy_fraction <= 1.0);
    assert!(stats.average_cell_energy.is_finite());
    assert!(stats.enval_std_dev.is_finite());
    world.check_invariants().unwrap();
}

#[test]
fn enval_stddev_reports_uniform_and_nonuniform_fields() {
    let mut world = World::new(only_a_config("enval-stddev", 4, 4)).unwrap();
    world.set_all_enval(0.5).unwrap();
    assert_eq!(world.stats().enval_std_dev.to_bits(), 0.0_f32.to_bits());

    let tile = world.tile_id(0, 0).unwrap();
    world.set_tile_enval(tile, -0.5).unwrap();
    assert!(world.stats().enval_std_dev > 0.0);
}

#[test]
fn reaction_and_operation_counters_are_deterministic() {
    fn run_once() -> crate::stats::WorldStats {
        let mut world = World::new(small_config("counter-determinism", 12, 10)).unwrap();
        world.spawn_founder_cells(5).unwrap();
        for _ in 0..40 {
            world.step();
        }
        world.stats()
    }

    let a = run_once();
    let b = run_once();
    assert_eq!(a, b);
    assert!(a.operation_counters.cell_steps > 0);
    assert!(a.operation_counters.enzyme_entries_seen > 0);
    assert!(a.operation_counters.metabolic_enzyme_attempts > 0);
    assert_eq!(
        a.operation_counters.metabolic_enzyme_attempts,
        a.reaction_counters.total_attempts()
    );
    assert!(a.operation_counters.local_enval_average_calls > 0);
}

#[test]
fn compact_stats_do_not_mutate_rng_or_require_percentiles() {
    let mut world = World::new(small_config("compact-stats-stability", 10, 8)).unwrap();
    world.spawn_founder_cells(3).unwrap();
    world.step_many(5);
    let rng_before = format!("{:?}", world.rng());
    let full = world.stats();
    let compact = world.compact_stats();
    let rng_after = format!("{:?}", world.rng());

    assert_eq!(rng_before, rng_after);
    assert_eq!(compact.tick_count, full.tick_count);
    assert_eq!(compact.live_cell_count, full.live_cell_count);
    assert_eq!(compact.system_element_amounts, full.system_element_amounts);
    assert_eq!(compact.total_element_amount, full.total_element_amount);
    assert_eq!(compact.enval_p05.to_bits(), 0.0_f32.to_bits());
    assert_eq!(compact.enval_p50.to_bits(), 0.0_f32.to_bits());
    assert_eq!(compact.enval_p95.to_bits(), 0.0_f32.to_bits());
    assert_eq!(full.enval_p50.to_bits(), world.stats().enval_p50.to_bits());
    world.check_invariants().unwrap();
}

fn niche_enzyme(enval_optimum: f32, rate: f32) -> Enzyme {
    let mut enzyme = inert_enzyme();
    enzyme.enval_optimum = enval_optimum;
    enzyme.rate = rate;
    enzyme
}

#[test]
fn niche_stats_match_scripted_optima_and_a_known_enval_field() {
    let mut world = World::new(only_a_config("niche-stats", 16, 16)).unwrap();
    world.set_all_enval(0.2).unwrap();
    let empty = world.stats();
    assert_eq!(empty.mixed_polarity_cells, 0);
    assert_eq!(empty.mean_enval_optimum_spread, 0.0);
    assert_eq!(empty.mean_niche_mismatch, 0.0);

    let scripted = [
        (
            vec![niche_enzyme(0.5, 1.0), niche_enzyme(-0.5, 3.0)],
            true,
            1.0,
            0.6,
        ),
        (
            vec![niche_enzyme(0.04, 1.0), niche_enzyme(-0.06, 1.0)],
            false,
            0.1,
            0.21,
        ),
        (vec![niche_enzyme(0.3, 2.0)], false, 0.0, 0.1),
        (
            vec![niche_enzyme(0.05, 1.0), niche_enzyme(-0.05, 1.0)],
            false,
            0.1,
            0.2,
        ),
        (
            vec![niche_enzyme(0.0, 1.0), niche_enzyme(-1.0, 1.0)],
            false,
            1.0,
            0.7,
        ),
    ];
    let count = scripted.len() as f64;
    let mut mixed = 0;
    let mut spread = 0.0_f64;
    let mut mismatch = 0.0_f64;
    for (index, (enzymes, is_mixed, cell_spread, cell_mismatch)) in scripted.into_iter().enumerate()
    {
        let position = Position::new(1.5 + index as f32 * 3.0, 8.5);
        spawn_with(&mut world, position, index as u64 + 1, enzymes);
        mixed += usize::from(is_mixed);
        spread += cell_spread;
        mismatch += cell_mismatch;
    }

    let full = world.stats();
    assert_eq!(full.mixed_polarity_cells, mixed);
    assert_eq!(full.mixed_polarity_cells, 1);
    assert!((full.mean_enval_optimum_spread - spread / count).abs() <= 1.0e-6);
    assert!((full.mean_niche_mismatch - mismatch / count).abs() <= 1.0e-6);

    let compact = world.compact_stats();
    assert_eq!(compact.mixed_polarity_cells, full.mixed_polarity_cells);
    assert_eq!(
        compact.mean_enval_optimum_spread,
        full.mean_enval_optimum_spread
    );
    assert_eq!(compact.mean_niche_mismatch, 0.0);
}

#[test]
fn niche_mismatch_reads_the_smoothed_local_enval_not_the_cells_tile() {
    let mut world = World::new(only_a_config("niche-stats-local", 9, 9)).unwrap();
    world.set_all_enval(0.0).unwrap();
    let tile = world.tile_id(4, 4).unwrap();
    world.set_tile_enval(tile, 1.0).unwrap();
    let local = tile_local_enval(&world, tile);
    assert!((local - 0.04).abs() <= 1.0e-6);
    let mut genome = Genome::random_founder(&mut world.rng, 0.0);
    genome.enzymes = vec![niche_enzyme(-0.5, 1.0)];
    world.spawn_cell_with_genome_at(tile, genome).unwrap();
    let stats = world.stats();
    assert!((stats.mean_niche_mismatch - f64::from(local + 0.5)).abs() <= 1.0e-6);
}

#[test]
fn niche_stats_read_only_the_world_and_leave_the_rng_alone() {
    let mut world = World::new(small_config("niche-stats-readonly", 10, 8)).unwrap();
    world.spawn_founder_cells(3).unwrap();
    world.step_many(5);
    let rng_before = format!("{:?}", world.rng());
    let first = world.stats();
    let second = world.stats();
    assert_eq!(first, second);
    assert_eq!(rng_before, format!("{:?}", world.rng()));
    assert_eq!(first.mixed_polarity_cells, 0);
    assert_eq!(first.mean_enval_optimum_spread, 0.0);
    assert!(first.mean_niche_mismatch > 0.0);
}

#[test]
fn profiled_step_records_actual_operation_counters() {
    let mut world = World::new(small_config("profile-counters", 10, 8)).unwrap();
    world.spawn_founder_cells(3).unwrap();
    let profile = world.step_profiled();
    assert!(profile.counters.cell_steps > 0);
    assert!(profile.counters.enzyme_entries_seen > 0);
    assert_eq!(
        profile.counters.cell_steps,
        world.operation_counters().cell_steps
    );
    world.check_invariants().unwrap();
}

#[test]
fn continuous_founders_are_canonical_non_overlapping_and_seed_deterministic() {
    let mut first = World::new(only_a_config("continuous-founders", 12, 9)).unwrap();
    let mut second = World::new(only_a_config("continuous-founders", 12, 9)).unwrap();
    assert_eq!(first.spawn_founder_cells(20).unwrap(), 20);
    assert_eq!(second.spawn_founder_cells(20).unwrap(), 20);
    let positions = |world: &World| {
        live_ids(world)
            .iter()
            .map(|cell_id| world.cells[*cell_id].position)
            .collect::<Vec<_>>()
    };
    let first_positions = positions(&first);
    assert_eq!(first_positions, positions(&second));
    assert!(first_positions.iter().all(|position| {
        position.is_finite()
            && position.x >= 0.0
            && position.x < first.width as f32
            && position.y >= 0.0
            && position.y < first.height as f32
    }));
    for left in 0..first_positions.len() {
        for right in left + 1..first_positions.len() {
            assert!(
                toroidal_distance_squared(
                    first_positions[left],
                    first_positions[right],
                    first.width as f32,
                    first.height as f32,
                ) >= (REFERENCE_CELL_RADIUS * 2.0 - GEOMETRY_TOLERANCE).powi(2)
            );
        }
    }
    first.check_invariants().unwrap();
}

#[test]
fn newborns_do_not_receive_a_cell_step_during_their_birth_phase() {
    let mut world = World::new(only_a_config("newborn-waits", 8, 8)).unwrap();
    let mut genome = Genome::random_founder(&mut world.rng, world.avg_enval);
    make_inert(&mut genome);
    world.config.founder_energy = 10.0;
    genome.maintenance_cost_per_area = 0.0;
    let parent = world
        .spawn_cell_with_genome_at_position(Position::new(4.0, 4.0), genome)
        .unwrap();
    let before_steps = world.operation_counters.cell_steps;
    for element in ELEMENT_ORDER {
        world.cells[parent].boundary[element] *= 1.6;
    }
    world.step_cells();
    assert_eq!(live_ids(&world).len(), 2);
    assert_eq!(world.operation_counters.cell_steps - before_steps, 1);
    let child = live_ids(&world)
        .iter()
        .copied()
        .find(|cell_id| *cell_id != parent)
        .unwrap();
    assert_eq!(world.inspect_cell_fluxes(child, 64).unwrap().flux_count, 0);
}

#[test]
fn cell_phase_snapshot_reuses_its_allocation() {
    let mut world = World::new(only_a_config("cell-phase-scratch", 8, 8)).unwrap();
    for position in [
        Position::new(1.0, 1.0),
        Position::new(3.0, 1.0),
        Position::new(1.0, 3.0),
        Position::new(3.0, 3.0),
    ] {
        let mut genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        make_inert(&mut genome);
        world.config.founder_energy = 10.0;
        genome.maintenance_cost_per_area = 0.0;
        world
            .spawn_cell_with_genome_at_position(position, genome)
            .unwrap();
    }

    world.step_cells();
    let scratch_ptr = world.cell_phase_scratch.as_ptr();
    let scratch_capacity = world.cell_phase_scratch.capacity();
    world.step_cells();

    assert_eq!(world.cell_phase_scratch.as_ptr(), scratch_ptr);
    assert_eq!(world.cell_phase_scratch.capacity(), scratch_capacity);
    assert_eq!(world.cell_phase_scratch, live_ids(&world));
    world.check_invariants().unwrap();
}

#[test]
fn fractional_position_death_release_is_conservative() {
    let mut world = World::new(only_a_config("fractional-death", 5, 5)).unwrap();
    let genome = Genome::random_founder(&mut world.rng, world.avg_enval);
    let position = Position::new(1.25, 2.75);
    let cell_id = world
        .spawn_cell_with_genome_at_position(position, genome)
        .unwrap();
    assert_eq!(world.pick_cell(position), Some(cell_id));
    world.cells[cell_id].internal_elements = ElementAmounts::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let before = world.element_field_totals();
    let body = world.cells[cell_id].body;
    let boundary = world.cells[cell_id].boundary;
    world.kill_cell_and_release(cell_id);
    assert_eq!(world.pick_cell(position), None);
    assert_eq!(world.spatial_index.len(), 0);
    let after = world.element_field_totals();
    for element in ELEMENT_ORDER {
        assert!(
            (after[element.index()]
                - before[element.index()]
                - (element.index() + 1) as f64
                - f64::from(body[element])
                - f64::from(boundary[element]))
            .abs()
                <= 1.0e-5
        );
    }
    world.check_invariants().unwrap();
}

#[test]
fn failed_death_deposition_preserves_the_intracellular_reservoir() {
    let mut world = World::new(only_a_config("invalid-death-release", 5, 5)).unwrap();
    let genome = Genome::random_founder(&mut world.rng, world.avg_enval);
    let cell_id = world
        .spawn_cell_with_genome_at_position(Position::new(1.25, 2.75), genome)
        .unwrap();
    let reservoir = ElementAmounts::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    world.cells[cell_id].internal_elements = reservoir;
    world.cells[cell_id].position = Position::new(f32::NAN, 2.75);

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        world.kill_cell_and_release(cell_id);
    }));

    assert!(result.is_err());
    assert_eq!(world.cells[cell_id].internal_elements, reservoir);
    assert!(world.cell(cell_id).is_some());
}

#[test]
fn spatial_and_tile_caches_update_incrementally_for_cell_lifecycle() {
    let mut world = World::new(only_a_config("incremental-derived-caches", 6, 6)).unwrap();
    let first_genome = Genome::random_founder(&mut world.rng, world.avg_enval);
    let first = world
        .spawn_cell_with_genome_at_position(Position::new(1.02, 1.02), first_genome)
        .unwrap();
    let second_genome = Genome::random_founder(&mut world.rng, world.avg_enval);
    let second = world
        .spawn_cell_with_genome_at_position(Position::new(1.75, 1.75), second_genome)
        .unwrap();

    assert_eq!(world.inspect_tile_xy(1, 1).unwrap().cell_center_count, 2);
    assert_eq!(world.stats().occupied_tile_count, 1);

    world.update_cell_position(second, Position::new(3.25, 3.25));
    assert_eq!(world.inspect_tile_xy(1, 1).unwrap().cell_center_count, 1);
    assert_eq!(world.inspect_tile_xy(3, 3).unwrap().cell_center_count, 1);
    assert_eq!(world.stats().occupied_tile_count, 2);

    world.kill_cell_and_release(first);
    assert_eq!(world.inspect_tile_xy(1, 1).unwrap().cell_center_count, 0);
    assert_eq!(world.inspect_tile_xy(3, 3).unwrap().cell_center_count, 1);
    assert_eq!(world.stats().occupied_tile_count, 1);
    assert_eq!(
        world
            .spatial_index
            .query_radius(Position::new(3.25, 3.25), 0.01),
        vec![second]
    );
    world.check_invariants().unwrap();
}

#[test]
fn energy_ledger_closes_over_long_runs() {
    let mut world = World::new(small_config("energy-ledger-closure", 32, 24)).unwrap();
    world.spawn_founder_cells(12).unwrap();
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
    assert!(world.energy_ledger_residual().abs() <= ledger.closure_tolerance());
    assert!(ledger.founder_energy > 0.0);
    assert!(ledger.injected_energy > 0.0);
    assert!(ledger.extracted_energy > 0.0);
    assert!(ledger.chemical_harvest > 0.0);
    assert!(ledger.maintenance > 0.0);
    assert_eq!(world.stats().energy_ledger, ledger);
}

#[test]
fn energy_ledger_closure_detects_unledgered_energy_changes() {
    let mut world = World::new(small_config("energy-ledger-detects", 12, 10)).unwrap();
    world.spawn_founder_cells(3).unwrap();
    world.step_many(20);
    world.check_invariants().unwrap();
    let cell_id = live_ids(&world)[0];

    world.add_unledgered_energy(cell_id, 0.25);
    assert!(matches!(
        world.check_invariants(),
        Err(super::InvariantError::EnergyLedgerMismatch { .. })
    ));

    world.add_unledgered_energy(cell_id, -0.25);
    world.check_invariants().unwrap();
}

#[test]
fn enval_ledger_attributes_cell_exchange_and_user_edits() {
    let mut world = World::new(small_config("enval-ledger-basic", 24, 18)).unwrap();
    world.spawn_founder_cells(10).unwrap();
    for tick in 1..=2_000 {
        world.step();
        if tick == 500 {
            world.adjust_tile_enval(TileId(7), 0.75).unwrap();
        }
        if tick == 1_000 {
            world.set_tile_enval(TileId(11), -0.5).unwrap();
        }
    }
    let ledger = world.enval_ledger();
    assert!(ledger.edits != 0.0);
    assert!(ledger.cell_uptake != 0.0 || ledger.cell_emission != 0.0);
    let scale = 1.0 + world.tile_count() as f64;
    assert!(
        world.enval_ledger_residual().abs() <= 1.0e-4 * scale,
        "enval ledger residual {}",
        world.enval_ledger_residual()
    );
}

fn closed_config(seed: &str, width: usize, height: usize) -> Config {
    let mut config = small_config(seed, width, height);
    config.enval_landscape.relaxation_per_second = 0.0;
    config.enval_recharge.rate_per_second = 0.0;
    config
}

fn pump_engine_genome(world: &mut World, lineage: u64, enval_optimum: f32) -> Genome {
    let mut genome = Genome::random_founder(&mut world.rng, 0.0);
    let mut engine = Enzyme::metabolic(
        ElementAmounts::new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        ElementAmounts::new([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        2.0,
        0.0,
        0.0,
    );
    engine.enval_sigma = 1000.0;
    engine.enval_throughput = 0.5;
    engine.enval_pump = 0.4;
    engine.enval_optimum = enval_optimum;
    genome.enzymes = vec![engine];
    genome.lineage_id = LineageId(lineage);
    world.config.founder_energy = 5.0;
    genome.maintenance_cost_per_area = 0.0;
    genome
}

#[test]
fn paid_pumping_prevents_perpetual_motion_between_opposite_polarity_cells() {
    let mut config = closed_config("pump-no-perpetual-motion", 8, 8);
    config.catalyst_upkeep_per_sec = 0.0;
    config.element_fields.initial_amounts = ElementAmounts::new([0.0, 4.0, 0.0, 0.0, 0.0, 0.0]);
    let mut world = World::new(config).unwrap();
    world.set_all_enval(0.0).unwrap();
    let positive = pump_engine_genome(&mut world, 1, 0.3);
    let negative = pump_engine_genome(&mut world, 2, -0.3);
    world
        .spawn_cell_with_genome_at_position(Position::new(3.0, 4.0), positive)
        .unwrap();
    world
        .spawn_cell_with_genome_at_position(Position::new(4.0, 4.0), negative)
        .unwrap();

    let mut previous = world.total_live_cell_energy();
    let start = previous;
    for _ in 0..1_500 {
        world.step();
        let total = world.total_live_cell_energy();
        assert!(
            total <= previous + 1.0e-12,
            "energy rose: {previous} -> {total}"
        );
        previous = total;
    }
    let ledger = world.energy_ledger();
    assert!(
        ledger.enval_harvest > 0.0,
        "the pump/harvest loop never ran"
    );
    assert!(ledger.pump_cost > ledger.enval_harvest);
    assert_eq!(ledger.chemical_harvest, 0.0);
    assert!(previous < start);
    world.check_invariants().unwrap();
}

#[test]
fn perturbed_cell_free_tiles_relax_toward_their_targets_at_kappa_per_tick() {
    let mut config = small_config("landscape-relaxation", 96, 72);
    config.enval_diffusivity_per_second = 0.0;
    config.enval_recharge.rate_per_second = 0.0;
    let mut world = World::new(config).unwrap();
    let targets = world.enval_landscape().to_vec();
    let kappa = 1.0
        - (-f64::from(world.config.enval_landscape.relaxation_per_second)
            * world.config.dt_seconds)
            .exp();
    let edited = TileId(1234);
    world.set_tile_enval(edited, 7.5).unwrap();
    world.set_tile_enval(TileId(77), -3.0).unwrap();
    for _ in 0..300 {
        let before = world.enval.clone();
        let inflow_before = world.enval_ledger().source_inflow;
        world.step();
        let mut applied = 0.0;
        for tile in 0..world.tile_count() {
            let expected = f64::from(before[tile])
                + kappa * (f64::from(targets[tile]) - f64::from(before[tile]));
            assert!(
                (f64::from(world.enval[tile]) - expected).abs() <= 1.0e-6,
                "tile {tile}"
            );
            applied += f64::from(world.enval[tile]) - f64::from(before[tile]);
        }
        let inflow = world.enval_ledger().source_inflow - inflow_before;
        assert!((inflow - applied).abs() <= 1.0e-9);
    }
    let expected_gap = (7.5 - f64::from(targets[edited.index()])) * (1.0 - kappa).powi(300);
    let gap = f64::from(world.enval[edited.index()]) - f64::from(targets[edited.index()]);
    assert!((gap - expected_gap).abs() <= 1.0e-4 * expected_gap.abs());
    assert!(gap < 7.5 - f64::from(targets[edited.index()]));
    assert!(world.enval_ledger_residual().abs() <= 1.0e-6);

    let mut frozen_config = small_config("landscape-relaxation", 96, 72);
    frozen_config.enval_landscape.relaxation_per_second = 0.0;
    frozen_config.enval_recharge.rate_per_second = 0.0;
    frozen_config.enval_diffusivity_per_second = 0.0;
    let mut frozen = World::new(frozen_config).unwrap();
    frozen.set_tile_enval(edited, 7.5).unwrap();
    let before = frozen.enval.clone();
    frozen.step_many(200);
    assert_eq!(frozen.enval_ledger().source_inflow, 0.0);
    assert_eq!(frozen.enval, before);
}

#[test]
fn landscape_is_seed_deterministic_bounded_periodic_and_differs_between_seeds() {
    let first = World::new(small_config("world-landscape", 320, 240)).unwrap();
    let second = World::new(small_config("world-landscape", 320, 240)).unwrap();
    let other = World::new(small_config("world-landscape-other", 320, 240)).unwrap();
    assert_eq!(first.enval_landscape(), second.enval_landscape());
    assert_eq!(first.enval, second.enval);
    assert_ne!(first.enval_landscape(), other.enval_landscape());
    let magnitude = first.config.enval_landscape.magnitude;
    let peak = first
        .enval_landscape()
        .iter()
        .fold(0.0_f32, |peak, value| peak.max(value.abs()));
    assert_eq!(peak, magnitude);
    assert!(
        first
            .enval_landscape()
            .iter()
            .all(|value| value.abs() <= magnitude)
    );
    let at = |x: usize, y: usize| first.enval_landscape()[first.index_xy(x, y)];
    let mut interior = 0.0_f32;
    for x in 0..first.width - 1 {
        for y in 0..first.height - 1 {
            interior = interior
                .max((at(x + 1, y) - at(x, y)).abs())
                .max((at(x, y + 1) - at(x, y)).abs());
        }
    }
    let mut seam = 0.0_f32;
    for y in 0..first.height {
        seam = seam.max((at(0, y) - at(first.width - 1, y)).abs());
    }
    for x in 0..first.width {
        seam = seam.max((at(x, 0) - at(x, first.height - 1)).abs());
    }
    assert!(
        interior > 0.0 && seam <= interior,
        "seam {seam} interior {interior}"
    );
    let positive = first
        .enval_landscape()
        .iter()
        .filter(|value| **value > 0.0)
        .count();
    assert!(positive > first.tile_count() / 5 && positive < 4 * first.tile_count() / 5);
}

#[test]
fn enval_ledger_closes_with_landscape_cells_recharge_and_user_edits() {
    let mut config = small_config("enval-ledger-landscape", 64, 48);
    config.initial_founder_count = 12;
    let mut world = World::new(config).unwrap();
    world.spawn_founder_cells(12).unwrap();
    let mut max_residual = 0.0_f64;
    for tick in 1..=2_000 {
        world.step();
        match tick {
            300 => world.adjust_tile_enval(TileId(5), 2.0).unwrap(),
            900 => world.set_tile_enval(TileId(3), -3.0).unwrap(),
            1_400 => world.set_all_enval(0.05).unwrap(),
            _ => {}
        }
        max_residual = max_residual.max(world.enval_ledger_residual().abs());
    }
    let ledger = world.enval_ledger();
    assert!(ledger.source_inflow != 0.0);
    assert!(ledger.recharge != 0.0);
    assert!(ledger.edits != 0.0);
    assert!(ledger.cell_uptake != 0.0 || ledger.cell_emission != 0.0);
    assert!(
        max_residual <= 1.0e-3,
        "enval ledger residual {max_residual}"
    );
    world.check_invariants().unwrap();
}

#[test]
fn cell_free_recharge_conserves_matter_never_flips_enval_and_matches_energy() {
    let mut config = small_config("recharge-cell-free", 64, 48);
    config.element_fields.initial_amounts = ElementAmounts::new([0.2, 0.2, 0.2, 0.0, 0.0, 0.6]);
    config.enval_recharge.rate_per_second = 0.5;
    let mut world = World::new(config).unwrap();
    let initial_total = world.stats().total_element_amount;
    let mut consumed = 0.0_f64;
    for _ in 0..1_500 {
        world.diffuse_element_fields();
        world.diffuse_enval();
        let before = world.enval.clone();
        let energy_before = world.enval_ledger.recharge_energy;
        world.apply_recharge();
        let mut tick_consumed = 0.0_f64;
        for (old, new) in before.iter().zip(&world.enval) {
            assert!(
                old.signum() == new.signum() || *new == 0.0,
                "recharge flipped enval"
            );
            assert!(new.abs() <= old.abs());
            tick_consumed += f64::from(old.abs()) - f64::from(new.abs());
        }
        let tick_energy = world.enval_ledger.recharge_energy - energy_before;
        assert!((tick_energy - tick_consumed).abs() <= 1.0e-5 + 1.0e-4 * tick_energy);
        consumed += tick_consumed;
        world.relax_enval_landscape();
        world.refresh_enval_sum();
        world.advance_time();
        assert!(
            world
                .element_fields
                .iter()
                .all(|amounts| amounts[Element::F] >= 0.0)
        );
    }
    let ledger = world.enval_ledger();
    assert!(ledger.recharge_amount > 1.0);
    assert!((ledger.recharge_energy - consumed).abs() <= 1.0e-3 * consumed);
    let total = world.stats().total_element_amount;
    assert!((total - initial_total).abs() <= 1.0e-5 * initial_total);
    world.check_invariants().unwrap();
}

#[test]
fn zero_recharge_rate_is_bit_identical_to_skipping_the_recharge_step() {
    let mut config = small_config("recharge-disabled", 48, 36);
    config.enval_recharge.rate_per_second = 0.0;
    let mut stepped = World::new(config).unwrap();
    stepped.spawn_founder_cells(8).unwrap();
    let mut manual = stepped.clone();
    for _ in 0..400 {
        stepped.step();
        manual.diffuse_element_fields();
        manual.step_cells();
        manual.drift_cells();
        manual.resolve_overlaps();
        manual.diffuse_enval();
        manual.relax_enval_landscape();
        manual.refresh_enval_sum();
        manual.advance_time();
    }
    assert_eq!(stepped.element_fields, manual.element_fields);
    assert_eq!(
        stepped
            .enval
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        manual
            .enval
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(stepped.enval_ledger().recharge_amount, 0.0);
}

#[test]
fn recharge_makes_d_only_where_enval_is_positive_and_e_only_where_negative() {
    let mut config = small_config("recharge-sign", 96, 72);
    config.enval_diffusivity_per_second = 0.0;
    config.element_fields.diffusivities = ElementAmounts::ZERO;
    config.element_fields.initial_amounts = ElementAmounts::new([0.0, 0.0, 0.0, 0.1, 0.1, 1.0]);
    config.element_fields.heterogeneity = 0.0;
    config.enval_recharge.rate_per_second = 0.5;
    let mut world = World::new(config).unwrap();
    world.step_many(300);
    let mut saw_d = false;
    let mut saw_e = false;
    for tile in 0..world.tile_count() {
        let amounts = world.element_fields[tile];
        let d_gain = amounts[Element::D] - 0.1;
        let e_gain = amounts[Element::E] - 0.1;
        let target = world.enval_landscape()[tile];
        if target >= 0.0 {
            assert!(e_gain.abs() <= 1.0e-7);
        }
        if target <= 0.0 {
            assert!(d_gain.abs() <= 1.0e-7);
        }
        if target > 0.05 {
            assert!(d_gain > 0.0);
            saw_d = true;
        } else if target < -0.05 {
            assert!(e_gain > 0.0);
            saw_e = true;
        }
    }
    assert!(saw_d && saw_e);
}

#[test]
fn catalyst_upkeep_charges_every_enzyme_on_top_of_base_maintenance() {
    let drain_with = |enzymes: Vec<Enzyme>| {
        let mut world = World::new(closed_config("catalyst-upkeep", 6, 6)).unwrap();
        let mut genome = Genome::random_founder(&mut world.rng, 0.0);
        genome.enzymes = enzymes;
        world.config.founder_energy = 10.0;
        genome.maintenance_cost_per_area = 0.05;
        let cell_id = world
            .spawn_cell_with_genome_at_position(Position::new(2.0, 2.0), genome)
            .unwrap();
        let before = world.cells[cell_id].energy;
        world.step_cell_fresh(cell_id);
        let paid = before - world.cells[cell_id].energy;
        assert!((world.energy_ledger().maintenance - paid).abs() <= 1.0e-15);
        assert_eq!(
            world
                .inspect_cell_detail(cell_id, 0)
                .unwrap()
                .catalyst_upkeep_per_sec,
            0.01 * world.cells[cell_id].genome.enzymes.len() as f64
        );
        paid
    };
    let dt = Config::default().dt_seconds;
    let one = drain_with(vec![inert_enzyme()]);
    assert!(
        (one - (0.05 * f64::from(crate::spatial::REFERENCE_CELL_AREA) + 0.01) * dt).abs()
            <= 1.0e-10
    );
    let four = drain_with(vec![inert_enzyme(); 4]);
    assert!(
        (four - (0.05 * f64::from(crate::spatial::REFERENCE_CELL_AREA) + 0.04) * dt).abs()
            <= 1.0e-10
    );
}

#[test]
fn small_default_world_stays_alive_and_honest() {
    for seed in ["viability-1", "viability-2", "viability-3"] {
        let mut config = small_config(seed, 64, 48);
        config.initial_founder_count = 8;
        let mut world = World::new(config).unwrap();
        assert_eq!(world.spawn_founder_cells(8).unwrap(), 8);
        for tick in 1..=3_000 {
            world.step();
            if tick % 100 == 0 {
                assert!(
                    world.cell_count() > 0,
                    "seed {seed} went extinct by tick {tick}"
                );
            }
        }
        let residual = world.energy_ledger_residual();
        assert!(
            residual.abs() <= world.energy_ledger().closure_tolerance(),
            "seed {seed}: ledger residual {residual}"
        );
        world.check_invariants().unwrap();
        assert!(
            world.reaction_counters.divisions >= 1,
            "seed {seed} had no fission"
        );
        let stats = world.stats();
        assert!(stats.energy_ledger.enval_harvest > 0.0);
        assert!(stats.enval_ledger.recharge_energy > 0.0);
    }
}

#[test]
fn dead_cells_are_compacted_with_stable_unreused_ids() {
    let mut world = World::new(closed_config("dead-cell-compaction", 24, 18)).unwrap();
    world.spawn_founder_cells(12).unwrap();
    let ids = live_ids(&world);
    let positions = ids
        .iter()
        .map(|cell_id| (*cell_id, world.cells[*cell_id].position))
        .collect::<Vec<_>>();
    for victim in [ids[0], ids[5], ids[11]] {
        world.kill_cell_and_release(victim);
    }
    world.check_invariants().unwrap();
    assert_eq!(world.cell_count(), 9);
    for (cell_id, position) in positions {
        if [ids[0], ids[5], ids[11]].contains(&cell_id) {
            assert!(world.cell(cell_id).is_none());
            assert!(world.inspect_cell(cell_id).is_none());
            assert!(world.inspect_cell_detail(cell_id, 4).is_none());
            assert!(world.inspect_cell_fluxes(cell_id, 4).is_none());
            assert_ne!(world.pick_cell(position), Some(cell_id));
        } else {
            let cell = world.cell(cell_id).unwrap();
            assert_eq!(cell.id, cell_id);
            assert_eq!(cell.position, position);
            assert_eq!(world.inspect_cell(cell_id).unwrap().cell_id, cell_id);
            assert_eq!(world.pick_cell(position), Some(cell_id));
        }
    }
    let genome = Genome::random_founder(&mut world.rng, 0.0);
    let newborn = world
        .spawn_cell_with_genome_at_position(Position::new(0.5, 0.5), genome)
        .or_else(|_| {
            let genome = Genome::random_founder(&mut world.rng, 0.0);
            world.spawn_cell_with_genome_at_position(Position::new(12.3, 9.1), genome)
        })
        .unwrap();
    assert_eq!(newborn, crate::cell::CellId(12));
    assert!(world.cell(ids[0]).is_none());

    world.step_many(50);
    world.check_invariants().unwrap();
    let reloaded =
        crate::snapshot::from_bytes(&crate::snapshot::to_bytes(&world).unwrap()).unwrap();
    assert_eq!(live_ids(&reloaded), live_ids(&world));
    for cell_id in [ids[0], ids[5], ids[11]] {
        assert!(reloaded.cell(cell_id).is_none());
    }
    reloaded.check_invariants().unwrap();
}

#[test]
fn snapshot_size_does_not_grow_with_cumulative_deaths() {
    let populate = |deaths: usize| {
        let mut world = World::new(closed_config("snapshot-death-size", 40, 30)).unwrap();
        let mut genome = Genome::random_founder(&mut world.rng, 0.0);
        genome.lineage_id = LineageId(7);
        let mut churned = 0;
        while churned < deaths {
            let id = world
                .spawn_cell_with_genome_at_position(Position::new(20.0, 15.0), genome.clone())
                .unwrap();
            world.kill_cell_and_release(id);
            churned += 1;
        }
        for index in 0..100 {
            let position = Position::new((index % 20) as f32 * 2.0, (index / 20) as f32 * 2.0);
            world
                .spawn_cell_with_genome_at_position(position, genome.clone())
                .unwrap();
        }
        world.check_invariants().unwrap();
        world
    };
    let quiet = populate(0);
    let churned = populate(20_000);
    assert_eq!(quiet.cell_count(), churned.cell_count());
    assert_eq!(churned.stats().deaths, 20_000);
    let quiet_size = crate::snapshot::to_bytes(&quiet).unwrap().len() as f64;
    let churned_size = crate::snapshot::to_bytes(&churned).unwrap().len() as f64;
    assert!(
        (churned_size - quiet_size).abs() <= 0.05 * quiet_size,
        "{churned_size} vs {quiet_size} bytes"
    );
    assert!(churned.cells.issued_ids() >= 20_100);
}

fn surface_enzyme(reactants: [f32; 6], products: [f32; 6], rate: f32, exposure: f32) -> Enzyme {
    let mut enzyme = Enzyme::metabolic(
        ElementAmounts::new(reactants),
        ElementAmounts::new(products),
        rate,
        0.5,
        0.0,
    );
    enzyme.enval_sigma = 1000.0;
    enzyme.enval_throughput = 0.0;
    enzyme.enval_pump = 0.0;
    enzyme.exposure = exposure;
    enzyme
}

const E_ONLY: [f32; 6] = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
const A_ONLY: [f32; 6] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0];

fn spawn_with(
    world: &mut World,
    position: Position,
    lineage: u64,
    enzymes: Vec<Enzyme>,
) -> crate::cell::CellId {
    let mut genome = Genome::random_founder(&mut world.rng, 0.0);
    genome.enzymes = enzymes;
    genome.lineage_id = LineageId(lineage);
    genome.maintenance_cost_per_area = 0.0;
    world
        .spawn_cell_with_genome_at_position(position, genome)
        .unwrap()
}

fn reaction_context(world: &World, cell_id: crate::cell::CellId) -> CellReactionContext {
    let slot = world.cells.slot(cell_id).unwrap();
    let cell = world.cells.at_slot(slot);
    CellReactionContext {
        cell_id,
        slot,
        position: cell.position,
        stencil: world.tile_stencil(cell.position).unwrap(),
        flux_cell: crate::bio::FluxCell::new(
            world.local_enval_at(cell.position).unwrap(),
            std::f32::consts::PI * cell.radius * cell.radius,
            world.config.cell_body,
        ),
        watched: world.watched_cell_id == Some(cell_id),
    }
}

fn contacts_of(world: &World, cell_id: crate::cell::CellId) -> (Vec<Contact>, f64) {
    let cell = &world.cells[cell_id];
    let mut contacts = Vec::new();
    let field = world.collect_contacts(cell_id, cell.position, cell.radius, &mut contacts);
    (contacts, field)
}

fn strain_of(world: &World, cell_id: crate::cell::CellId) -> f64 {
    world.cells[cell_id].strain(world.config.cell_body.boundary_density)
}

#[test]
fn exposed_enzyme_digests_a_matching_neighbour_boundary_until_it_lyses() {
    let mut config = only_a_config("surface-digestion", 8, 8);
    config.catalyst_upkeep_per_sec = 0.0;
    config.founder_energy = 50.0;
    let mut world = World::new(config).unwrap();
    let digester = spawn_with(
        &mut world,
        Position::new(3.0, 4.0),
        1,
        vec![surface_enzyme(E_ONLY, A_ONLY, 2.0, 1.0)],
    );
    let prey = spawn_with(&mut world, Position::new(3.9, 4.0), 2, vec![inert_enzyme()]);
    let boundary = world.cells[prey].boundary.total() as f32;
    world.cells[prey].boundary = ElementAmounts::new([0.0, 0.0, 0.0, 0.0, boundary * 0.8, 0.0]);
    let digester_boundary = world.cells[digester].boundary;
    let initial_total = world.stats().total_element_amount;
    world.inspect_cell_fluxes(digester, 64).unwrap();

    let mut previous = strain_of(&world, prey);
    assert!(previous > world.config.cell_body.lysis_strain);
    let mut lysed = false;
    for _ in 0..2_000 {
        world.step_cell_fresh(digester);
        let Some(_) = world.cell(prey) else { break };
        let strain = strain_of(&world, prey);
        assert!(strain <= previous);
        previous = strain;
        world.step_cell_fresh(prey);
        world.advance_time();
        world.check_invariants().unwrap();
        if world.cell(prey).is_none() {
            lysed = true;
            break;
        }
    }

    assert!(lysed, "prey never lysed; strain {previous}");
    assert!(previous < world.config.cell_body.lysis_strain);
    assert_eq!(world.lysis_count, 1);
    assert_eq!(world.energy_death_count, 0);
    let counters = world.interaction_counters();
    assert_eq!(counters.lyses_after_digestion, 1);
    assert!(counters.boundary_digested > 0.0);
    assert_eq!(
        counters.cross_lineage_boundary_digested,
        counters.boundary_digested
    );
    assert!(counters.cell_surface_extent > 0.0);
    assert_eq!(world.cells[digester].boundary, digester_boundary);
    assert!(
        world
            .flux_log
            .fluxes_newest_first()
            .any(|record| record.site == crate::cell::ReactionSite::Cell(prey))
    );
    let total = world.stats().total_element_amount;
    assert!((total - initial_total).abs() <= 1.0e-5 * initial_total);
    world.check_invariants().unwrap();
}

#[test]
fn lyses_count_as_digestion_only_within_the_attribution_window() {
    for (since, attributed) in [(None, 0), (Some(100), 1), (Some(101), 0)] {
        let mut config = only_a_config("lysis-attribution", 8, 8);
        config.catalyst_upkeep_per_sec = 0.0;
        let mut world = World::new(config).unwrap();
        for _ in 0..200 {
            world.advance_time();
        }
        let id = spawn_with(&mut world, Position::new(2.5, 3.5), 1, vec![inert_enzyme()]);
        for element in ELEMENT_ORDER {
            world.cells[id].boundary[element] *= 0.6;
        }
        world.cells[id].last_digested_tick = since.map(|ago| world.tick_count - ago);
        world.step_cell_fresh(id);
        assert_eq!(world.lysis_count, 1);
        assert_eq!(
            world.interaction_counters().lyses_after_digestion,
            attributed,
            "{since:?}"
        );
    }
}

#[test]
fn surface_catalysts_never_digest_their_own_boundary() {
    let mut config = only_a_config("surface-self", 8, 8);
    config.catalyst_upkeep_per_sec = 0.0;
    config.founder_energy = 50.0;
    let mut world = World::new(config).unwrap();
    let cell = spawn_with(
        &mut world,
        Position::new(3.0, 4.0),
        1,
        vec![surface_enzyme(E_ONLY, A_ONLY, 5.0, 1.0)],
    );
    let neighbour = spawn_with(&mut world, Position::new(3.9, 4.0), 1, vec![inert_enzyme()]);
    let total = world.cells[cell].boundary.total() as f32;
    world.cells[cell].boundary = ElementAmounts::new([0.0, 0.0, 0.0, 0.0, total, 0.0]);
    let own_boundary = world.cells[cell].boundary;
    let neighbour_boundary = world.cells[neighbour].boundary;
    let (contacts, _) = contacts_of(&world, cell);
    assert_eq!(contacts.len(), 1);
    assert!(contacts.iter().all(|contact| contact.cell_id != cell));
    world.inspect_cell_fluxes(cell, 64).unwrap();

    for _ in 0..500 {
        world.step_cell_fresh(cell);
        world.advance_time();
    }

    assert_eq!(world.cells[cell].boundary, own_boundary);
    assert_eq!(world.cells[neighbour].boundary, neighbour_boundary);
    assert_eq!(world.cells[cell].last_digested_tick, None);
    let counters = world.interaction_counters();
    assert_eq!(counters.boundary_digested, 0.0);
    assert_eq!(counters.cell_surface_extent, 0.0);
    assert_eq!(counters.field_surface_extent, 0.0);
    assert!(world.reaction_counters().no_substrate >= 1_000);
    assert_eq!(world.flux_log.flux_count(), 0);
    world.check_invariants().unwrap();
}

#[test]
fn isolated_surface_catalysis_draws_exactly_on_the_field_and_conserves_matter() {
    let mut config = small_config("surface-field", 12, 10);
    config.founder_energy = 20.0;
    let mut world = World::new(config).unwrap();
    let mut enzyme = Enzyme::founder_downhill();
    enzyme.rate = 5.0;
    enzyme.secretion_fraction = 0.0;
    enzyme.body_fraction = 0.0;
    enzyme.boundary_fraction = 0.0;
    enzyme.enval_sigma = 1000.0;
    enzyme.exposure = 1.0;
    let cell = spawn_with(&mut world, Position::new(5.3, 4.7), 1, vec![enzyme]);
    let (contacts, field_share) = contacts_of(&world, cell);
    assert!(contacts.is_empty());
    assert_eq!(field_share, 1.0);

    let stencil = crate::spatial::BilinearStencil::new(Position::new(5.3, 4.7), 12, 10).unwrap();
    let tiles = stencil
        .samples
        .map(|sample| world.index_xy(sample.x, sample.y));
    let tiles_before = tiles.map(|tile| world.element_fields[tile]);
    let field_before = world.element_field_totals();
    let internal_before = world.cells[cell].internal_elements;
    let system_before = world.stats().system_element_amounts;
    world.inspect_cell_fluxes(cell, 4).unwrap();
    let context = reaction_context(&world, cell);
    let terms = crate::bio::EnzymeTerms::new(&enzyme, &context.flux_cell, world.config.dt_seconds);
    let reacting = super::cells::ReactingEnzyme {
        index: 0,
        enzyme: &enzyme,
        terms: &terms,
    };
    world.surface_reaction_on_field(&context, reacting, 1.0);

    let record = world.flux_log.latest_flux().unwrap().clone();
    assert_eq!(record.site, crate::cell::ReactionSite::Field);
    let consumed = f64::from(record.executed_extent);
    assert!(consumed > 0.0);
    let field_after = world.element_field_totals();
    let d = Element::D.index();
    assert!(
        (field_before[d] - field_after[d] - consumed).abs() <= 1.0e-6,
        "{} vs {consumed}",
        field_before[d] - field_after[d]
    );
    for element in ELEMENT_ORDER {
        if element != Element::D {
            assert_eq!(field_after[element.index()], field_before[element.index()]);
        }
    }
    let weighted = (0..4)
        .map(|corner| {
            f64::from(stencil.samples[corner].weight) * f64::from(tiles_before[corner][Element::D])
        })
        .collect::<Vec<_>>();
    let pool = weighted.iter().sum::<f64>();
    for corner in 0..4 {
        let taken = f64::from(tiles_before[corner][Element::D])
            - f64::from(world.element_fields[tiles[corner]][Element::D]);
        assert!((taken - consumed * weighted[corner] / pool).abs() <= 1.0e-6);
    }
    let gained = f64::from(world.cells[cell].internal_elements[Element::A])
        - f64::from(internal_before[Element::A]);
    assert!((gained - consumed).abs() <= 1.0e-6);
    let system_after = world.stats().system_element_amounts;
    for element in ELEMENT_ORDER {
        let index = element.index();
        let expected = if element == Element::D {
            -consumed
        } else if element == Element::A {
            consumed
        } else {
            0.0
        };
        assert!((system_after[index] - system_before[index] - expected).abs() <= 1.0e-5);
    }
    assert!(world.interaction_counters().field_surface_extent > 0.0);
    assert_eq!(world.stats().interaction, world.interaction_counters());
    world.check_invariants().unwrap();

    let initial_total = world.stats().total_element_amount;
    for tick in 1..=1_000 {
        world.step();
        if tick % 100 == 0 {
            world.check_invariants().unwrap();
        }
    }
    let total = world.stats().total_element_amount;
    assert!((total - initial_total).abs() <= 1.0e-6 * initial_total);
    assert!(world.interaction_counters().field_surface_extent > consumed);
    assert_eq!(world.interaction_counters().boundary_digested, 0.0);
    assert!(world.energy_ledger_residual().abs() <= world.energy_ledger().closure_tolerance());
}

#[test]
fn contact_shares_follow_the_formula_sum_to_at_most_one_and_wrap_symmetrically() {
    let mut world = World::new(only_a_config("contact-shares", 20, 16)).unwrap();
    let gap = f64::from(world.config.surface.contact_gap);
    let isolated = spawn_with(
        &mut world,
        Position::new(15.0, 12.0),
        1,
        vec![inert_enzyme()],
    );
    let (contacts, field) = contacts_of(&world, isolated);
    assert!(contacts.is_empty());
    assert_eq!(field, 1.0);

    let mut pair_shares = Vec::new();
    for (a, b) in [
        (Position::new(5.0, 5.0), Position::new(5.95, 5.0)),
        (Position::new(0.2, 9.0), Position::new(19.25, 9.0)),
    ] {
        let first = spawn_with(&mut world, a, 1, vec![inert_enzyme()]);
        let second = spawn_with(&mut world, b, 2, vec![inert_enzyme()]);
        let (contacts, field) = contacts_of(&world, first);
        assert_eq!(contacts.len(), 1);
        let contact = contacts[0];
        assert_eq!(contact.cell_id, second);
        let ra = f64::from(world.cells[first].radius);
        let rb = f64::from(world.cells[second].radius);
        let closeness = ((ra + rb + gap - contact.distance) / gap).clamp(0.0, 1.0);
        assert!(closeness > 0.0 && closeness < 1.0);
        let expected = closeness * (rb / contact.distance).asin() / std::f64::consts::PI;
        assert!((contact.share - expected).abs() <= 1.0e-12);
        assert!((field + contact.share - 1.0).abs() <= 1.0e-12);
        let (reverse, _) = contacts_of(&world, second);
        assert_eq!(reverse[0].cell_id, first);
        assert!((reverse[0].distance - contact.distance).abs() <= 1.0e-5);
        assert!((reverse[0].share - contact.share).abs() <= 1.0e-5);
        pair_shares.push(contact.share);
    }
    assert!(
        (pair_shares[0] - pair_shares[1]).abs() <= 1.0e-5,
        "{pair_shares:?}"
    );
    assert!(pair_shares[0] > 0.0 && pair_shares[0] < 1.0 / 6.0);

    let hub = spawn_with(
        &mut world,
        Position::new(10.0, 4.0),
        1,
        vec![inert_enzyme()],
    );
    for index in 0..8 {
        let angle = index as f32 * std::f32::consts::TAU / 8.0;
        let position = Position::new(10.0 + 0.9 * angle.cos(), 4.0 + 0.9 * angle.sin());
        let id = spawn_with(
            &mut world,
            Position::new(1.0 + index as f32 * 2.0, 14.0),
            3,
            vec![inert_enzyme()],
        );
        world.update_cell_position(id, position);
    }
    let (contacts, field) = contacts_of(&world, hub);
    assert_eq!(contacts.len(), 8);
    let total = contacts.iter().map(|contact| contact.share).sum::<f64>();
    assert!((total - 1.0).abs() <= 1.0e-12);
    assert_eq!(field, 0.0);
    assert!(
        contacts
            .windows(2)
            .all(|pair| pair[0].cell_id.index() < pair[1].cell_id.index())
    );
}

#[test]
fn internal_and_surface_reactions_of_one_enzyme_never_exceed_its_capacity() {
    let mut config = small_config("surface-capacity", 12, 10);
    config.founder_energy = 100.0;
    let mut world = World::new(config).unwrap();
    let mut enzyme = surface_enzyme([0.0, 0.0, 0.0, 1.0, 0.0, 0.0], A_ONLY, 3.0, 0.6);
    enzyme.half_saturation = 1.0e-3;
    let actor = spawn_with(&mut world, Position::new(5.0, 5.0), 1, vec![enzyme]);
    for (index, offset) in [(0.0_f32, 0.9_f32), (0.95, 0.0), (0.0, -0.93)]
        .into_iter()
        .enumerate()
    {
        let id = spawn_with(
            &mut world,
            Position::new(5.0 + offset.0, 5.0 + offset.1),
            2 + index as u64,
            vec![inert_enzyme()],
        );
        world.cells[id].boundary[Element::D] += 2.0;
    }
    world.cells[actor].internal_elements[Element::D] = 5.0;
    world.inspect_cell_fluxes(actor, 64).unwrap();
    let radius = world.cells[actor].radius;
    let area = std::f32::consts::PI * radius * radius;
    world.step_cell_fresh(actor);

    let records = world
        .flux_log
        .fluxes_newest_first()
        .cloned()
        .collect::<Vec<_>>();
    assert!(
        records
            .iter()
            .any(|record| record.site == crate::cell::ReactionSite::Internal)
    );
    assert!(
        records
            .iter()
            .any(|record| record.site == crate::cell::ReactionSite::Field)
    );
    assert_eq!(
        records
            .iter()
            .filter(|record| matches!(record.site, crate::cell::ReactionSite::Cell(_)))
            .count(),
        3
    );
    let capacity = f64::from(enzyme.rate)
        * world.config.dt_seconds
        * f64::from(area / crate::spatial::REFERENCE_CELL_AREA);
    let requested = records
        .iter()
        .map(|record| f64::from(record.requested_extent))
        .sum::<f64>();
    assert!(
        requested <= capacity * (1.0 + 1.0e-6),
        "{requested} > {capacity}"
    );
    assert!(requested >= 0.9 * capacity, "{requested} vs {capacity}");
    world.check_invariants().unwrap();
}

#[test]
fn lineage_labels_never_change_the_simulation() {
    let run = |distinct_lineages: bool| {
        let mut config = small_config("lineage-blind", 16, 12);
        config.founder_energy = 4.0;
        let mut world = World::new(config).unwrap();
        let mut genome_rng = crate::rng::Rng::from_seed_str("lineage-blind-genomes");
        for index in 0..12_u64 {
            let mut genome = Genome::random_founder(&mut genome_rng, 0.0);
            genome.enzymes[0].exposure = 0.5;
            genome.mutation_rate = 0.5;
            genome.lineage_id = LineageId(if distinct_lineages { 100 + index } else { 7 });
            let position = Position::new(
                4.0 + (index % 4) as f32 * 0.92,
                4.0 + (index / 4) as f32 * 0.92,
            );
            let id = world
                .spawn_cell_with_genome_at_position(position, genome)
                .unwrap();
            for element in ELEMENT_ORDER {
                world.cells[id].boundary[element] *= 1.6;
            }
        }
        for _ in 0..600 {
            world.step();
        }
        world.check_invariants().unwrap();
        world
    };
    let same = run(false);
    let distinct = run(true);
    let physics = |world: &World| {
        let mut cells = world.cells.iter().cloned().collect::<Vec<_>>();
        for cell in &mut cells {
            cell.lineage_id = LineageId(0);
            cell.genome.lineage_id = LineageId(0);
        }
        let mut interaction = world.interaction_counters();
        interaction.cross_lineage_boundary_digested = 0.0;
        bincode::serialize(&(
            cells,
            &world.rng,
            &world.element_fields,
            &world.enval,
            world.energy_ledger,
            world.enval_ledger,
            world.reaction_counters,
            interaction,
            world.contact_pair_count,
            world.death_count,
            world.birth_count,
        ))
        .unwrap()
    };
    assert!(same.interaction_counters().boundary_digested > 0.0);
    assert_eq!(
        same.interaction_counters().cross_lineage_boundary_digested,
        0.0
    );
    assert!(
        distinct
            .interaction_counters()
            .cross_lineage_boundary_digested
            > 0.0
    );
    let stats = same.stats();
    assert!(
        stats.fissions > 0,
        "{} cells, {} deaths, {} lyses",
        stats.live_cell_count,
        stats.deaths,
        stats.lyses
    );
    assert_eq!(physics(&same), physics(&distinct));
}

fn drift_world(seed: &str, columns: usize, rows: usize) -> (World, Vec<crate::cell::CellId>) {
    let mut world = World::new(small_config(seed, columns * 3, rows * 3)).unwrap();
    let mut ids = Vec::new();
    for column in 0..columns {
        for row in 0..rows {
            let position = Position::new(column as f32 * 3.0 + 1.5, row as f32 * 3.0 + 1.5);
            ids.push(spawn_with(&mut world, position, 1, vec![inert_enzyme()]));
        }
    }
    (world, ids)
}

fn mean_squared_displacement(world: &mut World, ids: &[crate::cell::CellId], ticks: usize) -> f64 {
    let mut displacement = vec![(0.0_f64, 0.0_f64); ids.len()];
    for _ in 0..ticks {
        let before = ids
            .iter()
            .map(|id| world.cells[*id].position)
            .collect::<Vec<_>>();
        world.drift_cells();
        for (index, id) in ids.iter().enumerate() {
            let step = minimum_image_displacement(
                before[index],
                world.cells[*id].position,
                world.width as f32,
                world.height as f32,
            );
            displacement[index].0 += f64::from(step.x);
            displacement[index].1 += f64::from(step.y);
        }
    }
    displacement.iter().map(|(x, y)| x * x + y * y).sum::<f64>() / ids.len() as f64
}

#[test]
fn drift_mean_squared_displacement_is_four_d_t_and_scales_inversely_with_radius() {
    let (mut world, ids) = drift_world("drift-msd", 24, 24);
    let (small, reference) = ids.split_at(ids.len() / 2);
    for id in small {
        let cell = &mut world.cells[*id];
        for element in ELEMENT_ORDER {
            cell.body[element] *= 0.25;
        }
        cell.refresh_radius(world.config.cell_body.body_density);
    }
    world.refresh_max_cell_radius();
    let ticks = 1_000;
    let elapsed = ticks as f64 * world.config.dt_seconds;
    let diffusivity = f64::from(world.config.cell_motion.diffusivity);
    let radius_ratio =
        f64::from(world.cells[reference[0]].radius) / f64::from(world.cells[small[0]].radius);
    assert!((radius_ratio - 2.0).abs() < 1.0e-3);

    let mut reference_world = world.clone();
    let reference_msd = mean_squared_displacement(&mut reference_world, reference, ticks);
    let small_msd = mean_squared_displacement(&mut world, small, ticks);
    let expected = 4.0 * diffusivity * elapsed;
    assert!(
        (reference_msd / expected - 1.0).abs() <= 0.15,
        "reference MSD {reference_msd} vs {expected}"
    );
    let ratio = small_msd / reference_msd;
    assert!((ratio - 2.0).abs() <= 0.35, "half-radius MSD ratio {ratio}");
    world.check_invariants().unwrap();
    reference_world.check_invariants().unwrap();
}

#[test]
fn drift_draws_two_normals_per_cell_and_nothing_when_disabled() {
    let (mut world, ids) = drift_world("drift-rng", 4, 3);
    let positions = |world: &World| {
        ids.iter()
            .map(|id| world.cells[*id].position)
            .collect::<Vec<_>>()
    };

    let mut still = world.clone();
    still.config.cell_motion.diffusivity = 0.0;
    let before = positions(&still);
    let mut untouched = still.rng.clone();
    for _ in 0..10 {
        still.drift_cells();
    }
    assert_eq!(positions(&still), before);
    assert_eq!(
        still.rng.next_f64().to_bits(),
        untouched.next_f64().to_bits()
    );

    let mut expected = world.rng.clone();
    for _ in 0..2 * ids.len() {
        expected.gaussian(0.0, 1.0);
    }
    world.drift_cells();
    assert_ne!(positions(&world), before);
    assert_eq!(
        world.rng.next_f64().to_bits(),
        expected.next_f64().to_bits()
    );
    world.check_invariants().unwrap();
}

#[test]
fn zero_drift_reproduces_the_motionless_world() {
    let mut config = small_config("drift-off", 24, 18);
    config.cell_motion.diffusivity = 0.0;
    let mut stepped = World::new(config).unwrap();
    stepped.spawn_founder_cells(8).unwrap();
    let mut manual = stepped.clone();
    for _ in 0..300 {
        stepped.step();
        manual.diffuse_element_fields();
        manual.step_cells();
        manual.resolve_overlaps();
        manual.enval_phase();
        manual.advance_time();
    }
    assert_eq!(
        bincode::serialize(&stepped).unwrap(),
        bincode::serialize(&manual).unwrap()
    );
}

#[test]
fn drift_wraps_across_seams_and_keeps_spatial_caches_consistent() {
    let (mut world, ids) = drift_world("drift-seam", 2, 2);
    world.config.cell_motion.diffusivity = 1.0;
    for (index, id) in ids.iter().enumerate() {
        let corner = Position::new(
            if index % 2 == 0 { 0.001 } else { 5.999 },
            if index / 2 == 0 { 0.001 } else { 5.999 },
        );
        world.update_cell_position(*id, corner);
    }
    world.check_invariants().unwrap();
    let mut crossings = 0;
    for _ in 0..500 {
        let before = ids
            .iter()
            .map(|id| world.cells[*id].position)
            .collect::<Vec<_>>();
        world.drift_cells();
        for (index, id) in ids.iter().enumerate() {
            let after = world.cells[*id].position;
            assert!(after.x >= 0.0 && after.x < 6.0 && after.y >= 0.0 && after.y < 6.0);
            if (after.x - before[index].x).abs() > 3.0 || (after.y - before[index].y).abs() > 3.0 {
                crossings += 1;
            }
        }
        world.check_invariants().unwrap();
    }
    assert!(crossings > 0);
    world.rebuild_spatial_index();
    world.check_invariants().unwrap();
}

#[test]
fn interaction_stats_report_scripted_digestion_exposure_and_contacts() {
    let mut config = only_a_config("interaction-stats", 12, 10);
    config.catalyst_upkeep_per_sec = 0.0;
    config.founder_energy = 50.0;
    config.cell_motion.diffusivity = 0.0;
    let mut world = World::new(config).unwrap();
    let digester = spawn_with(
        &mut world,
        Position::new(3.0, 4.0),
        1,
        vec![surface_enzyme(E_ONLY, A_ONLY, 2.0, 1.0)],
    );
    let prey = spawn_with(&mut world, Position::new(3.9, 4.0), 2, vec![inert_enzyme()]);
    let sibling = spawn_with(&mut world, Position::new(3.0, 4.9), 1, vec![inert_enzyme()]);
    let mut low = inert_enzyme();
    low.exposure = 0.04;
    let mut lower = inert_enzyme();
    lower.exposure = 0.02;
    let loner = spawn_with(&mut world, Position::new(9.0, 8.0), 3, vec![low, lower]);
    let boundary = world.cells[prey].boundary.total() as f32;
    world.cells[prey].boundary = ElementAmounts::new([0.0, 0.0, 0.0, 0.0, boundary * 0.8, 0.0]);
    world.cells[sibling].boundary =
        ElementAmounts::new([boundary * 0.6, 0.0, 0.0, 0.0, boundary * 0.6, 0.0]);

    let fresh = world.stats();
    assert_eq!(
        fresh.interaction,
        crate::stats::InteractionCounters::default()
    );
    assert_eq!(fresh.exposed_cell_count, 1);
    assert_eq!(fresh.max_exposure, 1.0);
    assert!((fresh.mean_exposure - 1.06 / 5.0).abs() <= 1.0e-6);
    world.resolve_overlaps();
    let stats = world.compact_stats();
    assert_eq!(stats.contact_pairs, 2);
    assert_eq!(stats.cross_lineage_contact_pairs, 1);

    let inspection = world.inspect_cell(digester).unwrap();
    assert_eq!(
        inspection
            .contacts
            .iter()
            .map(|contact| contact.cell_id)
            .collect::<Vec<_>>(),
        vec![prey.index(), sibling.index()]
    );
    assert_eq!(inspection.contacts[0].lineage_id, LineageId(2));
    assert!(
        inspection
            .contacts
            .iter()
            .all(|contact| contact.share > 0.0)
    );
    let shares = inspection
        .contacts
        .iter()
        .map(|contact| contact.share)
        .sum::<f64>();
    assert!((inspection.field_share + shares - 1.0).abs() <= 1.0e-12);
    assert!(world.inspect_cell(loner).unwrap().contacts.is_empty());
    assert_eq!(world.inspect_cell(loner).unwrap().field_share, 1.0);
    assert!((world.inspect_lineage(LineageId(1)).unwrap().mean_exposure - 0.5).abs() <= 1.0e-9);
    assert!((world.inspect_lineage(LineageId(3)).unwrap().mean_exposure - 0.03).abs() <= 1.0e-6);

    for _ in 0..2_000 {
        for id in [digester, prey, sibling] {
            world.step_cell_fresh(id);
        }
        world.advance_time();
        if world.cell(prey).is_none() {
            break;
        }
    }
    assert!(world.cell(prey).is_none());
    world.resolve_overlaps();
    let full = world.stats();
    let compact = world.compact_stats();
    let counters = full.interaction;
    assert_eq!(counters, world.interaction_counters());
    assert_eq!(compact.interaction, counters);
    assert_eq!(counters.lyses_after_digestion, 1);
    assert_eq!(counters.field_surface_extent, 0.0);
    assert!(counters.cell_surface_extent > 0.0);
    assert!(counters.cross_lineage_boundary_digested > 0.0);
    assert!(counters.boundary_digested > counters.cross_lineage_boundary_digested);
    assert_eq!(full.contact_pairs, 1);
    assert_eq!(full.cross_lineage_contact_pairs, 0);
    assert_eq!(full.exposed_cell_count, 1);
    assert_eq!(compact.mean_exposure, full.mean_exposure);
    assert!((full.mean_exposure - 1.06 / 4.0).abs() <= 1.0e-6);
    world.check_invariants().unwrap();
}

#[test]
fn cells_without_exposed_enzymes_never_compute_contact_shares() {
    let mut world = World::new(small_config("unexposed-skip", 24, 18)).unwrap();
    world.spawn_founder_cells(12).unwrap();
    world.step_many(200);
    let counters = world.operation_counters();
    assert!(counters.cell_steps > 2_000);
    assert_eq!(counters.surface_contact_queries, 0);
    assert_eq!(
        world.interaction_counters(),
        crate::stats::InteractionCounters::default()
    );

    let exposed = live_ids(&world)[0];
    world.cells[exposed].genome.enzymes[0].exposure = 0.2;
    let steps_before = world.operation_counters().cell_steps;
    world.step();
    let counters = world.operation_counters();
    assert!(counters.cell_steps - steps_before > 1);
    assert_eq!(counters.surface_contact_queries, 1);
}

mod allocation_counter {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    pub struct CountingAllocator;

    thread_local! {
        static ARMED: Cell<bool> = const { Cell::new(false) };
        static COUNT: Cell<usize> = const { Cell::new(0) };
    }

    fn record() {
        let _ = ARMED.try_with(|armed| {
            if armed.get() {
                let _ = COUNT.try_with(|count| count.set(count.get() + 1));
            }
        });
    }

    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            record();
            unsafe { System.alloc(layout) }
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            record();
            unsafe { System.alloc_zeroed(layout) }
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            record();
            unsafe { System.realloc(ptr, layout, new_size) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            unsafe { System.dealloc(ptr, layout) }
        }
    }

    pub fn count_allocations(run: impl FnOnce()) -> usize {
        COUNT.with(|count| count.set(0));
        ARMED.with(|armed| armed.set(true));
        run();
        ARMED.with(|armed| armed.set(false));
        COUNT.with(|count| count.get())
    }
}

#[global_allocator]
static ALLOCATOR: allocation_counter::CountingAllocator = allocation_counter::CountingAllocator;

#[test]
fn steady_state_steps_allocate_nothing_per_reaction_query_or_pair() {
    let mut config = small_config("steady-state-allocations", 48, 36);
    config.catalyst_upkeep_per_sec = 0.0;
    let mut world = World::new(config).unwrap();
    let slow_growth = |genome: &mut Genome| {
        for engine in &mut genome.enzymes[1..] {
            engine.body_fraction = 0.02;
            engine.boundary_fraction = 0.02;
        }
    };
    for index in 0..80 {
        let mut genome = Genome::random_founder(&mut world.rng, world.avg_enval);
        world.config.founder_energy = 50.0;
        genome.maintenance_cost_per_area = 0.0;
        genome.enzymes[0].exposure = 0.05;
        slow_growth(&mut genome);
        let position = Position::new(
            (index % 10) as f32 * 4.3 + 1.0,
            (index / 10) as f32 * 4.1 + 1.0,
        );
        world
            .spawn_cell_with_genome_at_position(position, genome)
            .unwrap();
        if index % 2 == 0 {
            let mut genome = Genome::random_founder(&mut world.rng, world.avg_enval);
            genome.lineage_id = LineageId(900 + index as u64);
            genome.maintenance_cost_per_area = 0.0;
            slow_growth(&mut genome);
            let neighbour = Position::new(position.x + 0.91, position.y);
            let _ = world.spawn_cell_with_genome_at_position(neighbour, genome);
        }
    }
    world.step_many(150);
    let cells_before = world.cell_count();
    let counters_before = world.operation_counters();
    let interaction_before = world.interaction_counters();

    let allocations = allocation_counter::count_allocations(|| world.step_many(100));

    let counters = world.operation_counters().saturating_delta(counters_before);
    let interaction = world.interaction_counters();
    assert!(counters.reactions_succeeded > 10_000, "{counters:?}");
    assert!(interaction.field_surface_extent > interaction_before.field_surface_extent);
    assert!(interaction.cell_surface_extent > interaction_before.cell_surface_extent);
    assert!(counters.surface_contact_queries > 0);
    assert!(counters.surface_contact_queries < counters.cell_steps);
    assert!(world.contact_pair_count > 0);
    assert!(counters.overlap_candidates > 0 || counters.spatial_candidate_checks > 0);
    assert_eq!(counters.cell_divisions, 0);
    assert_eq!(world.cell_count(), cells_before);
    assert_eq!(
        allocations, 0,
        "steady-state steps allocated {allocations} times"
    );
    world.check_invariants().unwrap();
}

fn brute_force_pairs(world: &World, gap: f32) -> Vec<(u32, u32)> {
    let (width, height) = (world.width as f32, world.height as f32);
    let mut pairs = Vec::new();
    for slot_a in 0..world.cells.len() {
        for slot_b in slot_a + 1..world.cells.len() {
            let (a, b) = (world.cells.at_slot(slot_a), world.cells.at_slot(slot_b));
            if toroidal_distance_squared(a.position, b.position, width, height)
                <= (a.radius + b.radius + gap).powi(2)
            {
                pairs.push((slot_a as u32, slot_b as u32));
            }
        }
    }
    pairs
}

fn brute_force_step(world: &mut World) {
    world.diffuse_element_fields();
    world.contact_list_current = false;
    world.step_cells();
    world.drift_cells();
    world.refresh_max_cell_radius();
    let gap = world.config.surface.contact_gap;
    let contacts = brute_force_pairs(world, gap);
    world.contact_pair_count = contacts.len() as u64;
    world.cross_lineage_contact_pair_count = contacts
        .iter()
        .filter(|(a, b)| {
            world.cells.at_slot(*a as usize).lineage_id
                != world.cells.at_slot(*b as usize).lineage_id
        })
        .count() as u64;
    if world.cells.len() >= 2 {
        for pass in 0..super::OVERLAP_RELAXATION_PASSES {
            world.contact_pairs = brute_force_pairs(world, if pass == 0 { gap } else { 0.0 });
            world.contact_displacements.clear();
            world.contact_displacements.resize(world.cells.len(), 0.0);
            let corrected = world.overlap_pass();
            world.sync_spatial_index();
            if !corrected {
                break;
            }
        }
    }
    world.contact_list_current = false;
    world.enval_phase();
    world.advance_time();
}

#[test]
fn shared_contact_list_matches_brute_force_overlap_statistics_and_catalysis() {
    let mut world = World::new(small_config("shared-contact-list", 20, 16)).unwrap();
    world.config.founder_energy = 20.0;
    let mut index = 0;
    for column in 0..7 {
        for row in 0..6 {
            let mut genome = Genome::random_founder(&mut world.rng, world.avg_enval);
            genome.lineage_id = LineageId(1 + (column * 6 + row) as u64 % 3);
            for engine in &mut genome.enzymes[1..] {
                engine.body_fraction = 0.02;
                engine.boundary_fraction = 0.02;
            }
            if index % 2 == 0 {
                genome.enzymes[0].exposure = 0.2;
                genome.enzymes[0].reactants = ElementAmounts::new([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
                genome.enzymes[0].products = ElementAmounts::new([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
            }
            let position = Position::new(
                3.0 + column as f32 * 0.93 + (row % 2) as f32 * 0.4,
                3.0 + row as f32 * 0.88,
            );
            world
                .spawn_cell_with_genome_at_position(position, genome)
                .unwrap();
            index += 1;
        }
    }
    world.config.cell_motion.diffusivity = 0.5;
    let mut brute = world.clone();
    let mut overlapping_ticks = 0;
    let mut contact_sum = 0;
    let mut loose_ticks = 0;
    for tick in 0..200 {
        if tick % 20 == 10 {
            for target in [&mut world, &mut brute] {
                let ids = live_ids(target);
                let (mover, anchor) = (ids[tick % ids.len()], ids[(tick + 7) % ids.len()]);
                let anchor_position = target.cells[anchor].position;
                target.cells[mover].position =
                    Position::new(anchor_position.x + 0.05, anchor_position.y)
                        .wrapped(target.width as f32, target.height as f32)
                        .unwrap();
                target.rebuild_spatial_index();
            }
        }
        world.step();
        brute_force_step(&mut brute);
        assert!(world.contact_list_current);
        assert_eq!(
            world.contact_pair_count, brute.contact_pair_count,
            "tick {tick}"
        );
        assert_eq!(
            world.cross_lineage_contact_pair_count,
            brute.cross_lineage_contact_pair_count
        );
        for (listed, reference) in world.cells.iter().zip(brute.cells.iter()) {
            assert_eq!(listed.id, reference.id);
            assert_eq!(listed.position, reference.position, "tick {tick}");
            assert_eq!(listed.energy.to_bits(), reference.energy.to_bits());
        }
        contact_sum += world.contact_pair_count;
        if !world.contact_loose_extras.is_empty() {
            loose_ticks += 1;
        }
        if world.operation_counters.overlap_corrections > 0 {
            overlapping_ticks += 1;
        }
        world.operation_counters.overlap_corrections = 0;
    }
    assert_eq!(world.reaction_counters.divisions, 0);
    assert!(
        contact_sum > 200 * 20,
        "{contact_sum} contacts over the run"
    );
    assert!(loose_ticks >= 10, "{loose_ticks} ticks with loose cells");
    assert!(
        overlapping_ticks > 20,
        "{overlapping_ticks} ticks with overlaps"
    );
    assert!(world.interaction_counters.cell_surface_extent > 0.0);
    assert_eq!(
        world.interaction_counters.cell_surface_extent.to_bits(),
        brute.interaction_counters.cell_surface_extent.to_bits()
    );
    world.operation_counters = Default::default();
    brute.operation_counters = Default::default();
    assert_eq!(
        crate::snapshot::to_bytes(&world).unwrap(),
        crate::snapshot::to_bytes(&brute).unwrap()
    );
}

#[test]
fn overlap_mechanics_are_symmetric_wrapped_and_do_not_consume_rng() {
    let mut world = World::new(only_a_config("overlap-mechanics", 10, 8)).unwrap();
    let first_genome = Genome::random_founder(&mut world.rng, world.avg_enval);
    let first = world
        .spawn_cell_with_genome_at_position(Position::new(0.1, 2.0), first_genome)
        .unwrap();
    let second_genome = Genome::random_founder(&mut world.rng, world.avg_enval);
    let second = world
        .spawn_cell_with_genome_at_position(Position::new(1.1, 2.0), second_genome)
        .unwrap();
    world.cells[second].position = Position::new(9.8, 2.0);
    world.rebuild_spatial_index();
    let before = toroidal_distance_squared(
        world.cells[first].position,
        world.cells[second].position,
        world.width as f32,
        world.height as f32,
    );
    let first_before = world.cells[first].position;
    let second_before = world.cells[second].position;
    let mut expected_rng = world.rng.clone();
    world.resolve_overlaps();
    let first_movement = minimum_image_displacement(
        first_before,
        world.cells[first].position,
        world.width as f32,
        world.height as f32,
    );
    let second_movement = minimum_image_displacement(
        second_before,
        world.cells[second].position,
        world.width as f32,
        world.height as f32,
    );
    let after = toroidal_distance_squared(
        world.cells[first].position,
        world.cells[second].position,
        world.width as f32,
        world.height as f32,
    );
    assert!(after > before);
    assert!((first_movement.x + second_movement.x).abs() <= 1.0e-6);
    assert!((first_movement.y + second_movement.y).abs() <= 1.0e-6);
    for cell_id in [first, second] {
        let position = world.cells[cell_id].position;
        assert!(position.x >= 0.0 && position.x < world.width as f32);
        assert!(position.y >= 0.0 && position.y < world.height as f32);
    }
    assert_eq!(
        world.rng.next_f64().to_bits(),
        expected_rng.next_f64().to_bits()
    );
    world.cells[first].position = Position::new(4.0, 4.0);
    world.cells[second].position = Position::new(4.0, 4.0);
    world.rebuild_spatial_index();
    let mut replay = world.clone();
    world.resolve_overlaps();
    replay.resolve_overlaps();
    assert_eq!(world.cells[first].position, replay.cells[first].position);
    assert_eq!(world.cells[second].position, replay.cells[second].position);
    assert!(
        toroidal_distance_squared(
            world.cells[first].position,
            world.cells[second].position,
            world.width as f32,
            world.height as f32,
        ) > 0.0
    );
    world.check_invariants().unwrap();
    replay.check_invariants().unwrap();
}
