mod common;

use common::*;
use microcosmcore::snapshot::{self, SnapshotError};
use microcosmcore::{
    Config, ElementAmounts, Position, RenderBuffers, SNAPSHOT_VERSION, TileId, World,
    load_from_path, save_to_path,
};

const GOLDEN_FINGERPRINT: &str = "893a85bdd472ea31";

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
fn the_golden_fingerprint_pins_the_trajectory() {
    let mut world = golden_world();
    world.step_many(2_000);
    assert_eq!(
        format!("{:016x}", world.state_fingerprint()),
        GOLDEN_FINGERPRINT,
    );
}

#[test]
fn the_same_seed_makes_the_same_world_and_different_seeds_differ() {
    let mut first = founded_world("determinism", 32, 24, 10);
    let mut second = founded_world("determinism", 32, 24, 10);
    let mut other = founded_world("determinism-other", 32, 24, 10);
    assert_eq!(first.state_fingerprint(), second.state_fingerprint());
    assert_ne!(first.state_fingerprint(), other.state_fingerprint());
    for _ in 0..300 {
        first.step();
        second.step();
        other.step();
        assert_eq!(first.state_fingerprint(), second.state_fingerprint());
    }
    assert_ne!(first.state_fingerprint(), other.state_fingerprint());
    assert_eq!(first.stats(), second.stats());
    assert_eq!(
        snapshot::to_bytes(&first).unwrap(),
        snapshot::to_bytes(&second).unwrap()
    );
}

#[test]
fn step_many_step_and_step_profiled_agree() {
    let mut many = founded_world("stepping", 32, 24, 10);
    let mut single = many.clone();
    let mut profiled = many.clone();
    many.step_many(300);
    for _ in 0..300 {
        single.step();
        let profile = profiled.step_profiled();
        assert!(profile.total >= profile.cell_step);
    }
    assert_eq!(many.state_fingerprint(), single.state_fingerprint());
    assert_eq!(many.state_fingerprint(), profiled.state_fingerprint());
    assert_eq!(many.stats(), single.stats());
    assert_eq!(many.stats(), profiled.stats());
    assert_eq!(many.tick_count(), 300);
}

#[test]
fn observing_never_changes_the_trajectory() {
    let mut plain = founded_world("observation", 24, 18, 10);
    let mut observed = plain.clone();
    let mut buffers = RenderBuffers::default();
    let mut watched = None;
    let mut recorded_fluxes = false;
    for tick in 0..500 {
        if tick % 10 == 0 {
            if let Some(log) = watched.and_then(|id| observed.inspect_cell_fluxes(id, 16)) {
                recorded_fluxes |= log.flux_count > 0;
            }
            let _ = observed.stats();
            let _ = observed.compact_stats();
            let _ = observed.inspect_tile(TileId(tick % observed.tile_count()));
            let ids = observed.cells().map(|cell| cell.id).collect::<Vec<_>>();
            if let Some(id) = ids.get(tick / 10 % ids.len().max(1)).copied() {
                let _ = observed.inspect_cell(id);
                let _ = observed.inspect_cell_detail(id, 16);
                let _ = observed.inspect_cell_fluxes(id, 16);
                watched = Some(id);
            }
            let _ = observed.list_lineages(8);
            observed.write_render_buffers(&mut buffers);
        }
        plain.step();
        observed.step();
        assert_eq!(
            plain.state_fingerprint(),
            observed.state_fingerprint(),
            "tick {tick}"
        );
    }
    assert!(observed.cell_count() > 0);
    assert!(recorded_fluxes, "the watched cells never logged a flux");
    assert_eq!(
        snapshot::to_bytes(&plain).unwrap(),
        snapshot::to_bytes(&observed).unwrap()
    );
}

#[test]
fn snapshots_round_trip() {
    let mut world = founded_world("snapshot-round-trip", 24, 18, 10);
    world.step_many(200);
    let bytes = snapshot::to_bytes(&world).unwrap();
    let restored = snapshot::from_bytes(&bytes).unwrap();
    assert_eq!(restored.state_fingerprint(), world.state_fingerprint());
    assert_eq!(restored.stats(), world.stats());
    restored.check_invariants().unwrap();

    let path = std::env::temp_dir().join(format!(
        "microcosm-tests-{}.{}",
        std::process::id(),
        microcosmcore::SNAPSHOT_EXTENSION
    ));
    save_to_path(&world, &path).unwrap();
    let loaded = load_from_path(&path);
    std::fs::remove_file(&path).unwrap();
    let loaded = loaded.unwrap();
    assert_eq!(loaded.state_fingerprint(), world.state_fingerprint());
    assert_eq!(loaded.stats(), world.stats());
    assert!(matches!(load_from_path(&path), Err(SnapshotError::Io(_))));
}

#[test]
fn a_resumed_snapshot_continues_the_same_trajectory() {
    let mut uninterrupted = golden_world();
    uninterrupted.step_many(300);
    let mut interrupted = golden_world();
    interrupted.step_many(150);
    let mut resumed = snapshot::from_bytes(&snapshot::to_bytes(&interrupted).unwrap()).unwrap();
    assert_eq!(resumed.state_fingerprint(), interrupted.state_fingerprint());
    resumed.step_many(150);
    interrupted.step_many(150);
    assert_eq!(
        interrupted.state_fingerprint(),
        uninterrupted.state_fingerprint()
    );
    assert_eq!(
        resumed.state_fingerprint(),
        uninterrupted.state_fingerprint()
    );
}

#[test]
fn snapshots_carry_and_check_their_version() {
    assert_eq!(SNAPSHOT_VERSION, microcosmcore::VERSION);
    let world = founded_world("snapshot-version", 12, 10, 3);
    let bytes = snapshot::to_bytes(&world).unwrap();
    let length = SNAPSHOT_VERSION.len();
    assert_eq!(bytes[..8], (length as u64).to_le_bytes());
    assert_eq!(&bytes[8..8 + length], SNAPSHOT_VERSION.as_bytes());

    let mut tampered = bytes.clone();
    tampered[8] ^= 0x01;
    assert!(matches!(
        snapshot::from_bytes(&tampered),
        Err(SnapshotError::UnsupportedVersion { .. })
    ));
    assert!(matches!(
        snapshot::from_bytes(&bytes[..bytes.len() / 2]),
        Err(SnapshotError::Codec(_))
    ));
    assert!(snapshot::from_bytes(&bytes).is_ok());
}

#[test]
fn snapshot_size_follows_live_cells_not_history() {
    let populate = |rounds: usize| {
        let mut config = small_config("snapshot-churn", 40, 30);
        config.founder_energy = 0.0;
        let mut world = World::new(config).unwrap();
        let genome = inert_genome("snapshot-churn");
        let grid = |world: &mut World| {
            for column in 0..10 {
                for row in 0..10 {
                    let position = Position::new(column as f32 * 4.0 + 0.5, row as f32 * 3.0 + 0.5);
                    world
                        .spawn_cell_with_genome_at_position(position, genome.clone())
                        .unwrap();
                }
            }
        };
        for _ in 0..rounds {
            grid(&mut world);
            world.step();
            assert_eq!(world.cell_count(), 0);
        }
        grid(&mut world);
        world.check_invariants().unwrap();
        world
    };
    let quiet = populate(0);
    let churned = populate(200);
    assert_eq!(quiet.cell_count(), 100);
    assert_eq!(churned.cell_count(), 100);
    assert_eq!(churned.compact_stats().deaths, 20_000);
    let quiet_size = snapshot::to_bytes(&quiet).unwrap().len() as f64;
    let churned_size = snapshot::to_bytes(&churned).unwrap().len() as f64;
    assert!(
        (churned_size - quiet_size).abs() <= 0.05 * quiet_size,
        "{churned_size} vs {quiet_size} bytes"
    );
}

type Breakage = (&'static str, fn(&mut Config));

#[test]
fn config_validation_rejects_impossible_worlds() {
    let base = small_config("config-validation", 16, 12);
    base.validate().unwrap();
    Config::default().validate().unwrap();

    let mut zero_rates = base.clone();
    zero_rates.enval_diffusivity_per_second = 0.0;
    zero_rates.element_fields.diffusivities = ElementAmounts::ZERO;
    zero_rates.catalyst_upkeep_per_sec = 0.0;
    zero_rates.founder_energy = 0.0;
    zero_rates.enval_landscape.relaxation_per_second = 0.0;
    zero_rates.enval_recharge.rate_per_second = 0.0;
    zero_rates.cell_motion.diffusivity = 0.0;
    zero_rates.element_fields.initial_amounts = ElementAmounts::ZERO;
    zero_rates.element_fields.heterogeneity = 0.0;
    zero_rates.validate().unwrap();
    World::new(zero_rates).unwrap();

    let breakages: Vec<Breakage> = vec![
        ("width", |c| c.width = 0),
        ("height", |c| c.height = 0),
        ("tile count", |c| {
            c.width = usize::MAX;
            c.height = 2;
        }),
        ("dt zero", |c| c.dt_seconds = 0.0),
        ("dt negative", |c| c.dt_seconds = -0.01),
        ("dt nan", |c| c.dt_seconds = f64::NAN),
        ("enval diffusivity negative", |c| {
            c.enval_diffusivity_per_second = -1.0
        }),
        ("enval diffusivity nan", |c| {
            c.enval_diffusivity_per_second = f32::NAN
        }),
        ("enval diffusivity over one per tick", |c| {
            c.enval_diffusivity_per_second = 101.0
        }),
        ("element diffusivity negative", |c| {
            c.element_fields.diffusivities[microcosmcore::Element::C] = -0.1
        }),
        ("element diffusivity nan", |c| {
            c.element_fields.diffusivities[microcosmcore::Element::A] = f32::NAN
        }),
        ("element diffusivity over one per tick", |c| {
            c.element_fields.diffusivities[microcosmcore::Element::F] = 100.5
        }),
        ("permeability zero", |c| c.membrane_permeability = 0.0),
        ("permeability nan", |c| c.membrane_permeability = f32::NAN),
        ("upkeep negative", |c| c.catalyst_upkeep_per_sec = -0.01),
        ("upkeep infinite", |c| {
            c.catalyst_upkeep_per_sec = f64::INFINITY
        }),
        ("founder energy negative", |c| c.founder_energy = -1.0),
        ("founder energy nan", |c| c.founder_energy = f64::NAN),
        ("landscape magnitude zero", |c| {
            c.enval_landscape.magnitude = 0.0
        }),
        ("landscape magnitude nan", |c| {
            c.enval_landscape.magnitude = f32::NAN
        }),
        ("landscape scale", |c| c.enval_landscape.scale = 1.5),
        ("landscape relaxation", |c| {
            c.enval_landscape.relaxation_per_second = -0.1
        }),
        ("recharge negative", |c| {
            c.enval_recharge.rate_per_second = -0.1
        }),
        ("recharge nan", |c| {
            c.enval_recharge.rate_per_second = f32::NAN
        }),
        ("drift negative", |c| c.cell_motion.diffusivity = -0.1),
        ("drift infinite", |c| {
            c.cell_motion.diffusivity = f32::INFINITY
        }),
        ("contact gap zero", |c| c.surface.contact_gap = 0.0),
        ("contact gap negative", |c| c.surface.contact_gap = -0.1),
        ("element amount negative", |c| {
            c.element_fields.initial_amounts[microcosmcore::Element::B] = -0.5
        }),
        ("element amount nan", |c| {
            c.element_fields.initial_amounts[microcosmcore::Element::D] = f32::NAN
        }),
        ("heterogeneity one", |c| {
            c.element_fields.heterogeneity = 1.0
        }),
        ("heterogeneity negative", |c| {
            c.element_fields.heterogeneity = -0.1
        }),
        ("heterogeneity scale", |c| {
            c.element_fields.heterogeneity_scale = 1.0
        }),
        ("body density zero", |c| c.cell_body.body_density = 0.0),
        ("boundary density negative", |c| {
            c.cell_body.boundary_density = -0.2
        }),
        ("permeability sigma zero", |c| {
            c.cell_body.permeability_sigma = 0.0
        }),
        ("assembly half saturation", |c| {
            c.cell_body.assembly_half_saturation = 0.0
        }),
        ("assembly energy", |c| {
            c.cell_body.assembly_energy_per_unit = f64::NAN
        }),
    ];
    for (name, breakage) in breakages {
        let mut config = base.clone();
        breakage(&mut config);
        assert!(config.validate().is_err(), "{name} was accepted");
        assert!(World::new(config).is_err(), "{name} made a world");
    }
}
