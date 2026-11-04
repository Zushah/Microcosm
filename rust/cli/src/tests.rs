use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use microcosmcore::{Config, SNAPSHOT_EXTENSION, World};

use crate::args::{Command, OutputDestination, StatsMode, parse_args};
use crate::output::{
    StatsEmissionState, StatsInterval, collect_stats_for_output, resolve_output_requests,
    write_csv_record,
};
use crate::stats_table::{
    COMPACT_TEXT, FULL_TEXT, STAT_FIELDS, csv_header, format_is_valid, has_field, json_record,
    render_text, template_placeholders,
};
use crate::viability::{milestone_column, population_milestone_failures, run_viability_world};

const TEST_TIMESTAMP: &str = "05152026_164635";

fn unique_test_dir(name: &str) -> PathBuf {
    let dir = env::temp_dir().join(format!(
        "microcosm_cli_{name}_{}_{}",
        std::process::id(),
        TEST_TIMESTAMP
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn default_csv_and_snapshot_paths_share_timestamp() {
    let cwd = unique_test_dir("default_paths");
    let outputs = resolve_output_requests(
        Some(&OutputDestination::Default),
        Some(&OutputDestination::Default),
        &cwd,
        TEST_TIMESTAMP,
    )
    .unwrap();
    assert_eq!(
        outputs.csv.unwrap(),
        cwd.join("out").join("05152026_164635.csv")
    );
    assert_eq!(
        outputs.snapshot_out.unwrap(),
        cwd.join("out").join("05152026_164635.micosm")
    );
    assert!(cwd.join("out").is_dir());
}

#[test]
fn no_outputs_do_not_create_default_out_directory() {
    let cwd = unique_test_dir("no_outputs");
    let outputs = resolve_output_requests(None, None, &cwd, TEST_TIMESTAMP).unwrap();
    assert!(outputs.csv.is_none());
    assert!(outputs.snapshot_out.is_none());
    assert!(!cwd.join("out").exists());
}

#[test]
fn explicit_paths_append_expected_extensions_when_missing() {
    let cwd = unique_test_dir("explicit_extensions");
    let csv_path = cwd.join("logs").join("run");
    let snapshot_path = cwd.join("snapshots").join("final");
    let outputs = resolve_output_requests(
        Some(&OutputDestination::Path(csv_path)),
        Some(&OutputDestination::Path(snapshot_path)),
        &cwd,
        TEST_TIMESTAMP,
    )
    .unwrap();
    assert_eq!(outputs.csv.unwrap(), cwd.join("logs").join("run.csv"));
    assert_eq!(
        outputs.snapshot_out.unwrap(),
        cwd.join("snapshots").join("final.micosm")
    );
}

#[test]
fn explicit_directory_paths_use_timestamp_inside_directory() {
    let cwd = unique_test_dir("directory_paths");
    let csv_dir = cwd.join("csvs");
    let snapshot_dir = cwd.join("snapshots");
    fs::create_dir_all(&csv_dir).unwrap();
    fs::create_dir_all(&snapshot_dir).unwrap();
    let outputs = resolve_output_requests(
        Some(&OutputDestination::Path(csv_dir.clone())),
        Some(&OutputDestination::Path(snapshot_dir.clone())),
        &cwd,
        TEST_TIMESTAMP,
    )
    .unwrap();
    assert_eq!(outputs.csv.unwrap(), csv_dir.join("05152026_164635.csv"));
    assert_eq!(
        outputs.snapshot_out.unwrap(),
        snapshot_dir.join("05152026_164635.micosm")
    );
}

#[test]
fn optional_output_flags_parse_without_paths() {
    let args = vec![
        "run".to_owned(),
        "--csv".to_owned(),
        "--snapshot-out".to_owned(),
        "--steps".to_owned(),
        "1".to_owned(),
    ];
    let command = parse_args(args).unwrap().unwrap();
    let Command::Run(options) = command else {
        panic!("expected run command");
    };
    assert_eq!(options.csv, Some(OutputDestination::Default));
    assert_eq!(options.snapshot_out, Some(OutputDestination::Default));
    assert_eq!(options.steps, 1);
}

#[test]
fn initial_cells_is_the_canonical_population_flag() {
    let args = vec![
        "run".to_owned(),
        "--initial-cells".to_owned(),
        "7".to_owned(),
        "--steps".to_owned(),
        "1".to_owned(),
    ];
    let command = parse_args(args).unwrap().unwrap();
    let Command::Run(options) = command else {
        panic!("expected run command");
    };
    assert_eq!(options.initial_cells, Some(7));
    assert_eq!(options.config.initial_founder_count, 7);

    let args = vec![
        "bench".to_owned(),
        "--initial-cells".to_owned(),
        "11".to_owned(),
    ];
    let command = parse_args(args).unwrap().unwrap();
    let Command::Bench(options) = command else {
        panic!("expected bench command");
    };
    assert_eq!(options.initial_cells, 11);
    assert_eq!(options.config.initial_founder_count, 11);
}

#[test]
fn legacy_cli_aliases_are_not_accepted() {
    let args = vec!["run".to_owned(), "--founders".to_owned(), "3".to_owned()];
    assert!(parse_args(args).is_err());

    let args = vec!["bench".to_owned(), "--max-steps".to_owned(), "3".to_owned()];
    assert!(parse_args(args).is_err());

    let args = vec!["--steps".to_owned(), "1".to_owned()];
    assert!(parse_args(args).is_err());
}

#[test]
fn run_help_does_not_execute_default_run() {
    let args = vec!["run".to_owned(), "--help".to_owned()];
    let command = parse_args(args).unwrap();
    assert!(command.is_none());
}

#[test]
fn bench_help_does_not_execute_default_bench() {
    let args = vec!["bench".to_owned(), "--help".to_owned()];
    let command = parse_args(args).unwrap();
    assert!(command.is_none());
}

#[test]
fn current_snapshot_extension_is_micosm() {
    assert_eq!(SNAPSHOT_EXTENSION, "micosm");
}

#[test]
fn csv_header_is_wide_and_stable_enough_for_observability() {
    let header = csv_header();
    let columns = header.split(',').collect::<Vec<_>>();
    assert!(columns.contains(&"occupancy_fraction"));
    assert!(columns.contains(&"extracellular_a"));
    assert!(columns.contains(&"intracellular_f"));
    assert!(columns.contains(&"system_a"));
    assert!(columns.contains(&"total_element_amount"));
    assert!(columns.contains(&"executed_metabolic_flux"));
    assert!(columns.contains(&"uptake_flux"));
    assert!(columns.contains(&"secretion_flux"));
    assert!(columns.contains(&"rx_successes"));
    assert!(columns.contains(&"interval_cell_steps"));
    for column in [
        "mean_exposure",
        "max_exposure",
        "exposed_cells",
        "contact_pairs",
        "cross_lineage_contact_pairs",
        "field_surface_extent",
        "cell_surface_extent",
        "boundary_digested",
        "cross_lineage_boundary_digested",
        "lyses_after_digestion",
        "mixed_polarity_cells",
        "mean_enval_optimum_spread",
        "mean_niche_mismatch",
    ] {
        assert!(columns.contains(&column), "{column}");
    }
    assert!(columns.len() > 80);
}

#[test]
fn csv_rows_have_one_value_per_header_column_including_ledgers() {
    let mut world = World::new(Config {
        seed: "cli-csv-columns".to_owned(),
        width: 16,
        height: 12,
        ..Config::default()
    })
    .unwrap();
    world.spawn_founder_cells(4).unwrap();
    world.step_many(20);
    let mut state = StatsEmissionState::default();
    let first = world.stats();
    state.capture_interval(&first, Instant::now());
    world.step_many(5);
    let stats = world.stats();
    let interval = state.capture_interval(&stats, Instant::now());
    let mut buffer = Vec::new();
    write_csv_record(&mut buffer, &stats, &interval).unwrap();
    let row = String::from_utf8(buffer).unwrap();
    let header = csv_header();
    let header = header.split(',').collect::<Vec<_>>();
    assert_eq!(row.trim_end().split(',').count(), header.len());
    for column in [
        "energy_enval_harvest",
        "energy_pump_cost",
        "energy_ledger_residual",
        "renewable_coverage",
        "enval_source_inflow",
        "recharge_energy",
        "leak_flux",
    ] {
        assert!(header.contains(&column), "missing {column}");
    }
    for removed in ["cell_records", "dead_cells"] {
        assert!(!header.contains(&removed), "stale {removed}");
    }
    assert!(!header.iter().any(|column| column.contains("without_food")));
}

#[test]
fn trace_off_is_accepted_and_other_trace_modes_error() {
    let args = vec![
        "bench".to_owned(),
        "--trace".to_owned(),
        "off".to_owned(),
        "--steps".to_owned(),
        "1".to_owned(),
    ];
    assert!(parse_args(args).unwrap().is_some());

    let args = vec![
        "bench".to_owned(),
        "--trace".to_owned(),
        "reactions".to_owned(),
    ];
    assert!(parse_args(args).is_err());
}

#[test]
fn compact_output_is_default_and_verbose_stats_are_opt_in() {
    let args = vec!["run".to_owned(), "--steps".to_owned(), "1".to_owned()];
    let command = parse_args(args).unwrap().unwrap();
    let Command::Run(options) = command else {
        panic!("expected run command");
    };
    assert_eq!(options.stats_mode, StatsMode::Compact);

    let args = vec!["run".to_owned(), "--verbose-stats".to_owned()];
    let command = parse_args(args).unwrap().unwrap();
    let Command::Run(options) = command else {
        panic!("expected run command");
    };
    assert_eq!(options.stats_mode, StatsMode::Full);

    let args = vec!["bench".to_owned(), "--steps".to_owned(), "1".to_owned()];
    let command = parse_args(args).unwrap().unwrap();
    let Command::Bench(options) = command else {
        panic!("expected bench command");
    };
    assert_eq!(options.stats_mode, StatsMode::Compact);
}

#[test]
fn compact_stats_collection_skips_percentiles_unless_output_needs_full_stats() {
    let mut world = World::new(Config {
        seed: "cli-compact-stats".to_owned(),
        width: 4,
        height: 4,
        ..Config::default()
    })
    .unwrap();
    for x in 0..4 {
        for y in 0..4 {
            let tile = world.tile_id(x, y).unwrap();
            world.set_tile_enval(tile, 0.25).unwrap();
        }
    }

    let compact = collect_stats_for_output(&world, false, StatsMode::Compact, false);
    assert_eq!(compact.enval_p50.to_bits(), 0.0_f32.to_bits());

    let verbose = collect_stats_for_output(&world, false, StatsMode::Full, false);
    assert_eq!(verbose.enval_p50.to_bits(), 0.25_f32.to_bits());

    let csv = collect_stats_for_output(&world, true, StatsMode::Compact, false);
    assert_eq!(csv.enval_p50.to_bits(), 0.25_f32.to_bits());

    let json = collect_stats_for_output(&world, false, StatsMode::Compact, true);
    assert_eq!(json.enval_p50.to_bits(), 0.25_f32.to_bits());
}

#[test]
fn viability_defaults_and_flags_parse() {
    let command = parse_args(vec!["viability".to_owned()]).unwrap().unwrap();
    let Command::Viability(options) = command else {
        panic!("expected viability command");
    };
    assert_eq!(options.seeds, vec!["42", "1", "2", "3", "7"]);
    assert_eq!(options.steps, 60_000);
    assert!(options.jobs >= 1);

    let args = [
        "viability",
        "--seeds",
        "5, 9",
        "--steps",
        "300",
        "--jobs",
        "2",
        "--json",
    ]
    .map(str::to_owned);
    let command = parse_args(args).unwrap().unwrap();
    let Command::Viability(options) = command else {
        panic!("expected viability command");
    };
    assert_eq!(options.seeds, vec!["5", "9"]);
    assert_eq!(options.steps, 300);
    assert_eq!(options.jobs, 2);
    assert!(options.json);
    assert!(parse_args(["viability", "--jobs", "0"].map(str::to_owned)).is_err());
}

#[test]
fn until_cells_parses_and_stops_a_run_at_the_population_target() {
    let args = ["run", "--steps", "500", "--until-cells", "4000"].map(str::to_owned);
    let Command::Run(options) = parse_args(args).unwrap().unwrap() else {
        panic!("expected run command");
    };
    assert_eq!(options.until_cells, Some(4000));
    assert!(parse_args(["run", "--until-cells"].map(str::to_owned)).is_err());
}

#[test]
fn environment_flags_parse_for_every_world_command() {
    for command in ["run", "bench", "viability"] {
        let args = [
            command,
            "--permeability",
            "6",
            "--catalyst-upkeep",
            "0.02",
            "--landscape-magnitude",
            "1.5",
            "--landscape-scale",
            "30",
            "--landscape-relaxation",
            "0.4",
            "--enval-diffusivity",
            "12",
            "--recharge-rate",
            "0.2",
            "--heterogeneity",
            "0.3",
            "--heterogeneity-scale",
            "32",
            "--drift",
            "0.5",
            "--contact-gap",
            "0.25",
        ]
        .map(str::to_owned);
        let config = match parse_args(args).unwrap().unwrap() {
            Command::Run(options) => options.config,
            Command::Bench(options) => options.config,
            Command::Viability(options) => options.config,
            Command::Inspect(_) => panic!("unexpected inspect command"),
        };
        assert_eq!(config.membrane_permeability, 6.0);
        assert_eq!(config.catalyst_upkeep_per_sec, 0.02);
        assert_eq!(config.enval_landscape.magnitude, 1.5);
        assert_eq!(config.enval_landscape.scale, 30.0);
        assert_eq!(config.enval_landscape.relaxation_per_second, 0.4);
        assert_eq!(config.enval_diffusivity_per_second, 12.0);
        assert_eq!(config.enval_recharge.rate_per_second, 0.2);
        assert_eq!(config.element_fields.heterogeneity, 0.3);
        assert_eq!(config.element_fields.heterogeneity_scale, 32.0);
        assert_eq!(config.cell_motion.diffusivity, 0.5);
        assert_eq!(config.surface.contact_gap, 0.25);
    }
}

#[test]
fn viability_results_do_not_depend_on_thread_count() {
    let config = Config {
        width: 24,
        height: 18,
        initial_founder_count: 6,
        ..Config::default()
    };
    let run = |seed: &str| {
        let mut config = config.clone();
        config.seed = seed.to_owned();
        run_viability_world(config, 300, true)
    };
    let sequential = ["a", "b"].map(run);
    let parallel = std::thread::scope(|scope| {
        let handles = ["a", "b"].map(|seed| scope.spawn(move || run(seed)));
        handles.map(|handle| handle.join().unwrap())
    });
    for (left, right) in sequential.iter().zip(&parallel) {
        assert_eq!(left.final_live, right.final_live);
        assert_eq!(left.peak_live, right.peak_live);
        assert_eq!(
            left.chemical_harvest.to_bits(),
            right.chemical_harvest.to_bits()
        );
        assert_eq!(left.final_totals, right.final_totals);
    }
}

#[test]
fn population_milestones_fail_runs_short_of_either_threshold_and_print() {
    assert!(population_milestone_failures(&[None, None]).is_empty());
    assert!(population_milestone_failures(&[Some(5_000), None]).is_empty());
    assert!(population_milestone_failures(&[Some(5_000), Some(10_000)]).is_empty());
    assert_eq!(
        population_milestone_failures(&[Some(4_999), None]),
        vec!["4999 live cells at tick 20000 is below 5000".to_owned()]
    );
    assert_eq!(
        population_milestone_failures(&[Some(20_000), Some(9_999)]),
        vec!["9999 live cells at tick 60000 is below 10000".to_owned()]
    );
    assert_eq!(population_milestone_failures(&[Some(1), Some(2)]).len(), 2);
    assert_eq!(milestone_column(None), "-");
    assert_eq!(milestone_column(Some(17_584)), "17584");

    let config = Config {
        seed: "milestones".to_owned(),
        width: 24,
        height: 18,
        initial_founder_count: 6,
        ..Config::default()
    };
    let short = run_viability_world(config, 200, true);
    assert_eq!(short.milestone_cells, [None, None]);
    assert!(
        short.mean_niche_mismatch > 0.0,
        "the final report uses full stats"
    );
    assert!(
        short
            .failures
            .iter()
            .all(|failure| !failure.contains("live cells at tick"))
    );
}

#[test]
fn drift_flag_reaches_the_world_and_zero_drift_is_reproducible() {
    let parse = |drift: &str| {
        let args = [
            "viability",
            "--width",
            "24",
            "--height",
            "18",
            "--initial-cells",
            "6",
            "--drift",
            drift,
        ]
        .map(str::to_owned);
        let Command::Viability(mut options) = parse_args(args).unwrap().unwrap() else {
            panic!("expected viability command");
        };
        options.config.seed = "cli-drift".to_owned();
        options.config
    };
    let still = parse("0");
    assert_eq!(still.cell_motion.diffusivity, 0.0);
    let first = run_viability_world(still.clone(), 300, true);
    let second = run_viability_world(still, 300, true);
    let moving = run_viability_world(parse("0.2"), 300, true);
    assert_eq!(first.final_totals, second.final_totals);
    assert_eq!(
        first.peak_cross_lineage_contacts,
        second.peak_cross_lineage_contacts
    );
    assert_ne!(first.final_totals, moving.final_totals);
}

#[test]
fn fingerprint_flags_parse() {
    let args = ["run", "--steps", "3", "--fingerprint"].map(str::to_owned);
    let Command::Run(options) = parse_args(args).unwrap().unwrap() else {
        panic!("expected run command");
    };
    assert!(options.fingerprint);
    assert_eq!(options.fingerprint_every, 0);

    let args = ["run", "--fingerprint-every", "100"].map(str::to_owned);
    let Command::Run(options) = parse_args(args).unwrap().unwrap() else {
        panic!("expected run command");
    };
    assert!(options.fingerprint);
    assert_eq!(options.fingerprint_every, 100);
}

#[test]
fn stats_mode_and_json_flags_parse() {
    let args = vec![
        "run".to_owned(),
        "--stats-mode".to_owned(),
        "full".to_owned(),
        "--json".to_owned(),
        "--steps".to_owned(),
        "1".to_owned(),
    ];
    let command = parse_args(args).unwrap().unwrap();
    let Command::Run(options) = command else {
        panic!("expected run command");
    };
    assert_eq!(options.stats_mode, StatsMode::Full);
    assert!(options.json);

    let args = vec![
        "bench".to_owned(),
        "--verbose-stats".to_owned(),
        "--profile-json".to_owned(),
    ];
    let command = parse_args(args).unwrap().unwrap();
    let Command::Bench(options) = command else {
        panic!("expected bench command");
    };
    assert_eq!(options.stats_mode, StatsMode::Full);
    assert!(options.profile);
    assert!(options.profile_json);
}

fn sample_stats() -> (microcosmcore::WorldStats, StatsInterval) {
    let mut world = World::new(Config {
        seed: "cli-stat-table".to_owned(),
        width: 16,
        height: 12,
        ..Config::default()
    })
    .unwrap();
    world.spawn_founder_cells(4).unwrap();
    world.step_many(20);
    let mut state = StatsEmissionState::default();
    let first = world.stats();
    state.capture_interval(&first, Instant::now());
    world.step_many(5);
    let stats = world.stats();
    let interval = state.capture_interval(&stats, Instant::now());
    (stats, interval)
}

#[test]
fn stat_table_names_and_json_paths_are_unique() {
    let mut names = STAT_FIELDS
        .iter()
        .map(|field| field.name)
        .collect::<Vec<_>>();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), STAT_FIELDS.len());
    let mut paths = STAT_FIELDS
        .iter()
        .filter_map(|field| field.json)
        .collect::<Vec<_>>();
    paths.sort_unstable();
    let count = paths.len();
    paths.dedup();
    assert_eq!(paths.len(), count);
}

#[test]
fn text_lines_name_only_existing_stats_and_valid_formats() {
    for template in COMPACT_TEXT.iter().chain(FULL_TEXT) {
        for (name, spec) in template_placeholders(template) {
            assert!(has_field(name), "unknown stat {name}");
            assert!(format_is_valid(spec), "bad format {spec} for {name}");
        }
    }
}

#[test]
fn text_lines_render_every_placeholder() {
    let (stats, interval) = sample_stats();
    let compact = render_text(COMPACT_TEXT[0], &stats, &interval);
    assert!(compact.starts_with(&format!("tick={} time=", stats.tick_count)));
    assert!(compact.contains(&format!(" size={}x{} ", stats.width, stats.height)));
    assert!(compact.contains(" d_cells=+") || compact.contains(" d_cells=-"));
    assert!(!compact.contains('{') && !compact.contains('}'));
    for template in FULL_TEXT {
        let line = render_text(template, &stats, &interval);
        assert!(!line.contains('{') && !line.contains('}'), "{line}");
    }
}

#[test]
fn json_record_nests_paths_and_keeps_the_published_keys() {
    let (stats, interval) = sample_stats();
    let record = json_record(&stats, &interval);
    assert_eq!(record["tick"], stats.tick_count);
    assert_eq!(record["population"], stats.live_cell_count);
    assert_eq!(
        record["chemistry"]["system"]["F"],
        stats.system_element_amounts[5]
    );
    assert_eq!(
        record["metabolism"]["attempts"],
        stats.reaction_counters.attempts
    );
    assert_eq!(
        record["energy_ledger"]["assembly_cost"],
        stats.energy_ledger.assembly_cost
    );
    assert_eq!(record["interval"]["cell_steps"], interval.cell_steps);
    assert_eq!(
        record["niche"]["mixed_polarity_cells"],
        stats.mixed_polarity_cells
    );
    for section in [
        "chemistry",
        "metabolism",
        "interaction",
        "niche",
        "energy_ledger",
        "enval_ledger",
        "enval",
        "interval",
    ] {
        assert!(record[section].is_object(), "{section}");
    }
}
