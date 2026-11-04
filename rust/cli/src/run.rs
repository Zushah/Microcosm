use std::io::Write;
use std::process::ExitCode;
use std::time::Instant;

use microcosmcore::{World, WorldStats, load_from_path, save_to_path};

use crate::args::{InspectOptions, RunOptions, should_check_invariants};
use crate::bench::{CliProfile, print_profile_summary};
use crate::output::{
    StatsEmissionState, StatsInterval, StatsOutput, emit_world_stats_profiled, open_csv,
    print_text_lines, resolve_run_outputs,
};
use crate::stats_table::FULL_TEXT;

pub fn run_command(options: RunOptions) -> ExitCode {
    let outputs = match resolve_run_outputs(&options) {
        Ok(outputs) => outputs,
        Err(err) => {
            eprintln!("failed to resolve output paths: {err}");
            return ExitCode::from(1);
        }
    };

    let mut world = match initialize_run_world(&options) {
        Ok(world) => world,
        Err(message) => {
            eprintln!("{message}");
            return ExitCode::from(1);
        }
    };

    let mut csv = match open_csv(outputs.csv.as_ref()) {
        Ok(csv) => csv,
        Err(err) => {
            eprintln!("failed to open CSV output: {err}");
            return ExitCode::from(1);
        }
    };

    let mut profile = CliProfile::default();
    let mut emission_state = StatsEmissionState::default();
    let output = StatsOutput {
        print: !options.quiet,
        profile: options.profile,
        mode: options.stats_mode,
        json: options.json,
    };
    let wall_start = Instant::now();

    if options.check_invariants {
        let start = Instant::now();
        if let Err(err) = world.check_invariants() {
            eprintln!("invariant failure at tick {}: {err}", world.tick_count());
            return ExitCode::from(1);
        }
        if options.profile {
            profile.invariants += start.elapsed();
        }
    }

    if let Err(err) =
        emit_world_stats_profiled(&world, &mut csv, output, &mut profile, &mut emission_state)
    {
        eprintln!("failed to emit stats: {err}");
        return ExitCode::from(1);
    }
    let mut completed_steps = options.steps;
    for step_index in 1..=options.steps {
        if options.profile {
            profile.add_step(world.step_profiled());
        } else {
            world.step();
        }

        if options.fingerprint_every > 0 && world.tick_count() % options.fingerprint_every == 0 {
            println!(
                "tick={} fingerprint={:016x}",
                world.tick_count(),
                world.state_fingerprint()
            );
        }

        if should_check_invariants(
            options.check_invariants,
            options.check_invariants_every,
            step_index,
        ) {
            let start = Instant::now();
            if let Err(err) = world.check_invariants() {
                eprintln!("invariant failure at tick {}: {err}", world.tick_count());
                return ExitCode::from(1);
            }
            if options.profile {
                profile.invariants += start.elapsed();
            }
        }

        if options.stats_every > 0
            && step_index.is_multiple_of(options.stats_every)
            && let Err(err) = emit_world_stats_profiled(
                &world,
                &mut csv,
                output,
                &mut profile,
                &mut emission_state,
            )
        {
            eprintln!("failed to emit stats: {err}");
            return ExitCode::from(1);
        }

        if options
            .until_cells
            .is_some_and(|target| world.cell_count() >= target)
        {
            if !options.quiet {
                println!(
                    "until_cells reached tick={} cells={}",
                    world.tick_count(),
                    world.cell_count()
                );
            }
            completed_steps = step_index;
            break;
        }
    }

    if completed_steps > 0
        && (options.stats_every == 0 || !completed_steps.is_multiple_of(options.stats_every))
        && let Err(err) =
            emit_world_stats_profiled(&world, &mut csv, output, &mut profile, &mut emission_state)
    {
        eprintln!("failed to emit stats: {err}");
        return ExitCode::from(1);
    }

    if options.fingerprint {
        println!("fingerprint={:016x}", world.state_fingerprint());
    }

    if let Some(mut csv) = csv {
        let start = Instant::now();
        if let Err(err) = csv.flush() {
            eprintln!("failed to flush CSV output: {err}");
            return ExitCode::from(1);
        }
        if options.profile {
            profile.csv_flush += start.elapsed();
        }
        if !options.quiet
            && let Some(path) = outputs.csv.as_ref()
        {
            println!("csv_out={}", path.display());
        }
    }

    if let Some(path) = outputs.snapshot_out.as_ref() {
        let start = Instant::now();
        if let Err(err) = save_to_path(&world, path) {
            eprintln!("failed to write snapshot {}: {err}", path.display());
            return ExitCode::from(1);
        }
        if options.profile {
            profile.snapshot_io += start.elapsed();
        }
        if !options.quiet {
            println!("snapshot_out={}", path.display());
        }
    }

    if options.profile {
        print_profile_summary(
            completed_steps,
            wall_start.elapsed(),
            profile,
            options.profile_json || options.json,
        );
    }

    ExitCode::SUCCESS
}

pub fn inspect_command(options: InspectOptions) -> ExitCode {
    let world = match load_from_path(&options.snapshot) {
        Ok(world) => world,
        Err(err) => {
            eprintln!(
                "failed to read snapshot {}: {err}",
                options.snapshot.display()
            );
            return ExitCode::from(1);
        }
    };
    let stats = world.stats();
    print_inspect_summary(&world, &stats);
    match world.check_invariants() {
        Ok(()) => println!("invariants=ok"),
        Err(err) => {
            println!("invariants=failed error={err}");
            return ExitCode::from(1);
        }
    }
    ExitCode::SUCCESS
}

fn print_inspect_summary(world: &World, stats: &WorldStats) {
    println!("snapshot_summary:");
    print_text_lines(FULL_TEXT, stats, &StatsInterval::default());
    println!("config:");
    println!(
        "  seed={} size={}x{} dt_seconds={:.6}",
        world.config().seed,
        world.width(),
        world.height(),
        world.config().dt_seconds,
    );
    println!("enzyme_histogram:");
    for count in 1..stats.enzyme_count_histogram.len() {
        println!(
            "  enzymes={} cells={}",
            count, stats.enzyme_count_histogram[count]
        );
    }
    println!("lineages_top:");
    for (lineage, counters) in world.top_lineages(10) {
        let share = if stats.live_cell_count > 0 {
            counters.population as f64 / stats.live_cell_count as f64
        } else {
            0.0
        };
        println!(
            "  lineage={} population={} share={:.4} births={} deaths={}",
            lineage.raw(),
            counters.population,
            share,
            counters.births,
            counters.deaths
        );
    }
}

fn initialize_run_world(options: &RunOptions) -> Result<World, String> {
    let mut world = if let Some(path) = options.snapshot_in.as_ref() {
        load_from_path(path)
            .map_err(|err| format!("failed to load snapshot {}: {err}", path.display()))?
    } else {
        World::new(options.config.clone())
            .map_err(|err| format!("failed to initialize world: {err}"))?
    };

    if options.snapshot_in.is_none() {
        let target = options
            .initial_cells
            .unwrap_or(options.config.initial_founder_count);
        spawn_cells(&mut world, target)?;
    } else if let Some(extra_cells) = options.initial_cells {
        spawn_cells(&mut world, extra_cells)?;
    }

    if let Some(energy) = options.initial_energy {
        world
            .set_all_live_cell_energy(energy)
            .map_err(|err| format!("failed to set initial cell energy: {err}"))?;
    }
    Ok(world)
}

fn spawn_cells(world: &mut World, count: usize) -> Result<(), String> {
    match world.spawn_founder_cells(count) {
        Ok(spawned) if spawned == count => Ok(()),
        Ok(spawned) => {
            eprintln!(
                "warning: requested {} cells, spawned {} before the world filled",
                count, spawned
            );
            Ok(())
        }
        Err(err) => Err(format!("failed to spawn cells: {err}")),
    }
}
