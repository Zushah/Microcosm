#![recursion_limit = "256"]

use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use chrono::Local;
use microcosmcore::{
    Config, ELEMENT_COUNT, SNAPSHOT_EXTENSION, StepProfile, World, WorldStats, load_from_path,
    renewable_coverage, save_to_path,
};

const CSV_EXTENSION: &str = "csv";
const DEFAULT_OUTPUT_DIR: &str = "out";
const VIABILITY_DEFAULT_SEEDS: [&str; 5] = ["42", "1", "2", "3", "7"];
const VIABILITY_DEFAULT_STEPS: u64 = 60_000;
const VIABILITY_SAMPLE_EVERY: u64 = 100;
const VIABILITY_INVARIANTS_EVERY: u64 = 5_000;
const VIABILITY_PROGRESS_EVERY: u64 = 5_000;
const VIABILITY_MIN_LIVE_AFTER_TICK: u64 = 1_000;
const VIABILITY_POPULATION_CAP: usize = 60_000;
const VIABILITY_MIN_RENEWABLE_COVERAGE: f64 = 0.25;

#[derive(Debug, Clone)]
enum Command {
    Run(RunOptions),
    Bench(BenchOptions),
    Inspect(InspectOptions),
    Viability(ViabilityOptions),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum OutputDestination {
    Default,
    Path(PathBuf),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StatsMode {
    Compact,
    Full,
}

impl Default for StatsMode {
    fn default() -> Self {
        Self::Compact
    }
}

#[derive(Debug, Clone)]
struct RunOptions {
    config: Config,
    initial_cells: Option<usize>,
    initial_energy: Option<f64>,
    steps: u64,
    stats_every: u64,
    check_invariants: bool,
    check_invariants_every: u64,
    csv: Option<OutputDestination>,
    snapshot_in: Option<PathBuf>,
    snapshot_out: Option<OutputDestination>,
    profile: bool,
    profile_json: bool,
    quiet: bool,
    stats_mode: StatsMode,
    json: bool,
    predation_override: Option<bool>,
    until_cells: Option<usize>,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self {
            config: Config::default(),
            initial_cells: None,
            initial_energy: None,
            steps: 0,
            stats_every: 100,
            check_invariants: false,
            check_invariants_every: 1,
            csv: None,
            snapshot_in: None,
            snapshot_out: None,
            profile: false,
            profile_json: false,
            quiet: false,
            stats_mode: StatsMode::Compact,
            json: false,
            predation_override: None,
            until_cells: None,
        }
    }
}

#[derive(Debug, Clone)]
struct BenchOptions {
    config: Config,
    initial_cells: usize,
    initial_energy: Option<f64>,
    steps: u64,
    stats_every: u64,
    check_invariants_every: u64,
    profile: bool,
    profile_json: bool,
    quiet: bool,
    stats_mode: StatsMode,
    json: bool,
}

impl Default for BenchOptions {
    fn default() -> Self {
        let config = Config::default();
        Self {
            initial_cells: config.initial_founder_count,
            config,
            initial_energy: None,
            steps: 1000,
            stats_every: 100,
            check_invariants_every: 0,
            profile: false,
            profile_json: false,
            quiet: false,
            stats_mode: StatsMode::Compact,
            json: false,
        }
    }
}

#[derive(Debug, Clone)]
struct InspectOptions {
    snapshot: PathBuf,
}

#[derive(Debug, Clone)]
struct ViabilityOptions {
    config: Config,
    seeds: Vec<String>,
    steps: u64,
    jobs: usize,
    json: bool,
    quiet: bool,
}

impl Default for ViabilityOptions {
    fn default() -> Self {
        Self {
            config: Config::default(),
            seeds: VIABILITY_DEFAULT_SEEDS
                .iter()
                .map(|seed| (*seed).to_owned())
                .collect(),
            steps: VIABILITY_DEFAULT_STEPS,
            jobs: std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1),
            json: false,
            quiet: false,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct ViabilityResult {
    seed: String,
    passed: bool,
    failures: Vec<String>,
    final_tick: u64,
    extinction_tick: Option<u64>,
    min_live_after_warmup: Option<usize>,
    peak_live: usize,
    peak_tick: u64,
    final_live: usize,
    enval_harvest: f64,
    chemical_harvest: f64,
    recharge_energy: f64,
    renewable_coverage: f64,
    max_ledger_residual: f64,
    initial_totals: [f64; ELEMENT_COUNT],
    final_totals: [f64; ELEMENT_COUNT],
    wall_seconds: f64,
}

impl ViabilityResult {
    fn relative_total_drift(&self) -> f64 {
        let initial = self.initial_totals.iter().sum::<f64>();
        let final_total = self.final_totals.iter().sum::<f64>();
        (final_total - initial).abs() / initial.abs().max(1.0e-12)
    }
}

#[derive(Debug, Clone, Default)]
struct ResolvedOutputPaths {
    csv: Option<PathBuf>,
    snapshot_out: Option<PathBuf>,
}

#[derive(Clone, Debug, Default)]
struct CliProfile {
    step: StepProfile,
    step_count: u64,
    min_step: Option<Duration>,
    max_step: Duration,
    step_durations_ms: Vec<f64>,
    stats_output: Duration,
    invariants: Duration,
    snapshot_io: Duration,
    csv_flush: Duration,
}

impl CliProfile {
    fn add_step(&mut self, profile: StepProfile) {
        self.step_count = self.step_count.saturating_add(1);
        self.min_step = Some(
            self.min_step
                .map_or(profile.total, |min| min.min(profile.total)),
        );
        self.max_step = self.max_step.max(profile.total);
        self.step_durations_ms.push(duration_ms(profile.total));
        self.step.add_assign(profile);
    }
}

#[derive(Clone, Debug, Default)]
struct StatsEmissionState {
    previous_stats: Option<WorldStats>,
    previous_wall: Option<Instant>,
}

impl StatsEmissionState {
    fn capture_interval(&mut self, current: &WorldStats, now: Instant) -> StatsInterval {
        let interval = match (&self.previous_stats, self.previous_wall) {
            (Some(previous), Some(previous_wall)) => {
                StatsInterval::between(previous, current, now.duration_since(previous_wall))
            }
            _ => StatsInterval::initial(current),
        };
        self.previous_stats = Some(current.clone());
        self.previous_wall = Some(now);
        interval
    }
}

#[derive(Clone, Debug, Default)]
struct StatsInterval {
    tick_delta: u64,
    sim_seconds_delta: f64,
    wall_seconds: f64,
    population_delta: i64,
    births: u64,
    deaths: u64,
    predation_events: u64,
    cells_consumed: u64,
    divisions: u64,
    reaction_attempts: u64,
    reaction_successes: u64,
    executed_metabolic_flux: f64,
    uptake_flux: f64,
    leak_flux: f64,
    secretion_flux: f64,
    cell_steps: u64,
    enzyme_attempts: u64,
    steps_per_sec: f64,
    cell_steps_per_sec: f64,
    enzyme_attempts_per_sec: f64,
    reactions_per_sec: f64,
}

impl StatsInterval {
    fn initial(_current: &WorldStats) -> Self {
        Self::default()
    }

    fn between(previous: &WorldStats, current: &WorldStats, wall: Duration) -> Self {
        let wall_seconds = wall.as_secs_f64().max(0.0);
        let safe_wall = wall_seconds.max(1.0e-9);
        let tick_delta = current.tick_count.saturating_sub(previous.tick_count);
        let operations = current
            .operation_counters
            .saturating_delta(previous.operation_counters);
        let reactions = current
            .reaction_counters
            .saturating_delta(previous.reaction_counters);
        let reaction_attempts = reactions.total_attempts();
        let reaction_successes = reactions.total_successes();
        Self {
            tick_delta,
            sim_seconds_delta: current.sim_time_seconds - previous.sim_time_seconds,
            wall_seconds,
            population_delta: current.live_cell_count as i64 - previous.live_cell_count as i64,
            births: current.births.saturating_sub(previous.births),
            deaths: current.deaths.saturating_sub(previous.deaths),
            predation_events: current
                .predation_events
                .saturating_sub(previous.predation_events),
            cells_consumed: current
                .cells_consumed
                .saturating_sub(previous.cells_consumed),
            divisions: reactions.divisions,
            reaction_attempts,
            reaction_successes,
            executed_metabolic_flux: reactions.executed_metabolic_flux,
            uptake_flux: reactions.uptake_flux,
            leak_flux: reactions.leak_flux,
            secretion_flux: reactions.secretion_flux,
            cell_steps: operations.cell_steps,
            enzyme_attempts: operations.metabolic_enzyme_attempts,
            steps_per_sec: tick_delta as f64 / safe_wall,
            cell_steps_per_sec: operations.cell_steps as f64 / safe_wall,
            enzyme_attempts_per_sec: operations.metabolic_enzyme_attempts as f64 / safe_wall,
            reactions_per_sec: reaction_successes as f64 / safe_wall,
        }
    }
}

fn main() -> ExitCode {
    match parse_args(env::args().skip(1)) {
        Ok(Some(Command::Run(options))) => run_command(options),
        Ok(Some(Command::Bench(options))) => bench_command(options),
        Ok(Some(Command::Inspect(options))) => inspect_command(options),
        Ok(Some(Command::Viability(options))) => viability_command(options),
        Ok(None) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("error: {message}");
            print_usage();
            ExitCode::from(2)
        }
    }
}

fn run_command(options: RunOptions) -> ExitCode {
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

    if let Err(err) = emit_world_stats_profiled(
        &world,
        &mut csv,
        !options.quiet,
        options.profile,
        &mut profile,
        &mut emission_state,
        options.stats_mode,
        options.json,
    ) {
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

        if options.stats_every > 0 && step_index % options.stats_every == 0 {
            if let Err(err) = emit_world_stats_profiled(
                &world,
                &mut csv,
                !options.quiet,
                options.profile,
                &mut profile,
                &mut emission_state,
                options.stats_mode,
                options.json,
            ) {
                eprintln!("failed to emit stats: {err}");
                return ExitCode::from(1);
            }
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
        && (options.stats_every == 0 || completed_steps % options.stats_every != 0)
    {
        if let Err(err) = emit_world_stats_profiled(
            &world,
            &mut csv,
            !options.quiet,
            options.profile,
            &mut profile,
            &mut emission_state,
            options.stats_mode,
            options.json,
        ) {
            eprintln!("failed to emit stats: {err}");
            return ExitCode::from(1);
        }
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
        if !options.quiet {
            if let Some(path) = outputs.csv.as_ref() {
                println!("csv_out={}", path.display());
            }
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

fn bench_command(options: BenchOptions) -> ExitCode {
    let mut world = match World::new(options.config.clone()) {
        Ok(world) => world,
        Err(err) => {
            eprintln!("failed to initialize benchmark world: {err}");
            return ExitCode::from(1);
        }
    };

    let target_cells = options.initial_cells;
    match world.spawn_founder_cells(target_cells) {
        Ok(spawned) if spawned == target_cells => {}
        Ok(spawned) => eprintln!(
            "warning: requested {} initial cells, spawned {} before the world filled",
            target_cells, spawned
        ),
        Err(err) => {
            eprintln!("failed to spawn benchmark cells: {err}");
            return ExitCode::from(1);
        }
    }

    if let Some(energy) = options.initial_energy {
        if let Err(err) = world.set_all_live_cell_energy(energy) {
            eprintln!("failed to set benchmark cell energy: {err}");
            return ExitCode::from(1);
        }
    }

    let mut profile = CliProfile::default();
    let mut emission_state = StatsEmissionState::default();
    let wall_start = Instant::now();
    if !options.quiet {
        emit_bench_stats(
            &world,
            options.profile,
            &mut profile,
            &mut emission_state,
            options.stats_mode,
            options.json,
        );
    }

    for step_index in 1..=options.steps {
        if options.profile {
            profile.add_step(world.step_profiled());
        } else {
            world.step();
        }

        if should_check_invariants(
            options.check_invariants_every > 0,
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

        if !options.quiet && options.stats_every > 0 && step_index % options.stats_every == 0 {
            emit_bench_stats(
                &world,
                options.profile,
                &mut profile,
                &mut emission_state,
                options.stats_mode,
                options.json,
            );
        }
    }

    let elapsed = wall_start.elapsed();
    let stats = world.compact_stats();
    if !options.quiet && options.steps > 0 && options.steps % options.stats_every.max(1) != 0 {
        emit_bench_stats(
            &world,
            options.profile,
            &mut profile,
            &mut emission_state,
            options.stats_mode,
            options.json,
        );
    }
    let seconds = elapsed.as_secs_f64().max(1.0e-9);
    let actual_cell_steps = stats.operation_counters.cell_steps;
    println!(
        "bench steps={} elapsed_sec={:.6} steps_per_sec={:.3} final_cells={} final_elements={:.6} actual_cell_steps={} cell_steps_per_sec={:.3}",
        options.steps,
        seconds,
        options.steps as f64 / seconds,
        stats.live_cell_count,
        stats.total_element_amount,
        actual_cell_steps,
        actual_cell_steps as f64 / seconds,
    );
    if options.profile {
        print_profile_summary(
            options.steps,
            elapsed,
            profile,
            options.profile_json || options.json,
        );
    }

    ExitCode::SUCCESS
}

fn viability_command(options: ViabilityOptions) -> ExitCode {
    if options.seeds.is_empty() {
        eprintln!("viability requires at least one seed");
        return ExitCode::from(2);
    }
    let mut probe = options.config.clone();
    probe.seed = options.seeds[0].clone();
    if let Err(err) = probe.validate() {
        eprintln!("invalid viability config: {err}");
        return ExitCode::from(2);
    }

    let jobs = options.jobs.clamp(1, options.seeds.len());
    let next = AtomicUsize::new(0);
    let results = Mutex::new(vec![None; options.seeds.len()]);
    std::thread::scope(|scope| {
        for _ in 0..jobs {
            scope.spawn(|| {
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    let Some(seed) = options.seeds.get(index) else {
                        break;
                    };
                    let mut config = options.config.clone();
                    config.seed = seed.clone();
                    let result = run_viability_world(config, options.steps, options.quiet);
                    results.lock().expect("viability results lock")[index] = Some(result);
                }
            });
        }
    });
    let results = results
        .into_inner()
        .expect("viability results lock")
        .into_iter()
        .map(|result| result.expect("every viability seed produces a result"))
        .collect::<Vec<_>>();

    if options.json {
        print_viability_json(&results, options.steps);
    } else {
        print_viability_table(&results, options.steps);
    }
    if results.iter().all(|result| result.passed) {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}

fn run_viability_world(config: Config, steps: u64, quiet: bool) -> ViabilityResult {
    let wall_start = Instant::now();
    let mut result = ViabilityResult {
        seed: config.seed.clone(),
        ..ViabilityResult::default()
    };
    let founders = config.initial_founder_count;
    let mut world = match World::new(config) {
        Ok(world) => world,
        Err(err) => {
            result
                .failures
                .push(format!("world creation failed: {err}"));
            return result;
        }
    };
    if let Err(err) = world.spawn_founder_cells(founders) {
        result.failures.push(format!("founder spawn failed: {err}"));
        return result;
    }
    result.initial_totals = world.compact_stats().system_element_amounts;
    result.peak_live = world.cell_count();

    let mut honest = true;
    for tick in 1..=steps {
        world.step();
        let sample = tick % VIABILITY_SAMPLE_EVERY == 0 || tick == steps;
        let check_invariants = tick % VIABILITY_INVARIANTS_EVERY == 0 || tick == steps;
        if !sample && !check_invariants {
            continue;
        }
        let live = world.cell_count();
        result.final_tick = tick;
        result.final_live = live;
        if live > result.peak_live {
            result.peak_live = live;
            result.peak_tick = tick;
        }
        if tick >= VIABILITY_MIN_LIVE_AFTER_TICK {
            result.min_live_after_warmup = Some(
                result
                    .min_live_after_warmup
                    .map_or(live, |min| min.min(live)),
            );
        }
        let residual = world.energy_ledger_residual().abs();
        result.max_ledger_residual = result.max_ledger_residual.max(residual);
        if residual > world.energy_ledger().closure_tolerance() && honest {
            honest = false;
            result.failures.push(format!(
                "energy ledger residual {residual:.3e} exceeds tolerance at tick {tick}"
            ));
        }
        if check_invariants {
            if let Err(err) = world.check_invariants() {
                result
                    .failures
                    .push(format!("invariant failure at tick {tick}: {err}"));
                break;
            }
        }
        if live == 0 {
            result.extinction_tick = Some(tick);
            result.failures.push(format!("extinct at tick {tick}"));
            break;
        }
        if live > VIABILITY_POPULATION_CAP {
            result.failures.push(format!(
                "population {live} exceeds cap {VIABILITY_POPULATION_CAP} at tick {tick}"
            ));
            break;
        }
        if !quiet && tick % VIABILITY_PROGRESS_EVERY == 0 {
            let coverage = renewable_coverage(&world.energy_ledger(), &world.enval_ledger());
            eprintln!(
                "viability seed={} tick={} cells={} peak={} coverage={:.3} wall={:.1}s",
                result.seed,
                tick,
                live,
                result.peak_live,
                coverage,
                wall_start.elapsed().as_secs_f64()
            );
        }
    }

    let energy = world.energy_ledger();
    let enval = world.enval_ledger();
    result.enval_harvest = energy.enval_harvest;
    result.chemical_harvest = energy.chemical_harvest;
    result.recharge_energy = enval.recharge_energy;
    result.renewable_coverage = renewable_coverage(&energy, &enval);
    result.final_totals = world.compact_stats().system_element_amounts;
    if result.extinction_tick.is_none()
        && result.final_tick == steps
        && result.renewable_coverage < VIABILITY_MIN_RENEWABLE_COVERAGE
    {
        result.failures.push(format!(
            "renewable coverage {:.3} is below {VIABILITY_MIN_RENEWABLE_COVERAGE}",
            result.renewable_coverage
        ));
    }
    result.passed = result.failures.is_empty() && result.final_tick == steps;
    result.wall_seconds = wall_start.elapsed().as_secs_f64();
    result
}

fn print_viability_table(results: &[ViabilityResult], steps: u64) {
    println!(
        "{:<6} {:<4} {:>9} {:>8} {:>15} {:>7} {:>11} {:>11} {:>11} {:>8} {:>10} {:>14} {:>8}",
        "seed",
        "pass",
        "extinct",
        "min_live",
        "peak@tick",
        "final",
        "enval_harv",
        "chem_harv",
        "recharge_E",
        "coverage",
        "max_resid",
        "AF_total_drift",
        "wall_s"
    );
    for result in results {
        println!(
            "{:<6} {:<4} {:>9} {:>8} {:>15} {:>7} {:>11.1} {:>11.1} {:>11.1} {:>8.3} {:>10.2e} {:>14.2e} {:>8.1}",
            result.seed,
            if result.passed { "yes" } else { "NO" },
            result
                .extinction_tick
                .map_or_else(|| "-".to_owned(), |tick| tick.to_string()),
            result
                .min_live_after_warmup
                .map_or_else(|| "-".to_owned(), |live| live.to_string()),
            format!("{}@{}", result.peak_live, result.peak_tick),
            result.final_live,
            result.enval_harvest,
            result.chemical_harvest,
            result.recharge_energy,
            result.renewable_coverage,
            result.max_ledger_residual,
            result.relative_total_drift(),
            result.wall_seconds,
        );
        for failure in &result.failures {
            println!("       - {failure}");
        }
    }
    let passed = results.iter().filter(|result| result.passed).count();
    println!(
        "viability {} {}/{} seeds passed over {} ticks",
        if passed == results.len() {
            "PASS"
        } else {
            "FAIL"
        },
        passed,
        results.len(),
        steps
    );
}

fn print_viability_json(results: &[ViabilityResult], steps: u64) {
    let seeds = results
        .iter()
        .map(|result| {
            serde_json::json!({
                "seed": result.seed,
                "passed": result.passed,
                "failures": result.failures,
                "final_tick": result.final_tick,
                "extinction_tick": result.extinction_tick,
                "min_live_after_tick_1000": result.min_live_after_warmup,
                "peak_live": result.peak_live,
                "peak_tick": result.peak_tick,
                "final_live": result.final_live,
                "enval_harvest": result.enval_harvest,
                "chemical_harvest": result.chemical_harvest,
                "recharge_energy": result.recharge_energy,
                "renewable_coverage": result.renewable_coverage,
                "max_ledger_residual": result.max_ledger_residual,
                "initial_element_totals": element_amounts_json(result.initial_totals),
                "final_element_totals": element_amounts_json(result.final_totals),
                "relative_total_element_drift": result.relative_total_drift(),
                "wall_seconds": result.wall_seconds,
            })
        })
        .collect::<Vec<_>>();
    let value = serde_json::json!({
        "viability": {
            "steps": steps,
            "passed": results.iter().all(|result| result.passed),
            "seeds": seeds,
        }
    });
    println!("{}", value);
}

fn inspect_command(options: InspectOptions) -> ExitCode {
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
    print_full_stats(stats, &StatsInterval::default());
    println!("config:");
    println!(
        "  seed={} size={}x{} predation_enabled={} dt_seconds={:.6}",
        world.config().seed,
        world.width(),
        world.height(),
        world.predation_enabled(),
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

    if let Some(enabled) = options.predation_override {
        world.set_predation_enabled(enabled);
    }

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

fn parse_args<I>(args: I) -> Result<Option<Command>, String>
where
    I: IntoIterator<Item = String>,
{
    let mut args = args.into_iter().peekable();
    match args.peek().map(String::as_str) {
        None => {
            print_usage();
            Ok(None)
        }
        Some("-h" | "--help") => {
            print_usage();
            Ok(None)
        }
        Some("run") => {
            args.next();
            parse_run(args)
        }
        Some("bench") => {
            args.next();
            parse_bench(args)
        }
        Some("viability") => {
            args.next();
            parse_viability(args)
        }
        Some("inspect") => {
            args.next();
            let snapshot = args
                .next()
                .ok_or_else(|| "inspect requires a snapshot path".to_owned())?;
            if args.next().is_some() {
                return Err("inspect accepts exactly one snapshot path".to_owned());
            }
            Ok(Some(Command::Inspect(InspectOptions {
                snapshot: PathBuf::from(snapshot),
            })))
        }
        Some(other) => Err(format!("unrecognized command '{other}'")),
    }
}

fn parse_run<I>(args: std::iter::Peekable<I>) -> Result<Option<Command>, String>
where
    I: Iterator<Item = String>,
{
    let mut options = RunOptions::default();
    if parse_common_run_args(args, &mut options)? {
        Ok(Some(Command::Run(options)))
    } else {
        Ok(None)
    }
}

fn parse_common_run_args<I>(
    mut args: std::iter::Peekable<I>,
    options: &mut RunOptions,
) -> Result<bool, String>
where
    I: Iterator<Item = String>,
{
    while let Some(arg) = args.next() {
        if let Some(value) = arg.strip_prefix("--csv=") {
            options.csv = Some(OutputDestination::Path(PathBuf::from(value)));
            continue;
        }
        if let Some(value) = arg.strip_prefix("--snapshot-out=") {
            options.snapshot_out = Some(OutputDestination::Path(PathBuf::from(value)));
            continue;
        }
        match arg.as_str() {
            "-h" | "--help" => {
                print_usage();
                return Ok(false);
            }
            "--seed" => options.config.seed = next_value(&mut args, "--seed")?,
            "--width" => options.config.width = parse_value(&mut args, "--width")?,
            "--height" => options.config.height = parse_value(&mut args, "--height")?,
            "--initial-cells" => {
                let value = parse_value(&mut args, "--initial-cells")?;
                options.initial_cells = Some(value);
                options.config.initial_founder_count = value;
            }
            "--initial-energy" => {
                options.initial_energy = Some(parse_value(&mut args, "--initial-energy")?)
            }
            "--steps" => options.steps = parse_value(&mut args, "--steps")?,
            "--stats-every" => options.stats_every = parse_value(&mut args, "--stats-every")?,
            "--dt-seconds" => options.config.dt_seconds = parse_value(&mut args, "--dt-seconds")?,
            "--enval-alpha" => {
                options.config.enval_diffusion_alpha = parse_value(&mut args, "--enval-alpha")?;
            }
            "--check-invariants" => options.check_invariants = true,
            "--check-invariants-every" => {
                options.check_invariants = true;
                options.check_invariants_every =
                    parse_value(&mut args, "--check-invariants-every")?;
            }
            "--csv" => {
                options.csv = Some(match optional_value(&mut args) {
                    Some(value) => OutputDestination::Path(PathBuf::from(value)),
                    None => OutputDestination::Default,
                });
            }
            "--snapshot-in" => {
                options.snapshot_in = Some(PathBuf::from(next_value(&mut args, "--snapshot-in")?));
            }
            "--snapshot-out" => {
                options.snapshot_out = Some(match optional_value(&mut args) {
                    Some(value) => OutputDestination::Path(PathBuf::from(value)),
                    None => OutputDestination::Default,
                });
            }
            "--profile" => options.profile = true,
            "--profile-json" => {
                options.profile = true;
                options.profile_json = true;
            }
            "--stats-mode" => {
                options.stats_mode = parse_stats_mode(&next_value(&mut args, "--stats-mode")?)?;
            }
            "--verbose-stats" => options.stats_mode = StatsMode::Full,
            "--json" => options.json = true,
            "--quiet" => options.quiet = true,
            "--until-cells" => {
                options.until_cells = Some(parse_value(&mut args, "--until-cells")?);
            }
            "--predation" => {
                options.config.predation_enabled = true;
                options.predation_override = Some(true);
            }
            "--no-predation" => {
                options.config.predation_enabled = false;
                options.predation_override = Some(false);
            }
            "--trace" => parse_trace_mode(&next_value(&mut args, "--trace")?)?,
            other => {
                if !parse_environment_flag(other, &mut args, &mut options.config)? {
                    return Err(format!("unrecognized argument '{other}'"));
                }
            }
        }
    }
    Ok(true)
}

fn parse_bench<I>(mut args: std::iter::Peekable<I>) -> Result<Option<Command>, String>
where
    I: Iterator<Item = String>,
{
    let mut options = BenchOptions::default();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_usage();
                return Ok(None);
            }
            "--seed" => options.config.seed = next_value(&mut args, "--seed")?,
            "--width" => options.config.width = parse_value(&mut args, "--width")?,
            "--height" => options.config.height = parse_value(&mut args, "--height")?,
            "--initial-cells" => {
                options.initial_cells = parse_value(&mut args, "--initial-cells")?;
                options.config.initial_founder_count = options.initial_cells;
            }
            "--initial-energy" => {
                options.initial_energy = Some(parse_value(&mut args, "--initial-energy")?)
            }
            "--steps" => options.steps = parse_value(&mut args, "--steps")?,
            "--stats-every" => options.stats_every = parse_value(&mut args, "--stats-every")?,
            "--dt-seconds" => options.config.dt_seconds = parse_value(&mut args, "--dt-seconds")?,
            "--enval-alpha" => {
                options.config.enval_diffusion_alpha = parse_value(&mut args, "--enval-alpha")?;
            }
            "--check-invariants-every" => {
                options.check_invariants_every =
                    parse_value(&mut args, "--check-invariants-every")?;
            }
            "--check-invariants" => options.check_invariants_every = 1,
            "--profile" => options.profile = true,
            "--profile-json" => {
                options.profile = true;
                options.profile_json = true;
            }
            "--stats-mode" => {
                options.stats_mode = parse_stats_mode(&next_value(&mut args, "--stats-mode")?)?;
            }
            "--verbose-stats" => options.stats_mode = StatsMode::Full,
            "--json" => options.json = true,
            "--quiet" => options.quiet = true,
            "--predation" => options.config.predation_enabled = true,
            "--no-predation" => options.config.predation_enabled = false,
            "--trace" => parse_trace_mode(&next_value(&mut args, "--trace")?)?,
            other => {
                if !parse_environment_flag(other, &mut args, &mut options.config)? {
                    return Err(format!("unrecognized argument '{other}'"));
                }
            }
        }
    }
    Ok(Some(Command::Bench(options)))
}

fn parse_viability<I>(mut args: std::iter::Peekable<I>) -> Result<Option<Command>, String>
where
    I: Iterator<Item = String>,
{
    let mut options = ViabilityOptions::default();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_usage();
                return Ok(None);
            }
            "--seeds" => {
                options.seeds = next_value(&mut args, "--seeds")?
                    .split(',')
                    .map(str::trim)
                    .filter(|seed| !seed.is_empty())
                    .map(str::to_owned)
                    .collect();
                if options.seeds.is_empty() {
                    return Err("--seeds requires at least one seed".to_owned());
                }
            }
            "--steps" => options.steps = parse_value(&mut args, "--steps")?,
            "--jobs" => {
                options.jobs = parse_value(&mut args, "--jobs")?;
                if options.jobs == 0 {
                    return Err("--jobs must be at least 1".to_owned());
                }
            }
            "--json" => options.json = true,
            "--quiet" => options.quiet = true,
            "--width" => options.config.width = parse_value(&mut args, "--width")?,
            "--height" => options.config.height = parse_value(&mut args, "--height")?,
            "--initial-cells" => {
                options.config.initial_founder_count = parse_value(&mut args, "--initial-cells")?;
            }
            "--dt-seconds" => options.config.dt_seconds = parse_value(&mut args, "--dt-seconds")?,
            "--enval-alpha" => {
                options.config.enval_diffusion_alpha = parse_value(&mut args, "--enval-alpha")?;
            }
            "--predation" => options.config.predation_enabled = true,
            "--no-predation" => options.config.predation_enabled = false,
            other => {
                if !parse_environment_flag(other, &mut args, &mut options.config)? {
                    return Err(format!("unrecognized argument '{other}'"));
                }
            }
        }
    }
    Ok(Some(Command::Viability(options)))
}

fn parse_environment_flag<I>(
    arg: &str,
    args: &mut std::iter::Peekable<I>,
    config: &mut Config,
) -> Result<bool, String>
where
    I: Iterator<Item = String>,
{
    match arg {
        "--permeability" => config.membrane_permeability = parse_value(args, arg)?,
        "--catalyst-upkeep" => config.catalyst_upkeep_per_sec = parse_value(args, arg)?,
        "--source-pairs" => config.enval_sources.pairs = parse_value(args, arg)?,
        "--source-radius" => config.enval_sources.radius = parse_value(args, arg)?,
        "--source-magnitude" => config.enval_sources.magnitude = parse_value(args, arg)?,
        "--source-relaxation" => {
            config.enval_sources.relaxation_per_second = parse_value(args, arg)?;
        }
        "--recharge-rate" => config.enval_recharge.rate_per_second = parse_value(args, arg)?,
        "--heterogeneity" => config.element_fields.heterogeneity = parse_value(args, arg)?,
        "--heterogeneity-scale" => {
            config.element_fields.heterogeneity_scale = parse_value(args, arg)?;
        }
        _ => return Ok(false),
    }
    Ok(true)
}

fn next_value<I>(args: &mut std::iter::Peekable<I>, flag: &str) -> Result<String, String>
where
    I: Iterator<Item = String>,
{
    args.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn optional_value<I>(args: &mut std::iter::Peekable<I>) -> Option<String>
where
    I: Iterator<Item = String>,
{
    match args.peek() {
        Some(value) if !looks_like_flag(value) => args.next(),
        _ => None,
    }
}

fn looks_like_flag(value: &str) -> bool {
    value.starts_with('-') && value.len() > 1
}

fn parse_value<I, T>(args: &mut std::iter::Peekable<I>, flag: &str) -> Result<T, String>
where
    I: Iterator<Item = String>,
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let raw = next_value(args, flag)?;
    raw.parse::<T>()
        .map_err(|err| format!("invalid value for {flag}: {err}"))
}

fn parse_stats_mode(raw: &str) -> Result<StatsMode, String> {
    match raw {
        "compact" => Ok(StatsMode::Compact),
        "full" => Ok(StatsMode::Full),
        other => Err(format!(
            "unsupported stats mode '{other}'; expected compact or full"
        )),
    }
}

fn parse_trace_mode(raw: &str) -> Result<(), String> {
    match raw {
        "off" => Ok(()),
        other => Err(format!(
            "unsupported trace mode '{other}'; only '--trace off' is currently supported"
        )),
    }
}

fn should_check_invariants(enabled: bool, every: u64, step_index: u64) -> bool {
    enabled && every > 0 && step_index % every == 0
}

fn resolve_run_outputs(options: &RunOptions) -> std::io::Result<ResolvedOutputPaths> {
    let cwd = env::current_dir()?;
    let timestamp = local_timestamp();
    resolve_output_requests(
        options.csv.as_ref(),
        options.snapshot_out.as_ref(),
        &cwd,
        &timestamp,
    )
}

fn resolve_output_requests(
    csv: Option<&OutputDestination>,
    snapshot_out: Option<&OutputDestination>,
    cwd: &Path,
    timestamp: &str,
) -> std::io::Result<ResolvedOutputPaths> {
    Ok(ResolvedOutputPaths {
        csv: match csv {
            Some(request) => Some(resolve_output_destination(
                request,
                CSV_EXTENSION,
                cwd,
                timestamp,
            )?),
            None => None,
        },
        snapshot_out: match snapshot_out {
            Some(request) => Some(resolve_output_destination(
                request,
                SNAPSHOT_EXTENSION,
                cwd,
                timestamp,
            )?),
            None => None,
        },
    })
}

fn resolve_output_destination(
    request: &OutputDestination,
    extension: &str,
    cwd: &Path,
    timestamp: &str,
) -> std::io::Result<PathBuf> {
    match request {
        OutputDestination::Default => {
            let dir = cwd.join(DEFAULT_OUTPUT_DIR);
            fs::create_dir_all(&dir)?;
            Ok(dir.join(format!("{timestamp}.{extension}")))
        }
        OutputDestination::Path(path) => resolve_explicit_output_path(path, extension, timestamp),
    }
}

fn resolve_explicit_output_path(
    path: &Path,
    extension: &str,
    timestamp: &str,
) -> std::io::Result<PathBuf> {
    let raw = path.as_os_str().to_string_lossy();
    if path.is_dir() || raw.ends_with('/') || raw.ends_with('\\') {
        fs::create_dir_all(path)?;
        return Ok(path.join(format!("{timestamp}.{extension}")));
    }

    let mut path = path.to_path_buf();
    if path.extension().is_none() {
        path.set_extension(extension);
    }
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    Ok(path)
}

fn local_timestamp() -> String {
    Local::now().format("%m%d%Y_%H%M%S").to_string()
}

fn open_csv(path: Option<&PathBuf>) -> Result<Option<BufWriter<File>>, std::io::Error> {
    let Some(path) = path else {
        return Ok(None);
    };
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(writer, "{}", csv_header())?;
    Ok(Some(writer))
}

fn csv_header() -> &'static str {
    "tick,sim_time,population,live_cells,occupancy_fraction,occupied_tiles,empty_tiles,births,deaths,interval_births,interval_deaths,interval_pop_delta,predation_events,interval_predation,cells_consumed,interval_consumed,lineages,total_lineage_records,extinct_lineages,dominant_lineage,dominant_lineage_population,dominant_lineage_share,lineage_entropy,extracellular_a,extracellular_b,extracellular_c,extracellular_d,extracellular_e,extracellular_f,intracellular_a,intracellular_b,intracellular_c,intracellular_d,intracellular_e,intracellular_f,system_a,system_b,system_c,system_d,system_e,system_f,total_element_amount,avg_energy,min_energy,max_energy,total_energy,avg_age,max_age,avg_enzyme_count,min_enzyme_count,max_enzyme_count,cells_at_enzyme_cap,fraction_at_enzyme_cap,cells_with_attackase,cells_with_defensase,avg_attack,max_attack,avg_defense,max_defense,enz_metabolic,enz_defensase,enz_attackase,rx_attempts,rx_successes,rx_no_substrate,rx_success_metabolic,rx_energy_delta_total,rx_enval_input_total,rx_enval_output_total,executed_metabolic_flux,uptake_flux,leak_flux,secretion_flux,divisions,interval_divisions,interval_reaction_attempts,interval_reaction_successes,interval_executed_metabolic_flux,interval_uptake_flux,interval_leak_flux,interval_secretion_flux,cell_steps,interval_cell_steps,interval_cell_steps_per_sec,element_field_diffusion_tiles,element_uptake_events,enval_avg,enval_min,enval_max,enval_stddev,enval_p05,enval_p50,enval_p95,enval_positive_tiles,enval_negative_tiles,enval_near_zero_tiles,predator_energy_gained,avg_energy_gained_per_predation,enzyme_transfers,enzyme_replacements,energy_founder,energy_injected,energy_extracted,energy_chemical_harvest,energy_chemical_cost,energy_enval_harvest,energy_pump_cost,energy_maintenance,energy_death_loss,energy_predation_transfer,energy_ledger_residual,renewable_coverage,enval_source_inflow,enval_cell_uptake,enval_cell_emission,enval_recharge,enval_edits,recharge_amount,recharge_energy"
}

fn emit_world_stats_profiled(
    world: &World,
    csv: &mut Option<BufWriter<File>>,
    print: bool,
    profile_enabled: bool,
    profile: &mut CliProfile,
    state: &mut StatsEmissionState,
    stats_mode: StatsMode,
    json: bool,
) -> std::io::Result<()> {
    let start = Instant::now();
    let stats = collect_stats_for_output(world, csv.is_some(), stats_mode, json);
    let interval = state.capture_interval(&stats, Instant::now());
    emit_stats(&stats, &interval, csv, print, stats_mode, json)?;
    if profile_enabled {
        profile.stats_output += start.elapsed();
    }
    Ok(())
}

fn emit_bench_stats(
    world: &World,
    profile_enabled: bool,
    profile: &mut CliProfile,
    state: &mut StatsEmissionState,
    stats_mode: StatsMode,
    json: bool,
) {
    let start = Instant::now();
    let stats = collect_stats_for_output(world, false, stats_mode, json);
    let interval = state.capture_interval(&stats, Instant::now());
    print_stats_record(&stats, &interval, stats_mode, json);
    if profile_enabled {
        profile.stats_output += start.elapsed();
    }
}

fn collect_stats_for_output(
    world: &World,
    csv_enabled: bool,
    stats_mode: StatsMode,
    json: bool,
) -> WorldStats {
    if csv_enabled || json || matches!(stats_mode, StatsMode::Full) {
        world.stats()
    } else {
        world.compact_stats()
    }
}

fn emit_stats(
    stats: &WorldStats,
    interval: &StatsInterval,
    csv: &mut Option<BufWriter<File>>,
    print: bool,
    stats_mode: StatsMode,
    json: bool,
) -> std::io::Result<()> {
    if print {
        print_stats_record(stats, interval, stats_mode, json);
    }
    if let Some(writer) = csv.as_mut() {
        write_csv_record(writer, stats, interval)?;
    }
    Ok(())
}

fn write_csv_record<W: Write>(
    writer: &mut W,
    stats: &WorldStats,
    interval: &StatsInterval,
) -> std::io::Result<()> {
    let fields = vec![
        stats.tick_count.to_string(),
        format!("{:.6}", stats.sim_time_seconds),
        stats.cell_count.to_string(),
        stats.live_cell_count.to_string(),
        format!("{:.8}", stats.occupancy_fraction),
        stats.occupied_tile_count.to_string(),
        stats.empty_tile_count.to_string(),
        stats.births.to_string(),
        stats.deaths.to_string(),
        interval.births.to_string(),
        interval.deaths.to_string(),
        interval.population_delta.to_string(),
        stats.predation_events.to_string(),
        interval.predation_events.to_string(),
        stats.cells_consumed.to_string(),
        interval.cells_consumed.to_string(),
        stats.lineage_count.to_string(),
        stats.total_lineage_records.to_string(),
        stats.extinct_lineage_count.to_string(),
        stats.dominant_lineage_id.to_string(),
        stats.dominant_lineage_population.to_string(),
        format!("{:.8}", stats.dominant_lineage_share),
        format!("{:.8}", stats.lineage_entropy),
        format!("{:.8}", stats.extracellular_element_amounts[0]),
        format!("{:.8}", stats.extracellular_element_amounts[1]),
        format!("{:.8}", stats.extracellular_element_amounts[2]),
        format!("{:.8}", stats.extracellular_element_amounts[3]),
        format!("{:.8}", stats.extracellular_element_amounts[4]),
        format!("{:.8}", stats.extracellular_element_amounts[5]),
        format!("{:.8}", stats.intracellular_element_amounts[0]),
        format!("{:.8}", stats.intracellular_element_amounts[1]),
        format!("{:.8}", stats.intracellular_element_amounts[2]),
        format!("{:.8}", stats.intracellular_element_amounts[3]),
        format!("{:.8}", stats.intracellular_element_amounts[4]),
        format!("{:.8}", stats.intracellular_element_amounts[5]),
        format!("{:.8}", stats.system_element_amounts[0]),
        format!("{:.8}", stats.system_element_amounts[1]),
        format!("{:.8}", stats.system_element_amounts[2]),
        format!("{:.8}", stats.system_element_amounts[3]),
        format!("{:.8}", stats.system_element_amounts[4]),
        format!("{:.8}", stats.system_element_amounts[5]),
        format!("{:.8}", stats.total_element_amount),
        format!("{:.6}", stats.average_cell_energy),
        format!("{:.6}", stats.min_cell_energy),
        format!("{:.6}", stats.max_cell_energy),
        format!("{:.6}", stats.total_cell_energy),
        format!("{:.6}", stats.average_cell_age),
        format!("{:.6}", stats.max_cell_age),
        format!("{:.6}", stats.average_enzyme_count),
        stats.min_enzyme_count.to_string(),
        stats.max_enzyme_count.to_string(),
        stats.cells_at_enzyme_cap.to_string(),
        format!("{:.8}", stats.fraction_cells_at_enzyme_cap),
        stats.cells_with_attackase.to_string(),
        stats.cells_with_defensase.to_string(),
        format!("{:.6}", stats.average_attack_total),
        stats.max_attack_total.to_string(),
        format!("{:.6}", stats.average_defense_total),
        stats.max_defense_total.to_string(),
        stats.enzyme_type_totals.metabolic.to_string(),
        stats.enzyme_type_totals.defensase.to_string(),
        stats.enzyme_type_totals.attackase.to_string(),
        stats.reaction_counters.total_attempts().to_string(),
        stats.reaction_counters.total_successes().to_string(),
        stats
            .reaction_counters
            .no_substrate_by_type
            .total()
            .to_string(),
        stats
            .reaction_counters
            .successes_by_type
            .metabolic
            .to_string(),
        format!(
            "{:.6}",
            stats.reaction_counters.energy_delta_by_type.total()
        ),
        format!("{:.6}", stats.reaction_counters.enval_input_by_type.total()),
        format!(
            "{:.6}",
            stats.reaction_counters.enval_output_by_type.total()
        ),
        format!("{:.8}", stats.reaction_counters.executed_metabolic_flux),
        format!("{:.8}", stats.reaction_counters.uptake_flux),
        format!("{:.8}", stats.reaction_counters.leak_flux),
        format!("{:.8}", stats.reaction_counters.secretion_flux),
        stats.reaction_counters.divisions.to_string(),
        interval.divisions.to_string(),
        interval.reaction_attempts.to_string(),
        interval.reaction_successes.to_string(),
        format!("{:.8}", interval.executed_metabolic_flux),
        format!("{:.8}", interval.uptake_flux),
        format!("{:.8}", interval.leak_flux),
        format!("{:.8}", interval.secretion_flux),
        stats.operation_counters.cell_steps.to_string(),
        interval.cell_steps.to_string(),
        format!("{:.3}", interval.cell_steps_per_sec),
        stats
            .operation_counters
            .element_field_diffusion_tiles
            .to_string(),
        stats.operation_counters.element_uptake_events.to_string(),
        format!("{:.6}", stats.average_enval),
        format!("{:.6}", stats.min_enval),
        format!("{:.6}", stats.max_enval),
        format!("{:.6}", stats.enval_std_dev),
        format!("{:.6}", stats.enval_p05),
        format!("{:.6}", stats.enval_p50),
        format!("{:.6}", stats.enval_p95),
        stats.positive_enval_tile_count.to_string(),
        stats.negative_enval_tile_count.to_string(),
        stats.near_zero_enval_tile_count.to_string(),
        format!("{:.6}", stats.predator_energy_gained),
        format!("{:.6}", stats.average_energy_gained_per_predation),
        stats.predation_enzyme_transfers.to_string(),
        stats.predation_enzyme_replacements.to_string(),
        format!("{:.9}", stats.energy_ledger.founder_energy),
        format!("{:.9}", stats.energy_ledger.injected_energy),
        format!("{:.9}", stats.energy_ledger.extracted_energy),
        format!("{:.9}", stats.energy_ledger.chemical_harvest),
        format!("{:.9}", stats.energy_ledger.chemical_cost),
        format!("{:.9}", stats.energy_ledger.enval_harvest),
        format!("{:.9}", stats.energy_ledger.pump_cost),
        format!("{:.9}", stats.energy_ledger.maintenance),
        format!("{:.9}", stats.energy_ledger.death_loss),
        format!("{:.9}", stats.energy_ledger.predation_transfer),
        format!("{:.6e}", stats.energy_ledger_residual),
        format!("{:.8}", stats.renewable_coverage),
        format!("{:.9}", stats.enval_ledger.source_inflow),
        format!("{:.9}", stats.enval_ledger.cell_uptake),
        format!("{:.9}", stats.enval_ledger.cell_emission),
        format!("{:.9}", stats.enval_ledger.recharge),
        format!("{:.9}", stats.enval_ledger.edits),
        format!("{:.9}", stats.enval_ledger.recharge_amount),
        format!("{:.9}", stats.enval_ledger.recharge_energy),
    ];
    assert_eq!(fields.len(), csv_header().split(',').count());
    writeln!(writer, "{}", fields.join(","))
}

fn print_stats_record(stats: &WorldStats, interval: &StatsInterval, mode: StatsMode, json: bool) {
    if json {
        print_stats_json(stats, interval);
        return;
    }
    match mode {
        StatsMode::Compact => print_compact_stats(stats, interval),
        StatsMode::Full => print_full_stats(stats, interval),
    }
}

fn print_compact_stats(stats: &WorldStats, interval: &StatsInterval) {
    let extracellular = stats.extracellular_element_amounts.iter().sum::<f64>();
    let intracellular = stats.intracellular_element_amounts.iter().sum::<f64>();
    println!(
        "tick={} time={:.3}s size={}x{} tiles={} occ={:.3} cells={} d_cells={:+} births={} d_births={} deaths={} d_deaths={} predation={} d_predation={} consumed={} lineages={} total_elements={:.6} extracellular={:.6} intracellular={:.6} avg_energy={:.3} avg_enzymes={:.2} cap={:.3} rx_success={} d_rx_success={} flux={:.6} d_flux={:.6} uptake={:.6} secretion={:.6} cell_steps={} d_cell_steps={} enval_avg={:.6} enval_min={:.6} enval_max={:.6} enval_sd={:.6} system=A:{:.4} B:{:.4} C:{:.4} D:{:.4} E:{:.4} F:{:.4} energy=founder:{:.3} injected:{:.3} extracted:{:.3} chem_harvest:{:.3} chem_cost:{:.3} enval_harvest:{:.3} pump:{:.3} maintenance:{:.3} death_loss:{:.3} predation:{:.3} residual:{:.3e} coverage={:.4}",
        stats.tick_count,
        stats.sim_time_seconds,
        stats.width,
        stats.height,
        stats.tile_count,
        stats.occupancy_fraction,
        stats.live_cell_count,
        interval.population_delta,
        stats.births,
        interval.births,
        stats.deaths,
        interval.deaths,
        stats.predation_events,
        interval.predation_events,
        stats.cells_consumed,
        stats.lineage_count,
        stats.total_element_amount,
        extracellular,
        intracellular,
        stats.average_cell_energy,
        stats.average_enzyme_count,
        stats.fraction_cells_at_enzyme_cap,
        stats.reaction_counters.total_successes(),
        interval.reaction_successes,
        stats.reaction_counters.executed_metabolic_flux,
        interval.executed_metabolic_flux,
        stats.reaction_counters.uptake_flux,
        stats.reaction_counters.secretion_flux,
        stats.operation_counters.cell_steps,
        interval.cell_steps,
        stats.average_enval,
        stats.min_enval,
        stats.max_enval,
        stats.enval_std_dev,
        stats.system_element_amounts[0],
        stats.system_element_amounts[1],
        stats.system_element_amounts[2],
        stats.system_element_amounts[3],
        stats.system_element_amounts[4],
        stats.system_element_amounts[5],
        stats.energy_ledger.founder_energy,
        stats.energy_ledger.injected_energy,
        stats.energy_ledger.extracted_energy,
        stats.energy_ledger.chemical_harvest,
        stats.energy_ledger.chemical_cost,
        stats.energy_ledger.enval_harvest,
        stats.energy_ledger.pump_cost,
        stats.energy_ledger.maintenance,
        stats.energy_ledger.death_loss,
        stats.energy_ledger.predation_transfer,
        stats.energy_ledger_residual,
        stats.renewable_coverage,
    );
}

fn print_full_stats(stats: &WorldStats, interval: &StatsInterval) {
    println!(
        "tick={} time={:.3}s size={}x{} sim_interval={:.3}s wall_interval={:.3}s steps_per_sec={:.3}",
        stats.tick_count,
        stats.sim_time_seconds,
        stats.width,
        stats.height,
        interval.sim_seconds_delta,
        interval.wall_seconds,
        interval.steps_per_sec,
    );
    println!(
        "  grid occupied={} empty={} occupancy={:.6}",
        stats.occupied_tile_count, stats.empty_tile_count, stats.occupancy_fraction
    );
    println!(
        "  chemistry total={:.6} extracellular={:?} intracellular={:?} system={:?}",
        stats.total_element_amount,
        stats.extracellular_element_amounts,
        stats.intracellular_element_amounts,
        stats.system_element_amounts,
    );
    println!(
        "  cells live={} births={} (+{}) deaths={} (+{}) divisions={} (+{}) avg_energy={:.3} min_energy={:.3} max_energy={:.3} avg_age={:.3}s max_age={:.3}s",
        stats.live_cell_count,
        stats.births,
        interval.births,
        stats.deaths,
        interval.deaths,
        stats.reaction_counters.divisions,
        interval.divisions,
        stats.average_cell_energy,
        stats.min_cell_energy,
        stats.max_cell_energy,
        stats.average_cell_age,
        stats.max_cell_age,
    );
    println!(
        "  enzymes avg={:.2} min={} max={} cap={} cap_frac={:.3} hist_1_10={:?} attack_cells={} defense_cells={} avg_attack={:.2} max_attack={} avg_defense={:.2} max_defense={} totals=Met:{} Def:{} Atk:{}",
        stats.average_enzyme_count,
        stats.min_enzyme_count,
        stats.max_enzyme_count,
        stats.cells_at_enzyme_cap,
        stats.fraction_cells_at_enzyme_cap,
        &stats.enzyme_count_histogram[1..],
        stats.cells_with_attackase,
        stats.cells_with_defensase,
        stats.average_attack_total,
        stats.max_attack_total,
        stats.average_defense_total,
        stats.max_defense_total,
        stats.enzyme_type_totals.metabolic,
        stats.enzyme_type_totals.defensase,
        stats.enzyme_type_totals.attackase,
    );
    println!(
        "  lineages extant={} total_records={} extinct={} dominant={} dominant_pop={} dominant_share={:.3} entropy={:.3}",
        stats.extant_lineage_count,
        stats.total_lineage_records,
        stats.extinct_lineage_count,
        stats.dominant_lineage_id,
        stats.dominant_lineage_population,
        stats.dominant_lineage_share,
        stats.lineage_entropy,
    );
    println!(
        "  predation events={} (+{}) consumed={} (+{}) energy_gained={:.3} avg_gain={:.3} enzyme_transfers={} replacements={}",
        stats.predation_events,
        interval.predation_events,
        stats.cells_consumed,
        interval.cells_consumed,
        stats.predator_energy_gained,
        stats.average_energy_gained_per_predation,
        stats.predation_enzyme_transfers,
        stats.predation_enzyme_replacements,
    );
    println!(
        "  metabolism attempts={} (+{}) successes={} (+{}) no_substrate={} executed_flux={:.6} (+{:.6}) uptake_flux={:.6} (+{:.6}) secretion_flux={:.6} (+{:.6}) energy_delta={:.3} enval_in={:.3} enval_out={:.3}",
        stats.reaction_counters.total_attempts(),
        interval.reaction_attempts,
        stats.reaction_counters.total_successes(),
        interval.reaction_successes,
        stats.reaction_counters.no_substrate_by_type.total(),
        stats.reaction_counters.executed_metabolic_flux,
        interval.executed_metabolic_flux,
        stats.reaction_counters.uptake_flux,
        interval.uptake_flux,
        stats.reaction_counters.secretion_flux,
        interval.secretion_flux,
        stats.reaction_counters.energy_delta_by_type.total(),
        stats.reaction_counters.enval_input_by_type.total(),
        stats.reaction_counters.enval_output_by_type.total(),
    );
    println!(
        "  enval avg={:.6} min={:.6} p05={:.6} p50={:.6} p95={:.6} max={:.6} sd={:.6} pos={} neg={} near_zero={}",
        stats.average_enval,
        stats.min_enval,
        stats.enval_p05,
        stats.enval_p50,
        stats.enval_p95,
        stats.max_enval,
        stats.enval_std_dev,
        stats.positive_enval_tile_count,
        stats.negative_enval_tile_count,
        stats.near_zero_enval_tile_count,
    );
    println!(
        "  energy_ledger founder={:.6} injected={:.6} extracted={:.6} chemical_harvest={:.6} chemical_cost={:.6} enval_harvest={:.6} pump_cost={:.6} maintenance={:.6} death_loss={:.6} predation_transfer={:.6} residual={:.3e} renewable_coverage={:.6}",
        stats.energy_ledger.founder_energy,
        stats.energy_ledger.injected_energy,
        stats.energy_ledger.extracted_energy,
        stats.energy_ledger.chemical_harvest,
        stats.energy_ledger.chemical_cost,
        stats.energy_ledger.enval_harvest,
        stats.energy_ledger.pump_cost,
        stats.energy_ledger.maintenance,
        stats.energy_ledger.death_loss,
        stats.energy_ledger.predation_transfer,
        stats.energy_ledger_residual,
        stats.renewable_coverage,
    );
    println!(
        "  enval_ledger source_inflow={:.6} cell_uptake={:.6} cell_emission={:.6} recharge={:.6} edits={:.6} recharge_amount={:.6} recharge_energy={:.6}",
        stats.enval_ledger.source_inflow,
        stats.enval_ledger.cell_uptake,
        stats.enval_ledger.cell_emission,
        stats.enval_ledger.recharge,
        stats.enval_ledger.edits,
        stats.enval_ledger.recharge_amount,
        stats.enval_ledger.recharge_energy,
    );
    println!(
        "  interval ticks={} pop_delta={:+} cell_steps={} cell_steps_per_sec={:.3} enzyme_attempts={} enzyme_attempts_per_sec={:.3} reactions_per_sec={:.3}",
        interval.tick_delta,
        interval.population_delta,
        interval.cell_steps,
        interval.cell_steps_per_sec,
        interval.enzyme_attempts,
        interval.enzyme_attempts_per_sec,
        interval.reactions_per_sec,
    );
}

fn print_stats_json(stats: &WorldStats, interval: &StatsInterval) {
    let value = serde_json::json!({
        "tick": stats.tick_count,
        "sim_time": stats.sim_time_seconds,
        "width": stats.width,
        "height": stats.height,
        "population": stats.live_cell_count,
        "occupancy_fraction": stats.occupancy_fraction,
        "births": stats.births,
        "deaths": stats.deaths,
        "predation_events": stats.predation_events,
        "cells_consumed": stats.cells_consumed,
        "lineages": stats.lineage_count,
        "total_lineage_records": stats.total_lineage_records,
        "extinct_lineages": stats.extinct_lineage_count,
        "dominant_lineage": stats.dominant_lineage_id,
        "dominant_lineage_share": stats.dominant_lineage_share,
        "lineage_entropy": stats.lineage_entropy,
        "avg_energy": stats.average_cell_energy,
        "min_energy": stats.min_cell_energy,
        "max_energy": stats.max_cell_energy,
        "total_energy": stats.total_cell_energy,
        "avg_enzyme_count": stats.average_enzyme_count,
        "cells_at_enzyme_cap": stats.cells_at_enzyme_cap,
        "chemistry": {
            "extracellular": element_amounts_json(stats.extracellular_element_amounts),
            "intracellular": element_amounts_json(stats.intracellular_element_amounts),
            "system": element_amounts_json(stats.system_element_amounts),
            "total_element_amount": stats.total_element_amount,
        },
        "enzyme_type_totals": {
            "metabolic": stats.enzyme_type_totals.metabolic,
            "defensase": stats.enzyme_type_totals.defensase,
            "attackase": stats.enzyme_type_totals.attackase,
        },
        "metabolism": {
            "attempts": stats.reaction_counters.total_attempts(),
            "successes": stats.reaction_counters.total_successes(),
            "no_substrate": stats.reaction_counters.no_substrate_by_type.total(),
            "executed_metabolic_flux": stats.reaction_counters.executed_metabolic_flux,
            "uptake_flux": stats.reaction_counters.uptake_flux,
            "leak_flux": stats.reaction_counters.leak_flux,
            "secretion_flux": stats.reaction_counters.secretion_flux,
            "divisions": stats.reaction_counters.divisions,
            "energy_delta": stats.reaction_counters.energy_delta_by_type.total(),
            "enval_input": stats.reaction_counters.enval_input_by_type.total(),
            "enval_output": stats.reaction_counters.enval_output_by_type.total(),
        },
        "energy_ledger": {
            "founder_energy": stats.energy_ledger.founder_energy,
            "injected_energy": stats.energy_ledger.injected_energy,
            "extracted_energy": stats.energy_ledger.extracted_energy,
            "chemical_harvest": stats.energy_ledger.chemical_harvest,
            "chemical_cost": stats.energy_ledger.chemical_cost,
            "enval_harvest": stats.energy_ledger.enval_harvest,
            "pump_cost": stats.energy_ledger.pump_cost,
            "maintenance": stats.energy_ledger.maintenance,
            "death_loss": stats.energy_ledger.death_loss,
            "predation_transfer": stats.energy_ledger.predation_transfer,
            "residual": stats.energy_ledger_residual,
            "renewable_coverage": stats.renewable_coverage,
        },
        "enval_ledger": {
            "source_inflow": stats.enval_ledger.source_inflow,
            "cell_uptake": stats.enval_ledger.cell_uptake,
            "cell_emission": stats.enval_ledger.cell_emission,
            "recharge": stats.enval_ledger.recharge,
            "edits": stats.enval_ledger.edits,
            "recharge_amount": stats.enval_ledger.recharge_amount,
            "recharge_energy": stats.enval_ledger.recharge_energy,
        },
        "enval": {
            "average": stats.average_enval,
            "min": stats.min_enval,
            "max": stats.max_enval,
            "stddev": stats.enval_std_dev,
            "p05": stats.enval_p05,
            "p50": stats.enval_p50,
            "p95": stats.enval_p95,
        },
        "interval": {
            "ticks": interval.tick_delta,
            "sim_seconds": interval.sim_seconds_delta,
            "wall_seconds": interval.wall_seconds,
            "population_delta": interval.population_delta,
            "births": interval.births,
            "deaths": interval.deaths,
            "predation_events": interval.predation_events,
            "reaction_attempts": interval.reaction_attempts,
            "reaction_successes": interval.reaction_successes,
            "executed_metabolic_flux": interval.executed_metabolic_flux,
            "uptake_flux": interval.uptake_flux,
            "leak_flux": interval.leak_flux,
            "secretion_flux": interval.secretion_flux,
            "cell_steps": interval.cell_steps,
            "cell_steps_per_sec": interval.cell_steps_per_sec,
            "enzyme_attempts_per_sec": interval.enzyme_attempts_per_sec,
        },
    });
    println!("{}", value);
}

fn element_amounts_json(amounts: [f64; 6]) -> serde_json::Value {
    serde_json::json!({
        "A": amounts[0],
        "B": amounts[1],
        "C": amounts[2],
        "D": amounts[3],
        "E": amounts[4],
        "F": amounts[5],
    })
}

fn print_profile_summary(steps: u64, wall: Duration, mut profile: CliProfile, json: bool) {
    profile.step_durations_ms.sort_by(|a, b| a.total_cmp(b));
    let steps = steps.max(1) as f64;
    let wall_seconds = wall.as_secs_f64().max(1.0e-9);
    let measured_ms = duration_ms(profile.step.total);
    let element_ms = duration_ms(profile.step.element_field_diffusion);
    let cell_ms = duration_ms(profile.step.cell_step);
    let mechanics_ms = duration_ms(profile.step.cell_mechanics);
    let predation_ms = duration_ms(profile.step.predation);
    let enval_ms = duration_ms(profile.step.enval_diffusion);
    let pct = |part_ms: f64| -> f64 {
        if measured_ms > 0.0 {
            100.0 * part_ms / measured_ms
        } else {
            0.0
        }
    };
    let p50 = percentile_f64(&profile.step_durations_ms, 0.50);
    let p95 = percentile_f64(&profile.step_durations_ms, 0.95);
    let counters = profile.step.counters;
    if json {
        let value = serde_json::json!({
            "profile": {
                "wall_ms": duration_ms(wall),
                "avg_step_ms": measured_ms / steps,
                "min_step_ms": profile.min_step.map(duration_ms).unwrap_or(0.0),
                "max_step_ms": duration_ms(profile.max_step),
                "p50_step_ms": p50,
                "p95_step_ms": p95,
                "element_field_diffusion_ms_per_step": element_ms / steps,
                "cells_ms_per_step": cell_ms / steps,
                "cell_mechanics_ms_per_step": mechanics_ms / steps,
                "predation_ms_per_step": predation_ms / steps,
                "enval_ms_per_step": enval_ms / steps,
                "cell_steps": counters.cell_steps,
                "cell_steps_per_sec": counters.cell_steps as f64 / wall_seconds,
                "enzyme_attempts": counters.metabolic_enzyme_attempts,
                "enzyme_attempts_per_sec": counters.metabolic_enzyme_attempts as f64 / wall_seconds,
                "reactions_succeeded": counters.reactions_succeeded,
                "reactions_per_sec": counters.reactions_succeeded as f64 / wall_seconds,
                "predation_pairs_checked": counters.predation_pairs_checked,
                "predation_cells_considered": counters.predation_cells_considered,
                "predation_candidate_pairs": counters.predation_candidate_pairs,
                "predation_cross_lineage_pairs": counters.predation_cross_lineage_pairs,
                "spatial_candidate_checks": counters.spatial_candidate_checks,
                "overlap_candidates": counters.overlap_candidates,
                "overlap_corrections": counters.overlap_corrections,
                "combat_enzyme_skips": counters.combat_enzyme_skips,
                "element_field_diffusion_tiles": counters.element_field_diffusion_tiles,
                "element_uptake_events": counters.element_uptake_events,
                "cell_divisions": counters.cell_divisions,
                "cell_deaths": counters.cell_deaths
            }
        });
        println!("{}", value);
        return;
    }
    println!(
        "profile wall_ms={:.3} avg_step_ms={:.6} min_step_ms={:.6} max_step_ms={:.6} p50_step_ms={:.6} p95_step_ms={:.6} element_field_ms={:.6} cells_ms={:.6} mechanics_ms={:.6} predation_ms={:.6} enval_ms={:.6} measured_total_ms={:.6} element_field_pct={:.2} cells_pct={:.2} mechanics_pct={:.2} predation_pct={:.2} enval_pct={:.2} stats_output_ms={:.6} invariants_ms={:.6} snapshot_io_ms={:.6} csv_flush_ms={:.6} cell_steps={} cell_steps_per_sec={:.3} enzyme_entries={} enzyme_attempts={} enzyme_attempts_per_sec={:.3} reactions={} reactions_per_sec={:.3} element_field_tiles={} uptake_events={} divisions={} deaths={} predation_pairs={} predation_cells={} predation_candidates={} predation_cross_lineage={} predation_events={} consumed={} spatial_candidates={} overlap_candidates={} overlap_corrections={} combat_enzyme_skips={} enval_avg_calls={} enzyme_list_clones={} genome_clones={}",
        duration_ms(wall),
        measured_ms / steps,
        profile.min_step.map(duration_ms).unwrap_or(0.0),
        duration_ms(profile.max_step),
        p50,
        p95,
        element_ms / steps,
        cell_ms / steps,
        mechanics_ms / steps,
        predation_ms / steps,
        enval_ms / steps,
        measured_ms / steps,
        pct(element_ms),
        pct(cell_ms),
        pct(mechanics_ms),
        pct(predation_ms),
        pct(enval_ms),
        duration_ms(profile.stats_output),
        duration_ms(profile.invariants),
        duration_ms(profile.snapshot_io),
        duration_ms(profile.csv_flush),
        counters.cell_steps,
        counters.cell_steps as f64 / wall_seconds,
        counters.enzyme_entries_seen,
        counters.metabolic_enzyme_attempts,
        counters.metabolic_enzyme_attempts as f64 / wall_seconds,
        counters.reactions_succeeded,
        counters.reactions_succeeded as f64 / wall_seconds,
        counters.element_field_diffusion_tiles,
        counters.element_uptake_events,
        counters.cell_divisions,
        counters.cell_deaths,
        counters.predation_pairs_checked,
        counters.predation_cells_considered,
        counters.predation_candidate_pairs,
        counters.predation_cross_lineage_pairs,
        counters.predation_events,
        counters.predation_cells_consumed,
        counters.spatial_candidate_checks,
        counters.overlap_candidates,
        counters.overlap_corrections,
        counters.combat_enzyme_skips,
        counters.local_enval_average_calls,
        counters.enzyme_list_clones,
        counters.genome_clones,
    );
}

fn percentile_f64(sorted_values: &[f64], fraction: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }
    let max_index = sorted_values.len() - 1;
    let index = ((max_index as f64) * fraction.clamp(0.0, 1.0)).round() as usize;
    sorted_values[index.min(max_index)]
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn print_usage() {
    eprintln!(
        "usage:
  microcosm run [--seed S] [--width W] [--height H] [--initial-cells N] [--steps N] [--until-cells N] [--stats-every N] [--stats-mode compact|full] [--json] [--check-invariants] [--check-invariants-every N] [--csv [path]] [--snapshot-in path] [--snapshot-out [path]] [--profile|--profile-json] [--no-predation] [--trace off]
  microcosm bench [--seed S] [--width W] [--height H] [--initial-cells N] [--steps N] [--stats-every N] [--stats-mode compact|full] [--json] [--profile|--profile-json] [--no-predation] [--trace off]
  microcosm inspect snapshot.micosm
  microcosm viability [--seeds 42,1,2,3,7] [--steps 60000] [--jobs N] [--json] [--quiet] [--width W] [--height H] [--initial-cells N] [--no-predation]
environment flags (run, bench, viability):
  [--permeability P] [--catalyst-upkeep U] [--source-pairs N] [--source-radius R] [--source-magnitude M] [--source-relaxation K] [--recharge-rate K] [--heterogeneity H] [--heterogeneity-scale S]"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let columns = csv_header().split(',').collect::<Vec<_>>();
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
        let header = csv_header().split(',').collect::<Vec<_>>();
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
        world.set_all_enval(0.25).unwrap();

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
                "--source-pairs",
                "2",
                "--source-radius",
                "5",
                "--source-magnitude",
                "1.5",
                "--source-relaxation",
                "8",
                "--recharge-rate",
                "0.2",
                "--heterogeneity",
                "0.3",
                "--heterogeneity-scale",
                "32",
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
            assert_eq!(config.enval_sources.pairs, 2);
            assert_eq!(config.enval_sources.radius, 5.0);
            assert_eq!(config.enval_sources.magnitude, 1.5);
            assert_eq!(config.enval_sources.relaxation_per_second, 8.0);
            assert_eq!(config.enval_recharge.rate_per_second, 0.2);
            assert_eq!(config.element_fields.heterogeneity, 0.3);
            assert_eq!(config.element_fields.heterogeneity_scale, 32.0);
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
}
