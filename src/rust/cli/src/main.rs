use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use chrono::Local;
use microcosmcore::{
    load_from_path, save_to_path, Config, StepProfile, World, WorldStats, SNAPSHOT_EXTENSION,
};

const CSV_EXTENSION: &str = "csv";
const DEFAULT_OUTPUT_DIR: &str = "out";

#[derive(Debug, Clone)]
enum Command {
    Run(RunOptions),
    Bench(BenchOptions),
    Inspect(InspectOptions),
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
    molecule_uptakes: u64,
    molecule_outputs: u64,
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
            molecule_uptakes: reactions.molecule_uptakes,
            molecule_outputs: reactions.molecule_outputs,
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
    }

    if options.steps > 0 && (options.stats_every == 0 || options.steps % options.stats_every != 0) {
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
            options.steps,
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
        "bench steps={} elapsed_sec={:.6} steps_per_sec={:.3} final_cells={} final_molecules={} actual_cell_steps={} cell_steps_per_sec={:.3}",
        options.steps,
        seconds,
        options.steps as f64 / seconds,
        stats.live_cell_count,
        stats.molecule_count,
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
            "--predation" => {
                options.config.predation_enabled = true;
                options.predation_override = Some(true);
            }
            "--no-predation" => {
                options.config.predation_enabled = false;
                options.predation_override = Some(false);
            }
            "--trace" => parse_trace_mode(&next_value(&mut args, "--trace")?)?,
            other => return Err(format!("unrecognized argument '{other}'")),
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
            other => return Err(format!("unrecognized argument '{other}'")),
        }
    }
    Ok(Some(Command::Bench(options)))
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
    "tick,sim_time,population,live_cells,cell_records,dead_cells,occupancy_fraction,occupied_tiles,empty_tiles,births,deaths,interval_births,interval_deaths,interval_pop_delta,predation_events,interval_predation,cells_consumed,interval_consumed,lineages,total_lineage_records,extinct_lineages,dominant_lineage,dominant_lineage_share,molecules,tile_molecules,cell_molecules,free_molecule_records,active_molecule_records,molecule_arena_len,molecule_arena_high_water,molecule_slots_reused,molecule_slots_newly_allocated,total_atoms,tile_atoms,cell_atoms,avg_molecules_per_tile,avg_internal_molecules_per_cell,avg_atoms_per_cell,avg_energy,min_energy,max_energy,total_energy,avg_age,max_age,avg_time_without_food,avg_enzyme_count,min_enzyme_count,max_enzyme_count,cells_at_enzyme_cap,fraction_at_enzyme_cap,cells_with_attackase,cells_with_defensase,avg_attack,max_attack,avg_defense,max_defense,enz_anabolase,enz_catabolase,enz_transmutase,enz_defensase,enz_attackase,rx_attempts,rx_gates_passed,rx_successes,rx_no_substrate,rx_success_anabolase,rx_success_catabolase,rx_success_transmutase,rx_energy_delta_total,rx_enval_input_total,rx_enval_output_total,molecule_uptakes,molecule_outputs,divisions,interval_divisions,interval_reaction_attempts,interval_reaction_successes,interval_molecule_uptakes,interval_molecule_outputs,interval_cell_steps,interval_cell_steps_per_sec,enval_avg,enval_min,enval_max,enval_stddev,enval_p05,enval_p50,enval_p95,enval_positive_tiles,enval_negative_tiles,enval_near_zero_tiles,predator_energy_gained,avg_energy_gained_per_predation,enzyme_transfers,enzyme_replacements,elements_a,elements_b,elements_c,elements_d,elements_e,elements_f"
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

fn write_csv_record(
    writer: &mut BufWriter<File>,
    stats: &WorldStats,
    interval: &StatsInterval,
) -> std::io::Result<()> {
    let fields = vec![
        stats.tick_count.to_string(),
        format!("{:.6}", stats.sim_time_seconds),
        stats.cell_count.to_string(),
        stats.live_cell_count.to_string(),
        stats.cell_record_count.to_string(),
        stats.dead_cell_count.to_string(),
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
        format!("{:.8}", stats.dominant_lineage_share),
        stats.molecule_count.to_string(),
        stats.tile_molecule_count.to_string(),
        stats.cell_molecule_count.to_string(),
        stats.free_molecule_record_count.to_string(),
        stats.active_molecule_record_count.to_string(),
        stats.molecule_arena_len.to_string(),
        stats.molecule_arena_high_water_mark.to_string(),
        stats.molecule_slots_reused.to_string(),
        stats.molecule_slots_newly_allocated.to_string(),
        stats.total_atom_count.to_string(),
        stats.tile_atom_count.to_string(),
        stats.cell_atom_count.to_string(),
        format!("{:.6}", stats.average_molecules_per_tile),
        format!("{:.6}", stats.average_internal_molecules_per_live_cell),
        format!("{:.6}", stats.average_atoms_per_live_cell),
        format!("{:.6}", stats.average_cell_energy),
        format!("{:.6}", stats.min_cell_energy),
        format!("{:.6}", stats.max_cell_energy),
        format!("{:.6}", stats.total_cell_energy),
        format!("{:.6}", stats.average_cell_age),
        format!("{:.6}", stats.max_cell_age),
        format!("{:.6}", stats.average_time_without_food),
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
        stats.enzyme_type_totals.anabolase.to_string(),
        stats.enzyme_type_totals.catabolase.to_string(),
        stats.enzyme_type_totals.transmutase.to_string(),
        stats.enzyme_type_totals.defensase.to_string(),
        stats.enzyme_type_totals.attackase.to_string(),
        stats.reaction_counters.total_attempts().to_string(),
        stats
            .reaction_counters
            .gates_passed_by_type
            .total()
            .to_string(),
        stats.reaction_counters.total_successes().to_string(),
        stats
            .reaction_counters
            .no_substrate_by_type
            .total()
            .to_string(),
        stats
            .reaction_counters
            .successes_by_type
            .anabolase
            .to_string(),
        stats
            .reaction_counters
            .successes_by_type
            .catabolase
            .to_string(),
        stats
            .reaction_counters
            .successes_by_type
            .transmutase
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
        stats.reaction_counters.molecule_uptakes.to_string(),
        stats.reaction_counters.molecule_outputs.to_string(),
        stats.reaction_counters.divisions.to_string(),
        interval.divisions.to_string(),
        interval.reaction_attempts.to_string(),
        interval.reaction_successes.to_string(),
        interval.molecule_uptakes.to_string(),
        interval.molecule_outputs.to_string(),
        interval.cell_steps.to_string(),
        format!("{:.3}", interval.cell_steps_per_sec),
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
        stats.element_counts[0].to_string(),
        stats.element_counts[1].to_string(),
        stats.element_counts[2].to_string(),
        stats.element_counts[3].to_string(),
        stats.element_counts[4].to_string(),
        stats.element_counts[5].to_string(),
    ];
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

fn print_stats(stats: &WorldStats) {
    print_compact_stats(stats, &StatsInterval::default());
}

fn print_compact_stats(stats: &WorldStats, interval: &StatsInterval) {
    println!(
        "tick={} time={:.3}s size={}x{} tiles={} occ={:.3} molecules={} tile_mol={} cell_mol={} free_mol_records={} arena={} reused={} atoms={} cells={} d_cells={:+} births={} d_births={} deaths={} d_deaths={} predation={} d_predation={} consumed={} lineages={} avg_energy={:.3} avg_enzymes={:.2} cap={:.3} rx_success={} d_rx_success={} cell_steps={} d_cell_steps={} enval_avg={:.6} enval_min={:.6} enval_max={:.6} enval_sd={:.6} elements=A:{} B:{} C:{} D:{} E:{} F:{}",
        stats.tick_count,
        stats.sim_time_seconds,
        stats.width,
        stats.height,
        stats.tile_count,
        stats.occupancy_fraction,
        stats.molecule_count,
        stats.tile_molecule_count,
        stats.cell_molecule_count,
        stats.free_molecule_record_count,
        stats.molecule_arena_len,
        stats.molecule_slots_reused,
        stats.total_atom_count,
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
        stats.average_cell_energy,
        stats.average_enzyme_count,
        stats.fraction_cells_at_enzyme_cap,
        stats.reaction_counters.total_successes(),
        interval.reaction_successes,
        stats.operation_counters.cell_steps,
        interval.cell_steps,
        stats.average_enval,
        stats.min_enval,
        stats.max_enval,
        stats.enval_std_dev,
        stats.element_counts[0],
        stats.element_counts[1],
        stats.element_counts[2],
        stats.element_counts[3],
        stats.element_counts[4],
        stats.element_counts[5],
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
        "  molecules active={} tile={} cell={} free_records={} arena_len={} high_water={} slots_reused={} slots_new={} atoms total={} tile={} cell={} avg_tile_mol={:.3} avg_cell_mol={:.3} avg_cell_atoms={:.3}",
        stats.active_molecule_record_count,
        stats.tile_molecule_count,
        stats.cell_molecule_count,
        stats.free_molecule_record_count,
        stats.molecule_arena_len,
        stats.molecule_arena_high_water_mark,
        stats.molecule_slots_reused,
        stats.molecule_slots_newly_allocated,
        stats.total_atom_count,
        stats.tile_atom_count,
        stats.cell_atom_count,
        stats.average_molecules_per_tile,
        stats.average_internal_molecules_per_live_cell,
        stats.average_atoms_per_live_cell,
    );
    println!(
        "  cells live={} records={} dead_records={} births={} (+{}) deaths={} (+{}) divisions={} (+{}) avg_energy={:.3} min_energy={:.3} max_energy={:.3} avg_age={:.3}s max_age={:.3}s avg_no_food={:.3}",
        stats.live_cell_count,
        stats.cell_record_count,
        stats.dead_cell_count,
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
        stats.average_time_without_food,
    );
    println!(
        "  enzymes avg={:.2} min={} max={} cap={} cap_frac={:.3} hist_1_10={:?} attack_cells={} defense_cells={} avg_attack={:.2} max_attack={} avg_defense={:.2} max_defense={} totals=Abl:{} Cbl:{} Trn:{} Def:{} Atk:{}",
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
        stats.enzyme_type_totals.anabolase,
        stats.enzyme_type_totals.catabolase,
        stats.enzyme_type_totals.transmutase,
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
        "  reactions attempts={} (+{}) gates={} successes={} (+{}) no_substrate={} uptake={} (+{}) output={} (+{}) energy_delta={:.3} enval_in={:.3} enval_out={:.3}",
        stats.reaction_counters.total_attempts(),
        interval.reaction_attempts,
        stats.reaction_counters.gates_passed_by_type.total(),
        stats.reaction_counters.total_successes(),
        interval.reaction_successes,
        stats.reaction_counters.no_substrate_by_type.total(),
        stats.reaction_counters.molecule_uptakes,
        interval.molecule_uptakes,
        stats.reaction_counters.molecule_outputs,
        interval.molecule_outputs,
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
    fn merge_object(
        target: &mut serde_json::Map<String, serde_json::Value>,
        value: serde_json::Value,
    ) {
        if let serde_json::Value::Object(map) = value {
            target.extend(map);
        }
    }
    let mut value = serde_json::json!({
        "tick": stats.tick_count,
        "sim_time": stats.sim_time_seconds,
        "width": stats.width,
        "height": stats.height,
        "population": stats.live_cell_count,
        "cell_records": stats.cell_record_count,
        "dead_cells": stats.dead_cell_count,
        "occupancy_fraction": stats.occupancy_fraction,
        "molecules": stats.molecule_count,
        "tile_molecules": stats.tile_molecule_count,
        "cell_molecules": stats.cell_molecule_count,
        "free_molecule_records": stats.free_molecule_record_count,
        "active_molecule_records": stats.active_molecule_record_count,
        "molecule_arena_len": stats.molecule_arena_len,
        "molecule_arena_high_water": stats.molecule_arena_high_water_mark,
        "molecule_slots_reused": stats.molecule_slots_reused,
        "molecule_slots_newly_allocated": stats.molecule_slots_newly_allocated,
    });
    {
        let object = value
            .as_object_mut()
            .expect("top-level stats JSON should be an object");
        merge_object(
            object,
            serde_json::json!({
                "total_atoms": stats.total_atom_count,
                "tile_atoms": stats.tile_atom_count,
                "cell_atoms": stats.cell_atom_count,
                "births": stats.births,
                "deaths": stats.deaths,
                "predation_events": stats.predation_events,
                "cells_consumed": stats.cells_consumed,
                "lineages": stats.lineage_count,
                "dominant_lineage": stats.dominant_lineage_id,
                "dominant_lineage_share": stats.dominant_lineage_share,
                "avg_energy": stats.average_cell_energy,
                "min_energy": stats.min_cell_energy,
                "max_energy": stats.max_cell_energy,
                "avg_enzyme_count": stats.average_enzyme_count,
                "cells_at_enzyme_cap": stats.cells_at_enzyme_cap,
            }),
        );
        object.insert(
            "enzyme_type_totals".to_string(),
            serde_json::json!({
                "anabolase": stats.enzyme_type_totals.anabolase,
                "catabolase": stats.enzyme_type_totals.catabolase,
                "transmutase": stats.enzyme_type_totals.transmutase,
                "defensase": stats.enzyme_type_totals.defensase,
                "attackase": stats.enzyme_type_totals.attackase,
            }),
        );
        object.insert(
            "reactions".to_string(),
            serde_json::json!({
                "attempts": stats.reaction_counters.total_attempts(),
                "gates_passed": stats.reaction_counters.gates_passed_by_type.total(),
                "successes": stats.reaction_counters.total_successes(),
                "no_substrate": stats.reaction_counters.no_substrate_by_type.total(),
                "uptakes": stats.reaction_counters.molecule_uptakes,
                "outputs": stats.reaction_counters.molecule_outputs,
                "divisions": stats.reaction_counters.divisions,
            }),
        );
        object.insert(
            "enval".to_string(),
            serde_json::json!({
                "average": stats.average_enval,
                "min": stats.min_enval,
                "max": stats.max_enval,
                "stddev": stats.enval_std_dev,
                "p05": stats.enval_p05,
                "p50": stats.enval_p50,
                "p95": stats.enval_p95,
            }),
        );
        object.insert(
            "interval".to_string(),
            serde_json::json!({
                "ticks": interval.tick_delta,
                "sim_seconds": interval.sim_seconds_delta,
                "wall_seconds": interval.wall_seconds,
                "population_delta": interval.population_delta,
                "births": interval.births,
                "deaths": interval.deaths,
                "predation_events": interval.predation_events,
                "reaction_attempts": interval.reaction_attempts,
                "reaction_successes": interval.reaction_successes,
                "molecule_uptakes": interval.molecule_uptakes,
                "molecule_outputs": interval.molecule_outputs,
                "cell_steps": interval.cell_steps,
                "cell_steps_per_sec": interval.cell_steps_per_sec,
                "enzyme_attempts_per_sec": interval.enzyme_attempts_per_sec,
            }),
        );
    }
    println!("{}", value);
}

fn print_profile_summary(steps: u64, wall: Duration, mut profile: CliProfile, json: bool) {
    profile.step_durations_ms.sort_by(|a, b| a.total_cmp(b));
    let steps = steps.max(1) as f64;
    let wall_seconds = wall.as_secs_f64().max(1.0e-9);
    let measured_ms = duration_ms(profile.step.total);
    let molecule_ms = duration_ms(profile.step.molecule_diffusion);
    let cell_ms = duration_ms(profile.step.cell_step);
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
                "molecule_ms_per_step": molecule_ms / steps,
                "cells_ms_per_step": cell_ms / steps,
                "predation_ms_per_step": predation_ms / steps,
                "enval_ms_per_step": enval_ms / steps,
                "cell_steps": counters.cell_steps,
                "cell_steps_per_sec": counters.cell_steps as f64 / wall_seconds,
                "enzyme_attempts": counters.metabolic_enzyme_attempts,
                "enzyme_attempts_per_sec": counters.metabolic_enzyme_attempts as f64 / wall_seconds,
                "reactions_succeeded": counters.reactions_succeeded,
                "reactions_per_sec": counters.reactions_succeeded as f64 / wall_seconds,
                "predation_pairs_checked": counters.predation_pairs_checked,
                "predation_occupied_tiles_considered": counters.predation_occupied_tiles_considered,
                "predation_candidate_neighbor_pairs": counters.predation_candidate_neighbor_pairs,
                "predation_cross_lineage_pairs": counters.predation_cross_lineage_pairs,
                "combat_enzyme_skips": counters.combat_enzyme_skips,
                "molecule_diffusion_events": counters.molecule_diffusion_events,
                "molecule_moves": counters.molecule_moves,
                "molecule_slots_reused": counters.molecule_slots_reused,
                "molecule_slots_newly_allocated": counters.molecule_slots_newly_allocated
            }
        });
        println!("{}", value);
        return;
    }
    println!(
        "profile wall_ms={:.3} avg_step_ms={:.6} min_step_ms={:.6} max_step_ms={:.6} p50_step_ms={:.6} p95_step_ms={:.6} molecule_ms={:.6} cells_ms={:.6} predation_ms={:.6} enval_ms={:.6} measured_total_ms={:.6} molecule_pct={:.2} cells_pct={:.2} predation_pct={:.2} enval_pct={:.2} stats_output_ms={:.6} invariants_ms={:.6} snapshot_io_ms={:.6} csv_flush_ms={:.6} cell_steps={} cell_steps_per_sec={:.3} enzyme_entries={} enzyme_attempts={} enzyme_attempts_per_sec={:.3} reaction_gates={} reactions={} reactions_per_sec={:.3} substrate_candidates={} molecule_diffusion_events={} molecule_moves={} uptakes={} products={} byproducts={} divisions={} deaths={} predation_pairs={} predation_occupied_tiles={} predation_candidates={} predation_cross_lineage={} predation_events={} consumed={} combat_enzyme_skips={} enval_avg_calls={} molecule_slots_reused={} molecule_slots_new={} enzyme_list_clones={} genome_clones={}",
        duration_ms(wall),
        measured_ms / steps,
        profile.min_step.map(duration_ms).unwrap_or(0.0),
        duration_ms(profile.max_step),
        p50,
        p95,
        molecule_ms / steps,
        cell_ms / steps,
        predation_ms / steps,
        enval_ms / steps,
        measured_ms / steps,
        pct(molecule_ms),
        pct(cell_ms),
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
        counters.reaction_gates_passed,
        counters.reactions_succeeded,
        counters.reactions_succeeded as f64 / wall_seconds,
        counters.substrate_candidates_scanned,
        counters.molecule_diffusion_events,
        counters.molecule_moves,
        counters.molecule_uptakes,
        counters.products_created,
        counters.byproducts_created,
        counters.cell_divisions,
        counters.cell_deaths,
        counters.predation_pairs_checked,
        counters.predation_occupied_tiles_considered,
        counters.predation_candidate_neighbor_pairs,
        counters.predation_cross_lineage_pairs,
        counters.predation_events,
        counters.predation_cells_consumed,
        counters.combat_enzyme_skips,
        counters.local_enval_average_calls,
        counters.molecule_slots_reused,
        counters.molecule_slots_newly_allocated,
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
  microcosm run [--seed S] [--width W] [--height H] [--initial-cells N] [--steps N] [--stats-every N] [--stats-mode compact|full] [--json] [--check-invariants] [--check-invariants-every N] [--csv [path]] [--snapshot-in path] [--snapshot-out [path]] [--profile|--profile-json] [--no-predation] [--trace off]
  microcosm bench [--seed S] [--width W] [--height H] [--initial-cells N] [--steps N] [--stats-every N] [--stats-mode compact|full] [--json] [--profile|--profile-json] [--no-predation] [--trace off]
  microcosm inspect snapshot.micosm"
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
        assert!(columns.contains(&"tile_molecules"));
        assert!(columns.contains(&"cell_molecules"));
        assert!(columns.contains(&"active_molecule_records"));
        assert!(columns.contains(&"molecule_arena_len"));
        assert!(columns.contains(&"molecule_arena_high_water"));
        assert!(columns.contains(&"molecule_slots_reused"));
        assert!(columns.contains(&"molecule_slots_newly_allocated"));
        assert!(columns.contains(&"rx_successes"));
        assert!(columns.contains(&"interval_cell_steps"));
        assert!(columns.len() > 80);
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
