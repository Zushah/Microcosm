use std::path::PathBuf;

use microcosmcore::Config;

use crate::viability::{VIABILITY_DEFAULT_SEEDS, VIABILITY_DEFAULT_STEPS};

#[derive(Debug, Clone)]
pub enum Command {
    Run(RunOptions),
    Bench(BenchOptions),
    Inspect(InspectOptions),
    Viability(ViabilityOptions),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutputDestination {
    Default,
    Path(PathBuf),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatsMode {
    Compact,
    Full,
}

#[derive(Debug, Clone)]
pub struct RunOptions {
    pub config: Config,
    pub initial_cells: Option<usize>,
    pub initial_energy: Option<f64>,
    pub steps: u64,
    pub stats_every: u64,
    pub check_invariants: bool,
    pub check_invariants_every: u64,
    pub csv: Option<OutputDestination>,
    pub snapshot_in: Option<PathBuf>,
    pub snapshot_out: Option<OutputDestination>,
    pub profile: bool,
    pub profile_json: bool,
    pub quiet: bool,
    pub stats_mode: StatsMode,
    pub json: bool,
    pub until_cells: Option<usize>,
    pub fingerprint: bool,
    pub fingerprint_every: u64,
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
            until_cells: None,
            fingerprint: false,
            fingerprint_every: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BenchOptions {
    pub config: Config,
    pub initial_cells: usize,
    pub initial_energy: Option<f64>,
    pub steps: u64,
    pub stats_every: u64,
    pub check_invariants_every: u64,
    pub profile: bool,
    pub profile_json: bool,
    pub quiet: bool,
    pub stats_mode: StatsMode,
    pub json: bool,
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
pub struct InspectOptions {
    pub snapshot: PathBuf,
}

#[derive(Debug, Clone)]
pub struct ViabilityOptions {
    pub config: Config,
    pub seeds: Vec<String>,
    pub steps: u64,
    pub jobs: usize,
    pub json: bool,
    pub quiet: bool,
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

pub fn parse_args<I>(args: I) -> Result<Option<Command>, String>
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

pub fn parse_run<I>(args: std::iter::Peekable<I>) -> Result<Option<Command>, String>
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

pub fn parse_common_run_args<I>(
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
            "--enval-diffusivity" => {
                options.config.enval_diffusivity_per_second =
                    parse_value(&mut args, "--enval-diffusivity")?;
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
            "--fingerprint" => options.fingerprint = true,
            "--fingerprint-every" => {
                options.fingerprint = true;
                options.fingerprint_every = parse_value(&mut args, "--fingerprint-every")?;
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

pub fn parse_bench<I>(mut args: std::iter::Peekable<I>) -> Result<Option<Command>, String>
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
            "--enval-diffusivity" => {
                options.config.enval_diffusivity_per_second =
                    parse_value(&mut args, "--enval-diffusivity")?;
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

pub fn parse_viability<I>(mut args: std::iter::Peekable<I>) -> Result<Option<Command>, String>
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
            "--enval-diffusivity" => {
                options.config.enval_diffusivity_per_second =
                    parse_value(&mut args, "--enval-diffusivity")?;
            }
            other => {
                if !parse_environment_flag(other, &mut args, &mut options.config)? {
                    return Err(format!("unrecognized argument '{other}'"));
                }
            }
        }
    }
    Ok(Some(Command::Viability(options)))
}

pub fn parse_environment_flag<I>(
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
        "--landscape-magnitude" => config.enval_landscape.magnitude = parse_value(args, arg)?,
        "--landscape-scale" => config.enval_landscape.scale = parse_value(args, arg)?,
        "--landscape-relaxation" => {
            config.enval_landscape.relaxation_per_second = parse_value(args, arg)?;
        }
        "--recharge-rate" => config.enval_recharge.rate_per_second = parse_value(args, arg)?,
        "--heterogeneity" => config.element_fields.heterogeneity = parse_value(args, arg)?,
        "--heterogeneity-scale" => {
            config.element_fields.heterogeneity_scale = parse_value(args, arg)?;
        }
        "--drift" => config.cell_motion.diffusivity = parse_value(args, arg)?,
        "--contact-gap" => config.surface.contact_gap = parse_value(args, arg)?,
        _ => return Ok(false),
    }
    Ok(true)
}

pub fn next_value<I>(args: &mut std::iter::Peekable<I>, flag: &str) -> Result<String, String>
where
    I: Iterator<Item = String>,
{
    args.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

pub fn optional_value<I>(args: &mut std::iter::Peekable<I>) -> Option<String>
where
    I: Iterator<Item = String>,
{
    match args.peek() {
        Some(value) if !looks_like_flag(value) => args.next(),
        _ => None,
    }
}

pub fn looks_like_flag(value: &str) -> bool {
    value.starts_with('-') && value.len() > 1
}

pub fn parse_value<I, T>(args: &mut std::iter::Peekable<I>, flag: &str) -> Result<T, String>
where
    I: Iterator<Item = String>,
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let raw = next_value(args, flag)?;
    raw.parse::<T>()
        .map_err(|err| format!("invalid value for {flag}: {err}"))
}

pub fn parse_stats_mode(raw: &str) -> Result<StatsMode, String> {
    match raw {
        "compact" => Ok(StatsMode::Compact),
        "full" => Ok(StatsMode::Full),
        other => Err(format!(
            "unsupported stats mode '{other}'; expected compact or full"
        )),
    }
}

pub fn parse_trace_mode(raw: &str) -> Result<(), String> {
    match raw {
        "off" => Ok(()),
        other => Err(format!(
            "unsupported trace mode '{other}'; only '--trace off' is currently supported"
        )),
    }
}

pub fn should_check_invariants(enabled: bool, every: u64, step_index: u64) -> bool {
    enabled && every > 0 && step_index.is_multiple_of(every)
}

pub fn print_usage() {
    eprintln!(
        "usage:
  microcosm run [--seed S] [--width W] [--height H] [--initial-cells N] [--steps N] [--until-cells N] [--stats-every N] [--stats-mode compact|full] [--json] [--check-invariants] [--check-invariants-every N] [--csv [path]] [--snapshot-in path] [--snapshot-out [path]] [--profile|--profile-json] [--fingerprint] [--fingerprint-every N] [--trace off]
  microcosm bench [--seed S] [--width W] [--height H] [--initial-cells N] [--steps N] [--stats-every N] [--stats-mode compact|full] [--json] [--profile|--profile-json] [--trace off]
  microcosm inspect snapshot.micosm
  microcosm viability [--seeds 42,1,2,3,7] [--steps 60000] [--jobs N] [--json] [--quiet] [--width W] [--height H] [--initial-cells N]
environment flags (run, bench, viability):
  [--permeability P] [--catalyst-upkeep U] [--landscape-magnitude M] [--landscape-scale S] [--landscape-relaxation K] [--recharge-rate K] [--heterogeneity H] [--heterogeneity-scale S] [--drift D] [--contact-gap G]"
    );
}
