use std::env;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use chrono::Local;
use microcosmcore::{SNAPSHOT_EXTENSION, World, WorldStats};

use crate::args::{OutputDestination, RunOptions, StatsMode};
use crate::bench::CliProfile;
use crate::stats_table::{COMPACT_TEXT, FULL_TEXT, csv_header, csv_row, json_record, render_text};

const CSV_EXTENSION: &str = "csv";
const DEFAULT_OUTPUT_DIR: &str = "out";

#[derive(Debug, Clone, Default)]
pub struct ResolvedOutputPaths {
    pub csv: Option<PathBuf>,
    pub snapshot_out: Option<PathBuf>,
}

#[derive(Clone, Debug, Default)]
pub struct StatsEmissionState {
    previous_stats: Option<WorldStats>,
    previous_wall: Option<Instant>,
}

impl StatsEmissionState {
    pub fn capture_interval(&mut self, current: &WorldStats, now: Instant) -> StatsInterval {
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
pub struct StatsInterval {
    pub tick_delta: u64,
    pub sim_seconds_delta: f64,
    pub wall_seconds: f64,
    pub population_delta: i64,
    pub births: u64,
    pub deaths: u64,
    pub divisions: u64,
    pub reaction_attempts: u64,
    pub reaction_successes: u64,
    pub executed_metabolic_flux: f64,
    pub uptake_flux: f64,
    pub leak_flux: f64,
    pub secretion_flux: f64,
    pub cell_steps: u64,
    pub enzyme_attempts: u64,
    pub steps_per_sec: f64,
    pub cell_steps_per_sec: f64,
    pub enzyme_attempts_per_sec: f64,
    pub reactions_per_sec: f64,
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

pub fn resolve_run_outputs(options: &RunOptions) -> std::io::Result<ResolvedOutputPaths> {
    let cwd = env::current_dir()?;
    let timestamp = local_timestamp();
    resolve_output_requests(
        options.csv.as_ref(),
        options.snapshot_out.as_ref(),
        &cwd,
        &timestamp,
    )
}

pub fn resolve_output_requests(
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
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    Ok(path)
}

fn local_timestamp() -> String {
    Local::now().format("%m%d%Y_%H%M%S").to_string()
}

pub fn open_csv(path: Option<&PathBuf>) -> Result<Option<BufWriter<File>>, std::io::Error> {
    let Some(path) = path else {
        return Ok(None);
    };
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(writer, "{}", csv_header())?;
    Ok(Some(writer))
}

#[derive(Clone, Copy, Debug)]
pub struct StatsOutput {
    pub print: bool,
    pub profile: bool,
    pub mode: StatsMode,
    pub json: bool,
}

pub fn emit_world_stats_profiled(
    world: &World,
    csv: &mut Option<BufWriter<File>>,
    output: StatsOutput,
    profile: &mut CliProfile,
    state: &mut StatsEmissionState,
) -> std::io::Result<()> {
    let start = Instant::now();
    let stats = collect_stats_for_output(world, csv.is_some(), output.mode, output.json);
    let interval = state.capture_interval(&stats, Instant::now());
    if output.print {
        print_stats_record(&stats, &interval, output.mode, output.json);
    }
    if let Some(writer) = csv.as_mut() {
        write_csv_record(writer, &stats, &interval)?;
    }
    if output.profile {
        profile.stats_output += start.elapsed();
    }
    Ok(())
}

pub fn emit_bench_stats(
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

pub fn collect_stats_for_output(
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

pub fn write_csv_record<W: Write>(
    writer: &mut W,
    stats: &WorldStats,
    interval: &StatsInterval,
) -> std::io::Result<()> {
    writeln!(writer, "{}", csv_row(stats, interval))
}

pub fn print_stats_record(
    stats: &WorldStats,
    interval: &StatsInterval,
    mode: StatsMode,
    json: bool,
) {
    if json {
        println!("{}", json_record(stats, interval));
        return;
    }
    match mode {
        StatsMode::Compact => print_text_lines(COMPACT_TEXT, stats, interval),
        StatsMode::Full => print_text_lines(FULL_TEXT, stats, interval),
    }
}

pub fn print_text_lines(lines: &[&str], stats: &WorldStats, interval: &StatsInterval) {
    for line in lines {
        println!("{}", render_text(line, stats, interval));
    }
}
