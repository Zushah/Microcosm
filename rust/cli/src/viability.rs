use std::process::ExitCode;
use std::time::Instant;

use microcosmcore::{Config, World, renewable_coverage};

use crate::args::ViabilityOptions;
use crate::seeds::run_seeds;
use crate::table::print_table;

pub const VIABILITY_DEFAULT_SEEDS: [&str; 5] = ["42", "1", "2", "3", "7"];
pub const VIABILITY_DEFAULT_STEPS: u64 = 60_000;
const VIABILITY_SAMPLE_EVERY: u64 = 100;
const VIABILITY_INVARIANTS_EVERY: u64 = 1_000;
const VIABILITY_PROGRESS_EVERY: u64 = 5_000;
const VIABILITY_MIN_LIVE_AFTER_TICK: u64 = 1_000;
pub const VIABILITY_MAX_CELLS_PER_TILE: f64 = 0.75;
const VIABILITY_MIN_RENEWABLE_COVERAGE: f64 = 0.25;
const VIABILITY_RADIUS_MEDIAN_RANGE: std::ops::RangeInclusive<f32> = 0.2..=1.0;
const VIABILITY_MAX_RADIUS: f32 = 4.0;

#[derive(Debug, Clone, Default)]
pub struct ViabilityResult {
    pub seed: String,
    pub passed: bool,
    pub law_failures: Vec<String>,
    pub survival_failures: Vec<String>,
    pub final_tick: u64,
    pub extinction_tick: Option<u64>,
    pub min_live_after_warmup: Option<usize>,
    pub peak_live: usize,
    pub peak_tick: u64,
    pub final_live: usize,
    pub population_cap: usize,
    pub enval_harvest: f64,
    pub chemical_harvest: f64,
    pub recharge_energy: f64,
    pub renewable_coverage: f64,
    pub max_ledger_residual: f64,
    pub max_enval_residual: f64,
    pub max_enval_relative_residual: f64,
    pub max_matter_drift: f64,
    pub final_matter_drift: f64,
    pub fissions: u64,
    pub energy_deaths: u64,
    pub lyses: u64,
    pub total_deaths: u64,
    pub radius_median: f32,
    pub radius_p95: f32,
    pub radius_max: f32,
    pub max_radius_seen: f32,
    pub min_median_seen: f32,
    pub max_median_seen: f32,
    pub median_generation: u32,
    pub max_generation: u32,
    pub wall_seconds: f64,
}

impl ViabilityResult {
    pub fn setup_failure(seed: &str, failure: String) -> Self {
        Self {
            seed: seed.to_owned(),
            law_failures: vec![failure],
            ..Self::default()
        }
    }

    pub fn has_law_failure(&self) -> bool {
        !self.law_failures.is_empty()
    }

    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "seed": self.seed,
            "passed": self.passed,
            "law_failures": self.law_failures,
            "survival_failures": self.survival_failures,
            "final_tick": self.final_tick,
            "extinction_tick": self.extinction_tick,
            "min_live_after_tick_1000": self.min_live_after_warmup,
            "peak_live": self.peak_live,
            "peak_tick": self.peak_tick,
            "final_live": self.final_live,
            "population_cap": self.population_cap,
            "enval_harvest": self.enval_harvest,
            "chemical_harvest": self.chemical_harvest,
            "recharge_energy": self.recharge_energy,
            "renewable_coverage": self.renewable_coverage,
            "max_ledger_residual": self.max_ledger_residual,
            "max_enval_residual": self.max_enval_residual,
            "max_enval_relative_residual": self.max_enval_relative_residual,
            "max_matter_drift": self.max_matter_drift,
            "final_matter_drift": self.final_matter_drift,
            "fissions": self.fissions,
            "energy_deaths": self.energy_deaths,
            "lyses": self.lyses,
            "total_deaths": self.total_deaths,
            "radius_median": self.radius_median,
            "radius_p95": self.radius_p95,
            "radius_max": self.radius_max,
            "max_radius_seen": self.max_radius_seen,
            "min_median_seen": self.min_median_seen,
            "max_median_seen": self.max_median_seen,
            "median_generation": self.median_generation,
            "max_generation": self.max_generation,
            "wall_seconds": self.wall_seconds,
        })
    }

    pub fn failure_lines(&self) -> impl Iterator<Item = String> + '_ {
        self.law_failures
            .iter()
            .map(|failure| format!("law: {failure}"))
            .chain(
                self.survival_failures
                    .iter()
                    .map(|failure| format!("survival: {failure}")),
            )
    }
}

pub struct ViabilityMonitor {
    result: ViabilityResult,
    steps: u64,
    label: &'static str,
    wall_start: Instant,
    energy_ledger_closed: bool,
    enval_ledger_closed: bool,
}

impl ViabilityMonitor {
    pub fn start(world: &World, seed: &str, steps: u64, label: &'static str) -> Self {
        let population_cap =
            (VIABILITY_MAX_CELLS_PER_TILE * world.tile_count() as f64).floor() as usize;
        Self {
            result: ViabilityResult {
                seed: seed.to_owned(),
                min_median_seen: f32::INFINITY,
                peak_live: world.cell_count(),
                final_live: world.cell_count(),
                population_cap,
                ..ViabilityResult::default()
            },
            steps,
            label,
            wall_start: Instant::now(),
            energy_ledger_closed: true,
            enval_ledger_closed: true,
        }
    }

    pub fn observe(&mut self, world: &World, tick: u64, quiet: bool) -> bool {
        let is_final = tick == self.steps;
        let sample = tick.is_multiple_of(VIABILITY_SAMPLE_EVERY) || is_final;
        let check_invariants = tick.is_multiple_of(VIABILITY_INVARIANTS_EVERY) || is_final;
        if !sample && !check_invariants {
            return true;
        }
        let result = &mut self.result;
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
        if residual > world.energy_ledger().closure_tolerance() && self.energy_ledger_closed {
            self.energy_ledger_closed = false;
            result.law_failures.push(format!(
                "energy ledger residual {residual:.3e} exceeds tolerance at tick {tick}"
            ));
        }
        let enval_ledger = world.enval_ledger();
        let enval_residual = world.enval_ledger_residual().abs();
        let enval_magnitude = enval_ledger.magnitude();
        result.max_enval_residual = result.max_enval_residual.max(enval_residual);
        if enval_magnitude > 0.0 {
            result.max_enval_relative_residual = result
                .max_enval_relative_residual
                .max(enval_residual / enval_magnitude);
        }
        let enval_closed = enval_residual <= enval_ledger.closure_tolerance();
        if !enval_closed && self.enval_ledger_closed {
            self.enval_ledger_closed = false;
            result.law_failures.push(format!(
                "enval ledger residual {enval_residual:.3e} exceeds tolerance {:.3e} at tick {tick}",
                enval_ledger.closure_tolerance()
            ));
        }
        let matter_drift = world.matter_drift();
        result.max_matter_drift = result.max_matter_drift.max(matter_drift.abs());
        if check_invariants && let Err(err) = world.check_invariants() {
            result
                .law_failures
                .push(format!("invariant failure at tick {tick}: {err}"));
            return false;
        }

        if live == 0 {
            result.extinction_tick = Some(tick);
            result
                .survival_failures
                .push(format!("extinct at tick {tick}"));
            return false;
        }
        if live > result.population_cap {
            result.survival_failures.push(format!(
                "population {live} exceeds cap {} at tick {tick}",
                result.population_cap
            ));
            return false;
        }
        if sample {
            let stats = world.compact_stats();
            result.max_radius_seen = result.max_radius_seen.max(stats.radius_max);
            result.min_median_seen = result.min_median_seen.min(stats.radius_median);
            result.max_median_seen = result.max_median_seen.max(stats.radius_median);
            if !VIABILITY_RADIUS_MEDIAN_RANGE.contains(&stats.radius_median)
                || stats.radius_max > VIABILITY_MAX_RADIUS
            {
                result.survival_failures.push(format!(
                    "sizes out of bounds at tick {tick}: median={} max={}",
                    stats.radius_median, stats.radius_max
                ));
                return false;
            }
        }

        if !quiet && tick.is_multiple_of(VIABILITY_PROGRESS_EVERY) {
            let coverage = renewable_coverage(&world.energy_ledger(), &enval_ledger);
            eprintln!(
                "{} seed={} tick={} cells={} peak={} coverage={:.3} matter_drift={:+.2e} wall={:.1}s",
                self.label,
                result.seed,
                tick,
                live,
                result.peak_live,
                coverage,
                matter_drift,
                self.wall_start.elapsed().as_secs_f64()
            );
        }
        true
    }

    pub fn finish(self, world: &World) -> ViabilityResult {
        let Self {
            mut result,
            steps,
            wall_start,
            ..
        } = self;
        let energy = world.energy_ledger();
        let enval = world.enval_ledger();
        result.enval_harvest = energy.enval_harvest;
        result.chemical_harvest = energy.chemical_harvest;
        result.recharge_energy = enval.recharge_energy;
        result.renewable_coverage = renewable_coverage(&energy, &enval);
        let stats = world.compact_stats();
        result.final_live = stats.live_cell_count;
        result.final_tick = stats.tick_count;
        result.fissions = stats.fissions;
        result.energy_deaths = stats.energy_deaths;
        result.lyses = stats.lyses;
        result.total_deaths = stats.deaths;
        result.radius_median = stats.radius_median;
        result.radius_p95 = stats.radius_p95;
        result.radius_max = stats.radius_max;
        if !result.min_median_seen.is_finite() {
            result.min_median_seen = 0.0;
        }
        result.final_matter_drift = world.matter_drift();
        result.max_matter_drift = result.max_matter_drift.max(result.final_matter_drift.abs());
        let mut generations = world
            .cells()
            .map(|cell| cell.generation)
            .collect::<Vec<_>>();
        generations.sort_unstable();
        if let Some(max) = generations.last() {
            result.max_generation = *max;
            result.median_generation =
                generations[((generations.len() - 1) as f64 * 0.5).round() as usize];
        }
        if result.extinction_tick.is_none()
            && result.final_tick == steps
            && result.renewable_coverage < VIABILITY_MIN_RENEWABLE_COVERAGE
        {
            result.survival_failures.push(format!(
                "renewable coverage {:.3} is below {VIABILITY_MIN_RENEWABLE_COVERAGE}",
                result.renewable_coverage
            ));
        }
        result.passed = result.law_failures.is_empty()
            && result.survival_failures.is_empty()
            && result.final_tick == steps;
        result.wall_seconds = wall_start.elapsed().as_secs_f64();
        result
    }
}

pub fn create_world(config: Config) -> Result<World, String> {
    let founders = config.initial_founder_count;
    let mut world = World::new(config).map_err(|err| format!("world creation failed: {err}"))?;
    world
        .spawn_founder_cells(founders)
        .map_err(|err| format!("founder spawn failed: {err}"))?;
    Ok(world)
}

pub fn viability_command(options: ViabilityOptions) -> ExitCode {
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

    let results = run_seeds(&options.seeds, options.jobs, |seed| {
        let mut config = options.config.clone();
        config.seed = seed.to_owned();
        run_viability_world(config, options.steps, options.quiet)
    });

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

pub fn run_viability_world(config: Config, steps: u64, quiet: bool) -> ViabilityResult {
    let seed = config.seed.clone();
    let mut world = match create_world(config) {
        Ok(world) => world,
        Err(failure) => return ViabilityResult::setup_failure(&seed, failure),
    };
    let mut monitor = ViabilityMonitor::start(&world, &seed, steps, "viability");
    for tick in 1..=steps {
        world.step();
        if !monitor.observe(&world, tick, quiet) {
            break;
        }
    }
    monitor.finish(&world)
}

pub fn verdict_label(passed: bool) -> &'static str {
    if passed { "PASS" } else { "FAIL" }
}

fn print_viability_table(results: &[ViabilityResult], steps: u64) {
    let header = [
        "seed",
        "pass",
        "tick",
        "cells",
        "peak",
        "min>1k",
        "cap",
        "coverage",
        "matter_drift",
        "energy_resid",
        "enval_resid_rel",
        "r_med",
        "r_max",
        "gen_med",
        "gen_max",
        "fissions",
        "deaths",
        "wall_s",
    ];
    let rows = results
        .iter()
        .map(|result| {
            vec![
                result.seed.clone(),
                if result.passed { "yes" } else { "NO" }.to_owned(),
                result.final_tick.to_string(),
                result.final_live.to_string(),
                result.peak_live.to_string(),
                result
                    .min_live_after_warmup
                    .map_or_else(|| "-".to_owned(), |cells| cells.to_string()),
                result.population_cap.to_string(),
                format!("{:.3}", result.renewable_coverage),
                format!("{:.2e}", result.max_matter_drift),
                format!("{:.2e}", result.max_ledger_residual),
                format!("{:.2e}", result.max_enval_relative_residual),
                format!("{:.3}", result.radius_median),
                format!("{:.3}", result.max_radius_seen),
                result.median_generation.to_string(),
                result.max_generation.to_string(),
                result.fissions.to_string(),
                result.total_deaths.to_string(),
                format!("{:.1}", result.wall_seconds),
            ]
        })
        .collect::<Vec<_>>();
    print_table("", &header, &rows);
    for result in results {
        for line in result.failure_lines() {
            println!("seed {}: {line}", result.seed);
        }
    }
    let passed = results.iter().filter(|result| result.passed).count();
    println!(
        "viability {} {}/{} seeds passed over {} ticks",
        verdict_label(passed == results.len()),
        passed,
        results.len(),
        steps
    );
}

fn print_viability_json(results: &[ViabilityResult], steps: u64) {
    let value = serde_json::json!({
        "viability": {
            "steps": steps,
            "passed": results.iter().all(|result| result.passed),
            "seeds": results.iter().map(ViabilityResult::json).collect::<Vec<_>>(),
        }
    });
    println!("{}", value);
}
