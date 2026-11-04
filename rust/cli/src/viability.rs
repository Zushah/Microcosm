use std::process::ExitCode;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use microcosmcore::{Config, ELEMENT_COUNT, World, renewable_coverage};

use crate::args::ViabilityOptions;

pub const VIABILITY_DEFAULT_SEEDS: [&str; 5] = ["42", "1", "2", "3", "7"];
pub const VIABILITY_DEFAULT_STEPS: u64 = 60_000;
const VIABILITY_SAMPLE_EVERY: u64 = 100;
const VIABILITY_INVARIANTS_EVERY: u64 = 5_000;
const VIABILITY_PROGRESS_EVERY: u64 = 5_000;
const VIABILITY_MIN_LIVE_AFTER_TICK: u64 = 1_000;
const VIABILITY_POPULATION_CAP: usize = 60_000;
const VIABILITY_MIN_RENEWABLE_COVERAGE: f64 = 0.25;
const VIABILITY_POPULATION_MILESTONES: [(u64, usize); 2] = [(20_000, 5_000), (60_000, 10_000)];

#[derive(Debug, Clone, Default)]
pub struct ViabilityResult {
    pub seed: String,
    pub passed: bool,
    pub failures: Vec<String>,
    pub final_tick: u64,
    pub extinction_tick: Option<u64>,
    pub min_live_after_warmup: Option<usize>,
    pub peak_live: usize,
    pub peak_tick: u64,
    pub final_live: usize,
    pub enval_harvest: f64,
    pub chemical_harvest: f64,
    pub recharge_energy: f64,
    pub renewable_coverage: f64,
    pub max_ledger_residual: f64,
    pub initial_totals: [f64; ELEMENT_COUNT],
    pub final_totals: [f64; ELEMENT_COUNT],
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
    pub boundary_digested: f64,
    pub cross_lineage_boundary_digested: f64,
    pub lyses_after_digestion: u64,
    pub mean_exposure: f64,
    pub max_exposure: f32,
    pub exposed_cells: usize,
    pub mixed_polarity_cells: usize,
    pub mean_enval_optimum_spread: f64,
    pub mean_niche_mismatch: f64,
    pub peak_cross_lineage_contacts: u64,
    pub milestone_cells: [Option<usize>; 2],
    pub wall_seconds: f64,
}

impl ViabilityResult {
    fn relative_total_drift(&self) -> f64 {
        let initial = self.initial_totals.iter().sum::<f64>();
        let final_total = self.final_totals.iter().sum::<f64>();
        (final_total - initial).abs() / initial.abs().max(1.0e-12)
    }
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

pub fn run_viability_world(config: Config, steps: u64, quiet: bool) -> ViabilityResult {
    let wall_start = Instant::now();
    let mut result = ViabilityResult {
        seed: config.seed.clone(),
        min_median_seen: f32::INFINITY,
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
        for (cells, (milestone, _)) in result
            .milestone_cells
            .iter_mut()
            .zip(VIABILITY_POPULATION_MILESTONES)
        {
            if tick == milestone {
                *cells = Some(live);
            }
        }
        if sample {
            let stats = world.compact_stats();
            result.max_radius_seen = result.max_radius_seen.max(stats.radius_max);
            result.min_median_seen = result.min_median_seen.min(stats.radius_median);
            result.max_median_seen = result.max_median_seen.max(stats.radius_median);
            result.peak_cross_lineage_contacts = result
                .peak_cross_lineage_contacts
                .max(stats.cross_lineage_contact_pairs);
            if live > 0 && (!(0.2..=1.0).contains(&stats.radius_median) || stats.radius_max > 4.0) {
                result.failures.push(format!(
                    "sizes out of bounds at tick {tick}: median={} max={}",
                    stats.radius_median, stats.radius_max
                ));
                break;
            }
        }
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
        if check_invariants && let Err(err) = world.check_invariants() {
            result
                .failures
                .push(format!("invariant failure at tick {tick}: {err}"));
            break;
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
    let stats = world.stats();
    result.final_totals = stats.system_element_amounts;
    result.final_live = stats.live_cell_count;
    result.final_tick = stats.tick_count;
    result.fissions = stats.fissions;
    result.energy_deaths = stats.energy_deaths;
    result.lyses = stats.lyses;
    result.total_deaths = stats.deaths;
    result.radius_median = stats.radius_median;
    result.radius_p95 = stats.radius_p95;
    result.radius_max = stats.radius_max;
    result.boundary_digested = stats.interaction.boundary_digested;
    result.cross_lineage_boundary_digested = stats.interaction.cross_lineage_boundary_digested;
    result.lyses_after_digestion = stats.interaction.lyses_after_digestion;
    result.mean_exposure = stats.mean_exposure;
    result.max_exposure = stats.max_exposure;
    result.exposed_cells = stats.exposed_cell_count;
    result.mixed_polarity_cells = stats.mixed_polarity_cells;
    result.mean_enval_optimum_spread = stats.mean_enval_optimum_spread;
    result.mean_niche_mismatch = stats.mean_niche_mismatch;
    if stats.fissions < 200 {
        result
            .failures
            .push(format!("{} fissions is below 200", stats.fissions));
    }
    if stats.lyses as f64 > stats.deaths as f64 * 0.25 {
        result.failures.push(format!(
            "lysis deaths {} exceed 25% of {} deaths",
            stats.lyses, stats.deaths
        ));
    }
    if result.extinction_tick.is_none()
        && result.final_tick == steps
        && result.renewable_coverage < VIABILITY_MIN_RENEWABLE_COVERAGE
    {
        result.failures.push(format!(
            "renewable coverage {:.3} is below {VIABILITY_MIN_RENEWABLE_COVERAGE}",
            result.renewable_coverage
        ));
    }
    result
        .failures
        .extend(population_milestone_failures(&result.milestone_cells));
    result.passed = result.failures.is_empty() && result.final_tick == steps;
    result.wall_seconds = wall_start.elapsed().as_secs_f64();
    result
}

pub fn population_milestone_failures(milestone_cells: &[Option<usize>; 2]) -> Vec<String> {
    milestone_cells
        .iter()
        .zip(VIABILITY_POPULATION_MILESTONES)
        .filter_map(|(cells, (tick, minimum))| {
            cells
                .filter(|cells| *cells < minimum)
                .map(|cells| format!("{cells} live cells at tick {tick} is below {minimum}"))
        })
        .collect()
}

pub fn milestone_column(cells: Option<usize>) -> String {
    cells.map_or_else(|| "-".to_owned(), |cells| cells.to_string())
}

fn print_viability_table(results: &[ViabilityResult], steps: u64) {
    println!(
        "seed pass final peak cells_20k cells_60k coverage fissions energy_deaths lyses/all_deaths radius_median radius_p95 max_radius_seen median_range max_resid AF_total_drift digested cross_digested lyses_after_digestion mean_exposure max_exposure exposed_cells peak_cross_contacts mixed_polarity_cells mean_optimum_spread mean_niche_mismatch wall_s"
    );
    for result in results {
        println!(
            "{} {} {} {} {} {} {:.3} {} {} {}/{} {:.3} {:.3} {:.3} {:.3}..{:.3} {:.2e} {:.2e} {:.3} {:.3} {} {:.4} {:.3} {} {} {} {:.4} {:.4} {:.1}",
            result.seed,
            if result.passed { "yes" } else { "NO" },
            result.final_live,
            result.peak_live,
            milestone_column(result.milestone_cells[0]),
            milestone_column(result.milestone_cells[1]),
            result.renewable_coverage,
            result.fissions,
            result.energy_deaths,
            result.lyses,
            result.total_deaths,
            result.radius_median,
            result.radius_p95,
            result.max_radius_seen,
            result.min_median_seen,
            result.max_median_seen,
            result.max_ledger_residual,
            result.relative_total_drift(),
            result.boundary_digested,
            result.cross_lineage_boundary_digested,
            result.lyses_after_digestion,
            result.mean_exposure,
            result.max_exposure,
            result.exposed_cells,
            result.peak_cross_lineage_contacts,
            result.mixed_polarity_cells,
            result.mean_enval_optimum_spread,
            result.mean_niche_mismatch,
            result.wall_seconds
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
                "cells_20k": result.milestone_cells[0],
                "cells_60k": result.milestone_cells[1],
                "enval_harvest": result.enval_harvest,
                "chemical_harvest": result.chemical_harvest,
                "recharge_energy": result.recharge_energy,
                "renewable_coverage": result.renewable_coverage,
                "max_ledger_residual": result.max_ledger_residual,
                "initial_element_totals": element_amounts_json(result.initial_totals),
                "final_element_totals": element_amounts_json(result.final_totals),
                "relative_total_element_drift": result.relative_total_drift(),
                "fissions": result.fissions,
                "energy_deaths": result.energy_deaths,
                "lyses": result.lyses,
                "total_deaths": result.total_deaths,
                "radius_median": result.radius_median,
                "radius_p95": result.radius_p95,
                "radius_max": result.radius_max,
                "max_radius_seen": result.max_radius_seen,
                "min_median_seen": result.min_median_seen,
                "max_median_seen": result.max_median_seen,
                "boundary_digested": result.boundary_digested,
                "cross_lineage_boundary_digested": result.cross_lineage_boundary_digested,
                "lyses_after_digestion": result.lyses_after_digestion,
                "mean_exposure": result.mean_exposure,
                "max_exposure": result.max_exposure,
                "exposed_cells": result.exposed_cells,
                "mixed_polarity_cells": result.mixed_polarity_cells,
                "mean_enval_optimum_spread": result.mean_enval_optimum_spread,
                "mean_niche_mismatch": result.mean_niche_mismatch,
                "peak_cross_lineage_contacts": result.peak_cross_lineage_contacts,
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
