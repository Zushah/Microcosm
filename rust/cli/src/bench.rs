use std::process::ExitCode;
use std::time::{Duration, Instant};

use microcosmcore::{StepProfile, World};

use crate::args::{BenchOptions, should_check_invariants};
use crate::output::{StatsEmissionState, emit_bench_stats};

#[derive(Clone, Debug, Default)]
pub struct CliProfile {
    pub step: StepProfile,
    pub step_count: u64,
    pub min_step: Option<Duration>,
    pub max_step: Duration,
    pub step_durations_ms: Vec<f64>,
    pub stats_output: Duration,
    pub invariants: Duration,
    pub snapshot_io: Duration,
    pub csv_flush: Duration,
}

impl CliProfile {
    pub fn add_step(&mut self, profile: StepProfile) {
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

pub fn bench_command(options: BenchOptions) -> ExitCode {
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

    if let Some(energy) = options.initial_energy
        && let Err(err) = world.set_all_live_cell_energy(energy)
    {
        eprintln!("failed to set benchmark cell energy: {err}");
        return ExitCode::from(1);
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
    if !options.quiet
        && options.steps > 0
        && !options.steps.is_multiple_of(options.stats_every.max(1))
    {
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

pub fn print_profile_summary(steps: u64, wall: Duration, mut profile: CliProfile, json: bool) {
    profile.step_durations_ms.sort_by(|a, b| a.total_cmp(b));
    let steps = steps.max(1) as f64;
    let wall_seconds = wall.as_secs_f64().max(1.0e-9);
    let measured_ms = duration_ms(profile.step.total);
    let element_ms = duration_ms(profile.step.element_field_diffusion);
    let cell_ms = duration_ms(profile.step.cell_step);
    let mechanics_ms = duration_ms(profile.step.cell_mechanics);
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
                "enval_ms_per_step": enval_ms / steps,
                "cell_steps": counters.cell_steps,
                "cell_steps_per_sec": counters.cell_steps as f64 / wall_seconds,
                "enzyme_attempts": counters.metabolic_enzyme_attempts,
                "enzyme_attempts_per_sec": counters.metabolic_enzyme_attempts as f64 / wall_seconds,
                "reactions_succeeded": counters.reactions_succeeded,
                "reactions_per_sec": counters.reactions_succeeded as f64 / wall_seconds,
                "spatial_candidate_checks": counters.spatial_candidate_checks,
                "overlap_candidates": counters.overlap_candidates,
                "overlap_corrections": counters.overlap_corrections,
                "surface_contact_queries": counters.surface_contact_queries,
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
        "profile wall_ms={:.3} avg_step_ms={:.6} min_step_ms={:.6} max_step_ms={:.6} p50_step_ms={:.6} p95_step_ms={:.6} element_field_ms={:.6} cells_ms={:.6} mechanics_ms={:.6} enval_ms={:.6} measured_total_ms={:.6} element_field_pct={:.2} cells_pct={:.2} mechanics_pct={:.2} enval_pct={:.2} stats_output_ms={:.6} invariants_ms={:.6} snapshot_io_ms={:.6} csv_flush_ms={:.6} cell_steps={} cell_steps_per_sec={:.3} enzyme_entries={} enzyme_attempts={} enzyme_attempts_per_sec={:.3} reactions={} reactions_per_sec={:.3} element_field_tiles={} uptake_events={} divisions={} deaths={} spatial_candidates={} overlap_candidates={} overlap_corrections={} surface_contact_queries={} enval_avg_calls={}",
        duration_ms(wall),
        measured_ms / steps,
        profile.min_step.map(duration_ms).unwrap_or(0.0),
        duration_ms(profile.max_step),
        p50,
        p95,
        element_ms / steps,
        cell_ms / steps,
        mechanics_ms / steps,
        enval_ms / steps,
        measured_ms / steps,
        pct(element_ms),
        pct(cell_ms),
        pct(mechanics_ms),
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
        counters.spatial_candidate_checks,
        counters.overlap_candidates,
        counters.overlap_corrections,
        counters.surface_contact_queries,
        counters.local_enval_average_calls,
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

pub fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}
