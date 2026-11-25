use std::process::ExitCode;

use microcosmcore::{
    Census, Config, ELEMENT_ORDER, ENZYME_GENE_NAMES, Element, GENOME_GENE_NAMES, GeneSpread,
    Spread,
};
use serde_json::{Map, Value as Json};

use crate::args::CensusOptions;
use crate::seeds::run_seeds;
use crate::table::print_table;
use crate::viability::{ViabilityMonitor, ViabilityResult, create_world, verdict_label};

pub const CENSUS_SCHEMA: &str = "microcosm.census.v1";
const KIND_TABLE_GENES: [&str; 3] = ["rate", "energy_harvest_fraction", "enval_sigma"];

struct CensusRun {
    seed: String,
    verdict: ViabilityResult,
    samples: Vec<Census>,
}

pub fn census_command(options: CensusOptions) -> ExitCode {
    if options.seeds.is_empty() {
        eprintln!("census requires at least one seed");
        return ExitCode::from(2);
    }
    let mut probe = options.config.clone();
    probe.seed = options.seeds[0].clone();
    if let Err(err) = probe.validate() {
        eprintln!("invalid census config: {err}");
        return ExitCode::from(2);
    }

    let runs = run_seeds(&options.seeds, options.jobs, |seed| {
        let mut config = options.config.clone();
        config.seed = seed.to_owned();
        run_census_world(config, options.steps, options.every, options.quiet)
    });

    if options.json {
        print_census_json(&runs, &options);
    } else {
        print_census_tables(&runs, &options);
    }
    if runs.iter().any(|run| run.verdict.has_law_failure()) {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}

fn run_census_world(config: Config, steps: u64, every: u64, quiet: bool) -> CensusRun {
    let seed = config.seed.clone();
    let mut world = match create_world(config) {
        Ok(world) => world,
        Err(failure) => {
            return CensusRun {
                verdict: ViabilityResult::setup_failure(&seed, failure),
                seed,
                samples: Vec::new(),
            };
        }
    };
    let mut monitor = ViabilityMonitor::start(&world, &seed, steps, "census");
    let mut samples = vec![world.census()];
    for tick in 1..=steps {
        world.step();
        let keep_going = monitor.observe(&world, tick, quiet);
        if tick.is_multiple_of(every) || tick == steps || !keep_going {
            samples.push(world.census());
        }
        if !keep_going {
            break;
        }
    }
    CensusRun {
        verdict: monitor.finish(&world),
        seed,
        samples,
    }
}

struct Rates {
    seconds: f64,
    terms: Vec<(&'static str, &'static str, f64)>,
}

impl Rates {
    fn between(previous: &Census, current: &Census) -> Option<Self> {
        let seconds = current.sim_time_seconds - previous.sim_time_seconds;
        if seconds <= 0.0 {
            return None;
        }
        let per_second = |now: f64, before: f64| (now - before) / seconds;
        let mut terms = Vec::new();
        for ((name, now), (_, before)) in current
            .energy_ledger
            .terms()
            .into_iter()
            .zip(previous.energy_ledger.terms())
        {
            terms.push(("energy", name, per_second(now, before)));
        }
        for ((name, now), (_, before)) in current
            .enval_ledger
            .terms()
            .into_iter()
            .zip(previous.enval_ledger.terms())
        {
            terms.push(("enval", name, per_second(now, before)));
        }
        for (name, now, before) in [
            ("births", current.births, previous.births),
            ("deaths", current.deaths, previous.deaths),
            ("fissions", current.fissions, previous.fissions),
        ] {
            terms.push(("population", name, per_second(now as f64, before as f64)));
        }
        Some(Self { seconds, terms })
    }

    fn json(&self) -> Json {
        let mut groups = Map::new();
        groups.insert("seconds".to_owned(), self.seconds.into());
        for (group, name, value) in &self.terms {
            let entry = groups
                .entry(*group)
                .or_insert_with(|| Json::Object(Map::new()));
            if let Json::Object(terms) = entry {
                terms.insert((*name).to_owned(), (*value).into());
            }
        }
        Json::Object(groups)
    }
}

fn print_census_json(runs: &[CensusRun], options: &CensusOptions) {
    let seeds = runs
        .iter()
        .map(|run| {
            let samples = run
                .samples
                .iter()
                .enumerate()
                .map(|(index, sample)| {
                    let mut value = serde_json::to_value(sample).expect("census serializes");
                    let rates = index
                        .checked_sub(1)
                        .and_then(|previous| Rates::between(&run.samples[previous], sample))
                        .map(|rates| rates.json());
                    if let Json::Object(map) = &mut value {
                        map.insert("rates".to_owned(), rates.unwrap_or(Json::Null));
                    }
                    value
                })
                .collect::<Vec<_>>();
            serde_json::json!({
                "seed": run.seed,
                "extinction_tick": run.verdict.extinction_tick,
                "verdict": run.verdict.json(),
                "samples": samples,
            })
        })
        .collect::<Vec<_>>();
    let value = serde_json::json!({
        "census": {
            "schema": CENSUS_SCHEMA,
            "steps": options.steps,
            "every": options.every,
            "elements": ELEMENT_ORDER.iter().map(element_letter).collect::<Vec<_>>(),
            "enzyme_genes": ENZYME_GENE_NAMES,
            "genome_genes": GENOME_GENE_NAMES,
            "seeds": seeds,
        }
    });
    println!("{value}");
}

fn print_census_tables(runs: &[CensusRun], options: &CensusOptions) {
    for (index, run) in runs.iter().enumerate() {
        if index > 0 {
            println!();
        }
        print_census_run(run, options);
    }
    let passed = runs.iter().filter(|run| run.verdict.passed).count();
    let law_failures = runs
        .iter()
        .filter(|run| run.verdict.has_law_failure())
        .count();
    println!();
    println!(
        "census {}/{} seeds passed viability over {} ticks; {} with law failures",
        passed,
        runs.len(),
        options.steps,
        law_failures
    );
}

fn print_census_run(run: &CensusRun, options: &CensusOptions) {
    println!(
        "census seed={}  steps={}  every={}  verdict={}",
        run.seed,
        options.steps,
        options.every,
        verdict_label(run.verdict.passed)
    );
    for line in run.verdict.failure_lines() {
        println!("  {line}");
    }
    let (Some(first), Some(last)) = (run.samples.first(), run.samples.last()) else {
        return;
    };

    println!();
    println!("time series");
    let mut header = vec![
        "tick",
        "cells",
        "births",
        "deaths",
        "gen_p50",
        "gen_max",
        "lineages",
        "dom_share",
    ];
    header.extend(["A", "B", "C", "D", "E", "F"]);
    header.extend(["enval_mu", "enval_sd", "flat", "coverage", "matter_drift"]);
    let rows = run
        .samples
        .iter()
        .map(|sample| {
            let mut row = vec![
                sample.tick.to_string(),
                sample.live_cells.to_string(),
                sample.births.to_string(),
                sample.deaths.to_string(),
                spread_value(sample.generation.as_ref(), |spread| spread.p50, 0),
                spread_value(sample.generation.as_ref(), |spread| spread.max, 0),
                sample.extant_lineages.to_string(),
                format!("{:.3}", sample.dominant_lineage_share),
            ];
            row.extend(
                sample
                    .field_mean_per_tile
                    .iter()
                    .map(|value| format!("{value:.3}")),
            );
            row.extend([
                format!("{:+.4}", sample.enval_mean),
                format!("{:.4}", sample.enval_std_dev),
                format!("{:.3}", sample.enval_flattening),
                format!("{:.3}", sample.renewable_coverage),
                format!("{:+.2e}", sample.matter_drift),
            ]);
            row
        })
        .collect::<Vec<_>>();
    print_table("  ", &header, &rows);

    println!();
    println!(
        "genes: tick {} (first) against tick {} (last)",
        first.tick, last.tick
    );
    let header = [
        "gene", "min", "p05", "p50", "p95", "max", "|", "min", "p05", "p50", "p95", "max",
    ];
    let gene_rows = first
        .enzyme_genes
        .iter()
        .zip(&last.enzyme_genes)
        .chain(first.genome_genes.iter().zip(&last.genome_genes))
        .map(|(before, after)| gene_row(before, after))
        .collect::<Vec<_>>();
    print_table("  ", &header, &gene_rows);

    println!();
    println!("structure at tick {}", last.tick);
    let header = ["measure", "min", "p05", "p50", "p95", "max"];
    let rows = [
        ("enzymes_per_genome", last.enzymes_per_genome.as_ref()),
        (
            "stoichiometric_entries_per_enzyme",
            last.stoichiometric_entries_per_enzyme.as_ref(),
        ),
    ]
    .into_iter()
    .map(|(name, spread)| {
        let mut row = vec![name.to_owned()];
        row.extend(spread_columns(spread, 2));
        row
    })
    .collect::<Vec<_>>();
    print_table("  ", &header, &rows);
    println!(
        "  zero_exposure_enzyme_fraction {:.4}  exposed_cell_fraction {:.4}",
        last.zero_exposure_enzyme_fraction, last.exposed_cell_fraction
    );

    println!();
    let total_enzymes = last
        .reaction_kinds
        .iter()
        .map(|kind| kind.enzymes)
        .sum::<usize>();
    let shown = if options.top == 0 {
        last.reaction_kinds.len()
    } else {
        options.top.min(last.reaction_kinds.len())
    };
    println!(
        "reaction kinds at tick {} (top {} of {})",
        last.tick,
        shown,
        last.reaction_kinds.len()
    );
    let gene_indices = KIND_TABLE_GENES.map(|name| {
        ENZYME_GENE_NAMES
            .iter()
            .position(|gene| *gene == name)
            .expect("kind table genes are in the gene table")
    });
    let mut header = vec!["kind", "coupling", "enzymes", "share", "cells"];
    header.extend(KIND_TABLE_GENES);
    let rows = last.reaction_kinds[..shown]
        .iter()
        .map(|kind| {
            let mut row = vec![
                format!(
                    "{}→{}",
                    optional_element(kind.reactant),
                    optional_element(kind.product)
                ),
                match kind.coupling {
                    1 => "+",
                    -1 => "−",
                    _ => "0",
                }
                .to_owned(),
                kind.enzymes.to_string(),
                format!("{:.3}", kind.enzymes as f64 / total_enzymes.max(1) as f64),
                kind.cells.to_string(),
            ];
            row.extend(
                gene_indices
                    .iter()
                    .map(|index| format!("{:.4}", kind.gene_medians[*index])),
            );
            row
        })
        .collect::<Vec<_>>();
    print_table("  ", &header, &rows);

    println!();
    let previous = run
        .samples
        .len()
        .checked_sub(2)
        .map(|index| &run.samples[index]);
    match previous
        .and_then(|previous| Rates::between(previous, last).map(|rates| (previous, rates)))
    {
        Some((previous, rates)) => {
            println!(
                "rates per sim-second, ticks {} to {}",
                previous.tick, last.tick
            );
            let rows = rates
                .terms
                .iter()
                .map(|(group, name, value)| {
                    vec![format!("{group}.{name}"), format!("{value:+.4e}")]
                })
                .collect::<Vec<_>>();
            print_table("  ", &["term", "per_second"], &rows);
        }
        None => println!("rates: no interval"),
    }
}

fn gene_row(before: &GeneSpread, after: &GeneSpread) -> Vec<String> {
    let mut row = vec![before.name.to_owned()];
    row.extend(spread_columns(before.spread.as_ref(), 4));
    row.push("|".to_owned());
    row.extend(spread_columns(after.spread.as_ref(), 4));
    row
}

fn spread_columns(spread: Option<&Spread>, digits: usize) -> Vec<String> {
    [
        |spread: &Spread| spread.min,
        |spread: &Spread| spread.p05,
        |spread: &Spread| spread.p50,
        |spread: &Spread| spread.p95,
        |spread: &Spread| spread.max,
    ]
    .into_iter()
    .map(|field| spread_value(spread, field, digits))
    .collect()
}

fn spread_value(spread: Option<&Spread>, field: fn(&Spread) -> f64, digits: usize) -> String {
    spread.map_or_else(
        || "—".to_owned(),
        |spread| format!("{:.digits$}", field(spread)),
    )
}

fn optional_element(element: Option<Element>) -> &'static str {
    element.map_or("∅", |element| element_letter(&element))
}

fn element_letter(element: &Element) -> &'static str {
    ["A", "B", "C", "D", "E", "F"][element.index()]
}
