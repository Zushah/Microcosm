use std::collections::BTreeMap;

use super::World;
use super::stats::percentile_index;
use crate::census::{Census, GeneSpread, ReactionKind, Spread};
use crate::chem::{ELEMENT_COUNT, ELEMENT_ORDER, ElementAmounts};
use crate::genome::{ENZYME_GENE_NAMES, Enzyme, GENOME_GENE_NAMES};

const ENZYME_GENE_COUNT: usize = ENZYME_GENE_NAMES.len();
const GENOME_GENE_COUNT: usize = GENOME_GENE_NAMES.len();

type KindKey = (Option<usize>, Option<usize>, i8);

#[derive(Default)]
struct KindAccumulator {
    enzymes: usize,
    cells: usize,
    genes: Vec<[f32; ENZYME_GENE_COUNT]>,
}

impl World {
    pub fn census(&self) -> Census {
        let stats = self.compact_stats();
        let live_cells = stats.live_cell_count;

        let mut generations = Vec::with_capacity(live_cells);
        let mut enzymes_per_genome = Vec::with_capacity(live_cells);
        let mut genome_genes: [Vec<f64>; GENOME_GENE_COUNT] =
            std::array::from_fn(|_| Vec::with_capacity(live_cells));
        let total_enzymes = self
            .cells
            .iter()
            .map(|cell| cell.genome.enzymes.len())
            .sum::<usize>();
        let mut enzyme_genes: [Vec<f64>; ENZYME_GENE_COUNT] =
            std::array::from_fn(|_| Vec::with_capacity(total_enzymes));
        let mut stoichiometric_entries = Vec::with_capacity(total_enzymes);
        let mut zero_exposure_enzymes = 0_usize;
        let mut exposed_cells = 0_usize;
        let mut kinds = BTreeMap::<KindKey, KindAccumulator>::new();
        let mut cell_kinds = Vec::<KindKey>::new();

        for cell in self.cells.iter() {
            generations.push(f64::from(cell.generation));
            enzymes_per_genome.push(cell.genome.enzymes.len() as f64);
            for (values, value) in genome_genes.iter_mut().zip(cell.genome.gene_values()) {
                values.push(value);
            }
            let mut exposed = false;
            cell_kinds.clear();
            for enzyme in &cell.genome.enzymes {
                let genes = enzyme.gene_values();
                for (values, value) in enzyme_genes.iter_mut().zip(genes) {
                    values.push(f64::from(value));
                }
                stoichiometric_entries.push(stoichiometric_entry_count(enzyme) as f64);
                if enzyme.exposure == 0.0 {
                    zero_exposure_enzymes += 1;
                }
                exposed |= enzyme.exposure > 0.0;
                let key = reaction_kind_key(enzyme);
                let kind = kinds.entry(key).or_default();
                kind.enzymes += 1;
                kind.genes.push(genes);
                cell_kinds.push(key);
            }
            cell_kinds.sort_unstable();
            cell_kinds.dedup();
            for key in &cell_kinds {
                if let Some(kind) = kinds.get_mut(key) {
                    kind.cells += 1;
                }
            }
            if exposed {
                exposed_cells += 1;
            }
        }

        let mut reaction_kinds = kinds
            .into_iter()
            .map(|((reactant, product, coupling), kind)| ReactionKind {
                reactant: reactant.map(|index| ELEMENT_ORDER[index]),
                product: product.map(|index| ELEMENT_ORDER[index]),
                coupling,
                enzymes: kind.enzymes,
                cells: kind.cells,
                gene_medians: (0..ENZYME_GENE_COUNT)
                    .map(|gene| {
                        let values = kind.genes.iter().map(|genes| f64::from(genes[gene]));
                        spread_of(values.collect()).map_or(0.0, |spread| spread.p50)
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        reaction_kinds.sort_by_key(|kind| std::cmp::Reverse(kind.enzymes));

        let tile_count = self.tile_count.max(1) as f64;
        let mut field_mean_per_tile = [0.0_f64; ELEMENT_COUNT];
        for (mean, total) in field_mean_per_tile
            .iter_mut()
            .zip(stats.extracellular_element_amounts)
        {
            *mean = total / tile_count;
        }
        let total_matter = stats.total_element_amount;
        let matter_drift =
            (total_matter - self.initial_matter) / self.initial_matter.abs().max(1.0);

        let (enval_mean, enval_std_dev) = mean_and_std_dev(&self.enval);
        let (_, landscape_std_dev) = mean_and_std_dev(&self.enval_landscape);
        let enval_spread = spread_of(self.enval.iter().map(|value| f64::from(*value)).collect());
        let enval_flattening = if landscape_std_dev > 0.0 {
            enval_std_dev / landscape_std_dev
        } else {
            1.0
        };

        Census {
            tick: self.tick_count,
            sim_time_seconds: self.sim_time_seconds,
            live_cells,
            births: stats.births,
            deaths: stats.deaths,
            fissions: stats.fissions,
            energy_deaths: stats.energy_deaths,
            lyses: stats.lyses,
            extant_lineages: stats.extant_lineage_count,
            dominant_lineage_share: stats.dominant_lineage_share,
            generation: spread_of(generations),
            enzyme_genes: ENZYME_GENE_NAMES
                .iter()
                .zip(enzyme_genes)
                .map(|(name, values)| GeneSpread {
                    name,
                    spread: spread_of(values),
                })
                .collect(),
            genome_genes: GENOME_GENE_NAMES
                .iter()
                .zip(genome_genes)
                .map(|(name, values)| GeneSpread {
                    name,
                    spread: spread_of(values),
                })
                .collect(),
            enzymes_per_genome: spread_of(enzymes_per_genome),
            stoichiometric_entries_per_enzyme: spread_of(stoichiometric_entries),
            zero_exposure_enzyme_fraction: fraction(zero_exposure_enzymes, total_enzymes),
            exposed_cell_fraction: fraction(exposed_cells, live_cells),
            reaction_kinds,
            field_mean_per_tile,
            in_cells: stats.intracellular_element_amounts,
            total_matter,
            initial_matter: self.initial_matter,
            matter_drift,
            enval_mean,
            enval_std_dev,
            enval_p05: enval_spread.as_ref().map_or(0.0, |spread| spread.p05),
            enval_p50: enval_spread.as_ref().map_or(0.0, |spread| spread.p50),
            enval_p95: enval_spread.as_ref().map_or(0.0, |spread| spread.p95),
            positive_enval_fraction: fraction(stats.positive_enval_tile_count, self.tile_count),
            negative_enval_fraction: fraction(stats.negative_enval_tile_count, self.tile_count),
            landscape_std_dev,
            enval_flattening,
            energy_ledger: stats.energy_ledger,
            enval_ledger: stats.enval_ledger,
            energy_residual: stats.energy_ledger_residual,
            enval_residual: self.enval_ledger_residual(),
            renewable_coverage: stats.renewable_coverage,
        }
    }
}

fn spread_of(mut values: Vec<f64>) -> Option<Spread> {
    if values.is_empty() {
        return None;
    }
    values.sort_unstable_by(f64::total_cmp);
    let at = |fraction: f64| values[percentile_index(values.len(), fraction)];
    Some(Spread {
        count: values.len(),
        min: values[0],
        p05: at(0.05),
        p50: at(0.50),
        p95: at(0.95),
        max: values[values.len() - 1],
    })
}

fn mean_and_std_dev(values: &[f32]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let count = values.len() as f64;
    let mean = values.iter().map(|value| f64::from(*value)).sum::<f64>() / count;
    let variance = values
        .iter()
        .map(|value| (f64::from(*value) - mean).powi(2))
        .sum::<f64>()
        / count;
    (mean, variance.sqrt())
}

fn fraction(part: usize, whole: usize) -> f64 {
    if whole > 0 {
        part as f64 / whole as f64
    } else {
        0.0
    }
}

fn stoichiometric_entry_count(enzyme: &Enzyme) -> usize {
    enzyme
        .reactants
        .as_array()
        .iter()
        .chain(enzyme.products.as_array())
        .filter(|coefficient| **coefficient != 0.0)
        .count()
}

fn dominant_element(amounts: &ElementAmounts) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (index, value) in amounts.as_array().iter().enumerate() {
        if *value > 0.0 && best.is_none_or(|(_, best_value)| *value > best_value) {
            best = Some((index, *value));
        }
    }
    best.map(|(index, _)| index)
}

fn reaction_kind_key(enzyme: &Enzyme) -> KindKey {
    let coupling = if enzyme.enval_throughput > 0.0 {
        if enzyme.enval_optimum > 0.0 {
            1
        } else if enzyme.enval_optimum < 0.0 {
            -1
        } else {
            0
        }
    } else {
        0
    };
    (
        dominant_element(&enzyme.reactants),
        dominant_element(&enzyme.products),
        coupling,
    )
}
