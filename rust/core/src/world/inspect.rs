use std::collections::BTreeSet;

use serde::Serialize;

use super::{LineageCounters, TileId, World, WorldError};
use crate::cell::{CellId, FluxRecord};
use crate::chem::ELEMENT_COUNT;
use crate::genome::{Enzyme, Genome, GenomePatch, LineageId, MAX_CELL_ENZYMES, MIN_CELL_ENZYMES};

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TileInspection {
    pub tile_id: TileId,
    pub x: usize,
    pub y: usize,
    pub enval: f32,
    pub enval_landscape_target: f32,
    pub cell_center_count: u32,
    pub element_concentrations: [f32; ELEMENT_COUNT],
    pub total_element_concentration: f32,
    pub mass_density: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CellInspection {
    pub cell_id: CellId,
    pub x: f32,
    pub y: f32,
    pub radius: f32,
    pub body_total: f64,
    pub boundary_total: f64,
    pub area: f64,
    pub strain: f64,
    pub membrane_polarity: f64,
    pub energy: f64,
    pub lineage_id: LineageId,
    pub enzyme_count: usize,
    pub total_internal_elements: f32,
    pub age_seconds: f64,
    pub mean_enval_optimum: f32,
    pub enval_optimum_spread: f32,
    pub local_enval_average: f32,
    pub field_share: f64,
    pub contacts: Vec<ContactInspection>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ContactInspection {
    pub cell_id: usize,
    pub lineage_id: LineageId,
    pub distance: f64,
    pub share: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EnzymeDetailInspection {
    pub index: usize,
    pub reactants: [f32; ELEMENT_COUNT],
    pub products: [f32; ELEMENT_COUNT],
    pub rate: f32,
    pub half_saturation: f32,
    pub energy_harvest_fraction: f32,
    pub secretion_fraction: f32,
    pub body_fraction: f32,
    pub boundary_fraction: f32,
    pub enval_optimum: f32,
    pub enval_sigma: f32,
    pub enval_throughput: f32,
    pub enval_energy_fraction: f32,
    pub enval_release_fraction: f32,
    pub enval_pump: f32,
    pub exposure: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GenomeDetailInspection {
    pub mutation_rate: f32,
    pub maintenance_cost_per_area: f64,
    pub lineage_id: LineageId,
    pub enzyme_count: usize,
    pub min_cell_enzymes: usize,
    pub max_cell_enzymes: usize,
    pub enzymes: Vec<EnzymeDetailInspection>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FluxLogInspection {
    pub available: bool,
    pub reason: &'static str,
    pub limit: usize,
    pub truncated: bool,
    pub flux_count: usize,
    pub returned_count: usize,
    pub order: &'static str,
    pub fluxes: Vec<FluxRecord>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GenomeEditResult {
    pub target: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cell_id: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub center_x: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub center_y: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub brush_width: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub brush_height: Option<usize>,
    pub visited_tile_count: usize,
    pub patched_cell_count: usize,
    pub changed_fields: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CellDetailInspection {
    pub cell: CellInspection,
    pub maintenance_cost_per_area: f64,
    pub catalyst_upkeep_per_sec: f64,
    pub genome: GenomeDetailInspection,
    pub body: [f32; ELEMENT_COUNT],
    pub boundary: [f32; ELEMENT_COUNT],
    pub internal_elements: [f32; ELEMENT_COUNT],
    pub total_internal_elements: f32,
    pub recent_fluxes: FluxLogInspection,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LineageSummaryInspection {
    pub lineage_id: LineageId,
    pub population: u64,
    pub births: u64,
    pub deaths: u64,
    pub extinct: bool,
    pub share: f64,
    pub average_energy: f64,
    pub average_enzyme_count: f64,
    pub mean_exposure: f64,
    pub mean_enval_optimum: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LineageListInspection {
    pub extant_lineage_count: usize,
    pub total_lineage_records: usize,
    pub limit: usize,
    pub truncated: bool,
    pub lineages: Vec<LineageSummaryInspection>,
}

impl World {
    pub fn inspect_tile(&self, tile_id: TileId) -> Option<TileInspection> {
        if tile_id.index() >= self.tile_count {
            return None;
        }
        let (x, y) = self.tile_xy(tile_id)?;
        let amounts = self.element_fields[tile_id.index()];
        Some(TileInspection {
            tile_id,
            x,
            y,
            enval: self.enval[tile_id.index()],
            enval_landscape_target: self.enval_landscape[tile_id.index()],
            cell_center_count: self.tile_cell_counts[tile_id.index()],
            element_concentrations: *amounts.as_array(),
            total_element_concentration: amounts.total() as f32,
            mass_density: amounts.mass() as f32,
        })
    }

    pub fn inspect_tile_xy(&self, x: usize, y: usize) -> Option<TileInspection> {
        self.tile_id(x, y)
            .and_then(|tile_id| self.inspect_tile(tile_id))
    }

    pub fn inspect_cell(&self, cell_id: CellId) -> Option<CellInspection> {
        let cell = self.cells.get(cell_id)?;
        let mut contacts = Vec::new();
        let field_share = self.collect_contacts(cell_id, cell.position, cell.radius, &mut contacts);
        Some(CellInspection {
            cell_id,
            x: cell.position.x,
            y: cell.position.y,
            radius: cell.radius,
            body_total: cell.body.total(),
            boundary_total: cell.boundary.total(),
            area: cell.area(self.config.cell_body.body_density),
            strain: cell.strain(self.config.cell_body.boundary_density),
            membrane_polarity: cell.membrane_polarity(),
            energy: cell.energy,
            lineage_id: cell.lineage_id,
            enzyme_count: cell.genome.enzymes.len(),
            total_internal_elements: cell.internal_elements.total() as f32,
            age_seconds: (self.sim_time_seconds - cell.birth_sim_time).max(0.0),
            mean_enval_optimum: cell.genome.mean_enval_optimum(),
            enval_optimum_spread: cell.genome.enval_optimum_spread(),
            local_enval_average: self.local_enval_at(cell.position).unwrap_or(0.0),
            field_share,
            contacts: contacts
                .iter()
                .map(|contact| ContactInspection {
                    cell_id: contact.cell_id.index(),
                    lineage_id: self.cells[contact.cell_id].lineage_id,
                    distance: contact.distance,
                    share: contact.share,
                })
                .collect(),
        })
    }

    pub fn inspect_cell_detail(
        &mut self,
        cell_id: CellId,
        flux_limit: usize,
    ) -> Option<CellDetailInspection> {
        let recent_fluxes = self.inspect_cell_fluxes(cell_id, flux_limit)?;
        let cell = self.cells.get(cell_id)?;
        let summary = self.inspect_cell(cell_id)?;
        Some(CellDetailInspection {
            cell: summary,
            maintenance_cost_per_area: cell.maintenance_cost_per_area,
            catalyst_upkeep_per_sec: self.catalyst_upkeep_per_sec(cell),
            genome: inspect_genome(&cell.genome),
            body: *cell.body.as_array(),
            boundary: *cell.boundary.as_array(),
            internal_elements: *cell.internal_elements.as_array(),
            total_internal_elements: cell.internal_elements.total() as f32,
            recent_fluxes,
        })
    }

    pub fn inspect_cell_fluxes(
        &mut self,
        cell_id: CellId,
        limit: usize,
    ) -> Option<FluxLogInspection> {
        self.cells.get(cell_id)?;
        if self.watched_cell_id != Some(cell_id) {
            self.watched_cell_id = Some(cell_id);
            self.flux_log.clear();
        }
        let flux_count = self.flux_log.flux_count();
        let fluxes = self
            .flux_log
            .fluxes_newest_first()
            .take(limit)
            .cloned()
            .collect::<Vec<_>>();
        Some(FluxLogInspection {
            available: true,
            reason: "recorded",
            limit,
            truncated: flux_count > fluxes.len(),
            flux_count,
            returned_count: fluxes.len(),
            order: "newest_first",
            fluxes,
        })
    }

    pub fn inspect_lineage(&self, lineage_id: LineageId) -> Option<LineageSummaryInspection> {
        let counters = *self.lineage_counters.get(&lineage_id)?;
        Some(self.lineage_summary(lineage_id, counters))
    }

    pub fn list_lineages(&self, limit: usize) -> LineageListInspection {
        let extant_lineage_count = self.extant_lineage_count();
        let entries = self.top_lineages(limit);
        let lineages = entries
            .into_iter()
            .map(|(lineage_id, counters)| self.lineage_summary(lineage_id, counters))
            .collect::<Vec<_>>();
        LineageListInspection {
            extant_lineage_count,
            total_lineage_records: self.lineage_counters.len(),
            limit,
            truncated: extant_lineage_count > limit,
            lineages,
        }
    }

    pub fn apply_cell_genome_patch(
        &mut self,
        cell_id: CellId,
        patch: &GenomePatch,
    ) -> Result<GenomeEditResult, WorldError> {
        let changed_fields = self.apply_genome_patch_to_cell(cell_id, patch)?;
        Ok(GenomeEditResult {
            target: "cell",
            cell_id: Some(cell_id.index()),
            center_x: None,
            center_y: None,
            brush_width: None,
            brush_height: None,
            visited_tile_count: 1,
            patched_cell_count: 1,
            changed_fields,
        })
    }

    pub fn apply_genome_brush(
        &mut self,
        center_x: usize,
        center_y: usize,
        brush_width: usize,
        brush_height: usize,
        patch: &GenomePatch,
    ) -> Result<GenomeEditResult, WorldError> {
        let width = brush_width.min(self.width).max(1);
        let height = brush_height.min(self.height).max(1);
        let start_x = center_x as isize - (width as isize / 2);
        let start_y = center_y as isize - (height as isize / 2);
        let mut covered_tiles = vec![false; self.tile_count];
        for dx in 0..width {
            for dy in 0..height {
                let tile_id = self.wrapped_tile_id(start_x + dx as isize, start_y + dy as isize);
                covered_tiles[tile_id.index()] = true;
            }
        }
        let cells = self
            .cells
            .iter()
            .filter(|cell| {
                self.center_tile_id(cell.position)
                    .is_some_and(|tile_id| covered_tiles[tile_id.index()])
            })
            .map(|cell| cell.id)
            .collect::<Vec<_>>();

        let mut changed = BTreeSet::new();
        for cell_id in cells.iter().copied() {
            let Some(cell) = self.cells.get(cell_id) else {
                return Err(WorldError::InvalidCell(cell_id));
            };
            let mut genome = cell.genome.clone();
            let changed_fields = patch
                .apply_to_genome(&mut genome)
                .map_err(|err| WorldError::GenomePatch(err.to_string()))?;
            for field in changed_fields {
                changed.insert(field);
            }
        }

        let mut patched_cell_count = 0_usize;
        for cell_id in cells.iter().copied() {
            self.apply_genome_patch_to_cell(cell_id, patch)?;
            patched_cell_count += 1;
        }

        Ok(GenomeEditResult {
            target: "brush",
            cell_id: None,
            center_x: Some(center_x % self.width),
            center_y: Some(center_y % self.height),
            brush_width: Some(width),
            brush_height: Some(height),
            visited_tile_count: width.saturating_mul(height),
            patched_cell_count,
            changed_fields: changed.into_iter().collect(),
        })
    }

    pub(super) fn apply_genome_patch_to_cell(
        &mut self,
        cell_id: CellId,
        patch: &GenomePatch,
    ) -> Result<Vec<String>, WorldError> {
        let Some(cell) = self.cells.get(cell_id) else {
            return Err(WorldError::InvalidCell(cell_id));
        };
        let original_lineage = cell.lineage_id;
        let mut genome = cell.genome.clone();
        let changed_fields = patch
            .apply_to_genome(&mut genome)
            .map_err(|err| WorldError::GenomePatch(err.to_string()))?;
        genome.lineage_id = original_lineage;

        let cell = self
            .cells
            .get_mut(cell_id)
            .ok_or(WorldError::InvalidCell(cell_id))?;
        cell.genome = genome;
        cell.lineage_id = original_lineage;
        cell.maintenance_cost_per_area = cell.genome.maintenance_cost_per_area;
        Ok(changed_fields)
    }

    pub(super) fn lineage_summary(
        &self,
        lineage_id: LineageId,
        counters: LineageCounters,
    ) -> LineageSummaryInspection {
        let mut live_count = 0_u64;
        let mut energy_sum = 0.0_f64;
        let mut enzyme_sum = 0_u64;
        let mut exposure_sum = 0.0_f64;
        let mut rate_sum = 0.0_f64;
        let mut rate_weighted_optimum_sum = 0.0_f64;
        for cell in self.cells.iter() {
            if cell.lineage_id != lineage_id {
                continue;
            }
            live_count = live_count.saturating_add(1);
            energy_sum += cell.energy;
            enzyme_sum = enzyme_sum.saturating_add(cell.genome.enzymes.len() as u64);
            for enzyme in &cell.genome.enzymes {
                exposure_sum += f64::from(enzyme.exposure);
                rate_sum += f64::from(enzyme.rate);
                rate_weighted_optimum_sum +=
                    f64::from(enzyme.rate) * f64::from(enzyme.enval_optimum);
            }
        }
        let denominator = live_count.max(1) as f64;
        let total_live_cells = self.cells.len().max(1) as f64;
        LineageSummaryInspection {
            lineage_id,
            population: counters.population,
            births: counters.births,
            deaths: counters.deaths,
            extinct: counters.population == 0,
            share: counters.population as f64 / total_live_cells,
            average_energy: if live_count > 0 {
                energy_sum / denominator
            } else {
                0.0
            },
            average_enzyme_count: if live_count > 0 {
                enzyme_sum as f64 / denominator
            } else {
                0.0
            },
            mean_exposure: if enzyme_sum > 0 {
                exposure_sum / enzyme_sum as f64
            } else {
                0.0
            },
            mean_enval_optimum: if rate_sum > 0.0 {
                rate_weighted_optimum_sum / rate_sum
            } else {
                0.0
            },
        }
    }
}

fn inspect_genome(genome: &Genome) -> GenomeDetailInspection {
    GenomeDetailInspection {
        mutation_rate: genome.mutation_rate,
        maintenance_cost_per_area: genome.maintenance_cost_per_area,
        lineage_id: genome.lineage_id,
        enzyme_count: genome.enzymes.len(),
        min_cell_enzymes: MIN_CELL_ENZYMES,
        max_cell_enzymes: MAX_CELL_ENZYMES,
        enzymes: genome
            .enzymes
            .iter()
            .enumerate()
            .map(|(index, enzyme)| inspect_enzyme(index, enzyme))
            .collect(),
    }
}

fn inspect_enzyme(index: usize, enzyme: &Enzyme) -> EnzymeDetailInspection {
    EnzymeDetailInspection {
        index,
        reactants: *enzyme.reactants.as_array(),
        products: *enzyme.products.as_array(),
        rate: enzyme.rate,
        half_saturation: enzyme.half_saturation,
        energy_harvest_fraction: enzyme.energy_harvest_fraction,
        secretion_fraction: enzyme.secretion_fraction,
        body_fraction: enzyme.body_fraction,
        boundary_fraction: enzyme.boundary_fraction,
        enval_optimum: enzyme.enval_optimum,
        enval_sigma: enzyme.enval_sigma,
        enval_throughput: enzyme.enval_throughput,
        enval_energy_fraction: enzyme.enval_energy_fraction,
        enval_release_fraction: enzyme.enval_release_fraction,
        enval_pump: enzyme.enval_pump,
        exposure: enzyme.exposure,
    }
}
