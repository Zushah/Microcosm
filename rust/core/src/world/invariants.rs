use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use super::{GEOMETRY_TOLERANCE, TileId, World};
use crate::cell::CellId;
use crate::chem::{ELEMENT_ORDER, Element};
use crate::genome::{LineageId, MAX_CELL_ENZYMES, MIN_CELL_ENZYMES};

impl World {
    pub fn check_invariants(&self) -> Result<(), InvariantError> {
        if self.enval.len() != self.tile_count
            || self.enval_next.len() != self.tile_count
            || self.enval_landscape.len() != self.tile_count
            || self.element_fields.len() != self.tile_count
            || self.element_fields_next.len() != self.tile_count
            || self.tile_cell_counts.len() != self.tile_count
        {
            return Err(InvariantError::MismatchedWorldArrayLengths);
        }

        for tile_index in 0..self.tile_count {
            if !self.enval[tile_index].is_finite() {
                return Err(InvariantError::NonFiniteEnval(TileId(tile_index)));
            }
            for element in ELEMENT_ORDER {
                let value = self.element_fields[tile_index][element];
                if !value.is_finite() || value < 0.0 {
                    return Err(InvariantError::InvalidElementField {
                        tile: TileId(tile_index),
                        element,
                    });
                }
            }
        }

        if !self.cells.is_consistent() {
            return Err(InvariantError::CellStoreMismatch);
        }
        let mut expected_tile_cell_counts = vec![0_u32; self.tile_count];
        for cell in self.cells.iter() {
            let cell_id = cell.id;
            if !cell.energy.is_finite() || cell.energy < 0.0 {
                return Err(InvariantError::NonFiniteCellEnergy(cell_id));
            }
            if !cell.position.is_finite()
                || !cell.radius.is_finite()
                || cell.body.total() <= 0.0
                || (cell.radius
                    - (cell.area(self.config.cell_body.body_density) / std::f64::consts::PI).sqrt()
                        as f32)
                    .abs()
                    > GEOMETRY_TOLERANCE
                || cell.radius <= 0.0
                || cell.position.wrapped(self.width as f32, self.height as f32)
                    != Some(cell.position)
            {
                return Err(InvariantError::InvalidCellGeometry(cell_id));
            }
            if !self.spatial_index.contains_at(cell_id, cell.position) {
                return Err(InvariantError::SpatialIndexMismatch(cell_id));
            }
            if cell.genome.enzymes.len() < MIN_CELL_ENZYMES
                || cell.genome.enzymes.len() > MAX_CELL_ENZYMES
            {
                return Err(InvariantError::InvalidGenomeEnzymeCount(cell_id));
            }
            for enzyme in &cell.genome.enzymes {
                if enzyme.validate().is_err() {
                    return Err(InvariantError::InvalidCatalyst(cell_id));
                }
            }

            for element in ELEMENT_ORDER {
                let amount = cell.internal_elements[element];
                if [amount, cell.body[element], cell.boundary[element]]
                    .iter()
                    .any(|value| !value.is_finite() || *value < 0.0)
                {
                    return Err(InvariantError::CellElementReservoirMismatch {
                        cell: cell_id,
                        element,
                    });
                }
            }
            let tile_id = self
                .center_tile_id(cell.position)
                .ok_or(InvariantError::InvalidCellGeometry(cell_id))?;
            expected_tile_cell_counts[tile_id.index()] =
                expected_tile_cell_counts[tile_id.index()].saturating_add(1);
        }
        if self.spatial_index.len() != self.cells.len() {
            return Err(InvariantError::SpatialIndexCountMismatch {
                expected: self.cells.len(),
                actual: self.spatial_index.len(),
            });
        }
        for (tile_index, (expected, actual)) in expected_tile_cell_counts
            .iter()
            .zip(&self.tile_cell_counts)
            .enumerate()
        {
            if expected != actual {
                return Err(InvariantError::TileCellCountMismatch {
                    tile: TileId(tile_index),
                    expected: *expected,
                    actual: *actual,
                });
            }
        }

        let mut actual_lineage_population = BTreeMap::<LineageId, u64>::new();
        for cell in self.cells.iter() {
            *actual_lineage_population
                .entry(cell.lineage_id)
                .or_default() += 1;
        }
        for (lineage_id, counters) in &self.lineage_counters {
            let actual = actual_lineage_population
                .remove(lineage_id)
                .unwrap_or_default();
            if counters.population != actual {
                return Err(InvariantError::LineagePopulationMismatch {
                    lineage: *lineage_id,
                    expected: actual,
                    actual: counters.population,
                });
            }
        }
        if let Some((lineage, actual)) = actual_lineage_population.into_iter().next() {
            return Err(InvariantError::LineagePopulationMismatch {
                lineage,
                expected: actual,
                actual: 0,
            });
        }

        for (term, value) in self.energy_ledger.terms() {
            if !value.is_finite() || value < 0.0 {
                return Err(InvariantError::InvalidEnergyLedgerTerm { term, value });
            }
        }
        let actual = self.total_live_cell_energy();
        let expected = self.energy_ledger.expected_cell_energy();
        let tolerance = self.energy_ledger.closure_tolerance();
        if !actual.is_finite() || (actual - expected).abs() > tolerance {
            return Err(InvariantError::EnergyLedgerMismatch {
                expected,
                actual,
                tolerance,
            });
        }

        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum InvariantError {
    MismatchedWorldArrayLengths,
    NonFiniteEnval(TileId),
    InvalidElementField {
        tile: TileId,
        element: Element,
    },
    CellStoreMismatch,
    NonFiniteCellEnergy(CellId),
    InvalidCellGeometry(CellId),
    SpatialIndexMismatch(CellId),
    SpatialIndexCountMismatch {
        expected: usize,
        actual: usize,
    },
    TileCellCountMismatch {
        tile: TileId,
        expected: u32,
        actual: u32,
    },
    InvalidGenomeEnzymeCount(CellId),
    InvalidCatalyst(CellId),
    CellElementReservoirMismatch {
        cell: CellId,
        element: Element,
    },
    LineagePopulationMismatch {
        lineage: LineageId,
        expected: u64,
        actual: u64,
    },
    InvalidEnergyLedgerTerm {
        term: &'static str,
        value: f64,
    },
    EnergyLedgerMismatch {
        expected: f64,
        actual: f64,
        tolerance: f64,
    },
}

impl fmt::Display for InvariantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MismatchedWorldArrayLengths => {
                f.write_str("world arrays have mismatched lengths")
            }
            Self::NonFiniteEnval(tile) => write!(f, "tile {} has non-finite enval", tile.index()),
            Self::InvalidElementField { tile, element } => write!(
                f,
                "tile {} has an invalid {} field value",
                tile.index(),
                element
            ),
            Self::CellStoreMismatch => f.write_str("live cell store ids, slots and index disagree"),
            Self::NonFiniteCellEnergy(cell) => {
                write!(f, "cell {} has invalid energy", cell.index())
            }
            Self::InvalidCellGeometry(cell) => write!(
                f,
                "cell {} has invalid or non-canonical continuous geometry",
                cell.index()
            ),
            Self::SpatialIndexMismatch(cell) => write!(
                f,
                "cell {} is missing or misplaced in the spatial index",
                cell.index()
            ),
            Self::SpatialIndexCountMismatch { expected, actual } => write!(
                f,
                "spatial index count mismatch: expected {expected}, got {actual}"
            ),
            Self::TileCellCountMismatch {
                tile,
                expected,
                actual,
            } => write!(
                f,
                "tile {} cell count mismatch: expected {expected}, got {actual}",
                tile.index()
            ),
            Self::InvalidGenomeEnzymeCount(cell) => write!(
                f,
                "cell {} genome enzyme count is outside [{}, {}]",
                cell.index(),
                MIN_CELL_ENZYMES,
                MAX_CELL_ENZYMES
            ),
            Self::InvalidCatalyst(cell) => {
                write!(
                    f,
                    "cell {} has an invalid catalyst definition",
                    cell.index()
                )
            }
            Self::CellElementReservoirMismatch { cell, element } => write!(
                f,
                "cell {} has an invalid continuous {} reservoir",
                cell.index(),
                element
            ),
            Self::LineagePopulationMismatch {
                lineage,
                expected,
                actual,
            } => write!(
                f,
                "lineage {} population mismatch: expected {}, got {}",
                lineage.raw(),
                expected,
                actual
            ),
            Self::InvalidEnergyLedgerTerm { term, value } => write!(
                f,
                "energy ledger term {term} must be finite and nonnegative, got {value}"
            ),
            Self::EnergyLedgerMismatch {
                expected,
                actual,
                tolerance,
            } => write!(
                f,
                "live cell energy {actual} does not match the energy ledger balance {expected} (residual {}, tolerance {tolerance})",
                actual - expected
            ),
        }
    }
}

impl Error for InvariantError {}
