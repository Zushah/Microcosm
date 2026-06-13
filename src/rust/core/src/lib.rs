pub mod bio;
pub mod cell;
pub mod chem;
pub mod config;
pub mod genome;
pub mod molecule;
pub mod render_buffers;
pub mod rng;
pub mod snapshot;
pub mod stats;
pub mod world;

pub use cell::{
    Cell, CellId, CellState, ReactionMoleculeSummary, ReactionRecord, CELL_REACTION_LOG_CAPACITY,
};
pub use chem::{
    Composition, CompositionError, Element, ElementProperties, ELEMENT_COUNT, ELEMENT_ORDER,
};
pub use config::{Config, ConfigError, MoleculeSeedingConfig};
pub use genome::{
    Enzyme, EnzymeType, Genome, LineageId, PredationEnzymeTransferStats, MAX_CELL_ENZYMES,
    MIN_CELL_ENZYMES,
};
pub use molecule::{Molecule, MoleculeError};
pub use render_buffers::{RenderBuffers, EMPTY_CELL_ID};
pub use rng::Rng;
pub use snapshot::{
    load_from_path, save_to_path, SnapshotError, SNAPSHOT_EXTENSION, SNAPSHOT_VERSION,
};
pub use stats::{
    EnzymeTypeAmounts, EnzymeTypeCounts, OperationCounters, ReactionCounters, StepProfile,
    WorldStats, ENZYME_COUNT_HISTOGRAM_LEN,
};
pub use world::{
    CellDetailInspection, CellInspection, EnzymeDetailInspection, GenomeDetailInspection,
    InvariantError, LineageCounters, LineageListInspection, LineageSummaryInspection,
    MoleculeDetailInspection, MoleculeId, MoleculeListInspection, MoleculeOwner, NeighborIndices,
    ReactionLogInspection, TileId, TileInspection, World, WorldError,
};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
