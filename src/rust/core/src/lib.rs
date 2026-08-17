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
    CELL_REACTION_LOG_CAPACITY, Cell, CellId, CellState, ReactionMoleculeSummary, ReactionRecord,
};
pub use chem::{
    Composition, CompositionError, ELEMENT_COUNT, ELEMENT_ORDER, Element, ElementProperties,
};
pub use config::{Config, ConfigError, MoleculeSeedingConfig};
pub use genome::{
    Enzyme, EnzymeFieldPatch, EnzymePatchOperation, EnzymeType, GENOME_PATCH_SCHEMA, Genome,
    GenomeFieldPatch, GenomePatch, GenomePatchError, LineageId, MAX_CELL_ENZYMES, MIN_CELL_ENZYMES,
    PredationEnzymeTransferStats,
};
pub use molecule::{Molecule, MoleculeError};
pub use render_buffers::{
    EMPTY_CELL_ID, RenderBrushPreview, RenderBuffers, RenderDisplayMode, RenderVisualState,
};
pub use rng::Rng;
pub use snapshot::{
    SNAPSHOT_EXTENSION, SNAPSHOT_VERSION, SnapshotError, load_from_path, save_to_path,
};
pub use stats::{
    ENZYME_COUNT_HISTOGRAM_LEN, EnzymeTypeAmounts, EnzymeTypeCounts, OperationCounters,
    ReactionCounters, StepProfile, WorldStats,
};
pub use world::{
    CellDetailInspection, CellInspection, EnzymeDetailInspection, GenomeDetailInspection,
    GenomeEditResult, InvariantError, LineageCounters, LineageListInspection,
    LineageSummaryInspection, MoleculeDetailInspection, MoleculeId, MoleculeListInspection,
    MoleculeOwner, NeighborIndices, ReactionLogInspection, TileId, TileInspection, World,
    WorldError,
};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
