pub mod bio;
pub mod cell;
pub mod chem;
pub mod config;
pub mod genome;
pub mod render_buffers;
pub mod rng;
pub mod snapshot;
pub mod stats;
pub mod world;

pub use cell::{CELL_FLUX_LOG_CAPACITY, Cell, CellId, CellState, FluxRecord};
pub use chem::{
    ELEMENT_COUNT, ELEMENT_ORDER, Element, ElementAmounts, ElementAmountsError, ElementProperties,
};
pub use config::{
    Config, ConfigError, DEFAULT_ELEMENT_FIELD_AMOUNTS, DEFAULT_ELEMENT_FIELD_DIFFUSIVITIES,
    ElementFieldConfig,
};
pub use genome::{
    CatalystError, Enzyme, EnzymeFieldPatch, EnzymePatchOperation, EnzymeType, GENOME_PATCH_SCHEMA,
    Genome, GenomeFieldPatch, GenomePatch, GenomePatchError, LineageId, MAX_CELL_ENZYMES,
    MIN_CELL_ENZYMES, PredationEnzymeTransferStats,
};
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
    CellDetailInspection, CellInspection, EnzymeDetailInspection, FluxLogInspection,
    GenomeDetailInspection, GenomeEditResult, InvariantError, LineageCounters,
    LineageListInspection, LineageSummaryInspection, NeighborIndices, TileId, TileInspection,
    World, WorldError,
};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
