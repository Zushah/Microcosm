pub mod bio;
pub mod cell;
pub mod chem;
pub mod config;
pub mod environment;
pub mod genome;
pub mod render_buffers;
pub mod rng;
pub mod snapshot;
pub mod spatial;
pub mod stats;
pub mod world;

pub use cell::{CELL_FLUX_LOG_CAPACITY, Cell, CellId, CellStore, FluxRecord};
pub use chem::{
    ELEMENT_COUNT, ELEMENT_ORDER, Element, ElementAmounts, ElementAmountsError, ElementProperties,
};
pub use config::{
    Config, ConfigError, DEFAULT_ELEMENT_FIELD_AMOUNTS, DEFAULT_ELEMENT_FIELD_DIFFUSIVITIES,
    DEFAULT_MEMBRANE_PERMEABILITY, ElementFieldConfig, EnvalRechargeConfig, EnvalSourceConfig,
};
pub use environment::{ENVAL_PER_RECHARGE_ENERGY, EnvalSource, EnvalSources, RechargeOutcome};
pub use genome::{
    CatalystError, Enzyme, EnzymeFieldPatch, EnzymePatchOperation, EnzymeType, GENOME_PATCH_SCHEMA,
    Genome, GenomeFieldPatch, GenomePatch, GenomePatchError, LineageId, MAX_CELL_ENZYMES,
    MIN_CELL_ENZYMES, PredationEnzymeTransferStats,
};
pub use render_buffers::{RenderBrushPreview, RenderBuffers, RenderDisplayMode, RenderVisualState};
pub use rng::Rng;
pub use snapshot::{
    SNAPSHOT_EXTENSION, SNAPSHOT_VERSION, SnapshotError, load_from_path, save_to_path,
};
pub use spatial::{
    BilinearSample, BilinearStencil, DEFAULT_CELL_RADIUS, Position, SpatialIndex,
    minimum_image_delta, minimum_image_displacement, toroidal_distance, toroidal_distance_squared,
    wrap_coordinate,
};
pub use stats::{
    ENZYME_COUNT_HISTOGRAM_LEN, EnergyLedger, EnvalLedger, EnzymeTypeAmounts, EnzymeTypeCounts,
    OperationCounters, ReactionCounters, StepProfile, WorldStats, renewable_coverage,
};
pub use world::{
    CellDetailInspection, CellInspection, EnzymeDetailInspection, FluxLogInspection,
    GenomeDetailInspection, GenomeEditResult, InvariantError, LineageCounters,
    LineageListInspection, LineageSummaryInspection, NeighborIndices, TileId, TileInspection,
    World, WorldError,
};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
