use std::error::Error;
use std::fmt;
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::world::World;

pub const SNAPSHOT_VERSION: &str = crate::VERSION;
pub const SNAPSHOT_EXTENSION: &str = "micosm";

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SnapshotEnvelope {
    version: String,
    world: World,
}

#[derive(Debug)]
pub enum SnapshotError {
    Io(std::io::Error),
    Codec(bincode::Error),
    UnsupportedVersion {
        found: String,
        expected: &'static str,
    },
    Invariant(String),
}

impl fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "snapshot I/O error: {err}"),
            Self::Codec(err) => write!(f, "snapshot codec error: {err}"),
            Self::UnsupportedVersion { found, expected } => write!(
                f,
                "unsupported snapshot version {found}; expected {expected}"
            ),
            Self::Invariant(err) => write!(f, "snapshot invariant failure: {err}"),
        }
    }
}

impl Error for SnapshotError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::Codec(err) => Some(&**err),
            Self::UnsupportedVersion { .. } | Self::Invariant(_) => None,
        }
    }
}

impl From<std::io::Error> for SnapshotError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<bincode::Error> for SnapshotError {
    fn from(value: bincode::Error) -> Self {
        Self::Codec(value)
    }
}

pub fn to_bytes(world: &World) -> Result<Vec<u8>, SnapshotError> {
    let envelope = SnapshotEnvelope {
        version: SNAPSHOT_VERSION.to_owned(),
        world: world.clone(),
    };
    Ok(bincode::serialize(&envelope)?)
}

pub fn from_bytes(bytes: &[u8]) -> Result<World, SnapshotError> {
    let envelope: SnapshotEnvelope = bincode::deserialize(bytes)?;
    if envelope.version != SNAPSHOT_VERSION {
        return Err(SnapshotError::UnsupportedVersion {
            found: envelope.version,
            expected: SNAPSHOT_VERSION,
        });
    }
    let mut world = envelope.world;
    world.rebuild_derived_caches();
    world
        .check_invariants()
        .map_err(|err| SnapshotError::Invariant(err.to_string()))?;
    Ok(world)
}

pub fn save_to_path<P: AsRef<Path>>(world: &World, path: P) -> Result<(), SnapshotError> {
    let bytes = to_bytes(world)?;
    fs::write(path, bytes)?;
    Ok(())
}

pub fn load_from_path<P: AsRef<Path>>(path: P) -> Result<World, SnapshotError> {
    let bytes = fs::read(path)?;
    from_bytes(&bytes)
}

#[cfg(test)]
mod tests {
    use super::{from_bytes, to_bytes, SNAPSHOT_VERSION};
    use crate::{Config, World};

    fn small_world(seed: &str) -> World {
        let config = Config {
            seed: seed.to_owned(),
            width: 12,
            height: 10,
            ..Config::default()
        };
        let mut world = World::new(config).unwrap();
        world.spawn_founder_cells(4).unwrap();
        world
    }

    #[test]
    fn snapshot_version_matches_microcosm_release() {
        assert_eq!(SNAPSHOT_VERSION, "0.17.2");
    }

    #[test]
    fn snapshot_roundtrip_preserves_stats_and_invariants() {
        let mut world = small_world("snapshot-roundtrip");
        for _ in 0..20 {
            world.step();
        }
        let before = world.stats();
        let bytes = to_bytes(&world).unwrap();
        let loaded = from_bytes(&bytes).unwrap();
        loaded.check_invariants().unwrap();
        assert_eq!(before, loaded.stats());
    }

    #[test]
    fn snapshot_resume_matches_uninterrupted_run() {
        let mut uninterrupted = small_world("snapshot-resume");
        for _ in 0..50 {
            uninterrupted.step();
        }

        let mut resumed = small_world("snapshot-resume");
        for _ in 0..25 {
            resumed.step();
        }
        let bytes = to_bytes(&resumed).unwrap();
        let mut loaded = from_bytes(&bytes).unwrap();
        for _ in 0..25 {
            loaded.step();
        }

        assert_eq!(uninterrupted.stats(), loaded.stats());
        loaded.check_invariants().unwrap();
    }
}
