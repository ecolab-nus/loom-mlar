//! Optional architecture authoring and translation into the canonical MLAR core.
mod architecture;
pub mod archs;
mod builder;
pub mod compact;
pub mod selection;
mod yaml;

pub use architecture::{ArchLoadError, ChipYaml, ProcessorYaml};
pub use archs::{load_arch, load_arch_with_bindings};
pub use builder::{ArchitectureBuilder, Connection, ProcessorDefinition, ProcessorSourceFormat};
pub use compact::{LoomMemoryBinding, LoomParseError, lower_loom_source, parse_loom_source};
pub use mlar_rust::{PerfYamlError, PerformanceYaml};

pub fn write_artifact(
    architecture: &mlar_rust::Architecture,
    path: impl AsRef<std::path::Path>,
) -> Result<(), String> {
    architecture.validate().map_err(|error| error.to_string())?;
    let bytes = serde_json::to_vec_pretty(architecture).map_err(|error| error.to_string())?;
    std::fs::write(path, bytes).map_err(|error| error.to_string())
}
