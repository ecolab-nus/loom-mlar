//! Optional architecture authoring and translation into the canonical MLAR core.
mod architecture;
pub mod archs;
mod builder;
mod emission;
mod native;
pub mod selection;
mod templates;
mod yaml;

pub use architecture::{ArchLoadError, ChipYaml, ProcessorYaml};
pub use archs::{load_arch, load_arch_with_bindings};
pub use builder::{ArchitectureBuilder, Connection, NamedPort, ProcessorDefinition};
pub use emission::emit_processor_sources;
pub use mlar_rust::{PerfYamlError, PerformanceYaml};

pub fn registered_templates() -> impl Iterator<Item = &'static str> {
    templates::registered_names()
}

pub fn write_artifact(
    architecture: &mlar_rust::Architecture,
    path: impl AsRef<std::path::Path>,
) -> Result<(), String> {
    architecture.validate().map_err(|error| error.to_string())?;
    let bytes = serde_json::to_vec_pretty(architecture).map_err(|error| error.to_string())?;
    std::fs::write(path, bytes).map_err(|error| error.to_string())
}
