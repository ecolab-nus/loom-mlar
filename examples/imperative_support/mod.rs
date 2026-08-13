use std::error::Error;
use std::path::{Path, PathBuf};

use mlar_rust::arch::ProcessorYaml;
use mlar_rust::{Connection, MemoryEndpoint, ProcessorDefinition};

pub type ExampleResult<T> = Result<T, Box<dyn Error>>;

pub fn architecture_dir(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples/architectures")
        .join(name)
}

pub fn processor_definition(directory: &Path, name: &str) -> ExampleResult<ProcessorDefinition> {
    let path = directory.join(format!("{name}.yaml"));
    let yaml = ProcessorYaml::from_file(&path)?;
    Ok(yaml.build_definition(&path)?)
}

pub fn connection(domain: &[&str], inputs: &[&str], outputs: &[&str]) -> ExampleResult<Connection> {
    Ok(Connection::new(
        domain.iter().copied(),
        inputs
            .iter()
            .map(|endpoint| MemoryEndpoint::parse(endpoint))
            .collect::<Result<Vec<_>, _>>()?,
        outputs
            .iter()
            .map(|endpoint| MemoryEndpoint::parse(endpoint))
            .collect::<Result<Vec<_>, _>>()?,
    ))
}
