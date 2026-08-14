use std::error::Error;
use std::path::{Path, PathBuf};

use mlar_rust::{Connection, MemoryEndpoint};

pub type ExampleResult<T> = Result<T, Box<dyn Error>>;

pub fn architecture_dir(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples/declarative")
        .join(name)
}

/// Parse a connection over an ordered axis domain.
pub fn connection<'a>(
    domain: impl IntoIterator<Item = &'a str>,
    inputs: impl IntoIterator<Item = &'a str>,
    outputs: impl IntoIterator<Item = &'a str>,
) -> ExampleResult<Connection> {
    Ok(Connection::new(
        domain,
        parse_endpoints(inputs)?,
        parse_endpoints(outputs)?,
    ))
}

fn parse_endpoints<'a>(
    endpoints: impl IntoIterator<Item = &'a str>,
) -> ExampleResult<Vec<MemoryEndpoint>> {
    endpoints
        .into_iter()
        .map(MemoryEndpoint::parse)
        .collect::<Result<Vec<_>, _>>()
        .map_err(Into::into)
}
