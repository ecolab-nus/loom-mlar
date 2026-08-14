//! JSON-over-stdin query interface for fixed architecture descriptions.

use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::arch::Architecture;
use crate::mlir::architecture_to_mlir;

/// Query request payload for architecture-query binaries.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "query", rename_all = "snake_case")]
pub enum ArchitectureQuery {
    /// Return architecture MLIR in the `adl.*` dialect.
    Mlir,
}

/// In-process query result payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", content = "payload", rename_all = "snake_case")]
pub enum ArchitectureQueryResult {
    /// MLIR serialization of the architecture.
    Mlir(String),
}

/// Execute one architecture query in-process.
pub fn query_architecture(
    arch: &Architecture,
    query: &ArchitectureQuery,
) -> Result<ArchitectureQueryResult, String> {
    match query {
        ArchitectureQuery::Mlir => architecture_to_mlir(arch)
            .map(ArchitectureQueryResult::Mlir)
            .map_err(|error| error.to_string()),
    }
}

/// Read an [`ArchitectureQuery`] from stdin and write its result to stdout.
pub fn run_arch_query(arch: &Architecture) -> Result<(), Box<dyn std::error::Error>> {
    let stdin = std::io::stdin();
    let query: ArchitectureQuery = serde_json::from_reader(stdin.lock())?;

    let result = query_architecture(arch, &query)?;

    let mut stdout = std::io::stdout().lock();
    match result {
        ArchitectureQueryResult::Mlir(mlir) => stdout.write_all(mlir.as_bytes())?,
    }
    Ok(())
}

/// Run a query against a JSON-serialized architecture.
pub fn run_arch_query_from_json(arch_json: &str) -> Result<(), Box<dyn std::error::Error>> {
    let arch: Architecture = serde_json::from_str(arch_json)?;
    run_arch_query(&arch)
}

/// Build a standalone query binary at `output_dir/name`.
///
/// This invokes `cargo build --release` and requires `cargo` on `PATH`.
pub fn generate_arch_query_binary(
    arch: &Architecture,
    name: &str,
    output_dir: &Path,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let arch_json = serde_json::to_string(arch)?;

    let mlar_manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));

    let tmp_dir = output_dir.join(format!(".mlar_build_{name}_arch_query"));
    let src_dir = tmp_dir.join("src");
    std::fs::create_dir_all(&src_dir)?;

    let cargo_toml = format!(
        r#"[package]
name = "{name}"
version = "0.1.0"
edition = "2024"

[dependencies]
mlar-rust = {{ path = {mlar_path} }}
serde_json = "1.0"
"#,
        mlar_path = serde_json::to_string(&mlar_manifest_dir.to_string_lossy().as_ref())?
    );
    std::fs::write(tmp_dir.join("Cargo.toml"), cargo_toml)?;

    std::fs::write(src_dir.join("arch.json"), &arch_json)?;

    let main_rs = r#"fn main() {
    let arch_json = include_str!("arch.json");
    if let Err(e) = mlar_rust::run_arch_query_from_json(arch_json) {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
"#;
    std::fs::write(src_dir.join("main.rs"), main_rs)?;

    let status = std::process::Command::new("cargo")
        .args(["build", "--release"])
        .current_dir(&tmp_dir)
        .status()?;

    if !status.success() {
        return Err(format!("cargo build failed for architecture-query binary '{name}'").into());
    }

    std::fs::create_dir_all(output_dir)?;
    let compiled = tmp_dir.join("target/release").join(name);
    let output = output_dir.join(name);
    std::fs::copy(&compiled, &output)?;

    std::fs::remove_dir_all(&tmp_dir)?;

    Ok(output)
}

/// Define a query binary for an architecture expression.
///
/// # Example
///
/// ```ignore
/// // src/bin/query_my_arch.rs
/// use mlar_rust::mlar_arch_query;
///
/// fn build_arch() -> mlar_rust::Architecture {
///     // ... construct your architecture ...
///     # unimplemented!()
/// }
///
/// mlar_arch_query!(build_arch());
/// ```
#[macro_export]
macro_rules! mlar_arch_query {
    ($build_arch:expr) => {
        fn main() {
            let arch = $build_arch;
            if let Err(e) = $crate::run_arch_query(&arch) {
                eprintln!("{e}");
                std::process::exit(1);
            }
        }
    };
}
