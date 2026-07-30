//! Binary evaluator generation for architecture descriptions.
//!
//! This module provides the infrastructure to produce standalone executables
//! that evaluate schedules against a fixed architecture.  External (non-Rust)
//! tools can invoke the binary, pass a [`Schedule`] as JSON on **stdin**, and
//! receive the evaluated [`Schedule`] (with `scenarios` filled) as JSON on
//! **stdout**.
//!
//! # Three ways to create an evaluator binary
//!
//! ## 1. Macro — define the architecture in Rust code
//!
//! ```ignore
//! // src/bin/eval_my_arch.rs
//! use mlar_rust::mlar_evaluator;
//!
//! mlar_evaluator!(my_crate::build_architecture());
//! ```
//!
//! ## 2. Library function — call from your own `main()`
//!
//! ```ignore
//! fn main() {
//!     let arch = build_architecture();
//!     mlar_rust::run_evaluator(&arch).unwrap();
//! }
//! ```
//!
//! ## 3. Fully programmatic — generate a binary from a runtime `Architecture`
//!
//! ```ignore
//! let arch = build_architecture();
//! let binary = mlar_rust::generate_evaluator_binary(
//!     &arch,
//!     "my_arch_eval",
//!     Path::new("output/"),
//! )?;
//! // `binary` is the path to the compiled executable
//! ```

use std::path::{Path, PathBuf};

use crate::arch::Architecture;
use crate::schedule::evaluate::evaluate;
use crate::schedule::schedule::Schedule;

/// Run the schedule-evaluation pipeline:
///
/// 1. Read JSON from **stdin** (`Schedule`)
/// 2. Evaluate against `arch`
/// 3. Write evaluated `Schedule` JSON to **stdout**
pub fn run_evaluator(arch: &Architecture) -> Result<(), Box<dyn std::error::Error>> {
    let stdin = std::io::stdin();
    let input: Schedule = serde_json::from_reader(stdin.lock())?;

    let result = evaluate(&input, arch)?;

    let stdout = std::io::stdout();
    serde_json::to_writer(stdout.lock(), &result)?;
    Ok(())
}

/// Run the evaluator with an architecture deserialized from a JSON string.
///
/// This is the entry point used by binaries produced via
/// [`generate_evaluator_binary`] — the architecture JSON is embedded in the
/// binary at compile time.
pub fn run_evaluator_from_json(arch_json: &str) -> Result<(), Box<dyn std::error::Error>> {
    let arch: Architecture = serde_json::from_str(arch_json)?;
    run_evaluator(&arch)
}

/// Generate a standalone evaluator binary for the given architecture.
///
/// The function serializes `arch` to JSON, creates a temporary Cargo project
/// that embeds it, compiles with `cargo build --release`, and copies the
/// resulting binary to `output_dir/name`.
///
/// **Requires a Rust toolchain** (`cargo`) to be available on `$PATH`.
///
/// Returns the path to the generated binary.
pub fn generate_evaluator_binary(
    arch: &Architecture,
    name: &str,
    output_dir: &Path,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let arch_json = serde_json::to_string(arch)?;

    let mlar_manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));

    let tmp_dir = output_dir.join(format!(".mlar_build_{name}"));
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
    if let Err(e) = mlar_rust::run_evaluator_from_json(arch_json) {
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
        return Err(format!("cargo build failed for evaluator binary '{name}'").into());
    }

    std::fs::create_dir_all(output_dir)?;
    let compiled = tmp_dir.join("target/release").join(name);
    let output = output_dir.join(name);
    std::fs::copy(&compiled, &output)?;

    std::fs::remove_dir_all(&tmp_dir)?;

    Ok(output)
}

/// Generate a `main` function that builds an architecture and runs the
/// evaluator pipeline (stdin JSON → evaluate → stdout JSON).
///
/// # Example
///
/// ```ignore
/// // src/bin/eval_my_arch.rs
/// use mlar_rust::mlar_evaluator;
///
/// fn build_arch() -> mlar_rust::Architecture {
///     // ... construct your architecture ...
///     # unimplemented!()
/// }
///
/// mlar_evaluator!(build_arch());
/// ```
#[macro_export]
macro_rules! mlar_evaluator {
    ($build_arch:expr) => {
        fn main() {
            let arch = $build_arch;
            if let Err(e) = $crate::run_evaluator(&arch) {
                eprintln!("{e}");
                std::process::exit(1);
            }
        }
    };
}
