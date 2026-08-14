//! JSON-over-stdin schedule evaluator for fixed architecture descriptions.

use std::path::{Path, PathBuf};

use crate::arch::Architecture;
use crate::schedule::evaluate::evaluate;
use crate::schedule::schedule::Schedule;

/// Read a [`Schedule`] from stdin and write its evaluated form to stdout.
pub fn run_evaluator(arch: &Architecture) -> Result<(), Box<dyn std::error::Error>> {
    let stdin = std::io::stdin();
    let input: Schedule = serde_json::from_reader(stdin.lock())?;

    let result = evaluate(&input, arch)?;

    let stdout = std::io::stdout();
    serde_json::to_writer(stdout.lock(), &result)?;
    Ok(())
}

/// Evaluate stdin against a JSON-serialized architecture.
pub fn run_evaluator_from_json(arch_json: &str) -> Result<(), Box<dyn std::error::Error>> {
    let arch: Architecture = serde_json::from_str(arch_json)?;
    run_evaluator(&arch)
}

/// Build a standalone evaluator binary at `output_dir/name`.
///
/// This invokes `cargo build --release` and requires `cargo` on `PATH`.
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

/// Define an evaluator binary for an architecture expression.
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
