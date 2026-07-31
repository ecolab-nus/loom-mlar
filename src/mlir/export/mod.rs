mod emitter;
mod names;
mod rewrite;

use std::ffi::OsStr;
use std::fmt;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

use crate::arch::architecture::Architecture;

use emitter::MlirEmitter;
use names::prefixed_arch_name;
use rewrite::rewrite_mlir_source;

const QUANTITATIVE_RESOURCE_OP: &str = "adl.resource.quantitative";
const ADL_OPT: &str = env!("MLAR_BUILD_ADL_OPT");
const LOOM_OPT: &str = env!("MLAR_BUILD_LOOM_OPT");

/// Errors produced while exporting or validating architecture MLIR.
#[derive(Debug)]
pub enum MlirExportError {
    NonConcreteArchitecture,
    SourceRead {
        path: PathBuf,
        source: io::Error,
    },
    ToolNotFound {
        tool: &'static str,
        program: PathBuf,
    },
    ToolInvocation {
        tool: &'static str,
        program: PathBuf,
        source: io::Error,
    },
    InvalidAdl {
        program: PathBuf,
        stderr: String,
    },
    InvalidLoomMlir {
        program: PathBuf,
        stderr: String,
    },
    UnsupportedExperimentalFeature {
        feature: &'static str,
    },
}

impl fmt::Display for MlirExportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonConcreteArchitecture => write!(
                f,
                "architecture contains symbolic dimensions or memory sizes that cannot be concretized"
            ),
            Self::SourceRead { path, source } => {
                write!(
                    f,
                    "failed to read processor MLIR '{}': {source}",
                    path.display()
                )
            }
            Self::ToolNotFound { tool, program } => write!(
                f,
                "{tool} validator was not found at '{}'",
                program.display()
            ),
            Self::ToolInvocation {
                tool,
                program,
                source,
            } => write!(
                f,
                "failed to invoke {tool} validator '{}': {source}",
                program.display()
            ),
            Self::InvalidAdl { program, stderr } => write!(
                f,
                "ADL validation failed in '{}':\n{stderr}",
                program.display()
            ),
            Self::InvalidLoomMlir { program, stderr } => write!(
                f,
                "complete MLIR validation failed in '{}':\n{stderr}",
                program.display()
            ),
            Self::UnsupportedExperimentalFeature { feature } => write!(
                f,
                "experimental feature '{feature}' is emitted by loom-mlar but is not supported by the MLIR compiler"
            ),
        }
    }
}

impl std::error::Error for MlirExportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::SourceRead { source, .. } | Self::ToolInvocation { source, .. } => Some(source),
            _ => None,
        }
    }
}

struct GeneratedMlir {
    adl_only: String,
    complete: String,
    has_quantitative_resource: bool,
}

/// Serialize an [`Architecture`] and validate the result with both compiler
/// frontends. `adl-opt` validates the architecture-only module first, then
/// `loom-opt` validates the complete module including processor functionality.
///
/// Both validator paths are discovered and verified by loom-mlar's Cargo build
/// script, so callers do not need to configure environment variables or
/// modify `PATH`.
pub fn architecture_to_mlir(arch: &Architecture) -> Result<String, MlirExportError> {
    architecture_to_mlir_with_tools(arch, OsStr::new(ADL_OPT), OsStr::new(LOOM_OPT))
}

/// Serialize an [`Architecture`] without invoking external validators.
///
/// This is intended for debugging and experimental features such as
/// `adl.resource.quantitative`, which the current MLIR compiler does not yet
/// support.
pub fn architecture_to_mlir_unchecked(arch: &Architecture) -> Result<String, MlirExportError> {
    Ok(generate_mlir(arch)?.complete)
}

fn architecture_to_mlir_with_tools(
    arch: &Architecture,
    adl_opt: &OsStr,
    loom_opt: &OsStr,
) -> Result<String, MlirExportError> {
    let generated = generate_mlir(arch)?;
    if generated.has_quantitative_resource {
        return Err(MlirExportError::UnsupportedExperimentalFeature {
            feature: QUANTITATIVE_RESOURCE_OP,
        });
    }

    validate_mlir(
        adl_opt,
        "adl-opt",
        &generated.adl_only,
        ValidationStage::Adl,
    )?;
    validate_mlir(
        loom_opt,
        "loom-opt",
        &generated.complete,
        ValidationStage::Loom,
    )?;
    Ok(generated.complete)
}

fn generate_mlir(arch: &Architecture) -> Result<GeneratedMlir, MlirExportError> {
    let mut emitter = MlirEmitter::new();
    emitter
        .emit_architecture(arch)
        .ok_or(MlirExportError::NonConcreteArchitecture)?;

    let arch_name = arch.name().unwrap_or("unnamed");
    let module_header = format!("module @{} {{\n", prefixed_arch_name(arch_name));
    let architecture_body = indent(&emitter.output, 2);

    let mut adl_only = module_header.clone();
    adl_only.push_str(&architecture_body);
    adl_only.push_str("}\n");

    let mut complete = module_header;
    complete.push_str(&architecture_body);
    for path in &emitter.mlir_sources {
        let content =
            std::fs::read_to_string(path).map_err(|source| MlirExportError::SourceRead {
                path: PathBuf::from(path),
                source,
            })?;
        let rewritten = rewrite_mlir_source(
            &content,
            &emitter.processor_name_map,
            &emitter.memory_name_map,
        );
        complete.push('\n');
        complete.push_str(&indent(&rewritten, 2));
    }
    complete.push_str("}\n");

    Ok(GeneratedMlir {
        has_quantitative_resource: emitter.output.contains(QUANTITATIVE_RESOURCE_OP),
        adl_only,
        complete,
    })
}

#[derive(Clone, Copy)]
enum ValidationStage {
    Adl,
    Loom,
}

fn validate_mlir(
    program: &OsStr,
    tool: &'static str,
    mlir: &str,
    stage: ValidationStage,
) -> Result<(), MlirExportError> {
    let program_path = PathBuf::from(program);
    let mut child = Command::new(program)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|source| {
            if source.kind() == io::ErrorKind::NotFound {
                MlirExportError::ToolNotFound {
                    tool,
                    program: program_path.clone(),
                }
            } else {
                MlirExportError::ToolInvocation {
                    tool,
                    program: program_path.clone(),
                    source,
                }
            }
        })?;

    let write_result = child
        .stdin
        .take()
        .expect("piped validator stdin must be available")
        .write_all(mlir.as_bytes());
    let output = child
        .wait_with_output()
        .map_err(|source| MlirExportError::ToolInvocation {
            tool,
            program: program_path.clone(),
            source,
        })?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        return Err(match stage {
            ValidationStage::Adl => MlirExportError::InvalidAdl {
                program: program_path,
                stderr,
            },
            ValidationStage::Loom => MlirExportError::InvalidLoomMlir {
                program: program_path,
                stderr,
            },
        });
    }

    write_result.map_err(|source| MlirExportError::ToolInvocation {
        tool,
        program: program_path,
        source,
    })
}

/// Indent every line of `text` by `indent` spaces.
fn indent(text: &str, indent: usize) -> String {
    let prefix = " ".repeat(indent);
    let mut result = String::new();
    for line in text.lines() {
        if line.is_empty() {
            result.push('\n');
        } else {
            result.push_str(&prefix);
            result.push_str(line);
            result.push('\n');
        }
    }
    result
}

#[cfg(test)]
#[path = "../tests/export.rs"]
mod tests;
