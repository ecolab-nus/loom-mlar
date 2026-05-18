mod emitter;
mod names;
mod rewrite;

use crate::arch::architecture::Architecture;

use emitter::MlirEmitter;
use names::prefixed_arch_name;
use rewrite::rewrite_mlir_source;

/// Serialise an [`Architecture`] tree into the `adl.*` MLIR dialect,
/// wrapped in a top-level `module @<name> { ... }`.
///
/// Processor functionality MLIR sources (e.g. `matrix_lane.mlir`,
/// `vector_lane.mlir`) are appended inside the module, preserving
/// their own `module @xxx { ... }` structure.
///
/// Returns `None` if any dimension or memory size involves symbolic
/// expressions that cannot be simplified to constants.
pub fn architecture_to_mlir(arch: &Architecture) -> Option<String> {
    let mut emitter = MlirEmitter::new();
    emitter.emit_architecture(arch)?;

    let arch_name = arch.name().unwrap_or("unnamed");
    let mut result = format!("module @{} {{\n", prefixed_arch_name(arch_name));

    result.push_str(&indent(&emitter.output, 2));

    for path in &emitter.mlir_sources {
        if let Ok(content) = std::fs::read_to_string(path) {
            let rewritten = rewrite_mlir_source(
                &content,
                &emitter.processor_name_map,
                &emitter.memory_name_map,
            );
            result.push('\n');
            result.push_str(&indent(&rewritten, 2));
        }
    }

    result.push_str("}\n");
    Some(result)
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
