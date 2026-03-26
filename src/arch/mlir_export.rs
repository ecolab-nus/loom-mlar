use std::collections::HashMap;
use std::fmt::Write;

use super::architecture::Architecture;
use super::architecture_graph::ArchNodeComponent;
use super::data_mover::DataMover;
use super::memory::{MemoryBank, MemoryRegion};
use super::processor::Processor;
use super::size_dim::Dimension;

/// SSA-based emitter that serialises an [`Architecture`] tree into the
/// `adl.*` MLIR dialect.
struct MlirEmitter {
    counter: usize,
    output: String,
    /// Dimension name → SSA value (avoids emitting the same dim twice).
    dim_map: HashMap<String, String>,
    /// MLIR source paths collected from functionality modules
    /// (insertion-ordered, deduplicated).
    mlir_sources: Vec<String>,
}

impl MlirEmitter {
    fn new() -> Self {
        Self {
            counter: 0,
            output: String::new(),
            dim_map: HashMap::new(),
            mlir_sources: Vec::new(),
        }
    }

    fn next_ssa(&mut self) -> String {
        let name = format!("%{}", self.counter);
        self.counter += 1;
        name
    }

    /// Emit a `adl.spatial_dim` if this dimension has not been emitted yet.
    /// Returns `None` when the size cannot be simplified to a constant.
    fn emit_dim(&mut self, dim: &Dimension) -> Option<String> {
        if let Some(existing) = self.dim_map.get(&dim.name.0) {
            return Some(existing.clone());
        }
        let size = dim.size.simplify_constant()?;
        let ssa = self.next_ssa();
        writeln!(
            self.output,
            "{} = adl.spatial_dim \"{}\", {}",
            ssa, dim.name.0, size
        )
        .unwrap();
        self.dim_map.insert(dim.name.0.clone(), ssa.clone());
        Some(ssa)
    }

    /// Emit the `adl.memory.*` ops for a [`MemoryRegion`] tree.
    fn emit_memory(&mut self, region: &MemoryRegion) -> Option<String> {
        match region {
            MemoryRegion::Bank(bank) => self.emit_bank(bank),
            MemoryRegion::Array {
                name,
                dims,
                sub_regions,
            } => {
                let sub_ssa = self.emit_memory(sub_regions)?;
                let dim_ssas: Vec<String> = dims
                    .iter()
                    .map(|d| self.emit_dim(d))
                    .collect::<Option<Vec<_>>>()?;
                let ssa = self.next_ssa();
                let name_str = name.as_deref().unwrap_or("unnamed");
                let scaleout = dim_ssas.join(", ");
                writeln!(
                    self.output,
                    "{} = adl.memory.array \"{}\", ({}) of {}",
                    ssa, name_str, scaleout, sub_ssa
                )
                .unwrap();
                Some(ssa)
            }
        }
    }

    /// Emit a `adl.memory.bank` op.
    fn emit_bank(&mut self, bank: &MemoryBank) -> Option<String> {
        let (bsize, nblk) = if let Some(ref bs) = bank.block_size {
            let bsize = bs.simplify_constant()?;
            let cap = bank.capacity_bytes.simplify_constant()?;
            (bsize, cap / bsize)
        } else {
            (1, bank.capacity_bytes.simplify_constant()?)
        };

        let ssa = self.next_ssa();
        let name = bank.name.as_deref().unwrap_or("bank");
        writeln!(
            self.output,
            "{} = adl.memory.bank \"{}\", {{bsize = {}, nblk = {}}}",
            ssa, name, bsize, nblk
        )
        .unwrap();
        Some(ssa)
    }

    /// Emit a `adl.processor` op and record its MLIR source (if any).
    fn emit_processor(&mut self, proc: &Processor) -> String {
        let ssa = self.next_ssa();
        let name = proc.name.as_deref().unwrap_or("unnamed");

        let module_ref = proc
            .functionality
            .source
            .as_ref()
            .and_then(|s| s.mlir_module_name.as_deref())
            .or(proc.functionality.name.as_deref());

        if let Some(mod_name) = module_ref {
            writeln!(
                self.output,
                "{} = adl.processor \"{}\", @{}",
                ssa, name, mod_name
            )
            .unwrap();
        } else {
            writeln!(self.output, "{} = adl.processor \"{}\"", ssa, name).unwrap();
        }

        if let Some(ref source) = proc.functionality.source {
            if !self.mlir_sources.contains(&source.path) {
                self.mlir_sources.push(source.path.clone());
            }
        }

        ssa
    }

    /// Emit an `adl.dmover` op and record its MLIR source (if any).
    fn emit_data_mover(&mut self, mover: &DataMover) -> String {
        let ssa = self.next_ssa();
        let name = mover.name.as_deref().unwrap_or("unnamed");

        let module_ref = mover
            .functionality
            .source
            .as_ref()
            .and_then(|s| s.mlir_module_name.as_deref())
            .or(mover.functionality.name.as_deref());

        if let Some(mod_name) = module_ref {
            writeln!(
                self.output,
                "{} = adl.dmover \"{}\", @{}",
                ssa, name, mod_name
            )
            .unwrap();
        } else {
            writeln!(self.output, "{} = adl.dmover \"{}\"", ssa, name).unwrap();
        }

        if let Some(ref source) = mover.functionality.source {
            if !self.mlir_sources.contains(&source.path) {
                self.mlir_sources.push(source.path.clone());
            }
        }

        ssa
    }

    /// Recursively emit the full architecture tree, returning the SSA value
    /// for the top-level result.
    fn emit_architecture(&mut self, arch: &Architecture) -> Option<String> {
        match arch {
            Architecture::Unit(proc) => Some(self.emit_processor(proc)),
            Architecture::Graph(graph) => {
                let mut component_ssas = Vec::new();
                for node in &graph.nodes {
                    match &node.component {
                        ArchNodeComponent::Architecture(sub_arch) => {
                            if let Some(ssa) = self.emit_architecture(sub_arch) {
                                component_ssas.push(ssa);
                            }
                        }
                        ArchNodeComponent::MemoryRegion(region) => {
                            if let Some(ssa) = self.emit_memory(region) {
                                component_ssas.push(ssa);
                            }
                        }
                        ArchNodeComponent::DataMover(mover) => {
                            component_ssas.push(self.emit_data_mover(mover));
                        }
                        // Routers are not part of the df dialect.
                        _ => {}
                    }
                }
                let ssa = self.next_ssa();
                let components = component_ssas.join(", ");
                writeln!(
                    self.output,
                    "{} = adl.arch.compose \"{}\", [{}]",
                    ssa, graph.name, components
                )
                .unwrap();
                Some(ssa)
            }
            Architecture::Array {
                name, dims, elem, ..
            } => {
                let elem_ssa = self.emit_architecture(elem)?;
                let dim_ssas: Vec<String> = dims
                    .iter()
                    .map(|d| self.emit_dim(d))
                    .collect::<Option<Vec<_>>>()?;
                let ssa = self.next_ssa();
                let arch_name = name.as_deref().or_else(|| elem.name()).unwrap_or("unnamed");
                let dim_list = dim_ssas.join(", ");
                writeln!(
                    self.output,
                    "{} = adl.arch.scale \"{}\", ({}) of {}",
                    ssa, arch_name, dim_list, elem_ssa
                )
                .unwrap();
                Some(ssa)
            }
        }
    }
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
    let mut result = format!("module @{} {{\n", arch_name);

    result.push_str(&indent(&emitter.output, 2));

    for path in &emitter.mlir_sources {
        if let Ok(content) = std::fs::read_to_string(path) {
            result.push('\n');
            result.push_str(&indent(&content, 2));
        }
    }

    result.push_str("}\n");
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::{ArchGraph, MemoryBank, MemoryRegion, Processor, SizeExpr};

    #[test]
    fn single_processor_emits_df_processor() {
        let arch = Processor::new("vec_lane").into_elem();
        let mlir = architecture_to_mlir(&arch).expect("should emit");
        assert!(mlir.contains("adl.processor \"vec_lane\""));
    }

    #[test]
    fn output_wrapped_in_module() {
        let arch = Processor::new("p").into_elem();
        let mlir = architecture_to_mlir(&arch).expect("should emit");
        assert!(mlir.starts_with("module @p {\n"));
        assert!(mlir.ends_with("}\n"));
    }

    #[test]
    fn graph_emits_compose() {
        let l1 = MemoryRegion::bank(MemoryBank::new(SizeExpr::Const(1024))).with_name("L1");
        let lane = Processor::new("lane").into_elem();
        let arch: Architecture = ArchGraph::builder("core")
            .mem(&l1)
            .processor(&lane)
            .build()
            .into();
        let mlir = architecture_to_mlir(&arch).expect("should emit");
        assert!(mlir.contains("adl.memory.bank"));
        assert!(mlir.contains("adl.processor \"lane\""));
        assert!(mlir.contains("adl.arch.compose \"core\""));
    }

    #[test]
    fn array_emits_scale_with_dims() {
        let dim = Dimension::new_int("x", 8);
        let lane = Processor::new("lane").into_elem();
        let arch = lane.scale(&[dim]).with_name("mesh");
        let mlir = architecture_to_mlir(&arch).expect("should emit");
        assert!(mlir.contains("adl.spatial_dim \"x\", 8"));
        assert!(mlir.contains("adl.arch.scale \"mesh\""));
    }

    #[test]
    fn symbolic_dim_returns_none() {
        let dim = Dimension::new_sym("x", "N");
        let lane = Processor::new("lane").into_elem();
        let arch = lane.scale(&[dim]).with_name("mesh");
        assert!(architecture_to_mlir(&arch).is_none());
    }

    #[test]
    fn shared_dims_emitted_once() {
        let dim_x = Dimension::new_int("x", 4);
        let l1 = MemoryRegion::bank(MemoryBank::from_blocks(
            SizeExpr::Const(64),
            SizeExpr::Const(128),
        ))
        .scale(dim_x.as_slice())
        .with_name("L1");
        let lane = Processor::new("p").into_elem();
        let core: Architecture = ArchGraph::builder("core")
            .mem(&l1)
            .processor(&lane)
            .build()
            .into();
        let scaled = core.scale(&[dim_x]).with_name("mesh");
        let mlir = architecture_to_mlir(&scaled).expect("should emit");
        assert_eq!(
            mlir.matches("adl.spatial_dim \"x\"").count(),
            1,
            "dimension x should be emitted exactly once"
        );
    }
}
