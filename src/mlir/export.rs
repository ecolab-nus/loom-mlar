use std::collections::{HashMap, HashSet};
use std::fmt::Write;

use crate::arch::architecture::Architecture;
use crate::arch::architecture_graph::ArchNodeComponent;
use crate::arch::memory::{MemoryBank, MemoryRegion};
use crate::arch::processor::{DataMover, Processor};
use crate::arch::resource::Resource;
use crate::arch::size_dim::Dimension;

/// SSA-based emitter that serialises an [`Architecture`] tree into the
/// `adl.*` MLIR dialect.
struct MlirEmitter {
    counter: usize,
    output: String,
    /// Dimension name → SSA value (avoids emitting the same dim twice).
    dim_map: HashMap<String, String>,
    /// Memory-region name → SSA value.
    memory_map: HashMap<String, String>,
    /// Structural memory-region key → SSA value.
    memory_region_map: HashMap<String, String>,
    /// Resource id → SSA value.
    resource_map: HashMap<String, String>,
    /// Memory name → prefixed MLIR memory name used in exported ops.
    memory_name_map: HashMap<String, String>,
    /// Processor name → prefixed MLIR processor symbol.
    processor_name_map: HashMap<String, String>,
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
            memory_map: HashMap::new(),
            memory_region_map: HashMap::new(),
            resource_map: HashMap::new(),
            memory_name_map: HashMap::new(),
            processor_name_map: HashMap::new(),
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
            ssa,
            prefixed_dim_name(&dim.name.0),
            size
        )
        .unwrap();
        self.dim_map.insert(dim.name.0.clone(), ssa.clone());
        Some(ssa)
    }

    /// Emit the `adl.memory.*` ops for a [`MemoryRegion`] tree.
    fn emit_memory(&mut self, region: &MemoryRegion) -> Option<String> {
        let key = memory_region_key(region);
        let dedupe_named_region = region.name().is_some();
        if dedupe_named_region {
            if let Some(existing) = self.memory_region_map.get(&key).cloned() {
                self.record_memory_name(region, &existing);
                return Some(existing);
            }
        }

        let ssa = match region {
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
                let name_str = prefixed_memory_name(name.as_deref().unwrap_or("unnamed"));
                let scaleout = dim_ssas.join(", ");
                writeln!(
                    self.output,
                    "{} = adl.memory.array \"{}\", [{}] of {}",
                    ssa, name_str, scaleout, sub_ssa
                )
                .unwrap();
                Some(ssa)
            }
        }?;

        if dedupe_named_region {
            self.memory_region_map.insert(key, ssa.clone());
        }
        self.record_memory_name(region, &ssa);
        Some(ssa)
    }

    fn record_memory_name(&mut self, region: &MemoryRegion, ssa: &str) {
        if let Some(name) = region.name() {
            self.memory_map.insert(name.to_string(), ssa.to_string());
            self.memory_name_map
                .insert(name.to_string(), prefixed_memory_name(name));
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
        let name = prefixed_memory_name(bank.name.as_deref().unwrap_or("bank"));
        writeln!(
            self.output,
            "{} = adl.memory.bank \"{}\", {{bsize = {}, nblk = {}}}",
            ssa, name, bsize, nblk
        )
        .unwrap();
        Some(ssa)
    }

    /// Emit a typed `adl.resource.*` op if not already emitted, returning the SSA value.
    fn emit_resource(&mut self, resource: &Resource) -> String {
        if let Some(existing) = self.resource_map.get(resource.id().as_str()) {
            return existing.clone();
        }
        let ssa = self.next_ssa();
        match resource {
            Resource::Exclusive { id } => {
                let name = prefixed_resource_name(id.as_str());
                writeln!(self.output, "{} = adl.resource.exclusive \"{}\"", ssa, name).unwrap();
            }
            Resource::Quantitative { id, capacity } => {
                let name = prefixed_resource_name(id.as_str());
                writeln!(
                    self.output,
                    "{} = adl.resource.quantitative \"{}\", {{capacity = {}}}",
                    ssa, name, capacity
                )
                .unwrap();
            }
        }
        self.resource_map
            .insert(resource.id().as_str().to_string(), ssa.clone());
        ssa
    }

    /// Emit a `adl.processor` op and record its MLIR source (if any).
    fn emit_processor(&mut self, proc: &Processor) -> Option<String> {
        self.emit_processor_like("compute", proc)
    }

    /// Emit a `adl.processor.dmover` op and record its MLIR source (if any).
    fn emit_data_mover(&mut self, mover: &DataMover) -> Option<String> {
        self.emit_processor_like("dmover", mover)
    }

    fn emit_processor_like(&mut self, kind: &str, proc: &Processor) -> Option<String> {
        let ssa = self.next_ssa();
        let name = proc.name.as_deref().unwrap_or("unnamed");
        let prefixed_name = prefixed_processor_name(name);
        self.processor_name_map
            .insert(name.to_string(), prefixed_name.clone());
        let region_pairs = self.format_region_pairs(&proc.region_pairs)?;
        let proc_ref = if matches!(kind, "compute" | "dmover") {
            format!("@{}", prefixed_name)
        } else {
            format!("\"{}\"", prefixed_name)
        };

        let resource_clause = self.format_resource_clause(&proc.resources);

        writeln!(
            self.output,
            "{} = adl.processor.{} {}, {}{}",
            ssa, kind, proc_ref, region_pairs, resource_clause
        )
        .unwrap();

        if let Some(source_path) = proc.functionality.path.as_ref() {
            if !self.mlir_sources.contains(source_path) {
                self.mlir_sources.push(source_path.clone());
            }
        }

        Some(ssa)
    }

    fn format_resource_clause(&self, resources: &[Resource]) -> String {
        if resources.is_empty() {
            return String::new();
        }
        let ssas: Vec<&str> = resources
            .iter()
            .filter_map(|r| self.resource_map.get(r.id().as_str()))
            .map(|s| s.as_str())
            .collect();
        if ssas.is_empty() {
            return String::new();
        }
        format!(", with [{}]", ssas.join(", "))
    }

    fn format_region_pairs(&self, region_pairs: &[(MemoryRegion, MemoryRegion)]) -> Option<String> {
        let pairs: Vec<String> = region_pairs
            .iter()
            .map(|(src, dst)| {
                let src_name = src.name()?;
                let dst_name = dst.name()?;
                let src_ssa = self.memory_map.get(src_name)?;
                let dst_ssa = self.memory_map.get(dst_name)?;
                Some(format!("({}, {})", src_ssa, dst_ssa))
            })
            .collect::<Option<Vec<_>>>()?;
        Some(format!("[{}]", pairs.join(", ")))
    }

    /// Recursively emit the full architecture tree, returning the SSA value
    /// for the top-level result.
    fn emit_architecture(&mut self, arch: &Architecture) -> Option<String> {
        match arch {
            Architecture::Unit(proc) => self.emit_processor(proc),
            Architecture::Graph(graph) => {
                let mut arch_components: Vec<String> = Vec::new();
                let mut mem_components: Vec<String> = Vec::new();
                let mut memory_resource_ids: HashSet<String> = HashSet::new();
                for node in &graph.nodes {
                    match &node.component {
                        ArchNodeComponent::MemoryRegion(region) => {
                            if let Ok(resource) = region.generate_resource() {
                                memory_resource_ids.insert(resource.id().as_str().to_string());
                            }
                        }
                        ArchNodeComponent::DataMover(mover) => {
                            collect_region_pair_memory_resource_ids(
                                &mover.region_pairs,
                                &mut memory_resource_ids,
                            );
                        }
                        ArchNodeComponent::Architecture(sub_arch) => {
                            collect_arch_memory_resource_ids(sub_arch, &mut memory_resource_ids);
                        }
                        _ => {}
                    }
                }

                // Emit memory first so processors can safely reference memory SSA values.
                for node in &graph.nodes {
                    if let ArchNodeComponent::MemoryRegion(region) = &node.component {
                        mem_components.push(self.emit_memory(region)?);
                    }
                }

                // Emit resources so processors can reference them.
                for resource in &graph.resources {
                    if memory_resource_ids.contains(resource.id().as_str()) {
                        continue;
                    }
                    self.emit_resource(resource);
                }

                for node in &graph.nodes {
                    match &node.component {
                        ArchNodeComponent::Architecture(sub_arch) => {
                            arch_components.push(self.emit_architecture(sub_arch)?);
                        }
                        ArchNodeComponent::DataMover(mover) => {
                            self.emit_region_pair_array_memories(&mover.region_pairs)?;
                            arch_components.push(self.emit_data_mover(mover)?);
                        }
                        ArchNodeComponent::MemoryRegion(_) => {}
                        // Routers are not part of the df dialect.
                        _ => {}
                    }
                }
                let ssa = self.next_ssa();
                let arch = arch_components.join(", ");
                let mem = mem_components.join(", ");
                writeln!(
                    self.output,
                    "{} = adl.arch.compose \"{}\", arch[{}], mem[{}]",
                    ssa,
                    prefixed_arch_name(&graph.name),
                    arch,
                    mem
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
                    "{} = adl.arch.scale \"{}\", [{}] of {}",
                    ssa,
                    prefixed_arch_name(arch_name),
                    dim_list,
                    elem_ssa
                )
                .unwrap();
                Some(ssa)
            }
        }
    }

    fn emit_region_pair_array_memories(
        &mut self,
        region_pairs: &[(MemoryRegion, MemoryRegion)],
    ) -> Option<()> {
        for (src, dst) in region_pairs {
            if matches!(src, MemoryRegion::Array { .. }) {
                self.emit_memory(src)?;
            }
            if matches!(dst, MemoryRegion::Array { .. }) {
                self.emit_memory(dst)?;
            }
        }
        Some(())
    }
}

fn memory_region_key(region: &MemoryRegion) -> String {
    match region {
        MemoryRegion::Bank(bank) => format!(
            "bank:{:?}:{}:{:?}",
            bank.name, bank.capacity_bytes, bank.block_size
        ),
        MemoryRegion::Array {
            name,
            dims,
            sub_regions,
        } => {
            let dims = dims
                .iter()
                .map(|d| format!("{}={}", d.name.0, d.size))
                .collect::<Vec<_>>()
                .join(",");
            format!(
                "array:{:?}:[{}]:{}",
                name,
                dims,
                memory_region_key(sub_regions)
            )
        }
    }
}

/// Collect memory resource IDs used by region pairs.
fn collect_region_pair_memory_resource_ids(
    region_pairs: &[(MemoryRegion, MemoryRegion)],
    out: &mut HashSet<String>,
) {
    for (src, dst) in region_pairs {
        if let Some(id) = memory_region_resource_id(src) {
            out.insert(id);
        }
        if let Some(id) = memory_region_resource_id(dst) {
            out.insert(id);
        }
    }
}

fn memory_region_resource_id(region: &MemoryRegion) -> Option<String> {
    match region {
        MemoryRegion::Bank(_) => region
            .generate_resource()
            .ok()
            .map(|resource| resource.id().as_str().to_string()),
        MemoryRegion::Array { name, .. } => name
            .as_deref()
            .or_else(|| region.name())
            .map(ToString::to_string),
    }
}

/// Recursively collect memory resource IDs referenced by an
/// architecture tree (processors, data movers, and memory nodes).
fn collect_arch_memory_resource_ids(arch: &Architecture, out: &mut HashSet<String>) {
    match arch {
        Architecture::Unit(proc) => {
            collect_region_pair_memory_resource_ids(&proc.region_pairs, out);
        }
        Architecture::Array { elem, .. } => {
            collect_arch_memory_resource_ids(elem, out);
        }
        Architecture::Graph(graph) => {
            for node in &graph.nodes {
                match &node.component {
                    ArchNodeComponent::Architecture(sub_arch) => {
                        collect_arch_memory_resource_ids(sub_arch, out);
                    }
                    ArchNodeComponent::DataMover(mover) => {
                        collect_region_pair_memory_resource_ids(&mover.region_pairs, out);
                    }
                    ArchNodeComponent::MemoryRegion(region) => {
                        if let Ok(resource) = region.generate_resource() {
                            out.insert(resource.id().as_str().to_string());
                        }
                    }
                    _ => {}
                }
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

fn prefixed_resource_name(name: &str) -> String {
    format!("res_{name}")
}

fn prefixed_processor_name(name: &str) -> String {
    format!("proc_{name}")
}

fn prefixed_memory_name(name: &str) -> String {
    format!("mem_{name}")
}

fn prefixed_arch_name(name: &str) -> String {
    format!("arch_{name}")
}

fn prefixed_dim_name(name: &str) -> String {
    format!("dim_{name}")
}

fn extract_module_symbol(line: &str) -> Option<&str> {
    let trimmed = line.trim_start();
    let rest = trimmed.strip_prefix("module @")?;
    let end = rest
        .find(|c: char| c.is_whitespace() || c == '{')
        .unwrap_or(rest.len());
    if end == 0 {
        return None;
    }
    Some(&rest[..end])
}

fn rewrite_module_symbol_line(
    line: &str,
    processor_name_map: &HashMap<String, String>,
) -> Option<String> {
    let module_symbol = extract_module_symbol(line)?;
    let prefixed = processor_name_map.get(module_symbol)?;
    Some(line.replacen(&format!("@{module_symbol}"), &format!("@{prefixed}"), 1))
}

fn rewrite_bind_mem_line(line: &str, memory_name_map: &HashMap<String, String>) -> String {
    if !line.contains("loom.bind_mem") {
        return line.to_string();
    }
    let Some((before, after)) = line.split_once(", @") else {
        return line.to_string();
    };
    let symbol_end = after
        .find(|c: char| c.is_whitespace() || c == ':' || c == ',')
        .unwrap_or(after.len());
    let symbol = &after[..symbol_end];
    let Some(prefixed) = memory_name_map.get(symbol) else {
        return line.to_string();
    };

    format!("{before}, @{prefixed}{}", &after[symbol_end..])
}

fn rewrite_mem_symbol_after_marker(
    line: &str,
    marker: &str,
    memory_name_map: &HashMap<String, String>,
) -> String {
    let Some(idx) = line.find(marker) else {
        return line.to_string();
    };

    let symbol_start = idx + marker.len();
    let after = &line[symbol_start..];
    let symbol_end = after
        .find(|c: char| c.is_whitespace() || c == ',' || c == ':')
        .unwrap_or(after.len());
    let symbol = &after[..symbol_end];
    let Some(prefixed) = memory_name_map.get(symbol) else {
        return line.to_string();
    };

    format!(
        "{}{}{}",
        &line[..symbol_start],
        prefixed,
        &after[symbol_end..]
    )
}

fn rewrite_mlir_source(
    content: &str,
    processor_name_map: &HashMap<String, String>,
    memory_name_map: &HashMap<String, String>,
) -> String {
    let mut result = String::with_capacity(content.len());
    let mut module_rewritten = false;

    for line in content.lines() {
        let mut rewritten = line.to_string();
        if !module_rewritten {
            if let Some(module_line) = rewrite_module_symbol_line(&rewritten, processor_name_map) {
                rewritten = module_line;
                module_rewritten = true;
            }
        }
        rewritten = rewrite_bind_mem_line(&rewritten, memory_name_map);
        rewritten = rewrite_mem_symbol_after_marker(&rewritten, "src_mem_space @", memory_name_map);
        rewritten = rewrite_mem_symbol_after_marker(&rewritten, "dst_mem_space @", memory_name_map);
        result.push_str(&rewritten);
        result.push('\n');
    }

    if !content.ends_with('\n') {
        result.pop();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::{
        ArchGraph, ComputeProcessor, MemoryBank, MemoryRegion, Processor, Resource, SizeExpr,
    };

    #[test]
    fn single_processor_emits_df_processor() {
        let arch = Processor::new("vec_lane").into_elem();
        let mlir = architecture_to_mlir(&arch).expect("should emit");
        assert!(mlir.contains("adl.processor.compute @proc_vec_lane, []"));
    }

    #[test]
    fn output_wrapped_in_module() {
        let arch = Processor::new("p").into_elem();
        let mlir = architecture_to_mlir(&arch).expect("should emit");
        assert!(mlir.starts_with("module @arch_p {\n"));
        assert!(mlir.ends_with("}\n"));
    }

    #[test]
    fn graph_emits_compose() {
        let l1 = MemoryRegion::bank(MemoryBank::new(SizeExpr::Const(1024))).with_name("L1");
        let lane = Processor::new("lane").into_elem();
        let arch: Architecture = ArchGraph::builder("core")
            .mem(&l1)
            .architecture(&lane)
            .build()
            .into();
        let mlir = architecture_to_mlir(&arch).expect("should emit");
        assert!(mlir.contains("adl.memory.bank"));
        assert!(mlir.contains("adl.processor.compute @proc_lane, []"));
        assert!(!mlir.contains("adl.resource \"L1\""));
        assert!(mlir.contains("adl.arch.compose \"arch_core\", arch["));
        assert!(mlir.contains("], mem["));
    }

    #[test]
    fn memory_resources_not_emitted_as_adl_resource() {
        let l1 = MemoryRegion::bank(MemoryBank::new(SizeExpr::Const(1024))).with_name("L1");
        let lane = Processor::new("lane")
            .with_resources(vec![Resource::exclusive("alu")])
            .into_elem();
        let arch: Architecture = ArchGraph::builder("core")
            .mem(&l1)
            .architecture(&lane)
            .build()
            .into();
        let mlir = architecture_to_mlir(&arch).expect("should emit");
        assert!(mlir.contains("adl.resource.exclusive \"res_alu\""));
        assert!(!mlir.contains("adl.resource.exclusive \"L1\""));
        assert!(!mlir.contains("adl.resource.quantitative \"L1\""));
    }

    #[test]
    fn compute_builder_self_resource_is_emitted_and_referenced() {
        let lane = ComputeProcessor::builder()
            .named("lane")
            .finish()
            .into_elem();
        let arch: Architecture = ArchGraph::builder("core")
            .architecture(&lane)
            .build()
            .into();
        let mlir = architecture_to_mlir(&arch).expect("should emit");
        assert!(mlir.contains("adl.resource.exclusive \"res_lane\""));
        assert!(mlir.contains("adl.processor.compute @proc_lane, [], with ["));
    }

    #[test]
    fn quantitative_resource_is_emitted_with_capacity() {
        let lane = Processor::new("lane")
            .with_resources(vec![Resource::quantitative("l1_port", 2)])
            .into_elem();
        let arch: Architecture = ArchGraph::builder("core")
            .architecture(&lane)
            .build()
            .into();
        let mlir = architecture_to_mlir(&arch).expect("should emit");
        assert!(mlir.contains("adl.resource.quantitative \"res_l1_port\", {capacity = 2}"));
    }

    #[test]
    fn array_emits_scale_with_dims() {
        let dim = Dimension::new_int("x", 8);
        let lane = Processor::new("lane").into_elem();
        let arch = lane.scale(&[dim]).with_name("mesh");
        let mlir = architecture_to_mlir(&arch).expect("should emit");
        assert!(mlir.contains("adl.spatial_dim \"dim_x\", 8"));
        assert!(mlir.contains("adl.arch.scale \"arch_mesh\", ["));
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
            .architecture(&lane)
            .build()
            .into();
        let scaled = core.scale(&[dim_x]).with_name("mesh");
        let mlir = architecture_to_mlir(&scaled).expect("should emit");
        assert!(mlir.contains("adl.memory.array \"mem_L1\", ["));
        assert!(mlir.contains("adl.arch.scale \"arch_mesh\", ["));
        assert_eq!(
            mlir.matches("adl.spatial_dim \"dim_x\"").count(),
            1,
            "dimension x should be emitted exactly once"
        );
    }

    #[test]
    fn mlir_sources_rewrite_processor_module_and_memory_bindings() {
        let mut processor_name_map = HashMap::new();
        processor_name_map.insert("lane".to_string(), "proc_lane".to_string());

        let mut memory_name_map = HashMap::new();
        memory_name_map.insert("L1".to_string(), "mem_L1".to_string());

        let src = r#"module @lane {
  func.func @f(%a: memref<?xf16>) {
    loom.bind_mem %a, @L1 : memref<?xf16>
    loom.copy %a, %a src_mem_space @L1 dst_mem_space @L1, area: [1] : memref<?xf16> to memref<?xf16>
    return
  }
}
"#;
        let rewritten = rewrite_mlir_source(src, &processor_name_map, &memory_name_map);

        assert!(rewritten.contains("module @proc_lane {"));
        assert!(rewritten.contains("loom.bind_mem %a, @mem_L1 : memref<?xf16>"));
        assert!(rewritten.contains("src_mem_space @mem_L1 dst_mem_space @mem_L1"));
    }
}
