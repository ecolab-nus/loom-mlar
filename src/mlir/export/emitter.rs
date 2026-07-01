use std::collections::{HashMap, HashSet};
use std::fmt::Write;

use crate::arch::architecture::Architecture;
use crate::arch::memory::{MemoryBank, MemoryRegion};
use crate::arch::processor::{DataEffect, Processor};
use crate::arch::resource::Resource;
use crate::arch::size_dim::Dimension;

use super::names::{
    memory_region_key, memory_region_resource_id, prefixed_arch_name, prefixed_dim_name,
    prefixed_memory_name, prefixed_processor_name, prefixed_resource_name,
};

/// SSA-based emitter that serialises an [`Architecture`] tree into the
/// `adl.*` MLIR dialect.
pub(super) struct MlirEmitter {
    counter: usize,
    pub(super) output: String,
    /// Dimension name to SSA value, avoiding duplicate dim ops.
    dim_map: HashMap<String, String>,
    /// Memory-region name to SSA value.
    memory_map: HashMap<String, String>,
    /// Structural memory-region key to SSA value.
    memory_region_map: HashMap<String, String>,
    /// Resource id to SSA value.
    resource_map: HashMap<String, String>,
    /// Memory name to prefixed MLIR memory name used in exported ops.
    pub(super) memory_name_map: HashMap<String, String>,
    /// Processor name to prefixed MLIR processor symbol.
    pub(super) processor_name_map: HashMap<String, String>,
    /// MLIR source paths collected from functionality modules.
    pub(super) mlir_sources: Vec<String>,
}

impl MlirEmitter {
    pub(super) fn new() -> Self {
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

    /// Emit a typed `adl.resource.*` op if not already emitted.
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

    /// Emit a processor op and record its MLIR source, if any.
    fn emit_processor(&mut self, proc: &Processor) -> Option<String> {
        let kind = match proc.effect {
            DataEffect::Preserve => "dmover",
            DataEffect::Transform | DataEffect::Reduce | DataEffect::Accumulate => "compute",
        };
        self.emit_processor_like(kind, proc)
    }

    fn emit_processor_like(&mut self, kind: &str, proc: &Processor) -> Option<String> {
        let ssa = self.next_ssa();
        let name = proc.name.as_deref().unwrap_or("unnamed");
        let prefixed_name = prefixed_processor_name(name);
        self.processor_name_map
            .insert(name.to_string(), prefixed_name.clone());
        let region_pairs = self.format_region_pairs(proc)?;
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

    fn format_region_pairs(&self, proc: &Processor) -> Option<String> {
        let (Some(source), Some(destination)) = (&proc.source, &proc.destination) else {
            return Some("[]".to_string());
        };
        let src_ssa = self.memory_map.get(source.name.as_str())?;
        let dst_ssa = self.memory_map.get(destination.name.as_str())?;
        Some(format!("[({}, {})]", src_ssa, dst_ssa))
    }

    /// Recursively emit the full architecture tree, returning the SSA value
    /// for the top-level result.
    pub(super) fn emit_architecture(&mut self, arch: &Architecture) -> Option<String> {
        let elem_ssa = self.emit_architecture_scope(arch)?;
        if arch.dims.is_empty() {
            return Some(elem_ssa);
        }
        let dim_ssas: Vec<String> = arch
            .dims
            .iter()
            .map(|d| self.emit_dim(d))
            .collect::<Option<Vec<_>>>()?;
        let mem_region_ssas = self.emit_scaled_array_mem_regions(&arch.dims, arch)?;
        let ssa = self.next_ssa();
        let dim_list = dim_ssas.join(", ");
        let mem_region_clause = format_scale_mem_region_clause(&mem_region_ssas);
        writeln!(
            self.output,
            "{} = adl.arch.scale \"{}\", [{}] of {}{}",
            ssa,
            prefixed_arch_name(&arch.name),
            dim_list,
            elem_ssa,
            mem_region_clause
        )
        .unwrap();
        Some(ssa)
    }

    fn emit_architecture_scope(&mut self, arch: &Architecture) -> Option<String> {
        let mut arch_components: Vec<String> = Vec::new();
        let mut mem_components: Vec<String> = Vec::new();
        let mut memory_resource_ids: HashSet<String> = HashSet::new();

        collect_arch_memory_resource_ids(arch, &mut memory_resource_ids);

        for memory in &arch.memories {
            mem_components.push(self.emit_memory(memory)?);
        }

        for child in &arch.children {
            arch_components.push(self.emit_architecture(child)?);
        }

        for resource in &arch.resources {
            if memory_resource_ids.contains(resource.id().as_str()) {
                continue;
            }
            self.emit_resource(resource);
        }

        for processor in &arch.processors {
            arch_components.push(self.emit_processor(processor)?);
        }

        let ssa = self.next_ssa();
        let arch_values = arch_components.join(", ");
        let mem_values = mem_components.join(", ");
        writeln!(
            self.output,
            "{} = adl.arch.compose \"{}\", arch[{}], mem[{}]",
            ssa,
            prefixed_arch_name(&arch.name),
            arch_values,
            mem_values
        )
        .unwrap();
        Some(ssa)
    }

    fn emit_scaled_array_mem_regions(
        &mut self,
        dims: &[Dimension],
        elem: &Architecture,
    ) -> Option<Vec<String>> {
        let regions = collect_array_scaled_memory_regions(dims, elem);
        regions
            .iter()
            .map(|region| self.emit_memory(region))
            .collect()
    }
}

fn format_scale_mem_region_clause(mem_region_ssas: &[String]) -> String {
    match mem_region_ssas {
        [] => String::new(),
        [single] => format!(", mem_region {single}"),
        many => format!(", mem_region [{}]", many.join(", ")),
    }
}

fn collect_array_scaled_memory_regions(
    dims: &[Dimension],
    elem: &Architecture,
) -> Vec<MemoryRegion> {
    collect_arch_memory_regions_with_base_names(elem)
        .into_iter()
        .map(|(base_name, region)| region.scale(dims).with_name(format!("array_{base_name}")))
        .collect()
}

fn collect_arch_memory_regions_with_base_names(arch: &Architecture) -> Vec<(String, MemoryRegion)> {
    let mut regions = Vec::new();
    for region in &arch.memories {
        if let Some(name) = region.name() {
            regions.push((name.to_string(), region.clone()));
        }
    }
    for child in &arch.children {
        regions.extend(collect_arch_memory_regions_with_base_names(child));
    }
    regions
}

/// Recursively collect memory resource IDs referenced by an architecture tree.
fn collect_arch_memory_resource_ids(arch: &Architecture, out: &mut HashSet<String>) {
    for memory in &arch.memories {
        if let Some(id) = memory_region_resource_id(memory) {
            out.insert(id);
        }
    }
    for processor in &arch.processors {
        if let Some(source) = &processor.source {
            out.insert(source.name.clone());
        }
        if let Some(destination) = &processor.destination {
            out.insert(destination.name.clone());
        }
    }
    for child in &arch.children {
        collect_arch_memory_resource_ids(child, out);
    }
}
