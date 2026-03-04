use crate::core::{
    AffineExpr, AffineMap, Architecture, ArchitectureLabel, Dimension, Endpoint, Expr,
    MemoryRegion, MlirModuleRef, ProcessorElem, Resource, ResourceReq, SharingDomain, SizeExpr,
};
use serde::Serialize;
use serde_json::Value;
use std::collections::{HashMap, HashSet};

const GRAPH_SCHEMA_VERSION: &str = "mlar.arch-graph.v1";

/// Top-level JSON payload for web visualization.
#[derive(Debug, Clone, Serialize)]
pub struct ArchitectureGraphJson {
    pub schema_version: &'static str,
    pub architecture: GraphArchitectureMeta,
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GraphArchitectureMeta {
    pub name: String,
    pub labels: Vec<GraphArchitectureLabel>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GraphArchitectureLabel {
    pub name: String,
    pub dimensions: Vec<GraphDimension>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphNodeKind {
    Memory,
    Processor,
}

#[derive(Debug, Clone, Serialize)]
pub struct GraphNode {
    pub id: String,
    pub kind: GraphNodeKind,
    pub name: String,
    pub label: String,
    pub dimensions: Vec<GraphDimension>,
    pub details: GraphNodeDetails,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GraphNodeDetails {
    Memory {
        region: GraphMemoryRegion,
        resource: GraphResource,
    },
    Processor {
        element: GraphProcessorElem,
        total_instances: Option<u64>,
    },
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphEdgeKind {
    Link,
}

#[derive(Debug, Clone, Serialize)]
pub struct GraphEdge {
    pub id: String,
    pub kind: GraphEdgeKind,
    pub name: String,
    pub source: String,
    pub target: String,
    pub source_name: String,
    pub target_name: String,
    pub label: String,
    pub bandwidth: GraphExpr,
    pub latency: Option<GraphExpr>,
    pub constraints: String,
    pub sharing: String,
    pub map: GraphAffineMap,
}

#[derive(Debug, Clone, Serialize)]
pub struct GraphDimension {
    pub name: String,
    pub size_expr: String,
    pub size_const: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GraphExpr {
    pub expr: String,
    pub const_value: Option<i64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GraphSizeExpr {
    pub expr: String,
    pub const_value: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GraphAffineMap {
    pub source_dimensions: Vec<GraphDimension>,
    pub target_dimensions: Vec<GraphDimension>,
    pub expressions: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GraphResource {
    pub name: String,
    pub quantity: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct GraphResourceReq {
    pub resource: GraphResource,
    pub quantity: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct GraphMlirModuleRef {
    pub path: String,
    pub functions: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GraphProcessorElem {
    Unit {
        name: Option<String>,
        compute: Option<GraphMlirModuleRef>,
        resources: Vec<GraphResourceReq>,
    },
    Array {
        name: Option<String>,
        dimensions: Vec<GraphDimension>,
        elem: Box<GraphProcessorElem>,
    },
    Set {
        name: Option<String>,
        parts: Vec<GraphProcessorElem>,
    },
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GraphMemoryRegion {
    Bank {
        name: Option<String>,
        capacity_bytes: GraphSizeExpr,
        access_granularity: Option<GraphSizeExpr>,
    },
    Replicated {
        name: Option<String>,
        dimensions: Vec<GraphDimension>,
        elem: Box<GraphMemoryRegion>,
    },
    Group {
        name: Option<String>,
        parts: Vec<GraphMemoryRegion>,
    },
}

/// Convert an architecture to a JSON-ready graph representation.
pub fn architecture_to_graph_json(arch: &Architecture) -> ArchitectureGraphJson {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut used_ids = HashSet::new();
    let mut memory_node_ids = HashMap::new();
    let mut processor_node_ids = HashMap::new();

    for (idx, region) in arch.memory.iter().enumerate() {
        let name = named_or_fallback(region.name(), "memory", idx);
        let id = unique_id(&format!("mem:{}", slugify(&name)), &mut used_ids);
        memory_node_ids.insert(name.clone(), id.clone());
        nodes.push(memory_node_from_region(id, &name, region));
    }

    for (idx, proc) in arch.processors.iter().enumerate() {
        let name = named_or_fallback(proc.name(), "processor", idx);
        let id = unique_id(&format!("proc:{}", slugify(&name)), &mut used_ids);
        processor_node_ids.insert(name.clone(), id.clone());
        nodes.push(processor_node_from_elem(id, &name, proc));
    }

    for (idx, link) in arch.links.iter().enumerate() {
        let (source, source_name) = ensure_endpoint_node(
            &link.src,
            &mut nodes,
            &mut memory_node_ids,
            &mut processor_node_ids,
            &mut used_ids,
        );
        let (target, target_name) = ensure_endpoint_node(
            &link.dst,
            &mut nodes,
            &mut memory_node_ids,
            &mut processor_node_ids,
            &mut used_ids,
        );

        let edge_id = unique_id(
            &format!("edge:{}:{}", slugify(&link.name), idx),
            &mut used_ids,
        );
        let bandwidth = expr_to_json(&link.bandwidth);
        edges.push(GraphEdge {
            id: edge_id,
            kind: GraphEdgeKind::Link,
            name: link.name.clone(),
            source,
            target,
            source_name,
            target_name,
            label: format!("{} ({} B/cycle)", link.name, bandwidth.expr),
            bandwidth,
            latency: link.latency.as_ref().map(expr_to_json),
            constraints: link.constraints.to_string(),
            sharing: sharing_to_string(&link.sharing).to_string(),
            map: affine_map_to_json(&link.map),
        });
    }

    ArchitectureGraphJson {
        schema_version: GRAPH_SCHEMA_VERSION,
        architecture: GraphArchitectureMeta {
            name: arch.name.clone(),
            labels: arch.labels.iter().map(architecture_label_to_json).collect(),
        },
        nodes,
        edges,
    }
}

/// Convert an architecture graph to `serde_json::Value`.
pub fn architecture_to_graph_json_value(arch: &Architecture) -> Value {
    serde_json::to_value(architecture_to_graph_json(arch))
        .expect("architecture graph serialization must succeed")
}

/// Convert an architecture graph to compact JSON string.
pub fn architecture_to_graph_json_string(arch: &Architecture) -> Result<String, serde_json::Error> {
    serde_json::to_string(&architecture_to_graph_json(arch))
}

/// Convert an architecture graph to pretty-printed JSON string.
pub fn architecture_to_graph_json_string_pretty(
    arch: &Architecture,
) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(&architecture_to_graph_json(arch))
}

fn architecture_label_to_json(label: &ArchitectureLabel) -> GraphArchitectureLabel {
    GraphArchitectureLabel {
        name: label.name.clone(),
        dimensions: label.dims.iter().map(dimension_to_json).collect(),
    }
}

fn ensure_endpoint_node(
    endpoint: &Endpoint,
    nodes: &mut Vec<GraphNode>,
    memory_node_ids: &mut HashMap<String, String>,
    processor_node_ids: &mut HashMap<String, String>,
    used_ids: &mut HashSet<String>,
) -> (String, String) {
    match endpoint {
        Endpoint::Mem(region) => {
            let name = region
                .name()
                .map(|n| n.to_string())
                .unwrap_or_else(|| "unnamed_memory".to_string());

            if let Some(id) = memory_node_ids.get(&name) {
                return (id.clone(), name);
            }

            let id = unique_id(&format!("mem:{}", slugify(&name)), used_ids);
            nodes.push(memory_node_from_region(id.clone(), &name, region));
            memory_node_ids.insert(name.clone(), id.clone());
            (id, name)
        }
        Endpoint::Proc(proc) => {
            let name = proc
                .name()
                .map(|n| n.to_string())
                .unwrap_or_else(|| "unnamed_processor".to_string());

            if let Some(id) = processor_node_ids.get(&name) {
                return (id.clone(), name);
            }

            let id = unique_id(&format!("proc:{}", slugify(&name)), used_ids);
            nodes.push(processor_node_from_elem(id.clone(), &name, proc));
            processor_node_ids.insert(name.clone(), id.clone());
            (id, name)
        }
    }
}

fn memory_node_from_region(id: String, name: &str, region: &MemoryRegion) -> GraphNode {
    let dimensions = dedup_dimensions(collect_memory_dims(region))
        .iter()
        .map(dimension_to_json)
        .collect::<Vec<_>>();
    GraphNode {
        id,
        kind: GraphNodeKind::Memory,
        name: name.to_string(),
        label: node_label(name, &dimensions),
        dimensions,
        details: GraphNodeDetails::Memory {
            region: memory_region_to_json(region),
            resource: resource_to_json(&region.as_resource()),
        },
    }
}

fn processor_node_from_elem(id: String, name: &str, elem: &ProcessorElem) -> GraphNode {
    let dimensions = dedup_dimensions(collect_processor_dims(elem))
        .iter()
        .map(dimension_to_json)
        .collect::<Vec<_>>();
    GraphNode {
        id,
        kind: GraphNodeKind::Processor,
        name: name.to_string(),
        label: node_label(name, &dimensions),
        dimensions,
        details: GraphNodeDetails::Processor {
            element: processor_elem_to_json(elem),
            total_instances: elem.total_instances(),
        },
    }
}

fn node_label(name: &str, dims: &[GraphDimension]) -> String {
    if dims.is_empty() {
        return name.to_string();
    }

    let suffix = dims
        .iter()
        .map(|d| format!("{}={}", d.name, d.size_expr))
        .collect::<Vec<_>>()
        .join(", ");
    format!("{name} [{suffix}]")
}

fn named_or_fallback(name: Option<&str>, kind: &str, index: usize) -> String {
    name.map(|n| n.to_string())
        .unwrap_or_else(|| format!("{kind}_{index}"))
}

fn unique_id(base: &str, used: &mut HashSet<String>) -> String {
    if !used.contains(base) {
        used.insert(base.to_string());
        return base.to_string();
    }

    let mut counter = 1usize;
    loop {
        let candidate = format!("{base}__{counter}");
        if !used.contains(&candidate) {
            used.insert(candidate.clone());
            return candidate;
        }
        counter += 1;
    }
}

fn slugify(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            out.push(ch.to_ascii_lowercase());
        } else if ch.is_ascii_whitespace() || ch == ':' || ch == '/' || ch == '.' {
            out.push('_');
        }
    }

    if out.is_empty() {
        "unnamed".to_string()
    } else {
        out
    }
}

fn sharing_to_string(sharing: &SharingDomain) -> &'static str {
    match sharing {
        SharingDomain::SharedAcrossAll => "shared_across_all",
    }
}

fn expr_to_json(expr: &Expr) -> GraphExpr {
    GraphExpr {
        expr: expr.to_string(),
        const_value: expr.eval_const(),
    }
}

fn size_expr_to_json(expr: &SizeExpr) -> GraphSizeExpr {
    GraphSizeExpr {
        expr: expr.to_string(),
        const_value: expr.as_const(),
    }
}

fn resource_to_json(resource: &Resource) -> GraphResource {
    GraphResource {
        name: resource.name.clone(),
        quantity: resource.quantity,
    }
}

fn resource_req_to_json(req: &ResourceReq) -> GraphResourceReq {
    GraphResourceReq {
        resource: resource_to_json(&req.resource),
        quantity: req.quantity,
    }
}

fn mlir_module_to_json(module: &MlirModuleRef) -> GraphMlirModuleRef {
    GraphMlirModuleRef {
        path: module.path.clone(),
        functions: module.functions.clone(),
    }
}

fn dimension_to_json(dim: &Dimension) -> GraphDimension {
    GraphDimension {
        name: dim.name.0.clone(),
        size_expr: dim.size.to_string(),
        size_const: dim.size.as_const(),
    }
}

fn affine_map_to_json(map: &AffineMap) -> GraphAffineMap {
    GraphAffineMap {
        source_dimensions: map.src_dims.iter().map(dimension_to_json).collect(),
        target_dimensions: map.dst_dims.iter().map(dimension_to_json).collect(),
        expressions: map.exprs.iter().map(format_affine_expr).collect(),
    }
}

fn format_affine_expr(expr: &AffineExpr) -> String {
    match expr {
        AffineExpr::Var(dim) => dim.name.0.clone(),
        AffineExpr::Sym(sym) => sym.0.clone(),
        AffineExpr::Const(v) => v.to_string(),
        AffineExpr::Add(a, b) => format!("({} + {})", format_affine_expr(a), format_affine_expr(b)),
        AffineExpr::MulConst(c, e) => format!("({} * {})", c, format_affine_expr(e)),
        AffineExpr::Mod(a, b) => {
            format!("({} mod {})", format_affine_expr(a), format_affine_expr(b))
        }
        AffineExpr::CeilDiv(a, b) => {
            format!(
                "({} ceildiv {})",
                format_affine_expr(a),
                format_affine_expr(b)
            )
        }
    }
}

fn collect_memory_dims(region: &MemoryRegion) -> Vec<Dimension> {
    match region {
        MemoryRegion::Bank(_) => Vec::new(),
        MemoryRegion::Replicated { dims, elem, .. } => {
            let mut out = dims.clone();
            out.extend(collect_memory_dims(elem));
            out
        }
        MemoryRegion::Group { parts, .. } => {
            let mut out = Vec::new();
            for part in parts {
                out.extend(collect_memory_dims(part));
            }
            out
        }
    }
}

fn collect_processor_dims(elem: &ProcessorElem) -> Vec<Dimension> {
    match elem {
        ProcessorElem::Unit(_) => Vec::new(),
        ProcessorElem::Array { dims, elem, .. } => {
            let mut out = dims.clone();
            out.extend(collect_processor_dims(elem));
            out
        }
        ProcessorElem::Set { parts, .. } => {
            let mut out = Vec::new();
            for part in parts {
                out.extend(collect_processor_dims(part));
            }
            out
        }
    }
}

fn dedup_dimensions(dims: Vec<Dimension>) -> Vec<Dimension> {
    let mut seen = HashSet::new();
    let mut deduped = Vec::new();
    for dim in dims {
        if seen.insert(dim.name.0.clone()) {
            deduped.push(dim);
        }
    }
    deduped
}

fn memory_region_to_json(region: &MemoryRegion) -> GraphMemoryRegion {
    match region {
        MemoryRegion::Bank(bank) => GraphMemoryRegion::Bank {
            name: bank.name.clone(),
            capacity_bytes: size_expr_to_json(&bank.capacity_bytes),
            access_granularity: bank.access_granularity.as_ref().map(size_expr_to_json),
        },
        MemoryRegion::Replicated { name, dims, elem } => GraphMemoryRegion::Replicated {
            name: name.clone(),
            dimensions: dims.iter().map(dimension_to_json).collect(),
            elem: Box::new(memory_region_to_json(elem)),
        },
        MemoryRegion::Group { name, parts } => GraphMemoryRegion::Group {
            name: name.clone(),
            parts: parts.iter().map(memory_region_to_json).collect(),
        },
    }
}

fn processor_elem_to_json(elem: &ProcessorElem) -> GraphProcessorElem {
    match elem {
        ProcessorElem::Unit(proc) => GraphProcessorElem::Unit {
            name: proc.name.clone(),
            compute: proc.compute().map(mlir_module_to_json),
            resources: proc.resources.iter().map(resource_req_to_json).collect(),
        },
        ProcessorElem::Array { name, dims, elem } => GraphProcessorElem::Array {
            name: name.clone(),
            dimensions: dims.iter().map(dimension_to_json).collect(),
            elem: Box::new(processor_elem_to_json(elem)),
        },
        ProcessorElem::Set { name, parts } => GraphProcessorElem::Set {
            name: name.clone(),
            parts: parts.iter().map(processor_elem_to_json).collect(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::architecture_to_graph_json_value;
    use crate::core::{
        AffineMap, Architecture, Dimension, Link, MemoryBank, MemoryRegion, Processor, SizeExpr,
    };

    #[test]
    fn serializes_architecture_graph_schema() {
        let core_dim = Dimension::new_int("core", 4);
        let l1 = MemoryRegion::bank(MemoryBank::from_blocks(
            SizeExpr::Const(64),
            SizeExpr::Const(512),
        ))
        .replicate(core_dim.as_slice())
        .with_name("l1");
        let lane = Processor::new("lane").replicate(core_dim.as_slice());
        let map = AffineMap::identity(core_dim.as_slice());

        let link = Link::builder("l1_to_lane")
            .from_mem(&l1)
            .to_proc(&lane)
            .map(&map)
            .bandwidth(128)
            .build();

        let arch = Architecture::builder("unit")
            .mem(&l1)
            .processor(&lane)
            .link(link)
            .build();

        let value = architecture_to_graph_json_value(&arch);
        assert_eq!(value["schema_version"], "mlar.arch-graph.v1");
        assert_eq!(value["architecture"]["name"], "unit");
        assert_eq!(value["nodes"].as_array().map(|v| v.len()), Some(2));
        assert_eq!(value["edges"].as_array().map(|v| v.len()), Some(1));
        assert_eq!(value["edges"][0]["map"]["expressions"][0], "core");
        assert_eq!(value["edges"][0]["bandwidth"]["const_value"], 128);
    }
}
