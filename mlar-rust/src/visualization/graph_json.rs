use crate::arch::{
    Architecture, ArchitectureLabel, Dimension, Endpoint, LinkMapRelation, LinkTopology,
    MemoryRegion, MlirModuleRef, Processors, Resource, ResourceReq, SharingDomain, SizeExpr,
};
use crate::math::{AffineExpr, AffineMap, Expr};
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub intra_core: Option<Box<ArchitectureGraphJson>>,
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
        element: GraphProcessors,
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
    pub map_relation: GraphMapRelation,
    pub topology: GraphLinkTopology,
    pub map: GraphAffineMap,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphMapRelation {
    OneToOne,
    OneToMany,
    ManyToOne,
    ManyToMany,
    Unknown,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphLinkTopology {
    Ring,
    General,
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
    pub module_name: Option<String>,
    pub functions: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GraphProcessors {
    Unit {
        name: Option<String>,
        compute: Option<GraphMlirModuleRef>,
        resources: Vec<GraphResourceReq>,
    },
    Array {
        name: Option<String>,
        dimensions: Vec<GraphDimension>,
        elem: Box<GraphProcessors>,
    },
    Set {
        name: Option<String>,
        parts: Vec<GraphProcessors>,
    },
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GraphMemoryRegion {
    Bank {
        name: Option<String>,
        capacity_bytes: GraphSizeExpr,
        access_granularity: Option<GraphSizeExpr>,
        total_size_bytes: Option<u64>,
    },
    Replicated {
        name: Option<String>,
        dimensions: Vec<GraphDimension>,
        elem: Box<GraphMemoryRegion>,
        total_size_bytes: Option<u64>,
    },
    Group {
        name: Option<String>,
        parts: Vec<GraphMemoryRegion>,
        total_size_bytes: Option<u64>,
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
            map_relation: link_map_relation_to_json(link.map_relation()),
            topology: link_topology_to_json(link.topology()),
            map: affine_map_to_json(&link.map),
        });
    }

    // Build intra-core sub-graph when this architecture was produced by scale().
    let intra_core = if !arch.labels.is_empty() {
        Some(Box::new(extract_intra_core_graph(arch)))
    } else {
        None
    };

    ArchitectureGraphJson {
        schema_version: GRAPH_SCHEMA_VERSION,
        architecture: GraphArchitectureMeta {
            name: arch.name.clone(),
            labels: arch.labels.iter().map(architecture_label_to_json).collect(),
        },
        nodes,
        edges,
        intra_core,
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

fn processor_node_from_elem(id: String, name: &str, elem: &Processors) -> GraphNode {
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
            element: processors_to_json(elem),
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

fn link_map_relation_to_json(relation: LinkMapRelation) -> GraphMapRelation {
    match relation {
        LinkMapRelation::OneToOne => GraphMapRelation::OneToOne,
        LinkMapRelation::OneToMany => GraphMapRelation::OneToMany,
        LinkMapRelation::ManyToOne => GraphMapRelation::ManyToOne,
        LinkMapRelation::ManyToMany => GraphMapRelation::ManyToMany,
        LinkMapRelation::Unknown => GraphMapRelation::Unknown,
    }
}

fn link_topology_to_json(topology: LinkTopology) -> GraphLinkTopology {
    match topology {
        LinkTopology::Ring => GraphLinkTopology::Ring,
        LinkTopology::General => GraphLinkTopology::General,
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
        module_name: module.module_name.clone(),
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

fn collect_processor_dims(elem: &Processors) -> Vec<Dimension> {
    match elem {
        Processors::Unit(_) => Vec::new(),
        Processors::Array { dims, elem, .. } => {
            let mut out = dims.clone();
            out.extend(collect_processor_dims(elem));
            out
        }
        Processors::Set { parts, .. } => {
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
    let total_size_bytes = region.total_size_bytes();
    match region {
        MemoryRegion::Bank(bank) => GraphMemoryRegion::Bank {
            name: bank.name.clone(),
            capacity_bytes: size_expr_to_json(&bank.capacity_bytes),
            access_granularity: bank.access_granularity.as_ref().map(size_expr_to_json),
            total_size_bytes,
        },
        MemoryRegion::Replicated { name, dims, elem } => GraphMemoryRegion::Replicated {
            name: name.clone(),
            dimensions: dims.iter().map(dimension_to_json).collect(),
            elem: Box::new(memory_region_to_json(elem)),
            total_size_bytes,
        },
        MemoryRegion::Group { name, parts } => GraphMemoryRegion::Group {
            name: name.clone(),
            parts: parts.iter().map(memory_region_to_json).collect(),
            total_size_bytes,
        },
    }
}

fn processors_to_json(elem: &Processors) -> GraphProcessors {
    match elem {
        Processors::Unit(proc) => GraphProcessors::Unit {
            name: proc.name.clone(),
            compute: proc.compute().map(mlir_module_to_json),
            resources: proc.resources.iter().map(resource_req_to_json).collect(),
        },
        Processors::Array { name, dims, elem } => GraphProcessors::Array {
            name: name.clone(),
            dimensions: dims.iter().map(dimension_to_json).collect(),
            elem: Box::new(processors_to_json(elem)),
        },
        Processors::Set { name, parts } => GraphProcessors::Set {
            name: name.clone(),
            parts: parts.iter().map(processors_to_json).collect(),
        },
    }
}

/// Extract a single-instance "intra-core" sub-graph from a scaled architecture.
///
/// Strips the outermost scaling dimensions from each node and keeps only
/// intra-core links (those whose affine map is an identity on the scaling
/// dimensions, meaning src and dst share the same core coordinates).
fn extract_intra_core_graph(arch: &Architecture) -> ArchitectureGraphJson {
    // Collect all scaling dimension names from labels.
    let scaling_dim_names: HashSet<String> = arch
        .labels
        .iter()
        .flat_map(|label| label.dims.iter().map(|d| d.name.0.clone()))
        .collect();

    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut used_ids = HashSet::new();
    let mut memory_node_ids = HashMap::new();
    let mut processor_node_ids = HashMap::new();

    // Emit inner (unwrapped) memory nodes.
    for (idx, region) in arch.memory.iter().enumerate() {
        let name = named_or_fallback(region.name(), "memory", idx);
        let inner = unwrap_memory_scaling(region, &scaling_dim_names);
        let id = unique_id(&format!("mem:{}", slugify(&name)), &mut used_ids);
        memory_node_ids.insert(name.clone(), id.clone());
        nodes.push(memory_node_from_region(id, &name, &inner));
    }

    // Emit inner (unwrapped) processor nodes.
    for (idx, proc) in arch.processors.iter().enumerate() {
        let name = named_or_fallback(proc.name(), "processor", idx);
        let inner = unwrap_processor_scaling(proc, &scaling_dim_names);
        let id = unique_id(&format!("proc:{}", slugify(&name)), &mut used_ids);
        processor_node_ids.insert(name.clone(), id.clone());
        nodes.push(processor_node_from_elem(id, &name, &inner));
    }

    // Keep only intra-core edges (identity maps on scaling dims).
    for (idx, link) in arch.links.iter().enumerate() {
        if !is_identity_on_scaling_dims(&link.map, &scaling_dim_names) {
            continue;
        }

        let source_name = link.src.name().to_string();
        let target_name = link.dst.name().to_string();
        let source = memory_node_ids
            .get(&source_name)
            .or_else(|| processor_node_ids.get(&source_name))
            .cloned()
            .unwrap_or_else(|| format!("unknown:{}", source_name));
        let target = memory_node_ids
            .get(&target_name)
            .or_else(|| processor_node_ids.get(&target_name))
            .cloned()
            .unwrap_or_else(|| format!("unknown:{}", target_name));

        // Build the inner edge with the original (pre-scale) affine map.
        let inner_map = strip_scaling_from_map(&link.map, &scaling_dim_names);
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
            map_relation: link_map_relation_to_json(link.map_relation()),
            topology: link_topology_to_json(link.topology()),
            map: inner_map,
        });
    }

    // Determine the innermost label name for the core architecture name.
    let core_name = arch
        .labels
        .last()
        .map(|l| l.name.clone())
        .unwrap_or_else(|| "core".to_string());

    ArchitectureGraphJson {
        schema_version: GRAPH_SCHEMA_VERSION,
        architecture: GraphArchitectureMeta {
            name: core_name,
            labels: Vec::new(),
        },
        nodes,
        edges,
        intra_core: None,
    }
}

/// Unwrap outer `Replicated` layers only when they are pure scaling dimensions.
fn unwrap_memory_scaling(region: &MemoryRegion, scaling_dims: &HashSet<String>) -> MemoryRegion {
    let mut current = region;
    loop {
        match current {
            MemoryRegion::Replicated { dims, elem, .. }
                if !dims.is_empty() && dims.iter().all(|d| scaling_dims.contains(&d.name.0)) =>
            {
                current = elem;
            }
            other => return other.clone(),
        }
    }
}

/// Unwrap outer `Array` layers only when they are pure scaling dimensions.
fn unwrap_processor_scaling(elem: &Processors, scaling_dims: &HashSet<String>) -> Processors {
    let mut current = elem;
    loop {
        match current {
            Processors::Array { dims, elem, .. }
                if !dims.is_empty() && dims.iter().all(|d| scaling_dims.contains(&d.name.0)) =>
            {
                current = elem;
            }
            other => return other.clone(),
        }
    }
}

/// Check whether an affine map is an identity on the scaling dimensions.
///
/// For an intra-core link, the map produced by `prepend_identity_dims` is
/// simply the identity `[x, y] -> [x, y]: (x, y)`.  Inter-core links have
/// non-trivial expressions like `(x, (y+1) mod 8)`.
fn is_identity_on_scaling_dims(map: &AffineMap, scaling_dims: &HashSet<String>) -> bool {
    // Check each expression corresponding to a scaling dimension.
    for (dim, expr) in map.src_dims.iter().zip(map.exprs.iter()) {
        if !scaling_dims.contains(&dim.name.0) {
            continue;
        }
        // An identity expression for dim "x" is AffineExpr::Var(x).
        match expr {
            AffineExpr::Var(v) if v.name.0 == dim.name.0 => {}
            _ => return false,
        }
    }
    true
}

/// Strip scaling dimensions from an affine map, returning only the
/// inner (intra-core) portion.
fn strip_scaling_from_map(map: &AffineMap, scaling_dims: &HashSet<String>) -> GraphAffineMap {
    let inner_src: Vec<_> = map
        .src_dims
        .iter()
        .filter(|d| !scaling_dims.contains(&d.name.0))
        .collect();
    let inner_dst: Vec<_> = map
        .dst_dims
        .iter()
        .filter(|d| !scaling_dims.contains(&d.name.0))
        .collect();
    let inner_exprs: Vec<_> = map
        .src_dims
        .iter()
        .zip(map.exprs.iter())
        .filter(|(d, _)| !scaling_dims.contains(&d.name.0))
        .map(|(_, e)| format_affine_expr(e))
        .collect();

    GraphAffineMap {
        source_dimensions: inner_src.iter().map(|d| dimension_to_json(d)).collect(),
        target_dimensions: inner_dst.iter().map(|d| dimension_to_json(d)).collect(),
        expressions: inner_exprs,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        GraphMemoryRegion, GraphNodeDetails, architecture_to_graph_json,
        architecture_to_graph_json_value,
    };
    use crate::arch::{
        Architecture, Dimension, Link, MemoryBank, MemoryRegion, Processor, SizeExpr,
    };
    use crate::math::AffineMap;

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
        // No labels → no intra_core
        assert!(value.get("intra_core").is_none());
    }

    #[test]
    fn scaled_architecture_has_intra_core_graph() {
        let bank_dim = Dimension::new_int("nbank", 16);
        let l1 = MemoryRegion::bank(MemoryBank::from_blocks(
            SizeExpr::Const(64),
            SizeExpr::Const(512),
        ))
        .replicate(bank_dim.as_slice())
        .with_name("l1");
        let lane = Processor::new("lane").into_elem();
        let inner_map = AffineMap::new(&[], &[], vec![]);

        let link = Link::builder("l1_to_lane")
            .from_mem(&l1)
            .to_proc(&lane)
            .map(&inner_map)
            .bandwidth(128)
            .build();

        let core = Architecture::builder("core")
            .mem(&l1)
            .processor(&lane)
            .link(link)
            .build();

        let dim_x = Dimension::new_int("x", 4);
        let dim_y = Dimension::new_int("y", 4);
        let mesh = core.scale([&dim_x, &dim_y]).with_name("mesh");

        let graph = architecture_to_graph_json(&mesh);

        // Top-level should have intra_core.
        assert!(graph.intra_core.is_some());
        let intra = graph.intra_core.as_ref().unwrap();

        // Intra-core sub-graph has the core name.
        assert_eq!(intra.architecture.name, "core");
        assert!(intra.architecture.labels.is_empty());

        // Should have the unwrapped nodes (no Array/Replicated wrappers).
        assert_eq!(intra.nodes.len(), 2);
        assert_eq!(intra.edges.len(), 1);

        // The inner nodes should NOT have scaling dimensions.
        for node in &intra.nodes {
            assert!(
                node.dimensions
                    .iter()
                    .all(|d| d.name != "x" && d.name != "y"),
                "intra-core node should not have scaling dimensions"
            );
        }

        // Non-scaling memory hierarchy (nbank replication) should be preserved.
        let l1_node = intra
            .nodes
            .iter()
            .find(|n| n.name == "l1")
            .expect("l1 node should exist");
        assert!(l1_node.dimensions.iter().any(|d| d.name == "nbank"));
        match &l1_node.details {
            GraphNodeDetails::Memory { region, .. } => match region {
                GraphMemoryRegion::Replicated { dimensions, .. } => {
                    assert!(dimensions.iter().any(|d| d.name == "nbank"));
                }
                _ => panic!("l1 intra-core region should remain replicated by nbank"),
            },
            _ => panic!("l1 node should be memory"),
        }

        // Intra-core does not recurse.
        assert!(intra.intra_core.is_none());
    }
}
