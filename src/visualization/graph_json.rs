use crate::arch::{
    ArchGraph, ArchNode, ArchNodeComponent, Architecture, Dimension, LinkMapRelation, LinkTopology,
    MemoryRegion, Processors, Router, SizeExpr,
};
use crate::math::{AffineExpr, AffineMap, Expr};
use crate::schedule::Module;
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
    Router,
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
    },
    Processor {
        element: GraphProcessors,
        total_instances: Option<u64>,
    },
    Router {
        router: GraphRouter,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct GraphRouter {
    pub name: String,
    pub side_count: usize,
    pub endpoints: usize,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphEdgeKind {
    ScaleOutNetwork,
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
pub struct GraphFunctionalityModule {
    pub name: Option<String>,
    pub source_path: Option<String>,
    pub source_mlir_module_name: Option<String>,
    pub ops: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GraphProcessors {
    Unit {
        name: Option<String>,
        functionality: GraphFunctionalityModule,
    },
    Array {
        name: Option<String>,
        dimensions: Vec<GraphDimension>,
        elem: Box<GraphProcessors>,
    },
    Graph {
        name: String,
        processor_count: usize,
        link_count: usize,
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
    let synthetic_graph;
    let graph = if let Some(graph) = arch.as_graph() {
        graph
    } else {
        synthetic_graph = ArchGraph {
            name: arch.name().unwrap_or("architecture").to_string(),
            nodes: vec![ArchNode::from_architecture(arch)],
            edges: Vec::new(),
        };
        &synthetic_graph
    };

    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut used_ids = HashSet::new();
    let mut memory_node_ids = HashMap::new();
    let mut processor_node_ids = HashMap::new();
    let mut router_node_ids: HashMap<String, String> = HashMap::new();

    for (idx, node) in graph.nodes.iter().enumerate() {
        match &node.component {
            ArchNodeComponent::MemoryRegion(region) => {
                let name = named_or_fallback(region.name(), "memory", idx);
                if memory_node_ids.contains_key(&name) {
                    continue;
                }
                let id = unique_id(&format!("mem:{}", slugify(&name)), &mut used_ids);
                memory_node_ids.insert(name.clone(), id.clone());
                nodes.push(memory_node_from_region(id, &name, region));
            }
            ArchNodeComponent::Architecture(proc) => {
                let name = named_or_fallback(proc.name(), "processor", idx);
                if processor_node_ids.contains_key(&name) {
                    continue;
                }
                let id = unique_id(&format!("proc:{}", slugify(&name)), &mut used_ids);
                processor_node_ids.insert(name.clone(), id.clone());
                nodes.push(processor_node_from_elem(id, &name, proc));
            }
            ArchNodeComponent::Router(router) => {
                let name = if node.name.is_empty() {
                    format!("router_{idx}")
                } else {
                    node.name.clone()
                };
                if router_node_ids.contains_key(&name) {
                    continue;
                }
                let id = unique_id(&format!("router:{}", slugify(&name)), &mut used_ids);
                router_node_ids.insert(name.clone(), id.clone());
                nodes.push(router_node(id, &name, router));
            }
        }
    }

    let mut links = Vec::new();
    collect_connectivity_links(arch, &mut links);
    for (idx, link) in links.iter().enumerate() {
        let (source, source_name) =
            ensure_endpoint_node(link.src(), &mut nodes, &mut memory_node_ids, &mut used_ids);
        let (target, target_name) =
            ensure_endpoint_node(link.dst(), &mut nodes, &mut memory_node_ids, &mut used_ids);

        let link_name = link.name();
        let edge_id = unique_id(
            &format!("edge:{}:{}", slugify(link_name), idx),
            &mut used_ids,
        );
        let bandwidth = expr_to_json(link.bandwidth());
        edges.push(GraphEdge {
            id: edge_id,
            kind: GraphEdgeKind::ScaleOutNetwork,
            name: link_name.to_owned(),
            source,
            target,
            source_name,
            target_name,
            label: format!("{} ({} B/cycle)", link_name, bandwidth.expr),
            bandwidth,
            latency: link.latency().map(expr_to_json),
            constraints: String::new(),
            sharing: network_kind_label(link).to_owned(),
            map_relation: link_map_relation_to_json(link.map_relation()),
            topology: link_topology_to_json(link.topology()),
            map: affine_map_to_json(link.map()),
        });
    }

    ArchitectureGraphJson {
        schema_version: GRAPH_SCHEMA_VERSION,
        architecture: GraphArchitectureMeta {
            name: graph.name.clone(),
            labels: Vec::new(),
        },
        nodes,
        edges,
        intra_core: None,
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

fn collect_connectivity_links<'a>(
    arch: &'a Architecture,
    out: &mut Vec<&'a crate::arch::ScaleOutNetwork>,
) {
    match arch {
        Architecture::Unit(_) => {}
        Architecture::Array {
            connectivity, elem, ..
        } => {
            out.extend(connectivity.iter());
            collect_connectivity_links(elem, out);
        }
        Architecture::Graph(graph) => {
            for node in &graph.nodes {
                if let ArchNodeComponent::Architecture(node_arch) = &node.component {
                    collect_connectivity_links(node_arch, out);
                }
            }
        }
    }
}

fn ensure_endpoint_node(
    endpoint: &MemoryRegion,
    nodes: &mut Vec<GraphNode>,
    memory_node_ids: &mut HashMap<String, String>,
    used_ids: &mut HashSet<String>,
) -> (String, String) {
    let name = endpoint
        .name()
        .map(|n| n.to_string())
        .unwrap_or_else(|| "unnamed_memory".to_string());

    if let Some(id) = memory_node_ids.get(&name) {
        return (id.clone(), name);
    }

    let id = unique_id(&format!("mem:{}", slugify(&name)), used_ids);
    nodes.push(memory_node_from_region(id.clone(), &name, endpoint));
    memory_node_ids.insert(name.clone(), id.clone());
    (id, name)
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

fn router_node(id: String, name: &str, router: &Router) -> GraphNode {
    GraphNode {
        id,
        kind: GraphNodeKind::Router,
        name: name.to_string(),
        label: name.to_string(),
        dimensions: Vec::new(),
        details: GraphNodeDetails::Router {
            router: GraphRouter {
                name: router.name.clone(),
                side_count: router.side_count(),
                endpoints: router.total_endpoints(),
            },
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

fn network_kind_label(net: &crate::arch::ScaleOutNetwork) -> &'static str {
    match net {
        crate::arch::ScaleOutNetwork::Mesh(_) => "mesh",
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

fn functionality_to_json(module: &Module) -> GraphFunctionalityModule {
    GraphFunctionalityModule {
        name: module.name.clone(),
        source_path: module.source.as_ref().map(|s| s.path.clone()),
        source_mlir_module_name: module
            .source
            .as_ref()
            .and_then(|s| s.mlir_module_name.clone()),
        ops: module.ops.iter().map(|op| op.name.clone()).collect(),
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
        Processors::Graph(graph) => {
            let mut out = Vec::new();
            for node in &graph.nodes {
                if let ArchNodeComponent::Architecture(arch) = &node.component {
                    out.extend(collect_processor_dims(arch));
                }
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
            access_granularity: bank.block_size.as_ref().map(size_expr_to_json),
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
            functionality: functionality_to_json(&proc.functionality),
        },
        Processors::Array {
            name, dims, elem, ..
        } => GraphProcessors::Array {
            name: name.clone(),
            dimensions: dims.iter().map(dimension_to_json).collect(),
            elem: Box::new(processors_to_json(elem)),
        },
        Processors::Graph(graph) => GraphProcessors::Graph {
            name: graph.name.clone(),
            processor_count: graph
                .nodes
                .iter()
                .filter(|n| matches!(n.component, ArchNodeComponent::Architecture(_)))
                .count(),
            link_count: 0,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::{architecture_to_graph_json, architecture_to_graph_json_value};
    use crate::arch::{
        ArchGraph, Architecture, Dimension, MemoryBank, MemoryRegion, Processor, ScaleOutNetwork,
        SizeExpr,
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
        let l2 = MemoryRegion::bank(MemoryBank::from_blocks(
            SizeExpr::Const(64),
            SizeExpr::Const(1024),
        ))
        .replicate(core_dim.as_slice())
        .with_name("l2");
        let lane = Processor::new("lane").replicate(core_dim.as_slice());
        let map = AffineMap::identity(core_dim.as_slice());

        let link = ScaleOutNetwork::mesh("l1_to_l2")
            .from_mem(&l1)
            .to_mem(&l2)
            .map(&map)
            .bandwidth(128)
            .build();

        let arch: Architecture = ArchGraph::builder("unit")
            .mem(&l1)
            .processor(&lane)
            .build()
            .into();
        let arch = arch
            .scale(core_dim.as_slice())
            .with_connectivity(vec![link]);
        let value = architecture_to_graph_json_value(&arch);
        assert_eq!(value["schema_version"], "mlar.arch-graph.v1");
        assert_eq!(value["architecture"]["name"], "unit");
        assert_eq!(value["nodes"].as_array().map(|v| v.len()), Some(3));
        assert_eq!(value["edges"].as_array().map(|v| v.len()), Some(1));
        assert_eq!(value["edges"][0]["map"]["expressions"][0], "core");
        assert_eq!(value["edges"][0]["bandwidth"]["const_value"], 128);
        assert!(value.get("intra_core").is_none());
    }

    #[test]
    fn scaled_architecture_has_no_intra_core_graph() {
        let bank_dim = Dimension::new_int("nbank", 16);
        let l1 = MemoryRegion::bank(MemoryBank::from_blocks(
            SizeExpr::Const(64),
            SizeExpr::Const(512),
        ))
        .replicate(bank_dim.as_slice())
        .with_name("l1");
        let lane = Processor::new("lane").into_elem();

        let core: Architecture = ArchGraph::builder("core")
            .mem(&l1)
            .processor(&lane)
            .build()
            .into();

        let dim_x = Dimension::new_int("x", 4);
        let dim_y = Dimension::new_int("y", 4);
        let mesh = core.scale([&dim_x, &dim_y]).with_name("mesh");

        let graph = architecture_to_graph_json(&mesh);
        assert!(graph.intra_core.is_none());
        assert!(!graph.nodes.is_empty());
    }
}
