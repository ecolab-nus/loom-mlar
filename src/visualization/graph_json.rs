use crate::arch::{
    Architecture, DataEffect, Dimension, MemoryAccessMode, MemoryRegion, Processor, Resource,
    SizeExpr,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

const GRAPH_SCHEMA_VERSION: &str = "mlar.arch-graph.v1";

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphNodeKind {
    Memory,
    Processor,
    DataMover,
    Array,
    Graph,
    Router,
    Resource,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphEdgeKind {
    ScaleOutNetwork,
    IntraGraph,
    ResourceDependency,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphEdgeDirection {
    Directional,
    Bidirectional,
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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LinkMapRelation {
    OneToOne,
    OneToMany,
    ManyToOne,
    ManyToMany,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LinkTopology {
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
    pub op_details: Vec<GraphFunctionalityOp>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GraphFunctionalityOp {
    pub name: String,
    pub source_memories: Vec<String>,
    pub destination_memories: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GraphRouter {
    pub name: String,
    pub side_count: usize,
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
    Array {
        name: Option<String>,
        dimensions: Vec<GraphDimension>,
        sub_region: Box<GraphMemoryRegion>,
        total_size_bytes: Option<u64>,
    },
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphResourceKind {
    Quantitative,
    Exclusive,
}

#[derive(Debug, Clone, Serialize)]
pub struct GraphResourceInfo {
    pub id: String,
    pub kind: GraphResourceKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capacity: Option<i64>,
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
    Resource {
        resource: GraphResourceInfo,
    },
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
pub struct GraphEdge {
    pub id: String,
    pub kind: GraphEdgeKind,
    pub name: String,
    pub source: String,
    pub target: String,
    pub source_name: String,
    pub target_name: String,
    pub label: String,
    pub direction: GraphEdgeDirection,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bandwidth: Option<GraphExpr>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latency: Option<GraphExpr>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub constraints: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sharing: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub map_relation: Option<GraphMapRelation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub topology: Option<GraphLinkTopology>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub map: Option<GraphAffineMap>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub side: Option<u32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GraphArchitectureLabel {
    pub name: String,
    pub dimensions: Vec<GraphDimension>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GraphArchitectureMeta {
    pub name: String,
    pub labels: Vec<GraphArchitectureLabel>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ArchitectureGraphJson {
    pub schema_version: &'static str,
    pub architecture: GraphArchitectureMeta,
    pub nodes: Vec<GraphNode>,
    pub edges: Vec<GraphEdge>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub intra_core: Option<Box<ArchitectureGraphJson>>,
}

pub fn architecture_to_graph_json(arch: &Architecture) -> ArchitectureGraphJson {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    append_scope(arch, &arch.name, &mut nodes, &mut edges);
    ArchitectureGraphJson {
        schema_version: GRAPH_SCHEMA_VERSION,
        architecture: GraphArchitectureMeta {
            name: arch.name.clone(),
            labels: vec![GraphArchitectureLabel {
                name: arch.name.clone(),
                dimensions: arch.dims.iter().map(dimension_to_json).collect(),
            }],
        },
        nodes,
        edges,
        intra_core: None,
    }
}

pub fn architecture_to_graph_json_value(arch: &Architecture) -> Value {
    serde_json::to_value(architecture_to_graph_json(arch))
        .expect("graph serialization must succeed")
}

pub fn architecture_to_graph_json_string(arch: &Architecture) -> Result<String, serde_json::Error> {
    serde_json::to_string(&architecture_to_graph_json(arch))
}

pub fn architecture_to_graph_json_string_pretty(
    arch: &Architecture,
) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(&architecture_to_graph_json(arch))
}

fn append_scope(
    arch: &Architecture,
    path: &str,
    nodes: &mut Vec<GraphNode>,
    edges: &mut Vec<GraphEdge>,
) {
    for memory in &arch.memories {
        let name = memory.name().unwrap_or("memory").to_string();
        let id = scoped_id(path, "mem", &name);
        nodes.push(GraphNode {
            id,
            kind: GraphNodeKind::Memory,
            name: name.clone(),
            label: name,
            dimensions: collect_memory_dims(memory)
                .iter()
                .map(|dim| dimension_to_json(dim))
                .collect(),
            details: GraphNodeDetails::Memory {
                region: memory_region_to_json(memory),
            },
        });
    }

    for processor in &arch.processors {
        append_processor(path, processor, nodes, edges);
    }

    for resource in &arch.resources {
        let id = scoped_id(path, "res", resource.id().as_str());
        nodes.push(GraphNode {
            id,
            kind: GraphNodeKind::Resource,
            name: resource.id().as_str().to_string(),
            label: resource.id().as_str().to_string(),
            dimensions: Vec::new(),
            details: GraphNodeDetails::Resource {
                resource: resource_to_json(resource),
            },
        });
    }

    for child in &arch.children {
        append_scope(child, &format!("{path}/{}", child.name), nodes, edges);
    }
}

fn append_processor(
    path: &str,
    processor: &Processor,
    nodes: &mut Vec<GraphNode>,
    edges: &mut Vec<GraphEdge>,
) {
    let name = processor.name.as_deref().unwrap_or("processor").to_string();
    let id = scoped_id(path, "proc", &name);
    let kind = if processor.effect == DataEffect::Preserve {
        GraphNodeKind::DataMover
    } else {
        GraphNodeKind::Processor
    };
    nodes.push(GraphNode {
        id: id.clone(),
        kind,
        name: name.clone(),
        label: name.clone(),
        dimensions: Vec::new(),
        details: GraphNodeDetails::Processor {
            element: processor_to_json(processor),
            total_instances: Some(1),
        },
    });

    for access in &processor.accesses {
        let memory_id = scoped_id(path, "mem", &access.region.name);
        let direction = match access.mode {
            MemoryAccessMode::Read => (memory_id.clone(), id.clone(), "read"),
            MemoryAccessMode::Write => (id.clone(), memory_id.clone(), "write"),
            MemoryAccessMode::ReadWrite => (id.clone(), memory_id.clone(), "read_write"),
        };
        edges.push(GraphEdge {
            id: scoped_id(path, "edge", &format!("{}_{}", name, access.region.name)),
            kind: GraphEdgeKind::IntraGraph,
            name: direction.2.to_string(),
            source: direction.0.clone(),
            target: direction.1.clone(),
            source_name: direction.0,
            target_name: direction.1,
            label: direction.2.to_string(),
            direction: if access.mode == MemoryAccessMode::ReadWrite {
                GraphEdgeDirection::Bidirectional
            } else {
                GraphEdgeDirection::Directional
            },
            bandwidth: None,
            latency: None,
            constraints: None,
            sharing: None,
            map_relation: None,
            topology: None,
            map: None,
            side: None,
        });
    }
}

fn scoped_id(path: &str, kind: &str, name: &str) -> String {
    format!("{kind}::{}::{}", sanitize(path), sanitize(name))
}

fn sanitize(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn dimension_to_json(dim: &Dimension) -> GraphDimension {
    GraphDimension {
        name: dim.name.0.clone(),
        size_expr: dim.size.to_string(),
        size_const: dim.size.as_const(),
    }
}

fn size_expr_to_json(expr: &SizeExpr) -> GraphSizeExpr {
    GraphSizeExpr {
        expr: expr.to_string(),
        const_value: expr.as_const(),
    }
}

fn collect_memory_dims(region: &MemoryRegion) -> Vec<&Dimension> {
    match region {
        MemoryRegion::Bank(_) => Vec::new(),
        MemoryRegion::Array {
            dims, sub_regions, ..
        } => {
            let mut out: Vec<&Dimension> = dims.iter().collect();
            out.extend(collect_memory_dims(sub_regions));
            out
        }
    }
}

fn memory_region_to_json(region: &MemoryRegion) -> GraphMemoryRegion {
    match region {
        MemoryRegion::Bank(bank) => GraphMemoryRegion::Bank {
            name: bank.name.clone(),
            capacity_bytes: size_expr_to_json(&bank.capacity_bytes),
            access_granularity: bank.block_size.as_ref().map(size_expr_to_json),
            total_size_bytes: region.total_size_bytes(),
        },
        MemoryRegion::Array {
            name,
            dims,
            sub_regions,
        } => GraphMemoryRegion::Array {
            name: name.clone(),
            dimensions: dims.iter().map(dimension_to_json).collect(),
            sub_region: Box::new(memory_region_to_json(sub_regions)),
            total_size_bytes: region.total_size_bytes(),
        },
    }
}

fn processor_to_json(processor: &Processor) -> GraphProcessors {
    GraphProcessors::Unit {
        name: processor.name.clone(),
        functionality: GraphFunctionalityModule {
            name: processor.functionality.module_name.clone(),
            source_path: processor.functionality.path.clone(),
            source_mlir_module_name: processor.functionality.module_name.clone(),
            ops: processor
                .functionality
                .functions
                .iter()
                .map(|func| func.name.clone())
                .collect(),
            op_details: processor
                .functionality
                .functions
                .iter()
                .map(|func| GraphFunctionalityOp {
                    name: func.name.clone(),
                    source_memories: processor
                        .accesses
                        .iter()
                        .filter(|access| {
                            matches!(
                                access.mode,
                                MemoryAccessMode::Read | MemoryAccessMode::ReadWrite
                            )
                        })
                        .map(|access| access.region.name.clone())
                        .collect(),
                    destination_memories: processor
                        .accesses
                        .iter()
                        .filter(|access| {
                            matches!(
                                access.mode,
                                MemoryAccessMode::Write | MemoryAccessMode::ReadWrite
                            )
                        })
                        .map(|access| access.region.name.clone())
                        .collect(),
                })
                .collect(),
        },
    }
}

fn resource_to_json(resource: &Resource) -> GraphResourceInfo {
    match resource {
        Resource::Quantitative { id, capacity } => GraphResourceInfo {
            id: id.as_str().to_string(),
            kind: GraphResourceKind::Quantitative,
            capacity: Some(*capacity),
        },
        Resource::Exclusive { id } => GraphResourceInfo {
            id: id.as_str().to_string(),
            kind: GraphResourceKind::Exclusive,
            capacity: None,
        },
    }
}
