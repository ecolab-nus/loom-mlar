use crate::arch::{Architecture, Dimension, MemoryRegion, Processor, ScaleOutNetwork};
use serde::Serialize;
use serde_json::Value;

use super::graph_json::{GraphDimension, GraphExpr, GraphMemoryRegion, GraphRouter, GraphSizeExpr};

const HIERARCHY_SCHEMA_VERSION: &str = "mlar.arch-hierarchy.v1";

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HierarchyNodeKind {
    Unit,
    Array,
    Graph,
    Memory,
    Router,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum HierarchyNodeDetails {
    Processor {
        functions: Vec<String>,
    },
    Memory {
        region: GraphMemoryRegion,
        total_size_bytes: Option<u64>,
    },
    Router {
        router: GraphRouter,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct HierarchyConnectivity {
    pub name: String,
    pub kind: String,
    pub bandwidth: GraphExpr,
    pub latency: Option<GraphExpr>,
    pub topology: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct HierarchyNode {
    pub kind: HierarchyNodeKind,
    pub name: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub dimensions: Vec<GraphDimension>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_instances: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<HierarchyNodeDetails>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub connectivity: Vec<HierarchyConnectivity>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<HierarchyNode>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ArchitectureHierarchyJson {
    pub schema_version: &'static str,
    pub root: HierarchyNode,
}

pub fn architecture_to_hierarchy_json(arch: &Architecture) -> ArchitectureHierarchyJson {
    ArchitectureHierarchyJson {
        schema_version: HIERARCHY_SCHEMA_VERSION,
        root: architecture_to_hierarchy_node(arch),
    }
}

pub fn architecture_to_hierarchy_json_value(arch: &Architecture) -> Value {
    serde_json::to_value(architecture_to_hierarchy_json(arch))
        .expect("hierarchy serialization must succeed")
}

pub fn architecture_to_hierarchy_json_string_pretty(
    arch: &Architecture,
) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(&architecture_to_hierarchy_json(arch))
}

fn architecture_to_hierarchy_node(arch: &Architecture) -> HierarchyNode {
    let mut children = Vec::new();
    children.extend(arch.memories.iter().map(memory_region_to_hierarchy_node));
    children.extend(arch.processors.iter().map(processor_to_hierarchy_node));
    children.extend(arch.children.iter().map(architecture_to_hierarchy_node));

    HierarchyNode {
        kind: if arch.dims.is_empty() {
            HierarchyNodeKind::Graph
        } else {
            HierarchyNodeKind::Array
        },
        name: arch.name.clone(),
        dimensions: arch.dims.iter().map(dimension_to_json).collect(),
        total_instances: arch.total_instances(),
        details: None,
        connectivity: arch.networks.iter().map(connectivity_to_json).collect(),
        children,
    }
}

fn memory_region_to_hierarchy_node(region: &MemoryRegion) -> HierarchyNode {
    let dimensions: Vec<GraphDimension> = collect_memory_dims(region)
        .iter()
        .map(dimension_to_json)
        .collect();
    let total_size = region.total_size_bytes();

    HierarchyNode {
        kind: HierarchyNodeKind::Memory,
        name: region.name().unwrap_or("unnamed_memory").to_string(),
        dimensions,
        total_instances: None,
        details: Some(HierarchyNodeDetails::Memory {
            region: memory_region_detail(region),
            total_size_bytes: total_size,
        }),
        connectivity: Vec::new(),
        children: Vec::new(),
    }
}

fn processor_to_hierarchy_node(processor: &Processor) -> HierarchyNode {
    HierarchyNode {
        kind: HierarchyNodeKind::Unit,
        name: processor
            .name
            .clone()
            .unwrap_or_else(|| "unnamed_processor".to_string()),
        dimensions: Vec::new(),
        total_instances: Some(1),
        details: Some(HierarchyNodeDetails::Processor {
            functions: processor
                .functionality
                .functions
                .iter()
                .map(|op| op.name.clone())
                .collect(),
        }),
        connectivity: Vec::new(),
        children: Vec::new(),
    }
}

fn connectivity_to_json(net: &ScaleOutNetwork) -> HierarchyConnectivity {
    let topology = if net.is_ring_topology() {
        "ring"
    } else {
        "general"
    };
    HierarchyConnectivity {
        name: net.name().to_string(),
        kind: match net {
            ScaleOutNetwork::Mesh(_) => "mesh".to_string(),
        },
        bandwidth: GraphExpr {
            expr: net.bandwidth().to_string(),
            const_value: net.bandwidth().eval_const(),
        },
        latency: net.latency().map(|l| GraphExpr {
            expr: l.to_string(),
            const_value: l.eval_const(),
        }),
        topology: topology.to_string(),
    }
}

fn dimension_to_json(dim: &Dimension) -> GraphDimension {
    GraphDimension {
        name: dim.name.0.clone(),
        size_expr: dim.size.to_string(),
        size_const: dim.size.as_const(),
    }
}

fn collect_memory_dims(region: &MemoryRegion) -> Vec<Dimension> {
    match region {
        MemoryRegion::Bank(_) => Vec::new(),
        MemoryRegion::Array {
            dims,
            sub_regions: sub_region,
            ..
        } => {
            let mut out = dims.clone();
            out.extend(collect_memory_dims(sub_region));
            out
        }
    }
}

fn memory_region_detail(region: &MemoryRegion) -> GraphMemoryRegion {
    let total_size_bytes = region.total_size_bytes();
    match region {
        MemoryRegion::Bank(bank) => GraphMemoryRegion::Bank {
            name: bank.name.clone(),
            capacity_bytes: GraphSizeExpr {
                expr: bank.capacity_bytes.to_string(),
                const_value: bank.capacity_bytes.as_const(),
            },
            access_granularity: bank.block_size.as_ref().map(|bs| GraphSizeExpr {
                expr: bs.to_string(),
                const_value: bs.as_const(),
            }),
            total_size_bytes,
        },
        MemoryRegion::Array {
            name,
            dims,
            sub_regions: sub_region,
        } => GraphMemoryRegion::Array {
            name: name.clone(),
            dimensions: dims.iter().map(dimension_to_json).collect(),
            sub_region: Box::new(memory_region_detail(sub_region)),
            total_size_bytes,
        },
    }
}
