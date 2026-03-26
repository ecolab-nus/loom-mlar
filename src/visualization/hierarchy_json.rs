use crate::arch::{
    ArchNodeComponent, Architecture, Dimension, MemoryRegion, Router, ScaleOutNetwork,
};
use serde::Serialize;
use serde_json::Value;

use super::graph_json::{GraphDimension, GraphExpr, GraphMemoryRegion, GraphRouter, GraphSizeExpr};

const HIERARCHY_SCHEMA_VERSION: &str = "mlar.arch-hierarchy.v1";

#[derive(Debug, Clone, Serialize)]
pub struct ArchitectureHierarchyJson {
    pub schema_version: &'static str,
    pub root: HierarchyNode,
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
    match arch {
        Architecture::Unit(proc) => {
            let functions: Vec<String> = proc
                .functionality
                .functions
                .iter()
                .map(|op| op.name.clone())
                .collect();
            HierarchyNode {
                kind: HierarchyNodeKind::Unit,
                name: proc.name.clone().unwrap_or_else(|| "unnamed".into()),
                dimensions: Vec::new(),
                total_instances: Some(1),
                details: Some(HierarchyNodeDetails::Processor { functions }),
                connectivity: Vec::new(),
                children: Vec::new(),
            }
        }
        Architecture::Array {
            name,
            dims,
            elem,
            connectivity,
            ..
        } => {
            let dimensions: Vec<GraphDimension> = dims.iter().map(dimension_to_json).collect();
            let total = arch.total_instances();
            let conn: Vec<HierarchyConnectivity> =
                connectivity.iter().map(connectivity_to_json).collect();
            let child = architecture_to_hierarchy_node(elem);

            HierarchyNode {
                kind: HierarchyNodeKind::Array,
                name: name
                    .clone()
                    .or_else(|| elem.name().map(String::from))
                    .unwrap_or_else(|| "unnamed".into()),
                dimensions,
                total_instances: total,
                details: None,
                connectivity: conn,
                children: vec![child],
            }
        }
        Architecture::Graph(graph) => {
            let mut children = Vec::new();
            for node in &graph.nodes {
                children.push(graph_node_to_hierarchy_node(&node.component));
            }

            HierarchyNode {
                kind: HierarchyNodeKind::Graph,
                name: graph.name.clone(),
                dimensions: Vec::new(),
                total_instances: arch.total_instances(),
                details: None,
                connectivity: Vec::new(),
                children,
            }
        }
    }
}

fn graph_node_to_hierarchy_node(component: &ArchNodeComponent) -> HierarchyNode {
    match component {
        ArchNodeComponent::Architecture(arch) => architecture_to_hierarchy_node(arch),
        ArchNodeComponent::DataMover(mover) => HierarchyNode {
            kind: HierarchyNodeKind::Unit,
            name: mover
                .name
                .clone()
                .unwrap_or_else(|| "unnamed_data_mover".to_string()),
            dimensions: Vec::new(),
            total_instances: Some(1),
            details: Some(HierarchyNodeDetails::Processor {
                functions: mover
                    .functionality
                    .functions
                    .iter()
                    .map(|op| op.name.clone())
                    .collect(),
            }),
            connectivity: Vec::new(),
            children: Vec::new(),
        },
        ArchNodeComponent::MemoryRegion(region) => memory_region_to_hierarchy_node(region),
        ArchNodeComponent::Router(router) => router_to_hierarchy_node(router),
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

fn router_to_hierarchy_node(router: &Router) -> HierarchyNode {
    HierarchyNode {
        kind: HierarchyNodeKind::Router,
        name: router.name.clone(),
        dimensions: Vec::new(),
        total_instances: None,
        details: Some(HierarchyNodeDetails::Router {
            router: GraphRouter {
                name: router.name.clone(),
                side_count: router.side_count(),
            },
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::{
        ArchGraph, Dimension, MemoryBank, MemoryRegion, Processor, Router, SizeExpr,
    };

    #[test]
    fn hierarchy_captures_unit_processor() {
        let proc = Processor::new("lane").into_elem();
        let json = architecture_to_hierarchy_json(&proc);
        assert_eq!(json.schema_version, "mlar.arch-hierarchy.v1");
        assert_eq!(json.root.name, "lane");
        assert!(matches!(json.root.kind, HierarchyNodeKind::Unit));
        assert_eq!(json.root.total_instances, Some(1));
    }

    #[test]
    fn hierarchy_captures_graph_with_router() {
        let l1 = MemoryRegion::bank(MemoryBank::from_blocks(
            SizeExpr::Const(128),
            SizeExpr::Const(1024),
        ))
        .with_name("L1");
        let lane = Processor::new("lane").into_elem();
        let router = Router::new("xbar", 1);

        let core: Architecture = ArchGraph::builder("core")
            .mem(&l1)
            .processor(&lane)
            .router(&router)
            .build()
            .into();

        let json = architecture_to_hierarchy_json(&core);
        assert_eq!(json.root.name, "core");
        assert!(matches!(json.root.kind, HierarchyNodeKind::Graph));
        assert_eq!(json.root.children.len(), 3);

        let kinds: Vec<_> = json
            .root
            .children
            .iter()
            .map(|c| format!("{:?}", c.kind))
            .collect();
        assert!(kinds.contains(&"Memory".to_string()));
        assert!(kinds.contains(&"Unit".to_string()));
        assert!(kinds.contains(&"Router".to_string()));
    }

    #[test]
    fn hierarchy_captures_scaled_architecture() {
        let l1 = MemoryRegion::bank(MemoryBank::from_blocks(
            SizeExpr::Const(128),
            SizeExpr::Const(1024),
        ))
        .with_name("L1");
        let lane = Processor::new("lane").into_elem();
        let core: Architecture = ArchGraph::builder("core")
            .mem(&l1)
            .processor(&lane)
            .build()
            .into();

        let dim_x = Dimension::new_int("x", 4);
        let dim_y = Dimension::new_int("y", 4);
        let mesh = core.scale([&dim_x, &dim_y]).with_name("mesh");

        let json = architecture_to_hierarchy_json(&mesh);
        assert_eq!(json.root.name, "mesh");
        assert!(matches!(json.root.kind, HierarchyNodeKind::Array));
        assert_eq!(json.root.dimensions.len(), 2);
        assert_eq!(json.root.total_instances, Some(16));
        assert_eq!(json.root.children.len(), 1);

        let child = &json.root.children[0];
        assert!(matches!(child.kind, HierarchyNodeKind::Graph));
        assert_eq!(child.name, "core");
    }

    #[test]
    fn hierarchy_json_roundtrip() {
        let lane = Processor::new("lane").into_elem();
        let json_str = architecture_to_hierarchy_json_string_pretty(&lane).unwrap();
        let value: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(value["schema_version"], "mlar.arch-hierarchy.v1");
        assert_eq!(value["root"]["kind"], "unit");
        assert_eq!(value["root"]["name"], "lane");
    }

    #[test]
    fn generate_sample_hierarchy_json() {
        use crate::arch::{MeshNetworkInterface, ScaleOutNetwork};
        use crate::math::{AffineExpr, AffineMap, Expr};

        let dim_bank = Dimension::new_int("nbank", 16);
        let l1 = MemoryRegion::bank(MemoryBank::from_blocks(
            SizeExpr::Const(128),
            SizeExpr::Const(1024),
        ))
        .scale(dim_bank.as_slice())
        .with_name("L1");

        let matrix_lane = Processor::new("matrix_lane").into_elem();
        let vector_lane = Processor::new("vector_lane").into_elem();

        let core_router = Router::new("core_router", 2);

        let core: Architecture = ArchGraph::builder("core")
            .mem(&l1)
            .processor(&matrix_lane)
            .processor(&vector_lane)
            .router(&core_router)
            .build()
            .into();

        let dim_x = Dimension::new_int("x", 8);
        let dim_y = Dimension::new_int("y", 8);

        let scaled_l1 = l1.clone().scale(&[dim_x.clone(), dim_y.clone()]);

        let torus_y_map = AffineMap::new(
            &[dim_x.clone(), dim_y.clone()],
            &[dim_x.clone(), dim_y.clone()],
            vec![
                AffineExpr::var(dim_x.clone()),
                AffineExpr::modulo(
                    AffineExpr::add(AffineExpr::var(dim_y.clone()), AffineExpr::constant(1)),
                    AffineExpr::constant(8),
                ),
            ],
        );
        let io = MeshNetworkInterface::new(
            AffineMap::identity(&[dim_x.clone(), dim_y.clone()]),
            Expr::Const(64),
        );

        let torus_y = ScaleOutNetwork::mesh("L1_torus_y")
            .mem_region(&scaled_l1)
            .map(&torus_y_map)
            .io(&io)
            .link_bandwidth(64)
            .build();

        let torus_x_map = AffineMap::new(
            &[dim_x.clone(), dim_y.clone()],
            &[dim_x.clone(), dim_y.clone()],
            vec![
                AffineExpr::modulo(
                    AffineExpr::add(AffineExpr::var(dim_x.clone()), AffineExpr::constant(1)),
                    AffineExpr::constant(8),
                ),
                AffineExpr::var(dim_y.clone()),
            ],
        );
        let torus_x = ScaleOutNetwork::mesh("L1_torus_x")
            .mem_region(&scaled_l1)
            .map(&torus_x_map)
            .io(&io)
            .link_bandwidth(64)
            .build();

        let mesh = core
            .scale([&dim_x, &dim_y])
            .with_name("2d_mesh_torus")
            .with_connectivity(vec![torus_y, torus_x]);

        let json_str = architecture_to_hierarchy_json_string_pretty(&mesh).unwrap();

        let out_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tools/web-visualization/public/sample-hierarchy.json");
        std::fs::write(&out_path, &json_str).expect("Failed to write sample hierarchy JSON");

        let value: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(value["schema_version"], "mlar.arch-hierarchy.v1");
        assert_eq!(value["root"]["name"], "2d_mesh_torus");
        assert_eq!(value["root"]["kind"], "array");
        assert_eq!(
            value["root"]["dimensions"].as_array().map(|v| v.len()),
            Some(2)
        );
        assert!(
            value["root"]["connectivity"]
                .as_array()
                .is_some_and(|v| v.len() == 2)
        );
    }
}
