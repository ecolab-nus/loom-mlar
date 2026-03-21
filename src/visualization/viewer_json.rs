use super::graph_json::{ArchitectureGraphJson, architecture_to_graph_json};
use super::hierarchy_json::{HierarchyNode, architecture_to_hierarchy_json};
use crate::arch::{ArchNodeComponent, Architecture};
use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;

const VIEWER_SCHEMA_VERSION: &str = "mlar.arch-viewer.v1";

/// Combined payload for the web viewer: hierarchy tree + per-node graph views.
#[derive(Debug, Clone, Serialize)]
pub struct ArchitectureViewerJson {
    pub schema_version: &'static str,
    pub hierarchy: HierarchyNode,
    pub graphs: HashMap<String, ArchitectureGraphJson>,
}

/// Build the combined viewer payload for an architecture.
///
/// The `graphs` map is keyed by path strings (e.g. `""` for the root,
/// `"core"` for a child named "core", `"core/lane"` for a nested child).
/// Each value is a self-contained graph JSON suitable for React Flow rendering.
pub fn architecture_to_viewer_json(arch: &Architecture) -> ArchitectureViewerJson {
    let hierarchy = architecture_to_hierarchy_json(arch);
    let mut graphs = HashMap::new();
    collect_sub_graphs(arch, "", &mut graphs);

    ArchitectureViewerJson {
        schema_version: VIEWER_SCHEMA_VERSION,
        hierarchy: hierarchy.root,
        graphs,
    }
}

pub fn architecture_to_viewer_json_value(arch: &Architecture) -> Value {
    serde_json::to_value(architecture_to_viewer_json(arch))
        .expect("viewer serialization must succeed")
}

pub fn architecture_to_viewer_json_string_pretty(
    arch: &Architecture,
) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(&architecture_to_viewer_json(arch))
}

fn collect_sub_graphs(
    arch: &Architecture,
    path: &str,
    graphs: &mut HashMap<String, ArchitectureGraphJson>,
) {
    if matches!(arch, Architecture::Unit(_)) && !path.is_empty() {
        return;
    }

    graphs.insert(path.to_string(), architecture_to_graph_json(arch));

    match arch {
        Architecture::Unit(_) => {}
        Architecture::Array { elem, .. } => {
            let child_name = elem.name().unwrap_or("elem");
            let child_path = sub_path(path, child_name);
            collect_sub_graphs(elem, &child_path, graphs);
        }
        Architecture::Graph(graph) => {
            for node in &graph.nodes {
                if let ArchNodeComponent::Architecture(sub_arch) = &node.component {
                    let child_name = sub_arch.name().unwrap_or("unnamed");
                    let child_path = sub_path(path, child_name);
                    collect_sub_graphs(sub_arch, &child_path, graphs);
                }
            }
        }
    }
}

fn sub_path(parent: &str, child: &str) -> String {
    if parent.is_empty() {
        child.to_string()
    } else {
        format!("{parent}/{child}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::{
        ArchEdgeAttr, ArchGraph, Dimension, MemoryBank, MemoryRegion, MeshNetworkInterface,
        Processor, Router, ScaleOutNetwork, SizeExpr,
    };
    use crate::math::{AffineExpr, AffineMap, Expr};

    fn build_core_with_edges() -> (Architecture, MemoryRegion) {
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

        let mut core: Architecture = ArchGraph::builder("core")
            .mem(&l1)
            .processor(&matrix_lane)
            .processor(&vector_lane)
            .router(&core_router)
            .build()
            .into();

        {
            let graph = core.as_graph_mut().unwrap();
            let router_id = graph.router_ref("core_router").unwrap();
            let mem_id = graph.memory_ref("L1").unwrap();
            let mat_id = graph.processor_ref("matrix_lane").unwrap();
            let vec_id = graph.processor_ref("vector_lane").unwrap();

            let router_node = graph.get_node(&router_id).unwrap().clone();
            let mem_node = graph.get_node(&mem_id).unwrap().clone();
            let mat_node = graph.get_node(&mat_id).unwrap().clone();
            let vec_node = graph.get_node(&vec_id).unwrap().clone();

            graph.connect_with_attrs(&mat_node, &router_node, vec![ArchEdgeAttr::Side(0)]);
            graph.connect_with_attrs(&vec_node, &router_node, vec![ArchEdgeAttr::Side(0)]);
            graph.connect_with_attrs(&router_node, &mem_node, vec![ArchEdgeAttr::Side(1)]);
        }

        (core, l1)
    }

    #[test]
    fn viewer_json_has_both_hierarchy_and_graphs() {
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

        let json = architecture_to_viewer_json(&core);
        assert_eq!(json.schema_version, "mlar.arch-viewer.v1");
        assert_eq!(json.hierarchy.name, "core");
        assert!(json.graphs.contains_key(""));
    }

    #[test]
    fn viewer_json_has_sub_graphs_for_scaled_architecture() {
        let (core, l1) = build_core_with_edges();

        let dim_x = Dimension::new_int("x", 8);
        let dim_y = Dimension::new_int("y", 8);

        let scaled_l1 = l1.scale(&[dim_x.clone(), dim_y.clone()]);

        let io = MeshNetworkInterface::new(
            AffineMap::identity(&[dim_x.clone(), dim_y.clone()]),
            Expr::Const(64),
        );

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

        let json = architecture_to_viewer_json(&mesh);
        assert_eq!(json.schema_version, "mlar.arch-viewer.v1");
        assert_eq!(json.hierarchy.name, "2d_mesh_torus");

        assert!(json.graphs.contains_key(""), "root graph should exist");
        assert!(
            json.graphs.contains_key("core"),
            "core sub-graph should exist"
        );

        let root_graph = &json.graphs[""];
        assert!(!root_graph.nodes.is_empty(), "root graph should have nodes");

        let core_graph = &json.graphs["core"];
        assert!(!core_graph.nodes.is_empty(), "core graph should have nodes");
        assert!(
            !core_graph.edges.is_empty(),
            "core graph should have intra-graph edges"
        );
    }

    #[test]
    fn generate_sample_viewer_json() {
        let (core, l1) = build_core_with_edges();

        let dim_x = Dimension::new_int("x", 8);
        let dim_y = Dimension::new_int("y", 8);

        let scaled_l1 = l1.scale(&[dim_x.clone(), dim_y.clone()]);

        let io = MeshNetworkInterface::new(
            AffineMap::identity(&[dim_x.clone(), dim_y.clone()]),
            Expr::Const(64),
        );

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

        let json_str = architecture_to_viewer_json_string_pretty(&mesh).unwrap();

        let value: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(value["schema_version"], "mlar.arch-viewer.v1");
        assert_eq!(value["hierarchy"]["name"], "2d_mesh_torus");
        assert!(value["graphs"][""].is_object());
        assert!(value["graphs"]["core"].is_object());
    }
}
