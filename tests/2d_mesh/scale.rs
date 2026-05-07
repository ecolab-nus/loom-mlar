use mlar_rust::*;

use crate::data_movers::{dram_l1_bcst_h_mover, dram_l1_bcst_v_mover, dram_l1_mover};
use crate::memory::dram;
use crate::mesh::scaled_mesh;

/// Build the full system: 8×8 mesh with torus interconnects, DRAM, and a router.
pub fn scaled_mesh_torus() -> Architecture {
    let dram = dram();

    let mesh = scaled_mesh(|l1| {
        vec![
            dram_l1_mover(&dram, l1),
            dram_l1_bcst_v_mover(&dram, l1),
            dram_l1_bcst_h_mover(&dram, l1),
        ]
    });

    let mut system: Architecture = ArchGraph::builder("system")
        .architecture(&mesh)
        .mem(&dram)
        .build()
        .into();

    let graph = system
        .as_graph_mut()
        .expect("system architecture should be a graph");

    let router_id = graph.add_router(&Router::new("mesh_dram_router", 3));
    let mesh_id = graph.processor_ref("mesh").expect("mesh node");
    let mover_id = graph
        .data_mover_ref("dram_l1_mover")
        .expect("dram_l1_mover node");
    let bcst_v_id = graph
        .data_mover_ref("dram_l1_bcst_v")
        .expect("dram_l1_bcst_v node");
    let bcst_h_id = graph
        .data_mover_ref("dram_l1_bcst_h")
        .expect("dram_l1_bcst_h node");
    let dram_id = graph.memory_ref("DRAM").expect("DRAM node");

    let router_node = graph.get_node(&router_id).expect("router node").clone();
    let mesh_node = graph.get_node(&mesh_id).expect("mesh node").clone();
    let dram_node = graph.get_node(&dram_id).expect("DRAM node").clone();

    let mover_nodes: Vec<_> = [mover_id, bcst_v_id, bcst_h_id]
        .iter()
        .map(|id| graph.get_node(id).expect("mover node").clone())
        .collect();

    graph.connect_with_attrs(
        &mesh_node,
        &router_node,
        vec![
            ArchEdgeAttr::Side(0),
            ArchEdgeAttr::Direction(ArchEdgeDirection::Bidirectional),
        ],
    );

    for mover_node in &mover_nodes {
        graph.connect_with_attrs(
            &router_node,
            mover_node,
            vec![
                ArchEdgeAttr::Side(1),
                ArchEdgeAttr::Direction(ArchEdgeDirection::Bidirectional),
            ],
        );
        graph.connect_with_attrs(
            mover_node,
            &dram_node,
            vec![
                ArchEdgeAttr::Side(2),
                ArchEdgeAttr::Direction(ArchEdgeDirection::Bidirectional),
            ],
        );
    }

    system
}
