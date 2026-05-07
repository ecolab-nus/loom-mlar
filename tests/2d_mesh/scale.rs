use mlar_rust::*;

use crate::data_movers::{dram_l1_noc0, dram_l1_noc1};
use crate::memory::dram;
use crate::mesh::{mesh_array_l1, scaled_mesh};

/// Build the full system: 8×8 mesh, DRAM, two NoC data movers, and a router.
///
/// There is no explicit inter-core mesh connectivity; cross-core transfers are
/// modeled entirely through the NoC data movers attached to the system graph.
pub fn scaled_mesh_torus() -> Architecture {
    let dram = dram();
    let mesh = scaled_mesh();
    let array_l1 = mesh_array_l1(&mesh);

    let noc0 = dram_l1_noc0(&dram, &array_l1);
    let noc1 = dram_l1_noc1(&dram, &array_l1);

    let mut system: Architecture = ArchGraph::builder("system")
        .architecture(&mesh)
        .mem(&dram)
        .data_mover(&noc0)
        .data_mover(&noc1)
        .build()
        .into();

    let graph = system
        .as_graph_mut()
        .expect("system architecture should be a graph");

    let router_id = graph.add_router(&Router::new("mesh_dram_router", 3));
    let mesh_id = graph.processor_ref("mesh").expect("mesh node");
    let noc0_id = graph
        .data_mover_ref("dram_l1_noc0")
        .expect("dram_l1_noc0 node");
    let noc1_id = graph
        .data_mover_ref("dram_l1_noc1")
        .expect("dram_l1_noc1 node");
    let dram_id = graph.memory_ref("DRAM").expect("DRAM node");

    let router_node = graph.get_node(&router_id).expect("router node").clone();
    let mesh_node = graph.get_node(&mesh_id).expect("mesh node").clone();
    let dram_node = graph.get_node(&dram_id).expect("DRAM node").clone();

    let mover_nodes: Vec<_> = [noc0_id, noc1_id]
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
