use mlar_rust::*;

use crate::core_arch::single_core;
use crate::data_movers::dram_to_l1_mover;
use crate::dimensions::{dim_x, dim_y};
use crate::memory::dram;

/// Scale a single core to an 8×8 mesh and add torus interconnects.
///
/// Torus links connect neighboring L1 caches with wraparound:
/// - horizontal ring: L1(x, y) → L1(x, (y+1) mod 8)
/// - vertical ring:   L1(x, y) → L1((x+1) mod 8, y)
pub fn scaled_mesh_torus() -> Architecture {
    let core = single_core();
    let dim_x = dim_x();
    let dim_y = dim_y();
    let mesh = core.scale([&dim_x, &dim_y]).with_name("mesh");
    let dram = dram();
    let dram_to_l1 = dram_to_l1_mover();

    let scaled_l1 = mesh
        .get_memory_region("L1")
        .expect("scaled mesh should contain L1")
        .clone()
        .scale(&[dim_x.clone(), dim_y.clone()])
        .with_name("L1");

    // External IO is only at mesh boundaries: left edge (x = 0) and right edge (x = 7).
    let io_side = Dimension::new_int("io_side", 2);
    let io_map = AffineMapTemplate::parse("[io_side, y] -> [x, y]: (io_side * 7, y)")
        .expect("invalid affine map")
        .bind([&io_side, &dim_x, &dim_y])
        .expect("failed to bind");
    let io = MeshNetworkInterface::new(io_map, Expr::Const(64))
        .with_data_mover(dram_to_l1.clone());

    // Horizontal torus: y-neighbor with wraparound
    let torus_y_map = AffineMapTemplate::parse("[x, y] -> [x, y]: (x, (y + 1) mod 8)")
        .expect("invalid affine map")
        .bind([&dim_x, &dim_y])
        .expect("failed to bind");

    // Vertical torus: x-neighbor with wraparound
    let torus_x_map = AffineMapTemplate::parse("[x, y] -> [x, y]: ((x + 1) mod 8, y)")
        .expect("invalid affine map")
        .bind([&dim_x, &dim_y])
        .expect("failed to bind");

    let torus = ScaleOutNetwork::mesh("L1_torus")
        .mem_region(&scaled_l1)
        .links(&[torus_y_map.clone(), torus_x_map.clone()])
        .io(&io)
        .link_bandwidth(64)
        .build();

    let mesh = mesh.with_connectivity(vec![torus]);

    let mut system: Architecture = ArchGraph::builder("2d_mesh_torus")
        .processor(&mesh)
        .data_mover(&dram_to_l1)
        .mem(&dram)
        .build()
        .into();

    let graph = system
        .as_graph_mut()
        .expect("system architecture should be a graph");

    let router_id = graph.add_router(&Router::new("mesh_dram_router", 3));
    let mesh_id = graph.processor_ref("mesh").expect("mesh node");
    let mover_id = graph
        .data_mover_ref("dram_to_l1_mover")
        .expect("dram_to_l1_mover node");
    let dram_id = graph.memory_ref("DRAM").expect("DRAM node");

    let router_node = graph.get_node(&router_id).expect("router node").clone();
    let mesh_node = graph.get_node(&mesh_id).expect("mesh node").clone();
    let mover_node = graph.get_node(&mover_id).expect("mover node").clone();
    let dram_node = graph.get_node(&dram_id).expect("DRAM node").clone();

    graph.connect_with_attrs(
        &mesh_node,
        &router_node,
        vec![
            ArchEdgeAttr::Side(0),
            ArchEdgeAttr::Direction(ArchEdgeDirection::Bidirectional),
        ],
    );
    graph.connect_with_attrs(
        &router_node,
        &mover_node,
        vec![
            ArchEdgeAttr::Side(1),
            ArchEdgeAttr::Direction(ArchEdgeDirection::Bidirectional),
        ],
    );
    graph.connect_with_attrs(
        &mover_node,
        &dram_node,
        vec![
            ArchEdgeAttr::Side(2),
            ArchEdgeAttr::Direction(ArchEdgeDirection::Bidirectional),
        ],
    );

    system
}
