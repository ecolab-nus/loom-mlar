use mlar_rust::*;

use crate::matrix_lane::matrix_lane;
use crate::memory::l1;
use crate::vector_lane::vector_lane;

/// Build a single core with one router:
/// - side 0 (compute): matrix_lane + vector_lane
/// - side 1 (memory): L1
pub fn single_core() -> Architecture {
    let l1 = l1();
    let matrix_lane = matrix_lane();
    let vector_lane = vector_lane();

    let mut core: Architecture = ArchGraph::builder("core")
        .mem(&l1)
        .architecture(&matrix_lane)
        .architecture(&vector_lane)
        .build()
        .into();

    let graph = core
        .as_graph_mut()
        .expect("core builder should produce graph architecture");

    let core_router = Router::new("core_router", 2);
    let router_id = graph.add_router(&core_router);

    let mem_id = graph.memory_ref("L1").expect("L1 memory node");
    let mat_id = graph
        .processor_ref("matrix_lane")
        .expect("matrix_lane node");
    let vec_id = graph
        .processor_ref("vector_lane")
        .expect("vector_lane node");

    let router_node = graph.get_node(&router_id).unwrap().clone();
    let mem_node = graph.get_node(&mem_id).unwrap().clone();
    let mat_node = graph.get_node(&mat_id).unwrap().clone();
    let vec_node = graph.get_node(&vec_id).unwrap().clone();

    graph.connect_with_attrs(
        &mat_node,
        &router_node,
        vec![
            ArchEdgeAttr::Side(0),
            ArchEdgeAttr::Direction(ArchEdgeDirection::Bidirectional),
        ],
    );
    graph.connect_with_attrs(
        &vec_node,
        &router_node,
        vec![
            ArchEdgeAttr::Side(0),
            ArchEdgeAttr::Direction(ArchEdgeDirection::Bidirectional),
        ],
    );
    graph.connect_with_attrs(
        &router_node,
        &mem_node,
        vec![
            ArchEdgeAttr::Side(1),
            ArchEdgeAttr::Direction(ArchEdgeDirection::Bidirectional),
        ],
    );

    core
}
