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
        .processor(&matrix_lane)
        .processor(&vector_lane)
        .build()
        .into();

    let graph = core
        .as_graph_mut()
        .expect("core builder should produce graph architecture");

    let core_router = Router::new("core_router", 2);
    graph.add_router(&core_router);

    core
}
