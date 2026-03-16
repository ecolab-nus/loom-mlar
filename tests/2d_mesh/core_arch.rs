use mlar_rust::*;

use crate::matrix_lane::matrix_lane;
use crate::memory::l1;
use crate::vector_lane::vector_lane;

/// Build a single core with one router:
/// - compute side: matrix_lane + vector_lane
/// - memory side: all L1 banks
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
    let l1_ref = graph
        .memory_ref("L1")
        .expect("L1 memory region must be owned by core graph");
    let matrix_ref = graph
        .processor_ref("matrix_lane")
        .expect("matrix_lane processor must be owned by core graph");
    let vector_ref = graph
        .processor_ref("vector_lane")
        .expect("vector_lane processor must be owned by core graph");

    let core_router = Router::new("core_router")
        .side(
            RouterSide::new("compute")
                .endpoint(RouterEndpoint::from_proc_ref("matrix_lane", matrix_ref))
                .endpoint(RouterEndpoint::from_proc_ref("vector_lane", vector_ref)),
        )
        .side(RouterSide::from_memory_region_banks("memory", &l1, l1_ref));

    graph.add_router(&core_router);
    core
}
