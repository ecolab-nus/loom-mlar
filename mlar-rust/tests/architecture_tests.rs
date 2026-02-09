use mlar_rust::*;
use mlar_rust::lane::{MatMulLane, VecLane};
use std::fs;

#[test]
fn test_core_architecture() {
    // === Dimensions ===
    let dim_bank = Dimension::new("nbank", 16);
    let dim_x = Dimension::new("x", 8);
    let dim_y = Dimension::new("y", 8);

    // === Define a single core ===

    // L1: 16 banks, each bank has 1024 x 128B blocks = 16KB per bank, 256KB total
    let l1 = MemRegion::bank(Bank {
        block_size: Size::int(128),
        num_blocks: Size::int(1024),
    })
    .scale([&dim_bank]); // 16 banks => one L1

    // Matrix lane (single processor, reads from L1)
    let matrix_lane = FunctionalLane::new(
        "matrix_lane",
        vec![&l1, &l1],
        vec![&l1],
        MatMulLane,
    );
    let matrix_lane_set = ProcessorSet::from_lane(matrix_lane);

    // Vector lane (single processor, reads from L1)
    let vector_lane = FunctionalLane::new(
        "vector_lane",
        vec![&l1, &l1],
        vec![&l1],
        VecLane,
    );
    let vector_lane_set = ProcessorSet::from_lane(vector_lane);

    // All-to-one: all 16 L1 banks visible to the single lane
    let all_to_one_map = AffineMap::new(
        vec![dim_bank.clone()], // source: bank index
        vec![],                 // target: no dims (single processor)
        vec![],                 // no result expressions
    );

    let l1_to_matrix = MemoryProcessorInterconnect::builder("l1_to_matrix_lane")
        .source(&l1)
        .target(&matrix_lane_set)
        .affine_map(all_to_one_map.clone())
        .bandwidth(512)
        .build();

    let l1_to_vector = MemoryProcessorInterconnect::builder("l1_to_vector_lane")
        .source(&l1)
        .target(&vector_lane_set)
        .affine_map(all_to_one_map)
        .bandwidth(128)
        .build();

    // Build a single core as an Architecture
    let core = Architecture::builder("core")
        .dim(&dim_bank)
        .mem("l1", l1)
        .processor("matrix_lane", matrix_lane_set)
        .processor("vector_lane", vector_lane_set)
        .mem_proc_interconnect(l1_to_matrix)
        .mem_proc_interconnect(l1_to_vector)
        .build();

    // Verify single core
    assert_eq!(core.name, "core");
    assert_eq!(core.processor_sets.len(), 2);
    assert_eq!(core.memory_regions.len(), 1);
    assert_eq!(core.memory_processor_interconnects.len(), 2);
    assert_eq!(core.total_processing_elements(), Some(2));

    // Visualize single core
    let core_dot = architecture_to_dot(&core);
    fs::write("core_architecture.dot", &core_dot).expect("Failed to write DOT file");
    println!("Generated core_architecture.dot");

    // === Scale to 8x8 cores ===
    let cores = core.scale([&dim_x, &dim_y]);

    // Verify scaled architecture
    assert_eq!(cores.name, "core"); // name preserved
    assert_eq!(cores.dimensions.len(), 3); // x, y, nbank

    // Processor sets are now scaled by [x, y]
    assert_eq!(cores.processor_sets.len(), 2);
    let (mat_name, mat_set) = &cores.processor_sets[0];
    assert_eq!(mat_name, "matrix_lane");
    assert_eq!(mat_set.total_instances(), Some(64)); // 8x8

    let (vec_name, vec_set) = &cores.processor_sets[1];
    assert_eq!(vec_name, "vector_lane");
    assert_eq!(vec_set.total_instances(), Some(64)); // 8x8

    // Memory region "l1" is now scaled by [x, y] on top of [nbank]
    let all_l1s = cores.get_memory_region("l1").unwrap();
    match all_l1s {
        MemRegion::Indexed { indices, sub_region } => {
            // Outermost: [x, y]
            assert_eq!(indices.len(), 2);
            assert_eq!(indices[0].name, "x");
            assert_eq!(indices[1].name, "y");
            // Inner: [nbank] -> Bank
            match sub_region.as_ref() {
                MemRegion::Indexed { indices: inner, .. } => {
                    assert_eq!(inner.len(), 1);
                    assert_eq!(inner[0].name, "nbank");
                }
                _ => panic!("expected inner Indexed region"),
            }
        }
        _ => panic!("expected Indexed region"),
    }

    // Interconnect maps were replaced with identity on [x, y]
    let mpi = &cores.memory_processor_interconnects[0];
    assert_eq!(mpi.map.source_dims.len(), 2);
    assert_eq!(mpi.map.target_dims.len(), 2);
    assert_eq!(mpi.map.source_dims[0].name, "x");
    assert_eq!(mpi.map.source_dims[1].name, "y");

    // 2 lane types x 8x8 = 128 total processing elements
    assert_eq!(cores.total_processing_elements(), Some(128));

    // Visualize scaled architecture
    let cores_dot = architecture_to_dot(&cores);
    fs::write("cores_8x8_architecture.dot", &cores_dot).expect("Failed to write DOT file");
    println!("Generated cores_8x8_architecture.dot");

    let cores_expanded_dot = architecture_to_dot_expanded(&cores);
    fs::write("cores_8x8_expanded.dot", &cores_expanded_dot).expect("Failed to write DOT file");
    println!("Generated cores_8x8_expanded.dot");
}
