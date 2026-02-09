use mlar_rust::*;
use mlar_rust::lane::{MatMulLane, VecLane};

#[test]
fn test_core_architecture() {
    // === Dimensions ===
    let dim_bank = Dimension::new("nbank", 16);
    let dim_x = Dimension::new("x", 8);
    let dim_y = Dimension::new("y", 8);

    // === Memory hierarchy ===
    // Single L1: 16 banks, each bank has 1024 x 128B blocks = 16KB per bank, 256KB total
    let l1 = MemRegion::bank(Bank {
        block_size: Size::int(128),
        num_blocks: Size::int(1024),
    })
    .scale([&dim_bank]); // 16 banks => one L1

    // All L1s across the 8x8 core grid
    let all_l1s = l1.clone().scale([&dim_x, &dim_y]); // 64 L1s, one per core

    // === Compute lanes, scaled across 8x8 cores ===
    let matrix_lane = FunctionalLane::new(
        "matrix_lane",
        vec![&l1, &l1],
        vec![&l1],
        MatMulLane,
    );
    let matrix_lane_set = matrix_lane.scale([&dim_x, &dim_y]); // 64 matrix lanes

    let vector_lane = FunctionalLane::new(
        "vector_lane",
        vec![&l1, &l1],
        vec![&l1],
        VecLane,
    );
    let vector_lane_set = vector_lane.scale([&dim_x, &dim_y]); // 64 vector lanes

    // === Interconnects: each core's L1 connects to that core's lanes ===
    // Identity map [x, y] -> [x, y]: core (x,y) reads from L1 (x,y)
    let identity_map = AffineMap::new(
        vec![dim_x.clone(), dim_y.clone()], // source dims (from all_l1s)
        vec![dim_x.clone(), dim_y.clone()], // target dims (from lane set)
        vec![
            AffineExpr::dim(&dim_x),
            AffineExpr::dim(&dim_y),
        ],
    );

    let l1_to_matrix = MemoryProcessorInterconnect::builder("l1_to_matrix_lane")
        .source(&all_l1s)
        .target(&matrix_lane_set)
        .affine_map(identity_map.clone())
        .bandwidth(512) // 512 bits/cycle
        .build();

    let l1_to_vector = MemoryProcessorInterconnect::builder("l1_to_vector_lane")
        .source(&all_l1s)
        .target(&vector_lane_set)
        .affine_map(identity_map)
        .bandwidth(128) // 128 bits/cycle
        .build();

    // === Assemble the architecture ===
    let arch = Architecture {
        name: "8x8 Core Grid".to_string(),
        dimensions: vec![dim_x.clone(), dim_y.clone(), dim_bank.clone()],
        processor_sets: vec![matrix_lane_set, vector_lane_set],
        processor_aggregations: Vec::new(),
        memory_regions: vec![all_l1s],
        memory_interconnects: Vec::new(),
        memory_processor_interconnects: vec![l1_to_matrix, l1_to_vector],
        interconnects: Vec::new(),
    };

    // Verify structure
    assert_eq!(arch.name, "8x8 Core Grid");
    assert_eq!(arch.dimensions.len(), 3); // x, y, nbank
    assert_eq!(arch.processor_sets.len(), 2); // matrix + vector lane sets
    assert_eq!(arch.memory_regions.len(), 1); // all_l1s (hierarchical)
    assert_eq!(arch.memory_processor_interconnects.len(), 2); // one per lane type
    // 2 lane types x 8x8 cores = 128 total processing elements
    assert_eq!(arch.total_processing_elements(), Some(128));
}
