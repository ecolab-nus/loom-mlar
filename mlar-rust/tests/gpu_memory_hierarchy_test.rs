use mlar_rust::AffineMapTemplate;
use mlar_rust::lane::MatMulLane;
use mlar_rust::*;
use std::fs;

fn example_gpu_memory_hierarchy() -> Architecture {
    let dram_dim = Dimension::new("dram_dim", 4);   // Shared by DRAM and L2
    let warp_dim = Dimension::new("warp_dim", 32);  // Shared by L1, RF, and MatLane

    // DRAM: 4 banks, large capacity
    let dram_banks = MemRegion::bank(Bank {
        block_size: Size::int(256), // 256 bytes per block transfer
        num_blocks: Size::sym("DRAM_SIZE"),
    })
    .scale([&dram_dim]);

    // L2: 4 banks, each with many small blocks totaling 1MB
    let l2_banks = MemRegion::bank(Bank {
        block_size: Size::int(256),  // 256 bytes per block
        num_blocks: Size::int(4096), // 4096 blocks = 1MB total per bank
    })
    .scale([&dram_dim]);

    // L1: 32 banks, each with many small blocks totaling 64KB
    let l1_banks = MemRegion::bank(Bank {
        block_size: Size::int(64),   // 64 bytes per block
        num_blocks: Size::int(1024), // 1024 blocks = 64KB total per bank
    })
    .scale([&warp_dim]);

    // RF: same number of banks as L1, smaller size per bank
    let rf_banks = MemRegion::bank(Bank {
        block_size: Size::int(32),   // smaller block size than L1
        num_blocks: Size::int(128),  // 128 blocks = 4KB total per bank
    })
    .scale([&warp_dim]);

    // --- DRAM <-> L2 Connection (1-to-1) ---
    let dram_to_l2_map = AffineMapTemplate::parse("[dram_dim] -> [dram_dim]: (dram_dim)")
        .expect("invalid affine map")
        .bind([&dram_dim])
        .expect("failed to bind affine map");

    let dram_to_l2 = MemoryInterconnects::builder("DRAM_to_L2")
        .source(&dram_banks)
        .target(&l2_banks)
        .affine_map(dram_to_l2_map)
        .bandwidth(256)
        .build();

    // --- L2 <-> L1 Connection ---
    // Each L2 bank connects to 8 L1 banks (32/4 = 8)
    let l2_to_l1_map = AffineMapTemplate::parse("[dram_dim] -> [warp_dim]: (dram_dim * 8)")
        .expect("invalid affine map")
        .bind([&dram_dim, &warp_dim])
        .expect("failed to bind affine map"); // L2[i] -> L1[i*8..i*8+7]

    let l2_to_l1 = MemoryInterconnects::builder("L2_to_L1")
        .source(&l2_banks)
        .target(&l1_banks)
        .affine_map(l2_to_l1_map)
        .bandwidth(128)
        .build();


    // --- L1 <-> RF Connection (1-to-1) ---
    let l1_to_rf_map = AffineMapTemplate::parse("[warp_dim] -> [warp_dim]: (warp_dim)")
        .expect("invalid affine map")
        .bind([&warp_dim])
        .expect("failed to bind affine map");

    let l1_to_rf = MemoryInterconnects::builder("L1_to_RF")
        .source(&l1_banks)
        .target(&rf_banks)
        .affine_map(l1_to_rf_map)
        .bandwidth(64)
        .build();

    // Matrix lane per RF bank
    let mat_lane = FunctionalLane::new(
        "matmul_lane",
        vec![&rf_banks, &rf_banks],
        vec![&rf_banks],
        MatMulLane,
    );

    let mat_lane_set = mat_lane.scale([&warp_dim]);

    // RF -> Matrix Lane Connection (1-to-1)
    let rf_to_mat_map = AffineMapTemplate::parse("[warp_dim] -> [warp_dim]: (warp_dim)")
        .expect("invalid affine map")
        .bind([&warp_dim])
        .expect("failed to bind affine map");

    let rf_to_mat = MemoryProcessorInterconnect::builder("RF_to_MatLane")
        .source(&rf_banks)
        .target(&mat_lane_set)
        .affine_map(rf_to_mat_map)
        .bandwidth(64)
        .build();

    // Build architecture
    let arch = Architecture {
        name: "GPU".to_string(),
        dimensions: vec![dram_dim, warp_dim],
        processor_sets: vec![mat_lane_set],
        processor_aggregations: Vec::new(),
        memory_regions: Vec::new(),
        memory_interconnects: vec![dram_to_l2, l2_to_l1, l1_to_rf],
        memory_processor_interconnects: vec![rf_to_mat],
        interconnects: Vec::new(),
    };
        
    // Verify
    assert_eq!(arch.name, "GPU");
    assert_eq!(arch.memory_interconnects.len(), 3);
    assert_eq!(arch.memory_processor_interconnects.len(), 1);
    
    // 1. DRAM -> L2 Mapping
    assert_eq!(arch.memory_interconnects[0].name, "DRAM_to_L2");
    assert_eq!(arch.memory_interconnects[0].bandwidth, 256);

    // 2. L2 -> L1 Mapping
    assert_eq!(arch.memory_interconnects[1].name, "L2_to_L1");
    assert_eq!(arch.memory_interconnects[1].bandwidth, 128);

    // 3. L1 -> RF Mapping
    assert_eq!(arch.memory_interconnects[2].name, "L1_to_RF");
    assert_eq!(arch.memory_interconnects[2].bandwidth, 64);

    // 4. RF -> MatLane Mapping
    assert_eq!(arch.memory_processor_interconnects[0].name, "RF_to_MatLane");
    assert_eq!(arch.memory_processor_interconnects[0].bandwidth, 64);

    return arch;
    // In this architecture:
    // - DRAM banks [dram_dim:4] -> DRAM_to_L2 (1:1) -> L2 banks [dram_dim:4]
    // - L2 banks [dram_dim:4] -> L2_to_L1 (1:8) -> L1 banks [warp_dim:32]
    // - L1 banks [warp_dim:32] -> L1_to_RF (1:1) -> RF banks [warp_dim:32]
    // - RF banks [warp_dim:32] -> RF_to_MatLane (1:1) -> MatLane [warp_dim:32]
}

#[test]
fn test_gpu_memory_hierarchy() {
    example_gpu_memory_hierarchy();
}

/// Generate DOT visualization for the GPU memory hierarchy architecture
#[test]
fn test_gpu_memory_hierarchy_visualization() {
    let arch = example_gpu_memory_hierarchy();

    // Generate summary DOT (one block per memory level)
    let dot = architecture_to_dot(&arch);
    fs::write("gpu_memory_hierarchy.dot", &dot).expect("Failed to write DOT file");
    println!("Generated gpu_memory_hierarchy.dot:");
    println!("{}", dot);

    // Generate expanded DOT (all instances with affine-mapped edges)
    let expanded_dot = architecture_to_dot_expanded(&arch);
    fs::write("gpu_memory_hierarchy_expanded.dot", &expanded_dot).expect("Failed to write expanded DOT file");
    println!("\nGenerated gpu_memory_hierarchy_expanded.dot:");
    println!("{}", expanded_dot);
}

/// Print instructions for viewing the generated DOT files
#[test]
fn print_visualization_instructions() {
    println!("\n=== How to view the generated DOT files ===\n");
    println!("1. Install GraphViz: https://graphviz.org/download/");
    println!("   - macOS: brew install graphviz");
    println!("   - Ubuntu: sudo apt-get install graphviz");
    println!("   - Windows: choco install graphviz\n");
    println!("2. Convert DOT to PNG/SVG/PDF:");
    println!("   dot -Tpng gpu_memory_hierarchy.dot -o gpu_memory_hierarchy.png");
    println!("   dot -Tsvg 2d_mesh.dot -o 2d_mesh.svg");
    println!("   dot -Tpdf gpu_memory_hierarchy_simplified.dot -o gpu_memory_hierarchy_simplified.pdf\n");
    println!("3. Or use online viewers:");
    println!("   - https://dreampuf.github.io/GraphvizOnline/");
    println!("   - https://edotor.net/\n");
}
