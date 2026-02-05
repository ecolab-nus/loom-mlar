use mlar_rust::interconnect::AffineExpr;
use mlar_rust::lane::MatMulLane;
use mlar_rust::*;
use std::fs;

fn example_gpu_memory_hierarchy() -> Architecture {
    let dram_dim = Dimension::new("dram_banks", 4);
    let l2_dim = Dimension::new("l2_banks", 4);
    let l1_dim = Dimension::new("l1_banks", 32);

    // DRAM: 2 banks, large capacity
    let dram_banks = MemRegion::bank(Bank::builder()
            .block_size(256) // 256 bytes per block transfer
            .num_blocks(Size::symbolic("DRAM_SIZE"))
            .build())
        .scale([&dram_dim]);

    // L2: 4 banks, each with many small blocks totaling 1MB
    let l2_banks = MemRegion::bank(Bank::builder()
            .block_size(256)  // 256 bytes per block
            .num_blocks(4096)  // 4096 blocks = 1MB total per bank
            .build())
        .scale([&l2_dim]);

    // L1: 32 banks, each with many small blocks totaling 64KB
    let l1_banks = MemRegion::bank(Bank::builder()
            .block_size(64)   // 64 bytes per block
            .num_blocks(1024) // 1024 blocks = 64KB total per bank
            .build())
        .scale([&l1_dim]);

    // RF: same number of banks as L1, smaller size per bank
    let rf_banks = MemRegion::bank(Bank::builder()
            .block_size(32)   // smaller block size than L1
            .num_blocks(128)  // 128 blocks = 4KB total per bank
            .build())
        .scale([&l1_dim]);

    // --- DRAM <-> L2 Connection ---
    let dram_to_l2_map = AffineMap::builder()
        .source_dims(vec![&dram_dim])
        .target_dims(vec![&l2_dim])
        .result(AffineExpr::dim(0))
        .build();

    let dram_to_l2 = MemoryInterconnects::builder("DRAM_to_L2")
        .source(&dram_banks)
        .target(&l2_banks)
        .affine_map(dram_to_l2_map)
        .bandwidth(256)
        .build();

    // --- L2 <-> L1 Connection ---
    let l2_to_l1_map = AffineMap::builder()
        .num_dims(1)
        .result(AffineExpr::mul(
            AffineExpr::dim(0),
            AffineExpr::constant(2),
        ))
        .build();

    let l2_to_l1 = MemoryInterconnects::builder("L2_to_L1")
        .source(&l2_banks)
        .target(&l1_banks)
        .affine_map(l2_to_l1_map)
        .bandwidth(128)
        .build();


    // --- L1 <-> RF Connection ---
    let l1_to_rf_map = AffineMap::builder()
        .source_dims(vec![&l1_dim])
        .target_dims(vec![&l1_dim])
        .result(AffineExpr::dim(0))
        .build();

    let l1_to_rf = MemoryInterconnects::builder("L1_to_RF")
        .source(&l1_banks)
        .target(&rf_banks)
        .affine_map(l1_to_rf_map)
        .bandwidth(64)
        .build();
    
    // Matrix lane per L1 bank
    let mat_lane = FunctionalLane::new(
        "matmul_lane",
        vec![&rf_banks, &rf_banks],
        vec![&rf_banks],
        MatMulLane,
    );

    let mat_lane_set = mat_lane.scale(vec![l1_dim.clone()]);

    // RF -> Matrix Lane Connection (1-to-1)
    let rf_to_mat_map = AffineMap::builder()
        .source_dims(vec![&l1_dim])
        .target_dims(vec![&l1_dim])
        .result(AffineExpr::dim(0))
        .build();

    let rf_to_mat = MemoryProcessorInterconnect::builder("RF_to_MatLane")
        .source(&rf_banks)
        .target(mat_lane_set.clone())
        .affine_map(rf_to_mat_map)
        .bandwidth(64)
        .build();

    // Build architecture
    let arch = Architecture::builder("GPU")
        .dimension(dram_dim)
        .dimension(l2_dim)
        .dimension(l1_dim)
        .processor_set(mat_lane_set)
        .memory_interconnect(dram_to_l2)
        .memory_interconnect(l2_to_l1)
        .memory_interconnect(l1_to_rf)
        .memory_processor_interconnect(rf_to_mat)
        .build();
        
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
    // - DRAM banks (2 x Large) -> DRAM_to_L2 -> L2 banks
    // - L2 banks (4 x 1MB) -> L2_to_L1 -> L1 banks
    // - L1 banks (32 x 64KB) -> L1_to_RF -> RF banks (32 x 4KB)
}

#[test]
fn test_gpu_memory_hierarchy() {
    example_gpu_memory_hierarchy();
}

/// Generate DOT visualization for the GPU memory hierarchy architecture
#[test]
fn test_gpu_memory_hierarchy_visualization() {
    let arch = example_gpu_memory_hierarchy();

    // Generate DOT
    let dot = architecture_to_dot(&arch);

    // Write to file
    fs::write("gpu_memory_hierarchy.dot", &dot).expect("Failed to write DOT file");

    println!("Generated gpu_memory_hierarchy.dot:");
    println!("{}", dot);
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
