//! Tests for architecture visualization that generate DOT files.

use mlar_rust::lane::{MatMulLane, VecLane};
use mlar_rust::interconnect::AffineExpr;
use mlar_rust::visualization::ArchVisualizer;
use mlar_rust::*;
use std::fs;

/// Generate DOT visualization for the GPU memory hierarchy architecture
#[test]
fn test_gpu_memory_hierarchy_visualization() {
    // DRAM: 2 banks, large capacity
    let dram_banks = MemRegion::bank(
        Bank::builder()
            .block_size(256_usize)
            .num_blocks(Size::symbolic("DRAM_SIZE"))
            .build(),
    )
    .scale(vec![Dimension::new("dram_banks", 2)]);

    // L2: 4 banks, each with many small blocks totaling 1MB
    let l2_banks = MemRegion::bank(
        Bank::builder()
            .block_size(256_usize)
            .num_blocks(4096_usize)
            .build(),
    )
    .scale(vec![Dimension::new("l2_banks", 4)]);

    // L1: 8 banks, each with many small blocks totaling 64KB
    let l1_banks = MemRegion::bank(
        Bank::builder()
            .block_size(64_usize)
            .num_blocks(1024_usize)
            .build(),
    )
    .scale(vec![Dimension::new("l1_banks", 8)]);

    // --- DRAM <-> L2 Connection ---

    // DRAM -> L2 Shared Buffer
    let dram_l2_buffer = MemRegion::bank(
        Bank::builder()
            .block_size(256_usize)
            .num_blocks(1_usize)
            .build(),
    );

    // DRAM Output Aggregation (DRAM -> Buffer)
    let dram_bus_output = MemoryAggregation::builder("DRAM_bus_output")
        .source(dram_banks.clone())
        .target(dram_l2_buffer.clone())
        .bandwidth(256)
        .build();

    // L2 Input Aggregation (Buffer -> L2)
    let l2_bus_input = MemoryAggregation::builder("L2_bus_input")
        .source(dram_l2_buffer.clone())
        .target(l2_banks.clone())
        .bandwidth(128)
        .build();

    // --- L2 <-> L1 Connection ---

    // L2 -> L1 Shared Buffer
    let l2_l1_buffer = MemRegion::bank(
        Bank::builder()
            .block_size(256_usize)
            .num_blocks(1_usize)
            .build(),
    );

    // L2 Output Aggregation (L2 -> Buffer)
    let l2_bus_output = MemoryAggregation::builder("L2_bus_output")
        .source(l2_banks.clone())
        .target(l2_l1_buffer.clone())
        .bandwidth(128)
        .build();

    // L1 Input Aggregation (Buffer -> L1)
    let l1_bus_input = MemoryAggregation::builder("L1_bus_input")
        .source(l2_l1_buffer.clone())
        .target(l1_banks.clone())
        .bandwidth(64)
        .build();

    // Build architecture
    let arch = Architecture::builder("GPU_Memory_Hierarchy")
        .dimension(Dimension::new("dram_banks", 2))
        .dimension(Dimension::new("l2_banks", 4))
        .dimension(Dimension::new("l1_banks", 8))
        .memory_aggregation(dram_bus_output)
        .memory_aggregation(l2_bus_input)
        .memory_aggregation(l2_bus_output)
        .memory_aggregation(l1_bus_input)
        .build();

    // Generate DOT
    let dot = architecture_to_dot(&arch);

    // Write to file
    fs::write("gpu_memory_hierarchy.dot", &dot).expect("Failed to write DOT file");

    println!("Generated gpu_memory_hierarchy.dot:");
    println!("{}", dot);

    // Verify the DOT contains expected elements
    assert!(dot.contains("GPU_Memory_Hierarchy"));
    assert!(dot.contains("DRAM_bus_output"));
    assert!(dot.contains("L2_bus_input"));
    assert!(dot.contains("256 B/cycle"));
}

/// Generate DOT visualization for the 2D Mesh architecture
#[test]
fn test_2d_mesh_visualization() {
    // Create grid dimensions
    let dim_x = Dimension::new("x", 8);
    let dim_y = Dimension::new("y", 8);
    let dim_d = Dimension::new("d", 4);

    // Define L1 memory region: indexed by [x, y], each has 64KB blocks
    let l1_region = MemRegion::bank(
        Bank::builder()
            .block_size(65536_usize)
            .num_blocks(1_usize)
            .build(),
    )
    .scale(vec![dim_x.clone(), dim_y.clone()]);

    // Create functional units
    let mat_fu = FunctionalUnit::builder("matmul_32x32")
        .input_region(l1_region.clone())
        .input_region(l1_region.clone())
        .output_region(l1_region.clone())
        .latency(8)
        .build();

    let vec_fu = FunctionalUnit::builder("vec_add_32")
        .input_region(l1_region.clone())
        .input_region(l1_region.clone())
        .output_region(l1_region.clone())
        .latency(1)
        .build();

    // Create lanes
    let mat_lane = FunctionalLane::new(
        "matmul_lane",
        vec![l1_region.clone(), l1_region.clone()],
        vec![l1_region.clone()],
        MatMulLane,
    );

    let vec_lane = FunctionalLane::new(
        "vec_lane",
        vec![l1_region.clone(), l1_region.clone()],
        vec![l1_region.clone()],
        VecLane,
    );

    // Create ProcessorSets
    let mat_fu_set = mat_fu.scale(vec![dim_x.clone(), dim_y.clone()]);
    let vec_fu_set = vec_fu.scale(vec![dim_x.clone(), dim_y.clone()]);
    let mat_lane_set = mat_lane.scale(vec![dim_x.clone(), dim_y.clone()]);
    let vec_lane_set = vec_lane.scale(vec![dim_x.clone(), dim_y.clone()]);

    // Create interconnects with affine maps
    let noc_h_map = AffineMap::new(
        2,
        vec![
            AffineExpr::modulo(
                AffineExpr::add(AffineExpr::dim(0), AffineExpr::constant(1)),
                AffineExpr::constant(8),
            ),
            AffineExpr::dim(1),
        ],
    );

    let noc_h = Interconnect::builder("horizontal_noc")
        .grid(vec![dim_x.clone(), dim_y.clone()])
        .affine_map(noc_h_map)
        .bandwidth(32)
        .build();

    let noc_v_map = AffineMap::new(
        2,
        vec![
            AffineExpr::dim(0),
            AffineExpr::modulo(
                AffineExpr::add(AffineExpr::dim(1), AffineExpr::constant(1)),
                AffineExpr::constant(8),
            ),
        ],
    );

    let noc_v = Interconnect::builder("vertical_noc")
        .grid(vec![dim_x.clone(), dim_y.clone()])
        .affine_map(noc_v_map)
        .bandwidth(32)
        .build();

    let dram_map = AffineMap::new(
        2,
        vec![AffineExpr::add(
            AffineExpr::ceildiv(AffineExpr::dim(0), AffineExpr::constant(4)),
            AffineExpr::mul(
                AffineExpr::constant(2),
                AffineExpr::ceildiv(AffineExpr::dim(1), AffineExpr::constant(4)),
            ),
        )],
    );

    let to_dram = Interconnect::builder("to_dram")
        .grid(vec![dim_x.clone(), dim_y.clone()])
        .affine_map(dram_map)
        .bandwidth(64)
        .build();

    // Build the architecture
    let arch = Architecture::builder("2D_Mesh")
        .dimension(dim_x)
        .dimension(dim_y)
        .dimension(dim_d)
        .processor_set(mat_fu_set)
        .processor_set(vec_fu_set)
        .processor_set(mat_lane_set)
        .processor_set(vec_lane_set)
        .interconnect(noc_h)
        .interconnect(noc_v)
        .interconnect(to_dram)
        .build();

    // Generate DOT
    let dot = architecture_to_dot(&arch);

    // Write to file
    fs::write("2d_mesh.dot", &dot).expect("Failed to write DOT file");

    println!("Generated 2d_mesh.dot:");
    println!("{}", dot);

    // Verify the DOT contains expected elements
    assert!(dot.contains("2D_Mesh"));
    assert!(dot.contains("matmul_32x32"));
    assert!(dot.contains("vec_add_32"));
    assert!(dot.contains("horizontal_noc"));
    assert!(dot.contains("vertical_noc"));
    assert!(dot.contains("to_dram"));
}

/// Generate a simplified hierarchical view of the GPU memory hierarchy
#[test]
fn test_gpu_memory_hierarchy_simplified() {
    // Create a manually crafted graph showing the hierarchy more clearly
    let _viz = ArchVisualizer::new();

    // Add memory level nodes manually for a cleaner view
    let dram = MemRegion::bank(
        Bank::builder()
            .block_size(256_usize)
            .num_blocks(Size::symbolic("DRAM_SIZE"))
            .build(),
    )
    .scale(vec![Dimension::new("banks", 2)]);

    let l2 = MemRegion::bank(
        Bank::builder()
            .block_size(256_usize)
            .num_blocks(4096_usize)
            .build(),
    )
    .scale(vec![Dimension::new("banks", 4)]);

    let l1 = MemRegion::bank(
        Bank::builder()
            .block_size(64_usize)
            .num_blocks(1024_usize)
            .build(),
    )
    .scale(vec![Dimension::new("banks", 8)]);

    let reg = MemRegion::bank(
        Bank::builder()
            .block_size(1024_usize)
            .num_blocks(16_usize)
            .build(),
    );

    // Build the memory aggregations
    let dram_to_l2 = MemoryAggregation::builder("DRAM_to_L2")
        .source(dram)
        .target(l2.clone())
        .bandwidth(256)
        .build();

    let l2_to_l1 = MemoryAggregation::builder("L2_to_L1")
        .source(l2)
        .target(l1.clone())
        .bandwidth(128)
        .build();

    let l1_to_reg = MemoryAggregation::builder("L1_to_REG")
        .source(l1)
        .target(reg)
        .bandwidth(64)
        .build();

    // Generate DOT using the memory hierarchy function
    let dot = memory_hierarchy_to_dot(
        "GPU_Memory_Hierarchy_Simplified",
        &[dram_to_l2, l2_to_l1, l1_to_reg],
    );

    // Write to file
    fs::write("gpu_memory_hierarchy_simplified.dot", &dot).expect("Failed to write DOT file");

    println!("Generated gpu_memory_hierarchy_simplified.dot:");
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
