use mlar_rust::{lane::MatMulLane, *};

#[test]
fn test_gpu_memory_hierarchy() {
    let dram_dim = Dimension::new("dram_banks", 2);
    let l2_dim = Dimension::new("l2_banks", 4);
    let l1_dim = Dimension::new("l1_banks", 8);

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

    // L1: 8 banks, each with many small blocks totaling 64KB
    let l1_banks = MemRegion::bank(Bank::builder()
            .block_size(64)   // 64 bytes per block
            .num_blocks(1024) // 1024 blocks = 64KB total per bank
            .build())
        .scale([&l1_dim]);

    // --- DRAM <-> L2 Connection ---
    
    // DRAM -> L2 Shared Buffer
    let dram_l2_buffer = MemRegion::bank(Bank::builder()
        .block_size(256)
        .num_blocks(1)
        .build());

    // DRAM Output Aggregation (DRAM -> Buffer)
    let dram_bus_output = MemoryInterface::builder("DRAM_bus_output")
        .source(&dram_banks)
        .target(&dram_l2_buffer)
        .bandwidth(256)
        .build();

    // L2 Input Aggregation (Buffer -> L2)
    let l2_bus_input = MemoryInterface::builder("L2_bus_input")
        .source(&dram_l2_buffer)
        .target(&l2_banks)
        .bandwidth(128)
        .build();

    // --- L2 <-> L1 Connection ---

    // L2 -> L1 Shared Buffer
    let l2_l1_buffer = MemRegion::bank(Bank::builder()
        .block_size(256)  // Match L2 block size
        .num_blocks(1)
        .build());

    // L2 Output Aggregation (L2 -> Buffer)
    let l2_bus_output = MemoryInterface::builder("L2_bus_output")
        .source(&l2_banks)
        .target(&l2_l1_buffer)
        .bandwidth(128)
        .build();

    // L1 Input Aggregation (Buffer -> L1)
    let l1_bus_input = MemoryInterface::builder("L1_bus_input")
        .source(&l2_l1_buffer)
        .target(&l1_banks)
        .bandwidth(64)
        .build();

    // Create the input register file for the matrix lane
    let mat_input_reg = MemRegion::bank(Bank::builder()
        .block_size(1024)
        .num_blocks(16)
        .build());

    // create the output register file for the matrix lane
    let mat_output_reg = MemRegion::bank(Bank::builder()
        .block_size(1024)
        .num_blocks(16)
        .build());

    // The matrix lane
    let _mat_lane = FunctionalLane::new(
        "matmul_lane",
        vec![&mat_input_reg, &mat_output_reg],
        vec![&mat_output_reg],
        MatMulLane,
    );
    
    // Build architecture
    let arch = Architecture::builder("GPU")
        .dimension(dram_dim)
        .dimension(l2_dim)
        .dimension(l1_dim)
        .memory_interface(dram_bus_output)
        .memory_interface(l2_bus_input)
        .memory_interface(l2_bus_output)
        .memory_interface(l1_bus_input)
        .build();
        
    // Verify
    assert_eq!(arch.name, "GPU");
    assert_eq!(arch.memory_interfaces.len(), 4);
    
    // 1. DRAM Output Aggregation
    assert_eq!(arch.memory_interfaces[0].name, "DRAM_bus_output");
    assert_eq!(arch.memory_interfaces[0].bandwidth, 256);

    // 2. L2 Input Aggregation
    assert_eq!(arch.memory_interfaces[1].name, "L2_bus_input");
    assert_eq!(arch.memory_interfaces[1].bandwidth, 128);

    // 3. L2 Output Aggregation
    assert_eq!(arch.memory_interfaces[2].name, "L2_bus_output");
    assert_eq!(arch.memory_interfaces[2].sources.len(), 1);
    assert_eq!(arch.memory_interfaces[2].bandwidth, 128);
    
    // 4. L1 Input Aggregation
    assert_eq!(arch.memory_interfaces[3].name, "L1_bus_input");
    assert_eq!(arch.memory_interfaces[3].sources.len(), 1);
    assert_eq!(arch.memory_interfaces[3].bandwidth, 64);
    
    // In this architecture:
    // - DRAM banks (2 x Large) -> DRAM_bus_output -> dram_l2_buffer
    // - dram_l2_buffer -> L2_bus_input -> L2 banks
    // - L2 banks (4 x 1MB) -> L2_bus_output -> l2_l1_buffer
    // - l2_l1_buffer -> L1_bus_input -> L1 banks
}
