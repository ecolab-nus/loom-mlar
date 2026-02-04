use mlar_rust::*;

#[test]
fn test_gpu_memory_hierarchy() {
    // L2: 4 banks, each with many small blocks totaling 1MB
    let l2_banks = MemRegion::indexed(
        vec![Dimension::new("l2_banks", 4)],
        MemRegion::bank(Bank::builder()
            .block_size(256)  // 256 bytes per block
            .num_blocks(4096)  // 4096 blocks = 1MB total per bank
            .build()),
    );
    
    // L1: 8 banks, each with many small blocks totaling 64KB
    let l1_banks = MemRegion::indexed(
        vec![Dimension::new("l1_banks", 8)],
        MemRegion::bank(Bank::builder()
            .block_size(64)   // 64 bytes per block
            .num_blocks(1024) // 1024 blocks = 64KB total per bank
            .build()),
    );
    
    // Shared buffer - single block for transferring data between L2 and L1
    // L2 outputs to it, L1 inputs from it
    let shared_buffer = MemRegion::bank(Bank::builder()
        .block_size(256)  // Match L2 block size for efficient transfers
        .num_blocks(1)    // Single block buffer
        .build());
    
    // L2 output aggregation (for reading)
    // L2 banks output data to the shared buffer
    let l2_bus_output = MemoryAggregation::builder("L2_bus_output")
        .source(l2_banks.clone())
        .target(shared_buffer.clone())
        .bandwidth(128)  // bytes/cycle
        .build();
    
    // L1 input aggregation (for writing)
    // L1 banks input data from the shared buffer
    let l1_bus_input = MemoryAggregation::builder("L1_bus_input")
        .source(shared_buffer.clone())
        .target(l1_banks.clone())
        .bandwidth(64)  // bytes/cycle
        .build();
    
    // Build architecture
    let arch = Architecture::builder("GPU")
        .dimension(Dimension::new("l1_banks", 8))
        .dimension(Dimension::new("l2_banks", 4))
        .memory_aggregation(l2_bus_output)
        .memory_aggregation(l1_bus_input)
        .build();
        
    // Verify
    assert_eq!(arch.name, "GPU");
    assert_eq!(arch.memory_aggregations.len(), 2);
    
    // The L2 output aggregation allows reading from L2
    assert_eq!(arch.memory_aggregations[0].name, "L2_bus_output");
    assert_eq!(arch.memory_aggregations[0].sources.len(), 1);
    assert_eq!(arch.memory_aggregations[0].bandwidth, 128);
    
    // The L1 input aggregation allows writing to L1
    assert_eq!(arch.memory_aggregations[1].name, "L1_bus_input");
    assert_eq!(arch.memory_aggregations[1].sources.len(), 1);
    assert_eq!(arch.memory_aggregations[1].bandwidth, 64);
    
    // In this architecture:
    // - L2 banks (4 x 1MB) output data to shared buffer (256 bytes)
    // - L1 banks (8 x 64KB) input data from shared buffer
    // - The shared buffer acts as the connection point between L2 and L1
    // - Any L1 bank can read from any L2 bank via the shared buffer
}
