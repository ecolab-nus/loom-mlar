// Example MLAR dialect hardware description: 2D mesh architecture
// This example is adapted from the TensTorrent Wormhole architecture

module {
    // ============================================================
    // Functional unit definitions (mlar.fu) - fixed shapes, synchronous
    // Each function returns an index representing the latency (cycles)
    // ============================================================
    
    // Matrix multiplication functional unit (32x32 tiles)
    // Latency: 8 cycles for a 32x32 matmul
    func.func @matmul_32x32(%a: memref<32x32xf32>, %b: memref<32x32xf32>, %c: memref<32x32xf32>) -> index {
        linalg.matmul ins(%a, %b : memref<32x32xf32>, memref<32x32xf32>)
                      outs(%c : memref<32x32xf32>)
        %latency = arith.constant 8 : index
        return %latency : index
    }
    
    // Vector elementwise add functional unit (32-wide vectors)
    // Latency: 1 cycle for a 32-element add
    func.func @vec_add_32(%a: memref<32xf32>, %b: memref<32xf32>, %c: memref<32xf32>) -> index {
        linalg.add ins(%a, %b : memref<32xf32>, memref<32xf32>)
                   outs(%c : memref<32xf32>)
        %latency = arith.constant 1 : index
        return %latency : index
    }
    
    // ============================================================
    // Lane definitions (mlar.lane) - dynamic shapes, streaming
    // Dimension sizes are passed as index parameters to compute latency
    // ============================================================
    
    // Matrix lane processor for arbitrary MxNxK matmul
    // Latency: M*N*K / 64 cycles (streaming at 64 MACs/cycle)
    func.func @matmul_lane(%M: index, %N: index, %K: index,
                           %a: memref<?x?xf32>, %b: memref<?x?xf32>, %c: memref<?x?xf32>) -> index {
        linalg.matmul ins(%a, %b : memref<?x?xf32>, memref<?x?xf32>)
                      outs(%c : memref<?x?xf32>)
        // Compute latency: M * N * K / 64
        %c64 = arith.constant 64 : index
        %mn = arith.muli %M, %N : index
        %mnk = arith.muli %mn, %K : index
        %latency = arith.divui %mnk, %c64 : index
        return %latency : index
    }
    
    // Vector lane processor for arbitrary-length vectors
    // Latency: N / 32 cycles (streaming at 32 elements/cycle)
    func.func @vec_lane(%N: index,
                        %a: memref<?xf32>, %b: memref<?xf32>, %c: memref<?xf32>) -> index {
        linalg.add ins(%a, %b : memref<?xf32>, memref<?xf32>)
                   outs(%c : memref<?xf32>)
        // Compute latency: N / 32
        %c32 = arith.constant 32 : index
        %latency = arith.divui %N, %c32 : index
        return %latency : index
    }
    
    // ============================================================
    // Hardware architecture description
    // ============================================================
    
    // Fixed functional units (synchronous, small tiles)
    %mat_unit = mlar.fu @matmul_32x32
    %vec_unit = mlar.fu @vec_add_32
    
    // Streaming lane processors (dynamic shapes)
    %mat_lane = mlar.lane @matmul_lane
    %vec_lane = mlar.lane @vec_lane
    
    // Scale-out description (spatial dimensions)
    %x = mlar.spatial_dim "x", 8
    %y = mlar.spatial_dim "y", 8
    
    // Core declaration with scaleout and scalein (using fixed FUs)
    %cores = mlar.core "core" {scaleout=(%x, %y) , scalein=(%mat_unit, %vec_unit, [8,1])}
    
    // L1 memory per core
    %L1 = mlar.memory "L1" {scaleout=(%x, %y) , size = 1499136, bandwidth = 15}
    
    // Core to memory mapping (1:1)
    %core_to_mem = mlar.mux %cores, %L1, {map = affine_map<(d0, d1) -> (d0, d1)>}
    
    // Horizontal NoC links (ring in x dimension)
    %noc_h = mlar.interconnects "horizontal_links" %L1 : !mlar.memory, %L1 : !mlar.memory, {map = affine_map<(d0, d1) -> ((d0 + 1) mod 8, d1)>, bandwidth = 128, spatial_dims = [@y]} : !mlar.interconnect
    
    // Vertical NoC links (ring in y dimension)
    %noc_v = mlar.interconnects "vertical_links" %L1 : !mlar.memory, %L1 : !mlar.memory, {map = affine_map<(d0, d1) -> (d0, (d1 + 1) mod 8)>, bandwidth = 128, spatial_dims = [@x]} : !mlar.interconnect
    
    // DRAM resources
    %dram_idx = mlar.spatial_dim "d", 4
    %drams = mlar.memory "DRAM" {scaleout=(%dram_idx) , size = 34359738368, bandwidth = 288}
    
    // L1 to DRAM interconnect
    %to_dram = mlar.interconnects "NoC" %L1: !mlar.memory, %drams : !mlar.memory, {map = affine_map<(d0, d1) -> (d0 ceildiv 4 + 2 * (d1 ceildiv 4))>}
}
