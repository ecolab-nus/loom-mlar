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
    // cf.assert is used to validate preconditions for the performance model
    // ============================================================
    
    // Matrix lane processor for large MxNxK matmul
    // Performance model valid only when M >= 256 and N >= 256
    // Latency: M*N*K / 64 cycles (streaming at 64 MACs/cycle)
    func.func @matmul_lane(%M: index, %N: index, %K: index,
                           %a: memref<?x?xf32>, %b: memref<?x?xf32>, %c: memref<?x?xf32>) -> index {
        // Preconditions: performance model valid for large matrices
        %c256 = arith.constant 256 : index
        %m_ok = arith.cmpi sge, %M, %c256 : index
        cf.assert %m_ok, "matmul_lane requires M >= 256"
        %n_ok = arith.cmpi sge, %N, %c256 : index
        cf.assert %n_ok, "matmul_lane requires N >= 256"
        
        // Computation description
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
    // Performance model valid only when N >= 1024
    // Latency: N / 32 cycles (streaming at 32 elements/cycle)
    func.func @vec_lane(%N: index,
                        %a: memref<?xf32>, %b: memref<?xf32>, %c: memref<?xf32>) -> index {
        // Precondition: performance model valid for long vectors
        %c1024 = arith.constant 1024 : index
        %n_ok = arith.cmpi sge, %N, %c1024 : index
        cf.assert %n_ok, "vec_lane requires N >= 1024"
        
        // Computation description
        linalg.add ins(%a, %b : memref<?xf32>, memref<?xf32>)
                   outs(%c : memref<?xf32>)
        
        // Compute latency: N / 32
        %c32 = arith.constant 32 : index
        %latency = arith.divui %N, %c32 : index
        return %latency : index
    }

    // Horizontal NoC links (ring in x dimension)
    func.func @horizontal_links(%x: index, %y: index,
                               %src: memref<?x?xf32>, %dst: memref<?x?xf32>) -> index {
        // the latency is simply total size divided by bandwidth (assumed 32)
        %shape1 = memref.dim %src,0 : memref<?x?xf32>
        %shape2 = memref.dim %src,1 : memref<?x?xf32>
        %total_size = arith.muli %shape1, %shape2 : index
        %c32 = arith.constant 32 : index
        %latency = arith.divui %total_size, %c32 : index
        return %latency : index
    }
    
    // ============================================================
    // Hardware architecture description
    // ============================================================
    
    // Fixed functional units (synchronous, small tiles)
    %x = mlar.dim "x", 8
    %y = mlar.dim "y", 8
    %mat_unit = mlar.fu @matmul_32x32 <%x, %y>
    %vec_unit = mlar.fu @vec_add_32 <%x, %y>
    
    // Streaming lane processors (dynamic shapes, with preconditions)
    %mat_lane = mlar.lane @matmul_lane <%x, %y>
    %vec_lane = mlar.lane @vec_lane <%x, %y>
    
    // L1 memory per core
    %L1 = mlar.memory "L1" 65536 16 <%x, %y> : memref<8x8x65536x16xf32>
    
    // Horizontal NoC links (ring in x dimension)
    // The affine maps means automatical indexing of the first two dimensions
    %noc_h = mlar.interconnects @horizontal_spec <%x, %y> {map = affine_map<(d0, d1) -> ((d0 + 1) mod 8, d1)>}
    
    // Vertical NoC links (ring in y dimension)
    %noc_v = mlar.interconnects @vertical_spec <%x, %y> {map = affine_map<(d0, d1) -> (d0, (d1 + 1) mod 8)>}
    
    // DRAM resources
    %dram_idx = mlar.dim "d", 4
    %drams = mlar.memory "DRAM" 34359738368 288 <%dram_idx> : memref<4x34359738368x288xf32>
    
    // L1 to DRAM interconnect
    %to_dram = mlar.interconnects @dram_spec <%x, %y> {map = affine_map<(d0, d1) -> (d0 ceildiv 4 + 2 * (d1 ceildiv 4))>}
}
