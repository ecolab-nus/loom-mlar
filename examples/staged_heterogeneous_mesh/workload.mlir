module attributes {loom.tile_k = {is_reduction = true, upper_bound = 64 : index}, loom.tile_m = {is_reduction = false, upper_bound = 128 : index}, loom.tile_n = {is_reduction = false, upper_bound = 128 : index}} {
  func.func @staged_matmul(%A: memref<512x256xf16>, %B: memref<256x512xf16, 1>, %C: memref<512x512xf32, 2>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f32
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    %tile_m = loom.sym @tile_m {upper_bound = 128 : index} : index
    %tile_n = loom.sym @tile_n {upper_bound = 128 : index} : index
    %tile_k = loom.sym @tile_k {upper_bound = 64 : index} : index
    %m_trips = arith.ceildivui %c512, %tile_m : index
    %n_trips = arith.ceildivui %c512, %tile_n : index
    affine.parallel (%mi, %ni) = (0, 0) to (symbol(%m_trips), symbol(%n_trips)) {
      %m_offset = arith.muli %mi, %tile_m : index
      %n_offset = arith.muli %ni, %tile_n : index
      %C_tile = loom.subview %C[%m_offset, %n_offset] [%tile_m, %tile_n] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x512xf32, 2> to memref<?x?xf32, strided<[512, 1], offset: ?>, 2>
      %C_tensor = loom.init_tensor %C_tile[%tile_m, %tile_n] : memref<?x?xf32, strided<[512, 1], offset: ?>, 2> -> tensor<?x?xf32, {local_mem_kind = 2 : i64}>
      %init = linalg.fill ins(%zero : f32) outs(%C_tensor : tensor<?x?xf32, {local_mem_kind = 2 : i64}>) -> tensor<?x?xf32, {local_mem_kind = 2 : i64}>
      %stage_a = loom.alloc [%tile_m, %tile_k] on @STAGE : memref<?x?xf16, 2>
      %stage_a_owned = loom.semaphore_take %stage_a : memref<?x?xf16, 2> -> memref<?x?xf16, 2>
      %stage_b = loom.alloc [%tile_k, %tile_n] on @STAGE : memref<?x?xf16, 2>
      %stage_b_owned = loom.semaphore_take %stage_b : memref<?x?xf16, 2> -> memref<?x?xf16, 2>
      %k_trips = arith.ceildivui %c256, %tile_k : index
      %result = scf.for %ki = %c0 to %k_trips step %c1 iter_args(%acc = %init) -> (tensor<?x?xf32, {local_mem_kind = 2 : i64}>) {
        %k_offset = arith.muli %ki, %tile_k : index
        %A_tile = loom.subview %A[%m_offset, %k_offset] [%tile_m, %tile_k] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<512x256xf16> to memref<?x?xf16, strided<[256, 1], offset: ?>>
        loom.copy %A_tile, %stage_a_owned src_mem_space @mem_GCRAM_instance dst_mem_space @mem_STAGE_instance, area : [1, 1] : memref<?x?xf16, strided<[256, 1], offset: ?>> to memref<?x?xf16, 2>
        %A_tensor = loom.bufferize_to_tensor %stage_a_owned[%tile_m, %tile_k] : memref<?x?xf16, 2> -> tensor<?x?xf16, {local_mem_kind = 2 : i64}>
        %B_tile = loom.subview %B[%k_offset, %n_offset] [%tile_k, %tile_n] [1, 1], reuse : [seq = false, spat = false, temp = false] : memref<256x512xf16, 1> to memref<?x?xf16, strided<[512, 1], offset: ?>, 1>
        loom.copy %B_tile, %stage_b_owned src_mem_space @mem_RRAM_instance dst_mem_space @mem_STAGE_instance, area : [1, 1] : memref<?x?xf16, strided<[512, 1], offset: ?>, 1> to memref<?x?xf16, 2>
        %B_tensor = loom.bufferize_to_tensor %stage_b_owned[%tile_k, %tile_n] : memref<?x?xf16, 2> -> tensor<?x?xf16, {local_mem_kind = 2 : i64}>
        %next = linalg.matmul ins(%A_tensor, %B_tensor : tensor<?x?xf16, {local_mem_kind = 2 : i64}>, tensor<?x?xf16, {local_mem_kind = 2 : i64}>) outs(%acc : tensor<?x?xf32, {local_mem_kind = 2 : i64}>) -> tensor<?x?xf32, {local_mem_kind = 2 : i64}>
        scf.yield %next : tensor<?x?xf32, {local_mem_kind = 2 : i64}>
      } {loom.iter_type = #loom.iter_type<sequential>}
      loom.semaphore_give %stage_b_owned : memref<?x?xf16, 2>
      loom.semaphore_give %stage_a_owned : memref<?x?xf16, 2>
    }
    return
  }
}
