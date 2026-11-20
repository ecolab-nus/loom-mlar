module @processor {
  func.func @l1_gather(%arg0: memref<?x?xf16, 1>, %arg1: memref<?x?x?xf16, 1>) {
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    %B = loom.sym @B : index
    %effective_bandwidth = loom.sym @effective_bandwidth : index
    %gather_x = loom.sym @gather_x : index
    %gather_y = loom.sym @gather_y : index
    loom.bind_shape %arg0, [%M, %N] : memref<?x?xf16, 1>
    loom.bind_shape %arg1, [%B, %M, %N] : memref<?x?x?xf16, 1>
    loom.bind_mem %arg0, @src : memref<?x?xf16, 1>
    loom.bind_mem %arg1, @dst : memref<?x?x?xf16, 1>
    loom.gather %arg0, %arg1 src_mem_space @src dst_mem_space @dst area : [%gather_x, %gather_y] : memref<?x?xf16, 1> to memref<?x?x?xf16, 1>
    return
  }
}
