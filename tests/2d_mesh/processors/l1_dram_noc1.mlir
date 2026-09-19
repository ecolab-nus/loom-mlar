module @processor {
  func.func @l1_to_dram_f16(%arg0: memref<?x?xf16, 1>, %arg1: memref<?x?xf16>) {
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    loom.bind_shape %arg0, [%M, %N] : memref<?x?xf16, 1>
    loom.bind_shape %arg1, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %arg0, @src : memref<?x?xf16, 1>
    loom.bind_mem %arg1, @dst : memref<?x?xf16>
    loom.copy %arg0, %arg1 src_mem_space @src : 1 dst_mem_space @dst : 0, area : [1, 1] : memref<?x?xf16, 1> to memref<?x?xf16>
    return
  }
}
