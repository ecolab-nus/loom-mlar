module @processor {
  func.func @dram_to_l1_S_f16(%arg0: memref<?x?xf16>, %arg1: memref<?x?xf16, 1>) {
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    %effective_bandwidth = loom.sym @effective_bandwidth : index
    loom.bind_shape %arg0, [%M, %N] : memref<?x?xf16>
    loom.bind_shape %arg1, [%M, %N] : memref<?x?xf16, 1>
    loom.bind_mem %arg0, @src : memref<?x?xf16>
    loom.bind_mem %arg1, @dst_s : memref<?x?xf16, 1>
    loom.copy %arg0, %arg1 src_mem_space @src : 0 dst_mem_space @dst_s : 1, area : [1, 1] : memref<?x?xf16> to memref<?x?xf16, 1>
    return
  }
  func.func @dram_to_l1_S_bcst(%arg0: memref<?x?xf16>, %arg1: memref<?x?xf16, 1>) {
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    %effective_bandwidth = loom.sym @effective_bandwidth : index
    %bcst_x = loom.sym @bcst_x : index
    %bcst_y = loom.sym @bcst_y : index
    loom.bind_shape %arg0, [%M, %N] : memref<?x?xf16>
    loom.bind_shape %arg1, [%M, %N] : memref<?x?xf16, 1>
    loom.bind_mem %arg0, @src : memref<?x?xf16>
    loom.bind_mem %arg1, @dst_s : memref<?x?xf16, 1>
    loom.copy %arg0, %arg1 src_mem_space @src : 0 dst_mem_space @dst_s : 1, area : [%bcst_x, %bcst_y] : memref<?x?xf16> to memref<?x?xf16, 1>
    return
  }
  func.func @dram_to_l1_R_f16(%arg0: memref<?x?xf16>, %arg1: memref<?x?xf16>) {
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    %effective_bandwidth = loom.sym @effective_bandwidth : index
    loom.bind_shape %arg0, [%M, %N] : memref<?x?xf16>
    loom.bind_shape %arg1, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %arg0, @src : memref<?x?xf16>
    loom.bind_mem %arg1, @dst_r : memref<?x?xf16>
    loom.copy %arg0, %arg1 src_mem_space @src : 0 dst_mem_space @dst_r : 0, area : [1, 1] : memref<?x?xf16> to memref<?x?xf16>
    return
  }
  func.func @dram_to_l1_R_bcst(%arg0: memref<?x?xf16>, %arg1: memref<?x?xf16>) {
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    %effective_bandwidth = loom.sym @effective_bandwidth : index
    %bcst_x = loom.sym @bcst_x : index
    %bcst_y = loom.sym @bcst_y : index
    loom.bind_shape %arg0, [%M, %N] : memref<?x?xf16>
    loom.bind_shape %arg1, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %arg0, @src : memref<?x?xf16>
    loom.bind_mem %arg1, @dst_r : memref<?x?xf16>
    loom.copy %arg0, %arg1 src_mem_space @src : 0 dst_mem_space @dst_r : 0, area : [%bcst_x, %bcst_y] : memref<?x?xf16> to memref<?x?xf16>
    return
  }
}
