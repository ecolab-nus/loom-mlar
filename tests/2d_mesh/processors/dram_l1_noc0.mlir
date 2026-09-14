module @processor {
  func.func @dram_to_l1_S_f16(%dram_src: memref<?x?xf16>, %l1_dst: memref<?x?xf16>) {
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    %effective_bandwidth = loom.sym @effective_bandwidth : index
    loom.bind_shape %dram_src, [%M, %N] : memref<?x?xf16>
    loom.bind_shape %l1_dst, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %dram_src, @input_0 : memref<?x?xf16>
    loom.bind_mem %l1_dst, @output_0 : memref<?x?xf16>
    loom.copy %dram_src, %l1_dst src_mem_space @input_0 dst_mem_space @output_0, area: [1, 1] : memref<?x?xf16> to memref<?x?xf16>
    return
  }
  func.func @dram_to_l1_S_bcst(%dram_src: memref<?x?xf16>, %l1_dst: memref<?x?xf16>) {
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    %effective_bandwidth = loom.sym @effective_bandwidth : index
    %bcst_x = loom.sym @bcst_x : index
    %bcst_y = loom.sym @bcst_y : index
    loom.bind_shape %dram_src, [%M, %N] : memref<?x?xf16>
    loom.bind_shape %l1_dst, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %dram_src, @input_0 : memref<?x?xf16>
    loom.bind_mem %l1_dst, @output_0 : memref<?x?xf16>
    loom.copy %dram_src, %l1_dst src_mem_space @input_0 dst_mem_space @output_0, area: [%bcst_x, %bcst_y] : memref<?x?xf16> to memref<?x?xf16>
    return
  }
  func.func @dram_to_l1_R_f16(%dram_src: memref<?x?xf16>, %l1_dst: memref<?x?xf16>) {
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    %effective_bandwidth = loom.sym @effective_bandwidth : index
    loom.bind_shape %dram_src, [%M, %N] : memref<?x?xf16>
    loom.bind_shape %l1_dst, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %dram_src, @input_0 : memref<?x?xf16>
    loom.bind_mem %l1_dst, @output_0 : memref<?x?xf16>
    loom.copy %dram_src, %l1_dst src_mem_space @input_0 dst_mem_space @output_0 : 1, area: [1, 1] : memref<?x?xf16> to memref<?x?xf16>
    return
  }
  func.func @dram_to_l1_R_bcst(%dram_src: memref<?x?xf16>, %l1_dst: memref<?x?xf16>) {
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    %effective_bandwidth = loom.sym @effective_bandwidth : index
    %bcst_x = loom.sym @bcst_x : index
    %bcst_y = loom.sym @bcst_y : index
    loom.bind_shape %dram_src, [%M, %N] : memref<?x?xf16>
    loom.bind_shape %l1_dst, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %dram_src, @input_0 : memref<?x?xf16>
    loom.bind_mem %l1_dst, @output_0 : memref<?x?xf16>
    loom.copy %dram_src, %l1_dst src_mem_space @input_0 dst_mem_space @output_0 : 1, area: [%bcst_x, %bcst_y] : memref<?x?xf16> to memref<?x?xf16>
    return
  }
}
