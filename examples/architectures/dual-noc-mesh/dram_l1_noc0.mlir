module @dram_l1_noc0 {
  func.func @dram_to_l1_S_f16(
      %src: memref<?x?xf16>,
      %dst: memref<?x?xf16>
  ) {
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    %effective_bandwidth = loom.sym @effective_bandwidth : index
    loom.bind_shape %src, [%M, %N] : memref<?x?xf16>
    loom.bind_shape %dst, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %src, @DRAM : memref<?x?xf16>
    loom.bind_mem %dst, @array_L1 : memref<?x?xf16>
    loom.copy %src, %dst src_mem_space @DRAM dst_mem_space @array_L1, area: [1, 1] : memref<?x?xf16> to memref<?x?xf16>
    return
  }

  func.func @dram_to_l1_S_bcst(
      %src: memref<?x?xf16>,
      %dst: memref<?x?xf16>
  ) {
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    %effective_bandwidth = loom.sym @effective_bandwidth : index
    %bcst_x = loom.sym @bcst_x : index
    %bcst_y = loom.sym @bcst_y : index
    loom.bind_shape %src, [%M, %N] : memref<?x?xf16>
    loom.bind_shape %dst, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %src, @DRAM : memref<?x?xf16>
    loom.bind_mem %dst, @array_L1 : memref<?x?xf16>
    loom.copy %src, %dst src_mem_space @DRAM dst_mem_space @array_L1, area: [%bcst_x, %bcst_y] : memref<?x?xf16> to memref<?x?xf16>
    return
  }

  func.func @dram_to_l1_R_f16(
      %src: memref<?x?xf16>,
      %dst: memref<?x?xf16>
  ) {
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    %effective_bandwidth = loom.sym @effective_bandwidth : index
    loom.bind_shape %src, [%M, %N] : memref<?x?xf16>
    loom.bind_shape %dst, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %src, @DRAM : memref<?x?xf16>
    loom.bind_mem %dst, @array_L1 : memref<?x?xf16>
    loom.copy %src, %dst src_mem_space @DRAM dst_mem_space @array_L1 : 1, area: [1, 1] : memref<?x?xf16> to memref<?x?xf16>
    return
  }

  func.func @dram_to_l1_R_bcst(
      %src: memref<?x?xf16>,
      %dst: memref<?x?xf16>
  ) {
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    %effective_bandwidth = loom.sym @effective_bandwidth : index
    %bcst_x = loom.sym @bcst_x : index
    %bcst_y = loom.sym @bcst_y : index
    loom.bind_shape %src, [%M, %N] : memref<?x?xf16>
    loom.bind_shape %dst, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %src, @DRAM : memref<?x?xf16>
    loom.bind_mem %dst, @array_L1 : memref<?x?xf16>
    loom.copy %src, %dst src_mem_space @DRAM dst_mem_space @array_L1 : 1, area: [%bcst_x, %bcst_y] : memref<?x?xf16> to memref<?x?xf16>
    return
  }
}
