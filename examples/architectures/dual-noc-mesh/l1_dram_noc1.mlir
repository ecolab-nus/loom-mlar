module @l1_dram_noc1 {
  func.func @l1_to_dram_f16(
      %src: memref<?x?xf16>,
      %dst: memref<?x?xf16>
  ) {
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    loom.bind_shape %src, [%M, %N] : memref<?x?xf16>
    loom.bind_shape %dst, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %src, @array_L1 : memref<?x?xf16>
    loom.bind_mem %dst, @DRAM : memref<?x?xf16>
    loom.copy %src, %dst src_mem_space @array_L1 dst_mem_space @DRAM, area: [1, 1] : memref<?x?xf16> to memref<?x?xf16>
    return
  }
}
