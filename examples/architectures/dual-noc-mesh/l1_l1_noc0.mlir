module @l1_l1_noc0 {
  func.func @l1_gather(
      %src: memref<?x?xf16>,
      %dst: memref<?x?x?xf16>
  ) {
    %B = loom.sym @B : index
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    %effective_bandwidth = loom.sym @effective_bandwidth : index
    %gather_x = loom.sym @gather_x : index
    %gather_y = loom.sym @gather_y : index
    loom.bind_shape %src, [%M, %N] : memref<?x?xf16>
    loom.bind_shape %dst, [%B, %M, %N] : memref<?x?x?xf16>
    loom.bind_mem %src, @array_L1 : memref<?x?xf16>
    loom.bind_mem %dst, @array_L1 : memref<?x?x?xf16>
    loom.gather %src, %dst src_mem_space @array_L1 dst_mem_space @array_L1 area: [%gather_x, %gather_y] : memref<?x?xf16> to memref<?x?x?xf16>
    return
  }
}
