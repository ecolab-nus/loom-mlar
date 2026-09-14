module @processor {
  func.func @l1_gather(%src: memref<?x?xf16>, %dst: memref<?x?x?xf16>) {
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    %B = loom.sym @B : index
    %effective_bandwidth = loom.sym @effective_bandwidth : index
    %gather_x = loom.sym @gather_x : index
    %gather_y = loom.sym @gather_y : index
    loom.bind_shape %src, [%M, %N] : memref<?x?xf16>
    loom.bind_shape %dst, [%B, %M, %N] : memref<?x?x?xf16>
    loom.bind_mem %src, @input_0 : memref<?x?xf16>
    loom.bind_mem %dst, @output_0 : memref<?x?x?xf16>
    loom.gather %src, %dst src_mem_space @input_0 dst_mem_space @output_0 area: [%gather_x, %gather_y] : memref<?x?xf16> to memref<?x?x?xf16>
    return
  }
}
