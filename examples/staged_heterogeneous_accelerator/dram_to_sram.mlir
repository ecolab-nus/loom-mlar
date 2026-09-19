module @processor {
  func.func @dram_to_sram_f16(%src: memref<?x?xf16>, %dst: memref<?x?xf16, 1>) {
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    loom.bind_shape %src, [%M, %N] : memref<?x?xf16>
    loom.bind_shape %dst, [%M, %N] : memref<?x?xf16, 1>
    loom.bind_mem %src, @src : memref<?x?xf16>
    loom.bind_mem %dst, @dst : memref<?x?xf16, 1>
    loom.copy %src, %dst src_mem_space @src dst_mem_space @dst, area: [1, 1] : memref<?x?xf16> to memref<?x?xf16, 1>
    return
  }
}
