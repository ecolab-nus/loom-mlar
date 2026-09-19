module @processor {
  func.func @l1_to_dram_f16(%l1_src: memref<?x?xf16>, %dram_dst: memref<?x?xf16>) {
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    loom.bind_shape %l1_src, [%M, %N] : memref<?x?xf16>
    loom.bind_shape %dram_dst, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %l1_src, @src : memref<?x?xf16>
    loom.bind_mem %dram_dst, @dst : memref<?x?xf16>
    loom.copy %l1_src, %dram_dst src_mem_space @src : 0 dst_mem_space @dst : 0, area: [1, 1] : memref<?x?xf16> to memref<?x?xf16>
    return
  }
}
