module @l1_l2_dma {
  func.func @writeback_l2(%src: memref<?xf16>, %dst: memref<?xf16>) {
    %L = loom.sym @L : index
    loom.bind_shape %src, [%L] : memref<?xf16>
    loom.bind_mem %src, @array_L1 : memref<?xf16>
    loom.bind_shape %dst, [%L] : memref<?xf16>
    loom.bind_mem %dst, @L2 : memref<?xf16>
    loom.copy %src, %dst src_mem_space @array_L1 dst_mem_space @L2, area: [1] : memref<?xf16> to memref<?xf16>
    return
  }
}
