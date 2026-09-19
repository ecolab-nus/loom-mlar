module @processor {
  func.func @load_sram_f16(%src: memref<?x?xf16, 1>, %dst: memref<?x?xf16, 2>) {
    %M = loom.sym @M : index
    %K = loom.sym @K : index
    loom.bind_shape %src, [%M, %K] : memref<?x?xf16, 1>
    loom.bind_shape %dst, [%M, %K] : memref<?x?xf16, 2>
    loom.bind_mem %src, @sram : memref<?x?xf16, 1>
    loom.bind_mem %dst, @stage : memref<?x?xf16, 2>
    loom.copy %src, %dst src_mem_space @sram dst_mem_space @stage, area: [1, 1] : memref<?x?xf16, 1> to memref<?x?xf16, 2>
    return
  }
  func.func @load_sram_f16_broadcast(%src: memref<?x?xf16, 1>, %dst: memref<?x?xf16, 2>) {
    %M = loom.sym @M : index
    %K = loom.sym @K : index
    %bcst_x = loom.sym @bcst_x : index
    %bcst_y = loom.sym @bcst_y : index
    loom.bind_shape %src, [%M, %K] : memref<?x?xf16, 1>
    loom.bind_shape %dst, [%M, %K] : memref<?x?xf16, 2>
    loom.bind_mem %src, @sram : memref<?x?xf16, 1>
    loom.bind_mem %dst, @stage : memref<?x?xf16, 2>
    loom.copy %src, %dst src_mem_space @sram dst_mem_space @stage, area: [%bcst_x, %bcst_y] : memref<?x?xf16, 1> to memref<?x?xf16, 2>
    return
  }
}
