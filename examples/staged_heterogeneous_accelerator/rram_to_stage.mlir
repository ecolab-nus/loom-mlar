module @processor {
  func.func @load_rram_f16(%src: memref<?x?xf16>, %dst: memref<?x?xf16, 2>) {
    %K = loom.sym @K : index
    %N = loom.sym @N : index
    loom.bind_shape %src, [%K, %N] : memref<?x?xf16>
    loom.bind_shape %dst, [%K, %N] : memref<?x?xf16, 2>
    loom.bind_mem %src, @rram : memref<?x?xf16>
    loom.bind_mem %dst, @stage : memref<?x?xf16, 2>
    loom.copy %src, %dst src_mem_space @rram dst_mem_space @stage, area: [1, 1] : memref<?x?xf16> to memref<?x?xf16, 2>
    return
  }
  func.func @load_rram_f16_broadcast(%src: memref<?x?xf16>, %dst: memref<?x?xf16, 2>) {
    %K = loom.sym @K : index
    %N = loom.sym @N : index
    %bcst_x = loom.sym @bcst_x : index
    %bcst_y = loom.sym @bcst_y : index
    loom.bind_shape %src, [%K, %N] : memref<?x?xf16>
    loom.bind_shape %dst, [%K, %N] : memref<?x?xf16, 2>
    loom.bind_mem %src, @rram : memref<?x?xf16>
    loom.bind_mem %dst, @stage : memref<?x?xf16, 2>
    loom.copy %src, %dst src_mem_space @rram dst_mem_space @stage, area: [%bcst_x, %bcst_y] : memref<?x?xf16> to memref<?x?xf16, 2>
    return
  }
}
