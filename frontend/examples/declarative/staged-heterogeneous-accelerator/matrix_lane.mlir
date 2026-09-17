module @processor {
  func.func @matmul_staged_f16_f32(%A: memref<?x?xf16, 2>, %B: memref<?x?xf16, 2>, %C: memref<?x?xf32, 2>) {
    %M = loom.sym @M : index
    %K = loom.sym @K : index
    %N = loom.sym @N : index
    loom.bind_shape %A, [%M, %K] : memref<?x?xf16, 2>
    loom.bind_shape %B, [%K, %N] : memref<?x?xf16, 2>
    loom.bind_shape %C, [%M, %N] : memref<?x?xf32, 2>
    loom.bind_mem %A, @stage_a : memref<?x?xf16, 2>
    loom.bind_mem %B, @stage_b : memref<?x?xf16, 2>
    loom.bind_mem %C, @result : memref<?x?xf32, 2>
    linalg.matmul ins(%A, %B : memref<?x?xf16, 2>, memref<?x?xf16, 2>) outs(%C : memref<?x?xf32, 2>)
    return
  }
}
