module @processor {
  func.func @matmul_staged_f16_f32(%A: memref<?x?xf16>, %B: memref<?x?xf16>, %C: memref<?x?xf32>) {
    %M = loom.sym @M : index
    %K = loom.sym @K : index
    %N = loom.sym @N : index
    loom.bind_shape %A, [%M, %K] : memref<?x?xf16>
    loom.bind_shape %B, [%K, %N] : memref<?x?xf16>
    loom.bind_shape %C, [%M, %N] : memref<?x?xf32>
    loom.bind_mem %A, @stage_a : memref<?x?xf16>
    loom.bind_mem %B, @stage_b : memref<?x?xf16>
    loom.bind_mem %C, @result : memref<?x?xf32>
    linalg.matmul ins(%A, %B : memref<?x?xf16>, memref<?x?xf16>) outs(%C : memref<?x?xf32>)
    return
  }
}
