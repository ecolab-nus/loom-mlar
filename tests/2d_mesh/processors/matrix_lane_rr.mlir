module @processor {
  func.func @matmul_f16(%A: memref<?x?xf16, 1>, %B: memref<?x?xf16, 1>, %C: memref<?x?xf16>) {
    %M = loom.sym @M : index
    %K = loom.sym @K : index
    %N = loom.sym @N : index
    loom.bind_shape %A, [%M, %K] : memref<?x?xf16, 1>
    loom.bind_shape %B, [%K, %N] : memref<?x?xf16, 1>
    loom.bind_shape %C, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %A, @lhs : memref<?x?xf16, 1>
    loom.bind_mem %B, @rhs : memref<?x?xf16, 1>
    loom.bind_mem %C, @result : memref<?x?xf16>
    linalg.matmul ins(%A, %B : memref<?x?xf16, 1>, memref<?x?xf16, 1>) outs(%C : memref<?x?xf16>)
    return
  }
  func.func @batch_matmul_f16(%A: memref<?x?x?xf16, 1>, %Bmat: memref<?x?x?xf16, 1>, %C: memref<?x?x?xf16>) {
    %B = loom.sym @B : index
    %M = loom.sym @M : index
    %K = loom.sym @K : index
    %N = loom.sym @N : index
    loom.bind_shape %A, [%B, %M, %K] : memref<?x?x?xf16, 1>
    loom.bind_shape %Bmat, [%B, %K, %N] : memref<?x?x?xf16, 1>
    loom.bind_shape %C, [%B, %M, %N] : memref<?x?x?xf16>
    loom.bind_mem %A, @lhs : memref<?x?x?xf16, 1>
    loom.bind_mem %Bmat, @rhs : memref<?x?x?xf16, 1>
    loom.bind_mem %C, @result : memref<?x?x?xf16>
    linalg.batch_matmul ins(%A, %Bmat : memref<?x?x?xf16, 1>, memref<?x?x?xf16, 1>) outs(%C : memref<?x?x?xf16>)
    return
  }
}
