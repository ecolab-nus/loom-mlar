module @processor {
  func.func @matmul_f16(%arg0: memref<?x?xf16, 1>, %arg1: memref<?x?xf16, 1>, %arg2: memref<?x?xf16, 1>) {
    %M = loom.sym @M : index
    %K = loom.sym @K : index
    %N = loom.sym @N : index
    loom.bind_shape %arg0, [%M, %K] : memref<?x?xf16, 1>
    loom.bind_shape %arg1, [%K, %N] : memref<?x?xf16, 1>
    loom.bind_shape %arg2, [%M, %N] : memref<?x?xf16, 1>
    loom.bind_mem %arg0, @lhs : memref<?x?xf16, 1>
    loom.bind_mem %arg1, @rhs : memref<?x?xf16, 1>
    loom.bind_mem %arg2, @result : memref<?x?xf16, 1>
    linalg.matmul ins(%arg0, %arg1 : memref<?x?xf16, 1>, memref<?x?xf16, 1>) outs(%arg2 : memref<?x?xf16, 1>)
    return
  }
  func.func @batch_matmul_f16(%arg0: memref<?x?x?xf16, 1>, %arg1: memref<?x?x?xf16, 1>, %arg2: memref<?x?x?xf16, 1>) {
    %B = loom.sym @B : index
    %M = loom.sym @M : index
    %K = loom.sym @K : index
    %N = loom.sym @N : index
    loom.bind_shape %arg0, [%B, %M, %K] : memref<?x?x?xf16, 1>
    loom.bind_shape %arg1, [%B, %K, %N] : memref<?x?x?xf16, 1>
    loom.bind_shape %arg2, [%B, %M, %N] : memref<?x?x?xf16, 1>
    loom.bind_mem %arg0, @lhs : memref<?x?x?xf16, 1>
    loom.bind_mem %arg1, @rhs : memref<?x?x?xf16, 1>
    loom.bind_mem %arg2, @result : memref<?x?x?xf16, 1>
    linalg.batch_matmul ins(%arg0, %arg1 : memref<?x?x?xf16, 1>, memref<?x?x?xf16, 1>) outs(%arg2 : memref<?x?x?xf16, 1>)
    return
  }
}
