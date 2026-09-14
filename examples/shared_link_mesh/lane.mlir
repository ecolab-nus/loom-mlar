module @processor {
  func.func @matmul(%lhs: memref<?x?xf16>, %rhs: memref<?x?xf16>, %out: memref<?x?xf16>) {
    %M = loom.sym @M : index
    %K = loom.sym @K : index
    %N = loom.sym @N : index
    loom.bind_shape %lhs, [%M, %K] : memref<?x?xf16>
    loom.bind_shape %rhs, [%K, %N] : memref<?x?xf16>
    loom.bind_shape %out, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %lhs, @input_0 : memref<?x?xf16>
    loom.bind_mem %rhs, @input_0 : memref<?x?xf16>
    loom.bind_mem %out, @output_0 : memref<?x?xf16>
    linalg.matmul ins(%lhs, %rhs : memref<?x?xf16>, memref<?x?xf16>) outs(%out : memref<?x?xf16>)
    return
  }
}
