module @processor {
  func.func @pipeline_matmul_f16(%A: memref<?x?xf16>, %B: memref<?x?xf16>, %C: memref<?x?xf16>) {
    %M = loom.sym @M : index
    %K = loom.sym @K : index
    %N = loom.sym @N : index
    loom.bind_shape %A, [%M, %K] : memref<?x?xf16>
    loom.bind_shape %B, [%K, %N] : memref<?x?xf16>
    loom.bind_shape %C, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %A, @input_0 : memref<?x?xf16>
    loom.bind_mem %B, @input_1 : memref<?x?xf16>
    loom.bind_mem %C, @output_0 : memref<?x?xf16>
    linalg.matmul ins(%A, %B : memref<?x?xf16>, memref<?x?xf16>) outs(%C : memref<?x?xf16>)
    return
  }
}
