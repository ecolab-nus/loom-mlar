// Matrix lane compute semantics — fp16 matrix multiplication.
//
// C[M, N] = A[M, K] * B[K, N]
//
// @M, @N, @K are symbolic variables retrieved via `loom.sym`, then `loom.bind_shape`
// ties each memref dimension to those symbols.
// Memrefs are bound to @L1 via `loom.bind_mem`.
//
// This is the canonical matmul expressed in linalg-on-memref style.

module @matrix_lane {

func.func @matmul_f16(
    %A: memref<?x?xf16>,
    %B: memref<?x?xf16>,
    %C: memref<?x?xf16>
) {
  %M = loom.sym @M : index
  %N = loom.sym @N : index
  %K = loom.sym @K : index
  loom.bind_shape %A, [%M, %K] : memref<?x?xf16>
  loom.bind_mem %A, @L1
  loom.bind_shape %B, [%K, %N] : memref<?x?xf16>
  loom.bind_mem %B, @L1
  loom.bind_shape %C, [%M, %N] : memref<?x?xf16>
  loom.bind_mem %C, @L1
  linalg.matmul
      ins(%A, %B : memref<?x?xf16>, memref<?x?xf16>)
      outs(%C : memref<?x?xf16>)
  return
}

// C[B, M, N] = A[B, M, K] * B[B, K, N]  (batched matmul)
func.func @batch_matmul_f16(
    %A: memref<?x?x?xf16>,
    %Bmat: memref<?x?x?xf16>,
    %C: memref<?x?x?xf16>
) {
  %B = loom.sym @B : index
  %M = loom.sym @M : index
  %N = loom.sym @N : index
  %K = loom.sym @K : index
  loom.bind_shape %A, [%B, %M, %K] : memref<?x?x?xf16>
  loom.bind_mem %A, @L1
  loom.bind_shape %Bmat, [%B, %K, %N] : memref<?x?x?xf16>
  loom.bind_mem %Bmat, @L1
  loom.bind_shape %C, [%B, %M, %N] : memref<?x?x?xf16>
  loom.bind_mem %C, @L1
  linalg.batch_matmul
      ins(%A, %Bmat : memref<?x?x?xf16>, memref<?x?x?xf16>)
      outs(%C : memref<?x?x?xf16>)
  return
}

}
