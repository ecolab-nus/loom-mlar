// Matrix lane compute semantics — fp32 matrix multiplication.
//
// C[M, N] = A[M, K] * B[K, N]
//
// @M, @N, @K are symbolic variables retrieved via `loom.sym`, then `loom.bind`
// ties each tensor dimension to those symbols.
//
// This is the canonical matmul expressed in the linalg-on-tensors dialect.

module @matrix_lane {

func.func @matmul_f32(
    %A: tensor<?x?xf32>,
    %B: tensor<?x?xf32>,
    %C: tensor<?x?xf32>
) -> tensor<?x?xf32> {
  %M = loom.sym @M : index
  %N = loom.sym @N : index
  %K = loom.sym @K : index
  loom.bind %A, [%M, %K] : tensor<?x?xf32>
  loom.bind %B, [%K, %N] : tensor<?x?xf32>
  loom.bind %C, [%M, %N] : tensor<?x?xf32>
  %result = linalg.matmul
      ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%C : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}

}
