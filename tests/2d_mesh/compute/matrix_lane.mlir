// Matrix lane compute semantics — fp32 matrix multiplication.
//
// C[M, N] = A[M, K] * B[K, N]
//
// M, N, K are symbolic function arguments (`loom.sym`), then `loom.bind`
// ties each tensor dimension to those symbols.
//
// This is the canonical matmul expressed in the linalg-on-tensors dialect.

module @matrix_lane {

func.func @matmul_f32(
    %M: loom.sym,
    %N: loom.sym,
    %K: loom.sym,
    %A: tensor<?x?xf32>,
    %B: tensor<?x?xf32>,
    %C: tensor<?x?xf32>
) -> tensor<?x?xf32> {
  loom.bind %A, [%M, %K]
  loom.bind %B, [%K, %N]
  loom.bind %C, [%M, %N]
  %result = linalg.matmul
      ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%C : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}

}
