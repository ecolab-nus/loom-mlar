// Matrix lane compute semantics — fp32 matrix multiplication.
//
// C[M, N] = A[M, K] * B[K, N]
//
// This is the canonical matmul expressed in the linalg-on-tensors dialect.

module @matrix_lane {

func.func @matmul_f32(
    %A: tensor<?x?xf32>,
    %B: tensor<?x?xf32>,
    %C: tensor<?x?xf32>
) -> tensor<?x?xf32> {
  %result = linalg.matmul
      ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%C : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %result : tensor<?x?xf32>
}

}
