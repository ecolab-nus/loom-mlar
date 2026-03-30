// Matrix lane compute semantics — fp16 matrix kernels and row-wise reductions.
//
// C[M, N] = A[M, K] * B[K, N]
//
// @M, @N, @K are symbolic variables retrieved via `loom.sym`, then `loom.bind_shape`
// ties each memref dimension to those symbols.
// Memrefs are bound to @L1 via `loom.bind_mem`.
//
// This includes canonical matmul plus reduction kernels used by 2d_mesh.

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
  loom.bind_mem %A, @L1 : memref<?x?xf16>
  loom.bind_shape %B, [%K, %N] : memref<?x?xf16>
  loom.bind_mem %B, @L1 : memref<?x?xf16>
  loom.bind_shape %C, [%M, %N] : memref<?x?xf16>
  loom.bind_mem %C, @L1 : memref<?x?xf16>
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
  loom.bind_mem %A, @L1 : memref<?x?x?xf16>
  loom.bind_shape %Bmat, [%B, %K, %N] : memref<?x?x?xf16>
  loom.bind_mem %Bmat, @L1 : memref<?x?x?xf16>
  loom.bind_shape %C, [%B, %M, %N] : memref<?x?x?xf16>
  loom.bind_mem %C, @L1 : memref<?x?x?xf16>
  linalg.batch_matmul
      ins(%A, %Bmat : memref<?x?x?xf16>, memref<?x?x?xf16>)
      outs(%C : memref<?x?x?xf16>)
  return
}

// out[p] = sum(a[p, r]) for r in [0, R), for p in [0, P)
func.func @vec_vsum_f16(
  %a: memref<?x?xf16>,
  %out: memref<?xf16>
) {
  %P = loom.sym @P : index
  %R = loom.sym @R : index
  loom.bind_shape %a, [%P, %R] : memref<?x?xf16>
  loom.bind_mem %a, @L1 : memref<?x?xf16>
  loom.bind_shape %out, [%P] : memref<?xf16>
  loom.bind_mem %out, @L1 : memref<?xf16>
  linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0)>
    ],
    iterator_types = ["parallel", "reduction"]
  }
  ins(%a : memref<?x?xf16>)
  outs(%out : memref<?xf16>) {
    ^bb0(%x: f16, %acc: f16):
      %s = arith.addf %x, %acc : f16
      linalg.yield %s : f16
  }
  return
}

// out[p] = max(a[p, r]) for r in [0, R), for p in [0, P)
func.func @vec_vmax_f16(
  %a: memref<?x?xf16>,
  %out: memref<?xf16>
) {
  %P = loom.sym @P : index
  %R = loom.sym @R : index
  loom.bind_shape %a, [%P, %R] : memref<?x?xf16>
  loom.bind_mem %a, @L1 : memref<?x?xf16>
  loom.bind_shape %out, [%P] : memref<?xf16>
  loom.bind_mem %out, @L1 : memref<?xf16>
  linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0)>
    ],
    iterator_types = ["parallel", "reduction"]
  }
  ins(%a : memref<?x?xf16>)
  outs(%out : memref<?xf16>) {
    ^bb0(%x: f16, %acc: f16):
      %m = arith.maximumf %x, %acc : f16
      linalg.yield %m : f16
  }
  return
}

// out[i] = max(a[i], b[i]), for i in [0, L)
func.func @vec_max1_f16(
  %a: memref<?xf16>,
  %b: memref<?xf16>,
  %out: memref<?xf16>
) {
  %L = loom.sym @L : index
  loom.bind_shape %a, [%L] : memref<?xf16>
  loom.bind_mem %a, @L1 : memref<?xf16>
  loom.bind_shape %b, [%L] : memref<?xf16>
  loom.bind_mem %b, @L1 : memref<?xf16>
  loom.bind_shape %out, [%L] : memref<?xf16>
  loom.bind_mem %out, @L1 : memref<?xf16>
  linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  }
  ins(%a, %b : memref<?xf16>, memref<?xf16>)
  outs(%out : memref<?xf16>) {
    ^bb0(%x: f16, %y: f16, %z: f16):
      %cmp = arith.cmpf ogt, %x, %y : f16
      %sel = arith.select %cmp, %x, %y : f16
      linalg.yield %sel : f16
  }
  return
}

}
