module @processor {
  func.func @matmul_f16(%lhs: memref<?x?xf16>, %rhs: memref<?x?xf16>, %out: memref<?x?xf16>) {
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    %K = loom.sym @K : index
    loom.bind_shape %lhs, [%M, %K] : memref<?x?xf16>
    loom.bind_mem %lhs, @lhs : memref<?x?xf16>
    loom.bind_shape %rhs, [%K, %N] : memref<?x?xf16>
    loom.bind_mem %rhs, @rhs : memref<?x?xf16>
    loom.bind_shape %out, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %out, @result : memref<?x?xf16>
    linalg.matmul ins(%lhs, %rhs : memref<?x?xf16>, memref<?x?xf16>) outs(%out : memref<?x?xf16>)
    return
  }
  func.func @batch_matmul_f16(%lhs: memref<?x?x?xf16>, %rhs: memref<?x?x?xf16>, %out: memref<?x?x?xf16>) {
    %B = loom.sym @B : index
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    %K = loom.sym @K : index
    loom.bind_shape %lhs, [%B, %M, %K] : memref<?x?x?xf16>
    loom.bind_mem %lhs, @lhs : memref<?x?x?xf16>
    loom.bind_shape %rhs, [%B, %K, %N] : memref<?x?x?xf16>
    loom.bind_mem %rhs, @rhs : memref<?x?x?xf16>
    loom.bind_shape %out, [%B, %M, %N] : memref<?x?x?xf16>
    loom.bind_mem %out, @result : memref<?x?x?xf16>
    linalg.batch_matmul ins(%lhs, %rhs : memref<?x?x?xf16>, memref<?x?x?xf16>) outs(%out : memref<?x?x?xf16>)
    return
  }
  func.func @vec_vsum_f16(%input: memref<?x?xf16>, %out: memref<?xf16>) {
    %P = loom.sym @P : index
    %R = loom.sym @R : index
    loom.bind_shape %input, [%P, %R] : memref<?x?xf16>
    loom.bind_mem %input, @lhs : memref<?x?xf16>
    loom.bind_shape %out, [%P] : memref<?xf16>
    loom.bind_mem %out, @result : memref<?xf16>
    linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]
    }
    ins(%input : memref<?x?xf16>)
    outs(%out : memref<?xf16>) {
    ^bb0(%value: f16, %accumulator: f16):
      %result = arith.addf %value, %accumulator : f16
      linalg.yield %result : f16
    }
    return
  }
  func.func @vec_vmax_f16(%input: memref<?x?xf16>, %out: memref<?xf16>) {
    %P = loom.sym @P : index
    %R = loom.sym @R : index
    loom.bind_shape %input, [%P, %R] : memref<?x?xf16>
    loom.bind_mem %input, @lhs : memref<?x?xf16>
    loom.bind_shape %out, [%P] : memref<?xf16>
    loom.bind_mem %out, @result : memref<?xf16>
    linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]
    }
    ins(%input : memref<?x?xf16>)
    outs(%out : memref<?xf16>) {
    ^bb0(%value: f16, %accumulator: f16):
      %result = arith.maximumf %value, %accumulator : f16
      linalg.yield %result : f16
    }
    return
  }
  func.func @vec_max1_f16(%a: memref<?xf16>, %b: memref<?xf16>, %out: memref<?xf16>) {
    %L = loom.sym @L : index
    loom.bind_shape %a, [%L] : memref<?xf16>
    loom.bind_shape %b, [%L] : memref<?xf16>
    loom.bind_shape %out, [%L] : memref<?xf16>
    loom.bind_mem %a, @lhs : memref<?xf16>
    loom.bind_mem %b, @rhs : memref<?xf16>
    loom.bind_mem %out, @result : memref<?xf16>
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
  func.func @elementwise_add_f16(%lhs: memref<?x?xf16>, %rhs: memref<?x?xf16>, %out: memref<?x?xf16>) {
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    loom.bind_shape %lhs, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %lhs, @lhs : memref<?x?xf16>
    loom.bind_shape %rhs, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %rhs, @rhs : memref<?x?xf16>
    loom.bind_shape %out, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %out, @result : memref<?x?xf16>
    linalg.add ins(%lhs, %rhs : memref<?x?xf16>, memref<?x?xf16>) outs(%out : memref<?x?xf16>)
    return
  }
  func.func @elementwise_mul_f16(%lhs: memref<?x?xf16>, %rhs: memref<?x?xf16>, %out: memref<?x?xf16>) {
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    loom.bind_shape %lhs, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %lhs, @lhs : memref<?x?xf16>
    loom.bind_shape %rhs, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %rhs, @rhs : memref<?x?xf16>
    loom.bind_shape %out, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %out, @result : memref<?x?xf16>
    linalg.mul ins(%lhs, %rhs : memref<?x?xf16>, memref<?x?xf16>) outs(%out : memref<?x?xf16>)
    return
  }
}
