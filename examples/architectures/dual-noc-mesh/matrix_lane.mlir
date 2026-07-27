module @matrix_lane {
  func.func @matmul_SS_f16(
      %lhs: memref<?x?xf16>,
      %rhs: memref<?x?xf16>,
      %out: memref<?x?xf16>
  ) {
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    %K = loom.sym @K : index
    loom.bind_shape %lhs, [%M, %K] : memref<?x?xf16>
    loom.bind_shape %rhs, [%K, %N] : memref<?x?xf16>
    loom.bind_shape %out, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %lhs, @L1 : memref<?x?xf16>
    loom.bind_mem %rhs, @L1 : memref<?x?xf16>
    loom.bind_mem %out, @L1 : memref<?x?xf16>
    linalg.matmul
        ins(%lhs, %rhs : memref<?x?xf16>, memref<?x?xf16>)
        outs(%out : memref<?x?xf16>)
    return
  }

  func.func @matmul_SR_f16(
      %lhs: memref<?x?xf16>,
      %rhs: memref<?x?xf16, 1>,
      %out: memref<?x?xf16>
  ) {
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    %K = loom.sym @K : index
    loom.bind_shape %lhs, [%M, %K] : memref<?x?xf16>
    loom.bind_shape %rhs, [%K, %N] : memref<?x?xf16, 1>
    loom.bind_shape %out, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %lhs, @L1 : memref<?x?xf16>
    loom.bind_mem %rhs, @L1 : memref<?x?xf16, 1>
    loom.bind_mem %out, @L1 : memref<?x?xf16>
    linalg.matmul
        ins(%lhs, %rhs : memref<?x?xf16>, memref<?x?xf16, 1>)
        outs(%out : memref<?x?xf16>)
    return
  }

  func.func @matmul_RS_f16(
      %lhs: memref<?x?xf16, 1>,
      %rhs: memref<?x?xf16>,
      %out: memref<?x?xf16>
  ) {
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    %K = loom.sym @K : index
    loom.bind_shape %lhs, [%M, %K] : memref<?x?xf16, 1>
    loom.bind_shape %rhs, [%K, %N] : memref<?x?xf16>
    loom.bind_shape %out, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %lhs, @L1 : memref<?x?xf16, 1>
    loom.bind_mem %rhs, @L1 : memref<?x?xf16>
    loom.bind_mem %out, @L1 : memref<?x?xf16>
    linalg.matmul
        ins(%lhs, %rhs : memref<?x?xf16, 1>, memref<?x?xf16>)
        outs(%out : memref<?x?xf16>)
    return
  }

  func.func @matmul_RR_f16(
      %lhs: memref<?x?xf16, 1>,
      %rhs: memref<?x?xf16, 1>,
      %out: memref<?x?xf16>
  ) {
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    %K = loom.sym @K : index
    loom.bind_shape %lhs, [%M, %K] : memref<?x?xf16, 1>
    loom.bind_shape %rhs, [%K, %N] : memref<?x?xf16, 1>
    loom.bind_shape %out, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %lhs, @L1 : memref<?x?xf16, 1>
    loom.bind_mem %rhs, @L1 : memref<?x?xf16, 1>
    loom.bind_mem %out, @L1 : memref<?x?xf16>
    linalg.matmul
        ins(%lhs, %rhs : memref<?x?xf16, 1>, memref<?x?xf16, 1>)
        outs(%out : memref<?x?xf16>)
    return
  }

  func.func @elementwise_add_f16(
      %lhs: memref<?x?xf16>,
      %rhs: memref<?x?xf16>,
      %out: memref<?x?xf16>
  ) {
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    loom.bind_shape %lhs, [%M, %N] : memref<?x?xf16>
    loom.bind_shape %rhs, [%M, %N] : memref<?x?xf16>
    loom.bind_shape %out, [%M, %N] : memref<?x?xf16>
    loom.bind_mem %lhs, @L1 : memref<?x?xf16>
    loom.bind_mem %rhs, @L1 : memref<?x?xf16>
    loom.bind_mem %out, @L1 : memref<?x?xf16>
    linalg.add
        ins(%lhs, %rhs : memref<?x?xf16>, memref<?x?xf16>)
        outs(%out : memref<?x?xf16>)
    return
  }
}
