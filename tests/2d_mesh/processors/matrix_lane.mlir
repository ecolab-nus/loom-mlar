module @processor {
  func.func @vec_vsum_f16(%arg0: memref<?x?xf16, 1>, %arg1: memref<?xf16, 1>) {
    %P = loom.sym @P : index
    %R = loom.sym @R : index
    loom.bind_shape %arg0, [%P, %R] : memref<?x?xf16, 1>
    loom.bind_shape %arg1, [%P] : memref<?xf16, 1>
    loom.bind_mem %arg0, @data : memref<?x?xf16, 1>
    loom.bind_mem %arg1, @result : memref<?xf16, 1>
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : memref<?x?xf16, 1>) outs(%arg1 : memref<?xf16, 1>) {
    ^bb0(%in: f16, %out: f16):
      %native = arith.addf %in, %out : f16
      linalg.yield %native : f16
    }
    return
  }
  func.func @vec_vmax_f16(%arg0: memref<?x?xf16, 1>, %arg1: memref<?xf16, 1>) {
    %P = loom.sym @P : index
    %R = loom.sym @R : index
    loom.bind_shape %arg0, [%P, %R] : memref<?x?xf16, 1>
    loom.bind_shape %arg1, [%P] : memref<?xf16, 1>
    loom.bind_mem %arg0, @data : memref<?x?xf16, 1>
    loom.bind_mem %arg1, @result : memref<?xf16, 1>
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : memref<?x?xf16, 1>) outs(%arg1 : memref<?xf16, 1>) {
    ^bb0(%in: f16, %out: f16):
      %native = arith.maximumf %in, %out : f16
      linalg.yield %native : f16
    }
    return
  }
  func.func @vec_max1_f16(%arg0: memref<?xf16, 1>, %arg1: memref<?xf16, 1>, %arg2: memref<?xf16, 1>) {
    %L = loom.sym @L : index
    loom.bind_shape %arg0, [%L] : memref<?xf16, 1>
    loom.bind_shape %arg1, [%L] : memref<?xf16, 1>
    loom.bind_shape %arg2, [%L] : memref<?xf16, 1>
    loom.bind_mem %arg0, @data : memref<?xf16, 1>
    loom.bind_mem %arg1, @data : memref<?xf16, 1>
    loom.bind_mem %arg2, @result : memref<?xf16, 1>
    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<?xf16, 1>, memref<?xf16, 1>) outs(%arg2 : memref<?xf16, 1>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %native = arith.cmpf ogt, %in, %in_0 : f16
      %native_1 = arith.select %native, %in, %in_0 : f16
      linalg.yield %native_1 : f16
    }
    return
  }
  func.func @elementwise_add_f16(%arg0: memref<?x?xf16, 1>, %arg1: memref<?x?xf16, 1>, %arg2: memref<?x?xf16, 1>) {
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    loom.bind_shape %arg0, [%M, %N] : memref<?x?xf16, 1>
    loom.bind_shape %arg1, [%M, %N] : memref<?x?xf16, 1>
    loom.bind_shape %arg2, [%M, %N] : memref<?x?xf16, 1>
    loom.bind_mem %arg0, @data : memref<?x?xf16, 1>
    loom.bind_mem %arg1, @data : memref<?x?xf16, 1>
    loom.bind_mem %arg2, @result : memref<?x?xf16, 1>
    linalg.add ins(%arg0, %arg1 : memref<?x?xf16, 1>, memref<?x?xf16, 1>) outs(%arg2 : memref<?x?xf16, 1>)
    return
  }
  func.func @elementwise_mul_f16(%arg0: memref<?x?xf16, 1>, %arg1: memref<?x?xf16, 1>, %arg2: memref<?x?xf16, 1>) {
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    loom.bind_shape %arg0, [%M, %N] : memref<?x?xf16, 1>
    loom.bind_shape %arg1, [%M, %N] : memref<?x?xf16, 1>
    loom.bind_shape %arg2, [%M, %N] : memref<?x?xf16, 1>
    loom.bind_mem %arg0, @data : memref<?x?xf16, 1>
    loom.bind_mem %arg1, @data : memref<?x?xf16, 1>
    loom.bind_mem %arg2, @result : memref<?x?xf16, 1>
    linalg.mul ins(%arg0, %arg1 : memref<?x?xf16, 1>, memref<?x?xf16, 1>) outs(%arg2 : memref<?x?xf16, 1>)
    return
  }
}
