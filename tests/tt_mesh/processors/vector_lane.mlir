module @processor {
  func.func @vec_max_f16(%arg0: memref<?xf16, 1>, %arg1: memref<?xf16, 1>, %arg2: memref<?xf16, 1>) {
    %L = loom.sym @L : index
    loom.bind_shape %arg0, [%L] : memref<?xf16, 1>
    loom.bind_shape %arg1, [%L] : memref<?xf16, 1>
    loom.bind_shape %arg2, [%L] : memref<?xf16, 1>
    loom.bind_mem %arg0, @data : memref<?xf16, 1>
    loom.bind_mem %arg1, @data : memref<?xf16, 1>
    loom.bind_mem %arg2, @result : memref<?xf16, 1>
    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<?xf16, 1>, memref<?xf16, 1>) outs(%arg2 : memref<?xf16, 1>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %native = arith.maximumf %in, %in_0 : f16
      linalg.yield %native : f16
    }
    return
  }
  func.func @vec_exp_f16(%arg0: memref<?xf16, 1>, %arg1: memref<?xf16, 1>) {
    %L = loom.sym @L : index
    loom.bind_shape %arg0, [%L] : memref<?xf16, 1>
    loom.bind_shape %arg1, [%L] : memref<?xf16, 1>
    loom.bind_mem %arg0, @data : memref<?xf16, 1>
    loom.bind_mem %arg1, @result : memref<?xf16, 1>
    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0 : memref<?xf16, 1>) outs(%arg1 : memref<?xf16, 1>) {
    ^bb0(%in: f16, %out: f16):
      %native = math.exp %in : f16
      linalg.yield %native : f16
    }
    return
  }
  func.func @vec_sum_f16(%arg0: memref<?xf16, 1>, %arg1: memref<f16, 1>) {
    %L = loom.sym @L : index
    loom.bind_shape %arg0, [%L] : memref<?xf16, 1>
    loom.bind_mem %arg0, @data : memref<?xf16, 1>
    loom.bind_mem %arg1, @result : memref<f16, 1>
    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>], iterator_types = ["reduction"]} ins(%arg0 : memref<?xf16, 1>) outs(%arg1 : memref<f16, 1>) {
    ^bb0(%in: f16, %out: f16):
      %native = arith.addf %in, %out : f16
      linalg.yield %native : f16
    }
    return
  }
  func.func @vec_add_f16(%arg0: memref<?xf16, 1>, %arg1: memref<?xf16, 1>, %arg2: memref<?xf16, 1>) {
    %L = loom.sym @L : index
    loom.bind_shape %arg0, [%L] : memref<?xf16, 1>
    loom.bind_shape %arg1, [%L] : memref<?xf16, 1>
    loom.bind_shape %arg2, [%L] : memref<?xf16, 1>
    loom.bind_mem %arg0, @data : memref<?xf16, 1>
    loom.bind_mem %arg1, @data : memref<?xf16, 1>
    loom.bind_mem %arg2, @result : memref<?xf16, 1>
    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<?xf16, 1>, memref<?xf16, 1>) outs(%arg2 : memref<?xf16, 1>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %native = arith.addf %in, %in_0 : f16
      linalg.yield %native : f16
    }
    return
  }
  func.func @vec_mul_f16(%arg0: memref<?xf16, 1>, %arg1: memref<?xf16, 1>, %arg2: memref<?xf16, 1>) {
    %L = loom.sym @L : index
    loom.bind_shape %arg0, [%L] : memref<?xf16, 1>
    loom.bind_shape %arg1, [%L] : memref<?xf16, 1>
    loom.bind_shape %arg2, [%L] : memref<?xf16, 1>
    loom.bind_mem %arg0, @data : memref<?xf16, 1>
    loom.bind_mem %arg1, @data : memref<?xf16, 1>
    loom.bind_mem %arg2, @result : memref<?xf16, 1>
    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<?xf16, 1>, memref<?xf16, 1>) outs(%arg2 : memref<?xf16, 1>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %native = arith.mulf %in, %in_0 : f16
      linalg.yield %native : f16
    }
    return
  }
  func.func @vec_div_f16(%arg0: memref<?xf16, 1>, %arg1: memref<?xf16, 1>, %arg2: memref<?xf16, 1>) {
    %L = loom.sym @L : index
    loom.bind_shape %arg0, [%L] : memref<?xf16, 1>
    loom.bind_shape %arg1, [%L] : memref<?xf16, 1>
    loom.bind_shape %arg2, [%L] : memref<?xf16, 1>
    loom.bind_mem %arg0, @data : memref<?xf16, 1>
    loom.bind_mem %arg1, @data : memref<?xf16, 1>
    loom.bind_mem %arg2, @result : memref<?xf16, 1>
    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<?xf16, 1>, memref<?xf16, 1>) outs(%arg2 : memref<?xf16, 1>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %native = arith.divf %in, %in_0 : f16
      linalg.yield %native : f16
    }
    return
  }
  func.func @vec_sub_f16(%arg0: memref<?xf16, 1>, %arg1: memref<?xf16, 1>, %arg2: memref<?xf16, 1>) {
    %L = loom.sym @L : index
    loom.bind_shape %arg0, [%L] : memref<?xf16, 1>
    loom.bind_shape %arg1, [%L] : memref<?xf16, 1>
    loom.bind_shape %arg2, [%L] : memref<?xf16, 1>
    loom.bind_mem %arg0, @data : memref<?xf16, 1>
    loom.bind_mem %arg1, @data : memref<?xf16, 1>
    loom.bind_mem %arg2, @result : memref<?xf16, 1>
    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<?xf16, 1>, memref<?xf16, 1>) outs(%arg2 : memref<?xf16, 1>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %native = arith.subf %in, %in_0 : f16
      linalg.yield %native : f16
    }
    return
  }
  func.func @vec_powf_f16(%arg0: memref<?xf16, 1>, %arg1: memref<?xf16, 1>, %arg2: memref<?xf16, 1>) {
    %L = loom.sym @L : index
    loom.bind_shape %arg0, [%L] : memref<?xf16, 1>
    loom.bind_shape %arg1, [%L] : memref<?xf16, 1>
    loom.bind_shape %arg2, [%L] : memref<?xf16, 1>
    loom.bind_mem %arg0, @data : memref<?xf16, 1>
    loom.bind_mem %arg1, @data : memref<?xf16, 1>
    loom.bind_mem %arg2, @result : memref<?xf16, 1>
    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<?xf16, 1>, memref<?xf16, 1>) outs(%arg2 : memref<?xf16, 1>) {
    ^bb0(%in: f16, %in_0: f16, %out: f16):
      %native = math.powf %in, %in_0 : f16
      linalg.yield %native : f16
    }
    return
  }
  func.func @vec_cmpf_ogt_f16(%arg0: memref<?xf16, 1>, %arg1: memref<?xf16, 1>, %arg2: memref<?xi1, 1>) {
    %L = loom.sym @L : index
    loom.bind_shape %arg0, [%L] : memref<?xf16, 1>
    loom.bind_shape %arg1, [%L] : memref<?xf16, 1>
    loom.bind_shape %arg2, [%L] : memref<?xi1, 1>
    loom.bind_mem %arg0, @data : memref<?xf16, 1>
    loom.bind_mem %arg1, @data : memref<?xf16, 1>
    loom.bind_mem %arg2, @result : memref<?xi1, 1>
    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<?xf16, 1>, memref<?xf16, 1>) outs(%arg2 : memref<?xi1, 1>) {
    ^bb0(%in: f16, %in_0: f16, %out: i1):
      %native = arith.cmpf ogt, %in, %in_0 : f16
      linalg.yield %native : i1
    }
    return
  }
  func.func @vec_select_f16(%arg0: memref<?xi1, 1>, %arg1: memref<?xf16, 1>, %arg2: memref<?xf16, 1>, %arg3: memref<?xf16, 1>) {
    %L = loom.sym @L : index
    loom.bind_shape %arg0, [%L] : memref<?xi1, 1>
    loom.bind_shape %arg1, [%L] : memref<?xf16, 1>
    loom.bind_shape %arg2, [%L] : memref<?xf16, 1>
    loom.bind_shape %arg3, [%L] : memref<?xf16, 1>
    loom.bind_mem %arg0, @data : memref<?xi1, 1>
    loom.bind_mem %arg1, @data : memref<?xf16, 1>
    loom.bind_mem %arg2, @data : memref<?xf16, 1>
    loom.bind_mem %arg3, @result : memref<?xf16, 1>
    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg1, %arg2 : memref<?xi1, 1>, memref<?xf16, 1>, memref<?xf16, 1>) outs(%arg3 : memref<?xf16, 1>) {
    ^bb0(%in: i1, %in_0: f16, %in_1: f16, %out: f16):
      %native = arith.select %in, %in_0, %in_1 : f16
      linalg.yield %native : f16
    }
    return
  }
  func.func @vec_log_f16(%arg0: memref<?xf16, 1>, %arg1: memref<?xf16, 1>) {
    %L = loom.sym @L : index
    loom.bind_shape %arg0, [%L] : memref<?xf16, 1>
    loom.bind_shape %arg1, [%L] : memref<?xf16, 1>
    loom.bind_mem %arg0, @data : memref<?xf16, 1>
    loom.bind_mem %arg1, @result : memref<?xf16, 1>
    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0 : memref<?xf16, 1>) outs(%arg1 : memref<?xf16, 1>) {
    ^bb0(%in: f16, %out: f16):
      %native = math.log %in : f16
      linalg.yield %native : f16
    }
    return
  }
}
