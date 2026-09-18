module @processor {
  func.func @vec_max_f16(%lhs: memref<?xf16>, %rhs: memref<?xf16>, %out: memref<?xf16>) {
    %L = loom.sym @L : index
    loom.bind_shape %lhs, [%L] : memref<?xf16>
    loom.bind_mem %lhs, @input : memref<?xf16>
    loom.bind_shape %rhs, [%L] : memref<?xf16>
    loom.bind_mem %rhs, @input : memref<?xf16>
    loom.bind_shape %out, [%L] : memref<?xf16>
    loom.bind_mem %out, @result : memref<?xf16>
    linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    }
    ins(%lhs, %rhs : memref<?xf16>, memref<?xf16>)
    outs(%out : memref<?xf16>) {
    ^bb0(%lhs_value: f16, %rhs_value: f16, %unused: f16):
      %result = arith.maximumf %lhs_value, %rhs_value : f16
      linalg.yield %result : f16
    }
    return
  }
  func.func @vec_exp_f16(%input: memref<?xf16>, %out: memref<?xf16>) {
    %L = loom.sym @L : index
    loom.bind_shape %input, [%L] : memref<?xf16>
    loom.bind_mem %input, @input : memref<?xf16>
    loom.bind_shape %out, [%L] : memref<?xf16>
    loom.bind_mem %out, @result : memref<?xf16>
    linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    }
    ins(%input : memref<?xf16>)
    outs(%out : memref<?xf16>) {
    ^bb0(%value: f16, %unused: f16):
      %result = math.exp %value : f16
      linalg.yield %result : f16
    }
    return
  }
  func.func @vec_sum_f16(%input: memref<?xf16>, %out: memref<f16>) {
    %L = loom.sym @L : index
    loom.bind_shape %input, [%L] : memref<?xf16>
    loom.bind_mem %input, @input : memref<?xf16>
    loom.bind_mem %out, @result : memref<f16>
    linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]
    }
    ins(%input : memref<?xf16>)
    outs(%out : memref<f16>) {
    ^bb0(%value: f16, %accumulator: f16):
      %result = arith.addf %value, %accumulator : f16
      linalg.yield %result : f16
    }
    return
  }
  func.func @vec_add_f16(%lhs: memref<?xf16>, %rhs: memref<?xf16>, %out: memref<?xf16>) {
    %L = loom.sym @L : index
    loom.bind_shape %lhs, [%L] : memref<?xf16>
    loom.bind_mem %lhs, @input : memref<?xf16>
    loom.bind_shape %rhs, [%L] : memref<?xf16>
    loom.bind_mem %rhs, @input : memref<?xf16>
    loom.bind_shape %out, [%L] : memref<?xf16>
    loom.bind_mem %out, @result : memref<?xf16>
    linalg.add ins(%lhs, %rhs : memref<?xf16>, memref<?xf16>) outs(%out : memref<?xf16>)
    return
  }
  func.func @vec_mul_f16(%lhs: memref<?xf16>, %rhs: memref<?xf16>, %out: memref<?xf16>) {
    %L = loom.sym @L : index
    loom.bind_shape %lhs, [%L] : memref<?xf16>
    loom.bind_mem %lhs, @input : memref<?xf16>
    loom.bind_shape %rhs, [%L] : memref<?xf16>
    loom.bind_mem %rhs, @input : memref<?xf16>
    loom.bind_shape %out, [%L] : memref<?xf16>
    loom.bind_mem %out, @result : memref<?xf16>
    linalg.mul ins(%lhs, %rhs : memref<?xf16>, memref<?xf16>) outs(%out : memref<?xf16>)
    return
  }
  func.func @vec_div_f16(%lhs: memref<?xf16>, %rhs: memref<?xf16>, %out: memref<?xf16>) {
    %L = loom.sym @L : index
    loom.bind_shape %lhs, [%L] : memref<?xf16>
    loom.bind_mem %lhs, @input : memref<?xf16>
    loom.bind_shape %rhs, [%L] : memref<?xf16>
    loom.bind_mem %rhs, @input : memref<?xf16>
    loom.bind_shape %out, [%L] : memref<?xf16>
    loom.bind_mem %out, @result : memref<?xf16>
    linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    }
    ins(%lhs, %rhs : memref<?xf16>, memref<?xf16>)
    outs(%out : memref<?xf16>) {
    ^bb0(%lhs_value: f16, %rhs_value: f16, %unused: f16):
      %result = arith.divf %lhs_value, %rhs_value : f16
      linalg.yield %result : f16
    }
    return
  }
  func.func @vec_sub_f16(%lhs: memref<?xf16>, %rhs: memref<?xf16>, %out: memref<?xf16>) {
    %L = loom.sym @L : index
    loom.bind_shape %lhs, [%L] : memref<?xf16>
    loom.bind_mem %lhs, @input : memref<?xf16>
    loom.bind_shape %rhs, [%L] : memref<?xf16>
    loom.bind_mem %rhs, @input : memref<?xf16>
    loom.bind_shape %out, [%L] : memref<?xf16>
    loom.bind_mem %out, @result : memref<?xf16>
    linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    }
    ins(%lhs, %rhs : memref<?xf16>, memref<?xf16>)
    outs(%out : memref<?xf16>) {
    ^bb0(%lhs_value: f16, %rhs_value: f16, %unused: f16):
      %result = arith.subf %lhs_value, %rhs_value : f16
      linalg.yield %result : f16
    }
    return
  }
  func.func @vec_powf_f16(%lhs: memref<?xf16>, %rhs: memref<?xf16>, %out: memref<?xf16>) {
    %L = loom.sym @L : index
    loom.bind_shape %lhs, [%L] : memref<?xf16>
    loom.bind_mem %lhs, @input : memref<?xf16>
    loom.bind_shape %rhs, [%L] : memref<?xf16>
    loom.bind_mem %rhs, @input : memref<?xf16>
    loom.bind_shape %out, [%L] : memref<?xf16>
    loom.bind_mem %out, @result : memref<?xf16>
    linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    }
    ins(%lhs, %rhs : memref<?xf16>, memref<?xf16>)
    outs(%out : memref<?xf16>) {
    ^bb0(%lhs_value: f16, %rhs_value: f16, %unused: f16):
      %result = math.powf %lhs_value, %rhs_value : f16
      linalg.yield %result : f16
    }
    return
  }
  func.func @vec_cmpf_ogt_f16(%lhs: memref<?xf16>, %rhs: memref<?xf16>, %out: memref<?xi1>) {
    %L = loom.sym @L : index
    loom.bind_shape %lhs, [%L] : memref<?xf16>
    loom.bind_mem %lhs, @input : memref<?xf16>
    loom.bind_shape %rhs, [%L] : memref<?xf16>
    loom.bind_mem %rhs, @input : memref<?xf16>
    loom.bind_shape %out, [%L] : memref<?xi1>
    loom.bind_mem %out, @result : memref<?xi1>
    linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    }
    ins(%lhs, %rhs : memref<?xf16>, memref<?xf16>)
    outs(%out : memref<?xi1>) {
    ^bb0(%lhs_value: f16, %rhs_value: f16, %unused: i1):
      %result = arith.cmpf ogt, %lhs_value, %rhs_value : f16
      linalg.yield %result : i1
    }
    return
  }
  func.func @vec_select_f16(%condition: memref<?xi1>, %on_true: memref<?xf16>, %on_false: memref<?xf16>, %out: memref<?xf16>) {
    %L = loom.sym @L : index
    loom.bind_shape %condition, [%L] : memref<?xi1>
    loom.bind_mem %condition, @input : memref<?xi1>
    loom.bind_shape %on_true, [%L] : memref<?xf16>
    loom.bind_mem %on_true, @input : memref<?xf16>
    loom.bind_shape %on_false, [%L] : memref<?xf16>
    loom.bind_mem %on_false, @input : memref<?xf16>
    loom.bind_shape %out, [%L] : memref<?xf16>
    loom.bind_mem %out, @result : memref<?xf16>
    linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    }
    ins(%condition, %on_true, %on_false : memref<?xi1>, memref<?xf16>, memref<?xf16>)
    outs(%out : memref<?xf16>) {
    ^bb0(%condition_value: i1, %true_value: f16, %false_value: f16, %unused: f16):
      %result = arith.select %condition_value, %true_value, %false_value : f16
      linalg.yield %result : f16
    }
    return
  }
  func.func @vec_log_f16(%input: memref<?xf16>, %out: memref<?xf16>) {
    %L = loom.sym @L : index
    loom.bind_shape %input, [%L] : memref<?xf16>
    loom.bind_mem %input, @input : memref<?xf16>
    loom.bind_shape %out, [%L] : memref<?xf16>
    loom.bind_mem %out, @result : memref<?xf16>
    linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    }
    ins(%input : memref<?xf16>)
    outs(%out : memref<?xf16>) {
    ^bb0(%value: f16, %unused: f16):
      %result = math.log %value : f16
      linalg.yield %result : f16
    }
    return
  }
  func.func @relu_f16(%input: memref<?xf16>, %output: memref<?xf16>) {
    %L = loom.sym @L : index
    loom.bind_shape %input, [%L] : memref<?xf16>
    loom.bind_shape %output, [%L] : memref<?xf16>
    loom.bind_mem %input, @input : memref<?xf16>
    loom.bind_mem %output, @result : memref<?xf16>
    linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%input : memref<?xf16>) outs(%output : memref<?xf16>) {
    ^bb0(%x: f16, %unused: f16):
      %zero = arith.constant 0.0 : f16
      %value = arith.maximumf %x, %zero : f16
      linalg.yield %value : f16
    }
    return
  }
}
