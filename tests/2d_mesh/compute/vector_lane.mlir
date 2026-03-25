// Vector lane compute semantics — fp32 element-wise and reduction operations.
//
// Supported operations:
//   - element-wise max of two vectors
//   - element-wise exponential
//   - sum reduction over a vector
//   - element-wise addition
//   - element-wise multiplication
//   - element-wise division
//
// Symbol convention:
//   - @L: logical vector length
//   - rank-1 tensors are bound to [%L] via `loom.bind_shape`

module @vector_lane {

// out[i] = max(a[i], b[i]), for i in [0, L)
func.func @vec_max_f16(
    %a: tensor<?xf16>,
    %b: tensor<?xf16>,
    %out: tensor<?xf16>
) -> tensor<?xf16> {
  %L = loom.sym @L : index
  loom.bind_shape %a, [%L] : tensor<?xf16>
  loom.bind_shape %b, [%L] : tensor<?xf16>
  loom.bind_shape %out, [%L] : tensor<?xf16>
  %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    }
    ins(%a, %b : tensor<?xf16>, tensor<?xf16>)
    outs(%out : tensor<?xf16>) {
    ^bb0(%x: f16, %y: f16, %z: f16):
      %m = arith.maximumf %x, %y : f16
      linalg.yield %m : f16
  } -> tensor<?xf16>
  return %result : tensor<?xf16>
}

// out[i] = exp(a[i]), for i in [0, L)
func.func @vec_exp_f16(
    %a: tensor<?xf16>,
    %out: tensor<?xf16>
) -> tensor<?xf16> {
  %L = loom.sym @L : index
  loom.bind_shape %a, [%L] : tensor<?xf16>
  loom.bind_shape %out, [%L] : tensor<?xf16>
  %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    }
    ins(%a : tensor<?xf16>)
    outs(%out : tensor<?xf16>) {
    ^bb0(%x: f16, %y: f16):
      %e = math.exp %x : f16
      linalg.yield %e : f16
  } -> tensor<?xf16>
  return %result : tensor<?xf16>
}

// scalar = sum(a[i]) for i in [0, L)
func.func @vec_sum_f16(
    %a: tensor<?xf16>,
    %init: tensor<f16>
) -> tensor<f16> {
  %L = loom.sym @L : index
  loom.bind_shape %a, [%L] : tensor<?xf16>
  %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> ()>
      ],
      iterator_types = ["reduction"]
    }
    ins(%a : tensor<?xf16>)
    outs(%init : tensor<f16>) {
    ^bb0(%x: f16, %acc: f16):
      %s = arith.addf %x, %acc : f16
      linalg.yield %s : f16
  } -> tensor<f16>
  return %result : tensor<f16>
}

func.func @vec_vsum_f16(
  %a: tensor<?x?xf16>,
  %out: tensor<?xf16>
) -> tensor<?xf16> {
  %P = loom.sym @P : index
  %R = loom.sym @R : index
  loom.bind_shape %a, [%P, %R] : tensor<?x?xf16>
  loom.bind_shape %out, [%P] : tensor<?xf16>
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0)>
    ],
    iterator_types = ["parallel", "reduction"]
  }
  ins(%a : tensor<?x?xf16>)
  outs(%out : tensor<?xf16>) {
    ^bb0(%x: f16, %acc: f16):
      %s = arith.addf %x, %acc : f16
      linalg.yield %s : f16
  } -> tensor<?xf16>
  return %result : tensor<?xf16>
}

// out[i] = a[i] + b[i], for i in [0, L)
func.func @vec_add_f16(
    %a: tensor<?xf16>,
    %b: tensor<?xf16>,
    %out: tensor<?xf16>
) -> tensor<?xf16> {
  %L = loom.sym @L : index
  loom.bind_shape %a, [%L] : tensor<?xf16>
  loom.bind_shape %b, [%L] : tensor<?xf16>
  loom.bind_shape %out, [%L] : tensor<?xf16>
  %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    }
    ins(%a, %b : tensor<?xf16>, tensor<?xf16>)
    outs(%out : tensor<?xf16>) {
    ^bb0(%x: f16, %y: f16, %z: f16):
      %r = arith.addf %x, %y : f16
      linalg.yield %r : f16
  } -> tensor<?xf16>
  return %result : tensor<?xf16>
}

// out[i] = a[i] * b[i], for i in [0, L)
func.func @vec_mul_f16(
    %a: tensor<?xf16>,
    %b: tensor<?xf16>,
    %out: tensor<?xf16>
) -> tensor<?xf16> {
  %L = loom.sym @L : index
  loom.bind_shape %a, [%L] : tensor<?xf16>
  loom.bind_shape %b, [%L] : tensor<?xf16>
  loom.bind_shape %out, [%L] : tensor<?xf16>
  %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    }
    ins(%a, %b : tensor<?xf16>, tensor<?xf16>)
    outs(%out : tensor<?xf16>) {
    ^bb0(%x: f16, %y: f16, %z: f16):
      %r = arith.mulf %x, %y : f16
      linalg.yield %r : f16
  } -> tensor<?xf16>
  return %result : tensor<?xf16>
}

// out[i] = a[i] / b[i], for i in [0, L)
func.func @vec_div_f16(
    %a: tensor<?xf16>,
    %b: tensor<?xf16>,
    %out: tensor<?xf16>
) -> tensor<?xf16> {
  %L = loom.sym @L : index
  loom.bind_shape %a, [%L] : tensor<?xf16>
  loom.bind_shape %b, [%L] : tensor<?xf16>
  loom.bind_shape %out, [%L] : tensor<?xf16>
  %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    }
    ins(%a, %b : tensor<?xf16>, tensor<?xf16>)
    outs(%out : tensor<?xf16>) {
    ^bb0(%x: f16, %y: f16, %z: f16):
      %r = arith.divf %x, %y : f16
      linalg.yield %r : f16
  } -> tensor<?xf16>
  return %result : tensor<?xf16>
}

// out[i] = a[i] - b[i], for i in [0, L)
func.func @vec_sub_f16(
    %a: tensor<?xf16>,
    %b: tensor<?xf16>,
    %out: tensor<?xf16>
) -> tensor<?xf16> {
  %L = loom.sym @L : index
  loom.bind_shape %a, [%L] : tensor<?xf16>
  loom.bind_shape %b, [%L] : tensor<?xf16>
  loom.bind_shape %out, [%L] : tensor<?xf16>
  %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    }
    ins(%a, %b : tensor<?xf16>, tensor<?xf16>)
    outs(%out : tensor<?xf16>) {
    ^bb0(%x: f16, %y: f16, %z: f16):
      %r = arith.subf %x, %y : f16
      linalg.yield %r : f16
  } -> tensor<?xf16>
  return %result : tensor<?xf16>
}

// out[i] = pow(a[i], b[i]), for i in [0, L)
func.func @vec_powf_f16(
    %a: tensor<?xf16>,
    %b: tensor<?xf16>,
    %out: tensor<?xf16>
) -> tensor<?xf16> {
  %L = loom.sym @L : index
  loom.bind_shape %a, [%L] : tensor<?xf16>
  loom.bind_shape %b, [%L] : tensor<?xf16>
  loom.bind_shape %out, [%L] : tensor<?xf16>
  %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    }
    ins(%a, %b : tensor<?xf16>, tensor<?xf16>)
    outs(%out : tensor<?xf16>) {
    ^bb0(%x: f16, %y: f16, %z: f16):
      %r = math.powf %x, %y : f16
      linalg.yield %r : f16
  } -> tensor<?xf16>
  return %result : tensor<?xf16>
}

// out[j] = max(a[j, k]) for k in [0, R), for j in [0, P)
func.func @vec_vmax_f16(
  %a: tensor<?x?xf16>,
  %out: tensor<?xf16>
) -> tensor<?xf16> {
  %P = loom.sym @P : index
  %R = loom.sym @R : index
  loom.bind_shape %a, [%P, %R] : tensor<?x?xf16>
  loom.bind_shape %out, [%P] : tensor<?xf16>
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0)>
    ],
    iterator_types = ["parallel", "reduction"]
  }
  ins(%a : tensor<?x?xf16>)
  outs(%out : tensor<?xf16>) {
    ^bb0(%x: f16, %acc: f16):
      %m = arith.maximumf %x, %acc : f16
      linalg.yield %m : f16
  } -> tensor<?xf16>
  return %result : tensor<?xf16>
}

func.func @vec_max1_f16(
  %a: tensor<?xf16>,
  %b: tensor<?xf16>,
  %out: tensor<?xf16>
) -> tensor<?xf16> {
  %L = loom.sym @L : index
  loom.bind_shape %a, [%L] : tensor<?xf16>
  loom.bind_shape %b, [%L] : tensor<?xf16>
  loom.bind_shape %out, [%L] : tensor<?xf16>
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  }
  ins(%a, %b : tensor<?xf16>, tensor<?xf16>)
  outs(%out : tensor<?xf16>) {
    ^bb0(%x: f16, %y: f16, %z: f16):
      %cmp = arith.cmpf ogt, %x, %y : f16
      %sel = arith.select %cmp, %x, %y : f16
      linalg.yield %sel : f16
  } -> tensor<?xf16>
  return %result : tensor<?xf16>
}

func.func @vec_cmpf_ogt_f16(
  %a: tensor<?xf16>,
  %b: tensor<?xf16>,
  %out: tensor<?xi1>
) -> tensor<?xi1> {
  %L = loom.sym @L : index
  loom.bind_shape %a, [%L] : tensor<?xf16>
  loom.bind_shape %b, [%L] : tensor<?xf16>
  loom.bind_shape %out, [%L] : tensor<?xi1>
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  }
  ins(%a, %b : tensor<?xf16>, tensor<?xf16>)
  outs(%out : tensor<?xi1>) {
    ^bb0(%x: f16, %y: f16, %z: i1):
      %m = arith.cmpf ogt, %x, %y : f16
      linalg.yield %m : i1
  } -> tensor<?xi1>
  return %result : tensor<?xi1>
}

func.func @vec_select_f16(
  %cond: tensor<?xi1>,
  %a: tensor<?xf16>,
  %b: tensor<?xf16>,
  %out: tensor<?xf16>
) -> tensor<?xf16> {
  %L = loom.sym @L : index
  loom.bind_shape %cond, [%L] : tensor<?xi1>
  loom.bind_shape %a, [%L] : tensor<?xf16>
  loom.bind_shape %b, [%L] : tensor<?xf16>
  loom.bind_shape %out, [%L] : tensor<?xf16>
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  }
  ins(%cond, %a, %b : tensor<?xi1>, tensor<?xf16>, tensor<?xf16>)
  outs(%out : tensor<?xf16>) {
    ^bb0(%c: i1, %x: f16, %y: f16, %z: f16):
      %m = arith.select %c, %x, %y : f16
      linalg.yield %m : f16
  } -> tensor<?xf16>
  return %result : tensor<?xf16>
}
}
