// Vector lane compute semantics — fp16 element-wise and reduction operations.
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
//   - rank-1 memrefs are bound to [%L] via `loom.bind_shape`
//   - memrefs are bound to @L1 via `loom.bind_mem`

module @vector_lane {

// out[i] = max(a[i], b[i]), for i in [0, L)
func.func @vec_max_f16(
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
      %m = arith.maximumf %x, %y : f16
      linalg.yield %m : f16
  }
  return
}

// out[i] = exp(a[i]), for i in [0, L)
func.func @vec_exp_f16(
    %a: memref<?xf16>,
    %out: memref<?xf16>
) {
  %L = loom.sym @L : index
  loom.bind_shape %a, [%L] : memref<?xf16>
  loom.bind_mem %a, @L1 : memref<?xf16>
  loom.bind_shape %out, [%L] : memref<?xf16>
  loom.bind_mem %out, @L1 : memref<?xf16>
  linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    }
    ins(%a : memref<?xf16>)
    outs(%out : memref<?xf16>) {
    ^bb0(%x: f16, %y: f16):
      %e = math.exp %x : f16
      linalg.yield %e : f16
  }
  return
}

// scalar = sum(a[i]) for i in [0, L)
func.func @vec_sum_f16(
    %a: memref<?xf16>,
    %init: memref<f16>
) {
  %L = loom.sym @L : index
  loom.bind_shape %a, [%L] : memref<?xf16>
  loom.bind_mem %a, @L1 : memref<?xf16>
  loom.bind_mem %init, @L1 : memref<f16>
  linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> ()>
      ],
      iterator_types = ["reduction"]
    }
    ins(%a : memref<?xf16>)
    outs(%init : memref<f16>) {
    ^bb0(%x: f16, %acc: f16):
      %s = arith.addf %x, %acc : f16
      linalg.yield %s : f16
  }
  return
}

// out[i] = a[i] + b[i], for i in [0, L)
func.func @vec_add_f16(
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
      %r = arith.addf %x, %y : f16
      linalg.yield %r : f16
  }
  return
}

// out[i] = a[i] * b[i], for i in [0, L)
func.func @vec_mul_f16(
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
      %r = arith.mulf %x, %y : f16
      linalg.yield %r : f16
  }
  return
}

// out[i] = a[i] / b[i], for i in [0, L)
func.func @vec_div_f16(
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
      %r = arith.divf %x, %y : f16
      linalg.yield %r : f16
  }
  return
}

// out[i] = a[i] - b[i], for i in [0, L)
func.func @vec_sub_f16(
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
      %r = arith.subf %x, %y : f16
      linalg.yield %r : f16
  }
  return
}

// out[i] = pow(a[i], b[i]), for i in [0, L)
func.func @vec_powf_f16(
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
      %r = math.powf %x, %y : f16
      linalg.yield %r : f16
  }
  return
}

func.func @vec_cmpf_ogt_f16(
  %a: memref<?xf16>,
  %b: memref<?xf16>,
  %out: memref<?xi1>
) {
  %L = loom.sym @L : index
  loom.bind_shape %a, [%L] : memref<?xf16>
  loom.bind_mem %a, @L1 : memref<?xf16>
  loom.bind_shape %b, [%L] : memref<?xf16>
  loom.bind_mem %b, @L1 : memref<?xf16>
  loom.bind_shape %out, [%L] : memref<?xi1>
  loom.bind_mem %out, @L1 : memref<?xi1>
  linalg.generic {
    indexing_maps = [
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  }
  ins(%a, %b : memref<?xf16>, memref<?xf16>)
  outs(%out : memref<?xi1>) {
    ^bb0(%x: f16, %y: f16, %z: i1):
      %m = arith.cmpf ogt, %x, %y : f16
      linalg.yield %m : i1
  }
  return
}

func.func @vec_select_f16(
  %cond: memref<?xi1>,
  %a: memref<?xf16>,
  %b: memref<?xf16>,
  %out: memref<?xf16>
) {
  %L = loom.sym @L : index
  loom.bind_shape %cond, [%L] : memref<?xi1>
  loom.bind_mem %cond, @L1 : memref<?xi1>
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
      affine_map<(d0) -> (d0)>,
      affine_map<(d0) -> (d0)>
    ],
    iterator_types = ["parallel"]
  }
  ins(%cond, %a, %b : memref<?xi1>, memref<?xf16>, memref<?xf16>)
  outs(%out : memref<?xf16>) {
    ^bb0(%c: i1, %x: f16, %y: f16, %z: f16):
      %m = arith.select %c, %x, %y : f16
      linalg.yield %m : f16
  }
  return
}
}
