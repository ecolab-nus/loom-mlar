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
//   - rank-1 tensors are bound to [%L] via `loom.bind`

module @vector_lane {

// out[i] = max(a[i], b[i]), for i in [0, L)
func.func @vec_max_f32(
    %a: tensor<?xf32>,
    %b: tensor<?xf32>,
    %out: tensor<?xf32>
) -> tensor<?xf32> {
  %L = loom.sym @L : index
  loom.bind %a, [%L] : tensor<?xf32>
  loom.bind %b, [%L] : tensor<?xf32>
  loom.bind %out, [%L] : tensor<?xf32>
  %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    }
    ins(%a, %b : tensor<?xf32>, tensor<?xf32>)
    outs(%out : tensor<?xf32>) {
    ^bb0(%x: f32, %y: f32, %z: f32):
      %m = arith.maximumf %x, %y : f32
      linalg.yield %m : f32
  } -> tensor<?xf32>
  return %result : tensor<?xf32>
}

// out[i] = exp(a[i]), for i in [0, L)
func.func @vec_exp_f32(
    %a: tensor<?xf32>,
    %out: tensor<?xf32>
) -> tensor<?xf32> {
  %L = loom.sym @L : index
  loom.bind %a, [%L] : tensor<?xf32>
  loom.bind %out, [%L] : tensor<?xf32>
  %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    }
    ins(%a : tensor<?xf32>)
    outs(%out : tensor<?xf32>) {
    ^bb0(%x: f32, %y: f32):
      %e = math.exp %x : f32
      linalg.yield %e : f32
  } -> tensor<?xf32>
  return %result : tensor<?xf32>
}

// scalar = sum(a[i]) for i in [0, L)
func.func @vec_sum_f32(
    %a: tensor<?xf32>,
    %init: tensor<f32>
) -> tensor<f32> {
  %L = loom.sym @L : index
  loom.bind %a, [%L] : tensor<?xf32>
  %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> ()>
      ],
      iterator_types = ["reduction"]
    }
    ins(%a : tensor<?xf32>)
    outs(%init : tensor<f32>) {
    ^bb0(%x: f32, %acc: f32):
      %s = arith.addf %x, %acc : f32
      linalg.yield %s : f32
  } -> tensor<f32>
  return %result : tensor<f32>
}

// out[i] = a[i] + b[i], for i in [0, L)
func.func @vec_add_f32(
    %a: tensor<?xf32>,
    %b: tensor<?xf32>,
    %out: tensor<?xf32>
) -> tensor<?xf32> {
  %L = loom.sym @L : index
  loom.bind %a, [%L] : tensor<?xf32>
  loom.bind %b, [%L] : tensor<?xf32>
  loom.bind %out, [%L] : tensor<?xf32>
  %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    }
    ins(%a, %b : tensor<?xf32>, tensor<?xf32>)
    outs(%out : tensor<?xf32>) {
    ^bb0(%x: f32, %y: f32, %z: f32):
      %r = arith.addf %x, %y : f32
      linalg.yield %r : f32
  } -> tensor<?xf32>
  return %result : tensor<?xf32>
}

// out[i] = a[i] * b[i], for i in [0, L)
func.func @vec_mul_f32(
    %a: tensor<?xf32>,
    %b: tensor<?xf32>,
    %out: tensor<?xf32>
) -> tensor<?xf32> {
  %L = loom.sym @L : index
  loom.bind %a, [%L] : tensor<?xf32>
  loom.bind %b, [%L] : tensor<?xf32>
  loom.bind %out, [%L] : tensor<?xf32>
  %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    }
    ins(%a, %b : tensor<?xf32>, tensor<?xf32>)
    outs(%out : tensor<?xf32>) {
    ^bb0(%x: f32, %y: f32, %z: f32):
      %r = arith.mulf %x, %y : f32
      linalg.yield %r : f32
  } -> tensor<?xf32>
  return %result : tensor<?xf32>
}

// out[i] = a[i] / b[i], for i in [0, L)
func.func @vec_div_f32(
    %a: tensor<?xf32>,
    %b: tensor<?xf32>,
    %out: tensor<?xf32>
) -> tensor<?xf32> {
  %L = loom.sym @L : index
  loom.bind %a, [%L] : tensor<?xf32>
  loom.bind %b, [%L] : tensor<?xf32>
  loom.bind %out, [%L] : tensor<?xf32>
  %result = linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    }
    ins(%a, %b : tensor<?xf32>, tensor<?xf32>)
    outs(%out : tensor<?xf32>) {
    ^bb0(%x: f32, %y: f32, %z: f32):
      %r = arith.divf %x, %y : f32
      linalg.yield %r : f32
  } -> tensor<?xf32>
  return %result : tensor<?xf32>
}

}
