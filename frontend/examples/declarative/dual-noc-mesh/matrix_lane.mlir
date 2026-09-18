module @processor {
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
}
