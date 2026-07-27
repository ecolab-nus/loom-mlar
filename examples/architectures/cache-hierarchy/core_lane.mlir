module @core_lane {
  func.func @elementwise_add(
      %lhs: memref<?xf16>,
      %rhs: memref<?xf16>,
      %out: memref<?xf16>
  ) {
    %L = loom.sym @L : index
    loom.bind_shape %lhs, [%L] : memref<?xf16>
    loom.bind_mem %lhs, @L1 : memref<?xf16>
    loom.bind_shape %rhs, [%L] : memref<?xf16>
    loom.bind_mem %rhs, @L1 : memref<?xf16>
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
    ins(%lhs, %rhs : memref<?xf16>, memref<?xf16>)
    outs(%out : memref<?xf16>) {
      ^bb0(%a: f16, %b: f16, %unused: f16):
        %sum = arith.addf %a, %b : f16
        linalg.yield %sum : f16
    }
    return
  }
}
