module @vector_lane {
  func.func @vec_exp_f16(
      %input: memref<?xf16>,
      %output: memref<?xf16>
  ) {
    %L = loom.sym @L : index
    loom.bind_shape %input, [%L] : memref<?xf16>
    loom.bind_shape %output, [%L] : memref<?xf16>
    loom.bind_mem %input, @L1 : memref<?xf16>
    loom.bind_mem %output, @L1 : memref<?xf16>
    linalg.generic {
      indexing_maps = [
        affine_map<(d0) -> (d0)>,
        affine_map<(d0) -> (d0)>
      ],
      iterator_types = ["parallel"]
    }
    ins(%input : memref<?xf16>)
    outs(%output : memref<?xf16>) {
      ^bb0(%value: f16, %unused: f16):
        %result = math.exp %value : f16
        linalg.yield %result : f16
    }
    return
  }
}
