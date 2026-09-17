module @processor {
  func.func @pipeline_exp_f16(%input: memref<?xf16>, %output: memref<?xf16>) {
    %L = loom.sym @L : index
    loom.bind_shape %input, [%L] : memref<?xf16>
    loom.bind_shape %output, [%L] : memref<?xf16>
    loom.bind_mem %input, @input : memref<?xf16>
    loom.bind_mem %output, @result : memref<?xf16>
    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%input : memref<?xf16>) outs(%output : memref<?xf16>) {
    ^bb0(%x: f16, %unused: f16):
      %value = math.exp %x : f16
      linalg.yield %value : f16
    }
    return
  }
}
