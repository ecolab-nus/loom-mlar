module @processor {
  func.func @pipeline_reduce_f16(%input: memref<?x?xf16>, %output: memref<?xf16>) {
    %P = loom.sym @P : index
    %R = loom.sym @R : index
    loom.bind_shape %input, [%P, %R] : memref<?x?xf16>
    loom.bind_shape %output, [%P] : memref<?xf16>
    loom.bind_mem %input, @input : memref<?x?xf16>
    loom.bind_mem %output, @result : memref<?xf16>
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%input : memref<?x?xf16>) outs(%output : memref<?xf16>) {
    ^bb0(%x: f16, %acc: f16):
      %value = arith.addf %x, %acc : f16
      linalg.yield %value : f16
    }
    return
  }
}
