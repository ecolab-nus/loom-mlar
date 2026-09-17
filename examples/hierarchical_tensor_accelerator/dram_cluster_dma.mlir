module @processor {
  func.func @load_cluster_f16(%src: memref<?xf16>, %dst: memref<?xf16>) {
    %L = loom.sym @L : index
    loom.bind_shape %src, [%L] : memref<?xf16>
    loom.bind_shape %dst, [%L] : memref<?xf16>
    loom.bind_mem %src, @input_0 : memref<?xf16>
    loom.bind_mem %dst, @output_0 : memref<?xf16>
    memref.copy %src, %dst : memref<?xf16> to memref<?xf16>
    return
  }
}
