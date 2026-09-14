module @processor {
  func.func @load_l1(%src: memref<?xf16>, %dst: memref<?xf16>) {
    %L = loom.sym @L : index
    loom.bind_shape %src, [%L] : memref<?xf16>
    loom.bind_shape %dst, [%L] : memref<?xf16>
    loom.bind_mem %src, @input_0 : memref<?xf16>
    loom.bind_mem %dst, @output_0 : memref<?xf16>
    loom.copy %src, %dst src_mem_space @input_0 dst_mem_space @output_0, area: [1, 1] : memref<?xf16> to memref<?xf16>
    return
  }
  func.func @load_l1_broadcast(%src: memref<?xf16>, %dst: memref<?xf16>) {
    %L = loom.sym @L : index
    %bcst_x = loom.sym @bcst_x : index
    %bcst_y = loom.sym @bcst_y : index
    loom.bind_shape %src, [%L] : memref<?xf16>
    loom.bind_shape %dst, [%L] : memref<?xf16>
    loom.bind_mem %src, @input_0 : memref<?xf16>
    loom.bind_mem %dst, @output_0 : memref<?xf16>
    loom.copy %src, %dst src_mem_space @input_0 dst_mem_space @output_0, area: [%bcst_x, %bcst_y] : memref<?xf16> to memref<?xf16>
    return
  }
}
