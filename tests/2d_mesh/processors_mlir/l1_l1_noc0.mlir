// L1 -> L1 gather transfers carried over NoC0.

module @l1_l1_noc0 {

func.func @l1_gather(
      %l1_src: memref<?x?xf16>,
      %l1_dst: memref<?x?x?xf16>
  ) {
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    %B = loom.sym @B : index
    %effective_bandwidth = loom.sym @effective_bandwidth : index
    loom.bind_shape %l1_src, [%M, %N] : memref<?x?xf16>
    loom.bind_shape %l1_dst, [%B, %M, %N] : memref<?x?x?xf16>
    loom.bind_mem %l1_src, @array_L1 : memref<?x?xf16>
    loom.bind_mem %l1_dst, @array_L1 : memref<?x?x?xf16>
    %gather_x = loom.sym @gather_x : index
    %gather_y = loom.sym @gather_y : index
    loom.gather %l1_src, %l1_dst src_mem_space @array_L1 dst_mem_space @array_L1 area: [%gather_x, %gather_y] : memref<?x?xf16> to memref<?x?x?xf16>
    return
  }

}
