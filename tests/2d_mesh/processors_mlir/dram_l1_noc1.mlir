// L1 -> DRAM writeback and L1 -> L1 gather transfers carried over NoC1.

module @dram_l1_noc1 {

func.func @l1_to_dram_f16(
    %l1_src: memref<?x?xf16>,
    %dram_dst: memref<?x?xf16>
) {
  %M = loom.sym @M : index
  %N = loom.sym @N : index
  loom.bind_shape %l1_src, [%M, %N] : memref<?x?xf16>
  loom.bind_shape %dram_dst, [%M, %N] : memref<?x?xf16>
  loom.bind_mem %l1_src, @array_L1 : memref<?x?xf16>
  loom.bind_mem %dram_dst, @DRAM : memref<?x?xf16>
  loom.copy %l1_src, %dram_dst src_mem_space @array_L1 dst_mem_space @DRAM, area: [1, 1] : memref<?x?xf16> to memref<?x?xf16>
  return
}

func.func @l1_gather(
      %l1_src: memref<?x?xf16>,
      %l1_dst: memref<?x?x?xf16>
  ) {
    %M = loom.sym @M : index
    %N = loom.sym @N : index
    %B = loom.sym @B : index
    loom.bind_shape %l1_src, [%M, %N] : memref<?x?xf16>
    loom.bind_shape %l1_dst, [%B, %M, %N] : memref<?x?x?xf16>
    loom.bind_mem %l1_src, @array_L1 : memref<?x?xf16>
    loom.bind_mem %l1_dst, @array_L1 : memref<?x?x?xf16>
    %gather_x = loom.sym @GATHER_X : index
    %gather_y = loom.sym @GATHER_Y : index
    loom.gather ins(%l1_src: memref<?x?xf16>) outs(%l1_dst: memref<?x?x?xf16>) area: [%gather_x, %gather_y]
    return
  }
}
