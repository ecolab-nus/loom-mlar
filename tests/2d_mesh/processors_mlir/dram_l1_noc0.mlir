// DRAM -> L1 transfers carried over NoC0:
// unicast plus a parameterized 2D broadcast.

module @dram_l1_noc0 {

func.func @dram_to_l1_f16(
    %dram_src: memref<?x?xf16>,
    %l1_dst: memref<?x?xf16>
) {
  %M = loom.sym @M : index
  %N = loom.sym @N : index
  loom.bind_shape %dram_src, [%M, %N] : memref<?x?xf16>
  loom.bind_shape %l1_dst, [%M, %N] : memref<?x?xf16>
  loom.bind_mem %dram_src, @DRAM : memref<?x?xf16>
  loom.bind_mem %l1_dst, @array_L1 : memref<?x?xf16>
  loom.copy %dram_src, %l1_dst src_mem_space @DRAM dst_mem_space @array_L1, area: [1, 1]
  return
}

func.func @dram_to_l1_bcst(
    %dram_src: memref<?x?xf16>,
    %l1_dst: memref<?x?xf16>
) {
  %M = loom.sym @M : index
  %N = loom.sym @N : index
  loom.bind_shape %dram_src, [%M, %N] : memref<?x?xf16>
  loom.bind_shape %l1_dst, [%M, %N] : memref<?x?xf16>
  loom.bind_mem %dram_src, @DRAM : memref<?x?xf16>
  loom.bind_mem %l1_dst, @array_L1 : memref<?x?xf16>
  %bcst_x = loom.sym @BCST_X : index
  %bcst_y = loom.sym @BCST_Y : index
  loom.copy %dram_src, %l1_dst src_mem_space @DRAM dst_mem_space @array_L1, area: [%bcst_x, %bcst_y]
  return
}

}
