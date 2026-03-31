// DRAM <-> L1 transfers that use both horizontal and vertical mesh links:
// unicast (dram_to_l1, l1_to_dram) and full 2D broadcast.

module @dram_l1_mover {

func.func @dram_to_l1_f16(
    %dram_src: memref<?x?xf16>,
    %l1_dst: memref<?x?xf16>
) {
  %M = loom.sym @M : index
  %N = loom.sym @N : index
  loom.bind_shape %dram_src, [%M, %N] : memref<?x?xf16>
  loom.bind_shape %l1_dst, [%M, %N] : memref<?x?xf16>
  loom.bind_mem %dram_src, @DRAM : memref<?x?xf16>
  loom.bind_mem %l1_dst, @L1 : memref<?x?xf16>
  loom.copy %dram_src, %l1_dst src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16>
  return
}

func.func @dram_to_l1_1d_bcst_f16(
    %dram_src: memref<?x?xf16>,
    %l1_dst: memref<?x?xf16>
) {
  %M = loom.sym @M : index
  %N = loom.sym @N : index
  loom.bind_shape %dram_src, [%M, %N] : memref<?x?xf16>
  loom.bind_shape %l1_dst, [%M, %N] : memref<?x?xf16>
  loom.bind_mem %dram_src, @DRAM : memref<?x?xf16>
  loom.bind_mem %l1_dst, @L1 : memref<?x?xf16>
  loom.copy %dram_src, %l1_dst src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 8] : memref<?x?xf16> to memref<?x?xf16>
  return
}

func.func @l1_to_dram_f16(
    %l1_src: memref<?x?xf16>,
    %dram_dst: memref<?x?xf16>
) {
  %M = loom.sym @M : index
  %N = loom.sym @N : index
  loom.bind_shape %l1_src, [%M, %N] : memref<?x?xf16>
  loom.bind_shape %dram_dst, [%M, %N] : memref<?x?xf16>
  loom.bind_mem %l1_src, @L1 : memref<?x?xf16>
  loom.bind_mem %dram_dst, @DRAM : memref<?x?xf16>
  loom.copy %l1_src, %dram_dst src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16>
  return
}

}
