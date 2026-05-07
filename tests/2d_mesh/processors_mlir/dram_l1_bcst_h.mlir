// Horizontal broadcast from DRAM to per-core L1.
//
// Uses only horizontal mesh links — can run in parallel with vertical broadcasts.

module @dram_l1_bcst_h {

func.func @dram_to_l1_1d_bcst_h_f16(
    %dram_src: memref<?x?xf16>,
    %l1_dst: memref<?x?xf16>
) {
  %M = loom.sym @M : index
  %N = loom.sym @N : index
  loom.bind_shape %dram_src, [%M, %N] : memref<?x?xf16>
  loom.bind_shape %l1_dst, [%M, %N] : memref<?x?xf16>
  loom.bind_mem %dram_src, @DRAM : memref<?x?xf16>
  loom.bind_mem %l1_dst, @array_L1 : memref<?x?xf16>
  loom.copy %dram_src, %l1_dst src_mem_space @DRAM dst_mem_space @array_L1, broadcast : [8, 1] : memref<?x?xf16> to memref<?x?xf16>
  return
}

}
