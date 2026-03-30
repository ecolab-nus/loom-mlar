// Data-mover semantics for DRAM <-> L1 transfers.
//
// Interface convention:
// - memref args are the real transfer endpoints (source and destination)
// - symbols are bound directly to input/output memrefs via `loom.bind_shape`
// - `loom.copy` specifies the transfer with explicit source/destination memory spaces and broadcast

module @dram_l1_mover {

func.func @dram_to_l1_f16(
    %dram_src: memref<?x?xf16>,
    %l1_dst: memref<?x?xf16>
) {
  %M = loom.sym @M : index
  %N = loom.sym @N : index
  loom.bind_shape %dram_src, [%M, %N] 
  loom.bind_shape %l1_dst, [%M, %N] 
  loom.bind_mem %dram_src, @DRAM
  loom.bind_mem %l1_dst, @L1
  loom.copy %dram_src, %l1_dst src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16>
  return
}

func.func @dram_to_l1_2d_bcst_f16(
    %dram_src: memref<?x?xf16>,
    %l1_dst: memref<?x?xf16>
) {
  %M = loom.sym @M : index
  %N = loom.sym @N : index
  loom.bind_shape %dram_src, [%M, %N] 
  loom.bind_shape %l1_dst, [%M, %N] 
  loom.bind_mem %dram_src, @DRAM
  loom.bind_mem %l1_dst, @L1
  loom.copy %dram_src, %l1_dst src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 8] : memref<?x?xf16> to memref<?x?xf16>
  return
}

func.func @dram_to_l1_1d_bcst_f16(
    %dram_src: memref<?x?xf16>,
    %l1_dst: memref<?x?xf16>
) {
  %M = loom.sym @M : index
  %N = loom.sym @N : index
  loom.bind_shape %dram_src, [%M, %N] 
  loom.bind_shape %l1_dst, [%M, %N] 
  loom.bind_mem %dram_src, @DRAM
  loom.bind_mem %l1_dst, @L1
  loom.copy %dram_src, %l1_dst src_mem_space @DRAM dst_mem_space @L1, broadcast : [1, 8] : memref<?x?xf16> to memref<?x?xf16>
  return
}

func.func @dram_to_l1_1d_bcst_m_f16(
    %dram_src: memref<?x?xf16>,
    %l1_dst: memref<?x?xf16>
) {
  %M = loom.sym @M : index
  %N = loom.sym @N : index
  loom.bind_shape %dram_src, [%M, %N] 
  loom.bind_shape %l1_dst, [%M, %N] 
  loom.bind_mem %dram_src, @DRAM
  loom.bind_mem %l1_dst, @L1
  loom.copy %dram_src, %l1_dst src_mem_space @DRAM dst_mem_space @L1, broadcast : [8, 1] : memref<?x?xf16> to memref<?x?xf16>
  return
}

func.func @l1_to_dram_f16(
    %l1_src: memref<?x?xf16>,
    %dram_dst: memref<?x?xf16>
) {
  %M = loom.sym @M : index
  %N = loom.sym @N : index
  loom.bind_shape %l1_src, [%M, %N] 
  loom.bind_shape %dram_dst, [%M, %N] 
  loom.bind_mem %l1_src, @L1
  loom.bind_mem %dram_dst, @DRAM
  loom.copy %l1_src, %dram_dst src_mem_space @L1 dst_mem_space @DRAM, broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16>
  return
}

}
