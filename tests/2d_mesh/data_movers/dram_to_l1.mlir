// Data-mover semantics for DRAM -> L1 transfers.
//
// Interface convention:
// - memref args are the real transfer endpoints (source and destination)
// - symbols are bound directly to input/output memrefs via `loom.bind_shape`
// - `loom.copy` specifies the transfer with memory regions, interconnect, and broadcast

module @data_movers {

func.func @dram_to_l1_f16(
    %dram_src: memref<?x?xf16>,
    %l1_dst: memref<?x?xf16>
) {
  %M = loom.sym @M : index
  %N = loom.sym @N : index
  loom.bind_shape %dram_src, [%M, %N] : memref<?x?xf16>
  loom.bind_shape %l1_dst, [%M, %N] : memref<?x?xf16>
  loom.copy %dram_src @DRAM, %l1_dst @L1, interconnect : [], broadcast : [1, 1] : memref<?x?xf16> to memref<?x?xf16>
  return
}

func.func @dram_to_l1_bcst_f16(
    %dram_src: memref<?x?xf16>,
    %l1_dst: memref<?x?xf16>
) {
  %M = loom.sym @M : index
  %N = loom.sym @N : index
  loom.bind_shape %dram_src, [%M, %N] : memref<?x?xf16>
  loom.bind_shape %l1_dst, [%M, %N] : memref<?x?xf16>
  loom.copy %dram_src @DRAM, %l1_dst @L1, interconnect : [], broadcast : [8, 8] : memref<?x?xf16> to memref<?x?xf16>
  return
}

}
