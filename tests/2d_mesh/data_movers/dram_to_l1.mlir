// Data-mover semantics for DRAM <-> L1 transfers (bi-directional).
//
// Interface convention:
// - memref args are the real transfer endpoints (source and destination)
// - symbols are bound directly to input/output memrefs via `loom.bind_shape`
// - memory regions are associated with memref args via `loom.bind_mem`

module @data_movers {

func.func @dram_to_l1_f16(
    %dram_src: memref<?x?xf16>,
    %l1_dst: memref<?x?xf16>
) {
  %M = loom.sym @M : index
  %N = loom.sym @N : index
  loom.bind_shape %dram_src, [%M, %N] : memref<?x?xf16>
  loom.bind_shape %l1_dst, [%M, %N] : memref<?x?xf16>
  loom.bind_mem @DRAM %dram_src
  loom.bind_mem @L1 %l1_dst
  memref.copy %dram_src, %l1_dst : memref<?x?xf16> to memref<?x?xf16>
  return
}

}
