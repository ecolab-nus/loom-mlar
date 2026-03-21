// Data-mover semantics for DRAM -> L1 transfers.
//
// Interface convention:
// - memref args are the real transfer endpoints (source and destination)
// - symbols are bound directly to input/output memrefs via `loom.bind`

module @data_movers {

func.func @dram_to_l1_f16(
    %dram_src: memref<?x?xf16>,
    %l1_dst: memref<?x?xf16>
) {
  %M = loom.sym @M : index
  %N = loom.sym @N : index
  loom.bind %dram_src, [%M, %N] : memref<?x?xf16>
  loom.bind %l1_dst, [%M, %N] : memref<?x?xf16>
  memref.copy %dram_src, %l1_dst : memref<?x?xf16> to memref<?x?xf16>
  return
}

}
