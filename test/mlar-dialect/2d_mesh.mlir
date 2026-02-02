// Example MLAR dialect hardware description: 2D mesh architecture
// This example is adapted from the TensTorrent Wormhole architecture

module {
    // Functional units description
    %mat_unit = mlar.mat "FPU" {shape = [32, 32, 32], throughput = 128}
    %vec_unit = mlar.vec "SFPU" {shape = [32]}
    
    // Scale-out description (spatial dimensions)
    %x = mlar.spatial_dim "x", 8
    %y = mlar.spatial_dim "y", 8
    
    // Core declaration with scaleout and scalein
    %cores = mlar.core "core" {scaleout=(%x, %y) , scalein=(%mat_unit, %vec_unit, [8,1])}
    
    // L1 memory per core
    %L1 = mlar.memory "L1" {scaleout=(%x, %y) , size = 1499136, bandwidth = 15}
    
    // Core to memory mapping (1:1)
    %core_to_mem = mlar.mux %cores, %L1, {map = affine_map<(d0, d1) -> (d0, d1)>}
    
    // Horizontal NoC links (ring in x dimension)
    %noc_h = mlar.interconnects "horizontal_links" %L1 : !mlar.memory, %L1 : !mlar.memory, {map = affine_map<(d0, d1) -> ((d0 + 1) mod 8, d1)>, bandwidth = 128, spatial_dims = [@y]} : !mlar.interconnect
    
    // Vertical NoC links (ring in y dimension)
    %noc_v = mlar.interconnects "vertical_links" %L1 : !mlar.memory, %L1 : !mlar.memory, {map = affine_map<(d0, d1) -> (d0, (d1 + 1) mod 8)>, bandwidth = 128, spatial_dims = [@x]} : !mlar.interconnect
    
    // DRAM resources
    %dram_idx = mlar.spatial_dim "d", 4
    %drams = mlar.memory "DRAM" {scaleout=(%dram_idx) , size = 34359738368, bandwidth = 288}
    
    // L1 to DRAM interconnect
    %to_dram = mlar.interconnects "NoC" %L1: !mlar.memory, %drams : !mlar.memory, {map = affine_map<(d0, d1) -> (d0 ceildiv 4 + 2 * (d1 ceildiv 4))>}
}
