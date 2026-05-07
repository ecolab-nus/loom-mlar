use mlar_rust::*;

use crate::core_arch::single_core;
use crate::dimensions::{dim_x, dim_y};

/// Build the scaled 8×8 mesh.
///
/// The mesh has no explicit inter-core scale-out connectivity: cross-core
/// transfers are modeled by the NoC data movers attached at the system level.
pub fn scaled_mesh() -> Architecture {
    let core = single_core();
    let dim_x = dim_x();
    let dim_y = dim_y();
    core.scale([&dim_x, &dim_y]).with_name("mesh")
}

/// Look up the scaled mesh-wide L1 region used as the destination of
/// DRAM<->L1 transfers.
pub fn mesh_array_l1(mesh: &Architecture) -> MemoryRegion {
    mesh.get_scaled_memory_region("L1")
        .expect("scaled mesh should expose mesh-wide L1")
}
