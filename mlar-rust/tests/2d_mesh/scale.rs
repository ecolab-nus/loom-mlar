use mlar_rust::*;

use crate::core_arch::single_core;
use crate::dimensions::{dim_x, dim_y};

/// Scale a single core to an 8×8 mesh with the "perf" name variant.
pub fn scaled_mesh() -> Architecture {
    let core = single_core();
    core.scale([&dim_x(), &dim_y()]).with_name("2d_mesh_perf")
}

/// Scale a single core to an 8×8 mesh and add torus interconnects.
///
/// Torus links connect neighboring L1 caches with wraparound:
/// - horizontal ring: L1(x, y) → L1(x, (y+1) mod 8)
/// - vertical ring:   L1(x, y) → L1((x+1) mod 8, y)
pub fn scaled_mesh_torus() -> Architecture {
    let core = single_core();
    let dim_x = dim_x();
    let dim_y = dim_y();
    let mut mesh = core.scale([&dim_x, &dim_y]).with_name("2d_mesh_torus");

    let scaled_l1 = mesh.get_memory_region("l1").unwrap().clone();

    // Horizontal torus: y-neighbor with wraparound
    let torus_y_map = AffineMapTemplate::parse("[x, y] -> [x, y]: (x, (y + 1) mod 8)")
        .expect("invalid affine map")
        .bind([&dim_x, &dim_y])
        .expect("failed to bind");

    let torus_y = Link::builder("l1_torus_y")
        .from_mem(&scaled_l1)
        .to_mem(&scaled_l1)
        .map(&torus_y_map)
        .bandwidth(64)
        .build();

    // Vertical torus: x-neighbor with wraparound
    let torus_x_map = AffineMapTemplate::parse("[x, y] -> [x, y]: ((x + 1) mod 8, y)")
        .expect("invalid affine map")
        .bind([&dim_x, &dim_y])
        .expect("failed to bind");

    let torus_x = Link::builder("l1_torus_x")
        .from_mem(&scaled_l1)
        .to_mem(&scaled_l1)
        .map(&torus_x_map)
        .bandwidth(64)
        .build();

    mesh.links.push(torus_y);
    mesh.links.push(torus_x);
    mesh
}
