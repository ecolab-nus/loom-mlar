use mlar_rust::*;

use crate::core_arch::single_core;
use crate::dimensions::{dim_x, dim_y};

/// Scale a single core to an 8×8 mesh and add torus interconnects.
///
/// Torus links connect neighboring L1 caches with wraparound:
/// - horizontal ring: L1(x, y) → L1(x, (y+1) mod 8)
/// - vertical ring:   L1(x, y) → L1((x+1) mod 8, y)
pub fn scaled_mesh_torus() -> Architecture {
    let core = single_core();
    let dim_x = dim_x();
    let dim_y = dim_y();
    let mesh = core.scale([&dim_x, &dim_y]).with_name("2d_mesh_torus");

    let scaled_l1 = mesh
        .get_memory_region("L1")
        .expect("scaled mesh should contain L1")
        .clone()
        .scale(&[dim_x.clone(), dim_y.clone()])
        .with_name("L1");

    let io = MeshNetworkInterface::new(
        AffineMap::identity(&[dim_x.clone(), dim_y.clone()]),
        Expr::Const(64),
    );

    // Horizontal torus: y-neighbor with wraparound
    let torus_y_map = AffineMapTemplate::parse("[x, y] -> [x, y]: (x, (y + 1) mod 8)")
        .expect("invalid affine map")
        .bind([&dim_x, &dim_y])
        .expect("failed to bind");

    let torus_y = ScaleOutNetwork::mesh("L1_torus_y")
        .mem_region(&scaled_l1)
        .map(&torus_y_map)
        .io(&io)
        .link_bandwidth(64)
        .build();

    // Vertical torus: x-neighbor with wraparound
    let torus_x_map = AffineMapTemplate::parse("[x, y] -> [x, y]: ((x + 1) mod 8, y)")
        .expect("invalid affine map")
        .bind([&dim_x, &dim_y])
        .expect("failed to bind");

    let torus_x = ScaleOutNetwork::mesh("L1_torus_x")
        .mem_region(&scaled_l1)
        .map(&torus_x_map)
        .io(&io)
        .link_bandwidth(64)
        .build();

    mesh.with_connectivity(vec![torus_y, torus_x])
}
