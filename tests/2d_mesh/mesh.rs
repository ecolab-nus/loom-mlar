use mlar_rust::*;

use crate::core_arch::single_core;
use crate::dimensions::{dim_x, dim_y};

pub const TORUS_NAME: &str = "L1_torus";
pub const H_LINK_NAME: &str = "h";
pub const V_LINK_NAME: &str = "v";

/// Resource for the horizontal torus links (y-neighbor with wraparound).
pub fn h_link_resource() -> Resource {
    Resource::exclusive(format!("{TORUS_NAME}_{H_LINK_NAME}"))
}

/// Resource for the vertical torus links (x-neighbor with wraparound).
pub fn v_link_resource() -> Resource {
    Resource::exclusive(format!("{TORUS_NAME}_{V_LINK_NAME}"))
}

/// Build the scaled 8×8 mesh with torus connectivity.
///
/// The torus has two link families:
/// - link 0 (horizontal): L1(x, y) → L1(x, (y+1) mod 8)
/// - link 1 (vertical):   L1(x, y) → L1((x+1) mod 8, y)
///
/// `data_movers` are attached to the mesh IO interface and will be
/// auto-registered in any parent `ArchGraph`.
pub fn scaled_mesh<F>(data_movers: F) -> Architecture
where
    F: FnOnce(&MemoryRegion) -> Vec<DataMover>,
{
    let core = single_core();
    let dim_x = dim_x();
    let dim_y = dim_y();
    let mesh = core.scale([&dim_x, &dim_y]).with_name("mesh");

    let scaled_l1 = mesh
        .get_scaled_memory_region("L1")
        .expect("scaled mesh should expose mesh-wide L1");

    // External IO is only at mesh boundaries: left edge (x = 0) and right edge (x = 7).
    let io_side = Dimension::new_int("io_side", 2);
    let io_map = AffineMapTemplate::parse("[io_side, y] -> [x, y]: (io_side * 7, y)")
        .expect("invalid affine map")
        .bind([&io_side, &dim_x, &dim_y])
        .expect("failed to bind");
    let io = MeshNetworkInterface::new(io_map, Expr::Const(64))
        .with_data_movers(data_movers(&scaled_l1));

    // Horizontal torus: y-neighbor with wraparound
    let h_map = AffineMapTemplate::parse("[x, y] -> [x, y]: (x, (y + 1) mod 8)")
        .expect("invalid affine map")
        .bind([&dim_x, &dim_y])
        .expect("failed to bind");

    // Vertical torus: x-neighbor with wraparound
    let v_map = AffineMapTemplate::parse("[x, y] -> [x, y]: ((x + 1) mod 8, y)")
        .expect("invalid affine map")
        .bind([&dim_x, &dim_y])
        .expect("failed to bind");

    let torus = ScaleOutNetwork::mesh(TORUS_NAME)
        .mem_region(&scaled_l1)
        .link(MeshLink::named(H_LINK_NAME, h_map))
        .link(MeshLink::named(V_LINK_NAME, v_map))
        .io(&io)
        .link_bandwidth(64)
        .build();

    mesh.with_connectivity(vec![torus])
}
