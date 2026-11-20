//! Blackhole: upstream wormhole fixture with its grid, DRAM, gather and matmul rates changed; unverified.
#[path = "../tt_mesh/mod.rs"]
mod tt_mesh;

use tt_mesh::{Chip, Perf};

const DIR: &str = "tests/blackhole";

const CHIP: Chip = Chip {
    x: 12,
    y: 10,
    dram_channels: 8,
    l1_bytes: 1_398_784,
    perf: Perf {
        matrix_lane: include_str!("perf/matrix_lane.perf.yaml"),
        matrix_lane_ss: include_str!("perf/matrix_lane_ss.perf.yaml"),
        vector_lane: include_str!("perf/vector_lane.perf.yaml"),
        dram_l1_noc0: include_str!("perf/dram_l1_noc0.perf.yaml"),
        l1_l1_noc0: include_str!("perf/l1_l1_noc0.perf.yaml"),
        l1_dram_noc1: include_str!("perf/l1_dram_noc1.perf.yaml"),
    },
};

#[test]
fn test_export_2d_mesh_torus_mlir() {
    tt_mesh::export_hw_spec(&CHIP, DIR);
}

#[test]
fn test_generate_system_evaluator_binary() {
    tt_mesh::generate_evaluator(&CHIP, DIR);
}
