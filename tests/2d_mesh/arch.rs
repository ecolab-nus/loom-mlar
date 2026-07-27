//! Wrappers over `mlar_rust::archs`, pinned to the in-tree processor files (test fixture).

use mlar_rust::Architecture;

const PROCESSOR_DIR: &str = "tests/2d_mesh/processors";

pub fn single_core() -> Architecture {
    mlar_rust::archs::single_core(PROCESSOR_DIR)
}

pub fn scaled_mesh_torus() -> Architecture {
    mlar_rust::archs::scaled_mesh_torus(PROCESSOR_DIR)
}
