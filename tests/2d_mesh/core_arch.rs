use mlar_rust::*;

use crate::dimensions::dim_bank;
use crate::matrix_lane::matrix_lane;
use crate::memory::l1;
use crate::vector_lane::vector_lane;

/// Build a single core: L1 + matrix_lane + vector_lane with all-to-one links.
pub fn single_core() -> Architecture {
    let l1 = l1();
    let matrix_lane = matrix_lane();
    let vector_lane = vector_lane();

    let all_to_one_map = AffineMap::new(dim_bank().as_slice(), &[], vec![]);

    let l1_to_matrix = Link::builder("L1_to_matrix_lane")
        .from_mem(&l1)
        .to_proc(&matrix_lane)
        .map(&all_to_one_map)
        .bandwidth(512)
        .build();

    let l1_to_vector = Link::builder("L1_to_vector_lane")
        .from_mem(&l1)
        .to_proc(&vector_lane)
        .map(&all_to_one_map)
        .bandwidth(128)
        .build();

    Architecture::builder("core")
        .mem(&l1)
        .processor(&matrix_lane)
        .processor(&vector_lane)
        .link(l1_to_matrix)
        .link(l1_to_vector)
        .build()
}
