use mlar_rust::*;

use crate::dimensions::dim_bank;

/// L1 cache: 16 banks, each 128KB (1024 blocks × 128 bytes).
pub fn l1() -> MemoryRegion {
    MemoryRegion::bank(MemoryBank::from_blocks(
        SizeExpr::Const(128),
        SizeExpr::Const(1024),
    ))
    .replicate(dim_bank().as_slice())
    .with_name("l1")
}
