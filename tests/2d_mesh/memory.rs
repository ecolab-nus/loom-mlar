use mlar_rust::*;

use crate::dimensions::{dim_bank, dim_dram_channel};

/// L1 cache: 16 banks, each 128KB (1024 blocks × 128 bytes).
pub fn l1() -> MemoryRegion {
    MemoryRegion::bank(MemoryBank::from_blocks(
        SizeExpr::Const(16),
        SizeExpr::Const(5856),
    ))
    .scale(dim_bank().as_slice())
    .with_name("L1")
}

/// DRAM: 8 channels, each modeled as one memory bank.
pub fn dram() -> MemoryRegion {
    MemoryRegion::bank(
        MemoryBank::from_blocks(SizeExpr::Const(8192), SizeExpr::Const(196608))
            .with_name("DRAM_bank"),
    )
    .scale(dim_dram_channel().as_slice())
    .with_name("DRAM")
}
