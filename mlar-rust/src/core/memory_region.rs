use super::size_dim::{Size, Dimension};

/// Represents a concrete block of memory
#[derive(Debug, Clone)]
pub struct MemoryBlock {
    pub block_size: Size,   // Size of each block
    pub num_blocks: Size,   // Number of blocks
}

impl MemoryBlock {
    pub fn new(block_size: Size, num_blocks: Size) -> Self {
        Self {
            block_size,
            num_blocks,
        }
    }

    /// Create a memory block with concrete sizes (convenience method)
    pub fn new_concrete(block_size: usize, num_blocks: usize) -> Self {
        Self {
            block_size: Size::Int(block_size),
            num_blocks: Size::Int(num_blocks),
        }
    }
}

/// Represents a hierarchical memory region
#[derive(Debug, Clone)]
pub enum MemRegion {
    /// Non-leaf: indexed region containing sub-regions
    Indexed {
        indices: Vec<Dimension>,
        sub_region: Box<MemRegion>,
    },
    /// Leaf: concrete memory block
    Leaf(MemoryBlock),
}

impl MemRegion {
    /// Create an indexed memory region
    pub fn indexed(indices: Vec<Dimension>, sub_region: MemRegion) -> Self {
        MemRegion::Indexed {
            indices,
            sub_region: Box::new(sub_region),
        }
    }

    /// Create a leaf memory region
    pub fn leaf(block: MemoryBlock) -> Self {
        MemRegion::Leaf(block)
    }

    /// Convenience: create a leaf with concrete sizes
    pub fn leaf_concrete(block_size: usize, num_blocks: usize) -> Self {
        MemRegion::Leaf(MemoryBlock::new_concrete(block_size, num_blocks))
    }
}
