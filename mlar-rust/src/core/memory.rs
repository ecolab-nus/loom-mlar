use super::size_dim::{Size, Dimension};
use crate::interconnect::AffineMap;
use crate::processor_aggregation::ProcessorSet;

/// Represents a concrete block of memory
#[derive(Debug, Clone)]
pub struct Bank {
    pub block_size: Size,   // Size of each block
    pub num_blocks: Size,   // Number of blocks
}

impl Bank {
    pub fn new(block_size: Size, num_blocks: Size) -> Self {
        Self {
            block_size,
            num_blocks,
        }
    }
}

/// Memory aggregation - acts as a processor moving data between regions
#[derive(Debug, Clone)]
pub struct MemoryInterface {
    pub name: String,
    pub sources: Vec<MemRegion>,
    pub target: MemRegion,
    pub bandwidth: usize,  // bytes/cycle
}

/// Memory interconnect - maps sub-regions between memory regions
#[derive(Debug, Clone)]
pub struct MemoryInterconnects {
    pub name: String,
    pub sources: Vec<MemRegion>,
    pub targets: Vec<MemRegion>,
    /// Affine map over source indices to target indices
    pub map: AffineMap,
    pub bandwidth: usize, // bytes/cycle
}

impl MemoryInterconnects {
    pub fn new(
        name: impl Into<String>,
        sources: Vec<MemRegion>,
        targets: Vec<MemRegion>,
        map: AffineMap,
        bandwidth: usize,
    ) -> Self {
        Self {
            name: name.into(),
            sources,
            targets,
            map,
            bandwidth,
        }
    }
}

/// Memory-to-processor interconnect - maps memory sub-regions to processors
#[derive(Debug, Clone)]
pub struct MemoryProcessorInterconnect {
    pub name: String,
    pub source: MemRegion,
    pub target: ProcessorSet,
    /// Affine map over source indices to processor indices
    pub map: AffineMap,
    pub bandwidth: usize, // bytes/cycle
}

impl MemoryProcessorInterconnect {
    pub fn new(
        name: impl Into<String>,
        source: MemRegion,
        target: ProcessorSet,
        map: AffineMap,
        bandwidth: usize,
    ) -> Self {
        Self {
            name: name.into(),
            source,
            target,
            map,
            bandwidth,
        }
    }
}

impl MemoryInterface {
    pub fn new(
        name: impl Into<String>,
        sources: Vec<MemRegion>,
        target: MemRegion,
        bandwidth: usize,
    ) -> Self {
        Self {
            name: name.into(),
            sources,
            target,
            bandwidth,
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
    /// Leaf: concrete memory bank
    Bank(Bank),
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
    pub fn bank(bank: Bank) -> Self {
        MemRegion::Bank(bank)
    }

    /// Convenience: create a leaf with concrete sizes
    pub fn leaf_concrete(block_size: usize, num_blocks: usize) -> Self {
        MemRegion::Bank(Bank {
            block_size: Size::int(block_size),
            num_blocks: Size::int(num_blocks),
        })
    }

    /// Scale this memory region across the given dimensions.
    /// Creates a new indexed region wrapping this one.
    pub fn scale<'a, I>(self, indices: I) -> Self
    where
        I: IntoIterator<Item = &'a Dimension>,
    {
        let indices = indices.into_iter().cloned().collect();
        MemRegion::Indexed {
            indices,
            sub_region: Box::new(self),
        }
    }
}

impl From<&MemRegion> for MemRegion {
    fn from(region: &MemRegion) -> Self {
        region.clone()
    }
}

impl From<&MemoryInterface> for MemoryInterface {
    fn from(interface: &MemoryInterface) -> Self {
        interface.clone()
    }
}

impl From<&MemoryInterconnects> for MemoryInterconnects {
    fn from(interconnects: &MemoryInterconnects) -> Self {
        interconnects.clone()
    }
}

impl From<&MemoryProcessorInterconnect> for MemoryProcessorInterconnect {
    fn from(interconnect: &MemoryProcessorInterconnect) -> Self {
        interconnect.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bank_construction() {
        let mb = Bank {
            block_size: Size::int(1024),
            num_blocks: Size::int(4),
        };
        
        assert!(matches!(mb.block_size, Size::Int(1024)));
        assert!(matches!(mb.num_blocks, Size::Int(4)));

        let mb_sym = Bank {
            block_size: Size::sym("N"),
            num_blocks: Size::sym("M"),
        };
            
        assert!(matches!(mb_sym.block_size, Size::Sym(_)));
        assert!(matches!(mb_sym.num_blocks, Size::Sym(_)));
    }
}
