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
        Self::builder()
            .block_size(block_size)
            .num_blocks(num_blocks)
            .build()
    }
    
    pub fn builder() -> BankBuilder {
        BankBuilder::default()
    }
}

#[derive(Default)]
pub struct BankBuilder {
    block_size: Option<Size>,
    num_blocks: Option<Size>,
}

impl BankBuilder {
    pub fn block_size(mut self, block_size: impl Into<Size>) -> Self {
        self.block_size = Some(block_size.into());
        self
    }

    pub fn num_blocks(mut self, num_blocks: impl Into<Size>) -> Self {
        self.num_blocks = Some(num_blocks.into());
        self
    }

    pub fn build(self) -> Bank {
        Bank {
            block_size: self.block_size.expect("block_size must be set"),
            num_blocks: self.num_blocks.expect("num_blocks must be set"),
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
    pub fn builder(name: impl Into<String>) -> MemoryInterconnectsBuilder {
        MemoryInterconnectsBuilder {
            name: name.into(),
            sources: Vec::new(),
            targets: Vec::new(),
            map: None,
            bandwidth: None,
        }
    }
}

pub struct MemoryInterconnectsBuilder {
    name: String,
    sources: Vec<MemRegion>,
    targets: Vec<MemRegion>,
    map: Option<AffineMap>,
    bandwidth: Option<usize>,
}

impl MemoryInterconnectsBuilder {
    pub fn source(mut self, region: impl Into<MemRegion>) -> Self {
        self.sources.push(region.into());
        self
    }

    pub fn target(mut self, region: impl Into<MemRegion>) -> Self {
        self.targets.push(region.into());
        self
    }

    pub fn affine_map(mut self, map: AffineMap) -> Self {
        self.map = Some(map);
        self
    }

    pub fn bandwidth(mut self, bandwidth: usize) -> Self {
        self.bandwidth = Some(bandwidth);
        self
    }

    pub fn build(self) -> MemoryInterconnects {
        MemoryInterconnects {
            name: self.name,
            sources: self.sources,
            targets: self.targets,
            map: self.map.expect("map must be set"),
            bandwidth: self.bandwidth.expect("bandwidth must be set"),
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
    pub fn builder(name: impl Into<String>) -> MemoryProcessorInterconnectBuilder {
        MemoryProcessorInterconnectBuilder {
            name: name.into(),
            source: None,
            target: None,
            map: None,
            bandwidth: None,
        }
    }
}

pub struct MemoryProcessorInterconnectBuilder {
    name: String,
    source: Option<MemRegion>,
    target: Option<ProcessorSet>,
    map: Option<AffineMap>,
    bandwidth: Option<usize>,
}

impl MemoryProcessorInterconnectBuilder {
    pub fn source(mut self, region: impl Into<MemRegion>) -> Self {
        self.source = Some(region.into());
        self
    }

    pub fn target(mut self, set: ProcessorSet) -> Self {
        self.target = Some(set);
        self
    }

    pub fn affine_map(mut self, map: AffineMap) -> Self {
        self.map = Some(map);
        self
    }

    pub fn bandwidth(mut self, bandwidth: usize) -> Self {
        self.bandwidth = Some(bandwidth);
        self
    }

    pub fn build(self) -> MemoryProcessorInterconnect {
        MemoryProcessorInterconnect {
            name: self.name,
            source: self.source.expect("source must be set"),
            target: self.target.expect("target must be set"),
            map: self.map.expect("map must be set"),
            bandwidth: self.bandwidth.expect("bandwidth must be set"),
        }
    }
}

impl MemoryInterface {
    pub fn builder(name: impl Into<String>) -> MemoryInterfaceBuilder {
        MemoryInterfaceBuilder {
            name: name.into(),
            sources: Vec::new(),
            target: None,
            bandwidth: None,
        }
    }
}

pub struct MemoryInterfaceBuilder {
    name: String,
    sources: Vec<MemRegion>,
    target: Option<MemRegion>,
    bandwidth: Option<usize>,
}

impl MemoryInterfaceBuilder {
    pub fn source(mut self, region: impl Into<MemRegion>) -> Self {
        self.sources.push(region.into());
        self
    }

    pub fn target(mut self, region: impl Into<MemRegion>) -> Self {
        self.target = Some(region.into());
        self
    }

    pub fn bandwidth(mut self, bandwidth: usize) -> Self {
        self.bandwidth = Some(bandwidth);
        self
    }

    pub fn build(self) -> MemoryInterface {
        MemoryInterface {
            name: self.name,
            sources: self.sources,
            target: self.target.expect("target must be set"),
            bandwidth: self.bandwidth.expect("bandwidth must be set"),
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
        MemRegion::Bank(
            Bank::builder()
                .block_size(block_size)
                .num_blocks(num_blocks)
                .build(),
        )
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
    fn test_bank_builder() {
        let mb = Bank::builder()
            .block_size(1024 as usize)
            .num_blocks(4 as usize)
            .build();
        
        assert!(matches!(mb.block_size, Size::Int(1024)));
        assert!(matches!(mb.num_blocks, Size::Int(4)));

        let mb_sym = Bank::builder()
            .block_size("N")
            .num_blocks("M")
            .build();
            
        assert!(matches!(mb_sym.block_size, Size::Sym(_)));
        assert!(matches!(mb_sym.num_blocks, Size::Sym(_)));
    }
}
