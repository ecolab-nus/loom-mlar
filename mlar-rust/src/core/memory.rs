use super::size_dim::{Size, Dimension};

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
pub struct MemoryAggregation {
    pub name: String,
    pub sources: Vec<MemRegion>,
    pub target: MemRegion,
    pub bandwidth: usize,  // bytes/cycle
}

impl MemoryAggregation {
    pub fn builder(name: impl Into<String>) -> MemoryAggregationBuilder {
        MemoryAggregationBuilder {
            name: name.into(),
            sources: Vec::new(),
            target: None,
            bandwidth: None,
        }
    }
}

pub struct MemoryAggregationBuilder {
    name: String,
    sources: Vec<MemRegion>,
    target: Option<MemRegion>,
    bandwidth: Option<usize>,
}

impl MemoryAggregationBuilder {
    pub fn source(mut self, region: MemRegion) -> Self {
        self.sources.push(region);
        self
    }

    pub fn target(mut self, region: MemRegion) -> Self {
        self.target = Some(region);
        self
    }

    pub fn bandwidth(mut self, bandwidth: usize) -> Self {
        self.bandwidth = Some(bandwidth);
        self
    }

    pub fn build(self) -> MemoryAggregation {
        MemoryAggregation {
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
