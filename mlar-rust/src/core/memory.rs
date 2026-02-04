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

/// Defines how memory regions are aggregated and exposed
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationType {
    /// All sub-regions connected via a bus, single port exposed
    Bus,
    /// Each sub-region exposed as separate port
    Separate,
}

/// Represents a hierarchical memory region
#[derive(Debug, Clone)]
pub enum MemRegion {
    /// Non-leaf: indexed region containing sub-regions
    Indexed {
        indices: Vec<Dimension>,
        sub_region: Box<MemRegion>,
        aggregation: AggregationType,
    },
    /// Leaf: concrete memory block
    Bank(Bank),
}

impl MemRegion {
    /// Create an indexed memory region with specified aggregation type
    pub fn indexed(
        indices: Vec<Dimension>,
        sub_region: MemRegion,
        aggregation: AggregationType,
    ) -> Self {
        MemRegion::Indexed {
            indices,
            sub_region: Box::new(sub_region),
            aggregation,
        }
    }

    /// Create an indexed memory region with bus aggregation (single port exposed)
    pub fn indexed_bus(indices: Vec<Dimension>, sub_region: MemRegion) -> Self {
        Self::indexed(indices, sub_region, AggregationType::Bus)
    }

    /// Create an indexed memory region with separate ports (each sub-region exposed)
    pub fn indexed_separate(indices: Vec<Dimension>, sub_region: MemRegion) -> Self {
        Self::indexed(indices, sub_region, AggregationType::Separate)
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
