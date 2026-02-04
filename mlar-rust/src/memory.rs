use crate::core::{Index, MemRegion};

/// Represents a memory resource (mlar.memory)
#[derive(Debug, Clone)]
pub struct Memory {
    pub name: String,
    pub region: MemRegion,        // Hierarchical memory region
    pub bandwidth: usize,          // Bandwidth in bytes/cycle
    pub capacity: usize,           // Computed total capacity
}

impl Memory {
    pub fn builder(name: impl Into<String>) -> MemoryBuilder {
        MemoryBuilder {
            name: name.into(),
            region: None,
            bandwidth: 0,
        }
    }

    /// Compute the latency for transferring a given amount of data
    pub fn transfer_latency(&self, data_size: usize) -> Index {
        if self.bandwidth == 0 {
            return 0;
        }
        (data_size + self.bandwidth - 1) / self.bandwidth // Ceiling division
    }
}

pub struct MemoryBuilder {
    name: String,
    region: Option<MemRegion>,
    bandwidth: usize,
}

impl MemoryBuilder {
    pub fn region(mut self, region: MemRegion) -> Self {
        self.region = Some(region);
        self
    }

    pub fn bandwidth(mut self, bytes_per_cycle: usize) -> Self {
        self.bandwidth = bytes_per_cycle;
        self
    }

    pub fn build(self) -> Memory {
        let region = self.region.expect("Memory must have a region");
        let capacity = compute_region_capacity(&region);
        
        Memory {
            name: self.name,
            region,
            bandwidth: self.bandwidth,
            capacity,
        }
    }
}

/// Compute the total capacity of a memory region
fn compute_region_capacity(region: &MemRegion) -> usize {
    match region {
        MemRegion::Leaf(block) => {
            // Multiply block_size * num_blocks (only if both are concrete)
            match (block.block_size.as_concrete(), block.num_blocks.as_concrete()) {
                (Some(size), Some(count)) => size * count,
                _ => 0, // Symbolic capacity is unknown, return 0
            }
        }
        MemRegion::Indexed { indices, sub_region } => {
            // Multiply index counts by sub-region capacity
            let index_count: usize = indices.iter()
                .filter_map(|d| d.size.as_concrete())
                .product();
            
            index_count * compute_region_capacity(sub_region)
        }
    }
}
