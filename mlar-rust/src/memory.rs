use crate::primitives::{Dimension, Index};

/// Represents a memory resource (mlar.memory)
#[derive(Debug)]
pub struct Memory {
    pub name: String,
    pub capacity: usize,      // Total capacity in bytes
    pub bandwidth: usize,      // Bandwidth in bytes/cycle
    pub grid: Vec<Dimension>,  // Grid dimensions (e.g., <x, y>)
}

impl Memory {
    pub fn builder(name: impl Into<String>) -> MemoryBuilder {
        MemoryBuilder {
            name: name.into(),
            capacity: 0,
            bandwidth: 0,
            grid: Vec::new(),
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
    capacity: usize,
    bandwidth: usize,
    grid: Vec<Dimension>,
}

impl MemoryBuilder {
    pub fn capacity(mut self, bytes: usize) -> Self {
        self.capacity = bytes;
        self
    }

    pub fn bandwidth(mut self, bytes_per_cycle: usize) -> Self {
        self.bandwidth = bytes_per_cycle;
        self
    }

    pub fn grid(mut self, dims: Vec<Dimension>) -> Self {
        self.grid = dims;
        self
    }

    pub fn build(self) -> Memory {
        Memory {
            name: self.name,
            capacity: self.capacity,
            bandwidth: self.bandwidth,
            grid: self.grid,
        }
    }
}
