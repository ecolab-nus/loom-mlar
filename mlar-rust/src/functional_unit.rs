use crate::memory::Memory;
use crate::core::{Dimension, Index, MemRegion, Processor, PerformanceModel};

/// Represents a functional unit (mlar.fu) - fixed shapes, synchronous operations
#[derive(Debug, Clone)]
pub struct FunctionalUnit {
    pub name: String,
    pub input_regions: Vec<MemRegion>,
    pub output_regions: Vec<MemRegion>,
    pub latency: Index,
    pub grid: Vec<Dimension>,
}

impl FunctionalUnit {
    pub fn builder(name: impl Into<String>) -> FunctionalUnitBuilder {
        FunctionalUnitBuilder {
            name: name.into(),
            input_regions: Vec::new(),
            output_regions: Vec::new(),
            latency: 0,
            grid: Vec::new(),
        }
    }
}

impl Processor for FunctionalUnit {
    fn name(&self) -> &str {
        &self.name
    }

    fn input_memories(&self) -> &[Memory] {
        // TODO: Extract Memory from MemRegion
        &[]
    }

    fn output_memories(&self) -> &[Memory] {
        // TODO: Extract Memory from MemRegion
        &[]
    }

    fn grid(&self) -> &[Dimension] {
        &self.grid
    }
}

pub struct FunctionalUnitBuilder {
    name: String,
    input_regions: Vec<MemRegion>,
    output_regions: Vec<MemRegion>,
    latency: Index,
    grid: Vec<Dimension>,
}

impl FunctionalUnitBuilder {
    pub fn input_region(mut self, region: MemRegion) -> Self {
        self.input_regions.push(region);
        self
    }

    pub fn output_region(mut self, region: MemRegion) -> Self {
        self.output_regions.push(region);
        self
    }

    pub fn latency(mut self, cycles: Index) -> Self {
        self.latency = cycles;
        self
    }

    pub fn grid(mut self, dims: Vec<Dimension>) -> Self {
        self.grid = dims;
        self
    }

    pub fn build(self) -> FunctionalUnit {
        FunctionalUnit {
            name: self.name,
            input_regions: self.input_regions,
            output_regions: self.output_regions,
            latency: self.latency,
            grid: self.grid,
        }
    }
}

// Example: Matrix multiplication functional unit (32x32 tiles)
pub struct MatMul32x32;

impl PerformanceModel for MatMul32x32 {
    fn compute_latency(&self, _dims: &[Index], _inputs: &[MemRegion]) -> Index {
        8 // Fixed latency of 8 cycles
    }
}

// Example: Vector add functional unit (32-wide vectors)
pub struct VecAdd32;

impl PerformanceModel for VecAdd32 {
    fn compute_latency(&self, _dims: &[Index], _inputs: &[MemRegion]) -> Index {
        1 // Fixed latency of 1 cycle
    }
}
