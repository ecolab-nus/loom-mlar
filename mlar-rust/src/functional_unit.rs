use crate::core::{Index, MemRegion, Processor, PerformanceModel};

/// Represents a functional unit (mlar.fu) - fixed shapes, synchronous operations
#[derive(Debug, Clone)]
pub struct FunctionalUnit {
    pub name: String,
    pub input_regions: Vec<MemRegion>,
    pub output_regions: Vec<MemRegion>,
    pub latency: Index,
}

impl FunctionalUnit {
    pub fn builder(name: impl Into<String>) -> FunctionalUnitBuilder {
        FunctionalUnitBuilder {
            name: name.into(),
            input_regions: Vec::new(),
            output_regions: Vec::new(),
            latency: 0,
        }
    }
}

impl Processor for FunctionalUnit {
    fn name(&self) -> &str {
        &self.name
    }

    fn input_memories(&self) -> &[MemRegion] {
        &self.input_regions
    }

    fn output_memories(&self) -> &[MemRegion] {
        &self.output_regions
    }
}

pub struct FunctionalUnitBuilder {
    name: String,
    input_regions: Vec<MemRegion>,
    output_regions: Vec<MemRegion>,
    latency: Index,
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

    pub fn build(self) -> FunctionalUnit {
        FunctionalUnit {
            name: self.name,
            input_regions: self.input_regions,
            output_regions: self.output_regions,
            latency: self.latency,
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
