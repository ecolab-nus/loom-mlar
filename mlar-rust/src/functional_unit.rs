use crate::core::{Dimension, Index, MemRegion, Processor, PerformanceModel};
use crate::processor_aggregation::{ProcessorSet, Scalable};

/// Represents a functional unit (mlar.fu) - fixed shapes, synchronous operations
#[derive(Debug, Clone)]
pub struct FunctionalUnit {
    pub name: String,
    pub input_regions: Vec<MemRegion>,
    pub output_regions: Vec<MemRegion>,
    pub latency: Index,
}

impl Scalable for FunctionalUnit {
    fn scale(self, indices: Vec<Dimension>) -> ProcessorSet {
        ProcessorSet::from_unit_indexed(self, indices)
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
