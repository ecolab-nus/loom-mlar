use crate::primitives::{Dimension, Index, MemRef, PerformanceModel};

/// Represents a functional unit (mlar.fu) - fixed shapes, synchronous operations
#[derive(Debug)]
pub struct FunctionalUnit {
    pub name: String,
    pub inputs: Vec<MemRef>,
    pub outputs: Vec<MemRef>,
    pub latency: Index,
    pub grid: Vec<Dimension>,
}

impl FunctionalUnit {
    pub fn builder(name: impl Into<String>) -> FunctionalUnitBuilder {
        FunctionalUnitBuilder {
            name: name.into(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            latency: 0,
            grid: Vec::new(),
        }
    }
}

pub struct FunctionalUnitBuilder {
    name: String,
    inputs: Vec<MemRef>,
    outputs: Vec<MemRef>,
    latency: Index,
    grid: Vec<Dimension>,
}

impl FunctionalUnitBuilder {
    pub fn input(mut self, memref: MemRef) -> Self {
        self.inputs.push(memref);
        self
    }

    pub fn output(mut self, memref: MemRef) -> Self {
        self.outputs.push(memref);
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
            inputs: self.inputs,
            outputs: self.outputs,
            latency: self.latency,
            grid: self.grid,
        }
    }
}

// Example: Matrix multiplication functional unit (32x32 tiles)
pub struct MatMul32x32;

impl PerformanceModel for MatMul32x32 {
    fn compute_latency(&self, _dims: &[Index], _inputs: &[MemRef]) -> Index {
        8 // Fixed latency of 8 cycles
    }
}

// Example: Vector add functional unit (32-wide vectors)
pub struct VecAdd32;

impl PerformanceModel for VecAdd32 {
    fn compute_latency(&self, _dims: &[Index], _inputs: &[MemRef]) -> Index {
        1 // Fixed latency of 1 cycle
    }
}
