use std::sync::Arc;
use crate::core::{Dimension, Index, MemRegion, Processor};
use crate::processor_aggregation::{ProcessorSet, Scalable};

/// Represents a lane processor (mlar.lane) - dynamic shapes, streaming operations
#[derive(Debug, Clone)]
pub struct FunctionalLane {
    pub name: String,
    pub input_regions: Vec<MemRegion>,
    pub output_regions: Vec<MemRegion>,
    pub model: Arc<dyn LaneModel>,
}

impl FunctionalLane {
    pub fn new<N, I, O, M>(
        name: N,
        input_regions: I,
        output_regions: O,
        model: M,
    ) -> Self
    where
        N: Into<String>,
        I: IntoIterator,
        I::Item: Into<MemRegion>,
        O: IntoIterator,
        O::Item: Into<MemRegion>,
        M: LaneModel + 'static,
    {
        Self {
            name: name.into(),
            input_regions: input_regions.into_iter().map(Into::into).collect(),
            output_regions: output_regions.into_iter().map(Into::into).collect(),
            model: Arc::new(model),
        }
    }

    pub fn compute_latency(&self, dims: &[Index], inputs: &[MemRegion]) -> Result<Index, String> {
        // Validate preconditions before computing latency
        self.model.validate_preconditions(dims)?;
        Ok(self.model.compute_latency(dims, inputs))
    }
}

impl Scalable for FunctionalLane {
    fn scale(self, indices: Vec<Dimension>) -> ProcessorSet {
        ProcessorSet::from_lane_indexed(self, indices)
    }
}

impl Processor for FunctionalLane {
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

/// Trait for lane models with precondition validation
pub trait LaneModel: std::fmt::Debug + Send + Sync {
    /// Validate that the preconditions for the performance model are met
    fn validate_preconditions(&self, dims: &[Index]) -> Result<(), String>;
    
    /// Compute the latency for this lane given dimensions and inputs
    fn compute_latency(&self, dims: &[Index], inputs: &[MemRegion]) -> Index;
}

// Example: Matrix lane processor for large MxNxK matmul
#[derive(Debug)]
pub struct MatMulLane;

impl LaneModel for MatMulLane {
    fn validate_preconditions(&self, dims: &[Index]) -> Result<(), String> {
        if dims.len() < 2 {
            return Err("MatMulLane requires at least M and N dimensions".to_string());
        }
        
        let m = dims[0];
        let n = dims[1];
        
        if m < 256 {
            return Err(format!("matmul_lane requires M >= 256, got {}", m));
        }
        if n < 256 {
            return Err(format!("matmul_lane requires N >= 256, got {}", n));
        }
        
        Ok(())
    }

    fn compute_latency(&self, dims: &[Index], _inputs: &[MemRegion]) -> Index {
        // Latency: M * N * K / 64 cycles (streaming at 64 MACs/cycle)
        let m = dims[0];
        let n = dims[1];
        let k = dims.get(2).copied().unwrap_or(1);
        
        (m * n * k) / 64
    }
}

// Example: Vector lane processor for arbitrary-length vectors
#[derive(Debug)]
pub struct VecLane;

impl LaneModel for VecLane {
    fn validate_preconditions(&self, dims: &[Index]) -> Result<(), String> {
        if dims.is_empty() {
            return Err("VecLane requires N dimension".to_string());
        }
        
        let n = dims[0];
        if n < 1024 {
            return Err(format!("vec_lane requires N >= 1024, got {}", n));
        }
        
        Ok(())
    }

    fn compute_latency(&self, dims: &[Index], _inputs: &[MemRegion]) -> Index {
        // Latency: N / 32 cycles (streaming at 32 elements/cycle)
        let n = dims[0];
        n / 32
    }
}
