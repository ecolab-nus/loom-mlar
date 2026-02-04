use super::size_dim::{Dimension, Index};
use super::memory::MemRegion;

/// Trait for all processor types (functional units, lanes, etc.)
/// Processors operate on memory regions, transforming data from input memories to output memories
pub trait Processor {
    /// Get the name of this processor
    fn name(&self) -> &str;
    
    /// Get the input memory regions this processor reads from
    fn input_memories(&self) -> &[MemRegion];
    
    /// Get the output memory regions this processor writes to
    fn output_memories(&self) -> &[MemRegion];
    
    /// Get the grid dimensions where this processor is replicated
    fn grid(&self) -> &[Dimension];
}

/// Trait for performance models that compute latency
pub trait PerformanceModel {
    fn compute_latency(&self, dims: &[Index], inputs: &[MemRegion]) -> Index;
}
