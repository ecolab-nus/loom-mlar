pub mod size_dim;
pub mod memory_region;
pub mod processor;

// Re-export commonly used types
pub use size_dim::{Size, Dimension, Index};
pub use memory_region::{MemoryBlock, MemRegion};
pub use processor::{Processor, PerformanceModel};
