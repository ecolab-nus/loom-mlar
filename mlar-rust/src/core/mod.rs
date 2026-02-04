pub mod size_dim;
pub mod memory;
pub mod processor;

// Re-export commonly used types
pub use size_dim::{Size, Dimension, Index};
pub use memory::{Bank, MemRegion, AggregationType};
pub use processor::{Processor, PerformanceModel};
