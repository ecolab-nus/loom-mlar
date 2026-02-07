pub mod size_dim;
pub mod memory;
pub mod processor;
pub mod affine;

// Re-export commonly used types
pub use size_dim::{Size, Dimension, Index};
pub use memory::{Bank, MemRegion, MemoryInterface, MemoryInterconnects, MemoryProcessorInterconnect};
pub use processor::{Processor, PerformanceModel};
pub use affine::{AffineExpr, AffineMap, AffineMapBuilder, AffineMapTemplate};
