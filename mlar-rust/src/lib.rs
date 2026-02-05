pub mod core;
pub mod processor_aggregation;
pub mod functional_unit;
pub mod lane;
pub mod interconnect;
pub mod architecture;
pub mod visualization;

// Re-export commonly used types from core
pub use core::{Dimension, Index, MemRegion, Bank, Size, Processor, PerformanceModel, MemoryAggregation};

// Re-export commonly used types from modules
pub use functional_unit::FunctionalUnit;
pub use lane::FunctionalLane;
pub use interconnect::{Interconnect, AffineMap, AffineExpr};
pub use architecture::Architecture;
pub use processor_aggregation::{ProcessorSet, ProcessorAggregation, Scalable};

// Re-export visualization utilities
pub use visualization::{ArchVisualizer, architecture_to_dot, memory_hierarchy_to_dot};
