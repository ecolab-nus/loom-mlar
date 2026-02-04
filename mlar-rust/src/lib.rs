pub mod core;
pub mod functional_unit;
pub mod lane;
pub mod interconnect;
pub mod architecture;

// Re-export commonly used types from core
pub use core::{Dimension, Index, MemRegion, Bank, Size, Processor, PerformanceModel, AggregationType};

// Re-export commonly used types from modules
pub use functional_unit::FunctionalUnit;
pub use lane::Lane;
pub use interconnect::{Interconnect, AffineMap, AffineExpr};
pub use architecture::Architecture;
