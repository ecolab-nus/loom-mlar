pub mod core;
pub mod functional_unit;
pub mod lane;
pub mod memory;
pub mod interconnect;
pub mod architecture;

// Re-export commonly used types from core
pub use core::{Dimension, Index, MemRegion, MemoryBlock, Size, Processor, PerformanceModel};

// Re-export commonly used types from modules
pub use functional_unit::FunctionalUnit;
pub use lane::Lane;
pub use memory::Memory;
pub use interconnect::{Interconnect, AffineMap, AffineExpr};
pub use architecture::Architecture;
