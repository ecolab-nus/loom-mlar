pub mod primitives;
pub mod functional_unit;
pub mod lane;
pub mod memory;
pub mod interconnect;
pub mod architecture;

// Re-export commonly used types
pub use primitives::{Dimension, Index, MemRef, Shape, Size, PerformanceModel};
pub use functional_unit::FunctionalUnit;
pub use lane::Lane;
pub use memory::Memory;
pub use interconnect::{Interconnect, AffineMap, AffineExpr};
pub use architecture::Architecture;
