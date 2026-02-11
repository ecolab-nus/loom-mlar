pub mod architecture;
pub mod core;
pub mod visualization;

// Re-export commonly used types from core
pub use core::{
    AffineExpr, AffineMap, AffineMapTemplate, ConstraintExpr, CostExpr, Dimension, Endpoint, Expr,
    IndexExpr, IndexSelector, Link, MemoryBank, MemoryRegion, ParseError, PerfModel, PrimitiveProc,
    Processor, SharingDomain, SizeExpr, Symbol,
};

// Re-export architecture types
pub use architecture::{Architecture, ArchitectureBuilder, ArchitectureLabel};

// Re-export visualization utilities
pub use visualization::{ArchVisualizer, architecture_to_dot, memory_hierarchy_to_dot};
