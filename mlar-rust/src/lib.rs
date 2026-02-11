pub mod core;
pub mod architecture;
pub mod visualization;

// Re-export commonly used types from core
pub use core::{
    Symbol, SizeExpr, Dimension,
    Expr, ConstraintExpr, ParseError, PerfModel, CostExpr,
    AffineExpr, AffineMap, AffineMapTemplate, IndexExpr, IndexSelector,
    MemoryBank, MemoryRegion,
    Processor, PrimitiveProc,
    Link, Endpoint, SharingDomain,
};

// Re-export architecture types
pub use architecture::{Architecture, ArchitectureBuilder};

// Re-export visualization utilities
pub use visualization::{ArchVisualizer, architecture_to_dot, architecture_to_dot_expanded, memory_hierarchy_to_dot};
