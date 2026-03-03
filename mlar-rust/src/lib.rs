pub mod core;
pub mod utils;

// Re-export commonly used types from core
pub use core::{
    AffineExpr, AffineMap, AffineMapTemplate, Architecture, ArchitectureBuilder, ArchitectureLabel,
    ConstraintExpr, TimeCostExpr, Dimension, Endpoint, Expr, IndexExpr, IndexSelector, Link,
    MemoryBank, MemoryRegion, MlirModuleRef, ParseError, PerfModel, PrimitiveProc, Processor,
    Resource, ResourceReq, SharingDomain, SizeExpr, Symbol,
};

// Re-export visualization utilities
pub use utils::visualization::{ArchVisualizer, architecture_to_dot, memory_hierarchy_to_dot};
