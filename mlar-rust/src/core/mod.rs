pub mod affine;
pub mod architecture;
pub mod constraint;
pub mod expr;
pub mod link;
pub mod memory;
pub mod parse;
pub mod perf;
pub mod processor;
pub mod resource;
pub mod size_dim;

// Re-export commonly used types
pub use affine::{AffineExpr, AffineMap, AffineMapTemplate, IndexExpr, IndexSelector};
pub use architecture::{Architecture, ArchitectureBuilder, ArchitectureLabel};
pub use constraint::ConstraintExpr;
pub use expr::Expr;
pub use link::{Endpoint, Link, SharingDomain};
pub use memory::{MemoryBank, MemoryRegion};
pub use parse::ParseError;
pub use perf::{TimeCostExpr, PerfModel};
pub use processor::{MlirModuleRef, PrimitiveProc, Processor};
pub use resource::{Resource, ResourceReq};
pub use size_dim::{Dimension, SizeExpr, Symbol};
