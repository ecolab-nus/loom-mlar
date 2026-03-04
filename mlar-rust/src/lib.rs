pub mod core;
pub mod visualization;

// Re-export commonly used types from core
pub use core::{
    AffineExpr, AffineMap, AffineMapTemplate, Architecture, ArchitectureBuilder, ArchitectureLabel,
    ConstraintExpr, Dimension, Endpoint, Expr, FuncPerfModel, IndexExpr, IndexSelector, Link,
    MemoryBank, MemoryRegion, MlirModuleRef, ParseError, PerfScenario, ProcPerfModel, Processor,
    ProcessorElem, Resource, ResourceReq, SharingDomain, SizeExpr, Sym, TimeCostExpr,
};

// Re-export visualization utilities
pub use visualization::graph_json::{
    ArchitectureGraphJson, architecture_to_graph_json, architecture_to_graph_json_string,
    architecture_to_graph_json_string_pretty, architecture_to_graph_json_value,
};
