pub mod arch;
pub mod math;
pub mod visualization;
pub mod schedule;

// Re-export commonly used architecture types
pub use arch::{
    Architecture, ArchitectureBuilder, ArchitectureLabel, Dimension, Endpoint, FuncPerfModel, Link,
    MemoryBank, MemoryRegion, MlirModuleRef, PerfScenario, ProcPerfModel, Processor, ProcessorElem,
    Resource, ResourceReq, SharingDomain, SizeExpr, Sym, TimeCostExpr,
};

// Re-export commonly used math types
pub use math::{
    AffineExpr, AffineMap, AffineMapTemplate, ConstraintExpr, Expr, IndexExpr, IndexSelector,
    ParseError,
};

// Re-export visualization utilities
pub use visualization::graph_json::{
    ArchitectureGraphJson, architecture_to_graph_json, architecture_to_graph_json_string,
    architecture_to_graph_json_string_pretty, architecture_to_graph_json_value,
};
