pub mod arch;
pub mod math;
pub mod schedule;
pub mod visualization;

// Re-export commonly used architecture types
pub use arch::{
    ArchEdge, ArchGraph, ArchGraphBuilder, ArchGraphNode, ArchNode, ArchNodeComponent, Architecture,
    Dimension, Endpoint, FuncPerfModel, FunctionProcessor, MemoryBank,
    MemoryRegion, PerfScenario, PerfScenarios, Processor, ProcessorSet, Processors, Router,
    RouterEndpoint, RouterEndpointTarget, RouterSide, ScaleOutNetwork, SharingDomain, SizeExpr,
    Sym, TimeCostExpr, TimeExpr,
};

// Re-export commonly used math types
pub use math::{
    AffineExpr, AffineMap, AffineMapTemplate, ConstraintExpr, Expr, IndexExpr, IndexSelector,
    ParseError,
};
pub use schedule::{
    MLIRFunc, MLIRFuncRef, MLIRModuleRef, MlirFunc, MlirFuncDetails, MlirModule,
    MlirTensorSymbolBinding, Module, ModuleSource, PerfResult, Schedule, ScheduleWithSymMap,
    SymbolicMapping, evaluate, evaluate_with_sym_map,
};

// Re-export visualization utilities
pub use visualization::graph_json::{
    ArchitectureGraphJson, architecture_to_graph_json, architecture_to_graph_json_string,
    architecture_to_graph_json_string_pretty, architecture_to_graph_json_value,
};
