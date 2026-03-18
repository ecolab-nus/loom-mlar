pub mod arch;
pub mod evaluator;
pub mod math;
pub mod schedule;
pub mod visualization;

// Re-export commonly used architecture types
pub use arch::{
    ArchEdge, ArchEdgeAttr, ArchEdgeId, ArchGraph, ArchGraphBuilder, ArchGraphNode, ArchNode,
    ArchNodeComponent, ArchNodeId, Architecture, Dimension, FuncPerfModel, FunctionProcessor,
    MemoryBank, MemoryRegion, MeshNetwork, PerfScenario, Processor, ProcessorSet, Processors,
    Router, RouterSide, ScaleOutNetwork, SimpleTimeCost,
    SizeExpr, Sym, TimeCost, TimeExpr,
};

// Re-export commonly used math types
pub use math::{
    AffineExpr, AffineMap, AffineMapTemplate, ConstraintExpr, Expr, IndexExpr, IndexSelector,
    ParseError,
};
pub use schedule::{
    MLIRFunc, MLIRFuncRef, MLIRModuleRef, MlirFunc, MlirFuncDetails, MlirModule,
    MlirTensorSymbolBinding, Module, ModuleSource, Schedule, SymbolicMapping, evaluate,
};

// Re-export visualization utilities
pub use visualization::graph_json::{
    ArchitectureGraphJson, architecture_to_graph_json, architecture_to_graph_json_string,
    architecture_to_graph_json_string_pretty, architecture_to_graph_json_value,
};
pub use visualization::hierarchy_json::{
    ArchitectureHierarchyJson, architecture_to_hierarchy_json,
    architecture_to_hierarchy_json_string_pretty, architecture_to_hierarchy_json_value,
};

// Re-export evaluator utilities
pub use evaluator::{generate_evaluator_binary, run_evaluator, run_evaluator_from_json};
