pub mod abi;
pub mod arch;
pub mod archs;
pub mod math;
pub mod mlir;
pub mod schedule;
pub mod visualization;

pub use abi::{arch_query, evaluator};

// Re-export commonly used architecture types
pub use arch::{
    AdlExportError, AffineExpression, AffineRelation, ArchLoadError, Architecture,
    ArchitectureBuilder, ArchitectureError, Banking, ChipYaml, ConnectionSpec, Dimension,
    EndpointIndex, EndpointParseError, FuncPerfModel, FuncPerfModelBuilder, FunctionProcessor,
    IndexDomain, MemoryArray, MemoryCatalog, MemoryDefinition, MemoryEndpoint, NamedMemoryRegion,
    PerfScenario, PerfYamlError, PerfYamlSpec, ProcessorArray, ProcessorDefinition,
    ProcessorSelection, ProcessorSelectionError, ProcessorSelector, ProcessorType, ProcessorYaml,
    ResolvedConnection, ResolvedMemoryEndpoint, ResourceArray, SimpleTimeCost, SizeExpr, Sym,
    TimeCost, TimeExpr, architecture_to_mlir,
};

// Re-export commonly used math types
pub use math::{
    AffineExpr, AffineMap, AffineMapTemplate, ConstraintExpr, Expr, IndexExpr, IndexSelector,
    ParseError,
};
pub use mlir::{LoomParseError, parse_loom_source};
pub use schedule::{
    MLIRFunc, MLIRFuncRef, MLIRModuleRef, MlirBroadcastDim, MlirCopyOp, MlirFunc, MlirFuncDetails,
    MlirGatherOp, MlirMemRegionBinding, MlirMemrefSymbolBinding, MlirModule,
    MlirTensorSymbolBinding, Schedule, SymbolicMapping, evaluate,
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
pub use visualization::viewer_json::{
    ArchitectureViewerJson, architecture_to_viewer_json, architecture_to_viewer_json_string_pretty,
    architecture_to_viewer_json_value,
};

// Re-export evaluator utilities
pub use abi::arch_query::{
    ArchitectureQuery, ArchitectureQueryResult, generate_arch_query_binary, query_architecture,
    run_arch_query, run_arch_query_from_json,
};
pub use abi::evaluator::{generate_evaluator_binary, run_evaluator, run_evaluator_from_json};
