pub mod abi;
pub mod arch;
pub mod math;
pub mod mlir;
pub mod schedule;
pub mod visualization;

pub use abi::{arch_query, evaluator};

// Re-export commonly used architecture types
pub use arch::{
    Architecture, ComputeProcessor, ComputeProcessorBuilder, DataEffect, DataMover,
    DataMoverBuilder, Dimension, FuncPerfModel, FuncPerfModelBuilder, FunctionDataMover,
    FunctionProcessor, MemoryBank, MemoryRegion, MemoryRegionRef, MeshLink, MeshNetwork,
    MeshNetworkInterface, PerfFunctionYaml, PerfScenario, PerfScenarioYaml, PerfYamlError,
    PerfYamlSpec, Processor, ProcessorModule, Resource, ResourceId, ScaleOutNetwork,
    ScaleOutNetworkBindings, SimpleCostYaml, SimpleTimeCost, SizeExpr, Sym, TimeCost, TimeCostYaml,
    TimeExpr,
};
pub use mlir::{
    MlirExportError, architecture_to_mlir, architecture_to_mlir_unchecked,
    mlir_validators_available,
};

// Re-export commonly used math types
pub use math::{
    AffineExpr, AffineMap, AffineMapTemplate, ConstraintExpr, Expr, IndexExpr, IndexSelector,
    ParseError,
};
pub use schedule::{
    MLIRFunc, MLIRFuncRef, MLIRModuleRef, MlirBroadcastDim, MlirCopyOp, MlirFunc, MlirFuncDetails,
    MlirGatherOp, MlirMemRegionBinding, MlirMemrefSymbolBinding, MlirModule,
    MlirTensorSymbolBinding, Schedule, SymbolicMapping, evaluate,
};

// Re-export the renderer-independent visualization document API.
pub use visualization::document::{
    VISUALIZATION_SCHEMA_VERSION, VisualizationAffineMap, VisualizationArchitecture,
    VisualizationComponent, VisualizationDataEffect, VisualizationDimension,
    VisualizationDocumentV1, VisualizationExportError, VisualizationMemoryRegion,
    VisualizationNetworkKind, VisualizationNetworkLink, VisualizationRelationship,
    VisualizationRelationshipKind, VisualizationResourceKind, VisualizationScope,
    VisualizationSignedExpression, VisualizationUnsignedExpression,
    architecture_to_visualization_document, architecture_to_visualization_yaml,
};

// Re-export evaluator utilities
pub use abi::arch_query::{
    ArchitectureQuery, ArchitectureQueryResult, generate_arch_query_binary, query_architecture,
    run_arch_query, run_arch_query_from_json,
};
pub use abi::evaluator::{generate_evaluator_binary, run_evaluator, run_evaluator_from_json};
