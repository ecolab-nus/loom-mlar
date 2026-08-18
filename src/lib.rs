pub mod abi;
pub mod arch;
pub mod archs;
pub mod math;
pub mod mlir;
pub mod schedule;
pub mod visualization;

pub use abi::{arch_query, evaluator};

// Common architecture authoring surface. Derived/query and loader-specific
// types remain under `arch`.
pub use arch::{
    AdlExportError, Architecture, ArchitectureBuilder, ArchitectureError, Axis, Banking,
    Connection, FuncPerfModel, MemoryAlias, MemoryDefinition, MemoryEndpoint, MemoryTechnology,
    NetworkInterface, NetworkLink, NetworkTopology, OperationModel, PerfScenario,
    ProcessorDefinition, ProcessorSelector, ProcessorSourceFormat, ProcessorType,
    ResolvedEndpointIndex, Resource, Scope, TimeCost, architecture_to_mlir,
    architecture_to_mlir_unchecked, mlir_validators_available,
};

// Re-export commonly used math types
pub use math::{AffineError, AffineExpr, AffineMap, ConstraintExpr, Expr, ParseError, Sym};
pub use mlir::{LoomParseError, MlirFunc, MlirModule, parse_loom_source};
pub use schedule::{ProcessorTarget, Schedule, SymbolicMapping, evaluate};

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
