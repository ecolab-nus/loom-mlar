pub mod abi;
pub mod arch;
pub mod math;
pub mod mlir;
pub mod schedule;
pub mod visualization;

pub use abi::{arch_query, evaluator};

// Common core model surface. Derived/query types remain under `arch`.
pub use arch::{
    AdlExportError, Architecture, ArchitectureBuilder, ArchitectureError, Axis, Banking,
    Connection, EndpointIndex, FuncPerfModel, MemoryArray, MemoryDefinition, MemoryDomain,
    MemoryEndpoint, MemoryIdentity, MemoryLocation, MemoryPort, NetworkInterface, NetworkLink,
    NetworkTopology, OperationModel, PerfScenario, PerfYamlError, PerformanceYaml,
    ProcessorDefinition, ProcessorSelector, ProcessorType, ResolvedEndpointIndex, Resource, Scope,
    TimeCost, architecture_to_mlir, architecture_to_mlir_unchecked, mlir_validators_available,
};

// Re-export commonly used math types
pub use math::{AffineError, AffineExpr, AffineMap, ConstraintExpr, Expr, ParseError, Sym};
pub use mlir::{MlirFunc, MlirModule};
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
