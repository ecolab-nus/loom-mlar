mod arch_yaml;
pub mod architecture;
pub mod index;
pub mod memory;
pub mod perf;
mod perf_yaml;
pub mod processor;
pub mod resource;
pub mod size_dim;

pub use crate::mlir::{AdlExportError, architecture_to_mlir, architecture_to_mlir_unchecked};
pub use crate::schedule::{
    MLIRFunc, MLIRFuncRef, MLIRModuleRef, MlirBroadcastDim, MlirCopyOp, MlirFunc, MlirFuncDetails,
    MlirGatherOp, MlirMemRegionBinding, MlirMemrefSymbolBinding, MlirModule,
    MlirTensorSymbolBinding,
};
pub use arch_yaml::{ArchLoadError, ChipYaml, ProcessorYaml};
pub use architecture::{Architecture, ArchitectureBuilder, ArchitectureError};
pub use index::{AffineExpression, EndpointParseError, IndexDomain};
pub use memory::{
    Banking, EndpointIndex, MemoryArray, MemoryCatalog, MemoryDefinition, MemoryEndpoint,
    NamedMemoryRegion,
};
pub use perf::{
    FuncPerfModel, FuncPerfModelBuilder, PerfScenario, SimpleTimeCost, TimeCost, TimeExpr,
};
pub use perf_yaml::{PerfYamlError, PerfYamlSpec};
pub use processor::{
    AffineRelation, ConnectionSpec, FunctionProcessor, ProcessorArray, ProcessorDefinition,
    ProcessorSelection, ProcessorSelectionError, ProcessorSelector, ProcessorType,
    ResolvedConnection, ResolvedMemoryEndpoint,
};
pub use resource::ResourceArray;
pub use size_dim::{Dimension, SizeExpr, Sym};
