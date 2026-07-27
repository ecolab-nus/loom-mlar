pub mod architecture;
pub mod memory;
pub mod network;
pub mod perf;
pub mod perf_yaml;
pub mod processor;
pub mod resource;
pub mod size_dim;

pub mod mlir_export {
    pub use crate::mlir::{MlirExportError, architecture_to_mlir, architecture_to_mlir_unchecked};
}

// Re-export commonly used types
pub use crate::mlir::{architecture_to_mlir, architecture_to_mlir_unchecked};
pub use crate::schedule::{
    MLIRFunc, MLIRFuncRef, MLIRModuleRef, MlirBroadcastDim, MlirCopyOp, MlirFunc, MlirFuncDetails,
    MlirGatherOp, MlirMemRegionBinding, MlirMemrefSymbolBinding, MlirModule,
    MlirTensorSymbolBinding,
};
pub use architecture::Architecture;
pub use memory::{MemoryBank, MemoryRegion};
pub use network::{
    MeshLink, MeshNetwork, MeshNetworkInterface, ScaleOutNetwork, ScaleOutNetworkBindings,
};
pub use perf::{
    FuncPerfModel, FuncPerfModelBuilder, PerfScenario, SimpleTimeCost, TimeCost, TimeExpr,
};
pub use perf_yaml::{
    PerfFunctionYaml, PerfScenarioYaml, PerfYamlError, PerfYamlSpec, SimpleCostYaml, TimeCostYaml,
};
pub use processor::{
    ComputeProcessor, ComputeProcessorBuilder, DataEffect, DataMover, DataMoverBuilder,
    FunctionDataMover, FunctionProcessor, MemoryRegionRef, Module as ProcessorModule, Processor,
};
pub use resource::{Resource, ResourceId};
pub use size_dim::{Dimension, SizeExpr, Sym};
