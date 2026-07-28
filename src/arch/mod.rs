mod arch_yaml;
pub mod architecture;
pub mod memory;
pub mod network;
pub mod perf;
mod perf_yaml;
pub mod processor;
pub mod resource;
pub mod size_dim;

pub mod mlir_export {
    pub use crate::mlir::architecture_to_mlir;
}

// Re-export commonly used types
pub use crate::mlir::architecture_to_mlir;
pub use crate::schedule::{
    MLIRFunc, MLIRFuncRef, MLIRModuleRef, MlirBroadcastDim, MlirCopyOp, MlirFunc, MlirFuncDetails,
    MlirGatherOp, MlirMemRegionBinding, MlirMemrefSymbolBinding, MlirModule,
    MlirTensorSymbolBinding,
};
pub use arch_yaml::{ArchLoadError, ChipYaml};
pub use architecture::Architecture;
pub use memory::{MemoryBank, MemoryRegion};
pub use network::{
    MeshLink, MeshNetwork, MeshNetworkInterface, ScaleOutNetwork, ScaleOutNetworkBindings,
};
pub use perf::{
    FuncPerfModel, FuncPerfModelBuilder, PerfScenario, SimpleTimeCost, TimeCost, TimeExpr,
};
pub use perf_yaml::{PerfYamlError, PerfYamlSpec};
pub use processor::{
    ComputeProcessor, ComputeProcessorBuilder, DataEffect, DataMover, DataMoverBuilder,
    FunctionDataMover, FunctionProcessor, MemoryRegionRef, Module as ProcessorModule, Processor,
};
pub use resource::{Resource, ResourceId};
pub use size_dim::{Dimension, SizeExpr, Sym};
