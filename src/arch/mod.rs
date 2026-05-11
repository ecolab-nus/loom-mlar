pub mod architecture;
pub mod architecture_graph;
pub mod memory;
pub mod network;
pub mod perf;
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
pub use architecture::Architecture;
pub use architecture_graph::{
    ArchEdge, ArchEdgeAttr, ArchEdgeDirection, ArchEdgeId, ArchGraph, ArchGraphBuilder,
    ArchGraphNode, ArchNode, ArchNodeComponent, ArchNodeId, Router, RouterSide,
};
pub use memory::{MemoryBank, MemoryRegion};
pub use network::{
    MeshLink, MeshNetwork, MeshNetworkInterface, ScaleOutNetwork, ScaleOutNetworkBindings,
};
pub use perf::{FuncPerfModel, PerfScenario, SimpleTimeCost, TimeCost, TimeExpr};
pub use processor::{
    ComputeProcessor, ComputeProcessorBuilder, DataMover, DataMoverBuilder, FunctionDataMover,
    FunctionProcessor, Module as ProcessorModule, Processor,
};
pub use resource::{Resource, ResourceId};
pub use size_dim::{Dimension, SizeExpr, Sym};
