pub mod architecture;
pub mod architecture_graph;
pub mod memory;
pub mod network;
pub mod perf;
pub mod processor;
pub mod router;
pub mod size_dim;

pub mod mlir_export {
    pub use crate::mlir::architecture_to_mlir;
}

// Re-export commonly used types
pub use crate::mlir::architecture_to_mlir;
pub use crate::schedule::{
    MLIRFunc, MLIRFuncRef, MLIRModuleRef, MlirCopyOp, MlirFunc, MlirFuncDetails,
    MlirMemRegionBinding, MlirMemrefSymbolBinding, MlirModule, MlirTensorSymbolBinding,
};
pub use architecture::Architecture;
pub use architecture_graph::{
    ArchEdge, ArchEdgeAttr, ArchEdgeDirection, ArchEdgeId, ArchGraph, ArchGraphBuilder,
    ArchGraphNode, ArchNode, ArchNodeComponent, ArchNodeId,
};
pub use memory::{MemoryBank, MemoryRegion};
pub use network::{MeshNetwork, MeshNetworkInterface, ScaleOutNetwork};
pub use perf::{FuncPerfModel, PerfScenario, SimpleTimeCost, TimeCost, TimeExpr};
pub use processor::{
    ComputeProcessor, ComputeProcessorBuilder, DataMover, DataMoverBuilder, FunctionDataMover,
    FunctionProcessor, HardwareProperty, Module as ProcessorModule, Processor,
};
pub use router::{Router, RouterSide};
pub use size_dim::{Dimension, SizeExpr, Sym};
