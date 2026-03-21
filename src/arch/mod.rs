pub mod architecture;
pub mod architecture_graph;
pub mod data_mover;
pub mod memory;
pub mod network;
pub mod perf;
pub mod processor;
pub mod router;
pub mod size_dim;

// Re-export commonly used types
pub use crate::schedule::{
    MLIRFunc, MLIRFuncRef, MLIRModuleRef, MlirFunc, MlirFuncDetails, MlirMemrefSymbolBinding,
    MlirModule, MlirTensorSymbolBinding,
};
pub use architecture::Architecture;
pub use architecture_graph::{
    ArchEdge, ArchEdgeAttr, ArchEdgeDirection, ArchEdgeId, ArchGraph, ArchGraphBuilder,
    ArchGraphNode, ArchNode, ArchNodeComponent, ArchNodeId,
};
pub use data_mover::{DataMover, FunctionDataMover};
pub use memory::{MemoryBank, MemoryRegion};
pub use network::{MeshNetwork, MeshNetworkInterface, ScaleOutNetwork};
pub use perf::{FuncPerfModel, PerfScenario, SimpleTimeCost, TimeCost, TimeExpr};
pub use processor::{FunctionProcessor, Processor};
pub use router::{Router, RouterSide};
pub use size_dim::{Dimension, SizeExpr, Sym};
