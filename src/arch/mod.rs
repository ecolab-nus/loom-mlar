pub mod architecture;
pub mod architecture_graph;
pub mod links;
pub mod memory;
pub mod network;
pub mod perf;
pub mod processor;
pub mod router;
pub mod size_dim;

// Re-export commonly used types
pub use crate::schedule::{
    MLIRFunc, MLIRFuncRef, MLIRModuleRef, MlirFunc, MlirFuncDetails, MlirModule,
    MlirTensorSymbolBinding,
};
pub use architecture::Architecture;
pub use architecture_graph::{
    ArchEdge, ArchGraph, ArchGraphBuilder, ArchGraphNode, ArchNode, ArchNodeComponent,
};
pub use memory::{MemoryBank, MemoryRegion};
pub use network::{Endpoint, LinkMapRelation, LinkTopology, MeshNetwork, ScaleOutNetwork};
pub use perf::{FuncPerfModel, PerfScenario, SimpleTimeCost, TimeCost, TimeExpr};
pub use processor::{FunctionProcessor, Processor, ProcessorSet, Processors};
pub use router::{Router, RouterEndpoint, RouterEndpointTarget, RouterSide};
pub use size_dim::{Dimension, SizeExpr, Sym};
