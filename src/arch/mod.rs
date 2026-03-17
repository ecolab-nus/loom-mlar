pub mod architecture;
pub mod links;
pub mod memory;
pub mod perf;
pub mod processor;
pub mod size_dim;

// Re-export commonly used types
pub use crate::schedule::{
    MLIRFunc, MLIRFuncRef, MLIRModuleRef, MlirFunc, MlirFuncDetails, MlirModule,
    MlirTensorSymbolBinding,
};
pub use architecture::{
    ArchEdge, ArchGraph, ArchGraphBuilder, ArchGraphNode, ArchNode, ArchNodeComponent, Architecture,
};
pub use links::{
    Endpoint, LinkMapRelation, LinkTopology, Router, RouterEndpoint, RouterEndpointTarget,
    RouterSide, ScaleOutNetwork, SharingDomain,
};
pub use memory::{MemoryBank, MemoryRegion};
pub use perf::{FuncPerfModel, PerfScenario, SimpleTimeCost, TimeCost, TimeExpr};
pub use processor::{FunctionProcessor, Processor, ProcessorSet, Processors};
pub use size_dim::{Dimension, SizeExpr, Sym};
