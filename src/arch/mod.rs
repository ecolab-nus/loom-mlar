pub mod architecture;
pub mod graph;
pub mod links;
pub mod memory;
pub mod perf;
pub mod processor;
pub mod resource;
pub mod size_dim;

// Re-export commonly used types
pub use crate::schedule::{
    MLIRFunc, MLIRFuncRef, MLIRModuleRef, MlirFunc, MlirFuncDetails, MlirModule,
    MlirTensorSymbolBinding,
};
pub use architecture::{Architecture, ArchitectureBuilder};
pub use graph::{ArchEdge, ArchGraph, ArchGraphNode, ArchNode, ArchNodeComponent};
pub use links::{
    Endpoint, LinkMapRelation, LinkTopology, Router, RouterEndpoint, RouterEndpointTarget,
    RouterSide, ScaleOutNetwork, SharingDomain,
};
pub use memory::{MemoryBank, MemoryRegion};
pub use perf::{FuncPerfModel, PerfScenario, TimeCostExpr, TimeExpr};
pub use processor::{FunctionProcessor, Processor, ProcessorSet, Processors};
pub use resource::{Resource, ResourceReq};
pub use size_dim::{Dimension, SizeExpr, Sym};
