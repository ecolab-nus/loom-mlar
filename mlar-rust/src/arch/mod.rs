pub mod architecture;
pub mod link;
pub mod memory;
pub mod perf;
pub mod processor;
pub mod resource;
pub mod size_dim;

// Re-export commonly used types
pub use architecture::{Architecture, ArchitectureBuilder, ArchitectureLabel};
pub use link::{Endpoint, Link, LinkMapRelation, LinkTopology, SharingDomain};
pub use memory::{MemoryBank, MemoryRegion};
pub use perf::{FuncPerfModel, PerfScenario, ProcPerfModel, TimeCostExpr};
pub use processor::{
    MLIRFuncRef, MLIRModuleRef, MlirFuncRef, MlirModuleRef, MlirTensorSymbolBinding, Processor,
    Processors,
};
pub use resource::{Resource, ResourceReq};
pub use size_dim::{Dimension, SizeExpr, Sym};
