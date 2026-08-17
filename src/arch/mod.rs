pub mod architecture;
pub mod memory;
pub mod network;
pub mod perf;
pub mod perf_yaml;
pub mod processor;
pub mod resource;
pub mod size_dim;

// Re-export architecture-domain types.
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
