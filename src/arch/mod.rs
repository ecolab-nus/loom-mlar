pub mod architecture;
pub mod axis;
pub mod memory;
pub mod network;
pub mod perf;
pub mod perf_yaml;
pub mod processor;
pub mod resource;
pub mod scope;

pub use crate::mlir::{
    AdlExportError, architecture_to_mlir, architecture_to_mlir_unchecked, mlir_validators_available,
};
pub use architecture::{Architecture, ArchitectureBuilder, ArchitectureError};
pub use axis::{Axis, EndpointParseError};
pub use memory::{
    Banking, EndpointIndex, MemoryArray, MemoryDefinition, MemoryEndpoint, MemoryTechnology,
};
pub use network::{NetworkEdge, NetworkInterface, NetworkLink, NetworkTopology};
pub use perf::{FuncPerfModel, FuncPerfModelBuilder, PerfScenario, TimeCost};
pub use perf_yaml::{PerfYamlError, PerformanceYaml};
pub use processor::{
    Connection, ConnectionInstance, MemoryLocation, MemoryPort, OperationModel, ProcessorArray,
    ProcessorDefinition, ProcessorSelection, ProcessorSelectionError, ProcessorSelector,
    ProcessorType, ResolvedEndpointIndex,
};
pub use resource::Resource;
pub use scope::Scope;
