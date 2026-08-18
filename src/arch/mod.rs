mod arch_yaml;
pub mod architecture;
pub mod axis;
pub mod memory;
pub mod network;
pub mod perf;
mod perf_yaml;
pub mod processor;
pub mod resource;
pub mod scope;

pub use crate::mlir::{
    AdlExportError, architecture_to_mlir, architecture_to_mlir_unchecked, mlir_validators_available,
};
pub use arch_yaml::{ArchLoadError, ChipYaml, ProcessorYaml};
pub use architecture::{Architecture, ArchitectureBuilder, ArchitectureError};
pub use axis::{Axis, EndpointParseError};
pub use memory::{
    Banking, EndpointIndex, MemoryAlias, MemoryArray, MemoryDefinition, MemoryEndpoint,
    MemoryTechnology,
};
pub use network::{NetworkEdge, NetworkInterface, NetworkLink, NetworkTopology};
pub use perf::{FuncPerfModel, FuncPerfModelBuilder, PerfScenario, TimeCost};
pub use perf_yaml::{PerfYamlError, PerformanceYaml};
pub use processor::{
    Connection, ConnectionInstance, MemoryLocation, OperationModel, ProcessorArray,
    ProcessorDefinition, ProcessorSelection, ProcessorSelectionError, ProcessorSelector,
    ProcessorSourceFormat, ProcessorType, ResolvedEndpointIndex,
};
pub use resource::Resource;
pub use scope::Scope;
