use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Deserializer, Serialize, de::Error as _};

use super::axis::Axis;
use super::memory::{
    EndpointIndex, MemoryArray, MemoryDefinition, MemoryEndpoint,
    validate_axes as validate_memory_axes, validate_selection, validate_static_bank,
};
use super::network::NetworkTopology;
use super::processor::{
    Connection, ConnectionInstance, MemoryLocation, ProcessorArray, ProcessorDefinition,
    ProcessorType, ResolvedEndpointIndex,
};
use super::resource::Resource;
use super::scope::Scope;
use crate::math::AffineExpr;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArchitectureError {
    DuplicateName {
        kind: &'static str,
        name: String,
    },
    UnknownReference {
        owner: String,
        kind: &'static str,
        name: String,
    },
    RankMismatch {
        object: String,
        expected: usize,
        actual: usize,
    },
    Invalid(String),
}

impl std::fmt::Display for ArchitectureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateName { kind, name } => write!(f, "duplicate {kind} '{name}'"),
            Self::UnknownReference { owner, kind, name } => {
                write!(f, "{owner} refers to unknown {kind} '{name}'")
            }
            Self::RankMismatch {
                object,
                expected,
                actual,
            } => write!(f, "{object} has rank {actual}; expected rank {expected}"),
            Self::Invalid(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for ArchitectureError {}

impl From<String> for ArchitectureError {
    fn from(message: String) -> Self {
        Self::Invalid(message)
    }
}

/// Canonical, flat, indexed architecture representation.
#[derive(Clone, Debug, Serialize)]
pub struct Architecture {
    pub(crate) name: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty", alias = "dimensions")]
    pub(crate) axes: Vec<Axis>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) memory_definitions: Vec<MemoryDefinition>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) memories: Vec<MemoryArray>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) processor_definitions: Vec<ProcessorDefinition>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) processors: Vec<ProcessorArray>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) resources: Vec<Resource>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) networks: Vec<NetworkTopology>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) scopes: Vec<Scope>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ArchitectureData {
    name: String,
    #[serde(default, alias = "dimensions")]
    axes: Vec<Axis>,
    #[serde(default)]
    memory_definitions: Vec<MemoryDefinition>,
    #[serde(default)]
    memories: Vec<MemoryArray>,
    #[serde(default)]
    processor_definitions: Vec<ProcessorDefinition>,
    #[serde(default)]
    processors: Vec<ProcessorArray>,
    #[serde(default)]
    resources: Vec<Resource>,
    #[serde(default)]
    networks: Vec<NetworkTopology>,
    #[serde(default)]
    scopes: Vec<Scope>,
}

impl<'de> Deserialize<'de> for Architecture {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let data = ArchitectureData::deserialize(deserializer)?;
        let architecture = Self {
            name: data.name,
            axes: data.axes,
            memory_definitions: data.memory_definitions,
            memories: data.memories,
            processor_definitions: data.processor_definitions,
            processors: data.processors,
            resources: data.resources,
            networks: data.networks,
            scopes: data.scopes,
        };
        architecture.validate().map_err(D::Error::custom)?;
        Ok(architecture)
    }
}

impl Architecture {
    pub fn builder(name: impl Into<String>) -> ArchitectureBuilder {
        ArchitectureBuilder::new(name)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn axes(&self) -> &[Axis] {
        &self.axes
    }

    pub fn axis(&self, name: &str) -> Option<&Axis> {
        self.axes.iter().find(|axis| axis.name() == name)
    }

    pub fn memories(&self) -> &[MemoryArray] {
        &self.memories
    }

    pub fn memory_definitions(&self) -> &[MemoryDefinition] {
        &self.memory_definitions
    }

    pub fn processor_definitions(&self) -> &[ProcessorDefinition] {
        &self.processor_definitions
    }

    pub fn processors(&self) -> &[ProcessorArray] {
        &self.processors
    }

    pub fn resources(&self) -> &[Resource] {
        &self.resources
    }

    pub fn networks(&self) -> &[NetworkTopology] {
        &self.networks
    }

    pub fn scopes(&self) -> &[Scope] {
        &self.scopes
    }

    pub fn memory(&self, name: &str) -> Option<&MemoryArray> {
        self.memories.iter().find(|memory| memory.name == name)
    }

    /// Memory arrays placed from one definition, in placement order.
    pub fn memories_of(&self, definition: &str) -> impl Iterator<Item = &MemoryArray> {
        self.memories
            .iter()
            .filter(move |memory| memory.definition == definition)
    }

    pub fn memory_definition(&self, memory: &MemoryArray) -> Option<&MemoryDefinition> {
        self.memory_definitions
            .iter()
            .find(|definition| definition.name == memory.definition)
    }

    pub fn connection_instances(&self, processor: &ProcessorArray) -> Vec<ConnectionInstance> {
        resolve_connection_instances(
            &processor.connection,
            &processor.axes,
            &self.memories,
            &self.memory_definitions,
        )
        .expect("canonical processor connection must evaluate")
    }

    pub fn processor_definition(&self, name: &str) -> Option<&ProcessorDefinition> {
        self.processor_definitions
            .iter()
            .find(|definition| definition.name == name)
    }

    pub fn processor_array(&self, name: &str) -> Option<&ProcessorArray> {
        self.processors
            .iter()
            .find(|processor| processor.name == name)
    }

    /// Processor arrays placed from one definition, in placement order.
    pub fn processors_of(&self, definition: &str) -> impl Iterator<Item = &ProcessorArray> {
        self.processors
            .iter()
            .filter(move |processor| processor.definition == definition)
    }

    pub fn get_function(&self, name: &str) -> Option<&super::processor::OperationModel> {
        self.processor_definitions
            .iter()
            .find_map(|definition| definition.get_function(name))
    }

    pub fn functions_named(
        &self,
        name: &str,
    ) -> impl Iterator<Item = (&ProcessorDefinition, &super::processor::OperationModel)> {
        self.processor_definitions
            .iter()
            .filter_map(move |definition| {
                definition
                    .get_function(name)
                    .map(|function| (definition, function))
            })
    }

    pub fn with_processor_type(
        mut self,
        definition: &str,
        processor_type: Option<ProcessorType>,
    ) -> Result<Self, ArchitectureError> {
        let target = self
            .processor_definitions
            .iter_mut()
            .find(|candidate| candidate.name == definition)
            .ok_or_else(|| ArchitectureError::UnknownReference {
                owner: "architecture".into(),
                kind: "processor definition",
                name: definition.into(),
            })?;
        target.processor_type = processor_type;
        Ok(self)
    }

    pub fn validate(&self) -> Result<(), ArchitectureError> {
        if self.name.is_empty() {
            return Err(ArchitectureError::Invalid(
                "architecture name cannot be empty".into(),
            ));
        }
        validate_unique(self.axes.iter().map(Axis::name), "axis")?;
        if let Some(axis) = self.axes.iter().find(|axis| axis.extent() == 0) {
            return Err(ArchitectureError::Invalid(format!(
                "axis '{}' extent must be positive",
                axis.name()
            )));
        }
        validate_memory_definitions(&self.memory_definitions)?;
        validate_unique(
            self.memories.iter().map(|memory| memory.name.as_str()),
            "memory",
        )?;
        validate_unique(
            self.processor_definitions
                .iter()
                .map(|definition| definition.name.as_str()),
            "processor definition",
        )?;
        validate_unique(
            self.processors
                .iter()
                .map(|processor| processor.name.as_str()),
            "processor array",
        )?;
        validate_unique(
            self.resources.iter().map(|resource| resource.name.as_str()),
            "resource",
        )?;
        for resource in &self.resources {
            resource.validate().map_err(ArchitectureError::Invalid)?;
        }
        validate_unique(
            self.networks.iter().map(|network| network.name.as_str()),
            "network",
        )?;

        let axes = self
            .axes
            .iter()
            .map(|axis| (axis.name(), axis.clone()))
            .collect::<BTreeMap<_, _>>();
        for memory in &self.memories {
            let definition = self
                .memory_definitions
                .iter()
                .find(|definition| definition.name == memory.definition)
                .ok_or_else(|| ArchitectureError::UnknownReference {
                    owner: format!("memory '{}'", memory.name),
                    kind: "memory definition",
                    name: memory.definition.clone(),
                })?;
            let _ = definition;
            validate_memory_axes(memory).map_err(ArchitectureError::Invalid)?;
            validate_axes(
                &format!("memory '{}'", memory.name),
                &memory.domain(),
                &axes,
            )?;
        }
        for definition in &self.processor_definitions {
            definition.validate().map_err(ArchitectureError::Invalid)?;
            for resource in &definition.resources {
                resource.validate().map_err(ArchitectureError::Invalid)?;
            }
        }
        for processor in &self.processors {
            if self.processor_definition(&processor.definition).is_none() {
                return Err(ArchitectureError::UnknownReference {
                    owner: format!("processor array '{}'", processor.name),
                    kind: "processor definition",
                    name: processor.definition.clone(),
                });
            }
            let connection = processor.connection.clone();
            validate_connection(&connection, &self.memories, &self.memory_definitions)?;
            validate_processor_ports(
                self.processor_definition(&processor.definition)
                    .expect("processor definition was checked"),
                &connection,
            )?;
            let domain = resolve_domain(&connection, &axes)?;
            resolve_connection_instances(
                &connection,
                &domain,
                &self.memories,
                &self.memory_definitions,
            )?;
            if processor.axes != domain {
                return Err(ArchitectureError::Invalid(format!(
                    "processor array '{}' stores axes inconsistent with its connection",
                    processor.name
                )));
            }
        }
        for network in &self.networks {
            network.validate().map_err(ArchitectureError::Invalid)?;
            validate_axes(
                &format!("network '{}'", network.name),
                &network.dimensions,
                &axes,
            )?;
            for interface in &network.interfaces {
                validate_endpoint_reference(
                    &interface.endpoint,
                    &self.memories,
                    &self.memory_definitions,
                )?;
            }
        }
        validate_scopes(
            &self.scopes,
            &axes,
            &self.memories,
            &self.processors,
            &self.networks,
            &self.resources,
        )?;
        Ok(())
    }
}

fn validate_axes(
    owner: &str,
    candidate: &[Axis],
    axes: &BTreeMap<&str, Axis>,
) -> Result<(), ArchitectureError> {
    for axis in candidate {
        match axes.get(axis.name()) {
            Some(expected) if expected == axis => {}
            Some(expected) => {
                return Err(ArchitectureError::Invalid(format!(
                    "{owner} axis '{}' has extent {}, expected {}",
                    axis.name(),
                    axis.extent(),
                    expected.extent()
                )));
            }
            None => {
                return Err(ArchitectureError::UnknownReference {
                    owner: owner.into(),
                    kind: "axis",
                    name: axis.name().into(),
                });
            }
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct ArchitectureBuilder {
    name: String,
    dimensions: Vec<Axis>,
    memory_definitions: Vec<MemoryDefinition>,
    placements: Vec<(String, String, Vec<String>)>,
    processor_definitions: Vec<ProcessorDefinition>,
    connections: Vec<(String, String, Connection)>,
    resources: Vec<Resource>,
    networks: Vec<NetworkTopology>,
    scopes: Vec<Scope>,
}

impl ArchitectureBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            dimensions: Vec::new(),
            memory_definitions: Vec::new(),
            placements: Vec::new(),
            processor_definitions: Vec::new(),
            connections: Vec::new(),
            resources: Vec::new(),
            networks: Vec::new(),
            scopes: Vec::new(),
        }
    }

    pub fn axis(mut self, name: impl Into<String>, extent: u64) -> Self {
        self.dimensions.push(Axis::new(name, extent));
        self
    }

    pub fn memory_definition(mut self, definition: MemoryDefinition) -> Self {
        self.memory_definitions.push(definition);
        self
    }

    /// Place a definition over ordered axes over `dimensions`.
    pub fn place_memory(
        self,
        definition: impl Into<String>,
        dimensions: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        let definition = definition.into();
        self.place_memory_as(definition.clone(), definition, dimensions)
    }

    /// Place a definition under `name` over ordered axes over `dimensions`.
    pub fn place_memory_as(
        self,
        name: impl Into<String>,
        definition: impl Into<String>,
        dimensions: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        let mut builder = self;
        builder.placements.push((
            name.into(),
            definition.into(),
            dimensions.into_iter().map(Into::into).collect(),
        ));
        builder
    }

    pub fn processor_definition(mut self, definition: ProcessorDefinition) -> Self {
        self.processor_definitions.push(definition);
        self
    }

    /// Place `definition` under its own name.
    ///
    /// Use [`ArchitectureBuilder::connect_as`] when one definition is placed
    /// more than once and the placements need distinct names.
    pub fn connect(self, definition: impl Into<String>, connection: Connection) -> Self {
        let definition = definition.into();
        self.connect_as(definition.clone(), definition, connection)
    }

    /// Place `processor_definition` under an explicit `placement_name`.
    pub fn connect_as(
        mut self,
        placement_name: impl Into<String>,
        processor_definition: impl Into<String>,
        connection: Connection,
    ) -> Self {
        self.connections.push((
            placement_name.into(),
            processor_definition.into(),
            connection,
        ));
        self
    }

    pub fn resource(mut self, resource: Resource) -> Self {
        self.resources.push(resource);
        self
    }

    pub fn network(mut self, network: NetworkTopology) -> Self {
        self.networks.push(network);
        self
    }

    pub fn scope(mut self, scope: Scope) -> Self {
        self.scopes.push(scope);
        self
    }

    pub fn build(self) -> Result<Architecture, ArchitectureError> {
        if self.name.is_empty() {
            return Err(ArchitectureError::Invalid(
                "architecture name cannot be empty".into(),
            ));
        }
        validate_unique(
            self.dimensions
                .iter()
                .map(|dimension| dimension.name.as_str()),
            "axis",
        )?;
        if let Some(axis) = self.dimensions.iter().find(|axis| axis.extent() == 0) {
            return Err(ArchitectureError::Invalid(format!(
                "axis '{}' extent must be positive",
                axis.name()
            )));
        }
        validate_memory_definitions(&self.memory_definitions)?;
        validate_unique(
            self.processor_definitions
                .iter()
                .map(|definition| definition.name.as_str()),
            "processor definition",
        )?;
        validate_unique(
            self.placements.iter().map(|(name, _, _)| name.as_str()),
            "memory placement",
        )?;

        let dimension_map = self
            .dimensions
            .iter()
            .map(|dimension| (dimension.name.as_str(), dimension.clone()))
            .collect::<BTreeMap<_, _>>();
        let mut memories = Vec::new();
        for (name, definition_name, placement) in &self.placements {
            let definition = self
                .memory_definitions
                .iter()
                .find(|definition| definition.name == *definition_name)
                .ok_or_else(|| ArchitectureError::UnknownReference {
                    owner: format!("memory placement '{name}'"),
                    kind: "memory definition",
                    name: definition_name.clone(),
                })?;
            let _ = definition;
            let axes = placement
                .iter()
                .map(|dimension| {
                    dimension_map
                        .get(dimension.as_str())
                        .cloned()
                        .ok_or_else(|| {
                            ArchitectureError::Invalid(format!(
                                "placement '{}' uses unknown dimension '{}'",
                                name, dimension
                            ))
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let memory = MemoryArray::new(name, definition_name, axes);
            validate_memory_axes(&memory).map_err(ArchitectureError::Invalid)?;
            memories.push(memory);
        }

        for definition in &self.processor_definitions {
            if definition.name.is_empty() {
                return Err(ArchitectureError::Invalid(
                    "processor definition name cannot be empty".into(),
                ));
            }
            definition.validate().map_err(ArchitectureError::Invalid)?;
            for resource in &definition.resources {
                resource.validate().map_err(ArchitectureError::Invalid)?;
            }
        }
        validate_unique(
            self.resources.iter().map(|resource| resource.name.as_str()),
            "resource",
        )?;
        for resource in &self.resources {
            resource.validate().map_err(ArchitectureError::Invalid)?;
        }
        validate_unique(
            self.networks.iter().map(|network| network.name.as_str()),
            "network",
        )?;
        for network in &self.networks {
            network.validate().map_err(ArchitectureError::Invalid)?;
            for dimension in &network.dimensions {
                match dimension_map.get(dimension.name.as_str()) {
                    Some(architecture_dimension) if architecture_dimension == dimension => {}
                    Some(architecture_dimension) => {
                        return Err(ArchitectureError::Invalid(format!(
                            "network '{}' dimension '{}' has size {}, architecture dimension has size {}",
                            network.name,
                            dimension.name,
                            dimension.extent,
                            architecture_dimension.extent
                        )));
                    }
                    None => {
                        return Err(ArchitectureError::Invalid(format!(
                            "network '{}' uses unknown dimension '{}'",
                            network.name, dimension.name
                        )));
                    }
                }
            }
            for interface in &network.interfaces {
                validate_endpoint_reference(
                    &interface.endpoint,
                    &memories,
                    &self.memory_definitions,
                )?;
            }
        }
        let shared_resources = self
            .resources
            .iter()
            .cloned()
            .map(|resource| (resource.name.clone(), resource))
            .collect::<BTreeMap<_, _>>();
        let mut processors = Vec::new();
        let mut resources = self.resources;
        for (name, definition_name, mut connection) in self.connections {
            let definition = self
                .processor_definitions
                .iter()
                .find(|definition| definition.name == definition_name)
                .ok_or_else(|| {
                    ArchitectureError::Invalid(format!(
                        "connection refers to unknown processor definition '{}'",
                        definition_name
                    ))
                })?;
            resolve_implicit_endpoints(&mut connection, &memories)?;
            let resolved_connection = connection;
            validate_connection(&resolved_connection, &memories, &self.memory_definitions)?;
            validate_processor_ports(definition, &resolved_connection)?;
            let domain = resolve_domain(&resolved_connection, &dimension_map)?;
            resolve_connection_instances(
                &resolved_connection,
                &domain,
                &memories,
                &self.memory_definitions,
            )?;
            let mut processor_resources = definition
                .resources
                .iter()
                .cloned()
                .map(|mut resource| {
                    resource.name = format!("{}.{}", name, resource.name);
                    resource.indexed(domain.clone())
                })
                .collect::<Vec<_>>();
            resources.extend(processor_resources.iter().cloned());
            for resource_name in &resolved_connection.resources {
                let resource = shared_resources.get(resource_name).ok_or_else(|| {
                    ArchitectureError::Invalid(format!(
                        "processor '{}' refers to unknown shared resource '{}'",
                        name, resource_name
                    ))
                })?;
                processor_resources.push(resource.clone());
            }
            validate_unique(
                processor_resources
                    .iter()
                    .map(|resource| resource.name.as_str()),
                "processor resource",
            )?;
            processors.push(ProcessorArray {
                name,
                definition: definition_name,
                connection: resolved_connection,
                axes: domain,
                resources: processor_resources,
            });
        }
        validate_unique(
            resources.iter().map(|resource| resource.name.as_str()),
            "resource",
        )?;
        validate_unique(
            processors.iter().map(|processor| processor.name.as_str()),
            "processor array",
        )?;

        validate_scopes(
            &self.scopes,
            &dimension_map,
            &memories,
            &processors,
            &self.networks,
            &resources,
        )?;

        Ok(Architecture {
            name: self.name,
            axes: self.dimensions,
            memory_definitions: self.memory_definitions,
            memories,
            processor_definitions: self.processor_definitions,
            processors,
            resources,
            networks: self.networks,
            scopes: self.scopes,
        })
    }
}

fn validate_endpoint_reference(
    endpoint: &super::memory::MemoryEndpoint,
    memories: &[MemoryArray],
    definitions: &[MemoryDefinition],
) -> Result<(), ArchitectureError> {
    let memory = memories
        .iter()
        .find(|memory| memory.name == endpoint.memory)
        .ok_or_else(|| {
            ArchitectureError::Invalid(format!(
                "network interface refers to unknown placed memory '{}'",
                endpoint.memory
            ))
        })?;
    let definition = definitions
        .iter()
        .find(|definition| definition.name == memory.definition)
        .expect("placed memory definition was validated");
    validate_selection(endpoint, memory).map_err(ArchitectureError::Invalid)?;
    validate_static_bank(endpoint, definition).map_err(ArchitectureError::Invalid)
}

fn validate_memory_definitions(definitions: &[MemoryDefinition]) -> Result<(), ArchitectureError> {
    validate_unique(
        definitions
            .iter()
            .map(|definition| definition.name.as_str()),
        "memory definition",
    )?;
    let mut kinds_by_name = BTreeMap::new();
    let mut names_by_kind = BTreeMap::new();
    for definition in definitions {
        definition.validate().map_err(ArchitectureError::Invalid)?;
        let Some(technology) = &definition.technology else {
            continue;
        };
        if let Some(kind) = kinds_by_name.insert(&technology.name, technology.kind)
            && kind != technology.kind
        {
            return Err(ArchitectureError::Invalid(format!(
                "memory technology '{}' uses both kind {kind} and kind {}",
                technology.name, technology.kind
            )));
        }
        if let Some(name) = names_by_kind.insert(technology.kind, &technology.name)
            && name != &technology.name
        {
            return Err(ArchitectureError::Invalid(format!(
                "memory technology kind {} is shared by '{}' and '{}'",
                technology.kind, name, technology.name
            )));
        }
    }
    Ok(())
}

fn validate_scopes(
    scopes: &[Scope],
    dimensions: &BTreeMap<&str, Axis>,
    memories: &[MemoryArray],
    processors: &[ProcessorArray],
    networks: &[NetworkTopology],
    resources: &[Resource],
) -> Result<(), ArchitectureError> {
    validate_unique(scopes.iter().map(|scope| scope.name.as_str()), "scope")?;
    let names = scopes
        .iter()
        .map(|scope| scope.name.as_str())
        .collect::<BTreeSet<_>>();
    for scope in scopes {
        if let Some(parent) = &scope.parent {
            if parent == &scope.name || !names.contains(parent.as_str()) {
                return Err(ArchitectureError::Invalid(format!(
                    "scope '{}' has invalid parent '{}'",
                    scope.name, parent
                )));
            }
        }
    }

    let mut resolved_domains = BTreeMap::new();
    for scope in scopes {
        validate_unique(scope.axes.iter().map(String::as_str), "scope axis")?;
        let scope_dimensions = scope
            .axes
            .iter()
            .map(|name| {
                dimensions.get(name.as_str()).cloned().ok_or_else(|| {
                    ArchitectureError::Invalid(format!(
                        "scope '{}' uses unknown dimension '{}'",
                        scope.name, name
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        validate_unique(scope.memories.iter().map(String::as_str), "scope memory")?;
        validate_unique(
            scope.processors.iter().map(String::as_str),
            "scope processor",
        )?;
        validate_unique(scope.networks.iter().map(String::as_str), "scope network")?;
        validate_unique(scope.resources.iter().map(String::as_str), "scope resource")?;
        validate_membership(
            &scope.name,
            &scope_dimensions,
            &scope.memories,
            memories
                .iter()
                .map(|memory| (memory.name.as_str(), memory.domain())),
            "memory",
        )?;
        validate_membership(
            &scope.name,
            &scope_dimensions,
            &scope.processors,
            processors
                .iter()
                .map(|processor| (processor.name.as_str(), processor.axes.clone())),
            "processor",
        )?;
        validate_membership(
            &scope.name,
            &scope_dimensions,
            &scope.networks,
            networks
                .iter()
                .map(|network| (network.name.as_str(), network.dimensions.clone())),
            "network",
        )?;
        validate_membership(
            &scope.name,
            &scope_dimensions,
            &scope.resources,
            resources
                .iter()
                .map(|resource| (resource.name.as_str(), resource.indices.clone())),
            "resource",
        )?;
        resolved_domains.insert(scope.name.as_str(), scope_dimensions);
    }

    let by_name = scopes
        .iter()
        .map(|scope| (scope.name.as_str(), scope))
        .collect::<BTreeMap<_, _>>();
    for scope in scopes {
        let mut seen = BTreeSet::new();
        let mut cursor = scope;
        while let Some(parent) = &cursor.parent {
            if !seen.insert(parent.as_str()) {
                return Err(ArchitectureError::Invalid(format!(
                    "scope hierarchy contains a cycle through '{}'",
                    parent
                )));
            }
            let parent_scope = by_name[parent.as_str()];
            if !domain_is_prefix(
                &resolved_domains[parent_scope.name.as_str()],
                &resolved_domains[scope.name.as_str()],
            ) {
                return Err(ArchitectureError::Invalid(format!(
                    "parent scope '{}' dimensions are not a prefix of child scope '{}'",
                    parent_scope.name, scope.name
                )));
            }
            cursor = parent_scope;
        }
    }
    validate_single_owner(scopes, |scope| &scope.memories, "memory")?;
    validate_single_owner(scopes, |scope| &scope.processors, "processor")?;
    validate_single_owner(scopes, |scope| &scope.networks, "network")?;
    validate_single_owner(scopes, |scope| &scope.resources, "resource")?;
    Ok(())
}

fn validate_membership<'a>(
    scope: &str,
    scope_dimensions: &[Axis],
    members: &[String],
    candidates: impl IntoIterator<Item = (&'a str, Vec<Axis>)>,
    kind: &'static str,
) -> Result<(), ArchitectureError> {
    let candidates = candidates.into_iter().collect::<BTreeMap<_, _>>();
    for member in members {
        let domain =
            candidates
                .get(member.as_str())
                .ok_or_else(|| ArchitectureError::UnknownReference {
                    owner: format!("scope '{scope}'"),
                    kind,
                    name: member.clone(),
                })?;
        if !domain_is_prefix(scope_dimensions, domain.as_slice()) {
            return Err(ArchitectureError::Invalid(format!(
                "scope '{scope}' domain is not a prefix of {kind} '{member}' domain"
            )));
        }
    }
    Ok(())
}

fn domain_is_prefix(prefix: &[Axis], domain: &[Axis]) -> bool {
    prefix.len() <= domain.len() && prefix.iter().zip(domain).all(|(lhs, rhs)| lhs == rhs)
}

fn validate_single_owner(
    scopes: &[Scope],
    members: impl Fn(&Scope) -> &[String],
    kind: &str,
) -> Result<(), ArchitectureError> {
    let mut owners = BTreeMap::<&str, &str>::new();
    for scope in scopes {
        for member in members(scope) {
            if let Some(previous) = owners.insert(member, &scope.name) {
                return Err(ArchitectureError::Invalid(format!(
                    "{kind} '{member}' is owned by both scope '{previous}' and '{}'",
                    scope.name
                )));
            }
        }
    }
    Ok(())
}

fn validate_unique<'a>(
    names: impl IntoIterator<Item = &'a str>,
    kind: &'static str,
) -> Result<(), ArchitectureError> {
    let mut unique = BTreeSet::new();
    for name in names {
        if !unique.insert(name) {
            return Err(ArchitectureError::DuplicateName {
                kind,
                name: name.to_string(),
            });
        }
    }
    Ok(())
}

fn validate_connection(
    connection: &Connection,
    memories: &[MemoryArray],
    definitions: &[MemoryDefinition],
) -> Result<(), ArchitectureError> {
    validate_unique(
        connection.inputs.iter().map(|port| port.name.as_str()),
        "input port",
    )?;
    validate_unique(
        connection.outputs.iter().map(|port| port.name.as_str()),
        "output port",
    )?;
    for input in &connection.inputs {
        if let Some(output) = connection
            .outputs
            .iter()
            .find(|output| output.name == input.name)
            && output.endpoint != input.endpoint
        {
            return Err(ArchitectureError::Invalid(format!(
                "processor port '{}' is used as both input and output with different memory endpoints",
                input.name
            )));
        }
    }
    for port in connection.inputs.iter().chain(&connection.outputs) {
        if !valid_port_name(&port.name) {
            return Err(ArchitectureError::Invalid(format!(
                "invalid processor port name '{}'",
                port.name
            )));
        }
        let endpoint = &port.endpoint;
        let memory = memories
            .iter()
            .find(|memory| memory.name == endpoint.memory)
            .ok_or_else(|| {
                ArchitectureError::Invalid(format!(
                    "connection refers to unknown placed memory '{}'",
                    endpoint.memory
                ))
            })?;
        let definition = definitions
            .iter()
            .find(|definition| definition.name == memory.definition)
            .ok_or_else(|| {
                ArchitectureError::Invalid(format!(
                    "placed memory '{}' has unknown definition '{}'",
                    memory.name, memory.definition
                ))
            })?;
        validate_selection(endpoint, memory).map_err(ArchitectureError::Invalid)?;
        validate_static_bank(endpoint, definition).map_err(ArchitectureError::Invalid)?;
    }
    Ok(())
}

fn valid_port_name(name: &str) -> bool {
    let mut characters = name.chars();
    matches!(characters.next(), Some(first) if first.is_ascii_alphabetic() || first == '_')
        && characters.all(|character| character.is_ascii_alphanumeric() || character == '_')
}

fn resolve_implicit_endpoints(
    connection: &mut Connection,
    memories: &[MemoryArray],
) -> Result<(), ArchitectureError> {
    let domain = connection
        .domain
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    for port in connection.inputs.iter_mut().chain(&mut connection.outputs) {
        let endpoint = &mut port.endpoint;
        let memory = memories
            .iter()
            .find(|memory| memory.name == endpoint.memory)
            .ok_or_else(|| {
                ArchitectureError::Invalid(format!(
                    "connection port '{}' refers to unknown placed memory '{}'",
                    port.name, endpoint.memory
                ))
            })?;
        if endpoint.indices.is_empty() && memory.rank() > 0 {
            let missing = memory
                .axes()
                .iter()
                .filter(|axis| !domain.contains(axis.name()))
                .map(Axis::name)
                .collect::<Vec<_>>();
            if !missing.is_empty() {
                return Err(ArchitectureError::Invalid(format!(
                    "connection port '{}' cannot infer a pointwise endpoint for memory '{}': axes {missing:?} are not in processor domain {:?}; use explicit selectors",
                    port.name, endpoint.memory, connection.domain
                )));
            }
            endpoint.indices = memory
                .axes()
                .iter()
                .map(|axis| EndpointIndex::Expression(AffineExpr::variable(axis.name())))
                .collect();
        }
    }
    Ok(())
}

fn validate_processor_ports(
    definition: &ProcessorDefinition,
    connection: &Connection,
) -> Result<(), ArchitectureError> {
    let inputs = connection
        .inputs
        .iter()
        .map(|port| port.name.as_str())
        .collect::<BTreeSet<_>>();
    let outputs = connection
        .outputs
        .iter()
        .map(|port| port.name.as_str())
        .collect::<BTreeSet<_>>();
    for function in &definition.functions {
        let Some(details) = &function.func.mlir_details else {
            continue;
        };
        for (side, memrefs, ports, other_ports) in [
            ("input", &details.source_memrefs, &inputs, &outputs),
            ("output", &details.target_memrefs, &outputs, &inputs),
        ] {
            for memref in memrefs {
                let binding = details
                    .mem_region_bindings
                    .iter()
                    .find(|binding| &binding.memref == memref)
                    .ok_or_else(|| {
                        ArchitectureError::Invalid(format!(
                            "MLIR function '{}' {side} '%{memref}' has no loom.bind_mem",
                            function.func.name
                        ))
                    })?;
                if !ports.contains(binding.region.as_str()) {
                    let reason = if other_ports.contains(binding.region.as_str()) {
                        "is declared on the wrong side"
                    } else {
                        "is not declared by the connection"
                    };
                    return Err(ArchitectureError::Invalid(format!(
                        "MLIR function '{}' binds {side} '%{memref}' to '@{}', which {reason}",
                        function.func.name, binding.region
                    )));
                }
            }
        }
    }
    Ok(())
}

fn resolve_domain(
    connection: &Connection,
    dimensions: &BTreeMap<&str, Axis>,
) -> Result<Vec<Axis>, ArchitectureError> {
    let unique = connection.domain.iter().collect::<BTreeSet<_>>();
    if unique.len() != connection.domain.len() {
        return Err(ArchitectureError::Invalid(
            "processor connection domain contains duplicate axes".into(),
        ));
    }
    let domain_names = connection
        .domain
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    for variable in connection.variables() {
        if !domain_names.contains(variable.as_str()) {
            return Err(ArchitectureError::Invalid(format!(
                "connection index variable '{variable}' is not in its declared domain"
            )));
        }
    }
    connection
        .domain
        .iter()
        .map(|axis| {
            dimensions.get(axis.as_str()).cloned().ok_or_else(|| {
                ArchitectureError::Invalid(format!(
                    "connection domain uses unknown architecture axis '{axis}'"
                ))
            })
        })
        .collect()
}

fn resolve_connection_instances(
    connection: &Connection,
    domain: &[Axis],
    memories: &[MemoryArray],
    definitions: &[MemoryDefinition],
) -> Result<Vec<ConnectionInstance>, ArchitectureError> {
    let mut points = vec![BTreeMap::<String, i64>::new()];
    for dimension in domain {
        let mut expanded = Vec::new();
        for point in points {
            for value in 0..dimension.extent {
                let mut point = point.clone();
                point.insert(dimension.name.clone(), value as i64);
                expanded.push(point);
            }
        }
        points = expanded;
    }

    let mut resolved = Vec::new();
    'point: for point in points {
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        for symbolic in &connection.inputs {
            let Some(endpoint) =
                resolve_endpoint(&symbolic.endpoint, &point, memories, definitions)?
            else {
                continue 'point;
            };
            inputs.push(endpoint);
        }
        for symbolic in &connection.outputs {
            let Some(endpoint) =
                resolve_endpoint(&symbolic.endpoint, &point, memories, definitions)?
            else {
                continue 'point;
            };
            outputs.push(endpoint);
        }
        resolved.push(ConnectionInstance {
            variables: point
                .iter()
                .map(|(name, value)| (name.clone(), *value as u64))
                .collect(),
            inputs,
            outputs,
        });
    }
    Ok(resolved)
}

fn resolve_endpoint(
    endpoint: &MemoryEndpoint,
    values: &BTreeMap<String, i64>,
    memories: &[MemoryArray],
    definitions: &[MemoryDefinition],
) -> Result<Option<MemoryLocation>, ArchitectureError> {
    let memory = memories
        .iter()
        .find(|memory| memory.name == endpoint.memory)
        .expect("connection was validated");
    let mut indices = Vec::new();
    for (selector, axis) in endpoint.indices.iter().zip(memory.axes()) {
        match selector {
            super::memory::EndpointIndex::All => indices.push(ResolvedEndpointIndex::All),
            super::memory::EndpointIndex::Expression(expression) => {
                let value = expression.evaluate(values).ok_or_else(|| {
                    ArchitectureError::Invalid(format!(
                        "could not evaluate index for memory '{}'",
                        endpoint.memory
                    ))
                })?;
                if value < 0 || value >= axis.extent as i64 {
                    return Ok(None);
                }
                indices.push(ResolvedEndpointIndex::Index(value as u64));
            }
        }
    }
    let definition = definitions
        .iter()
        .find(|definition| definition.name == memory.definition)
        .expect("placement was validated");
    let bank = endpoint
        .bank
        .as_ref()
        .map(|expression| {
            expression.evaluate(values).ok_or_else(|| {
                ArchitectureError::Invalid(format!(
                    "could not evaluate bank for memory '{}'",
                    endpoint.memory
                ))
            })
        })
        .transpose()?;
    if let Some(bank) = bank {
        let bank_count = definition
            .banking
            .as_ref()
            .expect("bank selection was validated")
            .banks;
        if bank < 0 || bank >= bank_count as i64 {
            return Ok(None);
        }
    }
    Ok(Some(MemoryLocation {
        memory: endpoint.memory.clone(),
        indices,
        bank: bank.map(|bank| bank as u64),
    }))
}
