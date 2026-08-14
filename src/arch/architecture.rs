use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use serde::{Deserialize, Deserializer, Serialize, de::Error as _};

use super::arch_yaml::ProcessorYaml;
use super::axis::Axis;
use super::memory::{
    MemoryAlias, MemoryArray, MemoryDefinition, MemoryEndpoint, validate_region_selector,
    validate_static_bank,
};
use super::network::NetworkTopology;
use super::processor::{
    Connection, ConnectionInstance, MemoryLocation, ProcessorArray, ProcessorDefinition,
    ProcessorSourceFormat, ProcessorType, ResolvedEndpointIndex, resolve_operand_memory_bindings,
};
use super::resource::Resource;
use super::scope::Scope;

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
    pub(crate) memory_aliases: Vec<MemoryAlias>,
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
struct ArchitectureData {
    name: String,
    #[serde(default, alias = "dimensions")]
    axes: Vec<Axis>,
    #[serde(default)]
    memory_definitions: Vec<MemoryDefinition>,
    #[serde(default)]
    memory_aliases: Vec<MemoryAlias>,
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
            memory_aliases: data.memory_aliases,
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

    pub fn memory_aliases(&self) -> &[MemoryAlias] {
        &self.memory_aliases
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

    pub fn memory_alias(&self, name: &str) -> Option<&MemoryAlias> {
        self.memory_aliases.iter().find(|alias| alias.name == name)
    }

    pub fn memory_definition(&self, memory: &MemoryArray) -> Option<&MemoryDefinition> {
        self.memory_definitions
            .iter()
            .find(|definition| definition.name == memory.definition)
    }

    pub fn connection_instances(&self, processor: &ProcessorArray) -> Vec<ConnectionInstance> {
        let mut connection = processor.connection.clone();
        resolve_memory_aliases(&mut connection, &self.memory_aliases)
            .expect("canonical processor connection must resolve");
        resolve_connection_instances(
            &connection,
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
        validate_memory_aliases(&self.memory_aliases)?;
        validate_unique(
            self.memories.iter().map(|memory| memory.name.as_str()),
            "memory",
        )?;
        validate_memory_alias_targets(
            &self.memory_aliases,
            &self.memories,
            &self.memory_definitions,
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
            if memory.indices.len() != definition.indices.len() {
                return Err(ArchitectureError::RankMismatch {
                    object: format!("memory '{}'", memory.name),
                    expected: definition.indices.len(),
                    actual: memory.indices.len(),
                });
            }
            validate_axes(&format!("memory '{}'", memory.name), &memory.indices, &axes)?;
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
            let mut connection = processor.connection.clone();
            resolve_memory_aliases(&mut connection, &self.memory_aliases)?;
            validate_connection(&connection, &self.memories, &self.memory_definitions)?;
            validate_processor_memory_bindings(
                &processor.name,
                self.processor_definition(&processor.definition)
                    .expect("definition existence was checked"),
                &connection,
                &self.memories,
                &self.memory_definitions,
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
    memory_aliases: Vec<MemoryAlias>,
    placements: Vec<(String, String, Vec<String>)>,
    processor_definitions: Vec<ProcessorDefinition>,
    connections: Vec<(String, String, Connection)>,
    resources: Vec<Resource>,
    networks: Vec<NetworkTopology>,
    scopes: Vec<Scope>,
    processor_source_dir: Option<PathBuf>,
    /// Load failures from [`ArchitectureBuilder::processor`], reported by
    /// [`ArchitectureBuilder::build`] so the chain itself stays infallible.
    deferred_errors: Vec<String>,
}

impl ArchitectureBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            dimensions: Vec::new(),
            memory_definitions: Vec::new(),
            memory_aliases: Vec::new(),
            placements: Vec::new(),
            processor_definitions: Vec::new(),
            connections: Vec::new(),
            resources: Vec::new(),
            networks: Vec::new(),
            scopes: Vec::new(),
            processor_source_dir: None,
            deferred_errors: Vec::new(),
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

    pub fn memory_alias(mut self, alias: MemoryAlias) -> Self {
        self.memory_aliases.push(alias);
        self
    }

    pub fn place_memory(
        mut self,
        definition: impl Into<String>,
        dimensions: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        let definition = definition.into();
        self.placements.push((
            definition.clone(),
            definition,
            dimensions.into_iter().map(Into::into).collect(),
        ));
        self
    }

    pub fn place_memory_as(
        mut self,
        name: impl Into<String>,
        definition: impl Into<String>,
        dimensions: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.placements.push((
            name.into(),
            definition.into(),
            dimensions.into_iter().map(Into::into).collect(),
        ));
        self
    }

    pub fn processor_definition(mut self, definition: ProcessorDefinition) -> Self {
        self.processor_definitions.push(definition);
        self
    }

    /// Directory holding `<name>.yaml` processor packages for
    /// [`ArchitectureBuilder::processor`].
    pub fn processor_source_dir(mut self, directory: impl Into<PathBuf>) -> Self {
        self.processor_source_dir = Some(directory.into());
        self
    }

    /// Load and register `<name>.yaml` from the configured source directory.
    /// Load failures are reported by [`ArchitectureBuilder::build`].
    pub fn processor(mut self, name: impl AsRef<str>) -> Self {
        let name = name.as_ref();
        let Some(directory) = self.processor_source_dir.as_ref() else {
            self.deferred_errors.push(format!(
                "processor '{name}' needs a `processor_source_dir`; set one or pass a \
                 built definition to `processor_definition`"
            ));
            return self;
        };
        let path = directory.join(format!("{name}.yaml"));
        match ProcessorYaml::from_file(&path).and_then(|yaml| yaml.build_definition(&path)) {
            Ok(definition) => self.processor_definitions.push(definition),
            Err(error) => self
                .deferred_errors
                .push(format!("processor '{name}' ({}): {error}", path.display())),
        }
        self
    }

    /// [`ArchitectureBuilder::processor`] for several names in order.
    pub fn processors(mut self, names: impl IntoIterator<Item = impl AsRef<str>>) -> Self {
        for name in names {
            self = self.processor(name);
        }
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
        if !self.deferred_errors.is_empty() {
            return Err(ArchitectureError::Invalid(self.deferred_errors.join("; ")));
        }
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
        validate_memory_aliases(&self.memory_aliases)?;
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
            if placement.len() != definition.indices.len() {
                return Err(ArchitectureError::RankMismatch {
                    object: format!("memory placement '{name}'"),
                    expected: definition.indices.len(),
                    actual: placement.len(),
                });
            }
            let indices = placement
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
            memories.push(MemoryArray::new(name, definition_name, indices));
        }
        validate_memory_alias_targets(&self.memory_aliases, &memories, &self.memory_definitions)?;

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
                    &self.memory_aliases,
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
        for (name, definition_name, connection) in self.connections {
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
            let mut resolved_connection = connection.clone();
            resolve_memory_aliases(&mut resolved_connection, &self.memory_aliases)?;
            validate_connection(&resolved_connection, &memories, &self.memory_definitions)?;
            validate_processor_memory_bindings(
                &name,
                definition,
                &resolved_connection,
                &memories,
                &self.memory_definitions,
            )?;
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
            for resource_name in &connection.resources {
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
                connection,
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
            memory_aliases: self.memory_aliases,
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
    aliases: &[MemoryAlias],
) -> Result<(), ArchitectureError> {
    let endpoint = aliases
        .iter()
        .find(|alias| alias.name == endpoint.memory)
        .map_or(endpoint, |alias| &alias.endpoint);
    let memory = memories
        .iter()
        .find(|memory| memory.name == endpoint.memory)
        .ok_or_else(|| {
            ArchitectureError::Invalid(format!(
                "network interface refers to unknown placed memory '{}'",
                endpoint.memory
            ))
        })?;
    if endpoint.indices.len() != memory.indices.len() {
        return Err(ArchitectureError::Invalid(format!(
            "network interface endpoint '{}' has {} indices; placed memory expects {}",
            endpoint.memory,
            endpoint.indices.len(),
            memory.indices.len()
        )));
    }
    let definition = definitions
        .iter()
        .find(|definition| definition.name == memory.definition)
        .expect("placed memory definition was validated");
    validate_region_selector(endpoint).map_err(ArchitectureError::Invalid)?;
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

fn validate_memory_aliases(aliases: &[MemoryAlias]) -> Result<(), ArchitectureError> {
    validate_unique(
        aliases.iter().map(|alias| alias.name.as_str()),
        "memory alias",
    )
}

fn validate_memory_alias_targets(
    aliases: &[MemoryAlias],
    memories: &[MemoryArray],
    definitions: &[MemoryDefinition],
) -> Result<(), ArchitectureError> {
    for alias in aliases {
        if memories.iter().any(|memory| memory.name == alias.name) {
            return Err(ArchitectureError::DuplicateName {
                kind: "memory or alias",
                name: alias.name.clone(),
            });
        }
        let memory = memories
            .iter()
            .find(|memory| memory.name == alias.endpoint.memory)
            .ok_or_else(|| ArchitectureError::UnknownReference {
                owner: format!("memory alias '{}'", alias.name),
                kind: "placed memory",
                name: alias.endpoint.memory.clone(),
            })?;
        if alias.endpoint.indices.len() != memory.indices.len() {
            return Err(ArchitectureError::RankMismatch {
                object: format!("memory alias '{}'", alias.name),
                expected: memory.indices.len(),
                actual: alias.endpoint.indices.len(),
            });
        }
        let definition = definitions
            .iter()
            .find(|definition| definition.name == memory.definition)
            .ok_or_else(|| ArchitectureError::UnknownReference {
                owner: format!("placed memory '{}'", memory.name),
                kind: "memory definition",
                name: memory.definition.clone(),
            })?;
        validate_region_selector(&alias.endpoint).map_err(ArchitectureError::Invalid)?;
        validate_static_bank(&alias.endpoint, definition).map_err(ArchitectureError::Invalid)?;
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
                .map(|memory| (memory.name.as_str(), memory.indices.as_slice())),
            "memory",
        )?;
        validate_membership(
            &scope.name,
            &scope_dimensions,
            &scope.processors,
            processors
                .iter()
                .map(|processor| (processor.name.as_str(), processor.axes.as_slice())),
            "processor",
        )?;
        validate_membership(
            &scope.name,
            &scope_dimensions,
            &scope.networks,
            networks
                .iter()
                .map(|network| (network.name.as_str(), network.dimensions.as_slice())),
            "network",
        )?;
        validate_membership(
            &scope.name,
            &scope_dimensions,
            &scope.resources,
            resources
                .iter()
                .map(|resource| (resource.name.as_str(), resource.indices.as_slice())),
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
    candidates: impl IntoIterator<Item = (&'a str, &'a [Axis])>,
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
        if !domain_is_prefix(scope_dimensions, domain) {
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

fn resolve_memory_aliases(
    connection: &mut Connection,
    aliases: &[MemoryAlias],
) -> Result<(), ArchitectureError> {
    for endpoint in connection.inputs.iter_mut().chain(&mut connection.outputs) {
        if let Some(alias) = aliases.iter().find(|alias| alias.name == endpoint.memory) {
            if !endpoint.indices.is_empty() || endpoint.bank.is_some() {
                return Err(ArchitectureError::Invalid(format!(
                    "memory alias '{}' cannot be further indexed",
                    alias.name
                )));
            }
            *endpoint = alias.endpoint.clone();
        }
    }
    Ok(())
}

fn validate_connection(
    connection: &Connection,
    memories: &[MemoryArray],
    definitions: &[MemoryDefinition],
) -> Result<(), ArchitectureError> {
    for endpoint in connection.inputs.iter().chain(&connection.outputs) {
        let memory = memories
            .iter()
            .find(|memory| memory.name == endpoint.memory)
            .ok_or_else(|| {
                ArchitectureError::Invalid(format!(
                    "connection refers to unknown placed memory '{}'",
                    endpoint.memory
                ))
            })?;
        if endpoint.indices.len() != memory.indices.len() {
            return Err(ArchitectureError::Invalid(format!(
                "endpoint '{}' has {} indices; placed memory expects {}",
                endpoint.memory,
                endpoint.indices.len(),
                memory.indices.len()
            )));
        }
        let definition = definitions
            .iter()
            .find(|definition| definition.name == memory.definition)
            .ok_or_else(|| {
                ArchitectureError::Invalid(format!(
                    "placed memory '{}' has unknown definition '{}'",
                    memory.name, memory.definition
                ))
            })?;
        validate_region_selector(endpoint).map_err(ArchitectureError::Invalid)?;
        validate_static_bank(endpoint, definition).map_err(ArchitectureError::Invalid)?;
    }
    Ok(())
}

fn validate_processor_memory_bindings(
    processor: &str,
    definition: &ProcessorDefinition,
    connection: &Connection,
    memories: &[MemoryArray],
    definitions: &[MemoryDefinition],
) -> Result<(), ArchitectureError> {
    if !matches!(definition.source_format, ProcessorSourceFormat::CompactLoom) {
        return Ok(());
    }
    let candidates = |endpoints: &[MemoryEndpoint]| {
        endpoints
            .iter()
            .map(|endpoint| {
                let memory = memories
                    .iter()
                    .find(|memory| memory.name == endpoint.memory)
                    .expect("connection memories were validated");
                let definition = definitions
                    .iter()
                    .find(|definition| definition.name == memory.definition)
                    .expect("connection memory definitions were validated");
                (endpoint.memory.clone(), definition.technology.clone())
            })
            .collect::<Vec<_>>()
    };
    let input_candidates = candidates(&connection.inputs);
    let output_candidates = candidates(&connection.outputs);
    for operation in &definition.functions {
        let Some(details) = &operation.func.mlir_details else {
            continue;
        };
        let requirements = details
            .memref_memory_requirements
            .iter()
            .cloned()
            .collect::<BTreeMap<_, _>>();
        let operands = |names: &[String]| {
            names
                .iter()
                .map(|name| (name.clone(), requirements.get(name).cloned()))
                .collect::<Vec<_>>()
        };
        for (role, names, candidates) in [
            ("input", &details.source_memrefs, &input_candidates),
            ("output", &details.target_memrefs, &output_candidates),
        ] {
            resolve_operand_memory_bindings(
                &operation.func.name,
                role,
                &operands(names),
                candidates,
            )
            .map_err(|error| {
                ArchitectureError::Invalid(format!("processor '{processor}': {error}"))
            })?;
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
            let Some(endpoint) = resolve_endpoint(symbolic, &point, memories, definitions)? else {
                continue 'point;
            };
            inputs.push(endpoint);
        }
        for symbolic in &connection.outputs {
            let Some(endpoint) = resolve_endpoint(symbolic, &point, memories, definitions)? else {
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
    for (selector, domain) in endpoint.indices.iter().zip(&memory.indices) {
        match selector {
            super::memory::EndpointIndex::All => indices.push(ResolvedEndpointIndex::All),
            super::memory::EndpointIndex::Expression(expression) => {
                let value = expression.evaluate(values).ok_or_else(|| {
                    ArchitectureError::Invalid(format!(
                        "could not evaluate index for memory '{}'",
                        endpoint.memory
                    ))
                })?;
                if value < 0 || value >= domain.extent as i64 {
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
