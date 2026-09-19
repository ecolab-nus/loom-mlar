use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Deserializer, de::Error as _, de::MapAccess, de::SeqAccess, de::Visitor};

use super::PerformanceYaml;
use crate::ArchitectureBuilder;
use crate::builder::FunctionProvider;
use crate::native::{NativeSourceIndex, compose_module};
use crate::selection::MemoryEndpoint;
use crate::templates::{self, FunctionSpec};
use crate::{Connection, NamedPort, ProcessorDefinition};
use mlar_rust::Architecture;
use mlar_rust::arch::network::{NetworkInterface, NetworkLink, NetworkTopology};
use mlar_rust::arch::resource::Resource;
use mlar_rust::arch::scope::Scope;
use mlar_rust::arch::{Banking, MemoryDefinition, MemoryDomain};
use mlar_rust::math::{AffineMap, Expr, Sym};
use mlar_rust::{OperationModel, ProcessorType};

#[derive(Debug)]
pub enum ArchLoadError {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Yaml {
        path: PathBuf,
        source: serde_yaml::Error,
    },
    Invalid(String),
}

impl fmt::Display for ArchLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(f, "failed to read '{}': {source}", path.display())
            }
            Self::Yaml { path, source } => {
                write!(f, "failed to parse '{}': {source}", path.display())
            }
            Self::Invalid(message) => write!(f, "invalid architecture package: {message}"),
        }
    }
}

impl std::error::Error for ArchLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Yaml { source, .. } => Some(source),
            Self::Invalid(_) => None,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChipYaml {
    name: String,
    #[serde(default = "default_memory_catalog_path")]
    memory: String,
    #[serde(default)]
    parameters: Vec<String>,
    #[serde(default, deserialize_with = "super::yaml::unique_map")]
    dimensions: BTreeMap<String, DimensionSizeYaml>,
    #[serde(default, deserialize_with = "super::yaml::unique_map")]
    memories: BTreeMap<String, MemoryPlacementYaml>,
    #[serde(default)]
    processors: ProcessorPlacementsYaml,
    #[serde(default)]
    resources: Vec<SharedResourceYaml>,
    #[serde(default)]
    networks: Vec<NetworkYaml>,
    #[serde(default)]
    scopes: Vec<ScopeYaml>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
enum DimensionSizeYaml {
    Literal(u64),
    Expression(String),
}

/// One element of an `axes:` list: a bare axis name, or a nested list that
/// starts the level below. `[cluster, [core]]` is a per-cluster array holding
/// per-core arrays; `[x, y]` is a single 2-d array.
#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
enum AxisSpecYaml {
    Name(String),
    Nested(Vec<AxisSpecYaml>),
}

impl AxisSpecYaml {
    /// Flatten one `axes:` list into levels, outer to inner.
    fn levels(specs: &[Self], memory: &str) -> Result<Vec<Vec<String>>, ArchLoadError> {
        let mut level = Vec::new();
        let mut nested = None;
        for spec in specs {
            match spec {
                Self::Name(name) => {
                    if nested.is_some() {
                        return Err(ArchLoadError::Invalid(format!(
                            "memory '{memory}': axis '{name}' follows a nested level; \
                             a nested list must be last in its level"
                        )));
                    }
                    level.push(name.clone());
                }
                Self::Nested(inner) => {
                    if nested.is_some() {
                        return Err(ArchLoadError::Invalid(format!(
                            "memory '{memory}': a level may hold at most one nested level, \
                             because adl.memory.array takes a single child"
                        )));
                    }
                    nested = Some(inner);
                }
            }
        }
        if level.is_empty() {
            return Err(ArchLoadError::Invalid(format!(
                "memory '{memory}': a level must declare at least one axis"
            )));
        }
        let mut levels = vec![level];
        if let Some(inner) = nested {
            levels.extend(Self::levels(inner, memory)?);
        }
        Ok(levels)
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged, deny_unknown_fields)]
enum MemoryPlacementYaml {
    /// `L1:` — placed once, no replication.
    Single,
    Direct(Vec<AxisSpecYaml>),
    Detailed {
        #[serde(default)]
        definition: Option<String>,
        domain: MemoryDomain,
        #[serde(default)]
        axes: Vec<AxisSpecYaml>,
    },
}

impl MemoryPlacementYaml {
    fn resolve(
        &self,
        name: &str,
    ) -> Result<(String, MemoryDomain, Vec<Vec<String>>), ArchLoadError> {
        let (definition, domain, specs) = match self {
            Self::Single => {
                return Err(ArchLoadError::Invalid(format!(
                    "memory placement '{name}' must declare `domain: DRAM` or `domain: L1`"
                )));
            }
            Self::Direct(specs) => {
                let _ = specs;
                return Err(ArchLoadError::Invalid(format!(
                    "memory placement '{name}' must declare `domain: DRAM` or `domain: L1`"
                )));
            }
            Self::Detailed {
                definition,
                domain,
                axes,
            } => (
                definition.clone().unwrap_or_else(|| name.to_string()),
                *domain,
                axes.as_slice(),
            ),
        };
        let levels = if specs.is_empty() {
            Vec::new()
        } else {
            AxisSpecYaml::levels(specs, name)?
        };
        Ok((definition, domain, levels))
    }
}

fn default_memory_catalog_path() -> String {
    "memory.yaml".into()
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProcessorPlacementYaml {
    definition: String,
    #[serde(default)]
    domain: Vec<String>,
    #[serde(
        default,
        alias = "ins",
        deserialize_with = "deserialize_named_memory_endpoints"
    )]
    inputs: Vec<NamedPort>,
    #[serde(
        default,
        alias = "outs",
        deserialize_with = "deserialize_named_memory_endpoints"
    )]
    outputs: Vec<NamedPort>,
    #[serde(default)]
    resources: Vec<String>,
}

#[derive(Clone, Debug, Default)]
struct ProcessorPlacementsYaml(Vec<(String, ProcessorPlacementYaml)>);

impl<'de> Deserialize<'de> for ProcessorPlacementsYaml {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let mapping = serde_yaml::Mapping::deserialize(deserializer)?;
        mapping
            .into_iter()
            .map(|(name, placement)| {
                let name = name
                    .as_str()
                    .ok_or_else(|| D::Error::custom("processor placement names must be strings"))?
                    .to_string();
                let placement =
                    ProcessorPlacementYaml::deserialize(placement).map_err(D::Error::custom)?;
                Ok((name, placement))
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Self)
    }
}

fn deserialize_named_memory_endpoints<'de, D>(deserializer: D) -> Result<Vec<NamedPort>, D::Error>
where
    D: Deserializer<'de>,
{
    struct NamedPortsVisitor;

    impl<'de> Visitor<'de> for NamedPortsVisitor {
        type Value = Vec<NamedPort>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter
                .write_str("a list of memory endpoints or a mapping from port aliases to endpoints")
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de>,
        {
            let mut ports = Vec::new();
            let mut names = std::collections::BTreeSet::new();
            while let Some(endpoint) = seq.next_element::<String>()? {
                let endpoint = MemoryEndpoint::parse(&endpoint).map_err(A::Error::custom)?;
                let name = endpoint.memory.clone();
                if !names.insert(name.clone()) {
                    return Err(A::Error::custom(format!(
                        "duplicate processor port '{name}'; use explicit aliases for multiple endpoints of the same memory"
                    )));
                }
                ports.push(NamedPort::new(name, endpoint));
            }
            Ok(ports)
        }

        fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
        where
            A: MapAccess<'de>,
        {
            let mut ports = Vec::new();
            let mut names = std::collections::BTreeSet::new();
            while let Some((name, endpoint)) = map.next_entry::<String, String>()? {
                if !names.insert(name.clone()) {
                    return Err(A::Error::custom(format!(
                        "duplicate processor port '{name}'"
                    )));
                }
                ports.push(NamedPort::new(
                    name,
                    MemoryEndpoint::parse(&endpoint).map_err(A::Error::custom)?,
                ));
            }
            Ok(ports)
        }
    }

    deserializer.deserialize_any(NamedPortsVisitor)
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SharedResourceYaml {
    name: String,
    #[serde(default)]
    capacity: Option<u64>,
    #[serde(default)]
    dimensions: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResourceYaml {
    name: String,
    #[serde(default)]
    capacity: Option<u64>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NetworkYaml {
    name: String,
    dimensions: Vec<String>,
    #[serde(default)]
    parameters: Vec<String>,
    #[serde(default)]
    links: Vec<NetworkLinkYaml>,
    #[serde(default)]
    interfaces: Vec<NetworkInterfaceYaml>,
    #[serde(default)]
    resources: Vec<ResourceYaml>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NetworkLinkYaml {
    name: String,
    map: String,
    bandwidth: String,
    #[serde(default)]
    latency: Option<String>,
    #[serde(default)]
    resource: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NetworkInterfaceYaml {
    name: String,
    endpoint: String,
    #[serde(default)]
    injection_bandwidth: Option<String>,
    #[serde(default)]
    ejection_bandwidth: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ScopeYaml {
    name: String,
    #[serde(default)]
    parent: Option<String>,
    #[serde(default)]
    dimensions: Vec<String>,
    #[serde(default)]
    memories: Vec<String>,
    #[serde(default)]
    processors: Vec<String>,
    #[serde(default)]
    networks: Vec<String>,
    #[serde(default)]
    resources: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MemoryCatalogYaml {
    #[serde(default)]
    memories: MemoryDefinitionsYaml,
}

#[derive(Clone, Debug, Default)]
struct MemoryDefinitionsYaml(Vec<(String, MemoryDefinitionYaml)>);

impl<'de> Deserialize<'de> for MemoryDefinitionsYaml {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let mapping = serde_yaml::Mapping::deserialize(deserializer)?;
        mapping
            .into_iter()
            .map(|(name, definition)| {
                let name = name
                    .as_str()
                    .ok_or_else(|| D::Error::custom("memory names must be strings"))?
                    .to_string();
                let definition =
                    MemoryDefinitionYaml::deserialize(definition).map_err(D::Error::custom)?;
                Ok((name, definition))
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Self)
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MemoryDefinitionYaml {
    capacity: ScalarExprYaml,
    word_size: ScalarExprYaml,
    #[serde(default)]
    banking: Option<BankingYaml>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged, deny_unknown_fields)]
enum BankingYaml {
    Count(ScalarExprYaml),
    Detailed { banks: ScalarExprYaml },
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
enum ScalarExprYaml {
    Literal(u64),
    Expression(String),
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessorYaml {
    #[serde(default)]
    name: Option<String>,
    #[serde(default, rename = "type")]
    processor_type: Option<ProcessorType>,
    #[serde(deserialize_with = "super::yaml::unique_entries")]
    functions: Vec<(String, FunctionSpec)>,
    #[serde(default)]
    resources: Vec<ResourceYaml>,
    #[serde(default)]
    performance: Option<PerformanceYaml>,
}

impl ChipYaml {
    pub fn from_yaml_str(input: &str) -> Result<Self, ArchLoadError> {
        serde_yaml::from_str(input).map_err(|source| ArchLoadError::Yaml {
            path: PathBuf::from("<string>"),
            source,
        })
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, ArchLoadError> {
        read_yaml(path.as_ref())
    }

    pub(crate) fn processor_definition_paths(&self, directory: &Path) -> Vec<PathBuf> {
        let mut paths = self
            .processors
            .0
            .iter()
            .map(|(_, placement)| directory.join(&placement.definition))
            .collect::<Vec<_>>();
        paths.sort();
        paths.dedup();
        paths
    }

    pub fn build(&self, artifact_dir: impl AsRef<Path>) -> Result<Architecture, ArchLoadError> {
        self.build_with_bindings(artifact_dir, std::iter::empty::<(&str, u64)>())
    }

    pub fn build_with_bindings(
        &self,
        artifact_dir: impl AsRef<Path>,
        bindings: impl IntoIterator<Item = (impl Into<String>, u64)>,
    ) -> Result<Architecture, ArchLoadError> {
        let artifact_dir = artifact_dir.as_ref();
        let catalog_path = artifact_dir.join(&self.memory);
        let catalog_yaml: MemoryCatalogYaml = read_yaml(&catalog_path)?;
        mlar_rust::arch::perf::validate_symbols(&Sym::from_names(self.parameters.iter().cloned()))
            .map_err(ArchLoadError::Invalid)?;
        let declared = self
            .parameters
            .iter()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>();
        if declared.len() != self.parameters.len() {
            return Err(ArchLoadError::Invalid(
                "duplicate architecture parameter".into(),
            ));
        }
        let mut unique_bindings = BTreeMap::new();
        for (name, value) in bindings {
            let name = name.into();
            if unique_bindings.insert(name.clone(), value).is_some() {
                return Err(ArchLoadError::Invalid(format!(
                    "duplicate binding for architecture parameter '{name}'"
                )));
            }
        }
        let bindings = unique_bindings;
        for name in bindings.keys() {
            if !declared.contains(name) {
                return Err(ArchLoadError::Invalid(format!(
                    "binding supplied for unknown architecture parameter '{name}'"
                )));
            }
        }
        for name in &declared {
            if !bindings.contains_key(name) {
                return Err(ArchLoadError::Invalid(format!(
                    "architecture parameter '{name}' has no binding"
                )));
            }
        }
        let substitutions = bindings
            .iter()
            .map(|(name, value)| (Sym::new(name), Expr::from(*value)))
            .collect::<Vec<_>>();
        let memory_definitions = catalog_yaml.build(&declared, &substitutions)?;

        let mut concrete_dimensions = BTreeMap::new();
        for (name, size) in &self.dimensions {
            let expression = match size {
                DimensionSizeYaml::Literal(size) => Expr::from(*size),
                DimensionSizeYaml::Expression(expression) => {
                    Expr::parse(expression).map_err(|error| {
                        ArchLoadError::Invalid(format!("dimension '{name}': {error}"))
                    })?
                }
            };
            for symbol in expression.free_symbols() {
                if !declared.contains(&symbol.0) {
                    return Err(ArchLoadError::Invalid(format!(
                        "dimension '{name}' uses undeclared parameter '{}'",
                        symbol.0
                    )));
                }
            }
            let expression = expression.substitute(&substitutions);
            let size = expression
                .as_const()
                .filter(|size| *size > 0)
                .ok_or_else(|| {
                    ArchLoadError::Invalid(format!(
                        "dimension '{name}' does not instantiate to a positive u64: {expression}"
                    ))
                })?;
            concrete_dimensions.insert(name.clone(), size);
        }

        let mut builder = ArchitectureBuilder::new(&self.name);
        for definition in memory_definitions {
            builder = builder.memory_definition(definition);
        }
        for (name, size) in &concrete_dimensions {
            builder = builder.axis(name, *size);
        }
        for (name, placement) in &self.memories {
            let (definition, domain, levels) = placement.resolve(name)?;
            builder = builder.place_memory_levels(name, definition, domain, levels);
        }
        for resource in &self.resources {
            let indices = resource
                .dimensions
                .iter()
                .map(|name| {
                    concrete_dimensions
                        .get(name)
                        .copied()
                        .map(|extent| mlar_rust::Axis::new(name, extent))
                        .ok_or_else(|| {
                            ArchLoadError::Invalid(format!(
                                "resource '{}' uses unknown dimension '{}'",
                                resource.name, name
                            ))
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let definition = ResourceYaml {
                name: resource.name.clone(),
                capacity: resource.capacity,
            };
            builder = builder.resource(definition.build().indexed(indices));
        }
        for network in &self.networks {
            let network = network.build(&concrete_dimensions, &bindings, &builder)?;
            builder = builder.network(network);
        }

        let mut loaded_definitions = BTreeMap::<String, String>::new();
        let native_sources =
            NativeSourceIndex::load(artifact_dir).map_err(ArchLoadError::Invalid)?;
        let mut placements = Vec::new();
        for (placement_name, placement) in &self.processors.0 {
            let definition_name = if let Some(name) = loaded_definitions.get(&placement.definition)
            {
                name.clone()
            } else {
                let path = artifact_dir.join(&placement.definition);
                let processor_yaml = ProcessorYaml::from_file(&path)?;
                let definition =
                    processor_yaml.build_definition_with_index(&path, &native_sources)?;
                let name = definition.name.clone();
                builder = builder.processor_definition(definition);
                loaded_definitions.insert(placement.definition.clone(), name.clone());
                name
            };
            placements.push((placement_name, definition_name, placement.build()));
        }
        for (placement_name, definition_name, connection) in placements {
            builder = builder.connect_as(placement_name, definition_name, connection);
        }
        for scope in &self.scopes {
            builder = builder.scope(scope.build());
        }
        builder
            .build()
            .map_err(|error| ArchLoadError::Invalid(error.to_string()))
    }
}

impl NetworkYaml {
    fn build(
        &self,
        dimensions: &BTreeMap<String, u64>,
        bindings: &BTreeMap<String, u64>,
        builder: &ArchitectureBuilder,
    ) -> Result<NetworkTopology, ArchLoadError> {
        let network_dimensions = self
            .dimensions
            .iter()
            .map(|name| {
                dimensions
                    .get(name)
                    .copied()
                    .map(|size| mlar_rust::Axis::new(name, size))
                    .ok_or_else(|| {
                        ArchLoadError::Invalid(format!(
                            "network '{}' uses unknown dimension '{}'",
                            self.name, name
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let expression_bindings = bindings
            .iter()
            .map(|(name, value)| (Sym::new(name), Expr::from(*value)))
            .collect::<Vec<_>>();
        for parameter in &self.parameters {
            if bindings.contains_key(parameter) {
                return Err(ArchLoadError::Invalid(format!(
                    "network '{}' parameter '{parameter}' conflicts with a bound architecture parameter",
                    self.name
                )));
            }
        }
        let mut network = NetworkTopology::new(&self.name, network_dimensions)
            .with_parameters(Sym::from_names(self.parameters.iter().cloned()));
        for resource in &self.resources {
            let indices = network.dimensions.clone();
            network = network.with_resource(resource.build().indexed(indices));
        }
        for link in &self.links {
            let affine_bindings = bindings
                .iter()
                .map(|(name, value)| {
                    i64::try_from(*value)
                        .map(|value| (name.clone(), value))
                        .map_err(|_| {
                            ArchLoadError::Invalid(format!(
                                "network '{}' binding '{}' is too large for affine evaluation",
                                self.name, name
                            ))
                        })
                })
                .collect::<Result<BTreeMap<_, _>, _>>()?;
            let map =
                AffineMap::parse_with_bindings(&link.map, &network.dimensions, &affine_bindings)
                    .map_err(|error| {
                        ArchLoadError::Invalid(format!(
                            "network '{}' link '{}': {error}",
                            self.name, link.name
                        ))
                    })?;
            let bandwidth = Expr::parse(&link.bandwidth)
                .map_err(|error| {
                    ArchLoadError::Invalid(format!(
                        "network '{}' link '{}' bandwidth: {error}",
                        self.name, link.name
                    ))
                })?
                .substitute(&expression_bindings);
            let mut built = NetworkLink::new(&link.name, map, bandwidth);
            if let Some(latency) = &link.latency {
                built = built.with_latency(
                    Expr::parse(latency)
                        .map_err(|error| {
                            ArchLoadError::Invalid(format!(
                                "network '{}' link '{}' latency: {error}",
                                self.name, link.name
                            ))
                        })?
                        .substitute(&expression_bindings),
                );
            }
            if let Some(resource) = &link.resource {
                built = built.with_resource(resource);
            }
            network = network.with_link(built);
        }
        for interface in &self.interfaces {
            let mut built = NetworkInterface::new(
                &interface.name,
                builder
                    .lower_endpoint(&MemoryEndpoint::parse(&interface.endpoint).map_err(
                        |error| {
                            ArchLoadError::Invalid(format!(
                                "network '{}' interface '{}': {error}",
                                self.name, interface.name
                            ))
                        },
                    )?)
                    .map_err(ArchLoadError::Invalid)?,
            );
            if let Some(bandwidth) = &interface.injection_bandwidth {
                built = built.with_injection_bandwidth(
                    Expr::parse(bandwidth)
                        .map_err(|error| {
                            ArchLoadError::Invalid(format!(
                                "network '{}' interface '{}' injection bandwidth: {error}",
                                self.name, interface.name
                            ))
                        })?
                        .substitute(&expression_bindings),
                );
            }
            if let Some(bandwidth) = &interface.ejection_bandwidth {
                built = built.with_ejection_bandwidth(
                    Expr::parse(bandwidth)
                        .map_err(|error| {
                            ArchLoadError::Invalid(format!(
                                "network '{}' interface '{}' ejection bandwidth: {error}",
                                self.name, interface.name
                            ))
                        })?
                        .substitute(&expression_bindings),
                );
            }
            network = network.with_interface(built);
        }
        Ok(network)
    }
}

impl ScopeYaml {
    fn build(&self) -> Scope {
        let mut scope = Scope::new(&self.name, self.dimensions.iter().cloned())
            .with_memories(self.memories.iter().cloned())
            .with_processors(self.processors.iter().cloned())
            .with_networks(self.networks.iter().cloned())
            .with_resources(self.resources.iter().cloned());
        if let Some(parent) = &self.parent {
            scope = scope.with_parent(parent);
        }
        scope
    }
}

impl MemoryCatalogYaml {
    fn build(
        self,
        declared: &std::collections::BTreeSet<String>,
        substitutions: &[(Sym, Expr)],
    ) -> Result<Vec<MemoryDefinition>, ArchLoadError> {
        let MemoryCatalogYaml { memories } = self;
        let mut definitions = memories
            .0
            .into_iter()
            .map(|(name, memory)| -> Result<_, ArchLoadError> {
                let capacity = memory.capacity.resolve(
                    &format!("memory '{name}' capacity"),
                    declared,
                    substitutions,
                )?;
                let word_size = memory.word_size.resolve(
                    &format!("memory '{name}' word_size"),
                    declared,
                    substitutions,
                )?;
                let banking = memory
                    .banking
                    .map(|banking| match banking {
                        BankingYaml::Count(banks) | BankingYaml::Detailed { banks } => banks,
                    })
                    .map(|banks| {
                        banks
                            .resolve(&format!("memory '{name}' banks"), declared, substitutions)
                            .map(|banks| Banking { banks })
                    })
                    .transpose()?;
                Ok(MemoryDefinition {
                    name,
                    capacity,
                    word_size,
                    banking,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        definitions.sort_by(|left, right| left.name.cmp(&right.name));
        Ok(definitions)
    }
}

impl ScalarExprYaml {
    fn resolve(
        &self,
        label: &str,
        declared: &std::collections::BTreeSet<String>,
        substitutions: &[(Sym, Expr)],
    ) -> Result<u64, ArchLoadError> {
        let expression = match self {
            Self::Literal(value) => Expr::from(*value),
            Self::Expression(expression) => Expr::parse(expression)
                .map_err(|error| ArchLoadError::Invalid(format!("{label}: {error}")))?,
        };
        for symbol in expression.free_symbols() {
            if !declared.contains(&symbol.0) {
                return Err(ArchLoadError::Invalid(format!(
                    "{label} uses undeclared parameter '{}'",
                    symbol.0
                )));
            }
        }
        let expression = expression.substitute(substitutions);
        expression
            .as_const()
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                ArchLoadError::Invalid(format!(
                    "{label} does not instantiate to a positive u64: {expression}"
                ))
            })
    }
}

impl ProcessorYaml {
    pub fn from_yaml_str(input: &str) -> Result<Self, ArchLoadError> {
        serde_yaml::from_str(input).map_err(|source| ArchLoadError::Yaml {
            path: PathBuf::from("<string>"),
            source,
        })
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, ArchLoadError> {
        read_yaml(path.as_ref())
    }

    pub fn build_definition(
        &self,
        processor_yaml_path: impl AsRef<Path>,
    ) -> Result<ProcessorDefinition, ArchLoadError> {
        let processor_yaml_path = processor_yaml_path.as_ref();
        let base_dir = processor_yaml_path
            .parent()
            .unwrap_or_else(|| Path::new("."));
        let index = NativeSourceIndex::load(base_dir).map_err(ArchLoadError::Invalid)?;
        self.build_definition_with_index(processor_yaml_path, &index)
    }

    pub(crate) fn build_definition_with_index(
        &self,
        processor_yaml_path: &Path,
        index: &NativeSourceIndex,
    ) -> Result<ProcessorDefinition, ArchLoadError> {
        if self.functions.is_empty() {
            return Err(ArchLoadError::Invalid(format!(
                "{} must define at least one function",
                processor_yaml_path.display()
            )));
        }
        let mut providers = Vec::new();
        let mut blocks = Vec::new();
        for (exposed_name, function) in &self.functions {
            let provider = if templates::is_registered(&function.source) {
                let block = templates::emit_preview(exposed_name, function).map_err(|error| {
                    ArchLoadError::Invalid(format!(
                        "{} function '{exposed_name}': {error}",
                        processor_yaml_path.display()
                    ))
                })?;
                blocks.push(block);
                FunctionProvider::Template(function.clone())
            } else {
                if function.extent.is_some() {
                    return Err(ArchLoadError::Invalid(format!(
                        "{} function '{exposed_name}' supplies template-only `extent` to native source '{}'",
                        processor_yaml_path.display(),
                        function.source
                    )));
                }
                if exposed_name != &function.source {
                    return Err(ArchLoadError::Invalid(format!(
                        "{} function '{exposed_name}' aliases native source '{}'; native aliases are unsupported",
                        processor_yaml_path.display(),
                        function.source
                    )));
                }
                let native = index.get(&function.source).ok_or_else(|| {
                    ArchLoadError::Invalid(format!(
                        "{} function '{exposed_name}' has unknown source '{}'",
                        processor_yaml_path.display(),
                        function.source
                    ))
                })?;
                blocks.push(native.block.clone());
                FunctionProvider::Native(native.clone(), function.bindings.clone())
            };
            providers.push((exposed_name.clone(), provider));
        }
        let source = compose_module(blocks);
        let module = mlar_rust::mlir::MlirModule::from_mlir_source(&source).map_err(|error| {
            ArchLoadError::Invalid(format!("{}: {error}", processor_yaml_path.display()))
        })?;
        for ((exposed_name, function), parsed) in self.functions.iter().zip(&module.functions) {
            if !templates::is_registered(&function.source) {
                validate_native_metadata(exposed_name, function, parsed).map_err(|error| {
                    ArchLoadError::Invalid(format!(
                        "{} function '{exposed_name}': {error}",
                        processor_yaml_path.display()
                    ))
                })?;
            }
        }
        let spec = self.performance.as_ref().ok_or_else(|| {
            ArchLoadError::Invalid(format!(
                "{} must define `performance`",
                processor_yaml_path.display()
            ))
        })?;
        let models = spec.models_for_module(&module).map_err(|error| {
            ArchLoadError::Invalid(format!("{}: {error}", processor_yaml_path.display()))
        })?;
        let functions = module
            .functions
            .into_iter()
            .zip(models)
            .map(|(mut function, perf)| {
                for symbol in &perf.symbols {
                    if !function.symbols.contains(symbol) {
                        function.symbols.push(symbol.clone());
                    }
                }
                function.symbols.sort();
                Ok(OperationModel::new(function, perf))
            })
            .collect::<Result<Vec<_>, ArchLoadError>>()?;
        let name = self.name.clone().unwrap_or_else(|| {
            processor_yaml_path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .unwrap_or("processor")
                .trim_end_matches(".processor")
                .to_string()
        });
        Ok(ProcessorDefinition {
            name,
            processor_type: self.processor_type.clone(),
            source,
            functions,
            resources: self.resources.iter().map(ResourceYaml::build).collect(),
            providers: Some(providers),
            authored_native: true,
        })
    }
}

fn validate_native_metadata(
    _name: &str,
    spec: &FunctionSpec,
    function: &mlar_rust::mlir::MlirFunc,
) -> Result<(), String> {
    if !spec.dimensions.is_empty() || !spec.symbols.is_empty() {
        let declared = spec
            .dimensions
            .iter()
            .chain(&spec.symbols)
            .cloned()
            .collect::<BTreeSet<_>>();
        if declared.len() != spec.dimensions.len() + spec.symbols.len() {
            return Err("`dimensions` and `symbols` contain a duplicate name".into());
        }
        let native = function
            .symbols
            .iter()
            .map(|symbol| symbol.0.clone())
            .collect::<BTreeSet<_>>();
        if declared != native {
            return Err(format!(
                "declared dimensions/symbols {declared:?} do not match native MLIR symbols {native:?}"
            ));
        }
    }

    if let Some(element_type) = &spec.element_type {
        let types = function
            .mlir_details
            .as_ref()
            .into_iter()
            .flat_map(|details| &details.memref_arg_types)
            .map(|(_, ty)| ty);
        if !types
            .into_iter()
            .any(|ty| memref_has_element_type(ty, element_type))
        {
            return Err(format!(
                "element type '{element_type}' does not occur in the native MLIR memref arguments"
            ));
        }
    }
    Ok(())
}

fn memref_has_element_type(memref: &str, element_type: &str) -> bool {
    let Some(body) = memref
        .strip_prefix("memref<")
        .and_then(|body| body.strip_suffix('>'))
    else {
        return false;
    };
    let shape_and_element = body.split(',').next().unwrap_or(body).trim();
    shape_and_element == element_type || shape_and_element.ends_with(&format!("x{element_type}"))
}

impl ProcessorPlacementYaml {
    fn build(&self) -> Connection {
        Connection::named(
            self.domain.iter().cloned(),
            self.inputs.clone(),
            self.outputs.clone(),
        )
        .with_resources(self.resources.iter().cloned())
    }
}

impl ResourceYaml {
    fn build(&self) -> Resource {
        match self.capacity {
            Some(capacity) => Resource::quantitative(&self.name, capacity),
            None => Resource::exclusive(&self.name),
        }
    }
}

fn read_text(path: &Path) -> Result<String, ArchLoadError> {
    std::fs::read_to_string(path).map_err(|source| ArchLoadError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn read_yaml<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, ArchLoadError> {
    let input = read_text(path)?;
    serde_yaml::from_str(&input).map_err(|source| ArchLoadError::Yaml {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::{MemoryCatalogYaml, MemoryDomain, ProcessorPlacementYaml};
    use crate::selection::MemoryEndpoint;
    use mlar_rust::{Expr, Sym};

    #[test]
    fn connection_values_deserialize_as_endpoints() {
        let connection: ProcessorPlacementYaml = serde_yaml::from_str(
            r#"
definition: lane.yaml
domain: [x, y]
inputs: {data: "L1[x, y]"}
outputs: {result: "L2[x floordiv 2, y floordiv 2]"}
"#,
        )
        .expect("connection should deserialize");

        assert_eq!(
            connection
                .inputs
                .iter()
                .map(|port| &port.endpoint)
                .collect::<Vec<_>>(),
            [&MemoryEndpoint::parse("L1[x, y]").unwrap()]
        );
        assert_eq!(
            connection
                .outputs
                .iter()
                .map(|port| &port.endpoint)
                .collect::<Vec<_>>(),
            [&MemoryEndpoint::parse("L2[x floordiv 2, y floordiv 2]").unwrap()]
        );
    }

    #[test]
    fn an_explicit_domain_and_absent_axes_mean_one_instance() {
        let chip: super::ChipYaml = serde_yaml::from_str(
            r#"
name: single
memories:
  L1: {domain: L1}
processors:
  lane:
    definition: lane.yaml
    inputs: {data: "L1"}
    outputs: {result: "L1"}
"#,
        )
        .expect("chip should deserialize");

        let (definition, domain, levels) = chip.memories["L1"].resolve("L1").expect("L1 levels");
        assert_eq!(definition, "L1");
        assert_eq!(domain, MemoryDomain::L1);
        assert!(levels.is_empty());
        assert!(chip.processors.0[0].1.domain.is_empty());
    }

    #[test]
    fn nested_axes_become_levels_outer_to_inner() {
        let chip: super::ChipYaml = serde_yaml::from_str(
            r#"
name: nested
memories:
  L1: {domain: L1, axes: [cluster, [core]]}
  L2: {domain: L1, axes: [x, y]}
"#,
        )
        .expect("chip should deserialize");

        let (definition, _, levels) = chip.memories["L1"].resolve("L1").expect("L1 levels");
        assert_eq!(definition, "L1");
        assert_eq!(
            levels,
            vec![vec!["cluster".to_string()], vec!["core".to_string()]]
        );

        let (_, _, flat) = chip.memories["L2"].resolve("L2").expect("L2 levels");
        assert_eq!(flat, vec![vec!["x".to_string(), "y".to_string()]]);
    }

    #[test]
    fn a_nested_level_must_be_last_in_its_level() {
        let chip: super::ChipYaml = serde_yaml::from_str(
            r#"
name: bad
memories:
  L1: {domain: L1, axes: [cluster, [core], extra]}
"#,
        )
        .expect("chip should deserialize");

        let error = chip.memories["L1"]
            .resolve("L1")
            .expect_err("an axis after a nested level must be rejected");
        assert!(error.to_string().contains("must be last in its level"));
    }

    #[test]
    fn invalid_endpoint_fails_deserialization() {
        let error = serde_yaml::from_str::<super::ChipYaml>(
            r#"
name: bad
memories:
  L1: [x]
processors:
  lane:
    definition: lane.yaml
    domain: [x]
    inputs: {data: "L1[:,"}
    outputs: {result: "L1[x]"}
"#,
        )
        .expect_err("invalid endpoint must fail while deserializing");

        assert!(error.to_string().contains("memory indices must end with"));
    }

    #[test]
    fn endpoint_lists_derive_names_and_preserve_order() {
        let placement: super::ProcessorPlacementYaml = serde_yaml::from_str(
            "definition: lane.yaml\ninputs: ['R[x]', 'S[x]']\noutputs: ['R[x]']\n",
        )
        .unwrap();
        assert_eq!(
            placement
                .inputs
                .iter()
                .map(|p| p.name.as_str())
                .collect::<Vec<_>>(),
            ["R", "S"]
        );
        assert_eq!(placement.outputs[0].name, "R");
        assert_eq!(
            placement.inputs[0].endpoint,
            super::MemoryEndpoint::parse("R[x]").unwrap()
        );
    }

    #[test]
    fn repeated_memory_endpoints_require_aliases() {
        for endpoints in ["['L1[x]', 'L1[x + 1]']", "['L1[x]', 'L1[x]']"] {
            let error = serde_yaml::from_str::<super::ProcessorPlacementYaml>(&format!(
                "definition: lane.yaml\ninputs: {endpoints}\n"
            ))
            .unwrap_err();
            assert!(
                error.to_string().contains("use explicit aliases"),
                "{error}"
            );
        }
        let placement: super::ProcessorPlacementYaml = serde_yaml::from_str(
            "definition: lane.yaml\ninputs: {local: 'L1[x]', neighbor: 'L1[x + 1]'}\n",
        )
        .unwrap();
        assert_eq!(placement.inputs[0].name, "local");
        assert_eq!(placement.inputs[1].name, "neighbor");
    }

    #[test]
    fn duplicate_processor_ports_fail_deserialization() {
        let error = serde_yaml::from_str::<super::ChipYaml>(
            r#"
name: bad
processors:
  lane:
    definition: lane.yaml
    inputs:
      data: L1
      data: L2
"#,
        )
        .expect_err("duplicate ports must be rejected");
        assert!(error.to_string().contains("duplicate"), "{error}");
    }

    #[test]
    fn memory_geometry_accepts_architecture_parameters() {
        let catalog: MemoryCatalogYaml = serde_yaml::from_str(
            r#"
memories:
  L1:
    capacity: "X * 256"
    word_size: 16
    banking: X
"#,
        )
        .expect("symbolic memory geometry syntax");
        let declared = ["X".to_string()].into_iter().collect();
        let definitions = catalog
            .build(&declared, &[(Sym::new("X"), Expr::Const(4))])
            .expect("symbolic memory geometry should instantiate");
        let l1 = definitions
            .iter()
            .find(|definition| definition.name == "L1")
            .unwrap();
        assert_eq!(l1.capacity, 1024);
        assert_eq!(l1.banking.as_ref().unwrap().banks, 4);
    }

    #[test]
    fn memory_catalog_rejects_technology_identity() {
        let error = serde_yaml::from_str::<MemoryCatalogYaml>(
            r#"
memories:
  cache_a:
    technology: gcram
    capacity: 1024
    word_size: 16
"#,
        )
        .expect_err("technology no longer defines memory identity");
        assert!(error.to_string().contains("technology"));
    }
}
