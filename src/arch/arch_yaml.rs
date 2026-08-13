use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Deserializer, de::Error as _};

use super::PerformanceYaml;
use super::architecture::{Architecture, ArchitectureBuilder};
use super::memory::{Banking, MemoryAlias, MemoryDefinition, MemoryEndpoint, MemoryTechnology};
use super::network::{NetworkInterface, NetworkLink, NetworkTopology};
use super::processor::{
    Connection, OperationModel, ProcessorDefinition, ProcessorSourceFormat, ProcessorType,
};
use super::resource::Resource;
use super::scope::Scope;
use crate::math::{AffineMap, Expr, Sym};
use crate::mlir::parse_loom_source;

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
    #[serde(default)]
    dimensions: BTreeMap<String, DimensionSizeYaml>,
    #[serde(default)]
    memories: BTreeMap<String, MemoryPlacementYaml>,
    #[serde(default)]
    processors: ProcessorPlacementsYaml,
    #[serde(default)]
    resources: Vec<ResourceYaml>,
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

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
enum MemoryPlacementYaml {
    Direct(Vec<String>),
    Detailed {
        model: String,
        dimensions: Vec<String>,
    },
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
        deserialize_with = "deserialize_memory_endpoints"
    )]
    inputs: Vec<MemoryEndpoint>,
    #[serde(
        default,
        alias = "outs",
        deserialize_with = "deserialize_memory_endpoints"
    )]
    outputs: Vec<MemoryEndpoint>,
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

fn deserialize_memory_endpoints<'de, D>(deserializer: D) -> Result<Vec<MemoryEndpoint>, D::Error>
where
    D: Deserializer<'de>,
{
    Vec::<String>::deserialize(deserializer)?
        .into_iter()
        .map(|endpoint| MemoryEndpoint::parse(&endpoint).map_err(D::Error::custom))
        .collect()
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
    #[serde(default, deserialize_with = "deserialize_memory_regions")]
    regions: BTreeMap<String, MemoryEndpoint>,
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

fn deserialize_memory_regions<'de, D>(
    deserializer: D,
) -> Result<BTreeMap<String, MemoryEndpoint>, D::Error>
where
    D: Deserializer<'de>,
{
    BTreeMap::<String, String>::deserialize(deserializer)?
        .into_iter()
        .map(|(name, endpoint)| {
            let endpoint = MemoryEndpoint::parse(&endpoint).map_err(|error| {
                D::Error::custom(format!("invalid memory region '{name}': {error}"))
            })?;
            Ok((name, endpoint))
        })
        .collect()
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MemoryDefinitionYaml {
    #[serde(default)]
    indices: Vec<String>,
    capacity: ScalarExprYaml,
    word_size: ScalarExprYaml,
    #[serde(default)]
    technology: Option<String>,
    #[serde(default)]
    banking: Option<BankingYaml>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
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
    source: String,
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
        let bindings = bindings
            .into_iter()
            .map(|(name, value)| (name.into(), value))
            .collect::<BTreeMap<_, _>>();
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
        let (memory_definitions, aliases) = catalog_yaml.build(&declared, &substitutions)?;

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
        for alias in aliases {
            builder = builder.memory_alias(alias);
        }
        for (name, size) in &concrete_dimensions {
            builder = builder.axis(name, *size);
        }
        for (name, placement) in &self.memories {
            match placement {
                MemoryPlacementYaml::Direct(dimensions) => {
                    builder = builder.place_memory_as(name, name, dimensions.iter().cloned());
                }
                MemoryPlacementYaml::Detailed { model, dimensions } => {
                    builder = builder.place_memory_as(name, model, dimensions.iter().cloned());
                }
            }
        }
        for resource in &self.resources {
            builder = builder.resource(resource.build());
        }
        for network in &self.networks {
            builder = builder.network(network.build(&concrete_dimensions, &bindings)?);
        }

        let mut loaded_definitions = BTreeMap::<String, String>::new();
        let mut placements = Vec::new();
        for (placement_name, placement) in &self.processors.0 {
            let definition_name = if let Some(name) = loaded_definitions.get(&placement.definition)
            {
                name.clone()
            } else {
                let path = artifact_dir.join(&placement.definition);
                let processor_yaml = ProcessorYaml::from_file(&path)?;
                let definition = processor_yaml.build_definition(&path)?;
                let name = definition.name.clone();
                builder = builder.processor_definition(definition);
                loaded_definitions.insert(placement.definition.clone(), name.clone());
                name
            };
            placements.push((placement_name, definition_name, placement.build()));
        }
        for (placement_name, definition_name, connection) in placements {
            builder = builder.connect(placement_name, definition_name, connection);
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
    ) -> Result<NetworkTopology, ArchLoadError> {
        let network_dimensions = self
            .dimensions
            .iter()
            .map(|name| {
                dimensions
                    .get(name)
                    .copied()
                    .map(|size| super::Axis::new(name, size))
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
        let mut network = NetworkTopology::new(&self.name, network_dimensions);
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
                MemoryEndpoint::parse(&interface.endpoint).map_err(|error| {
                    ArchLoadError::Invalid(format!(
                        "network '{}' interface '{}': {error}",
                        self.name, interface.name
                    ))
                })?,
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
    ) -> Result<(Vec<MemoryDefinition>, Vec<MemoryAlias>), ArchLoadError> {
        let MemoryCatalogYaml { memories, regions } = self;
        let mut technology_kinds = BTreeMap::<String, u64>::new();
        for (_, memory) in &memories.0 {
            if let Some(technology) = &memory.technology {
                let next_kind = technology_kinds.len() as u64;
                technology_kinds
                    .entry(technology.clone())
                    .or_insert(next_kind);
            }
        }
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
                let technology = memory
                    .technology
                    .map(|technology| {
                        technology_kinds
                            .get(&technology)
                            .copied()
                            .map(|kind| MemoryTechnology::new(&technology, kind))
                            .ok_or_else(|| {
                                ArchLoadError::Invalid(format!(
                                    "memory '{name}' references unknown technology '{technology}'"
                                ))
                            })
                    })
                    .transpose()?;
                Ok(MemoryDefinition {
                    name,
                    indices: memory.indices,
                    capacity,
                    word_size,
                    technology,
                    banking,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        definitions.sort_by(|left, right| left.name.cmp(&right.name));
        let regions = regions
            .into_iter()
            .map(|(name, endpoint)| MemoryAlias::new(name, endpoint))
            .collect();
        Ok((definitions, regions))
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
        let source_path = base_dir.join(&self.source);
        let source = read_text(&source_path)?;
        let source_format = match source_path
            .extension()
            .and_then(|extension| extension.to_str())
        {
            Some("loom") => ProcessorSourceFormat::CompactLoom,
            Some("mlir") => ProcessorSourceFormat::Mlir,
            _ => {
                return Err(ArchLoadError::Invalid(format!(
                    "{} must have a .loom or .mlir extension",
                    source_path.display()
                )));
            }
        };
        let module = match source_format {
            ProcessorSourceFormat::CompactLoom => parse_loom_source(&source).map_err(|error| {
                ArchLoadError::Invalid(format!("{}: {error}", source_path.display()))
            })?,
            ProcessorSourceFormat::Mlir => crate::mlir::MlirModule::from_mlir_source(&source)
                .map_err(|error| {
                    ArchLoadError::Invalid(format!("{}: {error}", source_path.display()))
                })?,
        };
        let spec = self.performance.as_ref().ok_or_else(|| {
            ArchLoadError::Invalid(format!(
                "{} must define `performance`",
                processor_yaml_path.display()
            ))
        })?;
        let source_names = module
            .functions
            .iter()
            .map(|function| function.name.as_str())
            .collect::<std::collections::BTreeSet<_>>();
        let perf_names = spec
            .function_names()
            .collect::<std::collections::BTreeSet<_>>();
        if source_names != perf_names {
            return Err(ArchLoadError::Invalid(format!(
                "{} function names do not match compact Loom source: source={source_names:?}, performance={perf_names:?}",
                processor_yaml_path.display()
            )));
        }
        let functions = module
            .functions
            .into_iter()
            .map(|mut function| {
                let perf = spec
                    .model_for_func(&function)
                    .map_err(|error| ArchLoadError::Invalid(error.to_string()))?;
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
            source_format,
            functions,
            resources: self.resources.iter().map(ResourceYaml::build).collect(),
        })
    }
}

impl ProcessorPlacementYaml {
    fn build(&self) -> Connection {
        Connection::new(
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
    use super::{MemoryCatalogYaml, ProcessorPlacementYaml};
    use crate::arch::{EndpointIndex, MemoryEndpoint, MemoryTechnology};
    use crate::{Expr, Sym};

    #[test]
    fn connection_values_deserialize_as_endpoints() {
        let connection: ProcessorPlacementYaml = serde_yaml::from_str(
            r#"
definition: lane.yaml
domain: [x, y]
inputs: ["L1[x, y]"]
outputs: ["L2[x floordiv 2, y floordiv 2]"]
"#,
        )
        .expect("connection should deserialize");

        assert_eq!(
            connection.inputs,
            [MemoryEndpoint::parse("L1[x, y]").unwrap()]
        );
        assert_eq!(
            connection.outputs,
            [MemoryEndpoint::parse("L2[x floordiv 2, y floordiv 2]").unwrap()]
        );
    }

    #[test]
    fn memory_region_values_deserialize_as_endpoints() {
        let catalog: MemoryCatalogYaml = serde_yaml::from_str(
            r#"
regions:
  all_l1: "L1[:, :]"
"#,
        )
        .expect("memory catalog should deserialize");

        assert_eq!(
            catalog.regions["all_l1"],
            MemoryEndpoint {
                memory: "L1".into(),
                indices: vec![EndpointIndex::All, EndpointIndex::All],
                bank: None,
            }
        );
    }

    #[test]
    fn invalid_memory_region_endpoint_fails_deserialization() {
        let error = serde_yaml::from_str::<MemoryCatalogYaml>(
            r#"
regions:
  all_l1: "L1[:,"
"#,
        )
        .expect_err("invalid endpoint must fail while deserializing");

        assert!(error.to_string().contains("invalid memory region 'all_l1'"));
    }

    #[test]
    fn memory_geometry_accepts_architecture_parameters() {
        let catalog: MemoryCatalogYaml = serde_yaml::from_str(
            r#"
memories:
  L1:
    indices: [x]
    technology: custom_local
    capacity: "X * 256"
    word_size: 16
    banking: X
"#,
        )
        .expect("symbolic memory geometry syntax");
        let declared = ["X".to_string()].into_iter().collect();
        let (definitions, _) = catalog
            .build(&declared, &[(Sym::new("X"), Expr::Const(4))])
            .expect("symbolic memory geometry should instantiate");
        let l1 = definitions
            .iter()
            .find(|definition| definition.name == "L1")
            .unwrap();
        assert_eq!(l1.capacity, 1024);
        assert_eq!(
            l1.technology,
            Some(MemoryTechnology::new("custom_local", 0))
        );
        assert_eq!(l1.banking.as_ref().unwrap().banks, 4);
    }

    #[test]
    fn memory_technology_kinds_follow_first_catalog_appearance() {
        let catalog: MemoryCatalogYaml = serde_yaml::from_str(
            r#"
memories:
  cache_a:
    technology: gcram
    capacity: 1024
    word_size: 16
  weights:
    technology: rram
    capacity: 2048
    word_size: 16
  cache_b:
    technology: gcram
    capacity: 4096
    word_size: 16
"#,
        )
        .expect("ordered memory catalog");
        let (definitions, _) = catalog
            .build(&Default::default(), &[])
            .expect("memory catalog should build");

        let technology = |name: &str| {
            definitions
                .iter()
                .find(|definition| definition.name == name)
                .unwrap()
                .technology
                .clone()
                .unwrap()
        };
        assert_eq!(technology("cache_a"), MemoryTechnology::new("gcram", 0));
        assert_eq!(technology("weights"), MemoryTechnology::new("rram", 1));
        assert_eq!(technology("cache_b"), MemoryTechnology::new("gcram", 0));
    }
}
