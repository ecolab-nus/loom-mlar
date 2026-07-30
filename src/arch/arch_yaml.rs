use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Deserializer, de::Error as _};

use super::PerfYamlSpec;
use super::architecture::{Architecture, ArchitectureBuilder};
use super::memory::{Banking, MemoryCatalog, MemoryDefinition, MemoryEndpoint, NamedMemoryRegion};
use super::processor::{ConnectionSpec, FunctionProcessor, ProcessorDefinition, ProcessorType};
use super::resource::ResourceArray;
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
    dimensions: BTreeMap<String, u64>,
    #[serde(default)]
    memories: BTreeMap<String, Vec<String>>,
    #[serde(default)]
    processor: ProcessorEntriesYaml,
    #[serde(default)]
    resources: Vec<ResourceYaml>,
}

fn default_memory_catalog_path() -> String {
    "memory.yaml".into()
}

#[derive(Clone, Debug)]
struct ProcessorEntryYaml {
    processor: String,
    connections: Vec<ConnectionYaml>,
}

#[derive(Clone, Debug, Default)]
struct ProcessorEntriesYaml(Vec<ProcessorEntryYaml>);

impl<'de> Deserialize<'de> for ProcessorEntriesYaml {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let mapping = serde_yaml::Mapping::deserialize(deserializer)?;
        let mut entries = Vec::with_capacity(mapping.len());
        for (processor, connections) in mapping {
            let processor = processor
                .as_str()
                .ok_or_else(|| D::Error::custom("processor keys must be YAML filenames"))?
                .to_string();
            let connections =
                Vec::<ConnectionYaml>::deserialize(connections).map_err(D::Error::custom)?;
            entries.push(ProcessorEntryYaml {
                processor,
                connections,
            });
        }
        Ok(Self(entries))
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConnectionYaml {
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
struct MemoryCatalogYaml {
    #[serde(default)]
    memories: BTreeMap<String, MemoryDefinitionYaml>,
    #[serde(default, deserialize_with = "deserialize_memory_regions")]
    regions: BTreeMap<String, MemoryEndpoint>,
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
    capacity: u64,
    word_size: u64,
    #[serde(default)]
    banking: Option<BankingYaml>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
enum BankingYaml {
    Count(u64),
    Detailed { banks: u64 },
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
    performance: Option<serde_yaml::Value>,
    #[serde(default)]
    functions: Option<serde_yaml::Value>,
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
        let artifact_dir = artifact_dir.as_ref();
        let catalog_path = artifact_dir.join(&self.memory);
        let catalog_yaml: MemoryCatalogYaml = read_yaml(&catalog_path)?;
        let catalog = catalog_yaml.build()?;

        let mut builder = ArchitectureBuilder::new(&self.name).memory_catalog(catalog);
        for (name, size) in &self.dimensions {
            builder = builder.dimension(name, *size);
        }
        for (name, dimensions) in &self.memories {
            builder = builder.place_memory(name, dimensions.iter().cloned());
        }
        for resource in &self.resources {
            builder = builder.resource(resource.build());
        }

        let mut processor_entries = Vec::new();
        for entry in &self.processor.0 {
            let path = artifact_dir.join(&entry.processor);
            let processor_yaml = ProcessorYaml::from_file(&path)?;
            let definition = processor_yaml.build_definition(&path)?;
            let definition_name = definition.name.clone();
            builder = builder.processor_definition(definition);
            processor_entries.push((definition_name, &entry.connections));
        }
        for (definition, connections) in processor_entries {
            for connection in connections {
                builder = builder.connect(definition.clone(), connection.build());
            }
        }
        builder
            .build()
            .map_err(|error| ArchLoadError::Invalid(error.to_string()))
    }
}

impl MemoryCatalogYaml {
    fn build(self) -> Result<MemoryCatalog, ArchLoadError> {
        let definitions = self
            .memories
            .into_iter()
            .map(|(name, memory)| {
                let mut definition =
                    MemoryDefinition::new(name, memory.indices, memory.capacity, memory.word_size);
                definition.banking = memory.banking.map(|banking| Banking {
                    banks: match banking {
                        BankingYaml::Count(banks) | BankingYaml::Detailed { banks } => banks,
                    },
                });
                definition
            })
            .collect();
        let regions = self
            .regions
            .into_iter()
            .map(|(name, endpoint)| NamedMemoryRegion::new(name, endpoint))
            .collect();
        let catalog = MemoryCatalog {
            definitions,
            regions,
        };
        catalog.validate().map_err(ArchLoadError::Invalid)?;
        Ok(catalog)
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
        let module = parse_loom_source(&source).map_err(|error| {
            ArchLoadError::Invalid(format!("{}: {error}", source_path.display()))
        })?;
        let performance = self.performance.clone().or_else(|| {
            self.functions.as_ref().map(|functions| {
                let mut map = serde_yaml::Mapping::new();
                map.insert(
                    serde_yaml::Value::String("functions".into()),
                    functions.clone(),
                );
                serde_yaml::Value::Mapping(map)
            })
        });
        let performance = performance.ok_or_else(|| {
            ArchLoadError::Invalid(format!(
                "{} must define `performance`",
                processor_yaml_path.display()
            ))
        })?;
        let performance = normalize_performance_yaml(performance);
        let performance_text =
            serde_yaml::to_string(&performance).map_err(|source| ArchLoadError::Yaml {
                path: processor_yaml_path.to_path_buf(),
                source,
            })?;
        let spec = PerfYamlSpec::from_yaml_str(&performance_text)
            .map_err(|error| ArchLoadError::Invalid(error.to_string()))?;
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
            .map(|function| {
                let perf = spec
                    .model_for_func(&function)
                    .map_err(|error| ArchLoadError::Invalid(error.to_string()))?;
                Ok(FunctionProcessor::new(function, perf))
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
        })
    }
}

fn normalize_performance_yaml(value: serde_yaml::Value) -> serde_yaml::Value {
    let has_functions = value
        .as_mapping()
        .is_some_and(|mapping| mapping.contains_key(serde_yaml::Value::String("functions".into())));
    if has_functions {
        value
    } else {
        let mut mapping = serde_yaml::Mapping::new();
        mapping.insert(serde_yaml::Value::String("functions".into()), value);
        serde_yaml::Value::Mapping(mapping)
    }
}

impl ConnectionYaml {
    fn build(&self) -> ConnectionSpec {
        ConnectionSpec::new(self.inputs.clone(), self.outputs.clone())
            .with_resources(self.resources.iter().cloned())
    }
}

impl ResourceYaml {
    fn build(&self) -> ResourceArray {
        match self.capacity {
            Some(capacity) => ResourceArray::quantitative(&self.name, capacity),
            None => ResourceArray::exclusive(&self.name),
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
    use super::{ConnectionYaml, MemoryCatalogYaml};
    use crate::arch::{EndpointIndex, MemoryEndpoint};

    #[test]
    fn connection_values_deserialize_as_endpoints() {
        let connection: ConnectionYaml = serde_yaml::from_str(
            r#"
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
}
