use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use super::{
    Architecture, ComputeProcessor, DataMover, Dimension, MemoryRegion, MeshLink, MeshNetwork,
    MeshNetworkInterface, PerfYamlSpec, Resource, ScaleOutNetwork, SizeExpr,
};
use crate::math::{AffineMapTemplate, Expr};
use crate::schedule::MlirModule;

#[derive(Debug)]
pub enum ArchLoadError {
    Io { path: PathBuf, source: std::io::Error },
    Yaml(serde_yaml::Error),
    Invalid(String),
}

impl fmt::Display for ArchLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(f, "failed to read '{}': {source}", path.display())
            }
            Self::Yaml(source) => write!(f, "failed to parse system YAML: {source}"),
            Self::Invalid(message) => write!(f, "invalid system YAML: {message}"),
        }
    }
}

impl std::error::Error for ArchLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Yaml(source) => Some(source),
            Self::Invalid(_) => None,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SystemYaml {
    pub version: u32,
    pub dimensions: BTreeMap<String, i64>,
    pub architecture: ScopeYaml,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScopeYaml {
    pub name: String,
    #[serde(default)]
    pub scale: Vec<String>,
    #[serde(default)]
    pub memories: Vec<MemoryYaml>,
    #[serde(default)]
    pub processors: Vec<ProcessorYaml>,
    #[serde(default)]
    pub networks: Vec<MeshNetworkYaml>,
    #[serde(default)]
    pub children: Vec<ScopeYaml>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryYaml {
    pub name: String,
    #[serde(default)]
    pub bank_name: Option<String>,
    pub block_size_bytes: i64,
    pub num_blocks: i64,
    #[serde(default)]
    pub scale: Vec<String>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProcessorKindYaml {
    Compute,
    DataMover,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProcessorYaml {
    pub name: String,
    pub kind: ProcessorKindYaml,
    pub from: String,
    pub to: String,
    #[serde(default)]
    pub resources: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshNetworkYaml {
    pub name: String,
    pub dimensions: Vec<String>,
    pub region: String,
    pub links: Vec<MeshLinkYaml>,
    pub link_bandwidth: String,
    pub io: MeshIoYaml,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshLinkYaml {
    pub name: String,
    pub map: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeshIoYaml {
    pub map: String,
    pub link_bandwidth: String,
}

impl SystemYaml {
    pub fn from_yaml_str(input: &str) -> Result<Self, ArchLoadError> {
        let spec: Self = serde_yaml::from_str(input).map_err(ArchLoadError::Yaml)?;
        spec.validate()?;
        Ok(spec)
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, ArchLoadError> {
        let path = path.as_ref();
        let input = std::fs::read_to_string(path).map_err(|source| ArchLoadError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Self::from_yaml_str(&input)
    }

    pub fn build(&self, artifact_dir: impl AsRef<Path>) -> Result<Architecture, ArchLoadError> {
        let dims = self
            .dimensions
            .iter()
            .map(|(name, size)| (name.clone(), Dimension::new_int(name, *size)))
            .collect::<BTreeMap<_, _>>();
        let mut function_names = HashSet::new();
        build_scope(
            &self.architecture,
            artifact_dir.as_ref(),
            &dims,
            &mut function_names,
        )
    }

    fn validate(&self) -> Result<(), ArchLoadError> {
        if self.version != 1 {
            return Err(invalid(format!(
                "unsupported version {}; expected 1",
                self.version
            )));
        }
        if self.dimensions.is_empty() {
            return Err(invalid("dimensions must not be empty"));
        }
        for (name, size) in &self.dimensions {
            if name.is_empty() || *size <= 0 {
                return Err(invalid(format!(
                    "dimension '{name}' must have a positive size"
                )));
            }
        }

        let mut scope_names = HashSet::new();
        let mut memory_names = HashSet::new();
        let mut processor_names = HashSet::new();
        validate_scope(
            &self.architecture,
            &self.dimensions,
            &mut scope_names,
            &mut memory_names,
            &mut processor_names,
        )
    }
}

fn validate_scope(
    scope: &ScopeYaml,
    dimensions: &BTreeMap<String, i64>,
    scope_names: &mut HashSet<String>,
    memory_names: &mut HashSet<String>,
    processor_names: &mut HashSet<String>,
) -> Result<(), ArchLoadError> {
    require_unique(scope_names, "scope", &scope.name)?;
    validate_dimensions(&scope.scale, dimensions, &format!("scope '{}'", scope.name))?;

    for memory in &scope.memories {
        require_unique(memory_names, "memory", &memory.name)?;
        if memory.block_size_bytes <= 0 || memory.num_blocks <= 0 {
            return Err(invalid(format!(
                "memory '{}' sizes must be positive",
                memory.name
            )));
        }
        validate_dimensions(
            &memory.scale,
            dimensions,
            &format!("memory '{}'", memory.name),
        )?;
    }

    for processor in &scope.processors {
        require_unique(processor_names, "processor", &processor.name)?;
        if processor.from.is_empty() || processor.to.is_empty() {
            return Err(invalid(format!(
                "processor '{}' must name both route endpoints",
                processor.name
            )));
        }
    }

    for network in &scope.networks {
        if network.links.is_empty() {
            return Err(invalid(format!(
                "network '{}' must contain at least one link",
                network.name
            )));
        }
        validate_dimensions(
            &network.dimensions,
            dimensions,
            &format!("network '{}'", network.name),
        )?;
    }

    for child in &scope.children {
        validate_scope(
            child,
            dimensions,
            scope_names,
            memory_names,
            processor_names,
        )?;
    }
    Ok(())
}

fn validate_dimensions(
    names: &[String],
    dimensions: &BTreeMap<String, i64>,
    owner: &str,
) -> Result<(), ArchLoadError> {
    let mut seen = HashSet::new();
    for name in names {
        if !dimensions.contains_key(name) {
            return Err(invalid(format!("{owner} uses unknown dimension '{name}'")));
        }
        if !seen.insert(name) {
            return Err(invalid(format!("{owner} repeats dimension '{name}'")));
        }
    }
    Ok(())
}

fn require_unique(
    names: &mut HashSet<String>,
    kind: &str,
    name: &str,
) -> Result<(), ArchLoadError> {
    if name.is_empty() {
        return Err(invalid(format!("{kind} name must not be empty")));
    }
    if !names.insert(name.to_string()) {
        return Err(invalid(format!("duplicate {kind} name '{name}'")));
    }
    Ok(())
}

fn build_scope(
    spec: &ScopeYaml,
    artifact_dir: &Path,
    dims: &BTreeMap<String, Dimension>,
    function_names: &mut HashSet<String>,
) -> Result<Architecture, ArchLoadError> {
    let scope_dims = resolve_dims(&spec.scale, dims)?;
    let mut arch = Architecture::scope(&spec.name);

    for memory in &spec.memories {
        let memory_dims = resolve_dims(&memory.scale, dims)?;
        let mut region = MemoryRegion::bank(
            SizeExpr::Const(memory.block_size_bytes),
            SizeExpr::Const(memory.num_blocks),
        );
        if let Some(bank_name) = &memory.bank_name {
            region = region.with_name(bank_name);
        }
        if !memory_dims.is_empty() {
            region = region.scale(&memory_dims);
        }
        arch.add_memory(region.with_name(&memory.name));
    }

    for child in &spec.children {
        arch.add_child(build_scope(child, artifact_dir, dims, function_names)?);
    }

    for processor in &spec.processors {
        let source = arch
            .get_scaled_memory_region(&processor.from)
            .ok_or_else(|| {
                invalid(format!(
                    "processor '{}' cannot resolve source memory '{}' from scope '{}'",
                    processor.name, processor.from, spec.name
                ))
            })?;
        let destination = arch
            .get_scaled_memory_region(&processor.to)
            .ok_or_else(|| {
                invalid(format!(
                    "processor '{}' cannot resolve destination memory '{}' from scope '{}'",
                    processor.name, processor.to, spec.name
                ))
            })?;
        let (functionality, perf) =
            load_functionality_and_perf(artifact_dir, &processor.name, function_names)?;
        let resources = processor
            .resources
            .iter()
            .map(Resource::exclusive)
            .collect::<Vec<_>>();

        let built = match processor.kind {
            ProcessorKindYaml::Compute => ComputeProcessor::builder()
                .named(&processor.name)
                .from_region(source)
                .to_region(destination)
                .with_resources(resources)
                .functionality(functionality)
                .perf(perf)
                .finish()
                .map(|processor| processor.into_processor()),
            ProcessorKindYaml::DataMover => DataMover::builder()
                .named(&processor.name)
                .from_region(source)
                .to_region(destination)
                .with_resources(resources)
                .functionality(functionality)
                .perf(perf)
                .finish()
                .map(|processor| processor.into_processor()),
        }
        .map_err(|message| {
            invalid(format!(
                "failed to build processor '{}': {message}",
                processor.name
            ))
        })?;
        arch.add_processor(built);
    }

    arch = arch.with_dims(&scope_dims);

    for network in &spec.networks {
        let network_dims = resolve_dims(&network.dimensions, dims)?;
        let region = arch
            .get_scaled_memory_region(&network.region)
            .ok_or_else(|| {
                invalid(format!(
                    "network '{}' cannot resolve memory '{}' from scope '{}'",
                    network.name, network.region, spec.name
                ))
            })?;
        if region.dims() != network_dims {
            return Err(invalid(format!(
                "network '{}' dimensions do not match the outer dimensions of region '{}'",
                network.name, network.region
            )));
        }

        let links = network
            .links
            .iter()
            .map(|link| {
                parse_affine_map(&link.map, &network_dims)
                    .map(|map| MeshLink::named(&link.name, map))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let io_map = parse_affine_map(&network.io.map, &network_dims)?;
        let link_bandwidth = Expr::parse(&network.link_bandwidth).map_err(|source| {
            invalid(format!(
                "network '{}' has invalid link bandwidth: {source}",
                network.name
            ))
        })?;
        let io_bandwidth = Expr::parse(&network.io.link_bandwidth).map_err(|source| {
            invalid(format!(
                "network '{}' has invalid IO bandwidth: {source}",
                network.name
            ))
        })?;
        let mesh = MeshNetwork {
            name: network.name.clone(),
            dimensions: network_dims,
            links,
            region,
            io: MeshNetworkInterface::new(io_map, io_bandwidth),
            link_bandwidth,
        };
        let network = ScaleOutNetwork::Mesh(mesh);
        network.validate_map_domains().map_err(|message| {
            invalid(format!("network '{}' is invalid: {message}", network.name()))
        })?;
        arch.add_network(network);
    }

    Ok(arch)
}

fn resolve_dims(
    names: &[String],
    dimensions: &BTreeMap<String, Dimension>,
) -> Result<Vec<Dimension>, ArchLoadError> {
    names
        .iter()
        .map(|name| {
            dimensions
                .get(name)
                .cloned()
                .ok_or_else(|| invalid(format!("unknown dimension '{name}'")))
        })
        .collect()
}

fn parse_affine_map(
    source: &str,
    dimensions: &[Dimension],
) -> Result<crate::math::AffineMap, ArchLoadError> {
    AffineMapTemplate::parse(source)
        .map_err(|error| invalid(format!("invalid affine map '{source}': {error}")))?
        .bind(dimensions.iter().cloned())
        .map_err(|error| invalid(format!("cannot bind affine map '{source}': {error}")))
}

fn load_functionality_and_perf(
    dir: &Path,
    processor_name: &str,
    global_names: &mut HashSet<String>,
) -> Result<(MlirModule, Vec<super::FuncPerfModel>), ArchLoadError> {
    let mlir_path = dir.join(format!("{processor_name}.mlir"));
    let perf_path = dir.join(format!("{processor_name}.perf.yaml"));
    let functionality = MlirModule::from_mlir(mlir_path.to_string_lossy().into_owned())
        .map_err(|message| invalid(message))?;
    let perf_spec = PerfYamlSpec::from_file(&perf_path)
        .map_err(|error| invalid(format!("{}: {error}", perf_path.display())))?;

    let mlir_names = functionality
        .functions
        .iter()
        .map(|function| function.name.clone())
        .collect::<HashSet<_>>();
    let perf_names = perf_spec.functions.keys().cloned().collect::<HashSet<_>>();
    if mlir_names != perf_names {
        let mut only_mlir = mlir_names.difference(&perf_names).cloned().collect::<Vec<_>>();
        let mut only_perf = perf_names.difference(&mlir_names).cloned().collect::<Vec<_>>();
        only_mlir.sort();
        only_perf.sort();
        return Err(invalid(format!(
            "processor '{processor_name}' MLIR/perf function sets differ; only in MLIR: {only_mlir:?}, only in perf: {only_perf:?}"
        )));
    }
    for name in &mlir_names {
        if !global_names.insert(name.clone()) {
            return Err(invalid(format!(
                "function name '{name}' is not globally unique"
            )));
        }
    }

    let perf = perf_spec
        .models_for_module(&functionality)
        .map_err(|error| invalid(format!("{}: {error}", perf_path.display())))?;
    Ok((functionality, perf))
}

fn invalid(message: impl Into<String>) -> ArchLoadError {
    ArchLoadError::Invalid(message.into())
}

#[cfg(test)]
mod tests {
    use super::SystemYaml;

    #[test]
    fn rejects_unknown_fields() {
        let error = SystemYaml::from_yaml_str(
            r#"
version: 1
dimensions: {x: 2}
architecture:
  name: system
  mystery: true
"#,
        )
        .expect_err("unknown fields must fail");
        assert!(error.to_string().contains("unknown field"));
    }

    #[test]
    fn rejects_duplicate_memory_names() {
        let error = SystemYaml::from_yaml_str(
            r#"
version: 1
dimensions: {x: 2}
architecture:
  name: system
  memories:
    - {name: L1, block_size_bytes: 16, num_blocks: 4}
  children:
    - name: core
      memories:
        - {name: L1, block_size_bytes: 16, num_blocks: 4}
"#,
        )
        .expect_err("ambiguous memory names must fail");
        assert!(error.to_string().contains("duplicate memory name 'L1'"));
    }

    #[test]
    fn lowers_affine_mesh_networks() {
        let spec = SystemYaml::from_yaml_str(
            r#"
version: 1
dimensions: {x: 4, y: 2}
architecture:
  name: mesh
  scale: [x, y]
  memories:
    - {name: L1, block_size_bytes: 16, num_blocks: 4}
  networks:
    - name: torus
      dimensions: [x, y]
      region: L1
      links:
        - {name: x, map: "[x, y] -> [x, y]: ((x + 1) mod 4, y)"}
        - {name: y, map: "[x, y] -> [x, y]: (x, (y + 1) mod 2)"}
      link_bandwidth: "64"
      io:
        map: "[x, y] -> [x, y]: (x, y)"
        link_bandwidth: "32"
"#,
        )
        .expect("schema should parse");
        let architecture = spec.build(".").expect("network should lower");

        assert_eq!(architecture.networks.len(), 1);
        assert_eq!(architecture.networks[0].mesh_links().len(), 2);
        assert_eq!(architecture.networks[0].dimensions(), architecture.dims());
    }

    #[test]
    fn generated_schema_matches_the_legacy_mesh_structure() {
        let spec = SystemYaml::from_yaml_str(
            r#"
version: 1
dimensions: {nbank: 16, x: 8, y: 8, dram_channel: 8}
architecture:
  name: system
  memories:
    - name: DRAM
      bank_name: DRAM_bank
      block_size_bytes: 8192
      num_blocks: 196608
      scale: [dram_channel]
  processors:
    - {name: dram_l1_noc0, kind: data_mover, from: DRAM, to: L1, resources: [noc0]}
    - {name: l1_l1_noc0, kind: data_mover, from: L1, to: L1, resources: [noc0]}
    - {name: l1_dram_noc1, kind: data_mover, from: L1, to: DRAM, resources: [noc1]}
  children:
    - name: mesh
      scale: [x, y]
      memories:
        - name: L1
          block_size_bytes: 16
          num_blocks: 5464
          scale: [nbank]
      processors:
        - {name: matrix_lane, kind: compute, from: L1, to: L1}
        - {name: vector_lane, kind: compute, from: L1, to: L1}
"#,
        )
        .expect("schema should parse");
        let generated = spec
            .build("tests/2d_mesh/processors")
            .expect("schema should lower");
        let legacy = crate::archs::scaled_mesh_torus("tests/2d_mesh/processors");

        assert_eq!(generated.name, legacy.name);
        assert_eq!(generated.children.len(), legacy.children.len());
        assert_eq!(generated.children[0].name, legacy.children[0].name);
        assert_eq!(generated.children[0].dims, legacy.children[0].dims);

        let memory_summary = |architecture: &crate::arch::Architecture| {
            let mut summary = architecture
                .memories_recursive()
                .into_iter()
                .map(|memory| {
                    (
                        memory.name().expect("memory should be named").to_string(),
                        memory.total_size_bytes(),
                    )
                })
                .collect::<Vec<_>>();
            summary.sort();
            summary
        };
        assert_eq!(memory_summary(&generated), memory_summary(&legacy));

        for generated_processor in generated.processors_recursive() {
            let name = generated_processor
                .name
                .as_deref()
                .expect("processor should be named");
            let legacy_processor = legacy
                .get_processor(name)
                .expect("legacy processor should exist");
            assert_eq!(generated_processor.source, legacy_processor.source);
            assert_eq!(
                generated_processor.destination,
                legacy_processor.destination
            );
            assert_eq!(
                generated_processor.functions.len(),
                legacy_processor.functions.len()
            );
            for (generated_function, legacy_function) in generated_processor
                .functions
                .iter()
                .zip(&legacy_processor.functions)
            {
                assert_eq!(generated_function.func.name, legacy_function.func.name);
                assert_eq!(
                    generated_function.func.symbols,
                    legacy_function.func.symbols
                );
                assert_eq!(
                    serde_json::to_value(&generated_function.perf)
                        .expect("perf should serialize"),
                    serde_json::to_value(&legacy_function.perf)
                        .expect("perf should serialize")
                );
            }
            let resources = |processor: &crate::arch::Processor| {
                let mut resources = processor
                    .resources
                    .iter()
                    .map(|resource| {
                        (resource.id().as_str().to_string(), resource.capacity())
                    })
                    .collect::<Vec<_>>();
                resources.sort();
                resources
            };
            assert_eq!(
                resources(generated_processor),
                resources(legacy_processor)
            );
        }
    }
}
