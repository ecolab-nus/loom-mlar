use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::path::{Path, PathBuf};

use serde::de::Error as _;
use serde::{Deserialize, Deserializer};

use super::{
    Architecture, ComputeProcessor, DataMover, Dimension, MemoryRegion, MeshLink, MeshNetwork,
    MeshNetworkInterface, PerfYamlSpec, Resource, ScaleOutNetwork, SizeExpr,
};
use crate::math::{AffineMapTemplate, Expr};
use crate::schedule::MlirModule;

#[derive(Debug)]
pub enum ArchLoadError {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Yaml(serde_yaml::Error),
    Invalid(String),
}

impl fmt::Display for ArchLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(f, "failed to read '{}': {source}", path.display())
            }
            Self::Yaml(source) => write!(f, "failed to parse chip YAML: {source}"),
            Self::Invalid(message) => write!(f, "invalid chip YAML: {message}"),
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
pub struct ChipYaml {
    #[serde(default)]
    dimensions: BTreeMap<String, i64>,
    architecture: ArchitectureYaml,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ArchitectureYaml {
    name: String,
    #[serde(default)]
    groups: Vec<GroupYaml>,
    #[serde(default)]
    memories: Vec<MemoryYaml>,
    #[serde(default)]
    processors: Vec<ProcessorYaml>,
    #[serde(default)]
    networks: Vec<MeshNetworkYaml>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GroupYaml {
    name: String,
    #[serde(default, rename = "in")]
    parent: Option<String>,
    #[serde(default)]
    scale: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AggregateShorthandYaml {
    name: String,
}

#[derive(Clone, Debug)]
enum MemoryYaml {
    Physical(PhysicalMemoryYaml),
    Aggregate(AggregateMemoryYaml),
}

#[derive(Clone, Debug)]
struct PhysicalMemoryYaml {
    name: String,
    placement: Option<String>,
    bank_name: Option<String>,
    block_size_bytes: i64,
    num_blocks: i64,
    scale: Vec<String>,
    aggregate: Option<AggregateShorthandYaml>,
}

#[derive(Clone, Debug)]
struct AggregateMemoryYaml {
    name: String,
    base: String,
    across: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawMemoryYaml {
    name: String,
    #[serde(default, rename = "in")]
    placement: Option<String>,
    #[serde(default)]
    bank_name: Option<String>,
    #[serde(default)]
    block_size_bytes: Option<i64>,
    #[serde(default)]
    num_blocks: Option<i64>,
    #[serde(default)]
    scale: Option<Vec<String>>,
    #[serde(default)]
    aggregate: Option<AggregateShorthandYaml>,
    #[serde(default)]
    of: Option<String>,
    #[serde(default)]
    across: Option<String>,
}

impl<'de> Deserialize<'de> for MemoryYaml {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawMemoryYaml::deserialize(deserializer)?;
        if raw.of.is_some() || raw.across.is_some() {
            let (Some(base), Some(across)) = (raw.of, raw.across) else {
                return Err(D::Error::custom(format!(
                    "aggregate memory '{}' must specify both 'of' and 'across'",
                    raw.name
                )));
            };
            if raw.placement.is_some()
                || raw.bank_name.is_some()
                || raw.block_size_bytes.is_some()
                || raw.num_blocks.is_some()
                || raw.scale.is_some()
                || raw.aggregate.is_some()
            {
                return Err(D::Error::custom(format!(
                    "aggregate memory '{}' cannot define placement, storage, scale, or another aggregate",
                    raw.name
                )));
            }
            return Ok(Self::Aggregate(AggregateMemoryYaml {
                name: raw.name,
                base,
                across,
            }));
        }

        let (Some(block_size_bytes), Some(num_blocks)) = (raw.block_size_bytes, raw.num_blocks)
        else {
            return Err(D::Error::custom(format!(
                "physical memory '{}' must specify block_size_bytes and num_blocks",
                raw.name
            )));
        };
        Ok(Self::Physical(PhysicalMemoryYaml {
            name: raw.name,
            placement: raw.placement,
            bank_name: raw.bank_name,
            block_size_bytes,
            num_blocks,
            scale: raw.scale.unwrap_or_default(),
            aggregate: raw.aggregate,
        }))
    }
}

impl MemoryYaml {
    fn name(&self) -> &str {
        match self {
            Self::Physical(memory) => &memory.name,
            Self::Aggregate(memory) => &memory.name,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum ProcessorKindYaml {
    Compute,
    DataMover,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProcessorYaml {
    name: String,
    #[serde(default, rename = "in")]
    placement: Option<String>,
    kind: ProcessorKindYaml,
    from: String,
    to: String,
    #[serde(default)]
    resources: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MeshNetworkYaml {
    name: String,
    #[serde(default, rename = "in")]
    placement: Option<String>,
    dimensions: Vec<String>,
    region: String,
    links: Vec<MeshLinkYaml>,
    link_bandwidth: String,
    io: MeshIoYaml,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MeshLinkYaml {
    name: String,
    map: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct MeshIoYaml {
    map: String,
    link_bandwidth: String,
}

#[derive(Clone, Debug)]
struct AggregateMemory {
    name: String,
    base: String,
    across: String,
    target: String,
    visible_in: Option<String>,
}

#[derive(Clone, Debug)]
struct MemoryCatalog {
    physical_placements: BTreeMap<String, Option<String>>,
    aggregates: BTreeMap<String, AggregateMemory>,
}

impl ChipYaml {
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
        let catalog = self.memory_catalog()?;
        let mut function_names = HashSet::new();
        build_scope(
            &self.architecture,
            None,
            artifact_dir.as_ref(),
            &dims,
            &catalog,
            &mut function_names,
        )
    }

    fn validate(&self) -> Result<(), ArchLoadError> {
        for (name, size) in &self.dimensions {
            if name.is_empty() || *size <= 0 {
                return Err(invalid(format!(
                    "dimension '{name}' must have a positive size"
                )));
            }
        }

        if self.architecture.name.is_empty() {
            return Err(invalid("architecture name must not be empty"));
        }

        let mut group_names = HashSet::new();
        for group in &self.architecture.groups {
            require_unique(&mut group_names, "group", &group.name)?;
            validate_dimensions(
                &group.scale,
                &self.dimensions,
                &format!("group '{}'", group.name),
            )?;
        }
        for group in &self.architecture.groups {
            if let Some(parent) = &group.parent {
                if parent == &group.name {
                    return Err(invalid(format!(
                        "group '{}' cannot be inside itself",
                        group.name
                    )));
                }
                if !group_names.contains(parent) {
                    return Err(invalid(format!(
                        "group '{}' uses unknown parent group '{parent}'",
                        group.name
                    )));
                }
            }
            ensure_group_acyclic(&group.name, &self.architecture.groups)?;
        }

        let catalog = self.memory_catalog()?;
        let mut processor_names = HashSet::new();
        for processor in &self.architecture.processors {
            require_unique(&mut processor_names, "processor", &processor.name)?;
            validate_placement(
                processor.placement.as_deref(),
                &group_names,
                &format!("processor '{}'", processor.name),
            )?;
            if processor.from.is_empty() || processor.to.is_empty() {
                return Err(invalid(format!(
                    "processor '{}' must name both route endpoints",
                    processor.name
                )));
            }
            validate_memory_reference(
                &processor.from,
                processor.placement.as_deref(),
                &catalog,
                &format!("processor '{}'", processor.name),
            )?;
            validate_memory_reference(
                &processor.to,
                processor.placement.as_deref(),
                &catalog,
                &format!("processor '{}'", processor.name),
            )?;
        }

        let mut network_names = HashSet::new();
        for network in &self.architecture.networks {
            require_unique(&mut network_names, "network", &network.name)?;
            validate_placement(
                network.placement.as_deref(),
                &group_names,
                &format!("network '{}'", network.name),
            )?;
            if network.links.is_empty() {
                return Err(invalid(format!(
                    "network '{}' must contain at least one link",
                    network.name
                )));
            }
            validate_dimensions(
                &network.dimensions,
                &self.dimensions,
                &format!("network '{}'", network.name),
            )?;
            validate_memory_reference(
                &network.region,
                network.placement.as_deref(),
                &catalog,
                &format!("network '{}'", network.name),
            )?;
        }
        Ok(())
    }

    fn memory_catalog(&self) -> Result<MemoryCatalog, ArchLoadError> {
        build_memory_catalog(&self.architecture, &self.dimensions)
    }
}

fn ensure_group_acyclic(name: &str, groups: &[GroupYaml]) -> Result<(), ArchLoadError> {
    let mut seen = HashSet::new();
    let mut current = Some(name);
    while let Some(group_name) = current {
        if !seen.insert(group_name) {
            return Err(invalid(format!(
                "group hierarchy contains a cycle through '{group_name}'"
            )));
        }
        current = groups
            .iter()
            .find(|group| group.name == group_name)
            .and_then(|group| group.parent.as_deref());
    }
    Ok(())
}

fn validate_placement(
    placement: Option<&str>,
    group_names: &HashSet<String>,
    owner: &str,
) -> Result<(), ArchLoadError> {
    if let Some(group) = placement {
        if !group_names.contains(group) {
            return Err(invalid(format!("{owner} uses unknown group '{group}'")));
        }
    }
    Ok(())
}

fn build_memory_catalog(
    architecture: &ArchitectureYaml,
    dimensions: &BTreeMap<String, i64>,
) -> Result<MemoryCatalog, ArchLoadError> {
    let group_names = architecture
        .groups
        .iter()
        .map(|group| group.name.clone())
        .collect::<HashSet<_>>();
    let mut all_names = HashSet::new();
    let mut physical_placements = BTreeMap::new();

    for memory in &architecture.memories {
        require_unique(&mut all_names, "memory", memory.name())?;
        let MemoryYaml::Physical(memory) = memory else {
            continue;
        };

        validate_placement(
            memory.placement.as_deref(),
            &group_names,
            &format!("memory '{}'", memory.name),
        )?;
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
        physical_placements.insert(memory.name.clone(), memory.placement.clone());
    }

    let mut aggregates = BTreeMap::new();
    for memory in &architecture.memories {
        let MemoryYaml::Physical(memory) = memory else {
            continue;
        };
        if let Some(shorthand) = &memory.aggregate {
            require_unique(&mut all_names, "memory", &shorthand.name)?;
            let Some(across) = memory.placement.as_ref() else {
                return Err(invalid(format!(
                    "memory '{}' cannot define an aggregate without belonging to a group",
                    memory.name
                )));
            };
            aggregates.insert(
                shorthand.name.clone(),
                make_aggregate(
                    &shorthand.name,
                    &memory.name,
                    across,
                    architecture,
                    &physical_placements,
                )?,
            );
        }
    }
    for memory in &architecture.memories {
        let MemoryYaml::Aggregate(memory) = memory else {
            continue;
        };
        aggregates.insert(
            memory.name.clone(),
            make_aggregate(
                &memory.name,
                &memory.base,
                &memory.across,
                architecture,
                &physical_placements,
            )?,
        );
    }

    Ok(MemoryCatalog {
        physical_placements,
        aggregates,
    })
}

fn make_aggregate(
    name: &str,
    base: &str,
    across: &str,
    architecture: &ArchitectureYaml,
    physical_placements: &BTreeMap<String, Option<String>>,
) -> Result<AggregateMemory, ArchLoadError> {
    let Some(Some(base_group)) = physical_placements.get(base) else {
        return Err(invalid(format!(
            "aggregate memory '{name}' refers to unknown or non-group memory '{base}'"
        )));
    };
    if !is_group_ancestor(across, base_group, &architecture.groups) {
        return Err(invalid(format!(
            "aggregate memory '{name}' cannot aggregate '{base}' across unrelated group '{across}'"
        )));
    }
    let visible_in = architecture
        .groups
        .iter()
        .find(|group| group.name == across)
        .and_then(|group| group.parent.clone());
    let levels = group_distance(base_group, across, &architecture.groups)
        .ok_or_else(|| invalid(format!("cannot resolve aggregate memory '{name}'")))?;
    let mut target = base.to_string();
    for _ in 0..=levels {
        target = format!("array_{target}");
    }
    Ok(AggregateMemory {
        name: name.to_string(),
        base: base.to_string(),
        across: across.to_string(),
        target,
        visible_in,
    })
}

fn is_group_ancestor(ancestor: &str, group: &str, groups: &[GroupYaml]) -> bool {
    group_distance(group, ancestor, groups).is_some()
}

fn group_distance(group: &str, ancestor: &str, groups: &[GroupYaml]) -> Option<usize> {
    let mut current = group;
    let mut distance = 0;
    loop {
        if current == ancestor {
            return Some(distance);
        }
        current = groups
            .iter()
            .find(|candidate| candidate.name == current)?
            .parent
            .as_deref()?;
        distance += 1;
    }
}

fn validate_memory_reference(
    name: &str,
    placement: Option<&str>,
    catalog: &MemoryCatalog,
    owner: &str,
) -> Result<(), ArchLoadError> {
    if let Some(memory_placement) = catalog.physical_placements.get(name) {
        if memory_placement.as_deref() != placement {
            return Err(invalid(format!(
                "{owner} cannot use local memory '{name}' outside its placement group; use a declared aggregate"
            )));
        }
        return Ok(());
    }
    if let Some(aggregate) = catalog.aggregates.get(name) {
        if aggregate.visible_in.as_deref() != placement {
            return Err(invalid(format!(
                "{owner} cannot use aggregate memory '{}' outside the parent of group '{}'",
                aggregate.name, aggregate.across
            )));
        }
        return Ok(());
    }
    Err(invalid(format!("{owner} uses unknown memory '{name}'")))
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
    spec: &ArchitectureYaml,
    placement: Option<&str>,
    artifact_dir: &Path,
    dims: &BTreeMap<String, Dimension>,
    catalog: &MemoryCatalog,
    function_names: &mut HashSet<String>,
) -> Result<Architecture, ArchLoadError> {
    let (scope_name, scope_scale) = match placement {
        None => (spec.name.as_str(), &[][..]),
        Some(group_name) => {
            let group = spec
                .groups
                .iter()
                .find(|group| group.name == group_name)
                .ok_or_else(|| invalid(format!("unknown group '{group_name}'")))?;
            (group.name.as_str(), group.scale.as_slice())
        }
    };
    let scope_dims = resolve_dims(scope_scale, dims)?;
    let mut arch = Architecture::scope(scope_name);

    for memory in &spec.memories {
        let MemoryYaml::Physical(memory) = memory else {
            continue;
        };
        if memory.placement.as_deref() != placement {
            continue;
        }
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

    for group in spec
        .groups
        .iter()
        .filter(|group| group.parent.as_deref() == placement)
    {
        arch.add_child(build_scope(
            spec,
            Some(&group.name),
            artifact_dir,
            dims,
            catalog,
            function_names,
        )?);
    }

    for processor in spec
        .processors
        .iter()
        .filter(|processor| processor.placement.as_deref() == placement)
    {
        let source = resolve_memory_region(&arch, &processor.from, catalog).ok_or_else(|| {
            invalid(format!(
                "processor '{}' cannot resolve source memory '{}' from scope '{}'",
                processor.name, processor.from, scope_name
            ))
        })?;
        let destination =
            resolve_memory_region(&arch, &processor.to, catalog).ok_or_else(|| {
                invalid(format!(
                    "processor '{}' cannot resolve destination memory '{}' from scope '{}'",
                    processor.name, processor.to, scope_name
                ))
            })?;
        let (functionality, perf) =
            load_functionality_and_perf(artifact_dir, &processor.name, catalog, function_names)?;
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

    for network in spec
        .networks
        .iter()
        .filter(|network| network.placement.as_deref() == placement)
    {
        let network_dims = resolve_dims(&network.dimensions, dims)?;
        let region = resolve_memory_region(&arch, &network.region, catalog).ok_or_else(|| {
            invalid(format!(
                "network '{}' cannot resolve memory '{}' from scope '{}'",
                network.name, network.region, scope_name
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
            invalid(format!(
                "network '{}' is invalid: {message}",
                network.name()
            ))
        })?;
        arch.add_network(network);
    }

    Ok(arch.with_dims(&scope_dims))
}

fn resolve_memory_region(
    arch: &Architecture,
    source_name: &str,
    catalog: &MemoryCatalog,
) -> Option<MemoryRegion> {
    let base = catalog
        .aggregates
        .get(source_name)
        .map(|aggregate| aggregate.base.as_str())
        .unwrap_or(source_name);
    arch.get_scaled_memory_region(base)
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
    catalog: &MemoryCatalog,
    global_names: &mut HashSet<String>,
) -> Result<(MlirModule, Vec<super::FuncPerfModel>), ArchLoadError> {
    let mlir_path = dir.join(format!("{processor_name}.mlir"));
    let perf_path = dir.join(format!("{processor_name}.perf.yaml"));
    let mut functionality = MlirModule::from_mlir(mlir_path.to_string_lossy().into_owned())
        .map_err(|message| invalid(message))?;
    functionality.memory_aliases = catalog
        .aggregates
        .values()
        .map(|aggregate| (aggregate.name.clone(), aggregate.target.clone()))
        .collect();
    for function in &mut functionality.functions {
        let Some(details) = function.mlir_details.as_mut() else {
            continue;
        };
        for binding in &mut details.mem_region_bindings {
            canonicalize_memory_name(&mut binding.region, catalog);
        }
        for copy in &mut details.copy_ops {
            canonicalize_memory_name(&mut copy.src_region, catalog);
            canonicalize_memory_name(&mut copy.dst_region, catalog);
        }
    }
    let perf_spec = PerfYamlSpec::from_file(&perf_path)
        .map_err(|error| invalid(format!("{}: {error}", perf_path.display())))?;

    let mlir_names = functionality
        .functions
        .iter()
        .map(|function| function.name.clone())
        .collect::<HashSet<_>>();
    let perf_names = perf_spec
        .function_names()
        .map(str::to_string)
        .collect::<HashSet<_>>();
    if mlir_names != perf_names {
        let mut only_mlir = mlir_names
            .difference(&perf_names)
            .cloned()
            .collect::<Vec<_>>();
        let mut only_perf = perf_names
            .difference(&mlir_names)
            .cloned()
            .collect::<Vec<_>>();
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

fn canonicalize_memory_name(name: &mut String, catalog: &MemoryCatalog) {
    if let Some(aggregate) = catalog.aggregates.get(name) {
        *name = aggregate.target.clone();
    }
}

fn invalid(message: impl Into<String>) -> ArchLoadError {
    ArchLoadError::Invalid(message.into())
}

#[cfg(test)]
mod tests {
    use super::{ChipYaml, MemoryYaml};

    #[test]
    fn rejects_unknown_fields() {
        let error = ChipYaml::from_yaml_str(
            r#"
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
        let error = ChipYaml::from_yaml_str(
            r#"
dimensions: {x: 2}
architecture:
  name: system
  groups:
    - {name: core, scale: [x]}
  memories:
    - {name: L1, block_size_bytes: 16, num_blocks: 4}
    - {name: L1, in: core, block_size_bytes: 16, num_blocks: 4}
"#,
        )
        .expect_err("ambiguous memory names must fail");
        assert!(error.to_string().contains("duplicate memory name 'L1'"));
    }

    #[test]
    fn deserializes_memories_into_distinct_variants() {
        let spec = ChipYaml::from_yaml_str(
            r#"
dimensions: {x: 2}
architecture:
  name: system
  groups:
    - {name: cores, scale: [x]}
  memories:
    - {name: L1, in: cores, block_size_bytes: 16, num_blocks: 4}
    - {name: L1_all, of: L1, across: cores}
"#,
        )
        .expect("physical and aggregate memories should parse");

        assert!(matches!(
            spec.architecture.memories[0],
            MemoryYaml::Physical(_)
        ));
        assert!(matches!(
            spec.architecture.memories[1],
            MemoryYaml::Aggregate(_)
        ));
    }

    #[test]
    fn physical_memory_requires_complete_storage() {
        let error = ChipYaml::from_yaml_str(
            r#"
architecture:
  name: system
  memories:
    - {name: DRAM, block_size_bytes: 64}
"#,
        )
        .expect_err("an incomplete physical memory must fail during deserialization");

        assert!(
            error
                .to_string()
                .contains("must specify block_size_bytes and num_blocks")
        );
    }

    #[test]
    fn aggregate_memory_requires_complete_relation() {
        let error = ChipYaml::from_yaml_str(
            r#"
architecture:
  name: system
  memories:
    - {name: L1_all, of: L1}
"#,
        )
        .expect_err("an incomplete aggregate must fail during deserialization");

        assert!(
            error
                .to_string()
                .contains("must specify both 'of' and 'across'")
        );
    }

    #[test]
    fn lowers_affine_mesh_networks() {
        let spec = ChipYaml::from_yaml_str(
            r#"
dimensions: {x: 4, y: 2}
architecture:
  name: system
  groups:
    - {name: mesh, scale: [x, y]}
  memories:
    - name: L1
      in: mesh
      block_size_bytes: 16
      num_blocks: 4
      aggregate: {name: L1_mesh}
  networks:
    - name: torus
      dimensions: [x, y]
      region: L1_mesh
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
        assert_eq!(
            architecture.networks[0].dimensions(),
            architecture.children[0].dims()
        );
    }

    #[test]
    fn standalone_and_nested_aggregates_normalize_equally() {
        let nested = ChipYaml::from_yaml_str(
            r#"
dimensions: {x: 4}
architecture:
  name: system
  groups:
    - {name: cores, scale: [x]}
  memories:
    - name: L1
      in: cores
      block_size_bytes: 16
      num_blocks: 4
      aggregate: {name: L1_all}
"#,
        )
        .expect("nested aggregate should parse");
        let standalone = ChipYaml::from_yaml_str(
            r#"
dimensions: {x: 4}
architecture:
  name: system
  groups:
    - {name: cores, scale: [x]}
  memories:
    - {name: L1, in: cores, block_size_bytes: 16, num_blocks: 4}
    - {name: L1_all, of: L1, across: cores}
"#,
        )
        .expect("standalone aggregate should parse");

        let nested = nested.memory_catalog().expect("nested catalog");
        let standalone = standalone.memory_catalog().expect("standalone catalog");
        assert_eq!(
            nested.aggregates["L1_all"].target,
            standalone.aggregates["L1_all"].target
        );
        assert_eq!(nested.aggregates["L1_all"].target, "array_L1");
    }

    #[test]
    fn nested_group_aggregate_names_each_scaled_level() {
        let spec = ChipYaml::from_yaml_str(
            r#"
dimensions: {c: 2, x: 4}
architecture:
  name: system
  groups:
    - {name: clusters, scale: [c]}
    - {name: cores, in: clusters, scale: [x]}
  memories:
    - {name: L1, in: cores, block_size_bytes: 16, num_blocks: 4}
    - {name: L1_cluster, of: L1, across: cores}
    - {name: L1_system, of: L1, across: clusters}
"#,
        )
        .expect("nested groups should parse");
        let catalog = spec.memory_catalog().expect("catalog should build");
        assert_eq!(catalog.aggregates["L1_cluster"].target, "array_L1");
        assert_eq!(catalog.aggregates["L1_system"].target, "array_array_L1");
    }

    #[test]
    fn version_is_not_part_of_the_schema() {
        let error = ChipYaml::from_yaml_str(
            r#"
version: 1
architecture: {name: system}
"#,
        )
        .expect_err("version should be rejected as an obsolete field");
        assert!(error.to_string().contains("unknown field"));
    }

    #[test]
    fn local_memory_requires_an_aggregate_outside_its_group() {
        let error = ChipYaml::from_yaml_str(
            r#"
dimensions: {x: 4}
architecture:
  name: system
  groups:
    - {name: cores, scale: [x]}
  memories:
    - {name: L1, in: cores, block_size_bytes: 16, num_blocks: 4}
  processors:
    - {name: dma, kind: data_mover, from: L1, to: L1}
"#,
        )
        .expect_err("root processor must use an aggregate name");
        assert!(
            error
                .to_string()
                .contains("cannot use local memory 'L1' outside its placement group")
        );
    }

    #[test]
    fn aggregate_cannot_define_physical_storage() {
        let error = ChipYaml::from_yaml_str(
            r#"
dimensions: {x: 4}
architecture:
  name: system
  groups:
    - {name: cores, scale: [x]}
  memories:
    - {name: L1, in: cores, block_size_bytes: 16, num_blocks: 4}
    - {name: L1_all, of: L1, across: cores, num_blocks: 4}
"#,
        )
        .expect_err("aggregate storage fields must fail");
        assert!(
            error
                .to_string()
                .contains("cannot define placement, storage, scale, or another aggregate")
        );
    }
}
