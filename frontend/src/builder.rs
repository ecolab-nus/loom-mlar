use crate::{LoomMemoryBinding, ProcessorYaml, lower_loom_source, selection::MemoryEndpoint};
use mlar_rust::{
    Architecture, ArchitectureError, Axis, MemoryDefinition, NetworkTopology, OperationModel,
    ProcessorType, Resource, Scope,
};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProcessorSourceFormat {
    CompactLoom,
    Mlir,
}

#[derive(Clone, Debug)]
pub struct ProcessorDefinition {
    pub(crate) name: String,
    pub(crate) processor_type: Option<ProcessorType>,
    pub(crate) source: String,
    pub(crate) source_format: ProcessorSourceFormat,
    pub(crate) functions: Vec<OperationModel>,
    pub(crate) resources: Vec<Resource>,
}
impl ProcessorDefinition {
    pub fn new(
        name: impl Into<String>,
        source: impl Into<String>,
        functions: Vec<OperationModel>,
    ) -> Self {
        Self {
            name: name.into(),
            source: source.into(),
            source_format: ProcessorSourceFormat::CompactLoom,
            functions,
            processor_type: None,
            resources: vec![],
        }
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn source(&self) -> &str {
        &self.source
    }
    pub fn functions(&self) -> &[OperationModel] {
        &self.functions
    }
    pub fn resources(&self) -> &[Resource] {
        &self.resources
    }
    pub fn processor_type(&self) -> Option<&ProcessorType> {
        self.processor_type.as_ref()
    }
    pub fn source_format(&self) -> &ProcessorSourceFormat {
        &self.source_format
    }
    pub fn get_function(&self, name: &str) -> Option<&OperationModel> {
        self.functions.iter().find(|op| op.func.name == name)
    }
    pub fn with_type(mut self, kind: ProcessorType) -> Self {
        self.processor_type = Some(kind);
        self
    }
    pub fn with_resources(mut self, resources: Vec<Resource>) -> Self {
        self.resources = resources;
        self
    }
    fn lower(
        &self,
        inputs: &[LoomMemoryBinding],
        outputs: &[LoomMemoryBinding],
    ) -> Result<mlar_rust::ProcessorDefinition, String> {
        let mut definition = if self.source.trim().is_empty()
            || self.source_format == ProcessorSourceFormat::Mlir
        {
            mlar_rust::ProcessorDefinition::new(&self.name, &self.source, self.functions.clone())
        } else {
            let parsed =
                crate::parse_loom_source(&self.source).map_err(|error| error.to_string())?;
            if parsed.functions.len() != self.functions.len() {
                return Err(format!(
                    "processor '{}' source functions disagree with its operation models",
                    self.name
                ));
            }
            for operation in &self.functions {
                let function = parsed
                    .functions
                    .iter()
                    .find(|function| function.name == operation.func.name)
                    .ok_or_else(|| format!("unknown source function '{}'", operation.func.name))?;
                operation.validate()?;
                if operation.func.mlir_details != function.mlir_details
                    || operation
                        .func
                        .symbols
                        .iter()
                        .collect::<std::collections::BTreeSet<_>>()
                        != function
                            .symbols
                            .iter()
                            .collect::<std::collections::BTreeSet<_>>()
                {
                    return Err(format!(
                        "processor '{}' function '{}' metadata disagrees with its compact source",
                        self.name, function.name
                    ));
                }
                operation
                    .perf
                    .validate_for_func(function)
                    .map_err(|symbols| {
                        format!(
                            "function '{}' performance uses undeclared symbols: {symbols:?}",
                            function.name
                        )
                    })?;
            }
            let source = lower_loom_source(
                &self.source,
                "processor",
                &self
                    .functions
                    .iter()
                    .map(|op| (op.func.name.clone(), op.func.symbols.clone()))
                    .collect(),
                inputs,
                outputs,
            )
            .map_err(|error| error.to_string())?;
            mlar_rust::ProcessorDefinition::from_mlir_source(
                &self.name,
                source,
                self.functions
                    .iter()
                    .map(|op| (op.func.name.clone(), op.perf.clone())),
            )?
        }
        .with_resources(self.resources.clone());
        if let Some(kind) = &self.processor_type {
            definition = definition.with_type(kind.clone());
        }
        definition.validate()?;
        Ok(definition)
    }
}
impl From<mlar_rust::ProcessorDefinition> for ProcessorDefinition {
    fn from(def: mlar_rust::ProcessorDefinition) -> Self {
        Self {
            name: def.name().into(),
            source: def.source().into(),
            source_format: ProcessorSourceFormat::Mlir,
            functions: def.operations().to_vec(),
            resources: def.resources().to_vec(),
            processor_type: def.processor_type().cloned(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Connection {
    pub domain: Vec<String>,
    pub inputs: Vec<MemoryEndpoint>,
    pub outputs: Vec<MemoryEndpoint>,
    pub resources: Vec<String>,
}
impl Connection {
    pub fn new(
        domain: impl IntoIterator<Item = impl Into<String>>,
        inputs: Vec<MemoryEndpoint>,
        outputs: Vec<MemoryEndpoint>,
    ) -> Self {
        Self {
            domain: domain.into_iter().map(Into::into).collect(),
            inputs,
            outputs,
            resources: vec![],
        }
    }
    pub fn parse<'a>(
        domain: impl IntoIterator<Item = &'a str>,
        inputs: impl IntoIterator<Item = &'a str>,
        outputs: impl IntoIterator<Item = &'a str>,
    ) -> Result<Self, mlar_rust::arch::EndpointParseError> {
        Ok(Self::new(
            domain,
            inputs
                .into_iter()
                .map(MemoryEndpoint::parse)
                .collect::<Result<_, _>>()?,
            outputs
                .into_iter()
                .map(MemoryEndpoint::parse)
                .collect::<Result<_, _>>()?,
        ))
    }
    pub fn with_resources(
        mut self,
        resources: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.resources = resources.into_iter().map(Into::into).collect();
        self
    }
}

/// Authoring context; normalizes memory selections and lowers compact processors.
pub struct ArchitectureBuilder {
    core: mlar_rust::ArchitectureBuilder,
    axes: BTreeMap<String, Axis>,
    memories: BTreeMap<String, (String, Vec<Vec<String>>)>,
    memory_definitions: BTreeMap<String, MemoryDefinition>,
    definitions: Vec<ProcessorDefinition>,
    connections: Vec<(String, String, Connection)>,
    directory: Option<PathBuf>,
    errors: Vec<String>,
}
impl ArchitectureBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            core: mlar_rust::ArchitectureBuilder::new(name),
            axes: BTreeMap::new(),
            memories: BTreeMap::new(),
            memory_definitions: BTreeMap::new(),
            definitions: vec![],
            connections: vec![],
            directory: None,
            errors: vec![],
        }
    }
    pub fn axis(mut self, name: impl Into<String>, extent: u64) -> Self {
        let name = name.into();
        self.axes.insert(name.clone(), Axis::new(&name, extent));
        self.core = self.core.axis(name, extent);
        self
    }
    pub fn memory_definition(mut self, definition: MemoryDefinition) -> Self {
        self.memory_definitions
            .insert(definition.name.clone(), definition.clone());
        self.core = self.core.memory_definition(definition);
        self
    }
    pub fn place_memory(
        self,
        definition: impl Into<String>,
        dimensions: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        let name = definition.into();
        self.place_memory_as(name.clone(), name, dimensions)
    }
    pub fn place_memory_as(
        self,
        name: impl Into<String>,
        definition: impl Into<String>,
        dimensions: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        let axes = dimensions.into_iter().map(Into::into).collect::<Vec<_>>();
        self.place_memory_levels(
            name,
            definition,
            if axes.is_empty() { vec![] } else { vec![axes] },
        )
    }
    pub fn place_memory_levels(
        mut self,
        name: impl Into<String>,
        definition: impl Into<String>,
        levels: Vec<Vec<String>>,
    ) -> Self {
        let name = name.into();
        let definition = definition.into();
        if levels.iter().any(Vec::is_empty) {
            self.errors
                .push(format!("memory '{name}' has an empty axis level"));
        }
        self.core = self
            .core
            .place_memory_as(&name, &definition, levels.iter().flatten().cloned());
        self.memories.insert(name, (definition, levels));
        self
    }
    pub fn processor_definition(mut self, def: impl Into<ProcessorDefinition>) -> Self {
        self.definitions.push(def.into());
        self
    }
    pub fn processor_source_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.directory = Some(dir.into());
        self
    }
    pub fn processor(mut self, name: impl AsRef<str>) -> Self {
        let name = name.as_ref();
        match &self.directory {
            None => self
                .errors
                .push(format!("processor '{name}' needs a `processor_source_dir`")),
            Some(dir) => {
                let path = dir.join(format!("{name}.yaml"));
                match ProcessorYaml::from_file(&path).and_then(|yaml| yaml.build_definition(&path))
                {
                    Ok(def) => self.definitions.push(def),
                    Err(error) => self.errors.push(error.to_string()),
                }
            }
        }
        self
    }
    pub fn processors(mut self, names: impl IntoIterator<Item = impl AsRef<str>>) -> Self {
        for name in names {
            self = self.processor(name);
        }
        self
    }
    pub fn connect(self, definition: impl Into<String>, connection: Connection) -> Self {
        let name = definition.into();
        self.connect_as(name.clone(), name, connection)
    }
    pub fn connect_as(
        mut self,
        name: impl Into<String>,
        definition: impl Into<String>,
        connection: Connection,
    ) -> Self {
        self.connections
            .push((name.into(), definition.into(), connection));
        self
    }
    pub fn resource(mut self, resource: Resource) -> Self {
        self.core = self.core.resource(resource);
        self
    }
    pub fn network(mut self, network: NetworkTopology) -> Self {
        self.core = self.core.network(network);
        self
    }
    pub fn scope(mut self, scope: Scope) -> Self {
        self.core = self.core.scope(scope);
        self
    }
    pub(crate) fn lower_endpoint(
        &self,
        endpoint: &MemoryEndpoint,
    ) -> Result<mlar_rust::arch::MemoryEndpoint, String> {
        let (_, levels) = self
            .memories
            .get(&endpoint.memory)
            .ok_or_else(|| format!("unknown memory '{}'", endpoint.memory))?;
        endpoint.lower(levels)
    }
    fn binding(
        &self,
        endpoint: &mlar_rust::arch::MemoryEndpoint,
        symbol: String,
    ) -> Result<LoomMemoryBinding, String> {
        let (definition, levels) = &self.memories[&endpoint.memory];
        let technology = self
            .memory_definitions
            .get(definition)
            .ok_or_else(|| format!("unknown memory definition '{definition}'"))?
            .technology
            .clone();
        let scope_extent = endpoint
            .indices
            .iter()
            .zip(levels.iter().flatten())
            .filter_map(|(selector, axis)| {
                matches!(selector, mlar_rust::arch::EndpointIndex::All).then_some(axis)
            })
            .map(|axis| {
                self.axes
                    .get(axis)
                    .map(Axis::extent)
                    .ok_or_else(|| format!("unknown axis '{axis}'"))
            })
            .collect::<Result<_, _>>()?;
        Ok(LoomMemoryBinding {
            symbol,
            technology,
            scope_extent,
        })
    }
    pub fn build(self) -> Result<Architecture, ArchitectureError> {
        self.translate().map_err(ArchitectureError::Invalid)
    }
    fn translate(self) -> Result<Architecture, String> {
        if !self.errors.is_empty() {
            return Err(self.errors.join("; "));
        }
        let mut names = std::collections::BTreeSet::new();
        for def in &self.definitions {
            if !names.insert(&def.name) {
                return Err(format!("duplicate processor definition '{}'", def.name));
            }
        }
        let mut canonical = Vec::<(String, String, mlar_rust::ProcessorDefinition)>::new();
        let mut connections = Vec::new();
        for (name, definition, authored) in &self.connections {
            let inputs = authored
                .inputs
                .iter()
                .map(|ep| self.lower_endpoint(ep))
                .collect::<Result<Vec<_>, _>>()?;
            let outputs = authored
                .outputs
                .iter()
                .map(|ep| self.lower_endpoint(ep))
                .collect::<Result<Vec<_>, _>>()?;
            let def = self
                .definitions
                .iter()
                .find(|def| def.name == *definition)
                .ok_or_else(|| format!("unknown processor definition '{definition}'"))?;
            let input_bindings = inputs
                .iter()
                .enumerate()
                .map(|(i, ep)| self.binding(ep, format!("input_{i}")))
                .collect::<Result<Vec<_>, _>>()?;
            let output_bindings = outputs
                .iter()
                .enumerate()
                .map(|(i, ep)| self.binding(ep, format!("output_{i}")))
                .collect::<Result<Vec<_>, _>>()?;
            let lowered = def.lower(&input_bindings, &output_bindings)?;
            let fingerprint = serde_json::to_string(&lowered).map_err(|error| error.to_string())?;
            let canonical_name = if let Some((_, name, _)) =
                canonical.iter().find(|(fp, _, _)| *fp == fingerprint)
            {
                name.clone()
            } else {
                let index = canonical
                    .iter()
                    .filter(|(_, _, def)| {
                        def.name() == definition
                            || def.name().starts_with(&format!("{definition}__"))
                    })
                    .count();
                let canonical_name = if index == 0 {
                    definition.clone()
                } else {
                    format!("{definition}__{index}")
                };
                canonical.push((
                    fingerprint,
                    canonical_name.clone(),
                    lowered.with_name(&canonical_name),
                ));
                canonical_name
            };
            connections.push((
                name.clone(),
                canonical_name,
                mlar_rust::Connection::new(authored.domain.clone(), inputs, outputs)
                    .with_resources(authored.resources.clone()),
            ));
        }
        for def in &self.definitions {
            if !self
                .connections
                .iter()
                .any(|(_, name, _)| name == &def.name)
            {
                canonical.push((String::new(), def.name.clone(), def.lower(&[], &[])?));
            }
        }
        let mut core = self.core;
        for (_, _, def) in canonical {
            core = core.processor_definition(def);
        }
        for (name, def, connection) in connections {
            core = core.connect_as(name, def, connection);
        }
        core.build().map_err(|error| error.to_string())
    }
}
