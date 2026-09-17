use crate::native::{NativeFunction, NativeSourceIndex, compose_module};
use crate::templates::{self, FunctionSpec, ResolvedPort};
use crate::{ProcessorYaml, selection::MemoryEndpoint};
use mlar_rust::{
    Architecture, ArchitectureError, MemoryDefinition, NetworkTopology, OperationModel,
    ProcessorType, Resource, Scope,
};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub(crate) enum FunctionProvider {
    Template(FunctionSpec),
    Native(NativeFunction),
}

#[derive(Clone, Debug)]
pub struct ProcessorDefinition {
    pub(crate) name: String,
    pub(crate) processor_type: Option<ProcessorType>,
    pub(crate) source: String,
    pub(crate) functions: Vec<OperationModel>,
    pub(crate) resources: Vec<Resource>,
    pub(crate) providers: Option<Vec<(String, FunctionProvider)>>,
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
            functions,
            processor_type: None,
            resources: vec![],
            providers: None,
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
        inputs: &[ResolvedPort],
        outputs: &[ResolvedPort],
    ) -> Result<mlar_rust::ProcessorDefinition, String> {
        let source = if let Some(providers) = &self.providers {
            let blocks = providers
                .iter()
                .map(|(name, provider)| match provider {
                    FunctionProvider::Template(spec) => {
                        templates::emit(name, spec, inputs, outputs)
                    }
                    FunctionProvider::Native(function) => {
                        validate_native_bindings(
                            name,
                            &function.block,
                            inputs.len(),
                            outputs.len(),
                        )?;
                        Ok(function.block.clone())
                    }
                })
                .collect::<Result<Vec<_>, String>>()?;
            compose_module(blocks)
        } else {
            self.source.clone()
        };
        let mut definition = if source.trim().is_empty() {
            mlar_rust::ProcessorDefinition::new(&self.name, source, self.functions.clone())
        } else {
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

fn validate_native_bindings(
    function_name: &str,
    block: &str,
    input_count: usize,
    output_count: usize,
) -> Result<(), String> {
    let source = compose_module([block.to_string()]);
    let module = mlar_rust::mlir::MlirModule::from_mlir_source(&source)?;
    let details = module
        .functions
        .first()
        .and_then(|function| function.mlir_details.as_ref())
        .ok_or_else(|| format!("native function '{function_name}' has no parsed interface"))?;
    for binding in &details.mem_region_bindings {
        let is_input = details.source_memrefs.contains(&binding.memref);
        let is_output = details.target_memrefs.contains(&binding.memref);
        let (side, index, count) = if let Some(index) = binding.region.strip_prefix("input_") {
            if !is_input || is_output {
                return Err(format!(
                    "native function '{function_name}' binds non-input '{}' to @{}",
                    binding.memref, binding.region
                ));
            }
            ("input", index, input_count)
        } else if let Some(index) = binding.region.strip_prefix("output_") {
            if !is_output || is_input {
                return Err(format!(
                    "native function '{function_name}' binds non-output '{}' to @{}",
                    binding.memref, binding.region
                ));
            }
            ("output", index, output_count)
        } else {
            return Err(format!(
                "native function '{function_name}' binds '{}' to @{}; package functions must use @input_N or @output_N",
                binding.memref, binding.region
            ));
        };
        let index = index.parse::<usize>().map_err(|_| {
            format!(
                "native function '{function_name}' has invalid binding @{}",
                binding.region
            )
        })?;
        if index >= count {
            return Err(format!(
                "native function '{function_name}' binds '{}' to {side} {index}, but the connection has {count} {side}s",
                binding.memref
            ));
        }
    }
    Ok(())
}
impl From<mlar_rust::ProcessorDefinition> for ProcessorDefinition {
    fn from(def: mlar_rust::ProcessorDefinition) -> Self {
        Self {
            name: def.name().into(),
            source: def.source().into(),
            functions: def.operations().to_vec(),
            resources: def.resources().to_vec(),
            processor_type: def.processor_type().cloned(),
            providers: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct NamedPort {
    pub name: String,
    pub endpoint: MemoryEndpoint,
}

impl NamedPort {
    pub fn new(name: impl Into<String>, endpoint: MemoryEndpoint) -> Self {
        Self {
            name: name.into(),
            endpoint,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Connection {
    pub domain: Vec<String>,
    pub inputs: Vec<NamedPort>,
    pub outputs: Vec<NamedPort>,
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
            inputs: inputs
                .into_iter()
                .enumerate()
                .map(|(index, endpoint)| NamedPort::new(format!("input_{index}"), endpoint))
                .collect(),
            outputs: outputs
                .into_iter()
                .enumerate()
                .map(|(index, endpoint)| NamedPort::new(format!("output_{index}"), endpoint))
                .collect(),
            resources: vec![],
        }
    }
    pub fn named(
        domain: impl IntoIterator<Item = impl Into<String>>,
        inputs: Vec<NamedPort>,
        outputs: Vec<NamedPort>,
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
    pub fn parse_named<'a>(
        domain: impl IntoIterator<Item = &'a str>,
        inputs: impl IntoIterator<Item = (&'a str, &'a str)>,
        outputs: impl IntoIterator<Item = (&'a str, &'a str)>,
    ) -> Result<Self, mlar_rust::arch::EndpointParseError> {
        Ok(Self::named(
            domain,
            inputs
                .into_iter()
                .map(|(name, endpoint)| {
                    MemoryEndpoint::parse(endpoint).map(|endpoint| NamedPort::new(name, endpoint))
                })
                .collect::<Result<_, _>>()?,
            outputs
                .into_iter()
                .map(|(name, endpoint)| {
                    MemoryEndpoint::parse(endpoint).map(|endpoint| NamedPort::new(name, endpoint))
                })
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

/// Authoring context; normalizes memory selections and resolves processor sources.
pub struct ArchitectureBuilder {
    core: mlar_rust::ArchitectureBuilder,
    memories: BTreeMap<String, (String, Vec<Vec<String>>)>,
    memory_definitions: BTreeMap<String, MemoryDefinition>,
    definitions: Vec<ProcessorDefinition>,
    connections: Vec<(String, String, Connection)>,
    directory: Option<PathBuf>,
    native_sources: Option<NativeSourceIndex>,
    errors: Vec<String>,
}
impl ArchitectureBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            core: mlar_rust::ArchitectureBuilder::new(name),
            memories: BTreeMap::new(),
            memory_definitions: BTreeMap::new(),
            definitions: vec![],
            connections: vec![],
            directory: None,
            native_sources: None,
            errors: vec![],
        }
    }
    pub fn axis(mut self, name: impl Into<String>, extent: u64) -> Self {
        let name = name.into();
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
        let dir = dir.into();
        match NativeSourceIndex::load(&dir) {
            Ok(index) => self.native_sources = Some(index),
            Err(error) => self.errors.push(error),
        }
        self.directory = Some(dir);
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
                match ProcessorYaml::from_file(&path).and_then(|yaml| {
                    yaml.build_definition_with_index(
                        &path,
                        self.native_sources.as_ref().ok_or_else(|| {
                            crate::ArchLoadError::Invalid(
                                "native source index is unavailable".into(),
                            )
                        })?,
                    )
                }) {
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
    fn memory_space(
        &self,
        endpoint: &mlar_rust::arch::MemoryEndpoint,
    ) -> Result<Option<u64>, String> {
        let (definition, _) = self
            .memories
            .get(&endpoint.memory)
            .ok_or_else(|| format!("unknown memory '{}'", endpoint.memory))?;
        Ok(self
            .memory_definitions
            .get(definition)
            .ok_or_else(|| format!("unknown memory definition '{definition}'"))?
            .technology
            .as_ref()
            .map(|technology| technology.kind))
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
                .map(|port| self.lower_endpoint(&port.endpoint))
                .collect::<Result<Vec<_>, _>>()?;
            let outputs = authored
                .outputs
                .iter()
                .map(|port| self.lower_endpoint(&port.endpoint))
                .collect::<Result<Vec<_>, _>>()?;
            let def = self
                .definitions
                .iter()
                .find(|def| def.name == *definition)
                .ok_or_else(|| format!("unknown processor definition '{definition}'"))?;
            validate_port_names(name, "input", &authored.inputs)?;
            validate_port_names(name, "output", &authored.outputs)?;
            let input_ports = authored
                .inputs
                .iter()
                .zip(&inputs)
                .map(|(port, endpoint)| {
                    self.memory_space(endpoint)
                        .map(|space| ResolvedPort::new(&port.name, space))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let output_ports = authored
                .outputs
                .iter()
                .zip(&outputs)
                .map(|(port, endpoint)| {
                    self.memory_space(endpoint)
                        .map(|space| ResolvedPort::new(&port.name, space))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let lowered = def.lower(&input_ports, &output_ports)?;
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

fn validate_port_names(placement: &str, side: &str, ports: &[NamedPort]) -> Result<(), String> {
    let mut names = std::collections::BTreeSet::new();
    for port in ports {
        let mut chars = port.name.chars();
        if !matches!(chars.next(), Some(ch) if ch.is_ascii_alphabetic() || ch == '_')
            || !chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
        {
            return Err(format!(
                "processor placement '{placement}' has invalid {side} port name '{}'",
                port.name
            ));
        }
        if !names.insert(&port.name) {
            return Err(format!(
                "processor placement '{placement}' declares duplicate {side} port '{}'",
                port.name
            ));
        }
    }
    Ok(())
}
