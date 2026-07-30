use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::index::IndexDomain;
use super::memory::{
    MemoryArray, MemoryCatalog, MemoryDefinition, MemoryEndpoint, NamedMemoryRegion,
    validate_static_bank,
};
use super::processor::{
    AffineRelation, ConnectionSpec, ProcessorArray, ProcessorDefinition, ResolvedConnection,
    ResolvedMemoryEndpoint, endpoint_has_region_selector,
};
use super::resource::ResourceArray;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArchitectureError(pub String);

impl std::fmt::Display for ArchitectureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ArchitectureError {}

/// Canonical, flat, indexed architecture representation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Architecture {
    pub name: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dimensions: Vec<IndexDomain>,
    pub memory_catalog: MemoryCatalog,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub memories: Vec<MemoryArray>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub processor_definitions: Vec<ProcessorDefinition>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub processors: Vec<ProcessorArray>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub resources: Vec<ResourceArray>,
}

impl Architecture {
    pub fn builder(name: impl Into<String>) -> ArchitectureBuilder {
        ArchitectureBuilder::new(name)
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn dimension(&self, name: &str) -> Option<&IndexDomain> {
        self.dimensions
            .iter()
            .find(|dimension| dimension.name == name)
    }

    pub fn memory(&self, name: &str) -> Option<&MemoryArray> {
        self.memories.iter().find(|memory| memory.name == name)
    }

    pub fn memory_definition(&self, memory: &MemoryArray) -> Option<&MemoryDefinition> {
        self.memory_catalog.definition(&memory.definition)
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

    pub fn get_function(&self, name: &str) -> Option<&super::processor::FunctionProcessor> {
        self.processor_definitions
            .iter()
            .find_map(|definition| definition.get_function(name))
    }
}

#[derive(Clone, Debug)]
pub struct ArchitectureBuilder {
    name: String,
    dimensions: Vec<IndexDomain>,
    catalog: MemoryCatalog,
    placements: Vec<(String, String, Vec<String>)>,
    processor_definitions: Vec<ProcessorDefinition>,
    connections: Vec<(String, ConnectionSpec)>,
    resources: Vec<ResourceArray>,
}

impl ArchitectureBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            dimensions: Vec::new(),
            catalog: MemoryCatalog::default(),
            placements: Vec::new(),
            processor_definitions: Vec::new(),
            connections: Vec::new(),
            resources: Vec::new(),
        }
    }

    pub fn dimension(mut self, name: impl Into<String>, size: u64) -> Self {
        self.dimensions.push(IndexDomain::new(name, size));
        self
    }

    pub fn memory_definition(mut self, definition: MemoryDefinition) -> Self {
        self.catalog.definitions.push(definition);
        self
    }

    pub fn named_region(mut self, region: NamedMemoryRegion) -> Self {
        self.catalog.regions.push(region);
        self
    }

    pub fn memory_catalog(mut self, catalog: MemoryCatalog) -> Self {
        self.catalog = catalog;
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

    pub fn connect(
        mut self,
        processor_definition: impl Into<String>,
        connection: ConnectionSpec,
    ) -> Self {
        self.connections
            .push((processor_definition.into(), connection));
        self
    }

    pub fn resource(mut self, resource: ResourceArray) -> Self {
        self.resources.push(resource);
        self
    }

    pub fn build(self) -> Result<Architecture, ArchitectureError> {
        if self.name.is_empty() {
            return Err(ArchitectureError(
                "architecture name cannot be empty".into(),
            ));
        }
        validate_unique(
            self.dimensions
                .iter()
                .map(|dimension| dimension.name.as_str()),
            "dimension",
        )?;
        self.catalog.validate().map_err(ArchitectureError)?;
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
            let definition = self.catalog.definition(definition_name).ok_or_else(|| {
                ArchitectureError(format!(
                    "placement '{}' refers to unknown memory definition '{}'",
                    name, definition_name
                ))
            })?;
            if placement.len() != definition.indices.len() {
                return Err(ArchitectureError(format!(
                    "placement '{}' binds {} dimensions; memory definition '{}' expects {}",
                    name,
                    placement.len(),
                    definition_name,
                    definition.indices.len()
                )));
            }
            let indices = placement
                .iter()
                .map(|dimension| {
                    dimension_map
                        .get(dimension.as_str())
                        .cloned()
                        .ok_or_else(|| {
                            ArchitectureError(format!(
                                "placement '{}' uses unknown dimension '{}'",
                                name, dimension
                            ))
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            memories.push(MemoryArray::new(name, definition_name, indices));
        }

        for definition in &self.processor_definitions {
            if definition.name.is_empty() {
                return Err(ArchitectureError(
                    "processor definition name cannot be empty".into(),
                ));
            }
            for function in &definition.functions {
                function.validate().map_err(ArchitectureError)?;
            }
        }
        let mut function_names = BTreeSet::new();
        for function in self
            .processor_definitions
            .iter()
            .flat_map(|definition| &definition.functions)
        {
            if !function_names.insert(function.func.name.as_str()) {
                return Err(ArchitectureError(format!(
                    "function '{}' is defined by more than one processor",
                    function.func.name
                )));
            }
        }

        validate_unique(
            self.resources.iter().map(|resource| resource.name.as_str()),
            "resource",
        )?;
        let shared_resources = self
            .resources
            .iter()
            .cloned()
            .map(|resource| (resource.name.clone(), resource))
            .collect::<BTreeMap<_, _>>();
        let mut processors = Vec::new();
        let mut resources = self.resources;
        let mut counts = BTreeMap::<String, usize>::new();
        for (definition_name, connection) in self.connections {
            let definition = self
                .processor_definitions
                .iter()
                .find(|definition| definition.name == definition_name)
                .ok_or_else(|| {
                    ArchitectureError(format!(
                        "connection refers to unknown processor definition '{}'",
                        definition_name
                    ))
                })?;
            let mut resolved_connection = connection.clone();
            resolve_named_regions(&mut resolved_connection, &self.catalog)?;
            validate_connection(&resolved_connection, &memories, &self.catalog)?;
            let domain = infer_domain(&resolved_connection, &dimension_map)?;
            let instances = resolve_connection_instances(
                &resolved_connection,
                &domain,
                &memories,
                &self.catalog,
            )?;
            let index = counts.entry(definition_name.clone()).or_default();
            let name = if *index == 0 {
                definition_name.clone()
            } else {
                format!("{}__{}", definition_name, index)
            };
            *index += 1;
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
                    ArchitectureError(format!(
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
                relation: AffineRelation { domain, instances },
                resources: processor_resources,
            });
        }
        validate_unique(
            resources.iter().map(|resource| resource.name.as_str()),
            "resource",
        )?;

        Ok(Architecture {
            name: self.name,
            dimensions: self.dimensions,
            memory_catalog: self.catalog,
            memories,
            processor_definitions: self.processor_definitions,
            processors,
            resources,
        })
    }
}

fn validate_unique<'a>(
    names: impl IntoIterator<Item = &'a str>,
    kind: &str,
) -> Result<(), ArchitectureError> {
    let mut unique = BTreeSet::new();
    for name in names {
        if !unique.insert(name) {
            return Err(ArchitectureError(format!("duplicate {kind} '{name}'")));
        }
    }
    Ok(())
}

fn resolve_named_regions(
    connection: &mut ConnectionSpec,
    catalog: &MemoryCatalog,
) -> Result<(), ArchitectureError> {
    for endpoint in connection.inputs.iter_mut().chain(&mut connection.outputs) {
        if let Some(region) = catalog.region(&endpoint.memory) {
            if !endpoint.indices.is_empty() || endpoint.bank.is_some() {
                return Err(ArchitectureError(format!(
                    "named region '{}' cannot be further indexed",
                    region.name
                )));
            }
            *endpoint = region.endpoint.clone();
        }
    }
    Ok(())
}

fn validate_connection(
    connection: &ConnectionSpec,
    memories: &[MemoryArray],
    catalog: &MemoryCatalog,
) -> Result<(), ArchitectureError> {
    for endpoint in connection.inputs.iter().chain(&connection.outputs) {
        let memory = memories
            .iter()
            .find(|memory| memory.name == endpoint.memory)
            .ok_or_else(|| {
                ArchitectureError(format!(
                    "connection refers to unknown placed memory '{}'",
                    endpoint.memory
                ))
            })?;
        if endpoint.indices.len() != memory.indices.len() {
            return Err(ArchitectureError(format!(
                "endpoint '{}' has {} indices; placed memory expects {}",
                endpoint.memory,
                endpoint.indices.len(),
                memory.indices.len()
            )));
        }
        let definition = catalog.definition(&memory.definition).ok_or_else(|| {
            ArchitectureError(format!(
                "placed memory '{}' has unknown definition '{}'",
                memory.name, memory.definition
            ))
        })?;
        validate_static_bank(endpoint, definition).map_err(ArchitectureError)?;
    }
    Ok(())
}

fn infer_domain(
    connection: &ConnectionSpec,
    dimensions: &BTreeMap<&str, IndexDomain>,
) -> Result<Vec<IndexDomain>, ArchitectureError> {
    connection
        .variables()
        .into_iter()
        .map(|variable| {
            dimensions.get(variable.as_str()).cloned().ok_or_else(|| {
                ArchitectureError(format!(
                    "connection uses unknown free index variable '{}'",
                    variable
                ))
            })
        })
        .collect()
}

fn resolve_connection_instances(
    connection: &ConnectionSpec,
    domain: &[IndexDomain],
    memories: &[MemoryArray],
    catalog: &MemoryCatalog,
) -> Result<Vec<ResolvedConnection>, ArchitectureError> {
    let mut points = vec![BTreeMap::<String, i64>::new()];
    for dimension in domain {
        let mut expanded = Vec::new();
        for point in points {
            for value in 0..dimension.size {
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
            let Some(endpoint) = resolve_endpoint(symbolic, &point, memories, catalog)? else {
                continue 'point;
            };
            inputs.push(endpoint);
        }
        for symbolic in &connection.outputs {
            let Some(endpoint) = resolve_endpoint(symbolic, &point, memories, catalog)? else {
                continue 'point;
            };
            outputs.push(endpoint);
        }
        resolved.push(ResolvedConnection {
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
    catalog: &MemoryCatalog,
) -> Result<Option<ResolvedMemoryEndpoint>, ArchitectureError> {
    let memory = memories
        .iter()
        .find(|memory| memory.name == endpoint.memory)
        .expect("connection was validated");
    if endpoint_has_region_selector(endpoint) {
        return Ok(Some(ResolvedMemoryEndpoint {
            memory: endpoint.memory.clone(),
            indices: Vec::new(),
            bank: None,
        }));
    }
    let mut indices = Vec::new();
    for (selector, domain) in endpoint.indices.iter().zip(&memory.indices) {
        let super::memory::EndpointIndex::Expression(expression) = selector else {
            unreachable!()
        };
        let value = expression.evaluate(values).ok_or_else(|| {
            ArchitectureError(format!(
                "could not evaluate index for memory '{}'",
                endpoint.memory
            ))
        })?;
        if value < 0 || value >= domain.size as i64 {
            return Ok(None);
        }
        indices.push(value as u64);
    }
    let definition = catalog
        .definition(&memory.definition)
        .expect("placement was validated");
    let bank = endpoint
        .bank
        .as_ref()
        .map(|expression| {
            expression.evaluate(values).ok_or_else(|| {
                ArchitectureError(format!(
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
    Ok(Some(ResolvedMemoryEndpoint {
        memory: endpoint.memory.clone(),
        indices,
        bank: bank.map(|bank| bank as u64),
    }))
}
