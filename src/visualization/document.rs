use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::arch::{
    Architecture, DataEffect, Dimension, MemoryRegion, Processor, Resource, ScaleOutNetwork,
    SizeExpr,
};
use crate::math::{AffineExpr, AffineMap, Expr};

pub const VISUALIZATION_SCHEMA_VERSION: &str = "mlar.visualization.v1";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VisualizationDocumentV1 {
    pub schema_version: String,
    pub architecture: VisualizationArchitecture,
    pub scopes: Vec<VisualizationScope>,
    pub components: Vec<VisualizationComponent>,
    pub relationships: Vec<VisualizationRelationship>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VisualizationArchitecture {
    pub id: String,
    pub name: String,
    pub root_scope: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VisualizationScope {
    pub id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_scope: Option<String>,
    pub dimensions: Vec<VisualizationDimension>,
    pub replication_factor: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VisualizationDimension {
    pub name: String,
    pub size: VisualizationUnsignedExpression,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VisualizationUnsignedExpression {
    pub text: String,
    pub constant: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VisualizationSignedExpression {
    pub text: String,
    pub constant: Option<i64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum VisualizationMemoryRegion {
    Bank {
        name: Option<String>,
        capacity: VisualizationUnsignedExpression,
        block_size: Option<VisualizationUnsignedExpression>,
        total_size_bytes: Option<u64>,
    },
    Array {
        name: Option<String>,
        dimensions: Vec<VisualizationDimension>,
        element: Box<VisualizationMemoryRegion>,
        total_size_bytes: Option<u64>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum VisualizationComponent {
    Memory {
        id: String,
        scope: String,
        name: String,
        dimensions: Vec<VisualizationDimension>,
        region: VisualizationMemoryRegion,
        total_size_bytes: Option<u64>,
    },
    Processor {
        id: String,
        scope: String,
        name: String,
        effect: VisualizationDataEffect,
        functions: Vec<String>,
    },
    DataMover {
        id: String,
        scope: String,
        name: String,
        effect: VisualizationDataEffect,
        functions: Vec<String>,
    },
    Resource {
        id: String,
        scope: String,
        name: String,
        resource_kind: VisualizationResourceKind,
        capacity: Option<i64>,
    },
    Network {
        id: String,
        scope: String,
        name: String,
        network_kind: VisualizationNetworkKind,
        dimensions: Vec<VisualizationDimension>,
        bandwidth: VisualizationSignedExpression,
        latency: Option<VisualizationSignedExpression>,
        links: Vec<VisualizationNetworkLink>,
    },
}

impl VisualizationComponent {
    pub fn id(&self) -> &str {
        match self {
            Self::Memory { id, .. }
            | Self::Processor { id, .. }
            | Self::DataMover { id, .. }
            | Self::Resource { id, .. }
            | Self::Network { id, .. } => id,
        }
    }

    pub fn scope(&self) -> &str {
        match self {
            Self::Memory { scope, .. }
            | Self::Processor { scope, .. }
            | Self::DataMover { scope, .. }
            | Self::Resource { scope, .. }
            | Self::Network { scope, .. } => scope,
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Self::Memory { name, .. }
            | Self::Processor { name, .. }
            | Self::DataMover { name, .. }
            | Self::Resource { name, .. }
            | Self::Network { name, .. } => name,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VisualizationDataEffect {
    Preserve,
    Transform,
    Reduce,
    Accumulate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VisualizationResourceKind {
    Exclusive,
    Quantitative,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VisualizationNetworkKind {
    Mesh,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VisualizationNetworkLink {
    pub id: String,
    pub name: String,
    pub map: VisualizationAffineMap,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VisualizationAffineMap {
    pub source_dimensions: Vec<VisualizationDimension>,
    pub target_dimensions: Vec<VisualizationDimension>,
    pub expressions: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VisualizationRelationshipKind {
    Read,
    Write,
    Requires,
    NetworkAttachment,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VisualizationRelationship {
    pub id: String,
    pub kind: VisualizationRelationshipKind,
    pub source: String,
    pub target: String,
    pub label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bandwidth: Option<VisualizationSignedExpression>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub map: Option<VisualizationAffineMap>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VisualizationExportError {
    DuplicateId(String),
    AmbiguousReference {
        kind: &'static str,
        name: String,
        owner: String,
        candidates: Vec<String>,
    },
    MissingReference {
        kind: &'static str,
        name: String,
        owner: String,
    },
    Yaml(String),
}

impl fmt::Display for VisualizationExportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateId(id) => write!(f, "duplicate visualization id '{id}'"),
            Self::AmbiguousReference {
                kind,
                name,
                owner,
                candidates,
            } => write!(
                f,
                "ambiguous {kind} reference '{name}' for '{owner}'; candidates: {}",
                candidates.join(", ")
            ),
            Self::MissingReference { kind, name, owner } => {
                write!(f, "missing {kind} reference '{name}' for '{owner}'")
            }
            Self::Yaml(message) => write!(f, "visualization YAML serialization failed: {message}"),
        }
    }
}

impl std::error::Error for VisualizationExportError {}

pub fn architecture_to_visualization_document(
    architecture: &Architecture,
) -> Result<VisualizationDocumentV1, VisualizationExportError> {
    let root_path = vec![architecture.name.clone()];
    let root_scope = stable_id("scope", &root_path);
    let mut builder = DocumentBuilder::default();
    builder.collect_scope(architecture, None, &root_path, &[])?;
    builder.resolve_relationships()?;

    builder.scopes.sort_by(|a, b| a.id.cmp(&b.id));
    builder.components.sort_by(|a, b| a.id().cmp(b.id()));
    builder.relationships.sort_by(|a, b| a.id.cmp(&b.id));

    Ok(VisualizationDocumentV1 {
        schema_version: VISUALIZATION_SCHEMA_VERSION.to_string(),
        architecture: VisualizationArchitecture {
            id: stable_id("architecture", &root_path),
            name: architecture.name.clone(),
            root_scope,
        },
        scopes: builder.scopes,
        components: builder.components,
        relationships: builder.relationships,
    })
}

pub fn architecture_to_visualization_yaml(
    architecture: &Architecture,
) -> Result<String, VisualizationExportError> {
    let document = architecture_to_visualization_document(architecture)?;
    serde_yaml::to_string(&document)
        .map_err(|error| VisualizationExportError::Yaml(error.to_string()))
}

#[derive(Default)]
struct DocumentBuilder {
    ids: BTreeSet<String>,
    scopes: Vec<VisualizationScope>,
    components: Vec<VisualizationComponent>,
    relationships: Vec<VisualizationRelationship>,
    scope_parents: HashMap<String, Option<String>>,
    references: Vec<PendingReference>,
}

enum PendingReference {
    ProcessorMemory {
        owner_id: String,
        owner_name: String,
        owner_scope: String,
        memory_name: String,
        kind: VisualizationRelationshipKind,
    },
    ProcessorResource {
        owner_id: String,
        owner_name: String,
        owner_scope: String,
        resource_name: String,
    },
    NetworkMemory {
        network_id: String,
        network_name: String,
        owner_scope: String,
        memory_name: String,
        bandwidth: VisualizationSignedExpression,
        map: VisualizationAffineMap,
    },
}

impl DocumentBuilder {
    fn collect_scope(
        &mut self,
        architecture: &Architecture,
        parent_scope: Option<String>,
        path: &[String],
        inherited_dimensions: &[Dimension],
    ) -> Result<(), VisualizationExportError> {
        let scope_id = stable_id("scope", path);
        self.register_id(&scope_id)?;
        self.scope_parents
            .insert(scope_id.clone(), parent_scope.clone());
        self.scopes.push(VisualizationScope {
            id: scope_id.clone(),
            name: architecture.name.clone(),
            parent_scope,
            dimensions: architecture.dims.iter().map(dimension).collect(),
            replication_factor: dimension_product(&architecture.dims),
        });

        let mut effective_dimensions = inherited_dimensions.to_vec();
        effective_dimensions.extend(architecture.dims.iter().cloned());

        for memory in &architecture.memories {
            self.collect_memory(memory, &scope_id, path, &effective_dimensions)?;
        }
        for processor in &architecture.processors {
            self.collect_processor(processor, &scope_id, path)?;
        }
        for resource in &architecture.resources {
            self.collect_resource(resource, &scope_id, path)?;
        }
        for network in &architecture.networks {
            self.collect_network(network, &scope_id, path)?;
        }
        for child in &architecture.children {
            let mut child_path = path.to_vec();
            child_path.push(child.name.clone());
            self.collect_scope(
                child,
                Some(scope_id.clone()),
                &child_path,
                &effective_dimensions,
            )?;
        }
        Ok(())
    }

    fn collect_memory(
        &mut self,
        memory: &MemoryRegion,
        scope_id: &str,
        path: &[String],
        inherited_dimensions: &[Dimension],
    ) -> Result<(), VisualizationExportError> {
        let name = memory.name().unwrap_or("unnamed_memory").to_string();
        let mut memory_path = path.to_vec();
        memory_path.push(format!("memory:{name}"));
        let id = stable_id("memory", &memory_path);
        self.register_id(&id)?;
        let mut dimensions: Vec<VisualizationDimension> =
            inherited_dimensions.iter().map(dimension).collect();
        dimensions.extend(memory_dimensions(memory));
        self.components.push(VisualizationComponent::Memory {
            id,
            scope: scope_id.to_string(),
            name,
            dimensions,
            region: memory_region(memory),
            total_size_bytes: memory.total_size_bytes(),
        });
        Ok(())
    }

    fn collect_processor(
        &mut self,
        processor: &Processor,
        scope_id: &str,
        path: &[String],
    ) -> Result<(), VisualizationExportError> {
        let name = processor.name.as_deref().unwrap_or("unnamed_processor");
        let mut processor_path = path.to_vec();
        processor_path.push(format!("processor:{name}"));
        let id = stable_id("processor", &processor_path);
        self.register_id(&id)?;
        let functions = processor
            .functionality
            .functions
            .iter()
            .map(|function| function.name.clone())
            .collect();
        let component = if processor.effect == DataEffect::Preserve {
            VisualizationComponent::DataMover {
                id: id.clone(),
                scope: scope_id.to_string(),
                name: name.to_string(),
                effect: data_effect(&processor.effect),
                functions,
            }
        } else {
            VisualizationComponent::Processor {
                id: id.clone(),
                scope: scope_id.to_string(),
                name: name.to_string(),
                effect: data_effect(&processor.effect),
                functions,
            }
        };
        self.components.push(component);

        if let Some(source) = &processor.source {
            self.references.push(PendingReference::ProcessorMemory {
                owner_id: id.clone(),
                owner_name: name.to_string(),
                owner_scope: scope_id.to_string(),
                memory_name: source.name.clone(),
                kind: VisualizationRelationshipKind::Read,
            });
        }
        if let Some(destination) = &processor.destination {
            self.references.push(PendingReference::ProcessorMemory {
                owner_id: id.clone(),
                owner_name: name.to_string(),
                owner_scope: scope_id.to_string(),
                memory_name: destination.name.clone(),
                kind: VisualizationRelationshipKind::Write,
            });
        }
        for resource in &processor.resources {
            self.references.push(PendingReference::ProcessorResource {
                owner_id: id.clone(),
                owner_name: name.to_string(),
                owner_scope: scope_id.to_string(),
                resource_name: resource.id().as_str().to_string(),
            });
        }
        Ok(())
    }

    fn collect_resource(
        &mut self,
        resource: &Resource,
        scope_id: &str,
        path: &[String],
    ) -> Result<(), VisualizationExportError> {
        let name = resource.id().as_str();
        let mut resource_path = path.to_vec();
        resource_path.push(format!("resource:{name}"));
        let id = stable_id("resource", &resource_path);
        self.register_id(&id)?;
        let (resource_kind, capacity) = match resource {
            Resource::Exclusive { .. } => (VisualizationResourceKind::Exclusive, None),
            Resource::Quantitative { capacity, .. } => {
                (VisualizationResourceKind::Quantitative, Some(*capacity))
            }
        };
        self.components.push(VisualizationComponent::Resource {
            id,
            scope: scope_id.to_string(),
            name: name.to_string(),
            resource_kind,
            capacity,
        });
        Ok(())
    }

    fn collect_network(
        &mut self,
        network: &ScaleOutNetwork,
        scope_id: &str,
        path: &[String],
    ) -> Result<(), VisualizationExportError> {
        let name = network.name();
        let mut network_path = path.to_vec();
        network_path.push(format!("network:{name}"));
        let id = stable_id("network", &network_path);
        self.register_id(&id)?;
        let links = network
            .mesh_links()
            .iter()
            .enumerate()
            .map(|(index, link)| VisualizationNetworkLink {
                id: stable_id(
                    "network-link",
                    &[
                        id.clone(),
                        format!("{}:{index}", link.name.as_deref().unwrap_or("link")),
                    ],
                ),
                name: link.name.clone().unwrap_or_else(|| format!("link_{index}")),
                map: affine_map(&link.map),
            })
            .collect();
        self.components.push(VisualizationComponent::Network {
            id: id.clone(),
            scope: scope_id.to_string(),
            name: name.to_string(),
            network_kind: VisualizationNetworkKind::Mesh,
            dimensions: network.dimensions().iter().map(dimension).collect(),
            bandwidth: signed_expression(network.bandwidth()),
            latency: network.latency().map(signed_expression),
            links,
        });
        self.references.push(PendingReference::NetworkMemory {
            network_id: id,
            network_name: name.to_string(),
            owner_scope: scope_id.to_string(),
            memory_name: network
                .region()
                .name()
                .unwrap_or("unnamed_memory")
                .to_string(),
            bandwidth: signed_expression(&network.io().link_bandwidth),
            map: affine_map(&network.io().map),
        });
        Ok(())
    }

    fn resolve_relationships(&mut self) -> Result<(), VisualizationExportError> {
        let memory_index = self.component_index("memory");
        let resource_index = self.component_index("resource");
        let pending = std::mem::take(&mut self.references);
        for reference in pending {
            match reference {
                PendingReference::ProcessorMemory {
                    owner_id,
                    owner_name,
                    owner_scope,
                    memory_name,
                    kind,
                } => {
                    let memory_id = self.resolve_component(
                        "memory",
                        &memory_name,
                        &owner_name,
                        &owner_scope,
                        &memory_index,
                    )?;
                    let (source, target, label) = match kind {
                        VisualizationRelationshipKind::Read => (memory_id, owner_id, "read"),
                        VisualizationRelationshipKind::Write => (owner_id, memory_id, "write"),
                        _ => unreachable!(),
                    };
                    self.push_relationship(kind, source, target, label, None, None)?;
                }
                PendingReference::ProcessorResource {
                    owner_id,
                    owner_name,
                    owner_scope,
                    resource_name,
                } => {
                    let resource_id = self.resolve_component(
                        "resource",
                        &resource_name,
                        &owner_name,
                        &owner_scope,
                        &resource_index,
                    )?;
                    self.push_relationship(
                        VisualizationRelationshipKind::Requires,
                        owner_id,
                        resource_id,
                        "requires",
                        None,
                        None,
                    )?;
                }
                PendingReference::NetworkMemory {
                    network_id,
                    network_name,
                    owner_scope,
                    memory_name,
                    bandwidth,
                    map,
                } => {
                    let memory_id = self.resolve_component(
                        "memory",
                        &memory_name,
                        &network_name,
                        &owner_scope,
                        &memory_index,
                    )?;
                    self.push_relationship(
                        VisualizationRelationshipKind::NetworkAttachment,
                        network_id,
                        memory_id,
                        "attaches",
                        Some(bandwidth),
                        Some(map),
                    )?;
                }
            }
        }
        Ok(())
    }

    fn component_index(&self, kind: &'static str) -> BTreeMap<String, Vec<(String, String)>> {
        let mut index: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();
        for component in &self.components {
            let matches = matches!(
                (kind, component),
                ("memory", VisualizationComponent::Memory { .. })
                    | ("resource", VisualizationComponent::Resource { .. })
            );
            if !matches {
                continue;
            }
            index
                .entry(component.name().to_string())
                .or_default()
                .push((component.scope().to_string(), component.id().to_string()));
            if kind == "memory" && !component.name().starts_with("array_") {
                index
                    .entry(format!("array_{}", component.name()))
                    .or_default()
                    .push((component.scope().to_string(), component.id().to_string()));
            }
        }
        index
    }

    fn resolve_component(
        &self,
        kind: &'static str,
        name: &str,
        owner: &str,
        owner_scope: &str,
        index: &BTreeMap<String, Vec<(String, String)>>,
    ) -> Result<String, VisualizationExportError> {
        let Some(candidates) = index.get(name) else {
            return Err(VisualizationExportError::MissingReference {
                kind,
                name: name.to_string(),
                owner: owner.to_string(),
            });
        };
        let mut current_scope = Some(owner_scope.to_string());
        while let Some(scope) = current_scope {
            if let Some((_, id)) = candidates
                .iter()
                .find(|(candidate_scope, _)| candidate_scope == &scope)
            {
                return Ok(id.clone());
            }
            current_scope = self.scope_parents.get(&scope).cloned().flatten();
        }
        if candidates.len() == 1 {
            return Ok(candidates[0].1.clone());
        }
        Err(VisualizationExportError::AmbiguousReference {
            kind,
            name: name.to_string(),
            owner: owner.to_string(),
            candidates: candidates.iter().map(|(_, id)| id.clone()).collect(),
        })
    }

    fn push_relationship(
        &mut self,
        kind: VisualizationRelationshipKind,
        source: String,
        target: String,
        label: &str,
        bandwidth: Option<VisualizationSignedExpression>,
        map: Option<VisualizationAffineMap>,
    ) -> Result<(), VisualizationExportError> {
        let kind_name = match kind {
            VisualizationRelationshipKind::Read => "read",
            VisualizationRelationshipKind::Write => "write",
            VisualizationRelationshipKind::Requires => "requires",
            VisualizationRelationshipKind::NetworkAttachment => "network-attachment",
        };
        let id = stable_id(
            "relationship",
            &[kind_name.to_string(), source.clone(), target.clone()],
        );
        if self
            .relationships
            .iter()
            .any(|relationship| relationship.id == id)
        {
            return Ok(());
        }
        self.register_id(&id)?;
        self.relationships.push(VisualizationRelationship {
            id,
            kind,
            source,
            target,
            label: label.to_string(),
            bandwidth,
            map,
        });
        Ok(())
    }

    fn register_id(&mut self, id: &str) -> Result<(), VisualizationExportError> {
        if !self.ids.insert(id.to_string()) {
            return Err(VisualizationExportError::DuplicateId(id.to_string()));
        }
        Ok(())
    }
}

fn data_effect(effect: &DataEffect) -> VisualizationDataEffect {
    match effect {
        DataEffect::Preserve => VisualizationDataEffect::Preserve,
        DataEffect::Transform => VisualizationDataEffect::Transform,
        DataEffect::Reduce => VisualizationDataEffect::Reduce,
        DataEffect::Accumulate => VisualizationDataEffect::Accumulate,
    }
}

fn memory_dimensions(memory: &MemoryRegion) -> Vec<VisualizationDimension> {
    match memory {
        MemoryRegion::Bank(_) => Vec::new(),
        MemoryRegion::Array {
            dims, sub_regions, ..
        } => {
            let mut result: Vec<_> = dims.iter().map(dimension).collect();
            result.extend(memory_dimensions(sub_regions));
            result
        }
    }
}

fn memory_region(memory: &MemoryRegion) -> VisualizationMemoryRegion {
    match memory {
        MemoryRegion::Bank(bank) => VisualizationMemoryRegion::Bank {
            name: bank.name.clone(),
            capacity: unsigned_expression(&bank.capacity_bytes),
            block_size: bank.block_size.as_ref().map(unsigned_expression),
            total_size_bytes: memory.total_size_bytes(),
        },
        MemoryRegion::Array {
            name,
            dims,
            sub_regions,
        } => VisualizationMemoryRegion::Array {
            name: name.clone(),
            dimensions: dims.iter().map(dimension).collect(),
            element: Box::new(memory_region(sub_regions)),
            total_size_bytes: memory.total_size_bytes(),
        },
    }
}

fn dimension(value: &Dimension) -> VisualizationDimension {
    VisualizationDimension {
        name: value.name.0.clone(),
        size: unsigned_expression(&value.size),
    }
}

fn unsigned_expression(value: &SizeExpr) -> VisualizationUnsignedExpression {
    VisualizationUnsignedExpression {
        text: value.to_string(),
        constant: value.as_const(),
    }
}

fn signed_expression(value: &Expr) -> VisualizationSignedExpression {
    VisualizationSignedExpression {
        text: value.to_string(),
        constant: value.eval_const(),
    }
}

fn affine_map(value: &AffineMap) -> VisualizationAffineMap {
    VisualizationAffineMap {
        source_dimensions: value.src_dims.iter().map(dimension).collect(),
        target_dimensions: value.dst_dims.iter().map(dimension).collect(),
        expressions: value.exprs.iter().map(affine_expression).collect(),
    }
}

fn affine_expression(value: &AffineExpr) -> String {
    match value {
        AffineExpr::Var(dimension) => dimension.name.0.clone(),
        AffineExpr::Sym(symbol) => symbol.0.clone(),
        AffineExpr::Const(value) => value.to_string(),
        AffineExpr::Add(left, right) => {
            format!(
                "({} + {})",
                affine_expression(left),
                affine_expression(right)
            )
        }
        AffineExpr::MulConst(value, expression) => {
            format!("({value} * {})", affine_expression(expression))
        }
        AffineExpr::Mod(left, right) => {
            format!(
                "({} mod {})",
                affine_expression(left),
                affine_expression(right)
            )
        }
        AffineExpr::CeilDiv(left, right) => format!(
            "ceildiv({}, {})",
            affine_expression(left),
            affine_expression(right)
        ),
    }
}

fn dimension_product(dimensions: &[Dimension]) -> Option<u64> {
    dimensions.iter().try_fold(1u64, |product, dimension| {
        product.checked_mul(dimension.size.as_const()?)
    })
}

fn stable_id(kind: &str, path: &[String]) -> String {
    let raw = format!("{kind}:{}", path.join("/"));
    let readable = path
        .last()
        .map(|part| slug(part))
        .filter(|part| !part.is_empty())
        .unwrap_or_else(|| "item".to_string());
    format!(
        "{}-{}-{:016x}",
        slug(kind),
        readable,
        fnv1a64(raw.as_bytes())
    )
}

fn slug(value: &str) -> String {
    let mut result = String::new();
    let mut previous_dash = false;
    for character in value.chars() {
        if character.is_ascii_alphanumeric() {
            result.push(character.to_ascii_lowercase());
            previous_dash = false;
        } else if !previous_dash && !result.is_empty() {
            result.push('-');
            previous_dash = true;
        }
    }
    while result.ends_with('-') {
        result.pop();
    }
    if result.is_empty() || !result.starts_with(|character: char| character.is_ascii_lowercase()) {
        result.insert_str(0, "item-");
    }
    result
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::{MemoryRegion, MemoryRegionRef, MeshNetworkInterface, Processor, Resource};

    #[test]
    fn exports_stable_normalized_document() {
        let memory = MemoryRegion::leaf_concrete(16, 64).with_name("L1");
        let mut processor = Processor::new("lane");
        processor.source = Some(MemoryRegionRef::new("L1"));
        processor.destination = Some(MemoryRegionRef::new("L1"));
        let architecture = Architecture::scope("core")
            .with_memory(memory)
            .with_processor(processor);

        let first = architecture_to_visualization_document(&architecture).unwrap();
        let second = architecture_to_visualization_document(&architecture).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.schema_version, VISUALIZATION_SCHEMA_VERSION);
        assert_eq!(first.scopes.len(), 1);
        assert_eq!(first.scopes[0].replication_factor, Some(1));
        assert!(first.components.iter().any(
            |component| matches!(component, VisualizationComponent::Memory { name, .. } if name == "L1")
        ));
        assert!(
            first
                .relationships
                .iter()
                .any(|relationship| { relationship.kind == VisualizationRelationshipKind::Read })
        );
        assert!(
            first
                .relationships
                .iter()
                .any(|relationship| { relationship.kind == VisualizationRelationshipKind::Write })
        );
    }

    #[test]
    fn yaml_round_trips_through_public_document() {
        let architecture = Architecture::scope("empty");
        let yaml = architecture_to_visualization_yaml(&architecture).unwrap();
        let decoded: VisualizationDocumentV1 = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(decoded.schema_version, VISUALIZATION_SCHEMA_VERSION);
        assert_eq!(decoded.architecture.name, "empty");
    }

    #[test]
    fn stable_ids_do_not_depend_on_collection_order() {
        let left = stable_id("memory", &["system".to_string(), "memory:L1".to_string()]);
        let right = stable_id("memory", &["system".to_string(), "memory:L1".to_string()]);
        assert_eq!(left, right);
        assert!(left.starts_with("memory-memory-l1-"));
    }

    #[test]
    fn exports_network_topology_and_memory_attachment() {
        let dimensions = Dimension::new_int("x", 4);
        let map = AffineMap::identity(dimensions.as_slice());
        let memory = MemoryRegion::leaf_concrete(16, 64)
            .scale(dimensions.as_slice())
            .with_name("L1");
        let interface = MeshNetworkInterface::new(map.clone(), Expr::Const(32));
        let network = ScaleOutNetwork::mesh("noc")
            .mem_region(&memory)
            .map(&map)
            .io(&interface)
            .link_bandwidth(64)
            .build();
        let architecture = Architecture::scope("system")
            .with_memory(memory)
            .with_network(network);

        let document = architecture_to_visualization_document(&architecture).unwrap();
        assert!(document.components.iter().any(|component| {
            matches!(
                component,
                VisualizationComponent::Network { name, links, .. }
                    if name == "noc" && links.len() == 1
            )
        }));
        assert!(document.relationships.iter().any(|relationship| {
            relationship.kind == VisualizationRelationshipKind::NetworkAttachment
                && relationship
                    .bandwidth
                    .as_ref()
                    .and_then(|value| value.constant)
                    == Some(32)
                && relationship.map.is_some()
        }));
    }

    #[test]
    fn exports_child_resource_once_and_links_its_processor() {
        let mut processor = Processor::new("matrix_lane");
        processor.resources.push(Resource::exclusive("matrix_lane"));
        let architecture = Architecture::scope("system")
            .with_child(Architecture::scope("mesh").with_processor(processor));

        let document = architecture_to_visualization_document(&architecture).unwrap();
        let resources = document
            .components
            .iter()
            .filter_map(|component| match component {
                VisualizationComponent::Resource {
                    id, scope, name, ..
                } if name == "matrix_lane" => Some((id, scope)),
                _ => None,
            })
            .collect::<Vec<_>>();

        assert_eq!(resources.len(), 1);
        let (resource_id, resource_scope) = resources[0];
        let mesh_scope = document
            .scopes
            .iter()
            .find(|scope| scope.name == "mesh")
            .expect("mesh scope should be exported");
        assert_eq!(resource_scope, &mesh_scope.id);
        assert!(document.relationships.iter().any(|relationship| {
            relationship.kind == VisualizationRelationshipKind::Requires
                && relationship.target == *resource_id
        }));
    }
}
