use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::arch::{Architecture, Axis, MemoryDefinition, MemoryEndpoint, ProcessorType, Resource};
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
        capacity: Option<u64>,
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
    Unsupported {
        object: String,
        reason: String,
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
            Self::Unsupported { object, reason } => {
                write!(f, "cannot visualize {object}: {reason}")
            }
            Self::Yaml(message) => write!(f, "visualization YAML serialization failed: {message}"),
        }
    }
}

impl std::error::Error for VisualizationExportError {}

pub fn architecture_to_visualization_document(
    architecture: &Architecture,
) -> Result<VisualizationDocumentV1, VisualizationExportError> {
    DocumentBuilder::default().build(architecture)
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
    memory_ids: BTreeMap<String, String>,
    resource_ids: BTreeMap<String, String>,
}

struct ScopePlan {
    root_id: String,
    scopes: Vec<VisualizationScope>,
    domains: BTreeMap<String, Vec<Axis>>,
    memory_owners: BTreeMap<String, String>,
    processor_owners: BTreeMap<String, String>,
    network_owners: BTreeMap<String, String>,
    resource_owners: BTreeMap<String, String>,
    inferred_domains: Vec<(Vec<Axis>, String)>,
    explicit: bool,
}

impl ScopePlan {
    fn new(architecture: &Architecture) -> Self {
        let root_id = stable_id("scope", &[architecture.name().to_string()]);
        let mut scopes = vec![VisualizationScope {
            id: root_id.clone(),
            name: architecture.name().to_string(),
            parent_scope: None,
            dimensions: Vec::new(),
            replication_factor: Some(1),
        }];
        let mut domains = BTreeMap::from([(root_id.clone(), Vec::new())]);
        let mut memory_owners = BTreeMap::new();
        let mut processor_owners = BTreeMap::new();
        let mut network_owners = BTreeMap::new();
        let mut resource_owners = BTreeMap::new();
        let explicit = !architecture.scopes().is_empty();
        let mut inferred_domains = Vec::new();

        if explicit {
            let ids = architecture
                .scopes()
                .iter()
                .map(|scope| {
                    (
                        scope.name().to_string(),
                        stable_id(
                            "scope",
                            &[architecture.name().to_string(), scope.name().to_string()],
                        ),
                    )
                })
                .collect::<BTreeMap<_, _>>();
            let full_domains = architecture
                .scopes()
                .iter()
                .map(|scope| {
                    (
                        scope.name().to_string(),
                        scope
                            .axes()
                            .iter()
                            .map(|name| {
                                architecture
                                    .axis(name)
                                    .expect("validated scope axis")
                                    .clone()
                            })
                            .collect::<Vec<_>>(),
                    )
                })
                .collect::<BTreeMap<_, _>>();

            for scope in architecture.scopes() {
                let id = ids[scope.name()].clone();
                let parent_id = scope
                    .parent()
                    .map(|parent| ids[parent].clone())
                    .unwrap_or_else(|| root_id.clone());
                let parent_rank = scope
                    .parent()
                    .map(|parent| full_domains[parent].len())
                    .unwrap_or(0);
                let domain = full_domains[scope.name()].clone();
                scopes.push(VisualizationScope {
                    id: id.clone(),
                    name: scope.name().to_string(),
                    parent_scope: Some(parent_id),
                    dimensions: domain[parent_rank..].iter().map(dimension).collect(),
                    replication_factor: dimension_product(&domain[parent_rank..]),
                });
                domains.insert(id.clone(), domain);
                for name in scope.memories() {
                    memory_owners.insert(name.clone(), id.clone());
                }
                for name in scope.processors() {
                    processor_owners.insert(name.clone(), id.clone());
                }
                for name in scope.networks() {
                    network_owners.insert(name.clone(), id.clone());
                }
                for name in scope.resources() {
                    resource_owners.insert(name.clone(), id.clone());
                }
            }
        } else {
            let mut unique_domains = architecture
                .processors()
                .iter()
                .map(|processor| processor.axes().to_vec())
                .filter(|domain| !domain.is_empty())
                .collect::<Vec<_>>();
            unique_domains.sort_by(compare_domains);
            unique_domains.dedup();

            for domain in &unique_domains {
                let name = domain.iter().map(Axis::name).collect::<Vec<_>>().join("_");
                let id = stable_id(
                    "scope",
                    &[architecture.name().to_string(), format!("domain:{name}")],
                );
                inferred_domains.push((domain.clone(), id.clone()));
                domains.insert(id, domain.clone());
            }
            for (domain, id) in &inferred_domains {
                let parent = inferred_domains
                    .iter()
                    .filter(|(candidate, _)| {
                        candidate.len() < domain.len() && domain_is_prefix(candidate, domain)
                    })
                    .max_by_key(|(candidate, _)| candidate.len());
                let (parent_id, parent_rank) = parent
                    .map(|(candidate, id)| (id.clone(), candidate.len()))
                    .unwrap_or_else(|| (root_id.clone(), 0));
                scopes.push(VisualizationScope {
                    id: id.clone(),
                    name: domain.iter().map(Axis::name).collect::<Vec<_>>().join("_"),
                    parent_scope: Some(parent_id),
                    dimensions: domain[parent_rank..].iter().map(dimension).collect(),
                    replication_factor: dimension_product(&domain[parent_rank..]),
                });
            }
        }

        Self {
            root_id,
            scopes,
            domains,
            memory_owners,
            processor_owners,
            network_owners,
            resource_owners,
            inferred_domains,
            explicit,
        }
    }

    fn memory_owner(&self, name: &str, domain: &[Axis]) -> &str {
        self.owner(&self.memory_owners, name, domain)
    }

    fn processor_owner(&self, name: &str, domain: &[Axis]) -> &str {
        self.owner(&self.processor_owners, name, domain)
    }

    fn network_owner(&self, name: &str, domain: &[Axis]) -> &str {
        self.owner(&self.network_owners, name, domain)
    }

    fn resource_owner(&self, name: &str, domain: &[Axis]) -> &str {
        self.owner(&self.resource_owners, name, domain)
    }

    fn owner<'a>(
        &'a self,
        explicit_owners: &'a BTreeMap<String, String>,
        name: &str,
        domain: &[Axis],
    ) -> &'a str {
        if let Some(owner) = explicit_owners.get(name) {
            return owner;
        }
        if !self.explicit {
            if let Some((_, owner)) = self
                .inferred_domains
                .iter()
                .find(|(candidate, _)| candidate == domain)
            {
                return owner;
            }
        }
        &self.root_id
    }

    fn domain(&self, scope: &str) -> &[Axis] {
        &self.domains[scope]
    }
}

impl DocumentBuilder {
    fn build(
        mut self,
        architecture: &Architecture,
    ) -> Result<VisualizationDocumentV1, VisualizationExportError> {
        let plan = ScopePlan::new(architecture);
        for scope in &plan.scopes {
            self.register_id(&scope.id)?;
        }
        self.scopes = plan.scopes.clone();

        for memory in architecture.memories() {
            let definition = architecture
                .memory_definition(memory)
                .expect("validated memory definition");
            let scope = plan.memory_owner(memory.name(), memory.axes());
            let id = stable_id(
                "memory",
                &[architecture.name().to_string(), memory.name().to_string()],
            );
            self.register_id(&id)?;
            self.memory_ids
                .insert(memory.name().to_string(), id.clone());
            let owner_rank = plan.domain(scope).len().min(memory.axes().len());
            let region = memory_region(memory.name(), definition, &memory.axes()[owner_rank..]);
            self.components.push(VisualizationComponent::Memory {
                id,
                scope: scope.to_string(),
                name: memory.name().to_string(),
                dimensions: memory_dimensions(memory.axes(), definition),
                total_size_bytes: memory_region_size(&region),
                region,
            });
        }

        for resource in architecture.resources() {
            let scope = plan.resource_owner(resource.name(), resource.axes());
            let display_name = resource
                .name()
                .split_once('.')
                .filter(|(owner, _)| architecture.processor_array(owner).is_some())
                .map_or(resource.name(), |(_, local)| local);
            self.collect_resource(
                resource.name(),
                display_name,
                resource,
                scope,
                &[architecture.name().to_string()],
            )?;
        }

        for processor in architecture.processors() {
            let definition = architecture
                .processor_definition(processor.definition_name())
                .expect("validated processor definition");
            let processor_type = definition.processor_type().ok_or_else(|| {
                VisualizationExportError::Unsupported {
                    object: format!("processor '{}'", processor.name()),
                    reason: "its definition has no processor type".into(),
                }
            })?;
            let scope = plan.processor_owner(processor.name(), processor.axes());
            let id = stable_id(
                "processor",
                &[
                    architecture.name().to_string(),
                    processor.name().to_string(),
                ],
            );
            self.register_id(&id)?;
            let functions = definition
                .operations()
                .iter()
                .map(|operation| operation.func.name.clone())
                .collect();
            let component = match processor_type {
                ProcessorType::Compute => VisualizationComponent::Processor {
                    id: id.clone(),
                    scope: scope.to_string(),
                    name: processor.name().to_string(),
                    effect: VisualizationDataEffect::Transform,
                    functions,
                },
                ProcessorType::DataMover => VisualizationComponent::DataMover {
                    id: id.clone(),
                    scope: scope.to_string(),
                    name: processor.name().to_string(),
                    effect: VisualizationDataEffect::Preserve,
                    functions,
                },
            };
            self.components.push(component);

            for endpoint in &processor.connection().inputs {
                let memory = resolve_endpoint_memory(architecture, endpoint);
                let memory_id = self.memory_ids.get(memory).ok_or_else(|| {
                    VisualizationExportError::MissingReference {
                        kind: "memory",
                        name: memory.to_string(),
                        owner: processor.name().to_string(),
                    }
                })?;
                self.push_relationship(
                    VisualizationRelationshipKind::Read,
                    memory_id.clone(),
                    id.clone(),
                    "read",
                    None,
                    None,
                )?;
            }
            for endpoint in &processor.connection().outputs {
                let memory = resolve_endpoint_memory(architecture, endpoint);
                let memory_id = self.memory_ids.get(memory).ok_or_else(|| {
                    VisualizationExportError::MissingReference {
                        kind: "memory",
                        name: memory.to_string(),
                        owner: processor.name().to_string(),
                    }
                })?;
                self.push_relationship(
                    VisualizationRelationshipKind::Write,
                    id.clone(),
                    memory_id.clone(),
                    "write",
                    None,
                    None,
                )?;
            }
            for resource in processor.resources() {
                let resource_id = self.resource_ids.get(resource.name()).ok_or_else(|| {
                    VisualizationExportError::MissingReference {
                        kind: "resource",
                        name: resource.name().to_string(),
                        owner: processor.name().to_string(),
                    }
                })?;
                self.push_relationship(
                    VisualizationRelationshipKind::Requires,
                    id.clone(),
                    resource_id.clone(),
                    "requires",
                    None,
                    None,
                )?;
            }
        }

        for network in architecture.networks() {
            let scope = plan.network_owner(&network.name, &network.dimensions);
            let network_id = stable_id(
                "network",
                &[architecture.name().to_string(), network.name.clone()],
            );
            self.register_id(&network_id)?;

            for resource in &network.resources {
                let display_name = format!("{}.{}", network.name, resource.name());
                let resource_id = self.collect_resource(
                    &display_name,
                    resource.name(),
                    resource,
                    scope,
                    &[
                        architecture.name().to_string(),
                        format!("network:{}", network.name),
                    ],
                )?;
                self.push_relationship(
                    VisualizationRelationshipKind::Requires,
                    network_id.clone(),
                    resource_id,
                    "requires",
                    None,
                    None,
                )?;
            }

            let links = network
                .links
                .iter()
                .map(|link| VisualizationNetworkLink {
                    id: stable_id("network-link", &[network_id.clone(), link.name.clone()]),
                    name: link.name.clone(),
                    map: affine_map(&link.map),
                })
                .collect();
            self.components.push(VisualizationComponent::Network {
                id: network_id.clone(),
                scope: scope.to_string(),
                name: network.name.clone(),
                network_kind: VisualizationNetworkKind::Mesh,
                dimensions: network.dimensions.iter().map(dimension).collect(),
                bandwidth: summarize_expressions(
                    network.links.iter().map(|link| &link.bandwidth),
                    "unspecified",
                ),
                latency: summarize_optional_expressions(
                    network.links.iter().map(|link| link.latency.as_ref()),
                ),
                links,
            });

            for interface in &network.interfaces {
                let memory = resolve_endpoint_memory(architecture, &interface.endpoint);
                let memory_id = self.memory_ids.get(memory).ok_or_else(|| {
                    VisualizationExportError::MissingReference {
                        kind: "memory",
                        name: memory.to_string(),
                        owner: network.name.clone(),
                    }
                })?;
                let bandwidth = summarize_expressions(
                    [
                        interface.injection_bandwidth.as_ref(),
                        interface.ejection_bandwidth.as_ref(),
                    ]
                    .into_iter()
                    .flatten(),
                    "unspecified",
                );
                self.push_relationship(
                    VisualizationRelationshipKind::NetworkAttachment,
                    network_id.clone(),
                    memory_id.clone(),
                    &interface.name,
                    Some(bandwidth),
                    None,
                )?;
            }
        }

        self.scopes.sort_by(|left, right| left.id.cmp(&right.id));
        self.components
            .sort_by(|left, right| left.id().cmp(right.id()));
        self.relationships
            .sort_by(|left, right| left.id.cmp(&right.id));

        let root_path = vec![architecture.name().to_string()];
        Ok(VisualizationDocumentV1 {
            schema_version: VISUALIZATION_SCHEMA_VERSION.to_string(),
            architecture: VisualizationArchitecture {
                id: stable_id("architecture", &root_path),
                name: architecture.name().to_string(),
                root_scope: plan.root_id,
            },
            scopes: self.scopes,
            components: self.components,
            relationships: self.relationships,
        })
    }

    fn collect_resource(
        &mut self,
        canonical_name: &str,
        display_name: &str,
        resource: &Resource,
        scope: &str,
        path: &[String],
    ) -> Result<String, VisualizationExportError> {
        let mut resource_path = path.to_vec();
        resource_path.push(format!("resource:{canonical_name}"));
        let id = stable_id("resource", &resource_path);
        self.register_id(&id)?;
        self.components.push(VisualizationComponent::Resource {
            id: id.clone(),
            scope: scope.to_string(),
            name: display_name.to_string(),
            resource_kind: if resource.capacity().is_some() {
                VisualizationResourceKind::Quantitative
            } else {
                VisualizationResourceKind::Exclusive
            },
            capacity: resource.capacity(),
        });
        if path.len() == 1 {
            self.resource_ids
                .insert(canonical_name.to_string(), id.clone());
        }
        Ok(id)
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

fn compare_domains(left: &Vec<Axis>, right: &Vec<Axis>) -> std::cmp::Ordering {
    left.len().cmp(&right.len()).then_with(|| {
        left.iter()
            .map(Axis::name)
            .cmp(right.iter().map(Axis::name))
    })
}

fn domain_is_prefix(prefix: &[Axis], domain: &[Axis]) -> bool {
    prefix.len() <= domain.len() && prefix.iter().zip(domain).all(|(left, right)| left == right)
}

fn resolve_endpoint_memory<'a>(
    architecture: &'a Architecture,
    endpoint: &'a MemoryEndpoint,
) -> &'a str {
    architecture
        .memory_alias(&endpoint.memory)
        .map(|alias| alias.endpoint.memory.as_str())
        .unwrap_or(&endpoint.memory)
}

fn memory_dimensions(axes: &[Axis], definition: &MemoryDefinition) -> Vec<VisualizationDimension> {
    let mut dimensions = axes.iter().map(dimension).collect::<Vec<_>>();
    if let Some(banking) = &definition.banking {
        dimensions.push(VisualizationDimension {
            name: "bank".into(),
            size: unsigned_constant(banking.banks),
        });
    }
    dimensions
}

fn memory_region(
    name: &str,
    definition: &MemoryDefinition,
    placement_axes: &[Axis],
) -> VisualizationMemoryRegion {
    let banks = definition
        .banking
        .as_ref()
        .map_or(1, |banking| banking.banks);
    let bank_capacity = definition.capacity / banks;
    let mut region = VisualizationMemoryRegion::Bank {
        name: Some(format!("{name}_bank")),
        capacity: unsigned_constant(bank_capacity),
        block_size: Some(unsigned_constant(definition.word_size)),
        total_size_bytes: Some(bank_capacity),
    };
    if banks > 1 {
        region = VisualizationMemoryRegion::Array {
            name: Some(format!("{name}_banks")),
            dimensions: vec![VisualizationDimension {
                name: "bank".into(),
                size: unsigned_constant(banks),
            }],
            element: Box::new(region),
            total_size_bytes: Some(definition.capacity),
        };
    }
    if !placement_axes.is_empty() {
        let instances = dimension_product(placement_axes);
        let total = instances.and_then(|count| definition.capacity.checked_mul(count));
        region = VisualizationMemoryRegion::Array {
            name: Some(name.to_string()),
            dimensions: placement_axes.iter().map(dimension).collect(),
            element: Box::new(region),
            total_size_bytes: total,
        };
    }
    region
}

fn memory_region_size(region: &VisualizationMemoryRegion) -> Option<u64> {
    match region {
        VisualizationMemoryRegion::Bank {
            total_size_bytes, ..
        }
        | VisualizationMemoryRegion::Array {
            total_size_bytes, ..
        } => *total_size_bytes,
    }
}

fn dimension(axis: &Axis) -> VisualizationDimension {
    VisualizationDimension {
        name: axis.name().to_string(),
        size: unsigned_constant(axis.extent()),
    }
}

fn unsigned_constant(value: u64) -> VisualizationUnsignedExpression {
    VisualizationUnsignedExpression {
        text: value.to_string(),
        constant: Some(value),
    }
}

fn signed_expression(value: &Expr) -> VisualizationSignedExpression {
    VisualizationSignedExpression {
        text: value.to_string(),
        constant: value.eval_const(),
    }
}

fn summarize_expressions<'a>(
    values: impl IntoIterator<Item = &'a Expr>,
    empty: &str,
) -> VisualizationSignedExpression {
    let values = values.into_iter().collect::<Vec<_>>();
    let Some(first) = values.first() else {
        return VisualizationSignedExpression {
            text: empty.into(),
            constant: None,
        };
    };
    if values.iter().all(|value| *value == *first) {
        signed_expression(first)
    } else {
        VisualizationSignedExpression {
            text: "heterogeneous".into(),
            constant: None,
        }
    }
}

fn summarize_optional_expressions<'a>(
    values: impl IntoIterator<Item = Option<&'a Expr>>,
) -> Option<VisualizationSignedExpression> {
    let values = values.into_iter().flatten().collect::<Vec<_>>();
    (!values.is_empty()).then(|| summarize_expressions(values, "unspecified"))
}

fn affine_map(value: &AffineMap) -> VisualizationAffineMap {
    VisualizationAffineMap {
        source_dimensions: value.source_axes().iter().map(dimension).collect(),
        target_dimensions: value.target_axes().iter().map(dimension).collect(),
        expressions: value.expressions().iter().map(affine_expression).collect(),
    }
}

fn affine_expression(value: &AffineExpr) -> String {
    match value {
        AffineExpr::Constant(value) => value.to_string(),
        AffineExpr::Variable(name) => name.clone(),
        AffineExpr::Add(left, right) => format!(
            "({} + {})",
            affine_expression(left),
            affine_expression(right)
        ),
        AffineExpr::Sub(left, right) => format!(
            "({} - {})",
            affine_expression(left),
            affine_expression(right)
        ),
        AffineExpr::Mul(factor, expression) => {
            format!("({factor} * {})", affine_expression(expression))
        }
        AffineExpr::FloorDiv(expression, divisor) => {
            format!("floordiv({}, {divisor})", affine_expression(expression))
        }
        AffineExpr::CeilDiv(expression, divisor) => {
            format!("ceildiv({}, {divisor})", affine_expression(expression))
        }
        AffineExpr::Mod(expression, modulus) => {
            format!("({} mod {modulus})", affine_expression(expression))
        }
    }
}

fn dimension_product(dimensions: &[Axis]) -> Option<u64> {
    dimensions.iter().try_fold(1u64, |product, dimension| {
        product.checked_mul(dimension.extent())
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
    use crate::arch::{
        Connection, MemoryEndpoint, NetworkInterface, NetworkLink, NetworkTopology,
        ProcessorDefinition, ProcessorType,
    };

    #[test]
    fn exports_stable_normalized_document() {
        let architecture = Architecture::builder("core")
            .memory_definition(MemoryDefinition::new(
                "L1",
                std::iter::empty::<&str>(),
                1024,
                16,
            ))
            .place_memory("L1", std::iter::empty::<&str>())
            .processor_definition(
                ProcessorDefinition::new("lane", "", Vec::new()).with_type(ProcessorType::Compute),
            )
            .connect("lane", Connection::parse([], ["L1"], ["L1"]).unwrap())
            .build()
            .unwrap();

        let first = architecture_to_visualization_document(&architecture).unwrap();
        let second = architecture_to_visualization_document(&architecture).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.schema_version, VISUALIZATION_SCHEMA_VERSION);
        assert_eq!(first.scopes.len(), 1);
        assert!(first.components.iter().any(
            |component| matches!(component, VisualizationComponent::Memory { name, .. } if name == "L1")
        ));
        assert!(
            first
                .relationships
                .iter()
                .any(|relationship| relationship.kind == VisualizationRelationshipKind::Read)
        );
        assert!(
            first
                .relationships
                .iter()
                .any(|relationship| relationship.kind == VisualizationRelationshipKind::Write)
        );
    }

    #[test]
    fn yaml_round_trips_through_public_document() {
        let architecture = Architecture::builder("empty").build().unwrap();
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
        let axis = Axis::new("x", 4);
        let map = AffineMap::identity(std::slice::from_ref(&axis));
        let network = NetworkTopology::new("noc", vec![axis.clone()])
            .with_link(NetworkLink::new("east", map, Expr::Const(64)))
            .with_interface(
                NetworkInterface::new("l1", MemoryEndpoint::parse("L1[:]").unwrap())
                    .with_injection_bandwidth(Expr::Const(32)),
            );
        let architecture = Architecture::builder("system")
            .axis("x", 4)
            .memory_definition(MemoryDefinition::new("L1", ["x"], 1024, 16))
            .place_memory("L1", ["x"])
            .network(network)
            .build()
            .unwrap();

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
        }));
    }
}
