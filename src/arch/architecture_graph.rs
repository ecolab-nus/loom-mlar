use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;

use super::architecture::Architecture;
use super::memory::MemoryRegion;
use super::network::{ScaleOutNetwork, ScaleOutNetworkBindings};
use super::processor::{DataMover, Processor};
use super::resource::{Resource, ResourceId};

const NODE_ID_SUFFIX: &str = "::node";
const EDGE_ID_SUFFIX: &str = "::edge";

/// Numeric identifier for a router side (0-based).
pub type RouterSide = u32;

/// General router component: named node with numbered sides `0..num_sides`.
///
/// Connectivity between a router and other architecture nodes is expressed
/// through graph edges annotated with `ArchEdgeAttr::Side`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Router {
    pub name: String,
    pub num_sides: u32,
}

impl Router {
    pub fn new(name: impl Into<String>, num_sides: u32) -> Self {
        Self {
            name: name.into(),
            num_sides,
        }
    }

    pub fn side_count(&self) -> usize {
        self.num_sides as usize
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ArchNodeId(String);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ArchEdgeId(String);

/// Abstract node payload for heterogeneous architecture graph composition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArchNodeComponent {
    Architecture(Architecture),
    DataMover(DataMover),
    MemoryRegion(MemoryRegion),
    Router(Router),
}

/// Abstract architecture graph node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchNode {
    pub id: ArchNodeId,
    pub component: ArchNodeComponent,
}

/// Alias name for graph nodes.
pub type ArchGraphNode = ArchNode;

/// Typed attribute that can be attached to an architecture graph edge.
///
/// Each variant represents a different kind of metadata. An edge carries a
/// `Vec<ArchEdgeAttr>`, so it can hold zero or more attributes of any
/// combination of types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArchEdgeDirection {
    /// The edge direction is from source node to target node.
    Directional,
    /// The edge connects both directions.
    Bidirectional,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArchEdgeAttr {
    /// The router side this edge connects through.
    Side(RouterSide),
    /// Edge direction metadata.
    Direction(ArchEdgeDirection),
}

/// Edge between two architecture graph nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchEdge {
    pub id: ArchEdgeId,
    pub source: ArchNodeId,
    pub target: ArchNodeId,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub attrs: Vec<ArchEdgeAttr>,
}

/// Heterogeneous architecture composition using explicit nodes and edges.
///
/// An `ArchGraph` also owns a set of [`Resource`]s and a map recording which
/// nodes consume which resources.  Nodes that do **not** appear in the
/// resource map are treated as the sole consumer of an implicit private
/// resource — they never contend with other nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchGraph {
    pub name: String,
    pub nodes: Vec<ArchGraphNode>,
    pub edges: Vec<ArchEdge>,
    /// Resources available within the scope of this graph.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub resources: Vec<Resource>,
    /// Per-node resource associations.  Only nodes with contention need entries.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub resource_map: HashMap<ArchNodeId, Vec<ResourceId>>,
}

/// Builder for constructing an `ArchGraph`.
pub struct ArchGraphBuilder {
    graph: ArchGraph,
}

impl ArchNodeId {
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl fmt::Display for ArchNodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for ArchNodeId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for ArchNodeId {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

impl From<ArchNodeId> for String {
    fn from(value: ArchNodeId) -> Self {
        value.0
    }
}

impl ArchEdgeId {
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl fmt::Display for ArchEdgeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for ArchEdgeId {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for ArchEdgeId {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

impl From<ArchEdgeId> for String {
    fn from(value: ArchEdgeId) -> Self {
        value.0
    }
}

impl ArchNodeComponent {
    pub fn display_name(&self) -> Option<&str> {
        match self {
            ArchNodeComponent::Architecture(arch) => arch.name(),
            ArchNodeComponent::DataMover(mover) => mover.name.as_deref(),
            ArchNodeComponent::MemoryRegion(region) => region.name(),
            ArchNodeComponent::Router(router) => Some(router.name.as_str()),
        }
    }

    fn id_prefix(&self) -> &'static str {
        match self {
            ArchNodeComponent::Architecture(_) => "arch",
            ArchNodeComponent::DataMover(_) => "dm",
            ArchNodeComponent::MemoryRegion(_) => "mem",
            ArchNodeComponent::Router(_) => "router",
        }
    }
}

impl ArchNode {
    pub fn name(&self) -> Option<&str> {
        self.component.display_name()
    }

    pub fn from_component(component: ArchNodeComponent) -> Self {
        let base = component_node_base_id(&component);
        let id = ArchNodeId::from(format!("{base}{NODE_ID_SUFFIX}"));
        Self { id, component }
    }

    pub fn from_architecture(architecture: &Architecture) -> Self {
        Self::from_component(ArchNodeComponent::Architecture(architecture.clone()))
    }

    pub fn from_processor(proc: &Processor) -> Self {
        Self::from_architecture(&Architecture::Unit(proc.clone()))
    }

    pub fn from_data_mover(mover: &DataMover) -> Self {
        Self::from_component(ArchNodeComponent::DataMover(mover.clone()))
    }

    pub fn from_memory_region(region: &MemoryRegion) -> Self {
        Self::from_component(ArchNodeComponent::MemoryRegion(region.clone()))
    }

    pub fn from_router(router: &Router) -> Self {
        Self::from_component(ArchNodeComponent::Router(router.clone()))
    }
}

impl Default for ArchEdgeDirection {
    fn default() -> Self {
        Self::Directional
    }
}

impl ArchEdge {
    fn with_attrs(
        id: ArchEdgeId,
        source: ArchNodeId,
        target: ArchNodeId,
        attrs: Vec<ArchEdgeAttr>,
    ) -> Self {
        Self {
            id,
            source,
            target,
            attrs,
        }
    }

    /// Return the first `Side` attribute value, if any.
    pub fn side(&self) -> Option<RouterSide> {
        self.attrs.iter().find_map(|a| match a {
            ArchEdgeAttr::Side(s) => Some(*s),
            _ => None,
        })
    }

    /// Return the effective edge direction.
    ///
    /// Defaults to `Directional` when no explicit direction attribute exists.
    pub fn direction(&self) -> ArchEdgeDirection {
        self.attrs
            .iter()
            .find_map(|a| match a {
                ArchEdgeAttr::Direction(direction) => Some(*direction),
                _ => None,
            })
            .unwrap_or_default()
    }

    /// Return all attributes matching a predicate.
    pub fn attrs_where<F>(&self, predicate: F) -> Vec<&ArchEdgeAttr>
    where
        F: Fn(&ArchEdgeAttr) -> bool,
    {
        self.attrs.iter().filter(|a| predicate(a)).collect()
    }

    /// Add an attribute, returning `self` for builder-style chaining.
    pub fn with_attr(mut self, attr: ArchEdgeAttr) -> Self {
        self.attrs.push(attr);
        self
    }
}

impl ArchGraph {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            nodes: Vec::new(),
            edges: Vec::new(),
            resources: Vec::new(),
            resource_map: HashMap::new(),
        }
    }

    fn has_node_id(&self, id: &ArchNodeId) -> bool {
        self.nodes.iter().any(|n| n.id == *id)
    }

    fn has_node_id_str(&self, id: &str) -> bool {
        self.nodes.iter().any(|n| n.id.as_str() == id)
    }

    fn has_edge_id(&self, id: &ArchEdgeId) -> bool {
        self.edges.iter().any(|e| e.id == *id)
    }

    fn has_edge_id_str(&self, id: &str) -> bool {
        self.edges.iter().any(|e| e.id.as_str() == id)
    }

    fn assert_node_exists(&self, id: &ArchNodeId) {
        assert!(
            self.has_node_id(id),
            "unknown node ID '{}' in graph '{}'",
            id,
            self.name
        );
    }

    fn next_node_id_for_component(&self, component: &ArchNodeComponent) -> ArchNodeId {
        let base = component_node_base_id(component);
        let mut instance = 1usize;
        loop {
            let instance_tag = add_instance_tag(&base, instance);
            let candidate = ArchNodeId::from(format!("{instance_tag}{NODE_ID_SUFFIX}"));
            if !self.has_node_id(&candidate) && !self.has_edge_id_str(candidate.as_str()) {
                return candidate;
            }
            instance += 1;
        }
    }

    fn next_edge_id(&self, source: &ArchNodeId, target: &ArchNodeId) -> ArchEdgeId {
        let src = strip_suffix(source.as_str(), NODE_ID_SUFFIX);
        let tgt = strip_suffix(target.as_str(), NODE_ID_SUFFIX);
        let base = format!("edge::{src}_to_{tgt}");
        let mut instance = 1usize;
        loop {
            let instance_tag = add_instance_tag(&base, instance);
            let candidate = ArchEdgeId::from(format!("{instance_tag}{EDGE_ID_SUFFIX}"));
            if !self.has_edge_id(&candidate) && !self.has_node_id_str(candidate.as_str()) {
                return candidate;
            }
            instance += 1;
        }
    }

    fn edge_exists_between(&self, source: &ArchNodeId, target: &ArchNodeId) -> bool {
        self.edges.iter().any(|edge| {
            (edge.source == *source && edge.target == *target)
                || (edge.source == *target && edge.target == *source)
        })
    }

    pub fn add_component(&mut self, component: ArchNodeComponent) -> ArchNodeId {
        let resources = extract_resources(&component);
        let id = self.next_node_id_for_component(&component);
        self.nodes.push(ArchNode {
            id: id.clone(),
            component,
        });
        if !resources.is_empty() {
            let mut ids = Vec::with_capacity(resources.len());
            for r in resources {
                ids.push(r.id.clone());
                self.register_resource(r);
            }
            self.resource_map.insert(id.clone(), ids);
        }
        id
    }

    pub fn add_router(&mut self, router: &Router) -> ArchNodeId {
        self.add_component(ArchNodeComponent::Router(router.clone()))
    }

    pub fn memory_ref(&self, name: &str) -> Option<ArchNodeId> {
        self.node_id_by_name_and_kind(name, |component| {
            matches!(component, ArchNodeComponent::MemoryRegion(_))
        })
    }

    pub fn processor_ref(&self, name: &str) -> Option<ArchNodeId> {
        self.node_id_by_name_and_kind(name, |component| {
            matches!(component, ArchNodeComponent::Architecture(_))
        })
    }

    pub fn data_mover_ref(&self, name: &str) -> Option<ArchNodeId> {
        self.node_id_by_name_and_kind(name, |component| {
            matches!(component, ArchNodeComponent::DataMover(_))
        })
    }

    pub fn router_ref(&self, name: &str) -> Option<ArchNodeId> {
        self.node_id_by_name_and_kind(name, |component| {
            matches!(component, ArchNodeComponent::Router(_))
        })
    }

    pub fn add_node(&mut self, node: ArchNode) -> ArchNodeId {
        self.add_component(node.component)
    }

    pub fn connect(&mut self, source: &ArchNode, target: &ArchNode) -> &ArchEdge {
        self.connect_with_attrs(source, target, Vec::new())
    }

    pub fn connect_with_attrs(
        &mut self,
        source: &ArchNode,
        target: &ArchNode,
        attrs: Vec<ArchEdgeAttr>,
    ) -> &ArchEdge {
        self.assert_node_exists(&source.id);
        self.assert_node_exists(&target.id);
        assert!(
            !self.edge_exists_between(&source.id, &target.id),
            "edge already exists between '{}' and '{}' in graph '{}'",
            source.id,
            target.id,
            self.name
        );

        let edge_id = self.next_edge_id(&source.id, &target.id);
        let edge = ArchEdge::with_attrs(edge_id, source.id.clone(), target.id.clone(), attrs);
        self.edges.push(edge);
        self.edges
            .last()
            .expect("newly pushed edge must exist in graph")
    }

    pub fn get_node(&self, node_id: &ArchNodeId) -> Option<&ArchNode> {
        self.nodes.iter().find(|node| node.id == *node_id)
    }

    pub fn get_node_mut(&mut self, node_id: &ArchNodeId) -> Option<&mut ArchNode> {
        self.nodes.iter_mut().find(|node| node.id == *node_id)
    }

    pub fn add_memory_region(&mut self, region: &MemoryRegion) -> ArchNodeId {
        self.add_component(ArchNodeComponent::MemoryRegion(region.clone()))
    }

    pub fn add_architecture(&mut self, architecture: &Architecture) -> ArchNodeId {
        self.add_component(ArchNodeComponent::Architecture(architecture.clone()))
    }

    pub fn add_data_mover(&mut self, mover: &DataMover) -> ArchNodeId {
        self.add_component(ArchNodeComponent::DataMover(mover.clone()))
    }

    fn node_id_by_name_and_kind<F>(&self, name: &str, kind_check: F) -> Option<ArchNodeId>
    where
        F: Fn(&ArchNodeComponent) -> bool,
    {
        self.nodes
            .iter()
            .find(|node| node.name() == Some(name) && kind_check(&node.component))
            .map(|node| node.id.clone())
    }

    // ── Resource API ────────────────────────────────────────────────────

    /// Register a resource, deduplicating by ID.
    ///
    /// If a resource with the same ID already exists, asserts that the
    /// capacity matches (two processors sharing a resource must agree on
    /// its capacity).
    fn register_resource(&mut self, resource: Resource) {
        if let Some(existing) = self.resources.iter().find(|r| r.id == resource.id) {
            assert_eq!(
                existing.capacity, resource.capacity,
                "resource '{}' registered with conflicting capacities ({} vs {}) in graph '{}'",
                resource.id, existing.capacity, resource.capacity, self.name
            );
            return;
        }
        self.resources.push(resource);
    }

    /// Look up a resource definition by ID.
    pub fn get_resource(&self, id: &ResourceId) -> Option<&Resource> {
        self.resources.iter().find(|r| r.id == *id)
    }

    /// Return the resource IDs associated with a given node.
    ///
    /// Returns an empty slice for nodes not in the resource map (implicitly
    /// treated as sole consumers of their own private resource).
    pub fn node_resources(&self, node: &ArchNodeId) -> &[ResourceId] {
        self.resource_map
            .get(node)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Collect all node IDs that are associated with a given resource.
    pub fn resource_consumers(&self, resource: &ResourceId) -> Vec<&ArchNodeId> {
        self.resource_map
            .iter()
            .filter(|(_, ids)| ids.iter().any(|id| id == resource))
            .map(|(node_id, _)| node_id)
            .collect()
    }

    /// Extract the set of distinct `ResourceId`s referenced in the resource map.
    pub fn resource_ids_in_use(&self) -> Vec<ResourceId> {
        let mut seen = std::collections::HashSet::new();
        for ids in self.resource_map.values() {
            for id in ids {
                seen.insert(id.clone());
            }
        }
        seen.into_iter().collect()
    }

    /// Check whether two nodes share any resource.
    ///
    /// A node **not** in the resource map never conflicts with anything
    /// (it is the sole user of its own implicit resource).
    pub fn nodes_share_resource(&self, a: &ArchNodeId, b: &ArchNodeId) -> bool {
        let a_ids = match self.resource_map.get(a) {
            Some(v) => v,
            None => return false,
        };
        let b_ids = match self.resource_map.get(b) {
            Some(v) => v,
            None => return false,
        };
        a_ids.iter().any(|id| b_ids.contains(id))
    }

    /// Validate that every resource referenced in the resource map exists in
    /// `self.resources` and every node in the map exists in the graph.
    pub fn validate_resources(&self) -> Result<(), String> {
        for (node_id, ids) in &self.resource_map {
            if !self.has_node_id(node_id) {
                return Err(format!(
                    "resource map references unknown node '{}' in graph '{}'",
                    node_id, self.name
                ));
            }
            for rid in ids {
                if self.get_resource(rid).is_none() {
                    return Err(format!(
                        "node '{}' references unknown resource '{}' in graph '{}'",
                        node_id, rid, self.name
                    ));
                }
            }
        }
        Ok(())
    }

    // ── Builder entry point ─────────────────────────────────────────────

    /// Create a builder for constructing an `ArchGraph`.
    pub fn builder(name: impl Into<String>) -> ArchGraphBuilder {
        ArchGraphBuilder {
            graph: ArchGraph::new(name),
        }
    }
}

/// Extract resources from a component's underlying processor (if any).
fn extract_resources(component: &ArchNodeComponent) -> Vec<Resource> {
    match component {
        ArchNodeComponent::Architecture(arch) => arch_resources(arch),
        ArchNodeComponent::DataMover(dm) => dm.0.resources.clone(),
        ArchNodeComponent::MemoryRegion(_) | ArchNodeComponent::Router(_) => Vec::new(),
    }
}

fn arch_resources(arch: &Architecture) -> Vec<Resource> {
    match arch {
        Architecture::Unit(proc) => proc.resources.clone(),
        Architecture::Array { elem, .. } => arch_resources(elem),
        Architecture::Graph(_) => Vec::new(),
    }
}

fn component_node_base_id(component: &ArchNodeComponent) -> String {
    let name = component
        .display_name()
        .unwrap_or(default_component_name(component));
    format!("{}::{}", component.id_prefix(), name)
}

fn default_component_name(component: &ArchNodeComponent) -> &'static str {
    match component {
        ArchNodeComponent::Architecture(_) => "unnamed_architecture",
        ArchNodeComponent::DataMover(_) => "unnamed_data_mover",
        ArchNodeComponent::MemoryRegion(_) => "unnamed_memory",
        ArchNodeComponent::Router(_) => "unnamed_router",
    }
}

fn add_instance_tag(base: &str, instance: usize) -> String {
    if instance == 1 {
        base.to_string()
    } else {
        format!("{base}#{instance}")
    }
}

fn strip_suffix<'a>(value: &'a str, suffix: &str) -> &'a str {
    value.strip_suffix(suffix).unwrap_or(value)
}

impl ArchGraphBuilder {
    /// Add a memory region (borrows and clones).
    pub fn mem(mut self, region: &MemoryRegion) -> Self {
        self.graph.add_memory_region(region);
        self
    }

    /// Add an architecture node (borrows and clones).
    pub fn architecture(mut self, arch: &Architecture) -> Self {
        self.graph.add_architecture(arch);
        register_connectivity_components(&mut self.graph, arch);
        self
    }

    /// Add a data-mover node (borrows and clones).
    pub fn data_mover(mut self, mover: &DataMover) -> Self {
        self.graph.add_data_mover(mover);
        self
    }

    /// Graph architectures are node-only; scale-out links belong to `Architecture::Array`.
    pub fn link(self, _link: ScaleOutNetwork) -> Self {
        self
    }

    /// Add a router node.
    pub fn router(mut self, router: &Router) -> Self {
        self.graph.add_router(router);
        self
    }

    /// Add a pre-built abstract graph node by component.
    pub fn node(mut self, node: &ArchNode) -> Self {
        self.graph.add_component(node.component.clone());
        self
    }

    /// Add a graph node component.
    pub fn component(mut self, component: ArchNodeComponent) -> Self {
        self.graph.add_component(component);
        self
    }

    /// Build the `ArchGraph`.
    pub fn build(self) -> ArchGraph {
        self.graph
    }
}

fn register_connectivity_components(graph: &mut ArchGraph, arch: &Architecture) {
    let mut seen_named_movers = HashSet::new();
    register_connectivity_components_impl(graph, arch, &mut seen_named_movers);
}

fn register_connectivity_components_impl(
    graph: &mut ArchGraph,
    arch: &Architecture,
    seen_named_movers: &mut HashSet<String>,
) {
    match arch {
        Architecture::Array {
            connectivity, elem, ..
        } => {
            for network in connectivity {
                for resource in network.resources() {
                    graph.register_resource(resource);
                }
                for mover in network.data_movers() {
                    if let Some(name) = mover.name.as_deref() {
                        if !seen_named_movers.insert(name.to_string()) {
                            continue;
                        }
                        if graph.data_mover_ref(name).is_some() {
                            continue;
                        }
                    }
                    graph.add_data_mover(&mover);
                }
            }
            register_connectivity_components_impl(graph, elem, seen_named_movers);
        }
        Architecture::Graph(subgraph) => {
            for node in &subgraph.nodes {
                if let ArchNodeComponent::Architecture(sub_arch) = &node.component {
                    register_connectivity_components_impl(graph, sub_arch, seen_named_movers);
                }
            }
        }
        Architecture::Unit(_) => {}
    }
}
