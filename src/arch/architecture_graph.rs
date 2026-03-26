use serde::{Deserialize, Serialize};

use super::architecture::Architecture;
use super::memory::MemoryRegion;
use super::network::ScaleOutNetwork;
use super::processor::{DataMover, Processor};
use super::router::{Router, RouterSide};
use std::fmt;

const NODE_ID_SUFFIX: &str = "::node";
const EDGE_ID_SUFFIX: &str = "::edge";

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ArchNodeId(String);

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

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ArchEdgeId(String);

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

/// Abstract node payload for heterogeneous architecture graph composition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArchNodeComponent {
    Architecture(Architecture),
    DataMover(DataMover),
    MemoryRegion(MemoryRegion),
    Router(Router),
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

/// Abstract architecture graph node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchNode {
    pub id: ArchNodeId,
    pub component: ArchNodeComponent,
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

impl Default for ArchEdgeDirection {
    fn default() -> Self {
        Self::Directional
    }
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

/// Heterogeneous architecture composition using explicit nodes and edges.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchGraph {
    pub name: String,
    pub nodes: Vec<ArchGraphNode>,
    pub edges: Vec<ArchEdge>,
}

impl ArchGraph {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            nodes: Vec::new(),
            edges: Vec::new(),
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
        let id = self.next_node_id_for_component(&component);
        self.nodes.push(ArchNode {
            id: id.clone(),
            component,
        });
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

    /// Create a builder for constructing an `ArchGraph`.
    pub fn builder(name: impl Into<String>) -> ArchGraphBuilder {
        ArchGraphBuilder {
            graph: ArchGraph::new(name),
        }
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

/// Builder for constructing an `ArchGraph`.
pub struct ArchGraphBuilder {
    graph: ArchGraph,
}

impl ArchGraphBuilder {
    /// Add a memory region (borrows and clones).
    pub fn mem(mut self, region: &MemoryRegion) -> Self {
        self.graph.add_memory_region(region);
        self
    }

    /// Add a processor architecture (borrows and clones).
    pub fn processor(mut self, proc: &Architecture) -> Self {
        self.graph.add_architecture(proc);
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
