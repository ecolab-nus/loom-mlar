use serde::{Deserialize, Serialize};

use super::links::{Router, ScaleOutNetwork};
use super::memory::MemoryRegion;
use super::processor::Processor;
use super::size_dim::Dimension;
use crate::schedule::Module;
use std::collections::HashSet;
use std::ops::{Deref, DerefMut};

/// Abstract node payload for heterogeneous architecture graph composition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArchNodeComponent {
    Architecture(Architecture),
    MemoryRegion(MemoryRegion),
    Router(Router),
}

impl ArchNodeComponent {
    pub fn display_name(&self) -> Option<&str> {
        match self {
            ArchNodeComponent::Architecture(arch) => arch.name(),
            ArchNodeComponent::MemoryRegion(region) => region.name(),
            ArchNodeComponent::Router(router) => Some(router.name.as_str()),
        }
    }
}

/// Abstract architecture graph node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchNode {
    pub id: String,
    pub name: String,
    pub component: ArchNodeComponent,
}

impl ArchNode {
    pub fn from_architecture(architecture: &Architecture) -> Self {
        let name = architecture.name().unwrap_or("unnamed").to_string();
        Self {
            id: format!("arch::{name}"),
            name,
            component: ArchNodeComponent::Architecture(architecture.clone()),
        }
    }

    pub fn from_processor(proc: &Processor) -> Self {
        Self::from_architecture(&Architecture::Unit(proc.clone()))
    }

    pub fn from_memory_region(region: &MemoryRegion) -> Self {
        let name = region.name().unwrap_or("unnamed").to_string();
        Self {
            id: format!("mem::{name}"),
            name,
            component: ArchNodeComponent::MemoryRegion(region.clone()),
        }
    }

    pub fn from_router(router: &Router) -> Self {
        let name = router.name.clone();
        Self {
            id: format!("router::{name}"),
            name,
            component: ArchNodeComponent::Router(router.clone()),
        }
    }
}

/// Alias name for graph nodes.
pub type ArchGraphNode = ArchNode;

/// Directed edge between two architecture graph nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchEdge {
    pub id: String,
    pub source: String,
    pub target: String,
}

impl ArchEdge {
    /// Create a directed edge. The ID is auto-generated from the source and
    /// target node IDs (e.g. `"edge::L1_to_lane"`).
    pub fn new(source: impl Into<String>, target: impl Into<String>) -> Self {
        let source = source.into();
        let target = target.into();
        let src_name = source.rsplit("::").next().unwrap_or(&source);
        let tgt_name = target.rsplit("::").next().unwrap_or(&target);
        Self {
            id: format!("edge::{src_name}_to_{tgt_name}"),
            source,
            target,
        }
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

    fn has_id(&self, id: &str) -> bool {
        self.nodes.iter().any(|n| n.id == id) || self.edges.iter().any(|e| e.id == id)
    }

    fn assert_unique(&self, id: &str) {
        assert!(
            !self.has_id(id),
            "duplicate ID '{}' in graph '{}'",
            id,
            self.name
        );
    }

    pub fn add_router(&mut self, router: &Router) -> String {
        let node = ArchNode::from_router(router);
        let id = node.id.clone();
        self.add_node(node);
        id
    }

    pub fn memory_ref(&self, name: &str) -> Option<String> {
        self.node_id_by_name_and_kind(name, |component| {
            matches!(component, ArchNodeComponent::MemoryRegion(_))
        })
    }

    pub fn processor_ref(&self, name: &str) -> Option<String> {
        self.node_id_by_name_and_kind(name, |component| {
            matches!(component, ArchNodeComponent::Architecture(_))
        })
    }

    pub fn router_ref(&self, name: &str) -> Option<String> {
        self.node_id_by_name_and_kind(name, |component| {
            matches!(component, ArchNodeComponent::Router(_))
        })
    }

    pub fn add_node(&mut self, node: ArchNode) {
        self.assert_unique(&node.id);
        self.nodes.push(node);
    }

    pub fn add_edge(&mut self, edge: ArchEdge) {
        self.assert_unique(&edge.id);
        self.edges.push(edge);
    }

    pub fn connect(&mut self, source: impl Into<String>, target: impl Into<String>) -> &ArchEdge {
        let edge = ArchEdge::new(source, target);
        self.add_edge(edge);
        self.edges
            .last()
            .expect("newly pushed edge must exist in graph")
    }

    pub fn get_node(&self, node_id: &str) -> Option<&ArchNode> {
        self.nodes.iter().find(|node| node.id == node_id)
    }

    pub fn get_node_mut(&mut self, node_id: &str) -> Option<&mut ArchNode> {
        self.nodes.iter_mut().find(|node| node.id == node_id)
    }

    pub fn add_memory_region(&mut self, region: &MemoryRegion) -> String {
        let node = ArchNode::from_memory_region(region);
        let id = node.id.clone();
        self.add_node(node);
        id
    }

    pub fn add_architecture(&mut self, architecture: &Architecture) -> String {
        let node = ArchNode::from_architecture(architecture);
        let id = node.id.clone();
        self.add_node(node);
        id
    }

    fn node_id_by_name_and_kind<F>(&self, name: &str, kind_check: F) -> Option<String>
    where
        F: Fn(&ArchNodeComponent) -> bool,
    {
        self.nodes
            .iter()
            .find(|node| node.name == name && kind_check(&node.component))
            .map(|node| node.id.clone())
    }

    /// Create a builder for constructing an `ArchGraph`.
    pub fn builder(name: impl Into<String>) -> ArchGraphBuilder {
        ArchGraphBuilder {
            name: name.into(),
            nodes: Vec::new(),
            edges: Vec::new(),
            ids: HashSet::new(),
        }
    }
}

/// Builder for constructing an `ArchGraph`.
pub struct ArchGraphBuilder {
    name: String,
    nodes: Vec<ArchNode>,
    edges: Vec<ArchEdge>,
    ids: HashSet<String>,
}

impl ArchGraphBuilder {
    fn assert_unique(&self, id: &str) {
        assert!(
            !self.ids.contains(id),
            "duplicate ID '{}' in graph builder '{}'",
            id,
            self.name
        );
    }

    /// Add a memory region (borrows and clones).
    pub fn mem(mut self, region: &MemoryRegion) -> Self {
        let node = ArchNode::from_memory_region(region);
        self.assert_unique(&node.id);
        self.ids.insert(node.id.clone());
        self.nodes.push(node);
        self
    }

    /// Add a processor architecture (borrows and clones).
    pub fn processor(mut self, proc: &Architecture) -> Self {
        let node = ArchNode::from_architecture(proc);
        self.assert_unique(&node.id);
        self.ids.insert(node.id.clone());
        self.nodes.push(node);
        self
    }

    /// Graph architectures are node-only; scale-out links belong to `Architecture::Array`.
    pub fn link(self, _link: ScaleOutNetwork) -> Self {
        self
    }

    /// Add a router node.
    pub fn router(mut self, router: &Router) -> Self {
        let node = ArchNode::from_router(router);
        self.assert_unique(&node.id);
        self.ids.insert(node.id.clone());
        self.nodes.push(node);
        self
    }

    /// Add a pre-built abstract graph node.
    pub fn node(mut self, node: &ArchNode) -> Self {
        self.assert_unique(&node.id);
        self.ids.insert(node.id.clone());
        self.nodes.push(node.clone());
        self
    }

    /// Add a pre-built abstract graph edge.
    pub fn edge(mut self, edge: &ArchEdge) -> Self {
        self.assert_unique(&edge.id);
        self.ids.insert(edge.id.clone());
        self.edges.push(edge.clone());
        self
    }

    /// Build the `ArchGraph`.
    pub fn build(self) -> ArchGraph {
        ArchGraph {
            name: self.name,
            nodes: self.nodes,
            edges: self.edges,
        }
    }
}

/// Unified recursive architecture description.
///
/// One architecture value is either:
/// - `Unit`: one processor (atomic architecture)
/// - `Array`: homogeneous scaling of one sub-architecture
/// - `Graph`: heterogeneous composition of architecture/memory/router nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Architecture {
    /// Leaf: a single processor.
    Unit(Processor),
    /// Homogeneous array of an architecture.
    Array {
        name: Option<String>,
        dims: Vec<Dimension>,
        elem: Box<Architecture>,
        /// Connectivity among array instances.
        connectivity: Vec<ScaleOutNetwork>,
        /// Placeholder for outside-facing access interface.
        interface: Option<String>,
    },
    /// Explicit graph architecture.
    Graph(ArchGraph),
}

impl Architecture {
    /// Build an explicit graph architecture from parts.
    pub fn from_graph(graph: ArchGraph) -> Self {
        Architecture::Graph(graph)
    }

    /// Access this value as a graph architecture.
    pub fn as_graph(&self) -> Option<&ArchGraph> {
        match self {
            Architecture::Graph(graph) => Some(graph),
            _ => None,
        }
    }

    /// Mutable access to graph architecture.
    pub fn as_graph_mut(&mut self) -> Option<&mut ArchGraph> {
        match self {
            Architecture::Graph(graph) => Some(graph),
            _ => None,
        }
    }

    /// Look up a named memory region (graph architecture only).
    pub fn get_memory_region(&self, name: &str) -> Option<&MemoryRegion> {
        match self {
            Architecture::Graph(graph) => {
                graph.nodes.iter().find_map(|node| match &node.component {
                    ArchNodeComponent::MemoryRegion(region) if region.name() == Some(name) => {
                        Some(region)
                    }
                    _ => None,
                })
            }
            Architecture::Array { elem, .. } => elem.get_memory_region(name),
            Architecture::Unit(_) => None,
        }
    }

    /// Look up a named processor architecture.
    ///
    /// For graph architectures this searches architecture nodes.
    /// For array/unit architectures this recursively searches itself.
    pub fn get_processor(&self, name: &str) -> Option<&Architecture> {
        match self {
            Architecture::Graph(graph) => {
                for node in &graph.nodes {
                    if let ArchNodeComponent::Architecture(arch) = &node.component {
                        if arch.name() == Some(name) {
                            return Some(arch);
                        }
                        if let Some(found) = arch.get_processor(name) {
                            return Some(found);
                        }
                    }
                }
                None
            }
            Architecture::Unit(_) => self.name().filter(|n| *n == name).map(|_| self),
            Architecture::Array { elem, .. } => {
                if self.name() == Some(name) {
                    Some(self)
                } else {
                    elem.get_processor(name)
                }
            }
        }
    }

    /// Scale this architecture by prepending dimensions.
    ///
    /// Wraps the architecture in an outer `Array`.
    pub fn scale<'a, I>(self, dims: I) -> Architecture
    where
        I: IntoIterator<Item = &'a Dimension>,
    {
        Architecture::Array {
            name: None,
            dims: dims.into_iter().cloned().collect(),
            elem: Box::new(self),
            connectivity: Vec::new(),
            interface: None,
        }
    }

    /// Get the name for this architecture value.
    pub fn name(&self) -> Option<&str> {
        match self {
            Architecture::Unit(p) => p.name.as_deref(),
            Architecture::Array { name, elem, .. } => name.as_deref().or_else(|| elem.name()),
            Architecture::Graph(graph) => Some(graph.name.as_str()),
        }
    }

    /// Get functionality if this architecture is (or contains) a unit processor.
    pub fn functionality(&self) -> Option<&Module> {
        match self {
            Architecture::Unit(p) => Some(&p.functionality),
            Architecture::Array { elem, .. } => elem.functionality(),
            Architecture::Graph(_) => None,
        }
    }

    /// Set the name at the current level (builder-style, consumes self).
    pub fn with_name(self, n: impl Into<String>) -> Self {
        match self {
            Architecture::Unit(mut p) => {
                p.name = Some(n.into());
                Architecture::Unit(p)
            }
            Architecture::Array {
                dims,
                elem,
                connectivity,
                interface,
                ..
            } => Architecture::Array {
                name: Some(n.into()),
                dims,
                elem,
                connectivity,
                interface,
            },
            Architecture::Graph(mut graph) => {
                graph.name = n.into();
                Architecture::Graph(graph)
            }
        }
    }

    /// Set explicit array connectivity.
    pub fn with_connectivity(self, connectivity: Vec<ScaleOutNetwork>) -> Self {
        match self {
            Architecture::Array {
                name,
                dims,
                elem,
                interface,
                ..
            } => Architecture::Array {
                name,
                dims,
                elem,
                connectivity,
                interface,
            },
            other => other,
        }
    }

    /// Set placeholder outside-facing interface for array architectures.
    pub fn with_interface(self, interface: impl Into<String>) -> Self {
        match self {
            Architecture::Array {
                name,
                dims,
                elem,
                connectivity,
                ..
            } => Architecture::Array {
                name,
                dims,
                elem,
                connectivity,
                interface: Some(interface.into()),
            },
            other => other,
        }
    }

    /// Get the outermost dimensions (empty for Unit/Graph).
    pub fn dims(&self) -> &[Dimension] {
        match self {
            Architecture::Array { dims, .. } => dims,
            _ => &[],
        }
    }

    /// Compute total number of instances.
    ///
    /// Returns `None` if any dimension has symbolic size.
    pub fn total_instances(&self) -> Option<u64> {
        match self {
            Architecture::Unit(_) => Some(1),
            Architecture::Array { dims, elem, .. } => {
                let outer: u64 = dims
                    .iter()
                    .map(|d| d.size.as_const())
                    .collect::<Option<Vec<_>>>()?
                    .into_iter()
                    .product();
                let inner = elem.total_instances()?;
                Some(outer * inner)
            }
            Architecture::Graph(graph) => {
                let mut total = 0u64;
                for node in &graph.nodes {
                    if let ArchNodeComponent::Architecture(arch) = &node.component {
                        total += arch.total_instances()?;
                    }
                }
                Some(total)
            }
        }
    }

    /// Collect all dimensions recursively.
    pub fn all_dims(&self) -> Vec<&Dimension> {
        match self {
            Architecture::Unit(_) => vec![],
            Architecture::Array { dims, elem, .. } => {
                let mut result: Vec<&Dimension> = dims.iter().collect();
                result.extend(elem.all_dims());
                result
            }
            Architecture::Graph(graph) => {
                let mut result = Vec::new();
                for node in &graph.nodes {
                    if let ArchNodeComponent::Architecture(arch) = &node.component {
                        result.extend(arch.all_dims());
                    }
                }
                result
            }
        }
    }

    /// Get total number of processing elements.
    ///
    /// For graph architectures this sums all graph processors.
    pub fn total_processing_elements(&self) -> Option<u64> {
        self.total_instances()
    }
}

impl Deref for Architecture {
    type Target = ArchGraph;

    fn deref(&self) -> &Self::Target {
        match self {
            Architecture::Graph(graph) => graph,
            _ => panic!("Architecture is not Graph; use as_graph()/as_graph_mut()"),
        }
    }
}

impl DerefMut for Architecture {
    fn deref_mut(&mut self) -> &mut Self::Target {
        match self {
            Architecture::Graph(graph) => graph,
            _ => panic!("Architecture is not Graph; use as_graph()/as_graph_mut()"),
        }
    }
}

impl From<ArchGraph> for Architecture {
    fn from(graph: ArchGraph) -> Self {
        Architecture::Graph(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::Architecture;
    use crate::arch::{
        ArchEdge, ArchGraph, ArchNode, ArchNodeComponent, MemoryBank, MemoryRegion, Processor,
        Router, SizeExpr,
    };

    #[test]
    fn arch_graph_builder_materializes_memory_and_architecture_nodes() {
        let l1 = MemoryRegion::bank(MemoryBank::new(SizeExpr::Const(1024))).with_name("l1");
        let lane = Processor::new("lane").into_elem();
        let graph: Architecture = ArchGraph::builder("core")
            .mem(&l1)
            .processor(&lane)
            .router(&Router::new("router"))
            .build()
            .into();
        let graph = graph.as_graph().expect("must build graph");

        assert_eq!(graph.nodes.len(), 3);
        assert!(graph.edges.is_empty());
        assert!(
            graph
                .nodes
                .iter()
                .any(|n| matches!(n.component, ArchNodeComponent::MemoryRegion(_)))
        );
        assert!(
            graph
                .nodes
                .iter()
                .any(|n| matches!(n.component, ArchNodeComponent::Architecture(_)))
        );
        assert!(
            graph
                .nodes
                .iter()
                .any(|n| matches!(n.component, ArchNodeComponent::Router(_)))
        );
    }

    #[test]
    fn builder_accepts_custom_nodes() {
        let router = ArchNode::from_router(&Router::new("crossbar"));
        let arch: Architecture = ArchGraph::builder("mesh").node(&router).build().into();
        let graph = arch.as_graph().expect("builder must create graph");
        assert!(graph.nodes.iter().any(|n| n.id == "router::crossbar"));
    }

    #[test]
    fn builder_accepts_custom_edges() {
        let edge = ArchEdge::new("arch::foo", "arch::bar");
        let arch: Architecture = ArchGraph::builder("mesh").edge(&edge).build().into();
        let graph = arch.as_graph().expect("builder must create graph");
        assert_eq!(graph.edges.len(), 1);
        assert_eq!(graph.edges[0].id, "edge::foo_to_bar");
    }

    #[test]
    fn graph_supports_node_lookup_by_id() {
        let arch: Architecture = ArchGraph::builder("mesh")
            .router(&Router::new("router"))
            .build()
            .into();
        let graph = arch.as_graph().expect("builder must create graph");
        let router_id = graph
            .router_ref("router")
            .expect("router node ID should be available");
        let router = graph.get_node(&router_id).expect("router node must exist");
        assert_eq!(router.name, "router");
    }

    #[test]
    fn auto_generated_ids_use_component_names() {
        let l1 = MemoryRegion::bank(MemoryBank::new(SizeExpr::Const(1024))).with_name("L1");
        let lane = Processor::new("lane").into_elem();
        let graph: Architecture = ArchGraph::builder("core")
            .mem(&l1)
            .processor(&lane)
            .router(&Router::new("xbar"))
            .build()
            .into();
        let graph = graph.as_graph().expect("must build graph");

        assert!(graph.get_node("mem::L1").is_some());
        assert!(graph.get_node("arch::lane").is_some());
        assert!(graph.get_node("router::xbar").is_some());
    }

    #[test]
    #[should_panic(expected = "duplicate ID")]
    fn builder_rejects_duplicate_node_ids() {
        let r1 = Router::new("dup");
        let r2 = Router::new("dup");
        ArchGraph::builder("bad").router(&r1).router(&r2).build();
    }

    #[test]
    #[should_panic(expected = "duplicate ID")]
    fn graph_rejects_duplicate_node_ids() {
        let mut graph = ArchGraph::new("bad");
        let r = Router::new("r");
        graph.add_router(&r);
        graph.add_router(&r);
    }
}
