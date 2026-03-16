use super::architecture::Architecture;
use super::links::{Router, ScaleOutNetwork};
use super::memory::MemoryRegion;
use super::processor::Processor;
use std::collections::HashSet;

/// Abstract node payload for architecture graph nodes.
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
pub struct ArchNode {
    pub id: String,
    pub name: String,
    pub component: ArchNodeComponent,
}

impl ArchNode {
    pub fn from_architecture(architecture: &Architecture) -> Self {
        let name = architecture
            .name()
            .unwrap_or("unnamed")
            .to_string();
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

/// Backward-compatible name for existing code.
pub type ArchGraphNode = ArchNode;

/// Directed edge between two architecture graph nodes.
#[derive(Debug, Clone)]
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

/// Graph-style architecture description.
#[derive(Debug, Clone)]
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
