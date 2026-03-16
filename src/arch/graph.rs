use super::architecture::Architecture;
use super::links::{Router, ScaleOutNetwork};
use super::memory::MemoryRegion;
use super::processor::Processor;

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
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        component: ArchNodeComponent,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            component,
        }
    }

    pub fn from_architecture(id: impl Into<String>, architecture: &Architecture) -> Self {
        let name = architecture
            .name()
            .map(|n| n.to_string())
            .unwrap_or_else(|| "architecture".to_string());
        Self::new(
            id,
            name,
            ArchNodeComponent::Architecture(architecture.clone()),
        )
    }

    pub fn from_processor(id: impl Into<String>, proc: &Processor) -> Self {
        Self::from_architecture(id, &Architecture::Unit(proc.clone()))
    }

    pub fn from_memory_region(id: impl Into<String>, region: &MemoryRegion) -> Self {
        let name = region.name().unwrap_or("memory");
        Self::new(id, name, ArchNodeComponent::MemoryRegion(region.clone()))
    }

    pub fn from_router(id: impl Into<String>, router: &Router) -> Self {
        Self::new(
            id,
            router.name.clone(),
            ArchNodeComponent::Router(router.clone()),
        )
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
    pub fn new(
        id: impl Into<String>,
        source: impl Into<String>,
        target: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            source: source.into(),
            target: target.into(),
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

    pub fn add_router(&mut self, router: &Router) -> String {
        let node_index = self.nodes.len();
        let node_id = format!("router:{node_index}");
        self.nodes
            .push(ArchNode::from_router(node_id.clone(), router));
        node_id
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
        self.nodes.push(node);
    }

    pub fn add_edge(&mut self, edge: ArchEdge) {
        self.edges.push(edge);
    }

    pub fn connect(&mut self, source: impl Into<String>, target: impl Into<String>) -> &ArchEdge {
        let edge_index = self.edges.len();
        let edge_id = format!("edge:{edge_index}");
        self.edges.push(ArchEdge::new(edge_id, source, target));
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
        let node_index = self.nodes.len();
        let node_id = format!("mem:{node_index}");
        self.nodes
            .push(ArchNode::from_memory_region(node_id.clone(), region));
        node_id
    }

    pub fn add_architecture(&mut self, architecture: &Architecture) -> String {
        let node_index = self.nodes.len();
        let node_id = format!("arch:{node_index}");
        self.nodes
            .push(ArchNode::from_architecture(node_id.clone(), architecture));
        node_id
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
            mem_count: 0,
            arch_count: 0,
            router_count: 0,
        }
    }
}

/// Builder for constructing an `ArchGraph`.
pub struct ArchGraphBuilder {
    name: String,
    nodes: Vec<ArchNode>,
    edges: Vec<ArchEdge>,
    mem_count: usize,
    arch_count: usize,
    router_count: usize,
}

impl ArchGraphBuilder {
    /// Add a memory region (borrows and clones).
    pub fn mem(mut self, region: &MemoryRegion) -> Self {
        self.nodes.push(ArchNode::from_memory_region(
            format!("mem:{}", self.mem_count),
            region,
        ));
        self.mem_count += 1;
        self
    }

    /// Add a processor architecture (borrows and clones).
    pub fn processor(mut self, proc: &Architecture) -> Self {
        self.nodes.push(ArchNode::from_architecture(
            format!("arch:{}", self.arch_count),
            proc,
        ));
        self.arch_count += 1;
        self
    }

    /// Graph architectures are node-only; scale-out links belong to `Architecture::Array`.
    pub fn link(self, _link: ScaleOutNetwork) -> Self {
        self
    }

    /// Add a router node.
    pub fn router(mut self, router: &Router) -> Self {
        let node_id = format!("router:{}", self.router_count);
        self.nodes.push(ArchNode::from_router(node_id, router));
        self.router_count += 1;
        self
    }

    /// Add a pre-built abstract graph node.
    pub fn node(mut self, node: &ArchNode) -> Self {
        self.nodes.push(node.clone());
        self
    }

    /// Add a pre-built abstract graph edge.
    pub fn edge(mut self, edge: &ArchEdge) -> Self {
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
