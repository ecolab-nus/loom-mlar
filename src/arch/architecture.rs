use super::graph::{ArchGraph, ArchNodeComponent};
use super::links::ScaleOutNetwork;
use super::memory::MemoryRegion;
use super::processor::Processor;
use super::size_dim::Dimension;
use crate::schedule::Module;
use std::ops::{Deref, DerefMut};

/// Unified recursive architecture description.
///
/// `Architecture` replaces the former split between processor sets and
/// top-level architecture:
/// - `Unit`: atomic processor
/// - `Array`: homogeneous scaling-out of a sub-architecture
/// - `Graph`: explicit graph architecture (`ArchGraph`)
#[derive(Debug, Clone)]
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
        let router = ArchNode::from_router("router:3", &Router::new("crossbar"));
        let arch: Architecture = ArchGraph::builder("mesh").node(&router).build().into();
        let graph = arch.as_graph().expect("builder must create graph");
        assert!(graph.nodes.iter().any(|n| n.id == "router:3"));
    }

    #[test]
    fn builder_accepts_custom_edges() {
        let edge = ArchEdge::new("edge:0", "arch:0", "arch:1");
        let arch: Architecture = ArchGraph::builder("mesh").edge(&edge).build().into();
        let graph = arch.as_graph().expect("builder must create graph");
        assert_eq!(graph.edges.len(), 1);
        assert_eq!(graph.edges[0].id, "edge:0");
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
}
