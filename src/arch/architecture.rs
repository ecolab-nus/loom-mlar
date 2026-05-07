use serde::{Deserialize, Serialize};

use super::architecture_graph::{ArchGraph, ArchNodeComponent};
use super::memory::MemoryRegion;
use super::network::ScaleOutNetwork;
use super::processor::{DataMover, Processor};
use super::size_dim::Dimension;
use crate::schedule::MlirModule;
use std::ops::{Deref, DerefMut};

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

    /// Look up a memory region and materialize it through array scale levels.
    ///
    /// For an `Array`, this returns a new [`MemoryRegion::Array`] with the same
    /// outer dimensions as the architecture array and the matching element
    /// memory region as its sub-region. This is useful for components outside
    /// the array that need to refer to the whole distributed memory region
    /// rather than one element-local memory.
    pub fn get_scaled_memory_region(&self, name: &str) -> Option<MemoryRegion> {
        match self {
            Architecture::Graph(graph) => {
                for node in &graph.nodes {
                    match &node.component {
                        ArchNodeComponent::MemoryRegion(region) if region.name() == Some(name) => {
                            return Some(region.clone());
                        }
                        ArchNodeComponent::Architecture(arch) => {
                            if let Some(region) = arch.get_scaled_memory_region(name) {
                                return Some(region);
                            }
                        }
                        _ => {}
                    }
                }
                None
            }
            Architecture::Array { dims, elem, .. } => {
                let region = elem.get_scaled_memory_region(name)?;
                Some(region.scale(dims).with_name(format!("array_{name}")))
            }
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

    /// Look up a named data-mover node (graph architecture only).
    pub fn get_data_mover(&self, name: &str) -> Option<&DataMover> {
        match self {
            Architecture::Graph(graph) => {
                graph.nodes.iter().find_map(|node| match &node.component {
                    ArchNodeComponent::DataMover(mover) if mover.name.as_deref() == Some(name) => {
                        Some(mover)
                    }
                    _ => None,
                })
            }
            Architecture::Array { elem, .. } => elem.get_data_mover(name),
            Architecture::Unit(_) => None,
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
    pub fn functionality(&self) -> Option<&MlirModule> {
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
                ..
            } => Architecture::Array {
                name: Some(n.into()),
                dims,
                elem,
                connectivity,
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
                name, dims, elem, ..
            } => Architecture::Array {
                name,
                dims,
                elem,
                connectivity,
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
        ArchEdgeAttr, ArchEdgeDirection, ArchGraph, ArchNode, ArchNodeComponent, ComputeProcessor,
        Dimension, MemoryBank, MemoryRegion, MeshNetworkInterface, Processor, Router,
        ScaleOutNetwork, SizeExpr,
    };
    use crate::math::{AffineMap, Expr};

    #[test]
    fn arch_graph_builder_materializes_memory_and_architecture_nodes() {
        let l1 = MemoryRegion::bank(MemoryBank::new(SizeExpr::Const(1024))).with_name("l1");
        let lane = Processor::new("lane").into_elem();
        let graph: Architecture = ArchGraph::builder("core")
            .mem(&l1)
            .architecture(&lane)
            .router(&Router::new("router", 0))
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
        let router = ArchNode::from_router(&Router::new("crossbar", 0));
        let arch: Architecture = ArchGraph::builder("mesh").node(&router).build().into();
        let graph = arch.as_graph().expect("builder must create graph");
        assert!(
            graph
                .nodes
                .iter()
                .any(|n| n.id.as_str() == "router::crossbar::node")
        );
    }

    #[test]
    fn graph_connect_assigns_managed_edge_id() {
        let mut graph = ArchGraph::new("mesh");
        let src_id = graph.add_router(&Router::new("src", 0));
        let dst_id = graph.add_router(&Router::new("dst", 0));
        let src = graph
            .get_node(&src_id)
            .expect("source node should exist")
            .clone();
        let dst = graph
            .get_node(&dst_id)
            .expect("target node should exist")
            .clone();

        let edge = graph.connect(&src, &dst);
        assert_eq!(edge.id.as_str(), "edge::router::src_to_router::dst::edge");
        assert_eq!(edge.source, src_id);
        assert_eq!(edge.target, dst_id);
        assert_eq!(edge.direction(), ArchEdgeDirection::Directional);
    }

    #[test]
    fn graph_edge_direction_attribute_overrides_default_direction() {
        let mut graph = ArchGraph::new("mesh");
        let src_id = graph.add_router(&Router::new("src", 0));
        let dst_id = graph.add_router(&Router::new("dst", 0));
        let src = graph
            .get_node(&src_id)
            .expect("source node should exist")
            .clone();
        let dst = graph
            .get_node(&dst_id)
            .expect("target node should exist")
            .clone();

        let edge = graph.connect_with_attrs(
            &src,
            &dst,
            vec![ArchEdgeAttr::Direction(ArchEdgeDirection::Bidirectional)],
        );
        assert_eq!(edge.direction(), ArchEdgeDirection::Bidirectional);
    }

    #[test]
    fn graph_supports_node_lookup_by_id() {
        let arch: Architecture = ArchGraph::builder("mesh")
            .router(&Router::new("router", 0))
            .build()
            .into();
        let graph = arch.as_graph().expect("builder must create graph");
        let router_id = graph
            .router_ref("router")
            .expect("router node ID should be available");
        let router = graph.get_node(&router_id).expect("router node must exist");
        assert_eq!(router.name(), Some("router"));
    }

    #[test]
    fn auto_generated_ids_use_component_names() {
        let l1 = MemoryRegion::bank(MemoryBank::new(SizeExpr::Const(1024))).with_name("L1");
        let lane = Processor::new("lane").into_elem();
        let graph: Architecture = ArchGraph::builder("core")
            .mem(&l1)
            .architecture(&lane)
            .router(&Router::new("xbar", 0))
            .build()
            .into();
        let graph = graph.as_graph().expect("must build graph");

        assert!(graph.nodes.iter().any(|n| n.id.as_str() == "mem::L1::node"));
        assert!(
            graph
                .nodes
                .iter()
                .any(|n| n.id.as_str() == "arch::lane::node")
        );
        assert!(
            graph
                .nodes
                .iter()
                .any(|n| n.id.as_str() == "router::xbar::node")
        );
    }

    #[test]
    fn builder_suffixes_duplicate_node_components() {
        let r1 = Router::new("dup", 0);
        let r2 = Router::new("dup", 0);
        let arch: Architecture = ArchGraph::builder("ok")
            .router(&r1)
            .router(&r2)
            .build()
            .into();
        let graph = arch.as_graph().expect("builder must create graph");

        assert!(
            graph
                .nodes
                .iter()
                .any(|n| n.id.as_str() == "router::dup::node")
        );
        assert!(
            graph
                .nodes
                .iter()
                .any(|n| n.id.as_str() == "router::dup#2::node")
        );
    }

    #[test]
    #[should_panic(expected = "edge already exists between")]
    fn graph_rejects_duplicate_edges_between_same_nodes() {
        let mut graph = ArchGraph::new("bad");
        let src_id = graph.add_router(&Router::new("r1", 0));
        let dst_id = graph.add_router(&Router::new("r2", 0));
        let src = graph.get_node(&src_id).expect("source must exist").clone();
        let dst = graph.get_node(&dst_id).expect("target must exist").clone();
        graph.connect(&src, &dst);
        graph.connect(&src, &dst);
    }

    #[test]
    fn builder_processor_adds_connectivity_resources_and_data_movers() {
        let x = Dimension::new_int("x", 8);
        let lane = Processor::new("lane").into_elem();
        let l1 = MemoryRegion::bank(MemoryBank::new(SizeExpr::Const(1024)))
            .scale(x.as_slice())
            .with_name("L1");
        let io = MeshNetworkInterface::new(AffineMap::identity(x.as_slice()), Expr::Const(64))
            .with_data_mover(Processor::new("arr_dm").into());
        let map = AffineMap::identity(x.as_slice());
        let net = ScaleOutNetwork::mesh("l1_ring")
            .mem_region(&l1)
            .map(&map)
            .io(&io)
            .link_bandwidth(64)
            .build();
        let arr = lane
            .scale(x.as_slice())
            .with_name("mesh")
            .with_connectivity(vec![net]);

        let graph = ArchGraph::builder("top").architecture(&arr).build();

        assert!(graph.data_mover_ref("arr_dm").is_some());
        assert!(
            graph
                .resources
                .iter()
                .any(|resource| resource.id().as_str() == "l1_ring_link_0")
        );
    }

    #[test]
    fn array_materializes_scaled_memory_region_for_elements() {
        let nbank = Dimension::new_int("nbank", 16);
        let x = Dimension::new_int("x", 8);
        let y = Dimension::new_int("y", 4);
        let l1 = MemoryRegion::bank(MemoryBank::new(SizeExpr::Const(1024)))
            .scale(nbank.as_slice())
            .with_name("L1");
        let core: Architecture = ArchGraph::builder("core").mem(&l1).build().into();
        let mesh = core.scale([&x, &y]).with_name("mesh");

        let scaled_l1 = mesh
            .get_scaled_memory_region("L1")
            .expect("scaled architecture should expose mesh-wide L1");

        assert_eq!(scaled_l1.name(), Some("array_L1"));
        match scaled_l1 {
            MemoryRegion::Array {
                dims, sub_regions, ..
            } => {
                assert_eq!(
                    dims.iter().map(|d| d.name.0.as_str()).collect::<Vec<_>>(),
                    vec!["x", "y"]
                );
                assert_eq!(sub_regions.name(), Some("L1"));
                assert_eq!(sub_regions.dims()[0].name.0, "nbank");
            }
            MemoryRegion::Bank(_) => panic!("scaled L1 should be an array"),
        }
    }

    #[test]
    fn builder_memory_bank_auto_registers_resource() {
        let l1 = MemoryRegion::bank(MemoryBank::new(SizeExpr::Const(2048)).with_name("L1"));
        let graph = ArchGraph::builder("core").mem(&l1).build();

        let mem_id = graph.memory_ref("L1").expect("L1 memory node should exist");
        let memory_resource = graph
            .resources
            .iter()
            .find(|resource| resource.id().as_str() == "L1")
            .expect("L1 resource should be auto-registered");

        assert_eq!(memory_resource.capacity(), Some(2048));
        assert!(
            graph
                .node_resources(&mem_id)
                .iter()
                .any(|id| id.as_str() == "L1")
        );
    }

    #[test]
    fn builder_compute_processor_auto_registers_self_resource() {
        let lane = ComputeProcessor::builder()
            .named("lane")
            .finish()
            .into_elem();
        let graph = ArchGraph::builder("core").architecture(&lane).build();

        let lane_id = graph
            .processor_ref("lane")
            .expect("lane processor node should exist");
        let lane_resource = graph
            .resources
            .iter()
            .find(|resource| resource.id().as_str() == "lane")
            .expect("lane self resource should be auto-registered");

        assert!(lane_resource.is_exclusive());
        assert!(
            graph
                .node_resources(&lane_id)
                .iter()
                .any(|id| id.as_str() == "lane")
        );
    }
}
