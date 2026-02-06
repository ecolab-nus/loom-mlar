//! Visualization support for MLAR architectures using petgraph.
//!
//! This module provides functionality to convert MLAR architecture specifications
//! into graphs that can be exported to GraphViz DOT format for visualization.

use crate::architecture::Architecture;
use crate::core::{Dimension, MemRegion, MemoryInterconnects};
use crate::processor_aggregation::ProcessorSet;
use petgraph::dot::{Config, Dot};
use petgraph::graph::{DiGraph, NodeIndex};
use std::collections::HashMap;
use std::fmt::Write;

/// Node types in the architecture graph
#[derive(Debug, Clone)]
pub enum ArchNode {
    /// Memory region (L1, L2, DRAM, etc.)
    Memory {
        name: String,
        details: String,
    },
    /// Processing element (functional unit or lane)
    Processor {
        name: String,
        details: String,
    },
    /// Dimension node (for showing grid structure)
    Dimension {
        name: String,
        size: String,
    },
}

impl std::fmt::Display for ArchNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArchNode::Memory { name, details } => {
                if details.is_empty() {
                    write!(f, "{}", name)
                } else {
                    write!(f, "{}\\n{}", name, details)
                }
            }
            ArchNode::Processor { name, details } => {
                if details.is_empty() {
                    write!(f, "{}", name)
                } else {
                    write!(f, "{}\\n{}", name, details)
                }
            }
            ArchNode::Dimension { name, size } => {
                write!(f, "dim {}\\n[{}]", name, size)
            }
        }
    }
}

/// Edge types in the architecture graph
#[derive(Debug, Clone)]
pub enum ArchEdge {
    /// Memory interconnect mapping between regions
    MemoryInterconnect {
        name: String,
        bandwidth: usize,
        mapping: String,
    },
    /// Memory-to-processor interconnect mapping
    MemoryProcessorInterconnect {
        name: String,
        bandwidth: usize,
        mapping: String,
    },
    /// Interconnect connection
    Interconnect {
        name: String,
        bandwidth: usize,
        mapping: String,
    },
    /// Contains relationship (region contains sub-region)
    Contains,
    /// Uses relationship (processor uses memory)
    Uses,
    /// Scales across dimension
    ScaledBy,
}

impl std::fmt::Display for ArchEdge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArchEdge::MemoryInterconnect {
                name,
                bandwidth,
                mapping,
            } => write!(f, "{}\\n{}\\n({} B/cycle)", name, mapping, bandwidth),
            ArchEdge::MemoryProcessorInterconnect {
                name,
                bandwidth,
                mapping,
            } => write!(f, "{}\\n{}\\n({} B/cycle)", name, mapping, bandwidth),
            ArchEdge::Interconnect { name, bandwidth, mapping } => {
                write!(f, "{}\\n{}\\n({} B/cycle)", name, mapping, bandwidth)
            }
            ArchEdge::Contains => write!(f, "contains"),
            ArchEdge::Uses => write!(f, "uses"),
            ArchEdge::ScaledBy => write!(f, "scaled"),
        }
    }
}

/// Architecture graph for visualization
pub type ArchGraph = DiGraph<ArchNode, ArchEdge>;

/// Builder for creating architecture visualization graphs
pub struct ArchVisualizer {
    graph: ArchGraph,
    memory_nodes: HashMap<String, NodeIndex>,
    processor_nodes: HashMap<String, NodeIndex>,
    node_counter: usize,
}

impl ArchVisualizer {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            memory_nodes: HashMap::new(),
            processor_nodes: HashMap::new(),
            node_counter: 0,
        }
    }

    /// Generate a unique name for anonymous memory regions
    fn next_mem_name(&mut self) -> String {
        self.node_counter += 1;
        format!("mem_{}", self.node_counter)
    }

    /// Format dimensions for display in compact form: "x 4 (dram_dim)"
    fn format_dimensions_compact(dims: &[Dimension]) -> String {
        if dims.is_empty() {
            return String::new();
        }
        dims.iter()
            .map(|d| format!("x {} ({})", d.size, d.name))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Format dimensions for display in detailed form: "dram_dim:4"
    fn format_dimensions(dims: &[Dimension]) -> String {
        dims.iter()
            .map(|d| format!("{}:{}", d.name, d.size))
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Add a memory region to the graph, returning its node index
    fn add_memory_region(&mut self, region: &MemRegion, suggested_name: Option<&str>) -> NodeIndex {
        let name = suggested_name
            .map(|s| s.to_string())
            .unwrap_or_else(|| self.next_mem_name());

        // Check if we already have this named region
        if let Some(&idx) = self.memory_nodes.get(&name) {
            return idx;
        }

        let (node_name, details) = match region {
            MemRegion::Bank(bank) => {
                let details = format!(
                    "block_size: {}\\nnum_blocks: {}",
                    bank.block_size, bank.num_blocks
                );
                (name.clone(), details)
            }
            MemRegion::Indexed { indices, sub_region } => {
                let dims_str = Self::format_dimensions_compact(indices);
                let sub_details = match sub_region.as_ref() {
                    MemRegion::Bank(bank) => {
                        format!(
                            "block_size: {}\\nnum_blocks: {}",
                            bank.block_size, bank.num_blocks
                        )
                    }
                    MemRegion::Indexed { indices: inner_indices, .. } => {
                        format!("{}", Self::format_dimensions_compact(inner_indices))
                    }
                };
                // Format: "L2 x 4 (dram_dim)"
                let label = format!("{} {}", name, dims_str);
                (label, sub_details)
            }
        };

        let idx = self.graph.add_node(ArchNode::Memory {
            name: node_name,
            details,
        });
        self.memory_nodes.insert(name, idx);
        idx
    }

    /// Add a processor set to the graph
    fn add_processor_set(&mut self, set: &ProcessorSet) -> NodeIndex {
        let name = set.processor_name().to_string();

        if let Some(&idx) = self.processor_nodes.get(&name) {
            return idx;
        }

        let indices = set.indices();
        let details = if indices.is_empty() {
            String::new()
        } else {
            Self::format_dimensions_compact(indices)
        };

        // Format: "matmul_lane x 32 (warp_dim)"
        let label = if details.is_empty() {
            name.clone()
        } else {
            format!("{} {}", name, details)
        };

        let idx = self.graph.add_node(ArchNode::Processor {
            name: label,
            details: String::new(),
        });
        self.processor_nodes.insert(name, idx);
        idx
    }

    /// Add a dimension node to the graph
    /// Build a graph from memory interconnects (e.g., GPU memory hierarchy)
    pub fn from_memory_interconnects(interconnects: &[MemoryInterconnects]) -> Self {
        let mut viz = Self::new();

        for ic in interconnects {
            // Extract source/target names from interconnect name (e.g., "DRAM_to_L2" -> "DRAM", "L2")
            let (src_name, tgt_name) = extract_src_tgt_names(&ic.name);

            // Add source regions
            let source_indices: Vec<_> = ic
                .sources
                .iter()
                .enumerate()
                .map(|(i, src)| {
                    let name = if ic.sources.len() == 1 {
                        src_name.clone()
                    } else {
                        format!("{}_{}", src_name, i)
                    };
                    viz.add_memory_region(src, Some(&name))
                })
                .collect();

            // Add target regions
            let target_indices: Vec<_> = ic
                .targets
                .iter()
                .enumerate()
                .map(|(i, tgt)| {
                    let name = if ic.targets.len() == 1 {
                        tgt_name.clone()
                    } else {
                        format!("{}_{}", tgt_name, i)
                    };
                    viz.add_memory_region(tgt, Some(&name))
                })
                .collect();

            // Add edges from sources to target
            let mapping = format_affine_map_detailed(&ic.map);
            for src_idx in source_indices {
                for tgt_idx in &target_indices {
                    viz.graph.add_edge(
                        src_idx,
                        *tgt_idx,
                        ArchEdge::MemoryInterconnect {
                            name: ic.name.clone(),
                            bandwidth: ic.bandwidth,
                            mapping: mapping.clone(),
                        },
                    );
                }
            }
        }

        viz
    }

    /// Build a complete graph from an Architecture
    pub fn from_architecture(arch: &Architecture) -> Self {
        let mut viz = Self::new();

        // Add processor sets
        for set in &arch.processor_sets {
            viz.add_processor_set(set);
        }

        // Add memory interconnects with mapping edges
        for ic in &arch.memory_interconnects {
            // Extract source/target names from interconnect name (e.g., "DRAM_to_L2" -> "DRAM", "L2")
            let (src_name, tgt_name) = extract_src_tgt_names(&ic.name);

            let source_indices: Vec<_> = ic
                .sources
                .iter()
                .enumerate()
                .map(|(i, src)| {
                    let name = if ic.sources.len() == 1 {
                        src_name.clone()
                    } else {
                        format!("{}_{}", src_name, i)
                    };
                    viz.add_memory_region(src, Some(&name))
                })
                .collect();

            let target_indices: Vec<_> = ic
                .targets
                .iter()
                .enumerate()
                .map(|(i, tgt)| {
                    let name = if ic.targets.len() == 1 {
                        tgt_name.clone()
                    } else {
                        format!("{}_{}", tgt_name, i)
                    };
                    viz.add_memory_region(tgt, Some(&name))
                })
                .collect();

            let mapping = format_affine_map_detailed(&ic.map);
            for src_idx in source_indices {
                for tgt_idx in &target_indices {
                    viz.graph.add_edge(
                        src_idx,
                        *tgt_idx,
                        ArchEdge::MemoryInterconnect {
                            name: ic.name.clone(),
                            bandwidth: ic.bandwidth,
                            mapping: mapping.clone(),
                        },
                    );
                }
            }
        }

        // Add memory-processor interconnects
        for ic in &arch.memory_processor_interconnects {
            // Extract source name from interconnect name (e.g., "RF_to_MatLane" -> "RF")
            let (src_name, _) = extract_src_tgt_names(&ic.name);

            // Add source memory region
            let src_idx = viz.add_memory_region(&ic.source, Some(&src_name));

            // Add target processor set
            let tgt_idx = viz.add_processor_set(&ic.target);

            // Add edge from memory to processor
            let mapping = format_affine_map_detailed(&ic.map);
            viz.graph.add_edge(
                src_idx,
                tgt_idx,
                ArchEdge::MemoryProcessorInterconnect {
                    name: ic.name.clone(),
                    bandwidth: ic.bandwidth,
                    mapping,
                },
            );
        }

        // Add interconnects
        for ic in &arch.interconnects {
            // Create nodes for the interconnect grid
            let grid_name = format!("{}_grid", ic.name);
            let grid_dims = Self::format_dimensions(&ic.grid);
            let grid_idx = viz.graph.add_node(ArchNode::Memory {
                name: grid_name,
                details: format!("grid [{}]", grid_dims),
            });

            // Add self-loop to represent the interconnect topology
            let mapping = format_affine_map(&ic.affine_map);
            viz.graph.add_edge(
                grid_idx,
                grid_idx,
                ArchEdge::Interconnect {
                    name: ic.name.clone(),
                    bandwidth: ic.bandwidth,
                    mapping,
                },
            );
        }

        viz
    }

    /// Build a simplified hierarchical view for memory systems
    pub fn from_architecture_hierarchical(arch: &Architecture) -> Self {
        let mut viz = Self::new();

        // Create a linear chain of memory levels
        // Extract memory level names from interconnect names
        let mut levels: Vec<(String, NodeIndex)> = Vec::new();

        for ic in &arch.memory_interconnects {
            // Extract source/target names from interconnect name (e.g., "DRAM_to_L2" -> "DRAM", "L2")
            let (src_name, tgt_name) = extract_src_tgt_names(&ic.name);

            // Add source
            if let Some(src) = ic.sources.first() {
                if !levels.iter().any(|(n, _)| *n == src_name) {
                    let idx = viz.add_memory_region(src, Some(&src_name));
                    levels.push((src_name.clone(), idx));
                }
            }

            // Add target
            if let Some(tgt) = ic.targets.first() {
                if !levels.iter().any(|(n, _)| *n == tgt_name) {
                    let idx = viz.add_memory_region(tgt, Some(&tgt_name));
                    levels.push((tgt_name.clone(), idx));
                }
            }
        }

        // Add edges for memory interconnects
        for ic in &arch.memory_interconnects {
            let (src_name, tgt_name) = extract_src_tgt_names(&ic.name);

            if let (Some((_, src_idx)), Some((_, tgt_idx))) = (
                levels.iter().find(|(n, _)| *n == src_name),
                levels.iter().find(|(n, _)| *n == tgt_name),
            ) {
                viz.graph.add_edge(
                    *src_idx,
                    *tgt_idx,
                    ArchEdge::MemoryInterconnect {
                        name: ic.name.clone(),
                        bandwidth: ic.bandwidth,
                        mapping: format_affine_map_detailed(&ic.map),
                    },
                );
            }
        }

        // Add processor sets and memory-processor interconnects
        for ic in &arch.memory_processor_interconnects {
            let (src_name, _) = extract_src_tgt_names(&ic.name);
            let proc_idx = viz.add_processor_set(&ic.target);

            if let Some((_, src_idx)) = levels.iter().find(|(n, _)| *n == src_name) {
                viz.graph.add_edge(
                    *src_idx,
                    proc_idx,
                    ArchEdge::MemoryProcessorInterconnect {
                        name: ic.name.clone(),
                        bandwidth: ic.bandwidth,
                        mapping: format_affine_map_detailed(&ic.map),
                    },
                );
            }
        }

        viz
    }

    /// Export the graph to DOT format
    pub fn to_dot(&self) -> String {
        format!("{:?}", Dot::with_config(&self.graph, &[Config::EdgeNoLabel]))
    }

    /// Export the graph to DOT format with edge labels
    pub fn to_dot_with_labels(&self) -> String {
        format!("{:?}", Dot::new(&self.graph))
    }

    /// Export to a customized DOT format with better styling
    pub fn to_dot_styled(&self, title: &str) -> String {
        let mut dot = String::new();
        writeln!(dot, "digraph \"{}\" {{", title).unwrap();
        writeln!(dot, "    rankdir=TB;").unwrap();
        writeln!(dot, "    node [fontname=\"Helvetica\"];").unwrap();
        writeln!(dot, "    edge [fontname=\"Helvetica\", fontsize=10];").unwrap();
        writeln!(dot).unwrap();

        // Add nodes with styling based on type
        for idx in self.graph.node_indices() {
            let node = &self.graph[idx];
            let (shape, color, style) = match node {
                ArchNode::Memory { .. } => ("box", "lightblue", "filled"),
                ArchNode::Processor { .. } => ("ellipse", "lightgreen", "filled"),
                ArchNode::Dimension { .. } => ("diamond", "lightyellow", "filled"),
            };
            writeln!(
                dot,
                "    {} [label=\"{}\", shape={}, fillcolor={}, style={}];",
                idx.index(),
                node,
                shape,
                color,
                style
            )
            .unwrap();
        }

        writeln!(dot).unwrap();

        // Add edges with labels
        for edge in self.graph.edge_indices() {
            let (src, tgt) = self.graph.edge_endpoints(edge).unwrap();
            let weight = &self.graph[edge];
            let (label, color, style) = match weight {
                ArchEdge::MemoryInterconnect {
                    name,
                    bandwidth,
                    mapping,
                } => (
                    format!("{}\\n{}\\n{} B/cycle", name, mapping, bandwidth),
                    "blue",
                    "solid",
                ),
                ArchEdge::MemoryProcessorInterconnect {
                    name,
                    bandwidth,
                    mapping,
                } => (
                    format!("{}\\n{}\\n{} B/cycle", name, mapping, bandwidth),
                    "purple",
                    "solid",
                ),
                ArchEdge::Interconnect {
                    name,
                    bandwidth,
                    mapping,
                } => (
                    format!("{}\\n{}\\n{} B/cycle", name, mapping, bandwidth),
                    "red",
                    "dashed",
                ),
                ArchEdge::Contains => ("contains".to_string(), "gray", "dotted"),
                ArchEdge::Uses => ("uses".to_string(), "green", "dotted"),
                ArchEdge::ScaledBy => ("scaled".to_string(), "orange", "dashed"),
            };
            writeln!(
                dot,
                "    {} -> {} [label=\"{}\", color={}, style={}];",
                src.index(),
                tgt.index(),
                label,
                color,
                style
            )
            .unwrap();
        }

        writeln!(dot, "}}").unwrap();
        dot
    }

    /// Get the underlying graph for further manipulation
    pub fn graph(&self) -> &ArchGraph {
        &self.graph
    }

    /// Get mutable access to the underlying graph
    pub fn graph_mut(&mut self) -> &mut ArchGraph {
        &mut self.graph
    }
}

impl Default for ArchVisualizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Format an affine map for display
fn format_affine_map(map: &crate::interconnect::AffineMap) -> String {
    let src_dims = map.source_dim_names();
    let results: Vec<_> = map.results.iter().map(format_affine_expr).collect();
    format!(
        "({}) -> ({})",
        src_dims.join(", "),
        results.join(", ")
    )
}

/// Format an affine map with dimension details for display
fn format_affine_map_detailed(map: &crate::interconnect::AffineMap) -> String {
    let src_info = map
        .source_dims
        .iter()
        .map(|d| format!("{}:{}", d.name, d.size))
        .collect::<Vec<_>>()
        .join(", ");
    
    let tgt_info = map
        .target_dims
        .iter()
        .map(|d| format!("{}:{}", d.name, d.size))
        .collect::<Vec<_>>()
        .join(", ");
    
    let results: Vec<_> = map.results.iter().map(format_affine_expr).collect();
    format!(
        "[{}] -> [{}]\\n({})",
        src_info,
        tgt_info,
        results.join(", ")
    )
}

/// Format an affine expression for display
fn format_affine_expr(expr: &crate::interconnect::AffineExpr) -> String {
    use crate::interconnect::AffineExpr;
    match expr {
        AffineExpr::Dim(i) => format!("d{}", i),
        AffineExpr::Constant(c) => c.to_string(),
        AffineExpr::Add(a, b) => format!("({} + {})", format_affine_expr(a), format_affine_expr(b)),
        AffineExpr::Mul(a, b) => format!("({} * {})", format_affine_expr(a), format_affine_expr(b)),
        AffineExpr::Mod(a, b) => {
            format!("({} mod {})", format_affine_expr(a), format_affine_expr(b))
        }
        AffineExpr::CeilDiv(a, b) => format!(
            "({} ceildiv {})",
            format_affine_expr(a),
            format_affine_expr(b)
        ),
    }
}

/// Extract source and target names from interconnect name (e.g., "DRAM_to_L2" -> ("DRAM", "L2"))
fn extract_src_tgt_names(ic_name: &str) -> (String, String) {
    if ic_name.contains("_to_") {
        let mut parts = ic_name.split("_to_");
        let src = parts.next().unwrap_or(ic_name).to_string();
        let tgt = parts.next().unwrap_or(ic_name).to_string();
        (src, tgt)
    } else {
        // Fallback: use ic_name as prefix
        (format!("{}_src", ic_name), format!("{}_tgt", ic_name))
    }
}

/// Convenience function to generate DOT from an architecture
pub fn architecture_to_dot(arch: &Architecture) -> String {
    let viz = ArchVisualizer::from_architecture(arch);
    viz.to_dot_styled(&arch.name)
}

/// Convenience function to generate DOT from memory interconnects
pub fn memory_hierarchy_to_dot(name: &str, interconnects: &[MemoryInterconnects]) -> String {
    let viz = ArchVisualizer::from_memory_interconnects(interconnects);
    viz.to_dot_styled(name)
}

/// Convenience function to generate expanded DOT from an architecture
/// Shows all instances of memory regions and processors, with edges based on affine mapping
pub fn architecture_to_dot_expanded(arch: &Architecture) -> String {
    to_dot_expanded(arch)
}

/// Generate an expanded DOT visualization showing all instances
fn to_dot_expanded(arch: &Architecture) -> String {
    use crate::core::Size;
    
    let mut dot = String::new();
    writeln!(dot, "digraph \"{}\" {{", arch.name).unwrap();
    writeln!(dot, "    rankdir=TB;").unwrap();
    writeln!(dot, "    node [fontname=\"Helvetica\"];").unwrap();
    writeln!(dot, "    edge [fontname=\"Helvetica\", fontsize=10];").unwrap();
    writeln!(dot).unwrap();
    
    // Track node IDs for each instance
    let mut node_id = 0;
    let mut memory_instance_nodes: HashMap<String, Vec<usize>> = HashMap::new();
    let mut processor_instance_nodes: HashMap<String, Vec<usize>> = HashMap::new();
    
    // Helper function to get dimension size as usize
    fn get_dim_size(size: &Size) -> Option<usize> {
        match size {
            Size::Int(n) => Some(*n),
            Size::Sym(_) => None, // Can't expand symbolic sizes
        }
    }
    
    // Helper function to extract dimensions from a MemRegion
    fn get_region_dims(region: &MemRegion) -> Vec<&Dimension> {
        match region {
            MemRegion::Indexed { indices, .. } => indices.iter().collect(),
            MemRegion::Bank(_) => vec![],
        }
    }
    
    // Helper function to get bank info
    fn get_bank_info(region: &MemRegion) -> Option<(String, String)> {
        match region {
            MemRegion::Bank(bank) => Some((bank.block_size.to_string(), bank.num_blocks.to_string())),
            MemRegion::Indexed { sub_region, .. } => get_bank_info(sub_region),
        }
    }
    
    // Collect all unique memory levels from interconnects
    let mut memory_levels: Vec<(String, &MemRegion)> = Vec::new();
    let mut seen_names: std::collections::HashSet<String> = std::collections::HashSet::new();
    
    for ic in &arch.memory_interconnects {
        let (src_name, tgt_name) = extract_src_tgt_names(&ic.name);
        
        if let Some(src) = ic.sources.first() {
            if !seen_names.contains(&src_name) {
                memory_levels.push((src_name.clone(), src));
                seen_names.insert(src_name);
            }
        }
        if let Some(tgt) = ic.targets.first() {
            if !seen_names.contains(&tgt_name) {
                memory_levels.push((tgt_name.clone(), tgt));
                seen_names.insert(tgt_name);
            }
        }
    }
    
    for ic in &arch.memory_processor_interconnects {
        let (src_name, _) = extract_src_tgt_names(&ic.name);
        if !seen_names.contains(&src_name) {
            memory_levels.push((src_name.clone(), &ic.source));
            seen_names.insert(src_name);
        }
    }
    
    // Create subgraphs for each memory level with all instances
    for (level_name, region) in &memory_levels {
        let dims = get_region_dims(region);
        let bank_info = get_bank_info(region);
        
        // Calculate total instances
        let mut total_instances = 1usize;
        let mut expandable = true;
        for dim in &dims {
            if let Some(size) = get_dim_size(&dim.size) {
                total_instances *= size;
            } else {
                expandable = false;
                break;
            }
        }
        
        if !expandable || total_instances == 0 {
            // Create single node for symbolic/non-expandable regions
            let label = format!("{}\\n[symbolic]", level_name);
            writeln!(dot, "    {} [label=\"{}\", shape=box, fillcolor=lightblue, style=filled];", 
                node_id, label).unwrap();
            memory_instance_nodes.insert(level_name.clone(), vec![node_id]);
            node_id += 1;
        } else {
            // Create subgraph cluster for this memory level
            writeln!(dot, "    subgraph cluster_{} {{", level_name).unwrap();
            writeln!(dot, "        label=\"{}\";", level_name).unwrap();
            writeln!(dot, "        style=rounded;").unwrap();
            writeln!(dot, "        bgcolor=\"#E8F4FD\";").unwrap();
            
            let mut instance_ids = Vec::new();
            for i in 0..total_instances {
                let bank_label = if let Some((bs, nb)) = &bank_info {
                    format!("{}[{}]\\nbs:{} nb:{}", level_name, i, bs, nb)
                } else {
                    format!("{}[{}]", level_name, i)
                };
                writeln!(dot, "        {} [label=\"{}\", shape=box, fillcolor=lightblue, style=filled];",
                    node_id, bank_label).unwrap();
                instance_ids.push(node_id);
                node_id += 1;
            }
            
            writeln!(dot, "    }}").unwrap();
            memory_instance_nodes.insert(level_name.clone(), instance_ids);
        }
    }
    
    // Create processor instances
    for ic in &arch.memory_processor_interconnects {
        let (_, _tgt_name) = extract_src_tgt_names(&ic.name);
        let proc_name = ic.target.processor_name();
        let proc_dims = ic.target.indices();
        
        let mut total_instances = 1usize;
        let mut expandable = true;
        for dim in proc_dims {
            if let Some(size) = get_dim_size(&dim.size) {
                total_instances *= size;
            } else {
                expandable = false;
                break;
            }
        }
        
        if processor_instance_nodes.contains_key(proc_name) {
            continue;
        }
        
        if !expandable || total_instances == 0 {
            let label = format!("{}\\n[symbolic]", proc_name);
            writeln!(dot, "    {} [label=\"{}\", shape=ellipse, fillcolor=lightgreen, style=filled];",
                node_id, label).unwrap();
            processor_instance_nodes.insert(proc_name.to_string(), vec![node_id]);
            node_id += 1;
        } else {
            // Create subgraph cluster for processors
            writeln!(dot, "    subgraph cluster_{} {{", proc_name).unwrap();
            writeln!(dot, "        label=\"{}\";", proc_name).unwrap();
            writeln!(dot, "        style=rounded;").unwrap();
            writeln!(dot, "        bgcolor=\"#E8FDE8\";").unwrap();
            
            let mut instance_ids = Vec::new();
            for i in 0..total_instances {
                let label = format!("{}[{}]", proc_name, i);
                writeln!(dot, "        {} [label=\"{}\", shape=ellipse, fillcolor=lightgreen, style=filled];",
                    node_id, label).unwrap();
                instance_ids.push(node_id);
                node_id += 1;
            }
            
            writeln!(dot, "    }}").unwrap();
            processor_instance_nodes.insert(proc_name.to_string(), instance_ids);
        }
    }
    
    writeln!(dot).unwrap();
    
    // Draw edges based on affine maps
    for ic in &arch.memory_interconnects {
        let (src_name, tgt_name) = extract_src_tgt_names(&ic.name);
        
        let src_nodes = match memory_instance_nodes.get(&src_name) {
            Some(nodes) => nodes,
            None => continue,
        };
        let tgt_nodes = match memory_instance_nodes.get(&tgt_name) {
            Some(nodes) => nodes,
            None => continue,
        };
        
        let src_count = src_nodes.len();
        let tgt_count = tgt_nodes.len();
        
        if src_count == 0 || tgt_count == 0 {
            continue;
        }
        
        // For each source instance, compute which target instances it connects to
        for (src_idx, &src_node) in src_nodes.iter().enumerate() {
            // Apply the affine map to get the base target index
            let target_base = ic.map.apply(&[src_idx])[0] as usize;
            
            // Calculate fan-out: how many targets each source connects to
            // This is based on the ratio of target size to source size
            let fan_out = if src_count > 0 { tgt_count / src_count } else { 1 };
            
            // Connect to fan_out consecutive targets starting at target_base
            for offset in 0..fan_out {
                let tgt_idx = target_base + offset;
                if tgt_idx < tgt_nodes.len() {
                    let tgt_node = tgt_nodes[tgt_idx];
                    writeln!(dot, "    {} -> {} [color=blue];", src_node, tgt_node).unwrap();
                }
            }
        }
    }
    
    // Draw edges for memory-processor interconnects
    for ic in &arch.memory_processor_interconnects {
        let (src_name, _) = extract_src_tgt_names(&ic.name);
        let proc_name = ic.target.processor_name();
        
        let src_nodes = match memory_instance_nodes.get(&src_name) {
            Some(nodes) => nodes,
            None => continue,
        };
        let proc_nodes = match processor_instance_nodes.get(proc_name) {
            Some(nodes) => nodes,
            None => continue,
        };
        
        let src_count = src_nodes.len();
        let proc_count = proc_nodes.len();
        
        if src_count == 0 || proc_count == 0 {
            continue;
        }
        
        // For each source instance, compute which processor instances it connects to
        for (src_idx, &src_node) in src_nodes.iter().enumerate() {
            let target_base = ic.map.apply(&[src_idx])[0] as usize;
            let fan_out = if src_count > 0 { proc_count / src_count } else { 1 };
            
            for offset in 0..fan_out {
                let tgt_idx = target_base + offset;
                if tgt_idx < proc_nodes.len() {
                    let tgt_node = proc_nodes[tgt_idx];
                    writeln!(dot, "    {} -> {} [color=purple];", src_node, tgt_node).unwrap();
                }
            }
        }
    }
    
    writeln!(dot, "}}").unwrap();
    dot
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;

    #[test]
    fn test_simple_memory_visualization() {
        let bank = Bank {
            block_size: Size::int(1024),
            num_blocks: Size::int(16),
        };

        let region = MemRegion::bank(bank);

        let mut viz = ArchVisualizer::new();
        viz.add_memory_region(&region, Some("test_mem"));

        let dot = viz.to_dot_styled("test");
        assert!(dot.contains("test_mem"));
    }
}
