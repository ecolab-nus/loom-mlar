//! Visualization support for MLAR architectures using petgraph.
//!
//! This module provides functionality to convert MLAR architecture specifications
//! into graphs that can be exported to GraphViz DOT format for visualization.

use crate::architecture::Architecture;
use crate::core::{Dimension, MemRegion, MemoryAggregation};
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
    /// Data movement (memory aggregation)
    DataFlow {
        name: String,
        bandwidth: usize,
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
            ArchEdge::DataFlow { name, bandwidth } => {
                write!(f, "{}\\n({} B/cycle)", name, bandwidth)
            }
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
    dimension_nodes: HashMap<String, NodeIndex>,
    node_counter: usize,
}

impl ArchVisualizer {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            memory_nodes: HashMap::new(),
            processor_nodes: HashMap::new(),
            dimension_nodes: HashMap::new(),
            node_counter: 0,
        }
    }

    /// Generate a unique name for anonymous memory regions
    fn next_mem_name(&mut self) -> String {
        self.node_counter += 1;
        format!("mem_{}", self.node_counter)
    }

    /// Format dimensions for display
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
                let dims = Self::format_dimensions(indices);
                let sub_details = match sub_region.as_ref() {
                    MemRegion::Bank(bank) => {
                        format!(
                            "block_size: {}\\nnum_blocks: {}",
                            bank.block_size, bank.num_blocks
                        )
                    }
                    MemRegion::Indexed { indices: inner_indices, .. } => {
                        format!("indexed [{}]", Self::format_dimensions(inner_indices))
                    }
                };
                (name.clone(), format!("[{}]\\n{}", dims, sub_details))
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
            "single instance".to_string()
        } else {
            format!("[{}]", Self::format_dimensions(indices))
        };

        let idx = self.graph.add_node(ArchNode::Processor {
            name: name.clone(),
            details,
        });
        self.processor_nodes.insert(name, idx);
        idx
    }

    /// Add a dimension node to the graph
    fn add_dimension(&mut self, dim: &Dimension) -> NodeIndex {
        if let Some(&idx) = self.dimension_nodes.get(&dim.name) {
            return idx;
        }

        let idx = self.graph.add_node(ArchNode::Dimension {
            name: dim.name.clone(),
            size: dim.size.to_string(),
        });
        self.dimension_nodes.insert(dim.name.clone(), idx);
        idx
    }

    /// Build a graph from memory aggregations (e.g., GPU memory hierarchy)
    pub fn from_memory_aggregations(aggregations: &[MemoryAggregation]) -> Self {
        let mut viz = Self::new();

        for agg in aggregations {
            // Add source regions
            let source_indices: Vec<_> = agg
                .sources
                .iter()
                .enumerate()
                .map(|(i, src)| {
                    let name = format!("{}_src_{}", agg.name, i);
                    viz.add_memory_region(src, Some(&name))
                })
                .collect();

            // Add target region
            let target_name = format!("{}_tgt", agg.name);
            let target_idx = viz.add_memory_region(&agg.target, Some(&target_name));

            // Add edges from sources to target
            for src_idx in source_indices {
                viz.graph.add_edge(
                    src_idx,
                    target_idx,
                    ArchEdge::DataFlow {
                        name: agg.name.clone(),
                        bandwidth: agg.bandwidth,
                    },
                );
            }
        }

        viz
    }

    /// Build a complete graph from an Architecture
    pub fn from_architecture(arch: &Architecture) -> Self {
        let mut viz = Self::new();

        // Add dimensions
        for dim in &arch.dimensions {
            viz.add_dimension(dim);
        }

        // Add processor sets
        for set in &arch.processor_sets {
            viz.add_processor_set(set);
        }

        // Add memory aggregations with data flow edges
        for agg in &arch.memory_aggregations {
            let source_indices: Vec<_> = agg
                .sources
                .iter()
                .enumerate()
                .map(|(i, src)| {
                    let name = format!("{}_src_{}", agg.name, i);
                    viz.add_memory_region(src, Some(&name))
                })
                .collect();

            let target_name = format!("{}_tgt", agg.name);
            let target_idx = viz.add_memory_region(&agg.target, Some(&target_name));

            for src_idx in source_indices {
                viz.graph.add_edge(
                    src_idx,
                    target_idx,
                    ArchEdge::DataFlow {
                        name: agg.name.clone(),
                        bandwidth: agg.bandwidth,
                    },
                );
            }
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
        // Parse memory aggregation names to determine hierarchy
        let mut levels: Vec<(&str, NodeIndex)> = Vec::new();

        for agg in &arch.memory_aggregations {
            // Add source
            if let Some(src) = agg.sources.first() {
                let src_name = extract_level_name(&agg.name, true);
                if !levels.iter().any(|(n, _)| *n == src_name) {
                    let idx = viz.add_memory_region(src, Some(src_name));
                    levels.push((src_name, idx));
                }
            }

            // Add target
            let tgt_name = extract_level_name(&agg.name, false);
            if !levels.iter().any(|(n, _)| *n == tgt_name) {
                let idx = viz.add_memory_region(&agg.target, Some(tgt_name));
                levels.push((tgt_name, idx));
            }
        }

        // Add edges
        for agg in &arch.memory_aggregations {
            let src_name = extract_level_name(&agg.name, true);
            let tgt_name = extract_level_name(&agg.name, false);

            if let (Some((_, src_idx)), Some((_, tgt_idx))) = (
                levels.iter().find(|(n, _)| *n == src_name),
                levels.iter().find(|(n, _)| *n == tgt_name),
            ) {
                viz.graph.add_edge(
                    *src_idx,
                    *tgt_idx,
                    ArchEdge::DataFlow {
                        name: agg.name.clone(),
                        bandwidth: agg.bandwidth,
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
                ArchEdge::DataFlow { name, bandwidth } => {
                    (format!("{}\\n{} B/cycle", name, bandwidth), "blue", "solid")
                }
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
    let dims: Vec<_> = (0..map.num_dims).map(|i| format!("d{}", i)).collect();
    let results: Vec<_> = map.results.iter().map(format_affine_expr).collect();
    format!("({}) -> ({})", dims.join(", "), results.join(", "))
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

/// Extract level name from aggregation name (e.g., "DRAM_bus_output" -> "DRAM")
fn extract_level_name(agg_name: &str, is_source: bool) -> &str {
    // Common patterns: "X_bus_output", "X_bus_input"
    if agg_name.contains("_bus_output") {
        agg_name.split("_bus_output").next().unwrap_or(agg_name)
    } else if agg_name.contains("_bus_input") {
        if is_source {
            // For input, source is the buffer
            "buffer"
        } else {
            agg_name.split("_bus_input").next().unwrap_or(agg_name)
        }
    } else {
        agg_name
    }
}

/// Convenience function to generate DOT from an architecture
pub fn architecture_to_dot(arch: &Architecture) -> String {
    let viz = ArchVisualizer::from_architecture(arch);
    viz.to_dot_styled(&arch.name)
}

/// Convenience function to generate DOT from memory aggregations
pub fn memory_hierarchy_to_dot(name: &str, aggregations: &[MemoryAggregation]) -> String {
    let viz = ArchVisualizer::from_memory_aggregations(aggregations);
    viz.to_dot_styled(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;

    #[test]
    fn test_simple_memory_visualization() {
        let bank = Bank::builder()
            .block_size(1024_usize)
            .num_blocks(16_usize)
            .build();

        let region = MemRegion::bank(bank);

        let mut viz = ArchVisualizer::new();
        viz.add_memory_region(&region, Some("test_mem"));

        let dot = viz.to_dot_styled("test");
        assert!(dot.contains("test_mem"));
    }
}
