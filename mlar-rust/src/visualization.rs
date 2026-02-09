//! Visualization support for MLAR architectures using petgraph.
//!
//! Converts MLAR architecture specifications into graphs exportable to GraphViz DOT format.

use crate::architecture::Architecture;
use crate::core::{
    AffineExpr, AffineMap, Dimension, Endpoint, Link, MemoryBank, MemoryRegion, Processor,
};
use petgraph::dot::{Config, Dot};
use petgraph::graph::{DiGraph, NodeIndex};
use std::collections::HashMap;
use std::fmt::Write;

/// Node types in the architecture graph
#[derive(Debug, Clone)]
pub enum ArchNode {
    Memory { name: String, details: String },
    Processor { name: String, details: String },
    Dimension { name: String, size: String },
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
    /// A Link edge (memory-memory or memory-processor)
    Link {
        name: String,
        bandwidth: String,
        mapping: String,
    },
    /// Contains relationship (region contains sub-region)
    Contains,
    /// Scales across dimension
    ScaledBy,
}

impl std::fmt::Display for ArchEdge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArchEdge::Link {
                name,
                bandwidth,
                mapping,
            } => write!(f, "{}\\n{}\\n({} B/cycle)", name, mapping, bandwidth),
            ArchEdge::Contains => write!(f, "contains"),
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

    /// Add a memory region to the graph, returning its node index
    fn add_memory_region(
        &mut self,
        region: &MemoryRegion,
        suggested_name: Option<&str>,
    ) -> NodeIndex {
        let name = suggested_name
            .or_else(|| region.name())
            .map(|s| s.to_string())
            .unwrap_or_else(|| self.next_mem_name());

        if let Some(&idx) = self.memory_nodes.get(&name) {
            return idx;
        }

        let (node_name, details) = match region {
            MemoryRegion::Bank(bank) => {
                let details = format_bank_details(bank);
                (name.clone(), details)
            }
            MemoryRegion::Replicated { dims, elem, .. } => {
                let dims_str = Self::format_dimensions_compact(dims);
                let sub_details = match elem.as_ref() {
                    MemoryRegion::Bank(bank) => format_bank_details(bank),
                    MemoryRegion::Replicated { dims: inner_dims, .. } => {
                        Self::format_dimensions_compact(inner_dims)
                    }
                    MemoryRegion::Group { .. } => "group".to_string(),
                };
                let label = format!("{} {}", name, dims_str);
                (label, sub_details)
            }
            MemoryRegion::Group { name: gname, .. } => {
                let label = gname.clone().unwrap_or_else(|| name.clone());
                (label, "group".to_string())
            }
        };

        let idx = self.graph.add_node(ArchNode::Memory {
            name: node_name,
            details,
        });
        self.memory_nodes.insert(name, idx);
        idx
    }

    /// Add a processor to the graph, returning its node index
    fn add_processor(&mut self, proc: &Processor, suggested_name: &str) -> NodeIndex {
        let name = suggested_name.to_string();

        if let Some(&idx) = self.processor_nodes.get(&name) {
            return idx;
        }

        let dims = collect_all_dims(proc);
        let details = if dims.is_empty() {
            String::new()
        } else {
            Self::format_dimensions_compact(&dims)
        };

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

    /// Build a graph from Links (e.g., memory hierarchy).
    /// Uses the actual MemoryRegion/Processor data embedded in each endpoint.
    pub fn from_links(links: &[Link]) -> Self {
        let mut viz = Self::new();

        for link in links {
            let src_idx = viz.add_endpoint(&link.src);
            let dst_idx = viz.add_endpoint(&link.dst);

            let mapping = format_affine_map_detailed(&link.map);
            let bw = format_expr_short(&link.bandwidth);
            viz.graph.add_edge(
                src_idx,
                dst_idx,
                ArchEdge::Link {
                    name: link.name.clone(),
                    bandwidth: bw,
                    mapping,
                },
            );
        }

        viz
    }

    /// Add a node from an Endpoint, using the embedded data for details.
    fn add_endpoint(&mut self, endpoint: &Endpoint) -> NodeIndex {
        match endpoint {
            Endpoint::Mem(region) => self.add_memory_region(region, region.name()),
            Endpoint::Proc(proc) => {
                let name = proc.name().unwrap_or("unnamed");
                self.add_processor(proc, name)
            }
        }
    }

    /// Build a complete graph from an Architecture.
    /// Adds nodes from architecture components, then edges from links.
    pub fn from_architecture(arch: &Architecture) -> Self {
        let mut viz = Self::new();

        // Add named memory regions and processors from Architecture
        for region in &arch.memory {
            viz.add_memory_region(region, region.name());
        }
        for proc in &arch.processors {
            let name = proc.name().unwrap_or("unnamed");
            viz.add_processor(proc, name);
        }

        // Add link edges
        for link in &arch.links {
            let src_idx = viz.add_endpoint(&link.src);
            let dst_idx = viz.add_endpoint(&link.dst);

            let mapping = format_affine_map_detailed(&link.map);
            let bw = format_expr_short(&link.bandwidth);
            viz.graph.add_edge(
                src_idx,
                dst_idx,
                ArchEdge::Link {
                    name: link.name.clone(),
                    bandwidth: bw,
                    mapping,
                },
            );
        }

        viz
    }

    /// Export the graph to DOT format
    pub fn to_dot(&self) -> String {
        format!(
            "{:?}",
            Dot::with_config(&self.graph, &[Config::EdgeNoLabel])
        )
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

        for edge in self.graph.edge_indices() {
            let (src, tgt) = self.graph.edge_endpoints(edge).unwrap();
            let weight = &self.graph[edge];
            let (label, color, style) = match weight {
                ArchEdge::Link {
                    name,
                    bandwidth,
                    mapping,
                } => (
                    format!("{}\\n{}\\n{} B/cycle", name, mapping, bandwidth),
                    "blue",
                    "solid",
                ),
                ArchEdge::Contains => ("contains".to_string(), "gray", "dotted"),
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

    pub fn graph(&self) -> &ArchGraph {
        &self.graph
    }

    pub fn graph_mut(&mut self) -> &mut ArchGraph {
        &mut self.graph
    }
}

impl Default for ArchVisualizer {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Helper functions ────────────────────────────────────────────────────────

fn format_bank_details(bank: &MemoryBank) -> String {
    let cap = format!("capacity: {}", bank.capacity_bytes);
    if let Some(ref gran) = bank.access_granularity {
        format!("{}\\ngranularity: {}", cap, gran)
    } else {
        cap
    }
}

fn collect_all_dims(proc: &Processor) -> Vec<Dimension> {
    match proc {
        Processor::Primitive(_) => vec![],
        Processor::Replicated { dims, elem, .. } => {
            let mut result = dims.clone();
            result.extend(collect_all_dims(elem));
            result
        }
        Processor::Group { .. } => vec![],
    }
}

/// Format an affine map with dimension details for display
fn format_affine_map_detailed(map: &AffineMap) -> String {
    let src_info = map
        .src_dims
        .iter()
        .map(|d| d.name.0.clone())
        .collect::<Vec<_>>()
        .join(", ");

    let dst_info = map
        .dst_dims
        .iter()
        .map(|d| d.name.0.clone())
        .collect::<Vec<_>>()
        .join(", ");

    let results: Vec<_> = map.exprs.iter().map(format_affine_expr).collect();
    format!(
        "[{}] -> [{}]\\n({})",
        src_info,
        dst_info,
        results.join(", ")
    )
}

/// Format an affine expression for display
fn format_affine_expr(expr: &AffineExpr) -> String {
    match expr {
        AffineExpr::Var(dim) => dim.name.0.clone(),
        AffineExpr::Const(c) => c.to_string(),
        AffineExpr::Add(a, b) => {
            format!("({} + {})", format_affine_expr(a), format_affine_expr(b))
        }
        AffineExpr::MulConst(c, e) => {
            format!("({} * {})", format_affine_expr(e), c)
        }
        AffineExpr::Mod(a, b) => {
            format!("({} mod {})", format_affine_expr(a), format_affine_expr(b))
        }
        AffineExpr::CeilDiv(a, b) => {
            format!(
                "({} ceildiv {})",
                format_affine_expr(a),
                format_affine_expr(b)
            )
        }
    }
}

/// Format an Expr for short display (used for bandwidth labels)
fn format_expr_short(expr: &crate::core::Expr) -> String {
    use crate::core::Expr;
    match expr {
        Expr::Const(v) => v.to_string(),
        Expr::Sym(s) => s.0.clone(),
        Expr::Add(a, b) => format!("({} + {})", format_expr_short(a), format_expr_short(b)),
        Expr::Mul(a, b) => format!("({} * {})", format_expr_short(a), format_expr_short(b)),
        Expr::Div(a, b) => format!("({} / {})", format_expr_short(a), format_expr_short(b)),
        Expr::Min(a, b) => format!("min({}, {})", format_expr_short(a), format_expr_short(b)),
        Expr::Max(a, b) => format!("max({}, {})", format_expr_short(a), format_expr_short(b)),
    }
}

/// Convenience function to generate DOT from an architecture
pub fn architecture_to_dot(arch: &Architecture) -> String {
    let viz = ArchVisualizer::from_architecture(arch);
    viz.to_dot_styled(&arch.name)
}

/// Convenience function to generate DOT from links
pub fn memory_hierarchy_to_dot(name: &str, links: &[Link]) -> String {
    let viz = ArchVisualizer::from_links(links);
    viz.to_dot_styled(name)
}

/// Convenience function to generate expanded DOT from an architecture.
/// Shows all instances of memory regions and processors, with edges based on affine mapping.
pub fn architecture_to_dot_expanded(arch: &Architecture) -> String {
    to_dot_expanded(arch)
}

/// Generate an expanded DOT visualization showing all instances.
fn to_dot_expanded(arch: &Architecture) -> String {
    let mut dot = String::new();
    writeln!(dot, "digraph \"{}\" {{", arch.name).unwrap();
    writeln!(dot, "    rankdir=TB;").unwrap();
    writeln!(dot, "    node [fontname=\"Helvetica\"];").unwrap();
    writeln!(dot, "    edge [fontname=\"Helvetica\", fontsize=10];").unwrap();
    writeln!(dot).unwrap();

    let mut node_id = 0usize;
    let mut entity_nodes: HashMap<String, Vec<usize>> = HashMap::new();

    // Add memory region instances
    for region in &arch.memory {
        let name = region.name().unwrap_or("unnamed").to_string();
        let total = region_total_instances(region);

        if let Some(count) = total {
            if count > 1 {
                writeln!(dot, "    subgraph cluster_{} {{", name).unwrap();
                writeln!(dot, "        label=\"{}\";", name).unwrap();
                writeln!(dot, "        style=rounded;").unwrap();
                writeln!(dot, "        bgcolor=\"#E8F4FD\";").unwrap();

                let mut ids = Vec::new();
                for i in 0..count {
                    let bank_label = format_bank_instance_label(region, &name, i);
                    writeln!(dot, "        {} [label=\"{}\", shape=box, fillcolor=lightblue, style=filled];",
                        node_id, bank_label).unwrap();
                    ids.push(node_id);
                    node_id += 1;
                }
                writeln!(dot, "    }}").unwrap();
                entity_nodes.insert(name, ids);
            } else {
                writeln!(
                    dot,
                    "    {} [label=\"{}\", shape=box, fillcolor=lightblue, style=filled];",
                    node_id, name
                )
                .unwrap();
                entity_nodes.insert(name, vec![node_id]);
                node_id += 1;
            }
        } else {
            let label = format!("{}\\n[symbolic]", name);
            writeln!(
                dot,
                "    {} [label=\"{}\", shape=box, fillcolor=lightblue, style=filled];",
                node_id, label
            )
            .unwrap();
            entity_nodes.insert(name, vec![node_id]);
            node_id += 1;
        }
    }

    // Add processor instances
    for proc in &arch.processors {
        let name = proc.name().unwrap_or("unnamed").to_string();
        let total = proc.total_instances();

        if let Some(count) = total {
            if count > 1 {
                writeln!(dot, "    subgraph cluster_{} {{", name).unwrap();
                writeln!(dot, "        label=\"{}\";", name).unwrap();
                writeln!(dot, "        style=rounded;").unwrap();
                writeln!(dot, "        bgcolor=\"#E8FDE8\";").unwrap();

                let mut ids = Vec::new();
                for i in 0..count {
                    let label = format!("{}[{}]", name, i);
                    writeln!(dot, "        {} [label=\"{}\", shape=ellipse, fillcolor=lightgreen, style=filled];",
                        node_id, label).unwrap();
                    ids.push(node_id);
                    node_id += 1;
                }
                writeln!(dot, "    }}").unwrap();
                entity_nodes.insert(name, ids);
            } else {
                writeln!(
                    dot,
                    "    {} [label=\"{}\", shape=ellipse, fillcolor=lightgreen, style=filled];",
                    node_id, name
                )
                .unwrap();
                entity_nodes.insert(name, vec![node_id]);
                node_id += 1;
            }
        } else {
            let label = format!("{}\\n[symbolic]", name);
            writeln!(
                dot,
                "    {} [label=\"{}\", shape=ellipse, fillcolor=lightgreen, style=filled];",
                node_id, label
            )
            .unwrap();
            entity_nodes.insert(name, vec![node_id]);
            node_id += 1;
        }
    }

    writeln!(dot).unwrap();

    // Draw edges based on affine maps
    for link in &arch.links {
        let src_name = link.src.name();
        let dst_name = link.dst.name();

        let src_nodes = match entity_nodes.get(src_name) {
            Some(nodes) => nodes,
            None => continue,
        };
        let dst_nodes = match entity_nodes.get(dst_name) {
            Some(nodes) => nodes,
            None => continue,
        };

        let src_count = src_nodes.len();
        let dst_count = dst_nodes.len();

        if src_count == 0 || dst_count == 0 {
            continue;
        }

        let edge_color = if link.dst.is_proc() {
            "purple"
        } else {
            "blue"
        };

        for (src_idx, &src_node) in src_nodes.iter().enumerate() {
            let target_base = if !link.map.exprs.is_empty() {
                link.map.apply(&[src_idx as i64])[0] as usize
            } else {
                0
            };

            let fan_out = if src_count > 0 {
                dst_count / src_count
            } else {
                1
            };

            for offset in 0..fan_out {
                let tgt_idx = target_base + offset;
                if tgt_idx < dst_nodes.len() {
                    let tgt_node = dst_nodes[tgt_idx];
                    writeln!(
                        dot,
                        "    {} -> {} [color={}];",
                        src_node, tgt_node, edge_color
                    )
                    .unwrap();
                }
            }
        }
    }

    writeln!(dot, "}}").unwrap();
    dot
}

/// Compute total number of instances for a memory region
fn region_total_instances(region: &MemoryRegion) -> Option<u64> {
    match region {
        MemoryRegion::Bank(_) => Some(1),
        MemoryRegion::Replicated { dims, elem, .. } => {
            let outer: u64 = dims
                .iter()
                .map(|d| d.size.as_const())
                .collect::<Option<Vec<_>>>()?
                .into_iter()
                .product();
            let inner = region_total_instances(elem)?;
            Some(outer * inner)
        }
        MemoryRegion::Group { parts, .. } => {
            let mut total = 0u64;
            for p in parts {
                total += region_total_instances(p)?;
            }
            Some(total)
        }
    }
}

/// Format a bank instance label for the expanded view
fn format_bank_instance_label(region: &MemoryRegion, name: &str, index: u64) -> String {
    // Walk down to find bank details
    fn find_bank(region: &MemoryRegion) -> Option<&MemoryBank> {
        match region {
            MemoryRegion::Bank(b) => Some(b),
            MemoryRegion::Replicated { elem, .. } => find_bank(elem),
            MemoryRegion::Group { .. } => None,
        }
    }

    if let Some(bank) = find_bank(region) {
        if let Some(ref gran) = bank.access_granularity {
            format!("{}[{}]\\ncap:{} gran:{}", name, index, bank.capacity_bytes, gran)
        } else {
            format!("{}[{}]\\ncap:{}", name, index, bank.capacity_bytes)
        }
    } else {
        format!("{}[{}]", name, index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;

    #[test]
    fn test_simple_memory_visualization() {
        let bank = MemoryBank::from_blocks(SizeExpr::Const(1024), SizeExpr::Const(16));
        let region = MemoryRegion::bank(bank);

        let mut viz = ArchVisualizer::new();
        viz.add_memory_region(&region, Some("test_mem"));

        let dot = viz.to_dot_styled("test");
        assert!(dot.contains("test_mem"));
    }
}
