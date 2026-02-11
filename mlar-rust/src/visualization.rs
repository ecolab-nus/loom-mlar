//! Visualization support for MLAR architectures using petgraph.
//!
//! Converts MLAR architecture specifications into graphs exportable to GraphViz DOT format.

use crate::architecture::{Architecture, ArchitectureLabel};
use crate::core::{
    AffineExpr, AffineMap, Dimension, Endpoint, Link, MemoryBank, MemoryRegion, Processor,
};
use petgraph::dot::{Config, Dot};
use petgraph::graph::{DiGraph, NodeIndex};
use std::collections::{HashMap, HashSet};
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
                    MemoryRegion::Replicated {
                        dims: inner_dims, ..
                    } => Self::format_dimensions_compact(inner_dims),
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
        AffineExpr::Sym(sym) => sym.0.clone(),
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
    to_dot_expanded(arch)
}

/// Convenience function to generate DOT from links
pub fn memory_hierarchy_to_dot(name: &str, links: &[Link]) -> String {
    let viz = ArchVisualizer::from_links(links);
    viz.to_dot_styled(name)
}

/// Convenience function to generate expanded DOT from an architecture.
/// Shows all instances of memory regions and processors, with edges based on affine mapping.
pub fn architecture_to_dot_expanded(arch: &Architecture) -> String {
    architecture_to_dot(arch)
}

const MAX_RENDERED_INSTANCES_PER_ENTITY: usize = 4096;

#[derive(Debug, Clone)]
struct InstanceNode {
    id: usize,
    coord: Vec<i64>,
}

#[derive(Debug, Clone)]
struct EntityLayout {
    nodes: Vec<InstanceNode>,
    coord_to_node: HashMap<Vec<i64>, usize>,
    dim_names: Vec<String>,
    dim_positions: HashMap<String, usize>,
    dim_sizes: Option<Vec<u64>>,
}

#[derive(Debug, Clone)]
struct ArchitectureGrouping {
    label: String,
    dim_names: Vec<String>,
    dim_sizes: Vec<u64>,
}

/// Generate an expanded DOT visualization showing all instances.
fn to_dot_expanded(arch: &Architecture) -> String {
    let mut dot = String::new();
    writeln!(dot, "digraph \"{}\" {{", arch.name).unwrap();
    writeln!(dot, "    rankdir=TB;").unwrap();
    writeln!(
        dot,
        "    graph [nodesep=0.35, ranksep=0.9, splines=true, pad=0.2];"
    )
    .unwrap();
    writeln!(dot, "    node [fontname=\"Helvetica\"];").unwrap();
    writeln!(
        dot,
        "    edge [fontname=\"Helvetica\", fontsize=10, penwidth=1.2];"
    )
    .unwrap();
    writeln!(dot).unwrap();

    let entity_link_dims = collect_entity_link_dims(arch);
    let label_dim_names = arch.labels.last().map(|label| {
        label
            .dims
            .iter()
            .map(|d| d.name.0.clone())
            .collect::<Vec<_>>()
    });
    let mut node_id = 0usize;
    let mut entity_nodes: HashMap<String, EntityLayout> = HashMap::new();
    let mut memory_entity_names: HashSet<String> = HashSet::new();

    let mut memory_entities = Vec::new();
    for region in &arch.memory {
        let name = region.name().unwrap_or("unnamed");
        memory_entity_names.insert(name.to_string());
        let preferred_dims = merge_preferred_dims(
            entity_link_dims.get(name).map(|v| v.as_slice()),
            label_dim_names.as_deref(),
        );
        let dims = select_layout_dims(collect_memory_dims(region), preferred_dims.as_deref());
        memory_entities.push((region, name, dims));
    }

    let mut processor_entities = Vec::new();
    for proc in &arch.processors {
        let name = proc.name().unwrap_or("unnamed");
        let preferred_dims = merge_preferred_dims(
            entity_link_dims.get(name).map(|v| v.as_slice()),
            label_dim_names.as_deref(),
        );
        let dims = select_layout_dims(collect_processor_dims(proc), preferred_dims.as_deref());
        processor_entities.push((proc, name, dims));
    }

    let architecture_grouping =
        detect_architecture_grouping(arch, &memory_entities, &processor_entities);
    let use_entity_clusters = architecture_grouping.is_none();

    for (region, name, dims) in memory_entities {
        let symbolic = format_symbolic_label(name, dims.as_slice());
        let layout = add_entity_layout(
            &mut dot,
            &mut node_id,
            name,
            use_entity_clusters,
            "mem",
            dims,
            "box",
            "lightblue",
            "#E8F4FD",
            symbolic,
            |coord| format_memory_instance_label(region, name, coord),
        );
        entity_nodes.insert(name.to_string(), layout);
    }

    for (_proc, name, dims) in processor_entities {
        let symbolic = format_symbolic_label(name, dims.as_slice());
        let layout = add_entity_layout(
            &mut dot,
            &mut node_id,
            name,
            use_entity_clusters,
            "proc",
            dims,
            "ellipse",
            "lightgreen",
            "#E8FDE8",
            symbolic,
            |coord| format_processor_instance_label(name, coord),
        );
        entity_nodes.insert(name.to_string(), layout);
    }

    if let Some(grouping) = architecture_grouping.as_ref() {
        emit_architecture_clusters(&mut dot, grouping, &entity_nodes, &memory_entity_names);
    }

    writeln!(dot).unwrap();

    for link in &arch.links {
        let src_name = link.src.name();
        let dst_name = link.dst.name();
        let src_layout = match entity_nodes.get(src_name) {
            Some(layout) => layout,
            None => continue,
        };
        let dst_layout = match entity_nodes.get(dst_name) {
            Some(layout) => layout,
            None => continue,
        };

        if src_layout.nodes.is_empty() || dst_layout.nodes.is_empty() {
            continue;
        }

        let edge_attrs = format_link_edge_attrs(link, src_name, dst_name);

        // If either side is symbolic/aggregated, draw a coarse edge.
        if src_layout.dim_sizes.is_none() || dst_layout.dim_sizes.is_none() {
            let src_node = src_layout.nodes[0].id;
            let dst_node = dst_layout.nodes[0].id;
            writeln!(dot, "    {} -> {} [{}];", src_node, dst_node, edge_attrs).unwrap();
            continue;
        }

        let src_count = src_layout.nodes.len();
        let dst_count = dst_layout.nodes.len();
        let fan_out = if src_count > 0 && dst_count > src_count && dst_count % src_count == 0 {
            dst_count / src_count
        } else {
            1
        };

        for src_node in &src_layout.nodes {
            let src_inputs: Vec<i64> = link
                .map
                .src_dims
                .iter()
                .map(|d| {
                    src_layout
                        .dim_positions
                        .get(&d.name.0)
                        .and_then(|idx| src_node.coord.get(*idx))
                        .copied()
                        .unwrap_or(0)
                })
                .collect();

            let mapped = if link.map.exprs.is_empty() {
                Vec::new()
            } else {
                link.map.apply(&src_inputs)
            };

            let mut dst_coord = vec![0i64; dst_layout.dim_names.len()];
            for (idx, dst_dim_name) in dst_layout.dim_names.iter().enumerate() {
                if let Some(mapped_idx) = link
                    .map
                    .dst_dims
                    .iter()
                    .position(|d| d.name.0 == *dst_dim_name)
                {
                    dst_coord[idx] = *mapped.get(mapped_idx).unwrap_or(&0);
                } else if let Some(src_idx) = src_layout.dim_positions.get(dst_dim_name) {
                    dst_coord[idx] = src_node.coord.get(*src_idx).copied().unwrap_or(0);
                }
            }

            if fan_out == 1 {
                if let Some(&tgt_node) = dst_layout.coord_to_node.get(&dst_coord) {
                    writeln!(dot, "    {} -> {} [{}];", src_node.id, tgt_node, edge_attrs).unwrap();
                }
                continue;
            }

            let Some(dst_sizes) = dst_layout.dim_sizes.as_ref() else {
                continue;
            };
            let Some(base_flat) = coord_to_flat_index(&dst_coord, dst_sizes) else {
                continue;
            };

            for offset in 0..fan_out {
                let idx = base_flat + offset;
                if let Some(tgt_node) = dst_layout.nodes.get(idx) {
                    writeln!(
                        dot,
                        "    {} -> {} [{}];",
                        src_node.id, tgt_node.id, edge_attrs
                    )
                    .unwrap();
                }
            }
        }
    }

    writeln!(dot, "}}").unwrap();
    dot
}

fn add_entity_layout<F>(
    dot: &mut String,
    next_node_id: &mut usize,
    name: &str,
    use_entity_cluster: bool,
    cluster_prefix: &str,
    dims: Vec<Dimension>,
    shape: &str,
    fill_color: &str,
    cluster_color: &str,
    symbolic_label: String,
    label_for_coord: F,
) -> EntityLayout
where
    F: Fn(&[i64]) -> String,
{
    let dim_names: Vec<String> = dims.iter().map(|d| d.name.0.clone()).collect();
    let dim_positions: HashMap<String, usize> = dim_names
        .iter()
        .enumerate()
        .map(|(idx, name)| (name.clone(), idx))
        .collect();

    if dims.is_empty() {
        let id = *next_node_id;
        *next_node_id += 1;
        writeln!(
            dot,
            "    {} [label=\"{}\", shape={}, fillcolor={}, style=filled];",
            id,
            label_for_coord(&[]),
            shape,
            fill_color
        )
        .unwrap();

        let node = InstanceNode {
            id,
            coord: Vec::new(),
        };
        let mut coord_to_node = HashMap::new();
        coord_to_node.insert(Vec::new(), id);
        return EntityLayout {
            nodes: vec![node],
            coord_to_node,
            dim_names,
            dim_positions,
            dim_sizes: Some(Vec::new()),
        };
    }

    let sizes: Option<Vec<u64>> = dims
        .iter()
        .map(|d| d.size.as_const())
        .collect::<Option<Vec<_>>>();
    let render_count = sizes
        .as_ref()
        .and_then(|s| instance_count(s))
        .and_then(|n| usize::try_from(n).ok());

    if sizes.is_none()
        || render_count.is_none()
        || render_count.unwrap() > MAX_RENDERED_INSTANCES_PER_ENTITY
    {
        let id = *next_node_id;
        *next_node_id += 1;
        writeln!(
            dot,
            "    {} [label=\"{}\", shape={}, fillcolor={}, style=filled];",
            id, symbolic_label, shape, fill_color
        )
        .unwrap();

        let node = InstanceNode {
            id,
            coord: Vec::new(),
        };
        return EntityLayout {
            nodes: vec![node],
            coord_to_node: HashMap::new(),
            dim_names,
            dim_positions,
            dim_sizes: None,
        };
    }

    let sizes = sizes.unwrap();
    let coords = enumerate_coords(sizes.as_slice());
    let needs_cluster = use_entity_cluster && coords.len() > 1;
    let cluster_id = format!("cluster_{}_{}", cluster_prefix, sanitize_identifier(name));

    if needs_cluster {
        let dims_str = dim_names.join(", ");
        writeln!(dot, "    subgraph {} {{", cluster_id).unwrap();
        writeln!(dot, "        label=\"{} [{}]\";", name, dims_str).unwrap();
        writeln!(dot, "        style=rounded;").unwrap();
        writeln!(dot, "        bgcolor=\"{}\";", cluster_color).unwrap();
    }

    let indent = if needs_cluster { "        " } else { "    " };
    let mut nodes = Vec::with_capacity(coords.len());
    let mut coord_to_node = HashMap::new();
    let mut ordered_node_ids = Vec::with_capacity(coords.len());

    for coord in &coords {
        let id = *next_node_id;
        *next_node_id += 1;
        ordered_node_ids.push(id);
        coord_to_node.insert(coord.clone(), id);
        nodes.push(InstanceNode {
            id,
            coord: coord.clone(),
        });

        writeln!(
            dot,
            "{}{} [label=\"{}\", shape={}, fillcolor={}, style=filled];",
            indent,
            id,
            label_for_coord(coord),
            shape,
            fill_color
        )
        .unwrap();
    }

    if needs_cluster && sizes.len() == 2 {
        let rows = sizes[0] as usize;
        let cols = sizes[1] as usize;
        for row in 0..rows {
            let start = row * cols;
            let row_ids = &ordered_node_ids[start..start + cols];
            let row_decl = row_ids
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join("; ");
            writeln!(dot, "{}{{ rank=same; {}; }}", indent, row_decl).unwrap();

            for pair in row_ids.windows(2) {
                writeln!(
                    dot,
                    "{}{} -> {} [style=invis, weight=100];",
                    indent, pair[0], pair[1]
                )
                .unwrap();
            }

            if row + 1 < rows {
                let below = ordered_node_ids[(row + 1) * cols];
                writeln!(
                    dot,
                    "{}{} -> {} [style=invis, weight=50];",
                    indent, row_ids[0], below
                )
                .unwrap();
            }
        }
    }

    if needs_cluster {
        writeln!(dot, "    }}").unwrap();
    }

    EntityLayout {
        nodes,
        coord_to_node,
        dim_names,
        dim_positions,
        dim_sizes: Some(sizes),
    }
}

fn detect_architecture_grouping(
    arch: &Architecture,
    memory_entities: &[(&MemoryRegion, &str, Vec<Dimension>)],
    processor_entities: &[(&Processor, &str, Vec<Dimension>)],
) -> Option<ArchitectureGrouping> {
    let mut all_dims: Vec<&[Dimension]> = memory_entities
        .iter()
        .map(|(_, _, dims)| dims.as_slice())
        .collect();
    all_dims.extend(
        processor_entities
            .iter()
            .map(|(_, _, dims)| dims.as_slice()),
    );

    if all_dims.len() < 2 {
        return None;
    }

    if let Some(grouping) = grouping_from_arch_label(arch.labels.last(), &all_dims) {
        return Some(grouping);
    }

    let first = *all_dims.first()?;
    if first.is_empty() {
        return None;
    }

    let mut common_names: Vec<String> = first.iter().map(|d| d.name.0.clone()).collect();
    for dims in all_dims.iter().skip(1) {
        if dims.is_empty() {
            return None;
        }

        let names: HashSet<&str> = dims.iter().map(|d| d.name.0.as_str()).collect();
        common_names.retain(|n| names.contains(n.as_str()));
        if common_names.is_empty() {
            return None;
        }
    }

    let mut dim_names = Vec::new();
    let mut dim_sizes = Vec::new();
    for first_dim in first {
        if !common_names.iter().any(|n| n == &first_dim.name.0) {
            continue;
        }

        let expected = first_dim.size.as_const()?;
        for dims in all_dims.iter().skip(1) {
            let matched = dims.iter().find(|d| d.name.0 == first_dim.name.0)?;
            if matched.size.as_const()? != expected {
                return None;
            }
        }

        dim_names.push(first_dim.name.0.clone());
        dim_sizes.push(expected);
    }

    if dim_names.is_empty() {
        return None;
    }

    Some(ArchitectureGrouping {
        label: infer_architecture_group_name(arch),
        dim_names,
        dim_sizes,
    })
}

fn grouping_from_arch_label(
    label: Option<&ArchitectureLabel>,
    all_dims: &[&[Dimension]],
) -> Option<ArchitectureGrouping> {
    let label = label?;
    if label.dims.is_empty() {
        return None;
    }

    let mut dim_names = Vec::new();
    let mut dim_sizes = Vec::new();
    for dim in &label.dims {
        let size = dim.size.as_const()?;
        let appears_anywhere = all_dims
            .iter()
            .any(|dims| dims.iter().any(|d| d.name.0 == dim.name.0));
        if !appears_anywhere {
            return None;
        }
        dim_names.push(dim.name.0.clone());
        dim_sizes.push(size);
    }

    Some(ArchitectureGrouping {
        label: label.name.clone(),
        dim_names,
        dim_sizes,
    })
}

fn infer_architecture_group_name(arch: &Architecture) -> String {
    let lower = arch.name.to_ascii_lowercase();
    if lower.contains("mesh") || lower.contains("torus") {
        "core".to_string()
    } else {
        arch.name.clone()
    }
}

fn emit_architecture_clusters(
    dot: &mut String,
    grouping: &ArchitectureGrouping,
    entity_nodes: &HashMap<String, EntityLayout>,
    memory_entity_names: &HashSet<String>,
) {
    let coords = enumerate_coords(grouping.dim_sizes.as_slice());
    let mut memory_anchors: Vec<Option<usize>> = Vec::with_capacity(coords.len());
    let mut compute_anchors: Vec<Option<usize>> = Vec::with_capacity(coords.len());

    for coord in &coords {
        let mut memory_member_ids = Vec::new();
        let mut processor_member_ids = Vec::new();
        for (entity_name, layout) in entity_nodes {
            if layout.dim_sizes.is_none() {
                continue;
            }

            for node in &layout.nodes {
                if node_matches_arch_coord(node, layout, grouping, coord) {
                    if memory_entity_names.contains(entity_name) {
                        memory_member_ids.push(node.id);
                    } else {
                        processor_member_ids.push(node.id);
                    }
                }
            }
        }

        memory_member_ids.sort_unstable();
        memory_member_ids.dedup();
        processor_member_ids.sort_unstable();
        processor_member_ids.dedup();

        let mut member_ids = memory_member_ids.clone();
        member_ids.extend(processor_member_ids.iter().copied());
        member_ids.sort_unstable();
        member_ids.dedup();

        if member_ids.is_empty() {
            memory_anchors.push(None);
            compute_anchors.push(None);
            continue;
        }

        let coord_id = coord
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join("_");
        let cluster_id = format!("cluster_arch_{}", coord_id);
        let label = format!("{}[{}]", grouping.label, format_coord_suffix(coord));

        writeln!(dot, "    subgraph {} {{", cluster_id).unwrap();
        writeln!(dot, "        label=\"{}\";", label).unwrap();
        writeln!(dot, "        style=rounded;").unwrap();
        writeln!(dot, "        color=\"#C8CDD6\";").unwrap();
        writeln!(dot, "        bgcolor=\"#F5F7FA\";").unwrap();
        for member_id in &member_ids {
            writeln!(dot, "        {};", member_id).unwrap();
        }

        if let (Some(&mem_anchor), Some(&proc_anchor)) =
            (memory_member_ids.first(), processor_member_ids.first())
        {
            writeln!(
                dot,
                "        {} -> {} [style=invis, weight=260, minlen=2];",
                mem_anchor, proc_anchor
            )
            .unwrap();
        }

        if processor_member_ids.len() > 1 {
            let processor_rank = join_node_ids(processor_member_ids.as_slice());
            writeln!(dot, "        {{ rank=same; {}; }}", processor_rank).unwrap();
            for pair in processor_member_ids.windows(2) {
                writeln!(
                    dot,
                    "        {} -> {} [style=invis, weight=220];",
                    pair[0], pair[1]
                )
                .unwrap();
            }
        }

        writeln!(dot, "    }}").unwrap();

        memory_anchors.push(
            memory_member_ids
                .first()
                .copied()
                .or_else(|| member_ids.first().copied()),
        );
        compute_anchors.push(
            processor_member_ids
                .first()
                .copied()
                .or_else(|| member_ids.first().copied()),
        );
    }

    if grouping.dim_sizes.len() != 2 {
        return;
    }

    let rows = grouping.dim_sizes[0] as usize;
    let cols = grouping.dim_sizes[1] as usize;
    for row in 0..rows {
        let start = row * cols;
        let end = start + cols;
        let memory_row_anchors: Vec<usize> = memory_anchors[start..end]
            .iter()
            .filter_map(|id| *id)
            .collect();
        let compute_row_anchors: Vec<usize> = compute_anchors[start..end]
            .iter()
            .filter_map(|id| *id)
            .collect();

        if memory_row_anchors.len() > 1 {
            let row_decl = memory_row_anchors
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join("; ");
            writeln!(dot, "    {{ rank=same; {}; }}", row_decl).unwrap();

            for pair in memory_row_anchors.windows(2) {
                writeln!(
                    dot,
                    "    {} -> {} [style=invis, weight=120];",
                    pair[0], pair[1]
                )
                .unwrap();
            }
        }

        if compute_row_anchors.len() > 1 {
            let row_decl = compute_row_anchors
                .iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join("; ");
            writeln!(dot, "    {{ rank=same; {}; }}", row_decl).unwrap();

            for pair in compute_row_anchors.windows(2) {
                writeln!(
                    dot,
                    "    {} -> {} [style=invis, weight=140];",
                    pair[0], pair[1]
                )
                .unwrap();
            }
        }

        let memory_row_head = memory_anchors[start..end].iter().find_map(|id| *id);
        let compute_row_head = compute_anchors[start..end].iter().find_map(|id| *id);
        if let (Some(mem_head), Some(comp_head)) = (memory_row_head, compute_row_head) {
            writeln!(
                dot,
                "    {} -> {} [style=invis, weight=260, minlen=2];",
                mem_head, comp_head
            )
            .unwrap();
        }

        if row + 1 < rows {
            let next_start = (row + 1) * cols;
            let next_end = next_start + cols;
            let current_memory_anchor = memory_anchors[start..end].iter().find_map(|id| *id);
            let next_memory_anchor = memory_anchors[next_start..next_end]
                .iter()
                .find_map(|id| *id);
            let current_compute_anchor = compute_anchors[start..end].iter().find_map(|id| *id);
            let next_compute_anchor = compute_anchors[next_start..next_end]
                .iter()
                .find_map(|id| *id);

            if let (Some(a), Some(b)) = (current_memory_anchor, next_memory_anchor) {
                writeln!(dot, "    {} -> {} [style=invis, weight=80];", a, b).unwrap();
            }
            if let (Some(a), Some(b)) = (current_compute_anchor, next_compute_anchor) {
                writeln!(dot, "    {} -> {} [style=invis, weight=90];", a, b).unwrap();
            }
        }
    }

    // Lock each column across rows so the 2D grid remains visually aligned.
    for col in 0..cols {
        let mut memory_col = Vec::new();
        let mut compute_col = Vec::new();
        for row in 0..rows {
            let idx = row * cols + col;
            if let Some(id) = memory_anchors[idx] {
                memory_col.push(id);
            }
            if let Some(id) = compute_anchors[idx] {
                compute_col.push(id);
            }
        }

        for pair in memory_col.windows(2) {
            writeln!(
                dot,
                "    {} -> {} [style=invis, weight=180, minlen=3];",
                pair[0], pair[1]
            )
            .unwrap();
        }
        for pair in compute_col.windows(2) {
            writeln!(
                dot,
                "    {} -> {} [style=invis, weight=170, minlen=3];",
                pair[0], pair[1]
            )
            .unwrap();
        }
    }
}

fn node_matches_arch_coord(
    node: &InstanceNode,
    layout: &EntityLayout,
    grouping: &ArchitectureGrouping,
    coord: &[i64],
) -> bool {
    for (group_idx, group_dim_name) in grouping.dim_names.iter().enumerate() {
        let Some(&layout_idx) = layout.dim_positions.get(group_dim_name) else {
            return false;
        };
        let expected = coord.get(group_idx).copied().unwrap_or(0);
        if node.coord.get(layout_idx).copied().unwrap_or(0) != expected {
            return false;
        }
    }
    true
}

fn format_link_edge_attrs(link: &Link, src_name: &str, dst_name: &str) -> String {
    // Keep core-local compute edges structurally strong while preventing torus links
    // from distorting the node grid.
    if link.dst.is_proc() {
        return "color=purple, constraint=true, weight=70, arrowsize=0.7".to_string();
    }

    if link.src.is_mem() && link.dst.is_mem() && src_name == dst_name {
        return "color=blue, constraint=false, weight=3, arrowsize=0.6".to_string();
    }

    "color=blue, constraint=true, weight=25, arrowsize=0.7".to_string()
}

fn join_node_ids(node_ids: &[usize]) -> String {
    node_ids
        .iter()
        .map(|id| id.to_string())
        .collect::<Vec<_>>()
        .join("; ")
}

fn format_symbolic_label(name: &str, dims: &[Dimension]) -> String {
    if dims.is_empty() {
        return format!("{}\\n[symbolic]", name);
    }
    let dim_names = dims
        .iter()
        .map(|d| d.name.0.clone())
        .collect::<Vec<_>>()
        .join(", ");
    format!("{}\\n[symbolic {}]", name, dim_names)
}

fn format_coord_suffix(coord: &[i64]) -> String {
    coord
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn format_memory_instance_label(region: &MemoryRegion, name: &str, coord: &[i64]) -> String {
    let idx_label = if coord.is_empty() {
        name.to_string()
    } else {
        format!("{}[{}]", name, format_coord_suffix(coord))
    };

    if let Some(bank) = find_leaf_bank(region) {
        if let Some(ref gran) = bank.access_granularity {
            format!("{}\\ncap:{} gran:{}", idx_label, bank.capacity_bytes, gran)
        } else {
            format!("{}\\ncap:{}", idx_label, bank.capacity_bytes)
        }
    } else {
        idx_label
    }
}

fn format_processor_instance_label(name: &str, coord: &[i64]) -> String {
    if coord.is_empty() {
        name.to_string()
    } else {
        format!("{}[{}]", name, format_coord_suffix(coord))
    }
}

fn find_leaf_bank(region: &MemoryRegion) -> Option<&MemoryBank> {
    match region {
        MemoryRegion::Bank(b) => Some(b),
        MemoryRegion::Replicated { elem, .. } => find_leaf_bank(elem),
        MemoryRegion::Group { .. } => None,
    }
}

fn collect_entity_link_dims(arch: &Architecture) -> HashMap<String, Vec<String>> {
    let mut result: HashMap<String, Vec<String>> = HashMap::new();
    for link in &arch.links {
        let src_dims = result.entry(link.src.name().to_string()).or_default();
        for d in &link.map.src_dims {
            push_unique(src_dims, d.name.0.clone());
        }

        let dst_dims = result.entry(link.dst.name().to_string()).or_default();
        for d in &link.map.dst_dims {
            push_unique(dst_dims, d.name.0.clone());
        }
    }
    result
}

fn push_unique(items: &mut Vec<String>, value: String) {
    if !items.contains(&value) {
        items.push(value);
    }
}

fn merge_preferred_dims(
    primary: Option<&[String]>,
    secondary: Option<&[String]>,
) -> Option<Vec<String>> {
    if primary.is_none() && secondary.is_none() {
        return None;
    }

    let mut merged = Vec::new();
    if let Some(primary) = primary {
        for item in primary {
            push_unique(&mut merged, item.clone());
        }
    }
    if let Some(secondary) = secondary {
        for item in secondary {
            push_unique(&mut merged, item.clone());
        }
    }
    Some(merged)
}

fn collect_memory_dims(region: &MemoryRegion) -> Vec<Dimension> {
    fn walk(region: &MemoryRegion, out: &mut Vec<Dimension>) {
        match region {
            MemoryRegion::Bank(_) => {}
            MemoryRegion::Replicated { dims, elem, .. } => {
                out.extend(dims.iter().cloned());
                walk(elem, out);
            }
            MemoryRegion::Group { parts, .. } => {
                for part in parts {
                    walk(part, out);
                }
            }
        }
    }

    let mut out = Vec::new();
    walk(region, &mut out);
    out
}

fn collect_processor_dims(proc: &Processor) -> Vec<Dimension> {
    proc.all_dims().into_iter().cloned().collect()
}

fn select_layout_dims(all_dims: Vec<Dimension>, used_dims: Option<&[String]>) -> Vec<Dimension> {
    if all_dims.is_empty() {
        return all_dims;
    }

    let deduped = dedup_dims(all_dims);
    let Some(used_dims) = used_dims else {
        return deduped;
    };

    let used_set: HashSet<&str> = used_dims.iter().map(|s| s.as_str()).collect();
    let selected: Vec<Dimension> = deduped
        .iter()
        .filter(|d| used_set.contains(d.name.0.as_str()))
        .cloned()
        .collect();

    if selected.is_empty() {
        deduped
    } else {
        selected
    }
}

fn dedup_dims(dims: Vec<Dimension>) -> Vec<Dimension> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for dim in dims {
        if seen.insert(dim.name.0.clone()) {
            out.push(dim);
        }
    }
    out
}

fn instance_count(sizes: &[u64]) -> Option<u128> {
    sizes
        .iter()
        .try_fold(1u128, |acc, &sz| acc.checked_mul(sz as u128))
}

fn enumerate_coords(sizes: &[u64]) -> Vec<Vec<i64>> {
    let Some(total) = instance_count(sizes) else {
        return Vec::new();
    };
    let mut coords = Vec::with_capacity(total as usize);

    for flat in 0..(total as usize) {
        let mut idx = flat;
        let mut coord = vec![0i64; sizes.len()];
        for d in (0..sizes.len()).rev() {
            let size = sizes[d] as usize;
            coord[d] = (idx % size) as i64;
            idx /= size;
        }
        coords.push(coord);
    }

    coords
}

fn coord_to_flat_index(coord: &[i64], sizes: &[u64]) -> Option<usize> {
    if coord.len() != sizes.len() {
        return None;
    }

    let mut flat = 0usize;
    let mut stride = 1usize;

    for i in (0..coord.len()).rev() {
        let size = sizes[i] as usize;
        let value = coord[i];
        if value < 0 {
            return None;
        }
        let value = value as usize;
        if value >= size {
            return None;
        }
        flat = flat.checked_add(value.checked_mul(stride)?)?;
        stride = stride.checked_mul(size)?;
    }

    Some(flat)
}

fn sanitize_identifier(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
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
