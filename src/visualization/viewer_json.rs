use super::graph_json::{ArchitectureGraphJson, architecture_to_graph_json};
use super::hierarchy_json::{HierarchyNode, architecture_to_hierarchy_json};
use crate::arch::Architecture;
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;

const VIEWER_SCHEMA_VERSION: &str = "mlar.arch-viewer.v1";

/// Combined payload for the web viewer: hierarchy tree + per-node graph views.
#[derive(Debug, Clone, Serialize)]
pub struct ArchitectureViewerJson {
    pub schema_version: &'static str,
    pub hierarchy: HierarchyNode,
    pub graphs: BTreeMap<String, ArchitectureGraphJson>,
}

/// Build the combined viewer payload for an architecture.
///
/// The `graphs` map is keyed by path strings (e.g. `""` for the root,
/// `"core"` for a child named "core", `"core/lane"` for a nested child).
/// Each value is a self-contained graph JSON suitable for React Flow rendering.
pub fn architecture_to_viewer_json(arch: &Architecture) -> ArchitectureViewerJson {
    let hierarchy = architecture_to_hierarchy_json(arch);
    let mut graphs = BTreeMap::new();
    collect_sub_graphs(arch, "", &mut graphs);

    ArchitectureViewerJson {
        schema_version: VIEWER_SCHEMA_VERSION,
        hierarchy: hierarchy.root,
        graphs,
    }
}

pub fn architecture_to_viewer_json_value(arch: &Architecture) -> Value {
    serde_json::to_value(architecture_to_viewer_json(arch))
        .expect("viewer serialization must succeed")
}

pub fn architecture_to_viewer_json_string_pretty(
    arch: &Architecture,
) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(&architecture_to_viewer_json(arch))
}

fn collect_sub_graphs(
    arch: &Architecture,
    path: &str,
    graphs: &mut BTreeMap<String, ArchitectureGraphJson>,
) {
    graphs.insert(path.to_string(), architecture_to_graph_json(arch));

    for child in &arch.children {
        let child_path = sub_path(path, &child.name);
        collect_sub_graphs(child, &child_path, graphs);
    }
}

fn sub_path(parent: &str, child: &str) -> String {
    if parent.is_empty() {
        child.to_string()
    } else {
        format!("{parent}/{child}")
    }
}
