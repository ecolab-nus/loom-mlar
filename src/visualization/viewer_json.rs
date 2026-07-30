use std::collections::BTreeMap;

use crate::arch::Architecture;
use serde::Serialize;
use serde_json::Value;

use super::graph_json::{ArchitectureGraphJson, architecture_to_graph_json};
use super::hierarchy_json::{HierarchyNode, architecture_to_hierarchy_json};

const VIEWER_SCHEMA_VERSION: &str = "mlar.arch-viewer.v2";

#[derive(Debug, Clone, Serialize)]
pub struct ArchitectureViewerJson {
    pub schema_version: &'static str,
    pub hierarchy: HierarchyNode,
    pub graphs: BTreeMap<String, ArchitectureGraphJson>,
}

pub fn architecture_to_viewer_json(architecture: &Architecture) -> ArchitectureViewerJson {
    let hierarchy = architecture_to_hierarchy_json(architecture);
    let graphs = BTreeMap::from([("".into(), architecture_to_graph_json(architecture))]);
    ArchitectureViewerJson {
        schema_version: VIEWER_SCHEMA_VERSION,
        hierarchy: hierarchy.root,
        graphs,
    }
}

pub fn architecture_to_viewer_json_value(architecture: &Architecture) -> Value {
    serde_json::to_value(architecture_to_viewer_json(architecture))
        .expect("architecture viewer serialization must succeed")
}

pub fn architecture_to_viewer_json_string_pretty(
    architecture: &Architecture,
) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(&architecture_to_viewer_json(architecture))
}
