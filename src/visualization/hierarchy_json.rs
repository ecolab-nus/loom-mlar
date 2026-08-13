use crate::arch::Architecture;
use serde::Serialize;
use serde_json::{Value, json};

use super::graph_json::GraphDimension;

const HIERARCHY_SCHEMA_VERSION: &str = "mlar.arch-hierarchy.v2";

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HierarchyNodeKind {
    Architecture,
    MemoryArray,
    MemoryAlias,
    ProcessorArray,
    Resource,
}

#[derive(Debug, Clone, Serialize)]
pub struct HierarchyNode {
    pub kind: HierarchyNodeKind,
    pub name: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub dimensions: Vec<GraphDimension>,
    pub details: Value,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<HierarchyNode>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ArchitectureHierarchyJson {
    pub schema_version: &'static str,
    pub root: HierarchyNode,
}

pub fn architecture_to_hierarchy_json(architecture: &Architecture) -> ArchitectureHierarchyJson {
    let mut children = Vec::new();
    for memory in &architecture.memories {
        let definition = architecture
            .memory_definition(memory)
            .expect("canonical memory definition");
        children.push(HierarchyNode {
            kind: HierarchyNodeKind::MemoryArray,
            name: memory.name.clone(),
            dimensions: dims(&memory.indices),
            details: json!({
                "definition": memory.definition,
                "capacity_per_instance": definition.capacity,
                "word_size": definition.word_size,
                "banks": definition.banking.as_ref().map(|banking| banking.banks),
            }),
            children: Vec::new(),
        });
    }
    for alias in &architecture.memory_aliases {
        children.push(HierarchyNode {
            kind: HierarchyNodeKind::MemoryAlias,
            name: alias.name.clone(),
            dimensions: Vec::new(),
            details: json!({"endpoint": alias.endpoint}),
            children: Vec::new(),
        });
    }
    for processor in &architecture.processors {
        let valid_instances = processor.instances(architecture).len();
        children.push(HierarchyNode {
            kind: HierarchyNodeKind::ProcessorArray,
            name: processor.name.clone(),
            dimensions: dims(&processor.axes),
            details: json!({
                "definition": processor.definition,
                "connection": processor.connection,
                "valid_instances": valid_instances,
            }),
            children: processor
                .resources
                .iter()
                .map(|resource| HierarchyNode {
                    kind: HierarchyNodeKind::Resource,
                    name: resource.name.clone(),
                    dimensions: dims(&resource.indices),
                    details: json!({"capacity": resource.capacity}),
                    children: Vec::new(),
                })
                .collect(),
        });
    }
    ArchitectureHierarchyJson {
        schema_version: HIERARCHY_SCHEMA_VERSION,
        root: HierarchyNode {
            kind: HierarchyNodeKind::Architecture,
            name: architecture.name.clone(),
            dimensions: dims(&architecture.axes),
            details: json!({}),
            children,
        },
    }
}

pub fn architecture_to_hierarchy_json_value(architecture: &Architecture) -> Value {
    serde_json::to_value(architecture_to_hierarchy_json(architecture))
        .expect("architecture hierarchy serialization must succeed")
}

pub fn architecture_to_hierarchy_json_string_pretty(
    architecture: &Architecture,
) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(&architecture_to_hierarchy_json(architecture))
}

fn dims(indices: &[crate::arch::Axis]) -> Vec<GraphDimension> {
    indices
        .iter()
        .map(|index| GraphDimension {
            name: index.name.clone(),
            size: index.extent,
        })
        .collect()
}
