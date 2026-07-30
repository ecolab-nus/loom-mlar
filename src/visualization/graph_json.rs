use crate::arch::{Architecture, EndpointIndex, MemoryEndpoint};
use serde::Serialize;
use serde_json::{Value, json};

const GRAPH_SCHEMA_VERSION: &str = "mlar.arch-graph.v2";

#[derive(Debug, Clone, Serialize)]
pub struct GraphDimension {
    pub name: String,
    pub size: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphNodeKind {
    MemoryArray,
    NamedRegion,
    ProcessorArray,
    ResourceArray,
}

#[derive(Debug, Clone, Serialize)]
pub struct ArchitectureGraphNode {
    pub id: String,
    pub kind: GraphNodeKind,
    pub name: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub dimensions: Vec<GraphDimension>,
    pub details: Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct ArchitectureGraphEdge {
    pub id: String,
    pub kind: &'static str,
    pub source: String,
    pub target: String,
    pub processor: String,
    pub endpoint: MemoryEndpoint,
    pub valid_instances: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct GraphArchitecture {
    pub name: String,
    pub dimensions: Vec<GraphDimension>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ArchitectureGraphJson {
    pub schema_version: &'static str,
    pub architecture: GraphArchitecture,
    pub nodes: Vec<ArchitectureGraphNode>,
    pub edges: Vec<ArchitectureGraphEdge>,
}

pub fn architecture_to_graph_json(architecture: &Architecture) -> ArchitectureGraphJson {
    let mut nodes = Vec::new();
    for memory in &architecture.memories {
        let definition = architecture
            .memory_definition(memory)
            .expect("canonical memory definition");
        nodes.push(ArchitectureGraphNode {
            id: format!("memory:{}", memory.name),
            kind: GraphNodeKind::MemoryArray,
            name: memory.name.clone(),
            dimensions: dimensions(&memory.indices),
            details: json!({
                "definition": memory.definition,
                "capacity_per_instance": definition.capacity,
                "word_size": definition.word_size,
                "banks": definition.banking.as_ref().map(|banking| banking.banks),
                "instances": memory.instances(),
            }),
        });
    }
    for region in &architecture.memory_catalog.regions {
        nodes.push(ArchitectureGraphNode {
            id: format!("region:{}", region.name),
            kind: GraphNodeKind::NamedRegion,
            name: region.name.clone(),
            dimensions: Vec::new(),
            details: json!({"endpoint": region.endpoint}),
        });
    }
    for processor in &architecture.processors {
        let definition = architecture
            .processor_definition(&processor.definition)
            .expect("canonical processor definition");
        nodes.push(ArchitectureGraphNode {
            id: format!("processor:{}", processor.name),
            kind: GraphNodeKind::ProcessorArray,
            name: processor.name.clone(),
            dimensions: dimensions(&processor.relation.domain),
            details: json!({
                "definition": processor.definition,
                "type": definition.processor_type,
                "functions": definition.functions.iter().map(|function| &function.func.name).collect::<Vec<_>>(),
                "connection": processor.connection,
                "valid_instances": processor.relation.instances.len(),
            }),
        });
    }
    for resource in &architecture.resources {
        nodes.push(ArchitectureGraphNode {
            id: format!("resource:{}", resource.name),
            kind: GraphNodeKind::ResourceArray,
            name: resource.name.clone(),
            dimensions: dimensions(&resource.indices),
            details: json!({"capacity": resource.capacity}),
        });
    }

    let mut edges = Vec::new();
    for processor in &architecture.processors {
        for (index, endpoint) in processor.connection.inputs.iter().enumerate() {
            edges.push(ArchitectureGraphEdge {
                id: format!("{}:input:{index}", processor.name),
                kind: "affine_connection",
                source: endpoint_node_id(architecture, endpoint),
                target: format!("processor:{}", processor.name),
                processor: processor.name.clone(),
                endpoint: endpoint.clone(),
                valid_instances: processor.relation.instances.len(),
            });
        }
        for (index, endpoint) in processor.connection.outputs.iter().enumerate() {
            edges.push(ArchitectureGraphEdge {
                id: format!("{}:output:{index}", processor.name),
                kind: "affine_connection",
                source: format!("processor:{}", processor.name),
                target: endpoint_node_id(architecture, endpoint),
                processor: processor.name.clone(),
                endpoint: endpoint.clone(),
                valid_instances: processor.relation.instances.len(),
            });
        }
    }

    ArchitectureGraphJson {
        schema_version: GRAPH_SCHEMA_VERSION,
        architecture: GraphArchitecture {
            name: architecture.name.clone(),
            dimensions: dimensions(&architecture.dimensions),
        },
        nodes,
        edges,
    }
}

fn endpoint_node_id(architecture: &Architecture, endpoint: &MemoryEndpoint) -> String {
    if architecture
        .memory_catalog
        .region(&endpoint.memory)
        .is_some()
    {
        format!("region:{}", endpoint.memory)
    } else {
        format!("memory:{}", endpoint.memory)
    }
}

pub fn architecture_to_graph_json_value(architecture: &Architecture) -> Value {
    serde_json::to_value(architecture_to_graph_json(architecture))
        .expect("architecture graph serialization must succeed")
}

pub fn architecture_to_graph_json_string(
    architecture: &Architecture,
) -> Result<String, serde_json::Error> {
    serde_json::to_string(&architecture_to_graph_json(architecture))
}

pub fn architecture_to_graph_json_string_pretty(
    architecture: &Architecture,
) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(&architecture_to_graph_json(architecture))
}

fn dimensions(indices: &[crate::arch::IndexDomain]) -> Vec<GraphDimension> {
    indices
        .iter()
        .map(|index| GraphDimension {
            name: index.name.clone(),
            size: index.size,
        })
        .collect()
}

#[allow(dead_code)]
fn endpoint_is_region(endpoint: &MemoryEndpoint) -> bool {
    endpoint
        .indices
        .iter()
        .any(|index| matches!(index, EndpointIndex::All))
}
