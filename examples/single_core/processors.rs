use mlar_rust::{ProcessorDefinition, ProcessorType, Resource};

pub fn vector_lane() -> Result<ProcessorDefinition, String> {
    Ok(ProcessorDefinition::from_mlir_source_with_perf_yaml(
        "vector_lane",
        include_str!("vector_lane.mlir"),
        include_str!("vector_lane.perf.yaml"),
    )?
    .with_type(ProcessorType::Compute)
    .with_resources(vec![
        Resource::exclusive("vector_pipeline"),
        Resource::exclusive("vector_lane"),
    ]))
}
