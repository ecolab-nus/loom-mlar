use mlar_rust::{ProcessorDefinition, ProcessorType, Resource};

pub fn lane() -> Result<ProcessorDefinition, String> {
    Ok(ProcessorDefinition::from_mlir_source_with_perf_yaml(
        "lane",
        include_str!("lane.mlir"),
        include_str!("lane.perf.yaml"),
    )?
    .with_type(ProcessorType::Compute)
    .with_resources(vec![Resource::exclusive("lane_pipeline")]))
}

pub fn link_dma() -> Result<ProcessorDefinition, String> {
    Ok(ProcessorDefinition::from_mlir_source_with_perf_yaml(
        "link_dma",
        include_str!("link_dma.mlir"),
        include_str!("link_dma.perf.yaml"),
    )?
    .with_type(ProcessorType::DataMover))
}
