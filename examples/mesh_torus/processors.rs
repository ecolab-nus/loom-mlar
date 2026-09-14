use mlar_rust::{ProcessorDefinition, ProcessorType, Resource};

pub fn dram_l1_dma() -> Result<ProcessorDefinition, String> {
    Ok(ProcessorDefinition::from_mlir_source_with_perf_yaml(
        "dram_l1_dma",
        include_str!("dram_l1_dma.mlir"),
        include_str!("dram_l1_dma.perf.yaml"),
    )?
    .with_type(ProcessorType::DataMover))
}

pub fn l1_dram_dma() -> Result<ProcessorDefinition, String> {
    Ok(ProcessorDefinition::from_mlir_source_with_perf_yaml(
        "l1_dram_dma",
        include_str!("l1_dram_dma.mlir"),
        include_str!("l1_dram_dma.perf.yaml"),
    )?
    .with_type(ProcessorType::DataMover))
}

pub fn matrix_lane() -> Result<ProcessorDefinition, String> {
    Ok(ProcessorDefinition::from_mlir_source_with_perf_yaml(
        "matrix_lane",
        include_str!("matrix_lane.mlir"),
        include_str!("matrix_lane.perf.yaml"),
    )?
    .with_type(ProcessorType::Compute)
    .with_resources(vec![
        Resource::exclusive("matrix_pipeline"),
        Resource::exclusive("matrix_lane"),
    ]))
}
