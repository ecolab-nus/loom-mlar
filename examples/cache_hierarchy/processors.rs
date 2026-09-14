use mlar_rust::{ProcessorDefinition, ProcessorType, Resource};

pub fn core_lane() -> Result<ProcessorDefinition, String> {
    Ok(ProcessorDefinition::from_mlir_source_with_perf_yaml(
        "core_lane",
        include_str!("core_lane.mlir"),
        include_str!("core_lane.perf.yaml"),
    )?
    .with_type(ProcessorType::Compute)
    .with_resources(vec![
        Resource::exclusive("core_pipeline"),
        Resource::exclusive("core_lane"),
    ]))
}

pub fn dram_l2_dma() -> Result<ProcessorDefinition, String> {
    Ok(ProcessorDefinition::from_mlir_source_with_perf_yaml(
        "dram_l2_dma",
        include_str!("dram_l2_dma.mlir"),
        include_str!("dram_l2_dma.perf.yaml"),
    )?
    .with_type(ProcessorType::DataMover))
}

pub fn l1_l2_dma() -> Result<ProcessorDefinition, String> {
    Ok(ProcessorDefinition::from_mlir_source_with_perf_yaml(
        "l1_l2_dma",
        include_str!("l1_l2_dma.mlir"),
        include_str!("l1_l2_dma.perf.yaml"),
    )?
    .with_type(ProcessorType::DataMover))
}

pub fn l2_l1_dma() -> Result<ProcessorDefinition, String> {
    Ok(ProcessorDefinition::from_mlir_source_with_perf_yaml(
        "l2_l1_dma",
        include_str!("l2_l1_dma.mlir"),
        include_str!("l2_l1_dma.perf.yaml"),
    )?
    .with_type(ProcessorType::DataMover))
}
