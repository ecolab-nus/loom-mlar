use mlar_rust::{ProcessorDefinition, ProcessorType, Resource};

pub fn dram_l1_noc0() -> Result<ProcessorDefinition, String> {
    Ok(ProcessorDefinition::from_mlir_source_with_perf_yaml(
        "dram_l1_noc0",
        include_str!("dram_l1_noc0.mlir"),
        include_str!("dram_l1_noc0.perf.yaml"),
    )?
    .with_type(ProcessorType::DataMover))
}

pub fn l1_dram_noc1() -> Result<ProcessorDefinition, String> {
    Ok(ProcessorDefinition::from_mlir_source_with_perf_yaml(
        "l1_dram_noc1",
        include_str!("l1_dram_noc1.mlir"),
        include_str!("l1_dram_noc1.perf.yaml"),
    )?
    .with_type(ProcessorType::DataMover))
}

pub fn l1_l1_noc0() -> Result<ProcessorDefinition, String> {
    Ok(ProcessorDefinition::from_mlir_source_with_perf_yaml(
        "l1_l1_noc0",
        include_str!("l1_l1_noc0.mlir"),
        include_str!("l1_l1_noc0.perf.yaml"),
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
    .with_resources(vec![Resource::exclusive("matrix_lane")]))
}

pub fn vector_lane() -> Result<ProcessorDefinition, String> {
    Ok(ProcessorDefinition::from_mlir_source_with_perf_yaml(
        "vector_lane",
        include_str!("vector_lane.mlir"),
        include_str!("vector_lane.perf.yaml"),
    )?
    .with_type(ProcessorType::Compute)
    .with_resources(vec![Resource::exclusive("vector_lane")]))
}
