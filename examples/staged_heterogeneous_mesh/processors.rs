use mlar_rust::{ProcessorDefinition, ProcessorType};

pub fn gcram_to_stage() -> Result<ProcessorDefinition, String> {
    Ok(ProcessorDefinition::from_mlir_source_with_perf_yaml(
        "gcram_to_stage",
        include_str!("gcram_to_stage.mlir"),
        include_str!("gcram_to_stage.perf.yaml"),
    )?
    .with_type(ProcessorType::DataMover))
}

pub fn rram_to_stage() -> Result<ProcessorDefinition, String> {
    Ok(ProcessorDefinition::from_mlir_source_with_perf_yaml(
        "rram_to_stage",
        include_str!("rram_to_stage.mlir"),
        include_str!("rram_to_stage.perf.yaml"),
    )?
    .with_type(ProcessorType::DataMover))
}

pub fn matrix_lane() -> Result<ProcessorDefinition, String> {
    Ok(ProcessorDefinition::from_mlir_source_with_perf_yaml(
        "matrix_lane",
        include_str!("matrix_lane.mlir"),
        include_str!("matrix_lane.perf.yaml"),
    )?
    .with_type(ProcessorType::Compute))
}
