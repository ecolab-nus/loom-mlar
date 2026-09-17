use mlar_rust::{Expr, FuncPerfModel, ProcessorDefinition, ProcessorType, Resource};

pub fn core_lane() -> Result<ProcessorDefinition, String> {
    Ok(ProcessorDefinition::from_mlir_source(
        "core_lane",
        include_str!("core_lane.mlir"),
        [(
            "elementwise_add",
            FuncPerfModel::builder()
                .symbols(["L"])
                .simple_time_cost(Expr::Const(2), Expr::sym("L"), Expr::Const(32))
                .build(),
        )],
    )?
    .with_type(ProcessorType::Compute)
    .with_resources(vec![
        Resource::exclusive("core_pipeline"),
        Resource::exclusive("core_lane"),
    ]))
}

pub fn dram_l2_dma() -> Result<ProcessorDefinition, String> {
    Ok(ProcessorDefinition::from_mlir_source(
        "dram_l2_dma",
        include_str!("dram_l2_dma.mlir"),
        [(
            "load_l2",
            FuncPerfModel::builder()
                .symbols(["L"])
                .simple_time_cost(
                    Expr::Const(80),
                    Expr::parse("L * 2").map_err(|error| error.to_string())?,
                    Expr::Const(16),
                )
                .build(),
        )],
    )?
    .with_type(ProcessorType::DataMover))
}

pub fn l1_l2_dma() -> Result<ProcessorDefinition, String> {
    Ok(ProcessorDefinition::from_mlir_source(
        "l1_l2_dma",
        include_str!("l1_l2_dma.mlir"),
        [(
            "writeback_l2",
            FuncPerfModel::builder()
                .symbols(["L"])
                .simple_time_cost(
                    Expr::Const(12),
                    Expr::parse("L * 2").map_err(|error| error.to_string())?,
                    Expr::Const(32),
                )
                .build(),
        )],
    )?
    .with_type(ProcessorType::DataMover))
}

pub fn l2_l1_dma() -> Result<ProcessorDefinition, String> {
    Ok(ProcessorDefinition::from_mlir_source(
        "l2_l1_dma",
        include_str!("l2_l1_dma.mlir"),
        [(
            "load_l1",
            FuncPerfModel::builder()
                .symbols(["L"])
                .simple_time_cost(
                    Expr::Const(12),
                    Expr::parse("L * 2").map_err(|error| error.to_string())?,
                    Expr::Const(32),
                )
                .build(),
        )],
    )?
    .with_type(ProcessorType::DataMover))
}
