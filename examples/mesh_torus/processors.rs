use mlar_rust::{
    ConstraintExpr, Expr, FuncPerfModel, ProcessorDefinition, ProcessorType, Resource, TimeCost,
};

pub fn dram_l1_dma() -> Result<ProcessorDefinition, String> {
    Ok(ProcessorDefinition::from_mlir_source(
        "dram_l1_dma",
        include_str!("dram_l1_dma.mlir"),
        [
            (
                "load_l1",
                FuncPerfModel::builder()
                    .symbols(["L"])
                    .simple_time_cost(
                        Expr::Const(100),
                        Expr::parse("L * 2").map_err(|error| error.to_string())?,
                        Expr::Const(32),
                    )
                    .build(),
            ),
            (
                "load_l1_broadcast",
                FuncPerfModel::builder()
                    .symbols(["L", "bcst_x", "bcst_y"])
                    .simple_time_cost(
                        Expr::Const(100),
                        Expr::parse("L * 2").map_err(|error| error.to_string())?,
                        Expr::Const(32),
                    )
                    .build(),
            ),
        ],
    )?
    .with_type(ProcessorType::DataMover))
}

pub fn l1_dram_dma() -> Result<ProcessorDefinition, String> {
    Ok(ProcessorDefinition::from_mlir_source(
        "l1_dram_dma",
        include_str!("l1_dram_dma.mlir"),
        [(
            "writeback_dram",
            FuncPerfModel::builder()
                .symbols(["L"])
                .simple_time_cost(
                    Expr::Const(100),
                    Expr::parse("L * 2").map_err(|error| error.to_string())?,
                    Expr::Const(32),
                )
                .build(),
        )],
    )?
    .with_type(ProcessorType::DataMover))
}

pub fn matrix_lane() -> Result<ProcessorDefinition, String> {
    Ok(ProcessorDefinition::from_mlir_source(
        "matrix_lane",
        include_str!("matrix_lane.mlir"),
        [(
            "matmul",
            FuncPerfModel::builder()
                .symbols(["K", "M", "N"])
                .scenario_with_constraints(
                    ConstraintExpr::parse("M > 0 && N > 0 && K > 0")
                        .map_err(|error| error.to_string())?,
                    TimeCost::throughput(
                        Expr::Const(16),
                        Expr::parse("2 * M * N * K").map_err(|error| error.to_string())?,
                        Expr::Const(128),
                    ),
                )
                .build(),
        )],
    )?
    .with_type(ProcessorType::Compute)
    .with_resources(vec![
        Resource::exclusive("matrix_pipeline"),
        Resource::exclusive("matrix_lane"),
    ]))
}
