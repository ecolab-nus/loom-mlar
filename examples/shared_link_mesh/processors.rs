use mlar_rust::{
    ConstraintExpr, Expr, FuncPerfModel, ProcessorDefinition, ProcessorType, Resource, TimeCost,
};

pub fn lane() -> Result<ProcessorDefinition, String> {
    Ok(ProcessorDefinition::from_mlir_source(
        "lane",
        include_str!("lane.mlir"),
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
    .with_resources(vec![Resource::exclusive("lane_pipeline")]))
}

pub fn link_dma() -> Result<ProcessorDefinition, String> {
    Ok(ProcessorDefinition::from_mlir_source(
        "link_dma",
        include_str!("link_dma.mlir"),
        [(
            "link_copy",
            FuncPerfModel::builder()
                .symbols(["L"])
                .simple_time_cost(
                    Expr::Const(24),
                    Expr::parse("L * 2").map_err(|error| error.to_string())?,
                    Expr::Const(64),
                )
                .build(),
        )],
    )?
    .with_type(ProcessorType::DataMover))
}
