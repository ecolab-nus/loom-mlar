use mlar_rust::{
    ConstraintExpr, Expr, FuncPerfModel, ProcessorDefinition, ProcessorType, Resource, TimeCost,
};

fn mover(
    name: &str,
    source: &str,
    functions: impl IntoIterator<Item = (&'static str, FuncPerfModel)>,
) -> Result<ProcessorDefinition, String> {
    Ok(
        ProcessorDefinition::from_mlir_source(name, source, functions)?
            .with_type(ProcessorType::DataMover),
    )
}

pub fn dram_l1_noc0() -> Result<ProcessorDefinition, String> {
    mover(
        "dram_l1_noc0",
        include_str!("dram_l1_noc0.mlir"),
        [
            (
                "dram_to_l1_f16",
                FuncPerfModel::builder()
                    .symbols(["M", "N"])
                    .simple_time_cost(
                        Expr::Const(40),
                        Expr::parse("2 * M * N").map_err(|error| error.to_string())?,
                        Expr::Const(128),
                    )
                    .build(),
            ),
            (
                "dram_to_l1_broadcast_f16",
                FuncPerfModel::builder()
                    .symbols(["M", "N", "bcst_x", "bcst_y"])
                    .simple_time_cost(
                        Expr::Const(40),
                        Expr::parse("2 * M * N * bcst_x * bcst_y")
                            .map_err(|error| error.to_string())?,
                        Expr::Const(128),
                    )
                    .build(),
            ),
        ],
    )
}

pub fn l1_dram_noc1() -> Result<ProcessorDefinition, String> {
    mover(
        "l1_dram_noc1",
        include_str!("l1_dram_noc1.mlir"),
        [(
            "l1_to_dram_f16",
            FuncPerfModel::builder()
                .symbols(["M", "N"])
                .simple_time_cost(
                    Expr::Const(40),
                    Expr::parse("2 * M * N").map_err(|error| error.to_string())?,
                    Expr::Const(128),
                )
                .build(),
        )],
    )
}

pub fn l1_l1_noc0() -> Result<ProcessorDefinition, String> {
    mover(
        "l1_l1_noc0",
        include_str!("l1_l1_noc0.mlir"),
        [(
            "l1_gather_f16",
            FuncPerfModel::builder()
                .symbols(["B", "M", "N", "gather_x", "gather_y"])
                .simple_time_cost(
                    Expr::Const(12),
                    Expr::parse("2 * B * M * N").map_err(|error| error.to_string())?,
                    Expr::Const(64),
                )
                .build(),
        )],
    )
}

pub fn matrix_lane() -> Result<ProcessorDefinition, String> {
    Ok(ProcessorDefinition::from_mlir_source(
        "matrix_lane",
        include_str!("matrix_lane.mlir"),
        [(
            "matmul_f16",
            FuncPerfModel::builder()
                .symbols(["K", "M", "N"])
                .scenario_with_constraints(
                    ConstraintExpr::parse("M >= 16 && N >= 16 && K >= 16")
                        .map_err(|error| error.to_string())?,
                    TimeCost::throughput(
                        Expr::Const(8),
                        Expr::parse("2 * M * N * K").map_err(|error| error.to_string())?,
                        Expr::Const(512),
                    ),
                )
                .build(),
        )],
    )?
    .with_type(ProcessorType::Compute)
    .with_resources(vec![Resource::exclusive("matrix_pipeline")]))
}

pub fn vector_lane() -> Result<ProcessorDefinition, String> {
    Ok(ProcessorDefinition::from_mlir_source(
        "vector_lane",
        include_str!("vector_lane.mlir"),
        [(
            "relu_f16",
            FuncPerfModel::builder()
                .symbols(["L"])
                .simple_time_cost(Expr::Const(2), Expr::sym("L"), Expr::Const(128))
                .build(),
        )],
    )?
    .with_type(ProcessorType::Compute)
    .with_resources(vec![Resource::exclusive("vector_pipeline")]))
}
