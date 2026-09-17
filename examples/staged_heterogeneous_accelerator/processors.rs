use mlar_rust::{Expr, FuncPerfModel, PerfScenario, ProcessorDefinition, ProcessorType};

pub fn gcram_to_stage() -> Result<ProcessorDefinition, String> {
    Ok(ProcessorDefinition::from_mlir_source(
        "gcram_to_stage",
        include_str!("gcram_to_stage.mlir"),
        [
            (
                "load_gcram_f16",
                FuncPerfModel::builder()
                    .symbols(["K", "M"])
                    .scenario(PerfScenario::new(
                        Expr::parse("8 + (2 * M * K + 15) / 16")
                            .map_err(|error| error.to_string())?,
                    ))
                    .build(),
            ),
            (
                "load_gcram_f16_broadcast",
                FuncPerfModel::builder()
                    .symbols(["K", "M", "bcst_x", "bcst_y"])
                    .scenario(PerfScenario::new(
                        Expr::parse("8 + (2 * M * K * bcst_x * bcst_y + 15) / 16")
                            .map_err(|error| error.to_string())?,
                    ))
                    .build(),
            ),
        ],
    )?
    .with_type(ProcessorType::DataMover))
}

pub fn rram_to_stage() -> Result<ProcessorDefinition, String> {
    Ok(ProcessorDefinition::from_mlir_source(
        "rram_to_stage",
        include_str!("rram_to_stage.mlir"),
        [
            (
                "load_rram_f16",
                FuncPerfModel::builder()
                    .symbols(["K", "N"])
                    .scenario(PerfScenario::new(
                        Expr::parse("8 + (2 * K * N + 7) / 8")
                            .map_err(|error| error.to_string())?,
                    ))
                    .build(),
            ),
            (
                "load_rram_f16_broadcast",
                FuncPerfModel::builder()
                    .symbols(["K", "N", "bcst_x", "bcst_y"])
                    .scenario(PerfScenario::new(
                        Expr::parse("8 + (2 * K * N * bcst_x * bcst_y + 7) / 8")
                            .map_err(|error| error.to_string())?,
                    ))
                    .build(),
            ),
        ],
    )?
    .with_type(ProcessorType::DataMover))
}

pub fn matrix_lane() -> Result<ProcessorDefinition, String> {
    Ok(ProcessorDefinition::from_mlir_source(
        "matrix_lane",
        include_str!("matrix_lane.mlir"),
        [(
            "matmul_staged_f16_f32",
            FuncPerfModel::builder()
                .symbols(["K", "M", "N"])
                .scenario(PerfScenario::new(
                    Expr::parse(
                        "8 + max((M * N * K + 255) / 256, \
                         max((2 * (M * K + K * N) + 31) / 32, (4 * M * N + 15) / 16))",
                    )
                    .map_err(|error| error.to_string())?,
                ))
                .build(),
        )],
    )?
    .with_type(ProcessorType::Compute))
}
