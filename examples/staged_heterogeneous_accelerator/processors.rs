use mlar_rust::{Expr, FuncPerfModel, PerfScenario, ProcessorDefinition, ProcessorType};

fn mover(
    name: &str,
    source: &str,
    function: &'static str,
    throughput: i64,
) -> Result<ProcessorDefinition, String> {
    Ok(ProcessorDefinition::from_mlir_source(
        name,
        source,
        [(
            function,
            FuncPerfModel::builder()
                .symbols(["M", "N"])
                .simple_time_cost(
                    Expr::Const(40),
                    Expr::parse("2 * M * N").map_err(|error| error.to_string())?,
                    Expr::Const(throughput),
                )
                .build(),
        )],
    )?
    .with_type(ProcessorType::DataMover))
}

pub fn dram_to_sram() -> Result<ProcessorDefinition, String> {
    mover(
        "dram_to_sram",
        include_str!("dram_to_sram.mlir"),
        "dram_to_sram_f16",
        128,
    )
}

pub fn sram_to_dram() -> Result<ProcessorDefinition, String> {
    mover(
        "sram_to_dram",
        include_str!("sram_to_dram.mlir"),
        "sram_to_dram_f16",
        128,
    )
}

pub fn dram_to_rram() -> Result<ProcessorDefinition, String> {
    mover(
        "dram_to_rram",
        include_str!("dram_to_rram.mlir"),
        "dram_to_rram_f16",
        64,
    )
}

pub fn rram_to_dram() -> Result<ProcessorDefinition, String> {
    mover(
        "rram_to_dram",
        include_str!("rram_to_dram.mlir"),
        "rram_to_dram_f16",
        64,
    )
}

pub fn sram_to_stage() -> Result<ProcessorDefinition, String> {
    Ok(ProcessorDefinition::from_mlir_source(
        "sram_to_stage",
        include_str!("sram_to_stage.mlir"),
        [
            (
                "load_sram_f16",
                FuncPerfModel::builder()
                    .symbols(["K", "M"])
                    .scenario(PerfScenario::new(
                        Expr::parse("8 + (2 * M * K + 15) / 16")
                            .map_err(|error| error.to_string())?,
                    ))
                    .build(),
            ),
            (
                "load_sram_f16_broadcast",
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
