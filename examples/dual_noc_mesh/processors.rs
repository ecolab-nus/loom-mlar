use mlar_rust::{
    ConstraintExpr, Expr, FuncPerfModel, ProcessorDefinition, ProcessorType, Resource, TimeCost,
};

pub fn dram_l1_noc0() -> Result<ProcessorDefinition, String> {
    Ok(ProcessorDefinition::from_mlir_source(
        "dram_l1_noc0",
        include_str!("dram_l1_noc0.mlir"),
        [
            (
                "dram_to_l1_S_f16",
                FuncPerfModel::builder()
                    .symbols(["M", "N", "effective_bandwidth"])
                    .simple_time_cost(
                        Expr::Const(454),
                        Expr::parse("M * N * 2 * effective_bandwidth")
                            .map_err(|error| error.to_string())?,
                        Expr::Const(150),
                    )
                    .build(),
            ),
            (
                "dram_to_l1_S_bcst",
                FuncPerfModel::builder()
                    .symbols(["M", "N", "bcst_x", "bcst_y", "effective_bandwidth"])
                    .simple_time_cost(
                        Expr::Const(344),
                        Expr::parse("M * N * 2 * effective_bandwidth")
                            .map_err(|error| error.to_string())?,
                        Expr::Const(150),
                    )
                    .build(),
            ),
            (
                "dram_to_l1_R_f16",
                FuncPerfModel::builder()
                    .symbols(["M", "N", "effective_bandwidth"])
                    .simple_time_cost(
                        Expr::Const(888),
                        Expr::parse("M * N * 2 * effective_bandwidth")
                            .map_err(|error| error.to_string())?,
                        Expr::Const(888),
                    )
                    .build(),
            ),
            (
                "dram_to_l1_R_bcst",
                FuncPerfModel::builder()
                    .symbols(["M", "N", "bcst_x", "bcst_y", "effective_bandwidth"])
                    .simple_time_cost(
                        Expr::Const(888),
                        Expr::parse("M * N * 2 * effective_bandwidth")
                            .map_err(|error| error.to_string())?,
                        Expr::Const(888),
                    )
                    .build(),
            ),
        ],
    )?
    .with_type(ProcessorType::DataMover))
}

pub fn l1_dram_noc1() -> Result<ProcessorDefinition, String> {
    Ok(ProcessorDefinition::from_mlir_source(
        "l1_dram_noc1",
        include_str!("l1_dram_noc1.mlir"),
        [(
            "l1_to_dram_f16",
            FuncPerfModel::builder()
                .symbols(["M", "N"])
                .simple_time_cost(
                    Expr::Const(454),
                    Expr::parse("M * N * 2").map_err(|error| error.to_string())?,
                    Expr::Const(150),
                )
                .build(),
        )],
    )?
    .with_type(ProcessorType::DataMover))
}

pub fn l1_l1_noc0() -> Result<ProcessorDefinition, String> {
    Ok(ProcessorDefinition::from_mlir_source(
        "l1_l1_noc0",
        include_str!("l1_l1_noc0.mlir"),
        [(
            "l1_gather",
            FuncPerfModel::builder()
                .symbols(["B", "M", "N", "effective_bandwidth", "gather_x", "gather_y"])
                .simple_time_cost(
                    Expr::Const(344),
                    Expr::parse("B * M * N * 2 * effective_bandwidth")
                        .map_err(|error| error.to_string())?,
                    Expr::Const(28),
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
        [
            (
                "matmul_SS_f16",
                FuncPerfModel::builder()
                    .symbols(["K", "M", "N"])
                    .scenario_with_constraints(
                        ConstraintExpr::parse("(M >= 32 && N >= 32 && K >= 32) && (M * N >= 8192)")
                            .map_err(|error| error.to_string())?,
                        TimeCost::throughput(
                            Expr::parse("M * N / 2").map_err(|error| error.to_string())?,
                            Expr::parse("2 * M * N * K").map_err(|error| error.to_string())?,
                            Expr::Const(716),
                        ),
                    )
                    .scenario_with_constraints(
                        ConstraintExpr::parse("(M >= 32 && N >= 32 && K >= 32) && (M * N < 8192)")
                            .map_err(|error| error.to_string())?,
                        TimeCost::throughput(
                            Expr::parse("M * N / 2").map_err(|error| error.to_string())?,
                            Expr::parse("2 * M * N * K").map_err(|error| error.to_string())?,
                            Expr::parse("M * N * 716 / 8192").map_err(|error| error.to_string())?,
                        ),
                    )
                    .build(),
            ),
            (
                "matmul_SR_f16",
                FuncPerfModel::builder()
                    .symbols(["K", "M", "N"])
                    .scenario_with_constraints(
                        ConstraintExpr::parse("M >= 32 && N >= 32 && K >= 32")
                            .map_err(|error| error.to_string())?,
                        TimeCost::throughput(
                            Expr::Const(888),
                            Expr::parse("2 * M * N * K").map_err(|error| error.to_string())?,
                            Expr::Const(888),
                        ),
                    )
                    .build(),
            ),
            (
                "matmul_RS_f16",
                FuncPerfModel::builder()
                    .symbols(["K", "M", "N"])
                    .scenario_with_constraints(
                        ConstraintExpr::parse("M >= 32 && N >= 32 && K >= 32")
                            .map_err(|error| error.to_string())?,
                        TimeCost::throughput(
                            Expr::Const(888),
                            Expr::parse("2 * M * N * K").map_err(|error| error.to_string())?,
                            Expr::Const(888),
                        ),
                    )
                    .build(),
            ),
            (
                "matmul_RR_f16",
                FuncPerfModel::builder()
                    .symbols(["K", "M", "N"])
                    .scenario_with_constraints(
                        ConstraintExpr::parse("M >= 32 && N >= 32 && K >= 32")
                            .map_err(|error| error.to_string())?,
                        TimeCost::throughput(
                            Expr::Const(888),
                            Expr::parse("2 * M * N * K").map_err(|error| error.to_string())?,
                            Expr::Const(888),
                        ),
                    )
                    .build(),
            ),
            (
                "batch_matmul_SS_f16",
                FuncPerfModel::builder()
                    .symbols(["B", "K", "M", "N"])
                    .scenario_with_constraints(
                        ConstraintExpr::parse(
                            "(B >= 1 && M >= 32 && N >= 32 && K >= 32) && (M * N >= 8192)",
                        )
                        .map_err(|error| error.to_string())?,
                        TimeCost::throughput(
                            Expr::parse("M * N / 2").map_err(|error| error.to_string())?,
                            Expr::parse("2 * B * M * N * K").map_err(|error| error.to_string())?,
                            Expr::Const(716),
                        ),
                    )
                    .scenario_with_constraints(
                        ConstraintExpr::parse(
                            "(B >= 1 && M >= 32 && N >= 32 && K >= 32) && (M * N < 8192)",
                        )
                        .map_err(|error| error.to_string())?,
                        TimeCost::throughput(
                            Expr::parse("M * N / 2").map_err(|error| error.to_string())?,
                            Expr::parse("2 * B * M * N * K").map_err(|error| error.to_string())?,
                            Expr::parse("M * N * 716 / 8192").map_err(|error| error.to_string())?,
                        ),
                    )
                    .build(),
            ),
            (
                "batch_matmul_SR_f16",
                FuncPerfModel::builder()
                    .symbols(["B", "K", "M", "N"])
                    .scenario_with_constraints(
                        ConstraintExpr::parse("B >= 1 && M >= 32 && N >= 32 && K >= 32")
                            .map_err(|error| error.to_string())?,
                        TimeCost::throughput(
                            Expr::Const(888),
                            Expr::parse("2 * B * M * N * K").map_err(|error| error.to_string())?,
                            Expr::Const(888),
                        ),
                    )
                    .build(),
            ),
            (
                "batch_matmul_RS_f16",
                FuncPerfModel::builder()
                    .symbols(["B", "K", "M", "N"])
                    .scenario_with_constraints(
                        ConstraintExpr::parse("B >= 1 && M >= 32 && N >= 32 && K >= 32")
                            .map_err(|error| error.to_string())?,
                        TimeCost::throughput(
                            Expr::Const(888),
                            Expr::parse("2 * B * M * N * K").map_err(|error| error.to_string())?,
                            Expr::Const(888),
                        ),
                    )
                    .build(),
            ),
            (
                "batch_matmul_RR_f16",
                FuncPerfModel::builder()
                    .symbols(["B", "K", "M", "N"])
                    .scenario_with_constraints(
                        ConstraintExpr::parse("B >= 1 && M >= 32 && N >= 32 && K >= 32")
                            .map_err(|error| error.to_string())?,
                        TimeCost::throughput(
                            Expr::Const(888),
                            Expr::parse("2 * B * M * N * K").map_err(|error| error.to_string())?,
                            Expr::Const(888),
                        ),
                    )
                    .build(),
            ),
            (
                "vec_vsum_f16",
                FuncPerfModel::builder()
                    .symbols(["P", "R"])
                    .simple_time_cost(
                        Expr::Const(1),
                        Expr::parse("P * R").map_err(|error| error.to_string())?,
                        Expr::Const(128),
                    )
                    .build(),
            ),
            (
                "vec_vmax_f16",
                FuncPerfModel::builder()
                    .symbols(["P", "R"])
                    .simple_time_cost(
                        Expr::Const(1),
                        Expr::parse("P * R").map_err(|error| error.to_string())?,
                        Expr::Const(128),
                    )
                    .build(),
            ),
            (
                "vec_max1_f16",
                FuncPerfModel::builder()
                    .symbols(["L"])
                    .simple_time_cost(Expr::Const(1), Expr::sym("L"), Expr::Const(128))
                    .build(),
            ),
            (
                "elementwise_add_f16",
                FuncPerfModel::builder()
                    .symbols(["M", "N"])
                    .simple_time_cost(
                        Expr::Const(10),
                        Expr::parse("M * N").map_err(|error| error.to_string())?,
                        Expr::Const(43),
                    )
                    .build(),
            ),
            (
                "elementwise_mul_f16",
                FuncPerfModel::builder()
                    .symbols(["M", "N"])
                    .simple_time_cost(
                        Expr::Const(10),
                        Expr::parse("M * N").map_err(|error| error.to_string())?,
                        Expr::Const(15),
                    )
                    .build(),
            ),
        ],
    )?
    .with_type(ProcessorType::Compute)
    .with_resources(vec![Resource::exclusive("matrix_lane")]))
}

pub fn vector_lane() -> Result<ProcessorDefinition, String> {
    Ok(ProcessorDefinition::from_mlir_source(
        "vector_lane",
        include_str!("vector_lane.mlir"),
        [
            (
                "vec_max_f16",
                FuncPerfModel::builder()
                    .symbols(["L"])
                    .simple_time_cost(Expr::Const(1), Expr::sym("L"), Expr::Const(1024))
                    .build(),
            ),
            (
                "vec_exp_f16",
                FuncPerfModel::builder()
                    .symbols(["L"])
                    .simple_time_cost(Expr::Const(1), Expr::sym("L"), Expr::Const(7))
                    .build(),
            ),
            (
                "vec_sum_f16",
                FuncPerfModel::builder()
                    .symbols(["L"])
                    .simple_time_cost(Expr::Const(1), Expr::sym("L"), Expr::Const(1024))
                    .build(),
            ),
            (
                "vec_add_f16",
                FuncPerfModel::builder()
                    .symbols(["L"])
                    .simple_time_cost(Expr::Const(1), Expr::sym("L"), Expr::Const(1024))
                    .build(),
            ),
            (
                "vec_mul_f16",
                FuncPerfModel::builder()
                    .symbols(["L"])
                    .simple_time_cost(Expr::Const(1), Expr::sym("L"), Expr::Const(1024))
                    .build(),
            ),
            (
                "vec_div_f16",
                FuncPerfModel::builder()
                    .symbols(["L"])
                    .simple_time_cost(Expr::Const(1), Expr::sym("L"), Expr::Const(6))
                    .build(),
            ),
            (
                "vec_sub_f16",
                FuncPerfModel::builder()
                    .symbols(["L"])
                    .simple_time_cost(Expr::Const(1), Expr::sym("L"), Expr::Const(4))
                    .build(),
            ),
            (
                "vec_powf_f16",
                FuncPerfModel::builder()
                    .symbols(["L"])
                    .simple_time_cost(Expr::Const(1), Expr::sym("L"), Expr::Const(7))
                    .build(),
            ),
            (
                "vec_cmpf_ogt_f16",
                FuncPerfModel::builder()
                    .symbols(["L"])
                    .simple_time_cost(Expr::Const(1), Expr::sym("L"), Expr::Const(8))
                    .build(),
            ),
            (
                "vec_select_f16",
                FuncPerfModel::builder()
                    .symbols(["L"])
                    .simple_time_cost(Expr::Const(1), Expr::sym("L"), Expr::Const(8))
                    .build(),
            ),
            (
                "vec_log_f16",
                FuncPerfModel::builder()
                    .symbols(["L"])
                    .simple_time_cost(Expr::Const(1), Expr::sym("L"), Expr::Const(7))
                    .build(),
            ),
        ],
    )?
    .with_type(ProcessorType::Compute)
    .with_resources(vec![Resource::exclusive("vector_lane")]))
}
