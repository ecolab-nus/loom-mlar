use mlar_rust::*;

/// Matrix lane processor with matmul performance model (M, N, K symbols).
///
/// - Global constraints: M ≥ 32, N ≥ 32, K ≥ 32
/// - Scenario 1: M*N ≥ 8192 → throughput = 1024, latency = 1
/// - Scenario 2: M*N ≤ 8192 → throughput = (M*N / 8192) * 1024, latency = 1
pub fn matrix_lane() -> Architecture {
    let functionality = Module::from_mlir("tests/2d_mesh/compute/matrix_lane.mlir")
        .expect("tests/2d_mesh/compute/matrix_lane.mlir should parse");

    let mat_func_perf = FuncPerfModel {
        symbols: vec![Sym::new("M"), Sym::new("N"), Sym::new("K")],
        constraints: ConstraintExpr::And(vec![
            ConstraintExpr::Ge(Expr::sym("M"), Expr::Const(32)),
            ConstraintExpr::Ge(Expr::sym("N"), Expr::Const(32)),
            ConstraintExpr::Ge(Expr::sym("K"), Expr::Const(32)),
        ]),
        scenarios: vec![
            PerfScenario {
                constraints: ConstraintExpr::Ge(
                    Expr::mul(Expr::sym("M"), Expr::sym("N")),
                    Expr::Const(8192),
                ),
                time_cost: TimeCost::Simple(SimpleTimeCost {
                    fixed_latency: Expr::Const(1),
                    volume: Expr::mul(Expr::mul(Expr::sym("M"), Expr::sym("N")), Expr::sym("K")),
                    throughput: Expr::Const(1024),
                }),
            },
            PerfScenario {
                constraints: ConstraintExpr::Le(
                    Expr::mul(Expr::sym("M"), Expr::sym("N")),
                    Expr::Const(8192),
                ),
                time_cost: TimeCost::Simple(SimpleTimeCost {
                    fixed_latency: Expr::Const(1),
                    volume: Expr::mul(Expr::mul(Expr::sym("M"), Expr::sym("N")), Expr::sym("K")),
                    throughput: Expr::mul(
                        Expr::div(Expr::mul(Expr::sym("M"), Expr::sym("N")), Expr::Const(8192)),
                        Expr::Const(1024),
                    ),
                }),
            },
        ],
    };

    let batch_mat_func_perf = FuncPerfModel {
        symbols: vec![Sym::new("Batch"), Sym::new("M"), Sym::new("N"), Sym::new("K")],
        constraints: ConstraintExpr::And(vec![
            ConstraintExpr::Ge(Expr::sym("Batch"), Expr::Const(1)),
            ConstraintExpr::Ge(Expr::sym("M"), Expr::Const(32)),
            ConstraintExpr::Ge(Expr::sym("N"), Expr::Const(32)),
            ConstraintExpr::Ge(Expr::sym("K"), Expr::Const(32)),
        ]),
        scenarios: vec![
            PerfScenario {
                constraints: ConstraintExpr::Ge(
                    Expr::mul(Expr::sym("M"), Expr::sym("N")),
                    Expr::Const(8192),
                ),
                time_cost: TimeCost::Simple(SimpleTimeCost {
                    fixed_latency: Expr::Const(1),
                    volume: Expr::mul(
                        Expr::mul(Expr::mul(Expr::sym("Batch"), Expr::sym("M")), Expr::sym("N")),
                        Expr::sym("K"),
                    ),
                    throughput: Expr::Const(1024),
                }),
            },
            PerfScenario {
                constraints: ConstraintExpr::Le(
                    Expr::mul(Expr::sym("M"), Expr::sym("N")),
                    Expr::Const(8192),
                ),
                time_cost: TimeCost::Simple(SimpleTimeCost {
                    fixed_latency: Expr::Const(1),
                    volume: Expr::mul(
                        Expr::mul(Expr::mul(Expr::sym("Batch"), Expr::sym("M")), Expr::sym("N")),
                        Expr::sym("K"),
                    ),
                    throughput: Expr::mul(
                        Expr::div(Expr::mul(Expr::sym("M"), Expr::sym("N")), Expr::Const(8192)),
                        Expr::Const(1024),
                    ),
                }),
            },
        ],
    };

    Processor::from_module("matrix_lane", functionality, vec![mat_func_perf, batch_mat_func_perf])
        .expect("matrix_lane processor should link functionality and perf")
        .into_elem()
}
