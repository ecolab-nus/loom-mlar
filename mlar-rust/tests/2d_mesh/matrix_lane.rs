use mlar_rust::*;

use crate::memory::l1;

/// Matrix lane processor with matmul performance model (M, N, K symbols).
///
/// - Global constraints: M ≥ 32, N ≥ 32, K ≥ 32
/// - Scenario 1: M*N ≥ 8192 → throughput = 1024, latency = 1
/// - Scenario 2: M*N ≤ 8192 → throughput = (M*N / 8192) * 1024, latency = 1
pub fn matrix_lane() -> Processor {
    let mat_func_perf = FuncPerfModel {
        symbols: vec![Sym::new("M"), Sym::new("N"), Sym::new("K")],
        constraints: ConstraintExpr::And(vec![
            ConstraintExpr::Ge(Expr::sym("M"), Expr::Const(32)),
            ConstraintExpr::Ge(Expr::sym("N"), Expr::Const(32)),
            ConstraintExpr::Ge(Expr::sym("K"), Expr::Const(32)),
        ]),
        scenarios: vec![
            // Scenario 1: large M*N (≥ 8192)
            PerfScenario {
                constraints: ConstraintExpr::Ge(
                    Expr::mul(Expr::sym("M"), Expr::sym("N")),
                    Expr::Const(8192),
                ),
                time_cost: TimeCostExpr {
                    fixed_latency: Expr::Const(1),
                    throughput: Expr::Const(1024),
                },
            },
            // Scenario 2: small M*N (≤ 8192)
            PerfScenario {
                constraints: ConstraintExpr::Le(
                    Expr::mul(Expr::sym("M"), Expr::sym("N")),
                    Expr::Const(8192),
                ),
                time_cost: TimeCostExpr {
                    fixed_latency: Expr::Const(1),
                    throughput: Expr::mul(
                        Expr::div(
                            Expr::mul(Expr::sym("M"), Expr::sym("N")),
                            Expr::Const(8192),
                        ),
                        Expr::Const(1024),
                    ),
                },
            },
        ],
    };

    let mat_perf = ProcPerfModel {
        compute: MlirModuleRef::with_functions(
            "compute/matrix_lane.mlir",
            &["matmul_f32"],
        ),
        func_models: vec![mat_func_perf], // one function: matmul_f32
    };

    Processor::primitive_with_perf("matrix_lane", mat_perf)
        .with_resources(vec![ResourceReq::new(l1().as_resource(), 4)])
}
