use mlar_rust::*;

use crate::memory::l1;

/// Matrix lane processor with matmul performance model (M, N, K symbols).
///
/// - Valid when M, N, K ≥ 128
/// - Cost = 8 cycles fixed + M*N*K/1024 throughput
pub fn matrix_lane() -> Processor {
    let mat_perf = PerfModel {
        symbols: vec![Symbol::new("M"), Symbol::new("N"), Symbol::new("K")],
        constraints: ConstraintExpr::True,
        scenarios: vec![PerfScenario {
            constraints: ConstraintExpr::And(vec![
                ConstraintExpr::Ge(Expr::sym("M"), Expr::Const(128)),
                ConstraintExpr::Ge(Expr::sym("N"), Expr::Const(128)),
                ConstraintExpr::Ge(Expr::sym("K"), Expr::Const(128)),
            ]),
            time_cost: TimeCostExpr {
                fixed_latency: Expr::Const(8),
                throughput_latency: Expr::div(
                    Expr::mul(Expr::mul(Expr::sym("M"), Expr::sym("N")), Expr::sym("K")),
                    Expr::Const(1024),
                ),
            },
        }],
    };

    let mat_compute = MlirModuleRef::with_functions(
        "compute/matrix_lane.mlir",
        &["matmul_f32"],
    );

    Processor::primitive_with_perf_and_compute("matrix_lane", mat_perf, mat_compute)
        .with_resources(vec![ResourceReq::new(l1().as_resource(), 4)])
}
