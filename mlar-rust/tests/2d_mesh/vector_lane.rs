use mlar_rust::*;

use crate::memory::l1;

/// Vector lane processor with element-wise performance model (N symbol).
///
/// - Valid when N is divisible by 32
/// - Cost = 2 cycles fixed + N/32 throughput
pub fn vector_lane() -> Processor {
    let vec_perf = PerfModel {
        symbols: vec![Symbol::new("N")],
        constraints: ConstraintExpr::True,
        scenarios: vec![PerfScenario {
            constraints: ConstraintExpr::Divisible {
                x: Expr::sym("N"),
                by: Expr::Const(32),
            },
            time_cost: TimeCostExpr {
                fixed_latency: Expr::Const(2),
                throughput: Expr::div(Expr::sym("N"), Expr::Const(32)),
            },
        }],
    };

    let vec_compute = MlirModuleRef::with_functions(
        "compute/vector_lane.mlir",
        &[
            "vec_max_f32", "vec_exp_f32", "vec_sum_f32",
            "vec_add_f32", "vec_mul_f32", "vec_div_f32",
        ],
    );

    Processor::primitive_with_perf_and_compute("vector_lane", vec_perf, vec_compute)
        .with_resources(vec![ResourceReq::new(l1().as_resource(), 2)])
}
