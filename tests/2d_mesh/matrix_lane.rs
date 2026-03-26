use mlar_rust::*;

use crate::memory::l1;

/// Matrix lane processor with matmul performance model (M, N, K symbols).
///
/// - Global constraints: M ≥ 32, N ≥ 32, K ≥ 32
/// - Scenario 1: M*N ≥ 8192 → throughput = 1024, latency = 1
/// - Scenario 2: M*N ≤ 8192 → throughput = (M*N / 8192) * 1024, latency = 1
pub fn matrix_lane() -> Architecture {
    let functionality = MlirModule::from_mlir("tests/2d_mesh/processors_mlir/matrix_lane.mlir")
        .expect("tests/2d_mesh/processors_mlir/matrix_lane.mlir should parse");

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
        symbols: vec![
            Sym::new("Batch"),
            Sym::new("M"),
            Sym::new("N"),
            Sym::new("K"),
        ],
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
                        Expr::mul(
                            Expr::mul(Expr::sym("Batch"), Expr::sym("M")),
                            Expr::sym("N"),
                        ),
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
                        Expr::mul(
                            Expr::mul(Expr::sym("Batch"), Expr::sym("M")),
                            Expr::sym("N"),
                        ),
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

    let lane_shape = vec![HardwareProperty::LaneComputeShape(vec![32, 32, 32])];
    let l1_region = l1();

    let perf_models = vec![mat_func_perf, batch_mat_func_perf];

    let mut proc = ComputeProcessor::builder()
        .named("matrix_lane")
        .with_regions(vec![l1_region.clone()], vec![l1_region])
        .from_module(functionality, perf_models)
        .expect("matrix_lane processor should link functionality and perf")
        .into_processor();

    for fp in &mut proc.functions {
        fp.hardware_properties = lane_shape.clone();
    }

    proc.into_elem()
}
