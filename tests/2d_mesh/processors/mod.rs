pub mod dram_l1_noc0;
pub mod l1_dram_noc1;
pub mod l1_l1_noc0;
pub mod matrix_lane;
pub mod vector_lane;

use mlar_rust::*;

fn expr(input: &str) -> Expr {
    Expr::parse(input).expect("2d_mesh expression literal should parse")
}

fn constraint(input: &str) -> ConstraintExpr {
    ConstraintExpr::parse(input).expect("2d_mesh constraint literal should parse")
}

fn simple_cost(fixed_latency: &str, volume: &str, throughput: &str) -> SimpleTimeCost {
    SimpleTimeCost::new(expr(fixed_latency), expr(volume), expr(throughput))
}

fn scenario(
    constraints: &str,
    fixed_latency: &str,
    volume: &str,
    throughput: &str,
) -> PerfScenario {
    PerfScenario::with_constraints(
        constraint(constraints),
        simple_cost(fixed_latency, volume, throughput),
    )
}

fn simple_perf_model(fixed_latency: &str, volume: &str, throughput: &str) -> FuncPerfModel {
    FuncPerfModel::builder()
        .simple_time_cost(expr(fixed_latency), expr(volume), expr(throughput))
        .build()
}
