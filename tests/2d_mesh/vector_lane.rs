use mlar_rust::*;

/// Vector lane processor with per-function performance models.
///
/// Each function in the functionality module has its own `FuncPerfModel`:
/// - Most vector kernels in `compute/vector_lane.mlir` declare
///   `%L = loom.sym @L : index` as the logical vector length.
/// - `vec_vsum_*` declares `%P = loom.sym @P` and `%R = loom.sym @R`.
/// - `vec_max_*`, `vec_add_*`, `vec_sum_*`, `vec_mul_*`:
///   throughput = 1024, latency = 1
/// - `vec_exp_*`: throughput = 128, latency = 16
/// - `vec_div_*`: throughput = 256, latency = 8
pub fn vector_lane() -> Architecture {
    let functionality = Module::from_mlir("tests/2d_mesh/compute/vector_lane.mlir")
        .expect("tests/2d_mesh/compute/vector_lane.mlir should parse");

    let perf_for = |func: &str| -> FuncPerfModel {
        let op_prefix = func.rsplit_once('_').map(|(pre, _)| pre).unwrap_or(func);
        let (fixed_latency, throughput) = match op_prefix {
            "vec_max" | "vec_sum" | "vec_add" | "vec_mul" => (1, 1024),
            "vec_exp" => (16, 128),
            "vec_div" => (8, 256),
            "vec_cmpf_ogt" => (1, 1024),
            "vec_select" => (1, 1024),
            "vec_max1" => (1, 1024),
            "vec_vsum" => (1, 1024),
            _ => panic!("unexpected vector op '{}'", func),
        };

        let (symbols, volume) = if op_prefix == "vec_vsum" {
            (
                vec![Sym::new("P"), Sym::new("R")],
                Expr::mul(Expr::sym("P"), Expr::sym("R")),
            )
        } else {
            (vec![Sym::new("L")], Expr::sym("L"))
        };

        FuncPerfModel {
            symbols,
            constraints: ConstraintExpr::True,
            scenarios: vec![PerfScenario {
                constraints: ConstraintExpr::True,
                time_cost: TimeCost::Simple(SimpleTimeCost {
                    fixed_latency: Expr::Const(fixed_latency),
                    volume,
                    throughput: Expr::Const(throughput),
                }),
            }],
        }
    };

    let perf_models: Vec<FuncPerfModel> = functionality
        .ops
        .iter()
        .map(|op| perf_for(op.name.as_str()))
        .collect();

    Processor::from_module("vector_lane", functionality, perf_models)
        .expect("vector_lane processor should link functionality and perf")
        .into_elem()
}
