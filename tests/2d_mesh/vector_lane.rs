use mlar_rust::*;

/// Vector lane processor with per-function performance models.
///
/// Each function in the functionality module has its own `FuncPerfModel`:
/// - All vector kernels in `compute/vector_lane.mlir` take `%L: loom.sym`
///   as the logical vector length and bind vector tensors with `loom.bind`.
/// - `vec_max_f32`, `vec_add_f32`, `vec_sum_f32`, `vec_mul_f32`:
///   throughput = 1024, latency = 1
/// - `vec_exp_f32`: throughput = 128, latency = 16
/// - `vec_div_f32`: throughput = 256, latency = 8
pub fn vector_lane() -> Processors {
    let functionality = Module::from_mlir("tests/2d_mesh/compute/vector_lane.mlir")
        .expect("tests/2d_mesh/compute/vector_lane.mlir should parse");

    let perf_for = |func: &str| -> FuncPerfModel {
        let (fixed_latency, throughput) = match func {
            "vec_max_f32" | "vec_sum_f32" | "vec_add_f32" | "vec_mul_f32" => (1, 1024),
            "vec_exp_f32" => (16, 128),
            "vec_div_f32" => (8, 256),
            other => panic!("unexpected vector op '{}'", other),
        };

        FuncPerfModel {
            symbols: vec![Sym::new("L")],
            constraints: ConstraintExpr::True,
            scenarios: vec![PerfScenario {
                constraints: ConstraintExpr::True,
                time_cost: TimeCost::Simple(SimpleTimeCost {
                    fixed_latency: Expr::Const(fixed_latency),
                    volume: Expr::sym("L"),
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
