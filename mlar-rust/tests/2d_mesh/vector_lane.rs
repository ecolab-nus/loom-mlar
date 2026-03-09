use mlar_rust::*;

use crate::memory::l1;

/// Vector lane processor with per-function performance models.
///
/// Each function in the MLIR module has its own `FuncPerfModel`:
/// - All vector kernels in `compute/vector_lane.mlir` take `%L: loom.sym`
///   as the logical vector length and bind vector tensors with `loom.bind`.
/// - `vec_max_f32`, `vec_add_f32`, `vec_sum_f32`, `vec_mul_f32`:
///   throughput = 1024, latency = 1
/// - `vec_exp_f32`: throughput = 128, latency = 16
/// - `vec_div_f32`: throughput = 256, latency = 8
pub fn vector_lane() -> Processors {
    // Common model for fast ops: vec_max, vec_add, vec_sum, vec_mul.
    // These tests use constant costs (independent of L).
    let fast_op = FuncPerfModel {
        symbols: vec![],
        constraints: ConstraintExpr::True,
        scenarios: vec![PerfScenario {
            constraints: ConstraintExpr::True,
            time_cost: TimeCostExpr {
                fixed_latency: Expr::Const(1),
                throughput: Expr::Const(1024),
            },
        }],
    };

    // vec_exp: throughput = 128, latency = 16
    let exp_op = FuncPerfModel {
        symbols: vec![],
        constraints: ConstraintExpr::True,
        scenarios: vec![PerfScenario {
            constraints: ConstraintExpr::True,
            time_cost: TimeCostExpr {
                fixed_latency: Expr::Const(16),
                throughput: Expr::Const(128),
            },
        }],
    };

    // vec_div: throughput = 256, latency = 8
    let div_op = FuncPerfModel {
        symbols: vec![],
        constraints: ConstraintExpr::True,
        scenarios: vec![PerfScenario {
            constraints: ConstraintExpr::True,
            time_cost: TimeCostExpr {
                fixed_latency: Expr::Const(8),
                throughput: Expr::Const(256),
            },
        }],
    };

    // Function order matches MlirModuleRef:
    // vec_max_f32, vec_exp_f32, vec_sum_f32, vec_add_f32, vec_mul_f32, vec_div_f32
    let vec_perf = ProcPerfModel {
        compute: MlirModuleRef::from_mlir("tests/2d_mesh/compute/vector_lane.mlir")
            .expect("tests/2d_mesh/compute/vector_lane.mlir should parse"),
        func_models: vec![
            fast_op.clone(), // vec_max_f32
            exp_op,          // vec_exp_f32
            fast_op.clone(), // vec_sum_f32
            fast_op.clone(), // vec_add_f32
            fast_op,         // vec_mul_f32
            div_op,          // vec_div_f32
        ],
    };

    Processor::with_perf("vector_lane", vec_perf)
        .into_elem()
        .with_resources(vec![ResourceReq::new(l1().as_resource(), 2)])
}
