use mlar_rust::*;

use crate::memory::l1;

fn expr(input: &str) -> Expr {
    Expr::parse(input).expect("2d_mesh expression literal should parse")
}

fn constraint(input: &str) -> ConstraintExpr {
    ConstraintExpr::parse(input).expect("2d_mesh constraint literal should parse")
}

fn vector_op_prefix(func: &str) -> &str {
    func.rsplit_once('_').map(|(pre, _)| pre).unwrap_or(func)
}

fn vector_op_latency_throughput(func: &str, op_prefix: &str) -> (&'static str, &'static str) {
    match op_prefix {
        "vec_max" | "vec_sum" | "vec_add" | "vec_mul" => ("1", "1024"),
        "vec_exp" => ("1", "7"),
        "vec_div" => ("1", "6"),
        "vec_sub" => ("1", "4"),
        "vec_powf" => ("1", "7"),
        "vec_cmpf_ogt" => ("1", "8"),
        "vec_select" => ("1", "8"),
        _ => panic!("unexpected vector op '{}'", func),
    }
}

fn vector_op_symbols_and_volume(op_prefix: &str) -> (Vec<Sym>, &'static str) {
    let _ = op_prefix;
    (vec![Sym::new("L")], "L")
}

fn vector_func_perf_model(func: &str) -> FuncPerfModel {
    let op_prefix = vector_op_prefix(func);
    let (fixed_latency, throughput) = vector_op_latency_throughput(func, op_prefix);
    let (symbols, volume) = vector_op_symbols_and_volume(op_prefix);

    FuncPerfModel {
        symbols,
        constraints: constraint("true"),
        scenarios: vec![PerfScenario {
            constraints: constraint("true"),
            time_cost: TimeCost::Simple(SimpleTimeCost {
                fixed_latency: expr(fixed_latency),
                volume: expr(volume),
                throughput: expr(throughput),
            }),
        }],
    }
}

/// Vector lane processor with per-function performance models.
///
/// Each function in the functionality module has its own `FuncPerfModel`:
/// - Most vector kernels in `processors_mlir/vector_lane.mlir` declare
///   `%L = loom.sym @L : index` as the logical vector length.
/// - `vec_max_*`, `vec_sum_*`, `vec_add_*`, `vec_mul_*`:
///   throughput = 1024, latency = 1.
/// - `vec_exp_*` and `vec_powf_*`: throughput = 7, latency = 1.
/// - `vec_div_*`: throughput = 6, latency = 1.
/// - `vec_sub_*`: throughput = 4, latency = 1.
/// - `vec_cmpf_ogt_*` and `vec_select_*`: throughput = 8, latency = 1.
pub fn vector_lane() -> Architecture {
    let functionality = MlirModule::from_mlir("tests/2d_mesh/processors_mlir/vector_lane.mlir")
        .expect("tests/2d_mesh/processors_mlir/vector_lane.mlir should parse");

    let lane_shape = vec![HardwareProperty::LaneComputeShape(vec![32])];
    let l1_region = l1();

    let perf_models: Vec<FuncPerfModel> = functionality
        .functions
        .iter()
        .map(|op| vector_func_perf_model(op.name.as_str()))
        .collect();

    let mut proc = ComputeProcessor::builder()
        .named("vector_lane")
        .with_regions(vec![(l1_region.clone(), l1_region)])
        .from_module(functionality, perf_models)
        .expect("vector_lane processor should link functionality and perf")
        .into_processor();

    for fp in &mut proc.functions {
        fp.hardware_properties = lane_shape.clone();
    }

    proc.into_elem()
}
