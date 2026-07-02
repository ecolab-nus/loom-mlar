use mlar_rust::*;

use super::simple_perf_model;

pub const MLIR_PATH: &str = "tests/2d_mesh/processors/vector_lane.mlir";

pub fn functionality() -> MlirModule {
    MlirModule::from_mlir(MLIR_PATH).expect("vector_lane.mlir should parse")
}

pub fn perf(functionality: &MlirModule) -> Vec<FuncPerfModel> {
    functionality
        .functions
        .iter()
        .map(|op| func_perf_model(op.name.as_str()))
        .collect()
}

fn func_perf_model(func: &str) -> FuncPerfModel {
    let op_prefix = func.rsplit_once('_').map(|(pre, _)| pre).unwrap_or(func);
    let (fixed_latency, throughput) = match op_prefix {
        "vec_max" | "vec_sum" | "vec_add" | "vec_mul" => ("1", "1024"),
        "vec_exp" | "vec_log" => ("1", "7"),
        "vec_div" => ("1", "6"),
        "vec_sub" => ("1", "4"),
        "vec_powf" => ("1", "7"),
        "vec_cmpf_ogt" => ("1", "8"),
        "vec_select" => ("1", "8"),
        _ => panic!("unexpected vector op '{}'", func),
    };
    simple_perf_model(fixed_latency, "L", throughput)
}
