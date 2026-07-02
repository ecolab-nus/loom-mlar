use mlar_rust::*;

use super::{constraint, scenario, simple_perf_model};

pub const MLIR_PATH: &str = "tests/2d_mesh/processors/matrix_lane.mlir";

pub fn functionality() -> MlirModule {
    MlirModule::from_mlir(MLIR_PATH).expect("matrix_lane.mlir should parse")
}

pub fn perf(functionality: &MlirModule) -> Vec<FuncPerfModel> {
    functionality
        .functions
        .iter()
        .map(|op| match op.name.as_str() {
            "elementwise_add_f16" => simple_perf_model("10", "M * N", "43"),
            "elementwise_mul_f16" => simple_perf_model("10", "M * N", "15"),
            other => func_perf_model(other),
        })
        .collect()
}

fn func_perf_model(func: &str) -> FuncPerfModel {
    let op_prefix = func.rsplit_once('_').map(|(pre, _)| pre).unwrap_or(func);
    match op_prefix {
        "matmul" => FuncPerfModel::builder()
            .constraints(constraint("M >= 32 && N >= 32 && K >= 32"))
            .scenarios([
                scenario("M * N >= 8192", "M * N / 2", "2 * M * N * K", "716"),
                scenario(
                    "M * N < 8192 ",
                    "M * N / 2",
                    "2 * M * N * K",
                    "(M * N / 8192) * 716",
                ),
            ])
            .build(),
        "batch_matmul" => FuncPerfModel::builder()
            .constraints(constraint("B >= 1 && M >= 32 && N >= 32 && K >= 32"))
            .scenarios([
                scenario("M * N >= 8192", "M * N / 2", "2 * B * M * N * K", "716"),
                scenario(
                    "M * N < 8192",
                    "M * N / 2",
                    "2 * B * M * N * K",
                    "(M * N / 8192) * 716",
                ),
            ])
            .build(),
        "vec_vsum" | "vec_vmax" => simple_perf_model("1", "P * R", "128"),
        "vec_max1" => simple_perf_model("1", "L", "128"),
        _ => panic!("unexpected matrix op '{}'", func),
    }
}
