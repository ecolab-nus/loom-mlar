use mlar_rust::*;

use super::expr;

pub const MLIR_PATH: &str = "tests/2d_mesh/processors/l1_l1_noc0.mlir";

pub fn functionality() -> MlirModule {
    MlirModule::from_mlir(MLIR_PATH).expect("l1_l1_noc0.mlir should parse")
}

pub fn perf(functionality: &MlirModule) -> Vec<FuncPerfModel> {
    functionality
        .functions
        .iter()
        .map(|op| match op.name.as_str() {
            "l1_gather" => {
                let pm = FuncPerfModel::builder()
                    .symbols(["B", "M", "N", "gather_x", "gather_y", "effective_bandwidth"])
                    .simple_time_cost(
                        expr("344"),
                        expr("B * M * N * 2 * effective_bandwidth"),
                        expr("28"),
                    )
                    .build();
                pm.validate().expect("gather perf model should validate");
                pm
            }
            other => panic!("unexpected l1_l1_noc0 op '{}'", other),
        })
        .collect()
}
