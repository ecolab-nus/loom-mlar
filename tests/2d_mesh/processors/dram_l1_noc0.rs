use mlar_rust::*;

use super::{expr, simple_perf_model};

pub const MLIR_PATH: &str = "tests/2d_mesh/processors/dram_l1_noc0.mlir";

pub fn functionality() -> MlirModule {
    MlirModule::from_mlir(MLIR_PATH).expect("dram_l1_noc0.mlir should parse")
}

pub fn perf(functionality: &MlirModule) -> Vec<FuncPerfModel> {
    functionality
        .functions
        .iter()
        .map(|op| match op.name.as_str() {
            "dram_to_l1_f16" => {
                let pm = simple_perf_model("454", "M * N * 2 * effective_bandwidth", "150");
                pm.validate().expect("unicast perf model should validate");
                pm
            }
            "dram_to_l1_bcst" => {
                let pm = FuncPerfModel::builder()
                    .symbols(["M", "N", "bcst_x", "bcst_y", "effective_bandwidth"])
                    .simple_time_cost(
                        expr("344"),
                        expr("M * N * 2 * effective_bandwidth"),
                        expr("150"),
                    )
                    .build();
                pm.validate().expect("bcst perf model should validate");
                pm
            }
            other => panic!("unexpected dram_l1_noc0 op '{}'", other),
        })
        .collect()
}
