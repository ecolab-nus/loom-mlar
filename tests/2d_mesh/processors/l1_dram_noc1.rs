use mlar_rust::*;

use super::simple_perf_model;

pub const MLIR_PATH: &str = "tests/2d_mesh/processors/l1_dram_noc1.mlir";

pub fn functionality() -> MlirModule {
    MlirModule::from_mlir(MLIR_PATH).expect("l1_dram_noc1.mlir should parse")
}

pub fn perf(functionality: &MlirModule) -> Vec<FuncPerfModel> {
    functionality
        .functions
        .iter()
        .map(|op| match op.name.as_str() {
            "l1_to_dram_f16" => {
                let pm = simple_perf_model("454", "M * N * 2", "150");
                pm.validate().expect("unicast perf model should validate");
                pm
            }
            other => panic!("unexpected l1_dram_noc1 op '{}'", other),
        })
        .collect()
}
