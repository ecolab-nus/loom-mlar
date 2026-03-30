use mlar_rust::*;

use crate::memory::{dram, l1};

fn expr(input: &str) -> Expr {
    Expr::parse(input).expect("2d_mesh expression literal should parse")
}

fn constraint(input: &str) -> ConstraintExpr {
    ConstraintExpr::parse(input).expect("2d_mesh constraint literal should parse")
}

/// Data mover that models bi-directional transfers between DRAM and per-core L1.
pub fn dram_l1_mover() -> DataMover {
    let functionality = MlirModule::from_mlir("tests/2d_mesh/processors_mlir/dram_to_l1.mlir")
        .expect("tests/2d_mesh/processors_mlir/dram_to_l1.mlir should parse");


    let unicast_perf_model = FuncPerfModel {
        symbols: Sym::from_names(["M", "N"]),
        constraints: constraint("true"),
        scenarios: vec![PerfScenario {
            constraints: constraint("true"),
            time_cost: TimeCost::Simple(SimpleTimeCost {
                fixed_latency: expr("454"),
                volume: expr("M * N * 2"), // FP16
                throughput: expr("15"),
            }),
        }],
    };

    let bcst_1d_perf_model = FuncPerfModel {
        symbols: Sym::from_names(["M", "N"]),
        constraints: constraint("true"),
        scenarios: vec![PerfScenario {
            constraints: constraint("true"),
            time_cost: TimeCost::Simple(SimpleTimeCost {
                fixed_latency: expr("344"),
                volume: expr("M * N * 2"), // FP16
                throughput: expr("28"),
            }),
        }],
    };

    let bcst_1d_h_perf_model = FuncPerfModel {
        symbols: Sym::from_names(["M", "N"]),
        constraints: constraint("true"),
        scenarios: vec![PerfScenario {
            constraints: constraint("true"),
            time_cost: TimeCost::Simple(SimpleTimeCost {
                fixed_latency: expr("586"),
                volume: expr("M * N * 2"), // FP16
                throughput: expr("18"),
            }),
        }],
    };

    let bcst_1d_v_perf_model = FuncPerfModel {
        symbols: Sym::from_names(["M", "N"]),
        constraints: constraint("true"),
        scenarios: vec![PerfScenario {
            constraints: constraint("true"),
            time_cost: TimeCost::Simple(SimpleTimeCost {
                fixed_latency: expr("586"),
                volume: expr("M * N * 2"), // FP16
                throughput: expr("18"),
            }),
        }],
    };

    let perf_models: Vec<FuncPerfModel> = vec![unicast_perf_model.clone(), bcst_1d_perf_model, bcst_1d_h_perf_model, bcst_1d_v_perf_model, unicast_perf_model]
        .into_iter()
        .map(|perf_model| {
            perf_model.validate().expect("perf model should validate");
            perf_model
        })
        .collect();

    let dram = dram();
    let l1 = l1();
    DataMover::builder()
        .named("dram_l1_mover")
        .with_regions(vec![(dram.clone(), l1.clone()), (l1, dram)])
        .from_module(functionality, perf_models)
        .expect("dram_l1 data mover should link functionality and perf")
}
