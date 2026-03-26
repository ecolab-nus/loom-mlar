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

    let perf_models: Vec<FuncPerfModel> = functionality
        .functions
        .iter()
        .map(|op| {
            let throughput = if op.name.contains("bcst") {
                "8192"
            } else {
                "2048"
            };
            FuncPerfModel {
                symbols: vec![Sym::new("M"), Sym::new("N")],
                constraints: constraint("true"),
                scenarios: vec![PerfScenario {
                    constraints: constraint("true"),
                    time_cost: TimeCost::Simple(SimpleTimeCost {
                        fixed_latency: expr("20"),
                        volume: expr("M * N"),
                        throughput: expr(throughput),
                    }),
                }],
            }
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
