use mlar_rust::*;

use crate::memory::{dram, l1};

/// Data mover that models bi-directional transfers between DRAM and per-core L1.
pub fn dram_l1_mover() -> DataMover {
    let functionality = Module::from_mlir("tests/2d_mesh/data_movers/dram_to_l1.mlir")
        .expect("tests/2d_mesh/data_movers/dram_to_l1.mlir should parse");

    let perf_models: Vec<FuncPerfModel> = functionality
        .ops
        .iter()
        .map(|op| {
            let throughput = if op.name.contains("bcst") {
                Expr::Const(8192)
            } else {
                Expr::Const(2048)
            };
            FuncPerfModel {
                symbols: vec![Sym::new("M"), Sym::new("N")],
                constraints: ConstraintExpr::True,
                scenarios: vec![PerfScenario {
                    constraints: ConstraintExpr::True,
                    time_cost: TimeCost::Simple(SimpleTimeCost {
                        fixed_latency: Expr::Const(20),
                        volume: Expr::mul(Expr::sym("M"), Expr::sym("N")),
                        throughput,
                    }),
                }],
            }
        })
        .collect();

    let dram = dram();
    let l1 = l1();
    DataMover::builder()
        .named("dram_l1_mover")
        .with_regions(vec![dram.clone(), l1.clone()], vec![l1, dram])
        .from_module(functionality, perf_models)
        .expect("dram_l1 data mover should link functionality and perf")
}
