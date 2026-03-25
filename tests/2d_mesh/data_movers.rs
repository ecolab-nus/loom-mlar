use mlar_rust::*;

use crate::memory::{dram, l1};

/// Data mover that models bi-directional transfers between DRAM and per-core L1.
pub fn dram_to_l1_mover() -> DataMover {
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
    DataMover::from_module(
        "dram_to_l1_mover",
        functionality,
        perf_models,
        vec![dram.clone(), l1.clone()],
        vec![dram, l1],
    )
    .expect("dram_to_l1 data mover should link functionality and perf")
}

/// Data mover that models L1 -> DRAM writeback (no broadcast).
pub fn l1_to_dram_mover() -> DataMover {
    let functionality = Module::from_mlir("tests/2d_mesh/data_movers/l1_to_dram.mlir")
        .expect("tests/2d_mesh/data_movers/l1_to_dram.mlir should parse");

    let perf_models: Vec<FuncPerfModel> = functionality
        .ops
        .iter()
        .map(|_| FuncPerfModel {
            symbols: vec![Sym::new("M"), Sym::new("N")],
            constraints: ConstraintExpr::True,
            scenarios: vec![PerfScenario {
                constraints: ConstraintExpr::True,
                time_cost: TimeCost::Simple(SimpleTimeCost {
                    fixed_latency: Expr::Const(20),
                    volume: Expr::mul(Expr::sym("M"), Expr::sym("N")),
                    throughput: Expr::Const(2048),
                }),
            }],
        })
        .collect();

    let l1 = l1();
    let dram = dram();
    DataMover::from_module(
        "l1_to_dram_mover",
        functionality,
        perf_models,
        vec![l1],
        vec![dram],
    )
    .expect("l1_to_dram data mover should link functionality and perf")
}
