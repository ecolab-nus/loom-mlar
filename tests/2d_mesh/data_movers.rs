use mlar_rust::*;

/// Data mover that models copying a 2D tile from DRAM to per-core L1.
pub fn dram_to_l1_mover() -> DataMover {
    let functionality = Module::from_mlir("tests/2d_mesh/data_movers/dram_to_l1.mlir")
        .expect("tests/2d_mesh/data_movers/dram_to_l1.mlir should parse");

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

    DataMover::from_module("dram_to_l1_mover", functionality, perf_models)
        .expect("dram_to_l1 data mover should link functionality and perf")
}
