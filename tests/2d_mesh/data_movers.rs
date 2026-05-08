use mlar_rust::*;

fn expr(input: &str) -> Expr {
    Expr::parse(input).expect("2d_mesh expression literal should parse")
}

fn constraint(input: &str) -> ConstraintExpr {
    ConstraintExpr::parse(input).expect("2d_mesh constraint literal should parse")
}

/// Unicast (broadcast `[1, 1]`) DRAM<->L1 transfer cost.
fn unicast_perf() -> FuncPerfModel {
    let pm = FuncPerfModel {
        symbols: Sym::from_names(["M", "N"]),
        constraints: constraint("true"),
        scenarios: vec![PerfScenario {
            constraints: constraint("true"),
            time_cost: TimeCost::Simple(SimpleTimeCost {
                fixed_latency: expr("454"),
                volume: expr("M * N * 2"),
                throughput: expr("15"),
            }),
        }],
    };
    pm.validate().expect("unicast perf model should validate");
    pm
}

/// Parameterized 2D broadcast `[@BCST_X, @BCST_Y]` cost.
fn bcst_perf() -> FuncPerfModel {
    let pm = FuncPerfModel {
        symbols: Sym::from_names(["M", "N", "BCST_X", "BCST_Y"]),
        constraints: constraint("true"),
        scenarios: vec![PerfScenario {
            constraints: constraint("true"),
            time_cost: TimeCost::Simple(SimpleTimeCost {
                fixed_latency: expr("344 + BCST_X + BCST_Y"),
                volume: expr("M * N * 2"),
                throughput: expr("28/(BCST_X * BCST_Y)"),
            }),
        }],
    };
    pm.validate().expect("bcst perf model should validate");
    pm
}

/// DRAM->L1 transfers carried over NoC0.
///
/// Functions: `dram_to_l1_f16` (unicast), `dram_to_l1_bcst` (parameterized
/// 2D broadcast `[@BCST_X, @BCST_Y]`). NoC0 is read-only — there is no
/// L1->DRAM writeback path.
pub fn dram_l1_noc0(dram: &MemoryRegion, l1: &MemoryRegion) -> DataMover {
    let functionality = MlirModule::from_mlir("tests/2d_mesh/processors_mlir/dram_l1_noc0.mlir")
        .expect("dram_l1_noc0.mlir should parse");

    DataMover::builder()
        .named("dram_l1_noc0")
        .with_regions(vec![(dram.clone(), l1.clone())])
        .from_module(functionality, vec![unicast_perf(), bcst_perf()])
        .expect("dram_l1_noc0 data mover should link functionality and perf")
}

/// L1->L1 gather (`[@GATHER_X, @GATHER_Y]` area) cost — same structure as the
/// parameterized broadcast model but named for the gather symbols.
fn gather_perf() -> FuncPerfModel {
    let pm = FuncPerfModel {
        symbols: Sym::from_names(["M", "N", "B", "GATHER_X", "GATHER_Y"]),
        constraints: constraint("true"),
        scenarios: vec![PerfScenario {
            constraints: constraint("true"),
            time_cost: TimeCost::Simple(SimpleTimeCost {
                fixed_latency: expr("344 + GATHER_X + GATHER_Y"),
                volume: expr("B * M * N * 2"),
                throughput: expr("28/(GATHER_X * GATHER_Y)"),
            }),
        }],
    };
    pm.validate().expect("gather perf model should validate");
    pm
}

/// L1->DRAM writeback and L1->L1 gather transfers carried over NoC1.
///
/// Functions: `l1_to_dram_f16` (writeback), `l1_gather` (gather with
/// parameterized area `[@GATHER_X, @GATHER_Y]`). No DRAM->L1 load or
/// broadcast path.
pub fn dram_l1_noc1(dram: &MemoryRegion, l1: &MemoryRegion) -> DataMover {
    let functionality = MlirModule::from_mlir("tests/2d_mesh/processors_mlir/dram_l1_noc1.mlir")
        .expect("dram_l1_noc1.mlir should parse");

    DataMover::builder()
        .named("dram_l1_noc1")
        .with_regions(vec![(l1.clone(), dram.clone()), (l1.clone(), l1.clone())])
        .from_module(functionality, vec![unicast_perf(), gather_perf()])
        .expect("dram_l1_noc1 data mover should link functionality and perf")
}
