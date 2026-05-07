use mlar_rust::*;

fn expr(input: &str) -> Expr {
    Expr::parse(input).expect("2d_mesh expression literal should parse")
}

fn constraint(input: &str) -> ConstraintExpr {
    ConstraintExpr::parse(input).expect("2d_mesh constraint literal should parse")
}

/// Performance models for the four DRAM<->L1 transfer kinds carried by a NoC.
///
/// `bcst_2d_perf` describes the X-fixed-or-Y-fixed half-broadcast: one of the
/// broadcast dimensions is pinned to 8 by the MLIR (so the symbol has been
/// substituted away), while the other remains free and is named `free_bcst`.
fn noc_perf_models(free_bcst: &str) -> Vec<FuncPerfModel> {
    let unicast_perf = FuncPerfModel {
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

    let bcst_1d_perf = FuncPerfModel {
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

    let bcst_2d_perf = FuncPerfModel {
        symbols: Sym::from_names(["M", "N", free_bcst]),
        constraints: constraint("true"),
        scenarios: vec![PerfScenario {
            constraints: constraint("true"),
            time_cost: TimeCost::Simple(SimpleTimeCost {
                fixed_latency: expr(&format!("352 + {free_bcst}")),
                volume: expr("M * N * 2"),
                throughput: expr(&format!("28/(8 * {free_bcst})")),
            }),
        }],
    };

    vec![unicast_perf.clone(), bcst_1d_perf, bcst_2d_perf, unicast_perf]
        .into_iter()
        .inspect(|pm| {
            pm.validate().expect("perf model should validate");
        })
        .collect()
}

/// Build a DRAM<->L1 NoC data mover that exposes unicast, full symbolic 2D
/// broadcast, and a half-fixed 2D broadcast.
///
/// `name` selects the MLIR module under `processors_mlir/` (e.g. `dram_l1_noc0`),
/// and `free_bcst` is the broadcast symbol that remains free in the half-fixed
/// 2D broadcast (the other dim is pinned to 8 by the MLIR).
fn dram_l1_noc_mover(
    name: &str,
    free_bcst: &str,
    dram: &MemoryRegion,
    l1: &MemoryRegion,
) -> DataMover {
    let path = format!("tests/2d_mesh/processors_mlir/{name}.mlir");
    let functionality =
        MlirModule::from_mlir(&path).unwrap_or_else(|_| panic!("{name}.mlir should parse"));

    DataMover::builder()
        .named(name)
        .with_regions(vec![(dram.clone(), l1.clone()), (l1.clone(), dram.clone())])
        .from_module(functionality, noc_perf_models(free_bcst))
        .unwrap_or_else(|_| panic!("{name} data mover should link functionality and perf"))
}

/// DRAM<->L1 transfers carried over NoC0.
///
/// Functions: dram_to_l1_f16, dram_to_l1_1d_bcst_f16, dram_to_l1_2d_bcst_f16,
/// l1_to_dram_f16. The half-fixed 2D broadcast pins the X dimension to 8 and
/// leaves `BCST_Y` free.
pub fn dram_l1_noc0(dram: &MemoryRegion, l1: &MemoryRegion) -> DataMover {
    dram_l1_noc_mover("dram_l1_noc0", "BCST_Y", dram, l1)
}

/// DRAM<->L1 transfers carried over NoC1.
///
/// Mirrors `dram_l1_noc0`, but the half-fixed 2D broadcast pins the Y dimension
/// to 8 and leaves `BCST_X` free.
pub fn dram_l1_noc1(dram: &MemoryRegion, l1: &MemoryRegion) -> DataMover {
    dram_l1_noc_mover("dram_l1_noc1", "BCST_X", dram, l1)
}
