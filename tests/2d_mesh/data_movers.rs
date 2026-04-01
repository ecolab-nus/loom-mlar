use mlar_rust::*;

use crate::memory::{dram_ref, l1_ref};
use crate::mesh::{h_link_resource, v_link_resource};

fn expr(input: &str) -> Expr {
    Expr::parse(input).expect("2d_mesh expression literal should parse")
}

fn constraint(input: &str) -> ConstraintExpr {
    ConstraintExpr::parse(input).expect("2d_mesh constraint literal should parse")
}

/// Unicast and full-broadcast DRAM <-> L1 transfers.
///
/// Uses both horizontal and vertical links — contends with every other mover.
/// Functions: dram_to_l1_f16, dram_to_l1_1d_bcst_f16, l1_to_dram_f16.
pub fn dram_l1_mover() -> DataMover {
    let functionality = MlirModule::from_mlir("tests/2d_mesh/processors_mlir/dram_l1_mover.mlir")
        .expect("dram_l1_mover.mlir should parse");

    let unicast_perf_model = FuncPerfModel {
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

    let bcst_perf_model = FuncPerfModel {
        symbols: Sym::from_names(["M", "N"]),
        constraints: constraint("true"),
        scenarios: vec![PerfScenario {
            constraints: constraint("true"),
            time_cost: TimeCost::Simple(SimpleTimeCost {
                fixed_latency: expr("344"),
                volume: expr("M * N * 2"),
                throughput: expr("28"),
            }),
        }],
    };

    let perf_models: Vec<FuncPerfModel> = vec![
        unicast_perf_model.clone(),
        bcst_perf_model,
        unicast_perf_model,
    ]
    .into_iter()
    .map(|pm| {
        pm.validate().expect("perf model should validate");
        pm
    })
    .collect();

    let dram = dram_ref();
    let l1 = l1_ref();
    let mover = DataMover::builder()
        .named("dram_l1_mover")
        .with_regions(vec![(dram.clone(), l1.clone()), (l1, dram)])
        .from_module(functionality, perf_models)
        .expect("dram_l1_mover data mover should link functionality and perf");

    mover
        .into_processor()
        .with_resources(vec![h_link_resource(), v_link_resource()])
        .into()
}

/// Vertical broadcast from DRAM to L1 — uses only vertical links.
///
/// Can execute in parallel with horizontal broadcasts.
pub fn dram_l1_bcst_v_mover() -> DataMover {
    let functionality = MlirModule::from_mlir("tests/2d_mesh/processors_mlir/dram_l1_bcst_v.mlir")
        .expect("dram_l1_bcst_v.mlir should parse");

    let perf_model = FuncPerfModel {
        symbols: Sym::from_names(["M", "N"]),
        constraints: constraint("true"),
        scenarios: vec![PerfScenario {
            constraints: constraint("true"),
            time_cost: TimeCost::Simple(SimpleTimeCost {
                fixed_latency: expr("586"),
                volume: expr("M * N * 2"),
                throughput: expr("18"),
            }),
        }],
    };
    perf_model.validate().expect("perf model should validate");

    let dram = dram_ref();
    let l1 = l1_ref();
    let mover = DataMover::builder()
        .named("dram_l1_bcst_v")
        .with_regions(vec![(dram.clone(), l1.clone()), (l1, dram)])
        .from_module(functionality, vec![perf_model])
        .expect("dram_l1_bcst_v data mover should link functionality and perf");

    mover
        .into_processor()
        .with_resources(vec![v_link_resource()])
        .into()
}

/// Horizontal broadcast from DRAM to L1 — uses only horizontal links.
///
/// Can execute in parallel with vertical broadcasts.
pub fn dram_l1_bcst_h_mover() -> DataMover {
    let functionality = MlirModule::from_mlir("tests/2d_mesh/processors_mlir/dram_l1_bcst_h.mlir")
        .expect("dram_l1_bcst_h.mlir should parse");

    let perf_model = FuncPerfModel {
        symbols: Sym::from_names(["M", "N"]),
        constraints: constraint("true"),
        scenarios: vec![PerfScenario {
            constraints: constraint("true"),
            time_cost: TimeCost::Simple(SimpleTimeCost {
                fixed_latency: expr("586"),
                volume: expr("M * N * 2"),
                throughput: expr("18"),
            }),
        }],
    };
    perf_model.validate().expect("perf model should validate");

    let dram = dram_ref();
    let l1 = l1_ref();
    let mover = DataMover::builder()
        .named("dram_l1_bcst_h")
        .with_regions(vec![(dram.clone(), l1.clone()), (l1, dram)])
        .from_module(functionality, vec![perf_model])
        .expect("dram_l1_bcst_h data mover should link functionality and perf");

    mover
        .into_processor()
        .with_resources(vec![h_link_resource()])
        .into()
}
