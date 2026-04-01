use mlar_rust::*;

use crate::memory::l1_ref;

fn expr(input: &str) -> Expr {
    Expr::parse(input).expect("2d_mesh expression literal should parse")
}

fn constraint(input: &str) -> ConstraintExpr {
    ConstraintExpr::parse(input).expect("2d_mesh constraint literal should parse")
}

fn matrix_op_prefix(func: &str) -> &str {
    func.rsplit_once('_').map(|(pre, _)| pre).unwrap_or(func)
}

fn matmul_func_perf_model() -> FuncPerfModel {
    FuncPerfModel {
        symbols: Sym::from_names(["M", "N", "K"]),
        constraints: constraint("M >= 32 && N >= 32 && K >= 32"),
        scenarios: vec![
            PerfScenario {
                constraints: constraint("M * N >= 8192 && M == N"),
                time_cost: TimeCost::Simple(SimpleTimeCost {
                    fixed_latency: expr("100"),
                    volume: expr("M * N * K"),
                    throughput: expr("1024"),
                }),
            },
            PerfScenario {
                constraints: constraint("(M * N >= 8192) && (M != N)"),
                time_cost: TimeCost::Simple(SimpleTimeCost {
                    fixed_latency: expr("100"),
                    volume: expr("M * N * K"),
                    throughput: expr("716"), // 1024 * 0.7
                }),
            },
            PerfScenario {
                constraints: constraint("M * N < 8192 && M == N"),
                time_cost: TimeCost::Simple(SimpleTimeCost {
                    fixed_latency: expr("100"),
                    volume: expr("M * N * K"),
                    throughput: expr("(M * N / 8192) * 1024"),
                }),
            },
            PerfScenario {
                constraints: constraint("M * N < 8192 && M != N"),
                time_cost: TimeCost::Simple(SimpleTimeCost {
                    fixed_latency: expr("100"),
                    volume: expr("M * N * K"),
                    throughput: expr("(M * N / 8192) * 716"),
                }),
            },
        ],
    }
}

fn batch_matmul_func_perf_model() -> FuncPerfModel {
    FuncPerfModel {
        symbols: Sym::from_names(["B", "M", "N", "K"]),
        constraints: constraint("B >= 1 && M >= 32 && N >= 32 && K >= 32"),
        scenarios: vec![
            PerfScenario {
                constraints: constraint("(B * B * M * N >= 8192) && (M == N)"),
                time_cost: TimeCost::Simple(SimpleTimeCost {
                    fixed_latency: expr("100"),
                    volume: expr("B * M * N * K"),
                    throughput: expr("1024"),
                }),
            },
            PerfScenario {
                constraints: constraint("(B * B * M * N >= 8192) && (M != N)"),
                time_cost: TimeCost::Simple(SimpleTimeCost {
                    fixed_latency: expr("100"),
                    volume: expr("B * M * N * K"),
                    throughput: expr("716"), // 1024 * 0.7
                }),
            },
            PerfScenario {
                constraints: constraint("(B * B * M * N < 8192) && (M == N)"),
                time_cost: TimeCost::Simple(SimpleTimeCost {
                    fixed_latency: expr("100"),
                    volume: expr("B * M * N * K"),
                    throughput: expr("(B * B * M * N / 8192) * 1024"),
                }),
            },
            PerfScenario {
                constraints: constraint("(B * B * M * N < 8192) && (M != N)"),
                time_cost: TimeCost::Simple(SimpleTimeCost {
                    fixed_latency: expr("100"),
                    volume: expr("B * M * N * K"),
                    throughput: expr("(B * B * M * N / 8192) * 716"),
                }),
            },
        ],
    }
}

fn matrix_func_perf_model(func: &str) -> FuncPerfModel {
    match matrix_op_prefix(func) {
        "matmul" => matmul_func_perf_model(),
        "batch_matmul" => batch_matmul_func_perf_model(),
        "vec_vsum" | "vec_vmax" => FuncPerfModel {
            symbols: Sym::from_names(["P", "R"]),
            constraints: constraint("true"),
            scenarios: vec![PerfScenario {
                constraints: constraint("true"),
                time_cost: TimeCost::Simple(SimpleTimeCost {
                    fixed_latency: expr("1"),
                    volume: expr("P * R"),
                    throughput: expr("128"),
                }),
            }],
        },
        "vec_max1" => FuncPerfModel {
            symbols: Sym::from_names(["L"]),
            constraints: constraint("true"),
            scenarios: vec![PerfScenario {
                constraints: constraint("true"),
                time_cost: TimeCost::Simple(SimpleTimeCost {
                    fixed_latency: expr("1"),
                    volume: expr("L"),
                    throughput: expr("128"),
                }),
            }],
        },
        _ => panic!("unexpected matrix op '{}'", func),
    }
}

fn elementwise_add_perf_model() -> FuncPerfModel {
    FuncPerfModel {
        symbols: Sym::from_names(["M", "N"]),
        constraints: constraint("true"),
        scenarios: vec![PerfScenario {
            constraints: constraint("true"),
            time_cost: TimeCost::Simple(SimpleTimeCost {
                fixed_latency: expr("10"),
                volume: expr("M * N"),
                throughput: expr("43"),
            }),
        }],
    }
}

fn elementwise_mul_perf_model() -> FuncPerfModel {
    FuncPerfModel {
        symbols: Sym::from_names(["M", "N"]),
        constraints: constraint("true"),
        scenarios: vec![PerfScenario {
            constraints: constraint("true"),
            time_cost: TimeCost::Simple(SimpleTimeCost {
                fixed_latency: expr("10"),
                volume: expr("M * N"),
                throughput: expr("15"),
            }),
        }],
    }
}

/// Matrix lane processor with matmul plus vector-reduction performance models.
///
/// - `matmul_*`/`batch_matmul_*` use shape-aware throughput scenarios.
/// - `vec_vsum_*`/`vec_vmax_*`: symbols `P, R`, volume `P * R`, throughput `128`, latency `1`.
/// - `vec_max1_*`: symbol `L`, volume `L`, throughput `128`, latency `1`.
pub fn matrix_lane() -> Architecture {
    let functionality = MlirModule::from_mlir("tests/2d_mesh/processors_mlir/matrix_lane.mlir")
        .expect("tests/2d_mesh/processors_mlir/matrix_lane.mlir should parse");

    let lane_shape = vec![HardwareProperty::LaneComputeShape(vec![32, 32, 32])];
    let l1_region = l1_ref();

    let perf_models: Vec<FuncPerfModel> = functionality
        .functions
        .iter()
        .map(|op| match op.name.as_str() {
            "elementwise_add_f16" => elementwise_add_perf_model(),
            "elementwise_mul_f16" => elementwise_mul_perf_model(),
            _ => matrix_func_perf_model(op.name.as_str()),
        })
        .collect();

    let mut proc = ComputeProcessor::builder()
        .named("matrix_lane")
        .with_regions(vec![(l1_region.clone(), l1_region)])
        .from_module(functionality, perf_models)
        .expect("matrix_lane processor should link functionality and perf")
        .into_processor();

    for fp in &mut proc.functions {
        fp.hardware_properties = lane_shape.clone();
    }

    proc.into_elem()
}
