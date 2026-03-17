use std::fs;
use std::path::Path;
use std::process::Command;

use mlar_rust::*;

use crate::scale::scaled_mesh_torus;

const VEC_LANE_MLIR: &str = "tests/2d_mesh/compute/vector_lane.mlir";

/// Look up the full function name for a given operation prefix (e.g. `"vec_add"`)
/// from the vector-lane MLIR module. This makes tests resilient to datatype changes
/// (f16 ↔ f32 ↔ bf16 etc.) in the `.mlir` files.
fn vec_func(prefix: &str) -> String {
    Module::from_mlir(VEC_LANE_MLIR)
        .expect("vector_lane MLIR should parse")
        .ops
        .into_iter()
        .find(|op| op.name.starts_with(prefix))
        .unwrap_or_else(|| panic!("no function matching '{prefix}_*' in {VEC_LANE_MLIR}"))
        .name
}

#[test]
fn test_2d_mesh_torus_perf_models() {
    let mesh = scaled_mesh_torus();
    assert_eq!(mesh.total_processing_elements(), Some(128));

    // === Verify processor functionality + per-function models survive scaling ===
    let core_graph = match &mesh {
        Architecture::Array { elem, .. } => match elem.as_ref() {
            Architecture::Graph(graph) => graph,
            _ => panic!("expected core graph as array element"),
        },
        _ => panic!("expected scaled mesh to be an Array"),
    };
    let proc_nodes: Vec<&Architecture> = core_graph
        .nodes
        .iter()
        .filter_map(|node| match &node.component {
            ArchNodeComponent::Architecture(arch) => Some(arch),
            _ => None,
        })
        .collect();
    assert_eq!(proc_nodes.len(), 2);
    for proc in proc_nodes {
        match proc {
            Processors::Unit(p) => {
                assert!(
                    p.validate().is_ok(),
                    "processor {:?} should validate after scaling",
                    p.name
                );
                assert!(
                    p.functionality
                        .source
                        .as_ref()
                        .is_some_and(|s| s.path.ends_with(".mlir")),
                    "functionality source for {:?} should point to MLIR",
                    p.name
                );
            }
            _ => panic!("expected Unit processor nodes"),
        }
    }

    // === Verify matrix-lane functionality extracted from MLIR ===
    let mat_module = mesh
        .get_processor("matrix_lane")
        .expect("matrix_lane should exist")
        .functionality()
        .expect("matrix_lane should have functionality");
    assert_eq!(
        mat_module.source.as_ref().map(|s| s.path.as_str()),
        Some("tests/2d_mesh/compute/matrix_lane.mlir")
    );
    assert_eq!(mat_module.name.as_deref(), Some("matrix_lane"));
    assert_eq!(mat_module.ops.len(), 1);
    assert!(mat_module.ops[0].name.starts_with("matmul_"));
    let matmul_details = mat_module.ops[0]
        .mlir_details
        .as_ref()
        .expect("matmul_* should include MLIR details");
    assert_eq!(
        matmul_details
            .tensor_symbol_bindings
            .iter()
            .filter(|binding| binding.tensor != "C")
            .cloned()
            .collect::<Vec<_>>(),
        vec![
            MlirTensorSymbolBinding {
                tensor: "A".into(),
                symbols: vec![Sym::new("M"), Sym::new("K")],
            },
            MlirTensorSymbolBinding {
                tensor: "B".into(),
                symbols: vec![Sym::new("K"), Sym::new("N")],
            },
        ]
    );
    assert_eq!(matmul_details.output_tensors, vec!["C".to_string()]);
    assert_eq!(
        matmul_details
            .tensor_symbol_bindings
            .iter()
            .find(|binding| binding.tensor == "C")
            .expect("C binding should exist")
            .symbols,
        vec![Sym::new("M"), Sym::new("N")]
    );

    // === Verify vector-lane functionality extracted from MLIR ===
    let vec_module = mesh
        .get_processor("vector_lane")
        .expect("vector_lane should exist")
        .functionality()
        .expect("vector_lane should have functionality");
    assert_eq!(
        vec_module.source.as_ref().map(|s| s.path.as_str()),
        Some("tests/2d_mesh/compute/vector_lane.mlir")
    );
    assert_eq!(vec_module.name.as_deref(), Some("vector_lane"));
    assert_eq!(vec_module.ops.len(), 6);
    let op_names: Vec<&str> = vec_module.ops.iter().map(|op| op.name.as_str()).collect();
    for prefix in ["vec_max_", "vec_exp_", "vec_sum_", "vec_add_", "vec_mul_", "vec_div_"] {
        assert!(
            op_names.iter().any(|n| n.starts_with(prefix)),
            "expected a function starting with '{}' in vector_lane ops",
            prefix
        );
    }

    // === Verify per-function perf bindings for vector lane ===
    let vec_proc = mesh.get_processor("vector_lane").expect("vector_lane");
    match vec_proc {
        Processors::Unit(p) => {
            assert_eq!(p.functions.len(), 6);

            let max_name = vec_func("vec_max");
            let fast = p.get_function(&max_name).expect("vec_max_* binding");
            let fast_cost = fast.perf.scenarios[0].time_cost.as_simple().expect("Simple");
            assert_eq!(fast_cost.throughput.eval_const(), Some(1024));
            assert_eq!(fast_cost.fixed_latency.eval_const(), Some(1));

            let exp_name = vec_func("vec_exp");
            let exp = p.get_function(&exp_name).expect("vec_exp_* binding");
            let exp_cost = exp.perf.scenarios[0].time_cost.as_simple().expect("Simple");
            assert_eq!(exp_cost.throughput.eval_const(), Some(128));
            assert_eq!(exp_cost.fixed_latency.eval_const(), Some(16));

            let div_name = vec_func("vec_div");
            let div = p.get_function(&div_name).expect("vec_div_* binding");
            let div_cost = div.perf.scenarios[0].time_cost.as_simple().expect("Simple");
            assert_eq!(div_cost.throughput.eval_const(), Some(256));
            assert_eq!(div_cost.fixed_latency.eval_const(), Some(8));
        }
        _ => panic!("expected Unit"),
    }

}

#[test]
fn test_2d_mesh_torus() {
    let mesh = scaled_mesh_torus();

    // === Verify topology ===
    assert_eq!(mesh.name(), Some("2d_mesh_torus"));
    let (dims, connectivity, elem) = match &mesh {
        Architecture::Array {
            dims,
            connectivity,
            elem,
            ..
        } => (dims, connectivity, elem),
        _ => panic!("scaled mesh should be Array"),
    };
    assert_eq!(connectivity.len(), 2); // only torus scale-out networks
    match elem.as_ref() {
        Architecture::Graph(graph) => {
            assert!(
                graph.nodes.iter().any(|n| n.name == "core_router"),
                "scaled mesh should retain router node"
            );
            assert_eq!(
                graph
                    .nodes
                    .iter()
                    .filter(|n| matches!(n.component, ArchNodeComponent::MemoryRegion(_)))
                    .count(),
                1
            );
            assert_eq!(
                graph
                    .nodes
                    .iter()
                    .filter(|n| matches!(n.component, ArchNodeComponent::Architecture(_)))
                    .count(),
                2
            );
        }
        _ => panic!("array element should be graph"),
    }

    assert_eq!(mesh.total_processing_elements(), Some(128));
    assert_eq!(
        dims.iter().map(|d| d.name.0.as_str()).collect::<Vec<_>>(),
        vec!["x", "y"]
    );

    // === Verify torus links ===
    let torus_y_link = &connectivity[0];
    assert_eq!(torus_y_link.name, "L1_torus_y");
    assert_eq!(torus_y_link.map.apply(&[0, 0]), vec![0, 1]);
    assert_eq!(torus_y_link.map.apply(&[3, 5]), vec![3, 6]);
    assert_eq!(torus_y_link.map.apply(&[3, 7]), vec![3, 0]); // wraps

    let torus_x_link = &connectivity[1];
    assert_eq!(torus_x_link.name, "L1_torus_x");
    assert_eq!(torus_x_link.map.apply(&[0, 0]), vec![1, 0]);
    assert_eq!(torus_x_link.map.apply(&[5, 3]), vec![6, 3]);
    assert_eq!(torus_x_link.map.apply(&[7, 3]), vec![0, 3]); // wraps

    // === JSON export sanity for web visualization ===
    let json =
        architecture_to_graph_json_string(&mesh).expect("graph JSON serialization should succeed");
    assert!(json.contains("\"schema_version\":\"mlar.arch-graph.v1\""));
}

#[test]
fn test_export_2d_mesh_torus_graph_json() {
    let mesh = scaled_mesh_torus();
    let json = architecture_to_graph_json_string_pretty(&mesh)
        .expect("graph JSON serialization should succeed");

    let value: serde_json::Value =
        serde_json::from_str(&json).expect("serialized JSON should be valid");
    assert_eq!(value["schema_version"], "mlar.arch-graph.v1");
    assert_eq!(value["architecture"]["name"], "2d_mesh_torus");
    assert!(value["nodes"].as_array().is_some_and(|v| !v.is_empty()));
    assert!(value["edges"].as_array().is_some_and(|v| !v.is_empty()));

    let out_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/2d_mesh/2d_mesh_torus.json");
    fs::write(out_path, &json).expect("Failed to write JSON file");
}

/// Evaluate a sequential schedule of different vector-lane instructions
/// against the single-core architecture, with the per-func `sym_map` mapping
/// the MLIR symbol `L` to the expression `BM * BN`.
///
/// Schedule: vec_add → vec_exp → vec_mul → vec_div
///
/// Before substitution the per-function costs are:
///   vec_add: 1 + L/1024
///   vec_exp: 16 + L/128
///   vec_mul: 1 + L/1024
///   vec_div: 8 + L/256
///
/// After L → BM*BN the expressions use `BM` and `BN` instead.
#[test]
fn test_evaluate_vector_lane_sequential_schedule() {
    let core = crate::core_arch::single_core();

    let l_sym = vec![Sym::new("L")];
    let sym_map = {
        let mut m = SymbolicMapping::new();
        m.insert(
            Sym::new("L"),
            Expr::mul(Expr::sym("BM"), Expr::sym("BN")),
        );
        Some(m)
    };

    let schedule = Schedule::Sequential {
        schedules: vec![
            Schedule::Func {
                func: {
                    let mut f = MlirFunc::with_symbols(&vec_func("vec_add"), l_sym.clone());
                    f.sym_map = sym_map.clone();
                    f
                },
                processor: None,
                scenarios: None,
            },
            Schedule::Func {
                func: {
                    let mut f = MlirFunc::with_symbols(&vec_func("vec_exp"), l_sym.clone());
                    f.sym_map = sym_map.clone();
                    f
                },
                processor: None,
                scenarios: None,
            },
            Schedule::Func {
                func: {
                    let mut f = MlirFunc::with_symbols(&vec_func("vec_mul"), l_sym.clone());
                    f.sym_map = sym_map.clone();
                    f
                },
                processor: None,
                scenarios: None,
            },
            Schedule::Func {
                func: {
                    let mut f = MlirFunc::with_symbols(&vec_func("vec_div"), l_sym.clone());
                    f.sym_map = sym_map.clone();
                    f
                },
                processor: None,
                scenarios: None,
            },
        ],
        mlir_ref: None,
        processor: None,
        scenarios: None,
    };

    let result = evaluate(&schedule, &core).expect("sequential vector schedule should evaluate");

    // Extract per-func scenarios from the result schedule
    let func_scenarios: Vec<&Vec<PerfScenario>> = match &result {
        Schedule::Sequential { schedules, .. } => schedules
            .iter()
            .map(|s| match s {
                Schedule::Func { scenarios: Some(sc), .. } => sc,
                _ => panic!("expected Func with filled scenarios"),
            })
            .collect(),
        _ => panic!("expected Sequential"),
    };

    // Each function has 1 scenario
    for sc in &func_scenarios {
        assert_eq!(sc.len(), 1);
    }

    // vec_add: 1 + L/1024, after L → BM*BN
    let add_expr = func_scenarios[0][0].time_cost.to_expr();
    let free = add_expr.free_symbols();
    assert!(
        !free.contains(&Sym::new("L")),
        "L should have been substituted away, but symbols are: {:?}",
        free
    );
    assert!(
        free.contains(&Sym::new("BM")) && free.contains(&Sym::new("BN")),
        "BM and BN should appear after substitution, but symbols are: {:?}",
        free
    );

    // BM=32, BN=32 → L=1024 → vec_add: 1 + 1024/1024 = 2
    let at_32x32 = add_expr.substitute(&[
        (Sym::new("BM"), Expr::Const(32)),
        (Sym::new("BN"), Expr::Const(32)),
    ]);
    assert_eq!(at_32x32.eval_const(), Some(2));

    // All individual constraints are True
    assert_eq!(func_scenarios[0][0].constraints.eval_const(), Some(true));

    // sym_map is preserved on each func in the output schedule
    match &result {
        Schedule::Sequential { schedules, .. } => {
            for s in schedules {
                match s {
                    Schedule::Func { func, .. } => {
                        let sm = func.sym_map.as_ref().expect("sym_map should be present");
                        assert_eq!(sm.entries.len(), 1);
                        assert_eq!(sm.entries[0].0, Sym::new("L"));
                    }
                    _ => panic!("expected Func"),
                }
            }
        }
        _ => panic!("expected Sequential"),
    }

    // Schedule round-trips through JSON.
    let json = serde_json::to_string(&result).expect("Schedule should serialize");
    println!("{}", json);
    let decoded: Schedule =
        serde_json::from_str(&json).expect("Schedule should deserialize");
    let decoded_json = serde_json::to_string(&decoded).expect("decoded Schedule should serialize");
    assert_eq!(json, decoded_json);
}

/// Evaluate a schedule with per-func `sym_map` that maps the MLIR symbol `L`
/// to `BM * BN`.
///
/// The vector-lane functions all reference `L` in their cost expressions.
/// After `evaluate`, every occurrence of `L` should be replaced by `BM * BN`,
/// and the mapping should be preserved on each `MlirFunc` in the returned
/// `Schedule`.
///
/// Schedule: vec_add → vec_mul
///   vec_add: fixed=1, volume=L, throughput=1024  → 1 + L/1024
///   vec_mul: fixed=1, volume=L, throughput=1024  → 1 + L/1024
///
/// After substitution L → BM*BN:
///   per-func cost = 1 + (BM*BN)/1024
#[test]
fn test_evaluate_with_sym_map() {
    let core = crate::core_arch::single_core();

    let l_sym = vec![Sym::new("L")];
    let sym_map = {
        let mut m = SymbolicMapping::new();
        m.insert(
            Sym::new("L"),
            Expr::mul(Expr::sym("BM"), Expr::sym("BN")),
        );
        Some(m)
    };

    let schedule = Schedule::Sequential {
        schedules: vec![
            Schedule::Func {
                func: {
                    let mut f = MlirFunc::with_symbols(&vec_func("vec_add"), l_sym.clone());
                    f.sym_map = sym_map.clone();
                    f
                },
                processor: None,
                scenarios: None,
            },
            Schedule::Func {
                func: {
                    let mut f = MlirFunc::with_symbols(&vec_func("vec_mul"), l_sym.clone());
                    f.sym_map = sym_map.clone();
                    f
                },
                processor: None,
                scenarios: None,
            },
        ],
        mlir_ref: None,
        processor: None,
        scenarios: None,
    };

    let result = evaluate(&schedule, &core).expect("evaluate should succeed");

    // Extract per-func scenarios from the result
    let func_scenarios: Vec<&Vec<PerfScenario>> = match &result {
        Schedule::Sequential { schedules, .. } => schedules
            .iter()
            .map(|s| match s {
                Schedule::Func { scenarios: Some(sc), .. } => sc,
                _ => panic!("expected Func with filled scenarios"),
            })
            .collect(),
        _ => panic!("expected Sequential"),
    };

    assert_eq!(func_scenarios.len(), 2);
    for sc in &func_scenarios {
        assert_eq!(sc.len(), 1);
    }

    // --- Verify that the sym_map is preserved on each func ---
    match &result {
        Schedule::Sequential { schedules, .. } => {
            for s in schedules {
                match s {
                    Schedule::Func { func, .. } => {
                        let sm = func.sym_map.as_ref().expect("sym_map should be present");
                        assert_eq!(sm.entries.len(), 1);
                        assert_eq!(sm.entries[0].0, Sym::new("L"));
                    }
                    _ => panic!("expected Func"),
                }
            }
        }
        _ => panic!("expected Sequential"),
    }

    // --- Verify symbol substitution: L should no longer appear ---
    let s = &func_scenarios[0][0];
    assert!(
        s.time_cost.as_concrete().is_some(),
        "evaluated scenario should be Concrete"
    );
    let expr = s.time_cost.to_expr();
    let free = expr.free_symbols();
    assert!(
        !free.contains(&Sym::new("L")),
        "L should have been substituted away, but symbols are: {:?}",
        free
    );
    assert!(
        free.contains(&Sym::new("BM")) && free.contains(&Sym::new("BN")),
        "BM and BN should appear after substitution, but symbols are: {:?}",
        free
    );

    // --- Verify concrete evaluation ---
    // With BM=32, BN=32 → L=1024: per func = 1 + 1024/1024 = 2
    let at_32x32 = expr.substitute(&[
        (Sym::new("BM"), Expr::Const(32)),
        (Sym::new("BN"), Expr::Const(32)),
    ]);
    assert_eq!(at_32x32.eval_const(), Some(2));

    // With BM=0, BN=0 → L=0, only fixed latency: 1
    let at_0x0 = expr.substitute(&[
        (Sym::new("BM"), Expr::Const(0)),
        (Sym::new("BN"), Expr::Const(0)),
    ]);
    assert_eq!(at_0x0.eval_const(), Some(1));

    // Constraints remain True (all vector-lane scenarios have True constraints).
    assert_eq!(s.constraints.eval_const(), Some(true));

    // --- JSON round-trip ---
    let json = serde_json::to_string(&result).expect("Schedule should serialize");
    println!("{}", json);
    let decoded: Schedule =
        serde_json::from_str(&json).expect("Schedule should deserialize");
    let decoded_json = serde_json::to_string(&decoded).expect("decoded Schedule should serialize");
    assert_eq!(json, decoded_json);
}

/// Generate a standalone evaluator binary for the single-core architecture,
/// then verify it produces the correct evaluated Schedule when invoked externally.
#[test]
fn test_generate_core_evaluator_binary() {
    let core = crate::core_arch::single_core();

    let output_dir =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/2d_mesh/evaluators");
    let binary = generate_evaluator_binary(&core, "eval_core", &output_dir)
        .expect("binary generation should succeed");

    assert!(binary.exists(), "generated binary should exist at {binary:?}");

    let l_sym = vec![Sym::new("L")];
    let sym_map = {
        let mut m = SymbolicMapping::new();
        m.insert(
            Sym::new("L"),
            Expr::mul(Expr::sym("BM"), Expr::sym("BN")),
        );
        Some(m)
    };

    let schedule = Schedule::Sequential {
        schedules: vec![
            Schedule::Func {
                func: {
                    let mut f = MlirFunc::with_symbols(&vec_func("vec_add"), l_sym.clone());
                    f.sym_map = sym_map.clone();
                    f
                },
                processor: None,
                scenarios: None,
            },
            Schedule::Func {
                func: {
                    let mut f = MlirFunc::with_symbols(&vec_func("vec_mul"), l_sym.clone());
                    f.sym_map = sym_map.clone();
                    f
                },
                processor: None,
                scenarios: None,
            },
        ],
        mlir_ref: None,
        processor: None,
        scenarios: None,
    };
    let input_json = serde_json::to_string(&schedule).expect("input should serialize");

    let output = Command::new(&binary)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            child
                .stdin
                .as_mut()
                .unwrap()
                .write_all(input_json.as_bytes())
                .unwrap();
            child.wait_with_output()
        })
        .expect("binary should execute");

    assert!(
        output.status.success(),
        "binary exited with error: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let result: Schedule =
        serde_json::from_slice(&output.stdout).expect("binary output should be valid JSON");

    // Extract per-func scenarios
    let func_scenarios: Vec<&Vec<PerfScenario>> = match &result {
        Schedule::Sequential { schedules, .. } => schedules
            .iter()
            .map(|s| match s {
                Schedule::Func { scenarios: Some(sc), .. } => sc,
                _ => panic!("expected Func with filled scenarios"),
            })
            .collect(),
        _ => panic!("expected Sequential"),
    };

    assert_eq!(func_scenarios.len(), 2);
    for sc in &func_scenarios {
        assert_eq!(sc.len(), 1);
    }

    let expr = func_scenarios[0][0].time_cost.to_expr();
    let free = expr.free_symbols();
    assert!(!free.contains(&Sym::new("L")));
    assert!(free.contains(&Sym::new("BM")) && free.contains(&Sym::new("BN")));

    // BM=32, BN=32 → L=1024:
    //   vec_add: 1 + 1024/1024 = 2
    let at_32x32 = expr.substitute(&[
        (Sym::new("BM"), Expr::Const(32)),
        (Sym::new("BN"), Expr::Const(32)),
    ]);
    assert_eq!(at_32x32.eval_const(), Some(2));
}
