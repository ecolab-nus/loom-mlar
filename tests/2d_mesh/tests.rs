use std::fs;

use mlar_rust::*;

use crate::scale::scaled_mesh_torus;

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
    assert_eq!(mat_module.ops[0].name, "matmul_f32");
    let matmul_details = mat_module.ops[0]
        .mlir_details
        .as_ref()
        .expect("matmul_f32 should include MLIR details");
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
    assert!(op_names.contains(&"vec_max_f32"));
    assert!(op_names.contains(&"vec_exp_f32"));
    assert!(op_names.contains(&"vec_sum_f32"));
    assert!(op_names.contains(&"vec_add_f32"));
    assert!(op_names.contains(&"vec_mul_f32"));
    assert!(op_names.contains(&"vec_div_f32"));

    // === Verify per-function perf bindings for vector lane ===
    let vec_proc = mesh.get_processor("vector_lane").expect("vector_lane");
    match vec_proc {
        Processors::Unit(p) => {
            assert_eq!(p.functions.len(), 6);

            let fast = p.get_function("vec_max_f32").expect("vec_max_f32 binding");
            let fast_cost = fast.perf.scenarios[0].time_cost.as_simple().expect("Simple");
            assert_eq!(fast_cost.throughput.eval_const(), Some(1024));
            assert_eq!(fast_cost.fixed_latency.eval_const(), Some(1));

            let exp = p.get_function("vec_exp_f32").expect("vec_exp_f32 binding");
            let exp_cost = exp.perf.scenarios[0].time_cost.as_simple().expect("Simple");
            assert_eq!(exp_cost.throughput.eval_const(), Some(128));
            assert_eq!(exp_cost.fixed_latency.eval_const(), Some(16));

            let div = p.get_function("vec_div_f32").expect("vec_div_f32 binding");
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
/// against the single-core architecture, using `ScheduleWithSymMap` to map
/// the MLIR symbol `L` to the expression `BM * BN`.
///
/// Schedule: vec_add_f32 → vec_exp_f32 → vec_mul_f32 → vec_div_f32
///
/// Before substitution the per-function costs are:
///   vec_add_f32: 1 + L/1024
///   vec_exp_f32: 16 + L/128
///   vec_mul_f32: 1 + L/1024
///   vec_div_f32: 8 + L/256
///   total(L)   = 26 + L/1024 + L/128 + L/1024 + L/256
///
/// After L → BM*BN the expressions use `BM` and `BN` instead.
///   total(BM,BN) = 26 + (BM*BN)/1024 + (BM*BN)/128 + (BM*BN)/1024 + (BM*BN)/256
#[test]
fn test_evaluate_vector_lane_sequential_schedule() {
    let core = crate::core_arch::single_core();

    let l_sym = vec![Sym::new("L")];
    let schedule = Schedule::Sequential {
        schedules: vec![
            Schedule::Func {
                func: MlirFunc::with_symbols("vec_add_f32", l_sym.clone()),
                processor: None,
                time: None,
            },
            Schedule::Func {
                func: MlirFunc::with_symbols("vec_exp_f32", l_sym.clone()),
                processor: None,
                time: None,
            },
            Schedule::Func {
                func: MlirFunc::with_symbols("vec_mul_f32", l_sym.clone()),
                processor: None,
                time: None,
            },
            Schedule::Func {
                func: MlirFunc::with_symbols("vec_div_f32", l_sym.clone()),
                processor: None,
                time: None,
            },
        ],
        mlir_ref: None,
        processor: None,
        time: None,
    };

    let mut sym_map = SymbolicMapping::new();
    sym_map.insert(
        Sym::new("L"),
        Expr::mul(Expr::sym("BM"), Expr::sym("BN")),
    );

    let input = ScheduleWithSymMap::new(schedule, sym_map);
    let scenarios =
        evaluate_with_sym_map(&input, &core).expect("sequential vector schedule should evaluate");

    // 1 scenario per function → 1×1×1×1 = 1 combined scenario
    assert_eq!(scenarios.len(), 1);

    let s = &scenarios[0];
    assert!(s.time_cost.as_concrete().is_some(), "evaluated scenario should be Concrete");

    let expr = s.time_cost.to_expr();

    // L should have been replaced — only BM and BN remain.
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

    // BM=0, BN=0 → L=0, only fixed latencies: 1 + 16 + 1 + 8 = 26
    let at_0x0 = expr.substitute(&[
        (Sym::new("BM"), Expr::Const(0)),
        (Sym::new("BN"), Expr::Const(0)),
    ]);
    assert_eq!(at_0x0.eval_const(), Some(1 + 16 + 1 + 8));

    // BM=32, BN=32 → L=1024:
    //   (1 + 1024/1024) + (16 + 1024/128) + (1 + 1024/1024) + (8 + 1024/256)
    //   = 2 + 24 + 2 + 12 = 40
    let at_32x32 = expr.substitute(&[
        (Sym::new("BM"), Expr::Const(32)),
        (Sym::new("BN"), Expr::Const(32)),
    ]);
    assert_eq!(at_32x32.eval_const(), Some(40));

    // All individual constraints are True, so the fused constraint must also
    // evaluate to true.
    assert_eq!(s.constraints.eval_const(), Some(true));

    // --- sym_map is preserved in the output PerfScenarios ---
    let preserved_map = scenarios.sym_map.as_ref().expect("sym_map should be present");
    assert_eq!(preserved_map.entries.len(), 1);
    assert_eq!(preserved_map.entries[0].0, Sym::new("L"));

    // PerfScenarios (including sym_map) round-trips through JSON.
    let json = serde_json::to_string(&scenarios).expect("PerfScenarios should serialize");
    println!("{}", json);
    let decoded: PerfScenarios =
        serde_json::from_str(&json).expect("PerfScenarios should deserialize");
    assert_eq!(decoded.len(), scenarios.len());

    let decoded_map = decoded.sym_map.as_ref().expect("decoded sym_map should be present");
    assert_eq!(decoded_map.entries.len(), 1);
    assert_eq!(decoded_map.entries[0].0, Sym::new("L"));

    let decoded_at_32x32 = decoded[0].time_cost.to_expr().substitute(&[
        (Sym::new("BM"), Expr::Const(32)),
        (Sym::new("BN"), Expr::Const(32)),
    ]);
    assert_eq!(decoded_at_32x32.eval_const(), at_32x32.eval_const());
}

/// Evaluate a `ScheduleWithSymMap` that maps the MLIR symbol `L` to `BM * BN`.
///
/// The vector-lane functions all reference `L` in their cost expressions.
/// After `evaluate_with_sym_map`, every occurrence of `L` should be replaced
/// by `BM * BN`, and the mapping should be preserved in the returned
/// `PerfScenarios`.
///
/// Schedule: vec_add_f32 → vec_mul_f32
///   vec_add_f32: fixed=1, volume=L, throughput=1024  → 1 + L/1024
///   vec_mul_f32: fixed=1, volume=L, throughput=1024  → 1 + L/1024
///   total(L) = 2 + 2*L/1024
///
/// After substitution L → BM*BN:
///   total(BM, BN) = 2 + 2*(BM*BN)/1024
#[test]
fn test_evaluate_with_sym_map() {
    let core = crate::core_arch::single_core();

    let l_sym = vec![Sym::new("L")];
    let schedule = Schedule::Sequential {
        schedules: vec![
            Schedule::Func {
                func: MlirFunc::with_symbols("vec_add_f32", l_sym.clone()),
                processor: None,
                time: None,
            },
            Schedule::Func {
                func: MlirFunc::with_symbols("vec_mul_f32", l_sym.clone()),
                processor: None,
                time: None,
            },
        ],
        mlir_ref: None,
        processor: None,
        time: None,
    };

    let mut sym_map = SymbolicMapping::new();
    sym_map.insert(
        Sym::new("L"),
        Expr::mul(Expr::sym("BM"), Expr::sym("BN")),
    );

    let input = ScheduleWithSymMap::new(schedule, sym_map);
    let scenarios =
        evaluate_with_sym_map(&input, &core).expect("evaluate_with_sym_map should succeed");

    assert_eq!(scenarios.len(), 1);

    // --- Verify that the sym_map is preserved ---
    let preserved_map = scenarios.sym_map.as_ref().expect("sym_map should be present");
    assert_eq!(preserved_map.entries.len(), 1);
    assert_eq!(preserved_map.entries[0].0, Sym::new("L"));

    // --- Verify symbol substitution: L should no longer appear ---
    let s = &scenarios[0];
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
    // With BM=32, BN=32 → L=1024:
    //   total = (1 + 1024/1024) + (1 + 1024/1024) = 2 + 2 = 4
    let at_32x32 = expr.substitute(&[
        (Sym::new("BM"), Expr::Const(32)),
        (Sym::new("BN"), Expr::Const(32)),
    ]);
    assert_eq!(at_32x32.eval_const(), Some(4));

    // With BM=0, BN=0 → L=0, only fixed latencies: 1 + 1 = 2
    let at_0x0 = expr.substitute(&[
        (Sym::new("BM"), Expr::Const(0)),
        (Sym::new("BN"), Expr::Const(0)),
    ]);
    assert_eq!(at_0x0.eval_const(), Some(2));

    // Constraints remain True (all vector-lane scenarios have True constraints).
    assert_eq!(s.constraints.eval_const(), Some(true));

    // --- JSON round-trip preserves sym_map ---
    let json = serde_json::to_string(&scenarios).expect("PerfScenarios should serialize");
    println!("{}", json);
    let decoded: PerfScenarios =
        serde_json::from_str(&json).expect("PerfScenarios should deserialize");
    assert_eq!(decoded.len(), scenarios.len());
    let decoded_map = decoded.sym_map.as_ref().expect("decoded sym_map should be present");
    assert_eq!(decoded_map.entries.len(), 1);
    assert_eq!(decoded_map.entries[0].0, Sym::new("L"));

    let decoded_at_32x32 = decoded[0].time_cost.to_expr().substitute(&[
        (Sym::new("BM"), Expr::Const(32)),
        (Sym::new("BN"), Expr::Const(32)),
    ]);
    assert_eq!(decoded_at_32x32.eval_const(), at_32x32.eval_const());
}
