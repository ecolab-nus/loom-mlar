use std::fs;
use std::path::Path;
use std::process::Command;

use mlar_rust::mlir::{MlirMemrefSymbolBinding, MlirModule};
use mlar_rust::*;

use crate::arch::{scaled_mesh_torus, single_core};

const VEC_LANE_MLIR: &str = "tests/2d_mesh/processors/vector_lane.mlir";
const MATRIX_LANE_MLIR: &str = "tests/2d_mesh/processors/matrix_lane.mlir";
const DRAM_L1_NOC0_MLIR: &str = "tests/2d_mesh/processors/dram_l1_noc0.mlir";
const SCHEDULE_DIR: &str = "tests/2d_mesh/schedules";

fn mlir_function_text<'a>(module: &'a str, name: &str) -> &'a str {
    let marker = format!("func.func @{name}");
    let start = module
        .find(&marker)
        .unwrap_or_else(|| panic!("missing function {name}"));
    let text = &module[start..];
    let opening = text.find('{').unwrap();
    let mut depth = 0;
    for (offset, ch) in text[opening..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => depth -= 1,
            _ => {}
        }
        if depth == 0 {
            return &text[..opening + offset + 1];
        }
    }
    panic!("unclosed function {name}")
}

fn vec_func(prefix: &str) -> String {
    MlirModule::from_mlir(VEC_LANE_MLIR)
        .expect("vector_lane MLIR should parse")
        .functions
        .into_iter()
        .find(|op| op.name.starts_with(prefix))
        .unwrap_or_else(|| panic!("no function matching '{prefix}_*' in {VEC_LANE_MLIR}"))
        .name
}

fn load_example_schedule(name: &str) -> Schedule {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(SCHEDULE_DIR)
        .join(name);
    let json = fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("example schedule {path:?} should be readable: {err}"));
    serde_json::from_str(&json)
        .unwrap_or_else(|err| panic!("example schedule {path:?} should parse: {err}"))
}

fn node_scenarios(schedule: &Schedule) -> &[PerfScenario] {
    match schedule {
        Schedule::Func {
            scenarios: Some(s), ..
        }
        | Schedule::Sequential {
            scenarios: Some(s), ..
        }
        | Schedule::Parallel {
            scenarios: Some(s), ..
        } => s,
        _ => panic!("expected evaluated schedule node"),
    }
}

fn throughput_parts(cost: &TimeCost) -> (&Expr, &Expr, &Expr) {
    let TimeCost::Throughput {
        fixed_latency,
        volume,
        throughput,
    } = cost
    else {
        panic!("expected throughput components");
    };
    (fixed_latency, volume, throughput)
}

#[test]
fn test_2d_mesh_torus_perf_models() {
    let mesh = scaled_mesh_torus();
    let compute = mesh
        .processors()
        .iter()
        .filter(|array| {
            mesh.processor_definition(array.definition_name())
                .unwrap()
                .processor_type()
                == Some(&ProcessorType::Compute)
        })
        .collect::<Vec<_>>();
    assert_eq!(compute.len(), 2);
    assert_eq!(
        compute
            .iter()
            .map(|array| array.instances(&mesh).len())
            .sum::<usize>(),
        128
    );
    for array in compute {
        let definition = mesh.processor_definition(array.definition_name()).unwrap();
        definition.validate().unwrap();
        assert_eq!(
            definition.source(),
            fs::read_to_string(format!(
                "tests/2d_mesh/processors/{}.mlir",
                definition.name()
            ))
            .unwrap()
        );
    }

    let mat_module = MlirModule::from_mlir(MATRIX_LANE_MLIR).unwrap();
    assert_eq!(mat_module.path.as_deref(), Some(MATRIX_LANE_MLIR));
    assert_eq!(mat_module.module_name.as_deref(), Some("processor"));
    let matrix_mlir = fs::read_to_string(MATRIX_LANE_MLIR).unwrap();
    let matrix_lane = mesh.processor_definition("matrix_lane").unwrap();
    for (name, a_is_rram, b_is_rram) in [
        ("matmul_SS_f16", false, false),
        ("matmul_SR_f16", false, true),
        ("matmul_RS_f16", true, false),
        ("matmul_RR_f16", true, true),
        ("batch_matmul_SS_f16", false, false),
        ("batch_matmul_SR_f16", false, true),
        ("batch_matmul_RS_f16", true, false),
        ("batch_matmul_RR_f16", true, true),
    ] {
        assert!(
            mat_module
                .functions
                .iter()
                .any(|function| function.name == name)
        );
        let text = mlir_function_text(&matrix_mlir, name);
        let rank = if name.starts_with("batch_") {
            "?x?x?"
        } else {
            "?x?"
        };
        let a_type = format!(
            "%A: memref<{rank}xf16{}>",
            if a_is_rram { ", 1" } else { "" }
        );
        let b_type = format!(
            "%B{}: memref<{rank}xf16{}>",
            if name.starts_with("batch_") {
                "mat"
            } else {
                ""
            },
            if b_is_rram { ", 1" } else { "" }
        );
        assert!(text.contains(&a_type), "{name}: {a_type}");
        assert!(text.contains(&b_type), "{name}: {b_type}");
        assert!(text.contains(&format!("%C: memref<{rank}xf16>")));
        let model = matrix_lane.get_function(name).unwrap();
        let (latency, _, throughput) = throughput_parts(&model.perf.scenarios[0].time_cost);
        if a_is_rram || b_is_rram {
            assert_eq!(latency.eval_const(), Some(888));
            assert_eq!(throughput.eval_const(), Some(888));
        } else {
            assert_eq!(model.perf.scenarios.len(), 2);
            assert_eq!(throughput.eval_const(), Some(716));
        }
    }
    for prefix in ["vec_vsum_", "vec_vmax_", "vec_max1_"] {
        assert!(
            mat_module
                .functions
                .iter()
                .any(|function| function.name.starts_with(prefix))
        );
    }
    let matmul_details = mat_module
        .functions
        .iter()
        .find(|function| function.name == "matmul_SS_f16")
        .unwrap()
        .mlir_details
        .as_ref()
        .unwrap();
    assert!(matmul_details.tensor_args.is_empty());
    assert_eq!(matmul_details.memref_args, ["A", "B", "C"]);
    assert_eq!(
        matmul_details
            .memref_symbol_bindings
            .iter()
            .filter(|binding| binding.memref != "C")
            .cloned()
            .collect::<Vec<_>>(),
        vec![
            MlirMemrefSymbolBinding {
                memref: "A".into(),
                symbols: Sym::from_names(["M", "K"])
            },
            MlirMemrefSymbolBinding {
                memref: "B".into(),
                symbols: Sym::from_names(["K", "N"])
            },
        ]
    );
    assert!(matmul_details.output_tensors.is_empty());
    assert_eq!(
        matmul_details
            .memref_symbol_bindings
            .iter()
            .find(|binding| binding.memref == "C")
            .unwrap()
            .symbols,
        Sym::from_names(["M", "N"])
    );
    assert_eq!(matmul_details.mem_region_bindings.len(), 3);
    assert_eq!(
        matmul_details
            .mem_region_bindings
            .iter()
            .map(|binding| (binding.memref.as_str(), binding.region.as_str()))
            .collect::<Vec<_>>(),
        [("A", "input_0"), ("B", "input_0"), ("C", "output_0")],
    );

    let vec_module = MlirModule::from_mlir(VEC_LANE_MLIR).unwrap();
    assert_eq!(vec_module.path.as_deref(), Some(VEC_LANE_MLIR));
    assert_eq!(vec_module.module_name.as_deref(), Some("processor"));
    for prefix in [
        "vec_max_", "vec_exp_", "vec_sum_", "vec_add_", "vec_mul_", "vec_div_",
    ] {
        assert!(
            vec_module
                .functions
                .iter()
                .any(|function| function.name.starts_with(prefix))
        );
    }

    for mover_name in ["dram_l1_noc0", "l1_l1_noc0", "l1_dram_noc1"] {
        let definition = mesh.processor_definition(mover_name).unwrap();
        definition.validate().unwrap();
        assert_eq!(definition.processor_type(), Some(&ProcessorType::DataMover));
        assert_eq!(
            definition.source(),
            fs::read_to_string(format!("tests/2d_mesh/processors/{mover_name}.mlir")).unwrap()
        );
        let array = mesh.processor_array(mover_name).unwrap();
        assert!(!array.connection().inputs.is_empty());
        assert!(!array.connection().outputs.is_empty());
        assert!(
            !array
                .resources()
                .iter()
                .any(|resource| ["L1_torus_h", "L1_torus_v"].contains(&resource.name()))
        );
        let resource = if mover_name == "l1_dram_noc1" {
            "noc1"
        } else {
            "noc0"
        };
        assert!(array.resources().iter().any(|r| r.name() == resource));
    }

    let noc0 = mesh.processor_definition("dram_l1_noc0").unwrap();
    let copy_mlir = fs::read_to_string(DRAM_L1_NOC0_MLIR).unwrap();
    for (name, expected_dst_kind) in [
        ("dram_to_l1_S_f16", "dst_mem_space @output_0,"),
        ("dram_to_l1_S_bcst", "dst_mem_space @output_0,"),
        ("dram_to_l1_R_f16", "dst_mem_space @output_0 : 1,"),
        ("dram_to_l1_R_bcst", "dst_mem_space @output_0 : 1,"),
    ] {
        assert!(noc0.get_function(name).is_some());
        assert!(
            mlir_function_text(&copy_mlir, name).contains(expected_dst_kind),
            "{name}"
        );
    }
    for (name, latency, throughput) in [
        ("dram_to_l1_S_f16", 454, 150),
        ("dram_to_l1_S_bcst", 344, 150),
        ("dram_to_l1_R_f16", 888, 888),
        ("dram_to_l1_R_bcst", 888, 888),
    ] {
        let model = noc0.get_function(name).unwrap();
        let (fixed_latency, _, rate) = throughput_parts(&model.perf.scenarios[0].time_cost);
        assert_eq!(fixed_latency.eval_const(), Some(latency));
        assert_eq!(rate.eval_const(), Some(throughput));
    }
    for stale in [
        "batch_dram_to_l1_f16",
        "batch_dram_to_l1_bcst",
        "l1_to_dram_f16",
        "batch_l1_to_dram_f16",
        "dram_to_l1_1d_bcst_f16",
        "dram_to_l1_2d_bcst_f16",
    ] {
        assert!(noc0.get_function(stale).is_none());
    }
    let broadcast = noc0.get_function("dram_to_l1_S_bcst").unwrap();
    for symbol in Sym::from_names(["M", "N", "bcst_x", "bcst_y", "effective_bandwidth"]) {
        assert!(broadcast.func.symbols.contains(&symbol));
    }

    let noc1 = mesh.processor_definition("l1_dram_noc1").unwrap();
    assert!(
        noc1.get_function("l1_to_dram_f16")
            .unwrap()
            .func
            .mlir_details
            .is_some()
    );
    for stale in [
        "batch_l1_to_dram_f16",
        "dram_to_l1_S_f16",
        "dram_to_l1_R_f16",
        "batch_dram_to_l1_f16",
        "dram_to_l1_S_bcst",
        "dram_to_l1_R_bcst",
        "batch_dram_to_l1_bcst",
        "l1_gather",
        "batch_l1_gather",
        "dram_to_l1_1d_bcst_f16",
        "dram_to_l1_2d_bcst_f16",
    ] {
        assert!(noc1.get_function(stale).is_none());
    }
    let gather = mesh
        .processor_definition("l1_l1_noc0")
        .unwrap()
        .get_function("l1_gather")
        .unwrap();
    for symbol in Sym::from_names(["B", "M", "N", "gather_x", "gather_y"]) {
        assert!(gather.func.symbols.contains(&symbol));
    }
    let details = gather.func.mlir_details.as_ref().unwrap();
    assert_eq!(details.memref_args, ["l1_src", "l1_dst"]);
    assert_eq!(
        details.memref_symbol_bindings[0].symbols,
        Sym::from_names(["M", "N"])
    );
    assert_eq!(
        details.memref_symbol_bindings[1].symbols,
        Sym::from_names(["B", "M", "N"])
    );
    assert!(
        gather.perf.scenarios[0]
            .time_cost
            .to_expr()
            .free_symbols()
            .contains(&Sym::new("B"))
    );

    let details = noc0
        .get_function("dram_to_l1_S_f16")
        .unwrap()
        .func
        .mlir_details
        .as_ref()
        .unwrap();
    assert_eq!(details.memref_args, ["dram_src", "l1_dst"]);
    assert_eq!(details.source_memrefs, ["dram_src"]);
    assert_eq!(details.target_memrefs, ["l1_dst"]);
    assert!(details.tensor_args.is_empty());
    assert!(details.tensor_symbol_bindings.is_empty());
    assert_eq!(details.memref_symbol_bindings.len(), 2);
    assert_eq!(details.memref_symbol_bindings[0].memref, "dram_src");
    assert_eq!(details.memref_symbol_bindings[1].memref, "l1_dst");
    assert_eq!(
        details.memref_symbol_bindings[0].symbols,
        Sym::from_names(["M", "N"])
    );
    assert_eq!(
        details.memref_symbol_bindings[1].symbols,
        Sym::from_names(["M", "N"])
    );
    assert_eq!(
        details
            .mem_region_bindings
            .iter()
            .map(|binding| (binding.memref.as_str(), binding.region.as_str()))
            .collect::<Vec<_>>(),
        [("dram_src", "input_0"), ("l1_dst", "output_0")]
    );
}

#[test]
fn test_2d_mesh_torus() {
    let mesh = scaled_mesh_torus();
    assert_eq!(mesh.name(), "system");
    assert!(mesh.memory("DRAM").is_some());
    assert!(mesh.networks().is_empty());
    assert_eq!(mesh.memories().len(), 2);
    assert_eq!(mesh.processors().len(), 5);
    assert_eq!(
        mesh.memory("L1")
            .unwrap()
            .axes()
            .iter()
            .map(|axis| axis.name())
            .collect::<Vec<_>>(),
        ["x", "y"]
    );
    for name in ["matrix_lane", "vector_lane"] {
        assert_eq!(
            mesh.processor_array(name).unwrap().instances(&mesh).len(),
            64
        );
    }
    for name in ["dram_l1_noc0", "l1_l1_noc0", "l1_dram_noc1"] {
        assert_eq!(
            mesh.processor_array(name).unwrap().instances(&mesh).len(),
            1
        );
    }
    for stale in ["L1_torus_h", "L1_torus_v"] {
        assert!(
            !mesh
                .resources()
                .iter()
                .any(|resource| resource.name() == stale)
        );
    }
    let yaml = architecture_to_visualization_yaml(&mesh).unwrap();
    assert!(yaml.contains("schema_version: mlar.visualization.v1"));
    assert!(!yaml.contains("L1_torus_h") && !yaml.contains("L1_torus_v"));
    let canonical = serde_json::to_value(&mesh).unwrap();
    let restored: Architecture = serde_json::from_value(canonical.clone()).unwrap();
    assert_eq!(serde_json::to_value(restored).unwrap(), canonical);
}

#[test]
fn test_export_2d_mesh_torus_visualization_yaml() {
    let mesh = scaled_mesh_torus();
    let yaml = architecture_to_visualization_yaml(&mesh)
        .expect("visualization YAML serialization should succeed");
    let document: VisualizationDocumentV1 =
        serde_yaml::from_str(&yaml).expect("serialized visualization YAML should be valid");

    assert_eq!(document.schema_version, VISUALIZATION_SCHEMA_VERSION);
    assert_eq!(document.architecture.name, "system");
    assert_eq!(document.scopes.len(), 2);
    assert!(document.scopes.iter().any(|scope| {
        scope.name == "x_y"
            && scope.replication_factor == Some(64)
            && scope
                .dimensions
                .iter()
                .any(|dimension| dimension.name == "x")
            && scope
                .dimensions
                .iter()
                .any(|dimension| dimension.name == "y")
    }));
    assert!(document.components.iter().any(
        |component| matches!(component, VisualizationComponent::Memory { name, .. } if name == "DRAM")
    ));
    assert!(
        document
            .relationships
            .iter()
            .any(|relationship| { relationship.kind == VisualizationRelationshipKind::Read })
    );
}

#[test]
fn test_2d_mesh_example_schedules() {
    let core = single_core();

    let vector = load_example_schedule("core_vector_two_ops.json");
    let vector_result = evaluate(&vector, &core).expect("vector example schedule should evaluate");
    let vector_scenarios = node_scenarios(&vector_result);
    assert_eq!(vector_scenarios.len(), 1);
    assert_eq!(
        vector_scenarios[0]
            .time_cost
            .to_expr()
            .substitute(&[
                (Sym::new("BM"), Expr::Const(32)),
                (Sym::new("BN"), Expr::Const(32)),
            ])
            .eval_const(),
        Some(4)
    );

    let parallel_vector = load_example_schedule("core_parallel_vector.json");
    let parallel_vector_result = evaluate(&parallel_vector, &core)
        .expect("parallel vector example schedule should evaluate");
    let parallel_vector_scenarios = node_scenarios(&parallel_vector_result);
    assert_eq!(parallel_vector_scenarios.len(), 1);
    assert_eq!(
        parallel_vector_scenarios[0]
            .time_cost
            .to_expr()
            .substitute(&[
                (Sym::new("BM"), Expr::Const(32)),
                (Sym::new("BN"), Expr::Const(32)),
            ])
            .eval_const(),
        Some(147)
    );

    let nested = load_example_schedule("core_nested_parallel_sequential.json");
    let nested_result = evaluate(&nested, &core).expect("nested example schedule should evaluate");
    let nested_scenarios = node_scenarios(&nested_result);
    assert_eq!(nested_scenarios.len(), 1);
    assert_eq!(
        nested_scenarios[0]
            .time_cost
            .to_expr()
            .substitute(&[
                (Sym::new("BM"), Expr::Const(32)),
                (Sym::new("BN"), Expr::Const(32)),
            ])
            .eval_const(),
        Some(149)
    );

    let matmul = load_example_schedule("core_matmul.json");
    let matmul_result = evaluate(&matmul, &core).expect("matmul example schedule should evaluate");
    let matmul_scenarios = node_scenarios(&matmul_result);
    assert_eq!(matmul_scenarios.len(), 2);
    for scenario in matmul_scenarios {
        let mut free = scenario.time_cost.to_expr().free_symbols();
        free.extend(scenario.constraints.free_symbols());
        assert!(!free.contains(&Sym::new("M")));
        assert!(!free.contains(&Sym::new("N")));
        assert!(!free.contains(&Sym::new("K")));
        assert!(free.contains(&Sym::new("BM")));
        assert!(free.contains(&Sym::new("BN")));
        assert!(free.contains(&Sym::new("BK")));
    }

    let system = scaled_mesh_torus();
    let data_roundtrip = load_example_schedule("system_data_roundtrip.json");
    let data_result =
        evaluate(&data_roundtrip, &system).expect("system data example schedule should evaluate");
    let data_scenarios = node_scenarios(&data_result);
    assert_eq!(data_scenarios.len(), 1);
    assert_eq!(
        data_scenarios[0]
            .time_cost
            .to_expr()
            .substitute(&[
                (Sym::new("BM"), Expr::Const(30)),
                (Sym::new("BN"), Expr::Const(25)),
            ])
            .eval_const(),
        Some(928)
    );
}

#[test]
fn test_evaluate_vector_lane_sequential_schedule() {
    let core = crate::arch::single_core();

    let l_sym = vec![Sym::new("L")];
    let sym_map = {
        let mut m = SymbolicMapping::new();
        m.insert(Sym::new("L"), Expr::mul(Expr::sym("BM"), Expr::sym("BN")));
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
                scenarios: None,
            },
            Schedule::Func {
                func: {
                    let mut f = MlirFunc::with_symbols(&vec_func("vec_exp"), l_sym.clone());
                    f.sym_map = sym_map.clone();
                    f
                },
                scenarios: None,
            },
            Schedule::Func {
                func: {
                    let mut f = MlirFunc::with_symbols(&vec_func("vec_mul"), l_sym.clone());
                    f.sym_map = sym_map.clone();
                    f
                },
                scenarios: None,
            },
            Schedule::Func {
                func: {
                    let mut f = MlirFunc::with_symbols(&vec_func("vec_div"), l_sym.clone());
                    f.sym_map = sym_map.clone();
                    f
                },
                scenarios: None,
            },
        ],
        scenarios: None,
    };

    let result = evaluate(&schedule, &core).expect("sequential vector schedule should evaluate");

    let func_scenarios: Vec<&Vec<PerfScenario>> = match &result {
        Schedule::Sequential { schedules, .. } => schedules
            .iter()
            .map(|s| match s {
                Schedule::Func {
                    scenarios: Some(sc),
                    ..
                } => sc,
                _ => panic!("expected Func with filled scenarios"),
            })
            .collect(),
        _ => panic!("expected Sequential"),
    };

    for sc in &func_scenarios {
        assert_eq!(sc.len(), 1);
    }

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

    let at_32x32 = add_expr.substitute(&[
        (Sym::new("BM"), Expr::Const(32)),
        (Sym::new("BN"), Expr::Const(32)),
    ]);
    assert_eq!(at_32x32.eval_const(), Some(2));

    assert_eq!(func_scenarios[0][0].constraints.eval_const(), Some(true));

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

    let json = serde_json::to_string(&result).expect("Schedule should serialize");
    let decoded: Schedule = serde_json::from_str(&json).expect("Schedule should deserialize");
    let decoded_json = serde_json::to_string(&decoded).expect("decoded Schedule should serialize");
    assert_eq!(json, decoded_json);
}

#[test]
fn test_evaluate_with_sym_map() {
    let core = crate::arch::single_core();

    let l_sym = vec![Sym::new("L")];
    let sym_map = {
        let mut m = SymbolicMapping::new();
        m.insert(Sym::new("L"), Expr::mul(Expr::sym("BM"), Expr::sym("BN")));
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
                scenarios: None,
            },
            Schedule::Func {
                func: {
                    let mut f = MlirFunc::with_symbols(&vec_func("vec_mul"), l_sym.clone());
                    f.sym_map = sym_map.clone();
                    f
                },
                scenarios: None,
            },
        ],
        scenarios: None,
    };

    let result = evaluate(&schedule, &core).expect("evaluate should succeed");

    let func_scenarios: Vec<&Vec<PerfScenario>> = match &result {
        Schedule::Sequential { schedules, .. } => schedules
            .iter()
            .map(|s| match s {
                Schedule::Func {
                    scenarios: Some(sc),
                    ..
                } => sc,
                _ => panic!("expected Func with filled scenarios"),
            })
            .collect(),
        _ => panic!("expected Sequential"),
    };

    assert_eq!(func_scenarios.len(), 2);
    for sc in &func_scenarios {
        assert_eq!(sc.len(), 1);
    }

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

    let s = &func_scenarios[0][0];
    assert!(
        s.time_cost.as_expression().is_some(),
        "evaluated scenario should be Expression"
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

    let at_32x32 = expr.substitute(&[
        (Sym::new("BM"), Expr::Const(32)),
        (Sym::new("BN"), Expr::Const(32)),
    ]);
    assert_eq!(at_32x32.eval_const(), Some(2));

    let at_0x0 = expr.substitute(&[
        (Sym::new("BM"), Expr::Const(0)),
        (Sym::new("BN"), Expr::Const(0)),
    ]);
    assert_eq!(at_0x0.eval_const(), Some(1));

    assert_eq!(s.constraints.eval_const(), Some(true));

    let json = serde_json::to_string(&result).expect("Schedule should serialize");
    let decoded: Schedule = serde_json::from_str(&json).expect("Schedule should deserialize");
    let decoded_json = serde_json::to_string(&decoded).expect("decoded Schedule should serialize");
    assert_eq!(json, decoded_json);
}

#[test]
fn test_generate_core_evaluator_binary() {
    let core = crate::arch::single_core();

    let output_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/2d_mesh/bin");
    let binary = generate_evaluator_binary(&core, "eval_core", &output_dir)
        .expect("binary generation should succeed");

    assert!(
        binary.exists(),
        "generated binary should exist at {binary:?}"
    );

    let l_sym = vec![Sym::new("L")];
    let sym_map = {
        let mut m = SymbolicMapping::new();
        m.insert(Sym::new("L"), Expr::mul(Expr::sym("BM"), Expr::sym("BN")));
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
                scenarios: None,
            },
            Schedule::Func {
                func: {
                    let mut f = MlirFunc::with_symbols(&vec_func("vec_mul"), l_sym.clone());
                    f.sym_map = sym_map.clone();
                    f
                },
                scenarios: None,
            },
        ],
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

    let func_scenarios: Vec<&Vec<PerfScenario>> = match &result {
        Schedule::Sequential { schedules, .. } => schedules
            .iter()
            .map(|s| match s {
                Schedule::Func {
                    scenarios: Some(sc),
                    ..
                } => sc,
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

    let at_32x32 = expr.substitute(&[
        (Sym::new("BM"), Expr::Const(32)),
        (Sym::new("BN"), Expr::Const(32)),
    ]);
    assert_eq!(at_32x32.eval_const(), Some(2));
}

#[test]
fn test_evaluate_system_data_mover_schedule() {
    let system = scaled_mesh_torus();

    let mn_sym = vec![
        Sym::new("M"),
        Sym::new("N"),
        Sym::new("effective_bandwidth"),
    ];
    let sym_map = {
        let mut m = SymbolicMapping::new();
        m.insert(Sym::new("M"), Expr::sym("BM"));
        m.insert(Sym::new("N"), Expr::sym("BN"));
        m.insert(Sym::new("effective_bandwidth"), Expr::Const(1));
        Some(m)
    };

    let schedule = Schedule::Sequential {
        schedules: vec![
            Schedule::Func {
                func: {
                    let mut f = MlirFunc::with_symbols("dram_to_l1_S_f16", mn_sym.clone());
                    f.sym_map = sym_map.clone();
                    f
                },
                scenarios: None,
            },
            Schedule::Func {
                func: {
                    let mut f = MlirFunc::with_symbols("dram_to_l1_S_f16", mn_sym.clone());
                    f.sym_map = sym_map.clone();
                    f
                },
                scenarios: None,
            },
        ],
        scenarios: None,
    };

    let result = evaluate(&schedule, &system).expect("data mover schedule should evaluate");

    let func_scenarios = match &result {
        Schedule::Sequential {
            schedules,
            scenarios: Some(_seq_sc),
            ..
        } => {
            let per_func: Vec<&Vec<PerfScenario>> = schedules
                .iter()
                .map(|s| match s {
                    Schedule::Func {
                        scenarios: Some(sc),
                        ..
                    } => sc,
                    _ => panic!("expected Func with filled scenarios"),
                })
                .collect();
            per_func
        }
        _ => panic!("expected Sequential with filled scenarios"),
    };

    assert_eq!(func_scenarios.len(), 2);
    for sc in &func_scenarios {
        assert_eq!(sc.len(), 1);
    }

    let expr = func_scenarios[0][0].time_cost.to_expr();
    let free = expr.free_symbols();
    assert!(
        !free.contains(&Sym::new("M")) && !free.contains(&Sym::new("N")),
        "M and N should have been substituted away, but symbols are: {:?}",
        free
    );
    assert!(
        free.contains(&Sym::new("BM")) && free.contains(&Sym::new("BN")),
        "BM and BN should appear after substitution, but symbols are: {:?}",
        free
    );

    match &result {
        Schedule::Sequential { schedules, .. } => {
            for s in schedules {
                match s {
                    Schedule::Func { func, .. } => {
                        let sm = func.sym_map.as_ref().expect("sym_map should be present");
                        assert_eq!(sm.entries.len(), 3);
                        assert_eq!(sm.entries[0].0, Sym::new("M"));
                        assert_eq!(sm.entries[1].0, Sym::new("N"));
                        assert_eq!(sm.entries[2].0, Sym::new("effective_bandwidth"));
                    }
                    _ => panic!("expected Func"),
                }
            }
        }
        _ => panic!("expected Sequential"),
    }

    let json = serde_json::to_string(&result).expect("Schedule should serialize");
    let decoded: Schedule = serde_json::from_str(&json).expect("Schedule should deserialize");
    let decoded_json = serde_json::to_string(&decoded).expect("decoded Schedule should serialize");
    assert_eq!(json, decoded_json);
}

#[test]
#[cfg_attr(
    not(mlar_has_mlir_validators),
    ignore = "requires adl-opt and loom-opt"
)]
fn test_export_2d_mesh_torus_mlir() {
    if !mlir_validators_available() {
        eprintln!("skipping checked MLIR export: adl-opt and/or loom-opt is unavailable");
        return;
    }
    let mesh = scaled_mesh_torus();
    let mlir = architecture_to_mlir(&mesh).expect("MLIR export and validation should succeed");

    let out_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/2d_mesh/2d_mesh_torus.mlir");
    assert_eq!(mlir, include_str!("2d_mesh_torus.mlir"));
    fs::write(out_path, &mlir).expect("Failed to write MLIR file");
}

#[test]
fn test_generate_system_evaluator_binary() {
    let system = scaled_mesh_torus();

    let output_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/2d_mesh/bin");
    let binary = generate_evaluator_binary(&system, "eval_system", &output_dir)
        .expect("system binary generation should succeed");

    assert!(
        binary.exists(),
        "generated binary should exist at {binary:?}"
    );

    let mn_sym = vec![
        Sym::new("M"),
        Sym::new("N"),
        Sym::new("effective_bandwidth"),
    ];
    let sym_map = {
        let mut m = SymbolicMapping::new();
        m.insert(Sym::new("M"), Expr::sym("BM"));
        m.insert(Sym::new("N"), Expr::sym("BN"));
        m.insert(Sym::new("effective_bandwidth"), Expr::Const(1));
        Some(m)
    };

    let schedule = Schedule::Sequential {
        schedules: vec![
            Schedule::Func {
                func: {
                    let mut f = MlirFunc::with_symbols("dram_to_l1_S_f16", mn_sym.clone());
                    f.sym_map = sym_map.clone();
                    f
                },
                scenarios: None,
            },
            Schedule::Func {
                func: {
                    let mut f = MlirFunc::with_symbols("dram_to_l1_S_f16", mn_sym.clone());
                    f.sym_map = sym_map.clone();
                    f
                },
                scenarios: None,
            },
        ],
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

    let func_scenarios: Vec<&Vec<PerfScenario>> = match &result {
        Schedule::Sequential { schedules, .. } => schedules
            .iter()
            .map(|s| match s {
                Schedule::Func {
                    scenarios: Some(sc),
                    ..
                } => sc,
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
    assert!(!free.contains(&Sym::new("M")) && !free.contains(&Sym::new("N")));
    assert!(free.contains(&Sym::new("BM")) && free.contains(&Sym::new("BN")));
}

#[test]
#[cfg_attr(
    not(mlar_has_mlir_validators),
    ignore = "requires adl-opt and loom-opt"
)]
fn test_generate_system_arch_query_binary_mlir() {
    if !mlir_validators_available() {
        eprintln!("skipping MLIR query binary test: adl-opt and/or loom-opt is unavailable");
        return;
    }
    let system = scaled_mesh_torus();

    let output_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/2d_mesh/bin");
    let binary = generate_arch_query_binary(&system, "query_system", &output_dir)
        .expect("system query binary generation should succeed");

    assert!(
        binary.exists(),
        "generated binary should exist at {binary:?}"
    );

    let query_json =
        serde_json::to_string(&ArchitectureQuery::Mlir).expect("query should serialize");

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
                .write_all(query_json.as_bytes())
                .unwrap();
            child.wait_with_output()
        })
        .expect("binary should execute");

    assert!(
        output.status.success(),
        "binary exited with error: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let mlir = String::from_utf8(output.stdout).expect("binary output should be valid UTF-8");

    let expected =
        architecture_to_mlir(&system).expect("MLIR export and validation should succeed");
    assert_eq!(mlir, expected);
    assert!(mlir.starts_with("module @arch_system {\n"));
}
