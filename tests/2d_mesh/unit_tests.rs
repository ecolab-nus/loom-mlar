use mlar_rust::*;

// ── helpers ──────────────────────────────────────────────────────────────────

fn route_regions() -> (MemoryRegion, MemoryRegion) {
    let dim_bank = Dimension::new_int("nbank", 16);
    let dim_x = Dimension::new_int("x", 8);
    let dim_y = Dimension::new_int("y", 8);
    let dim_dram_channel = Dimension::new_int("dram_channel", 8);
    let l1 = MemoryRegion::bank(SizeExpr::Const(16), SizeExpr::Const(5856))
        .scale(&dim_bank)
        .with_name("L1");
    let array_l1 = l1.scale(&[dim_x, dim_y]).with_name("array_L1");
    let dram = MemoryRegion::bank(SizeExpr::Const(8192), SizeExpr::Const(196608))
        .with_name("DRAM_bank")
        .scale(&dim_dram_channel)
        .with_name("DRAM");
    (dram, array_l1)
}

// ── from src/arch/processor.rs ───────────────────────────────────────────────

#[test]
fn processor_builder_rejects_name_mismatch_with_mlir_module() {
    let module = MlirModule::from_mlir("tests/2d_mesh/processors/vector_lane.mlir")
        .expect("vector_lane should parse");
    let perf_models = vec![FuncPerfModel::trivial(); module.functions.len()];

    let err = ComputeProcessor::builder()
        .named("wrong_name")
        .functionality(module)
        .perf(perf_models)
        .finish()
        .expect_err("name mismatch should fail before interface validation");
    assert!(
        err.contains("Processor name 'wrong_name' does not match MLIR module name 'vector_lane'")
    );
}

#[test]
fn processor_builder_rejects_functionality_perf_count_mismatch() {
    let module =
        MlirModule::from_functions("toy", vec![MlirFunc::named("f0"), MlirFunc::named("f1")]);

    let err = ComputeProcessor::builder()
        .named("toy")
        .functionality(module)
        .perf(vec![FuncPerfModel::trivial()])
        .finish()
        .expect_err("each function should require one performance model");

    assert!(err.contains("has 1 performance models but functionality has 2 functions"));
}

#[test]
fn data_mover_validation_rejects_missing_memref_interface() {
    let functionality = MlirModule::from_mlir("tests/2d_mesh/processors/vector_lane.mlir")
        .expect("vector_lane should parse");
    let perf_models = vec![FuncPerfModel::trivial(); functionality.functions.len()];

    let (dram, array_l1) = route_regions();
    let err = DataMover::builder()
        .named("vector_lane")
        .from_region(dram)
        .to_region(array_l1)
        .functionality(functionality)
        .perf(perf_models)
        .finish()
        .expect_err("vector lane functions should not satisfy data-mover interface");
    assert!(
        err.contains("pure data-mover function must contain exactly one loom.copy or loom.gather")
    );
}

// From src/mlir/tests/parser.rs.

#[test]
fn mlir_module_ref_from_mlir_records_single_module_and_functions() {
    let module = MlirModule::from_mlir("tests/2d_mesh/processors/vector_lane.mlir")
        .expect("vector_lane.mlir should parse");
    assert_eq!(
        module.path.as_deref(),
        Some("tests/2d_mesh/processors/vector_lane.mlir")
    );
    assert_eq!(module.module_name.as_deref(), Some("vector_lane"));
    assert!(
        module
            .functions
            .iter()
            .any(|f| f.name.starts_with("vec_max_"))
    );
    assert!(
        module
            .functions
            .iter()
            .any(|f| f.name.starts_with("vec_div_"))
    );

    // Uppercase alias naming is also supported.
    let alias_module = MLIRModuleRef::from_mlir("tests/2d_mesh/processors/vector_lane.mlir")
        .expect("alias constructor should parse");
    assert_eq!(alias_module.module_name.as_deref(), Some("vector_lane"));
}

#[test]
fn mlir_func_ref_from_mlir_extracts_symbols_tensors_and_bindings() {
    let module = MlirModule::from_mlir("tests/2d_mesh/processors/matrix_lane.mlir")
        .expect("matrix_lane.mlir should parse");
    let func = module
        .functions
        .iter()
        .find(|f| f.name.starts_with("matmul_"))
        .expect("matmul_* function should exist");
    let details = func
        .mlir_details
        .as_ref()
        .expect("from_mlir should populate mlir_details");

    assert_eq!(func.symbols, vec!["M".into(), "N".into(), "K".into()]);
    assert!(details.tensor_args.is_empty());
    assert_eq!(details.memref_args, vec!["A", "B", "C"]);
    assert!(details.output_tensors.is_empty());
    assert_eq!(details.source_memrefs, vec!["A", "B"]);
    assert_eq!(details.target_memrefs, vec!["C"]);
    assert_eq!(details.mem_region_bindings.len(), 3);
    assert!(!details.linalg_ops.is_empty());
    assert!(details.tensor_symbol_bindings.is_empty());
    assert_eq!(details.memref_symbol_bindings.len(), 3);

    assert_eq!(details.memref_symbol_bindings[0].memref, "A");
    assert_eq!(
        details.memref_symbol_bindings[0].symbols,
        vec!["M".into(), "K".into()]
    );
    assert_eq!(details.memref_symbol_bindings[1].memref, "B");
    assert_eq!(
        details.memref_symbol_bindings[1].symbols,
        vec!["K".into(), "N".into()]
    );
    assert_eq!(details.memref_symbol_bindings[2].memref, "C");
    assert_eq!(
        details.memref_symbol_bindings[2].symbols,
        vec!["M".into(), "N".into()]
    );
}

// ── from src/schedule/schedule.rs ────────────────────────────────────────────

#[test]
fn schedule_serializes_and_deserializes() {
    let module = MlirModule::from_mlir("tests/2d_mesh/processors/vector_lane.mlir")
        .expect("vector_lane MLIR should parse");
    let add_func = module
        .functions
        .iter()
        .find(|func| func.name.starts_with("vec_add_"))
        .cloned()
        .expect("vec_add_* should exist");
    let mul_func = module
        .functions
        .iter()
        .find(|func| func.name.starts_with("vec_mul_"))
        .cloned()
        .expect("vec_mul_* should exist");
    let add_name = add_func.name.clone();
    let mul_name = mul_func.name.clone();
    let add_fp = FunctionProcessor::new(MlirFunc::named(&add_name), FuncPerfModel::trivial());
    let mul_fp = FunctionProcessor::new(MlirFunc::named(&mul_name), FuncPerfModel::trivial());

    let schedule = Schedule::Sequential {
        schedules: vec![
            Schedule::Func {
                func: add_func,
                processor: Some(add_fp),
                scenarios: None,
            },
            Schedule::Parallel {
                schedules: vec![Schedule::Func {
                    func: mul_func,
                    processor: Some(mul_fp),
                    scenarios: None,
                }],
                scenarios: Some(vec![PerfScenario {
                    constraints: ConstraintExpr::True,
                    time_cost: TimeCost::Concrete(Expr::Const(40)),
                }]),
            },
        ],
        scenarios: Some(vec![PerfScenario {
            constraints: ConstraintExpr::True,
            time_cost: TimeCost::Concrete(Expr::Const(150)),
        }]),
    };

    let value = serde_json::to_value(&schedule).expect("schedule should serialize");
    assert!(
        !value["Sequential"]
            .as_object()
            .unwrap()
            .contains_key("mlir_ref")
    );
    assert!(
        !value["Sequential"]
            .as_object()
            .unwrap()
            .contains_key("processor")
    );
    assert!(value["Sequential"]["scenarios"].is_array());
    assert!(
        value["Sequential"]["schedules"][0]["Func"]
            .get("scenarios")
            .is_none()
    );
    assert_eq!(
        value["Sequential"]["schedules"][0]["Func"]["func"]["name"],
        serde_json::json!(add_name)
    );
    assert_eq!(
        value["Sequential"]["schedules"][0]["Func"]["processor"]["func"]["name"],
        serde_json::json!(add_name)
    );
    assert_eq!(
        value["Sequential"]["schedules"][1]["Parallel"]["schedules"][0]["Func"]["func"]["name"],
        serde_json::json!(mul_name)
    );
    assert!(value["Sequential"]["schedules"][1]["Parallel"]["scenarios"].is_array());
    assert!(
        value["Sequential"]["schedules"][1]["Parallel"]["schedules"][0]["Func"]
            .get("scenarios")
            .is_none()
    );

    let decoded: Schedule =
        serde_json::from_value(value.clone()).expect("schedule should deserialize");
    let round_trip = serde_json::to_value(decoded).expect("schedule should serialize");
    assert_eq!(round_trip, value);
}

#[test]
fn schedule_serializes_and_deserializes_with_absent_optional_fields() {
    let func = MlirModule::from_mlir("tests/2d_mesh/processors/vector_lane.mlir")
        .expect("vector_lane MLIR should parse")
        .functions
        .into_iter()
        .find(|f| f.name.starts_with("vec_add_"))
        .expect("vec_add_* should exist");

    let schedule = Schedule::Sequential {
        schedules: vec![Schedule::Parallel {
            schedules: vec![Schedule::Func {
                func,
                processor: None,
                scenarios: None,
            }],
            scenarios: None,
        }],
        scenarios: None,
    };

    let value = serde_json::to_value(&schedule).expect("schedule should serialize");
    let seq = value["Sequential"]
        .as_object()
        .expect("Sequential payload should be an object");
    assert!(!seq.contains_key("mlir_ref"));
    assert!(!seq.contains_key("processor"));
    assert!(!seq.contains_key("scenarios"));

    let par = value["Sequential"]["schedules"][0]["Parallel"]
        .as_object()
        .expect("Parallel payload should be an object");
    assert!(!par.contains_key("mlir_ref"));
    assert!(!par.contains_key("processor"));
    assert!(!par.contains_key("scenarios"));

    let op = value["Sequential"]["schedules"][0]["Parallel"]["schedules"][0]["Func"]
        .as_object()
        .expect("Func payload should be an object");
    assert!(!op.contains_key("processor"));
    assert!(!op.contains_key("scenarios"));

    let decoded: Schedule =
        serde_json::from_value(value.clone()).expect("schedule should deserialize");
    let round_trip = serde_json::to_value(decoded).expect("schedule should serialize");
    assert_eq!(round_trip, value);
}
