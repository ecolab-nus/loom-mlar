use mlar_rust::*;

#[test]
fn processor_definition_names_are_independent_of_mlir_module_names() {
    let source = include_str!("processors/vector_lane.mlir");
    let yaml = include_str!("processors/vector_lane.perf.yaml");
    let definition =
        ProcessorDefinition::from_mlir_source_with_perf_yaml("different_name", source, yaml)
            .unwrap();
    assert_eq!(definition.name(), "different_name");
    definition.validate().unwrap();
}

#[test]
fn processor_builder_rejects_functionality_perf_count_mismatch() {
    let source = "module @toy {\n  func.func @f0() { return }\n  func.func @f1() { return }\n}";
    let error =
        ProcessorDefinition::from_mlir_source("toy", source, [("f0", FuncPerfModel::trivial())])
            .unwrap_err();
    assert!(error.contains("no performance model was supplied for function 'f1'"));
}

#[test]
fn data_mover_export_rejects_compute_functions() {
    let architecture = crate::arch::single_core()
        .with_processor_type("vector_lane", Some(ProcessorType::DataMover))
        .unwrap();
    assert!(
        matches!(architecture_to_mlir_unchecked(&architecture), Err(AdlExportError::DataMoverContainsCompute { processor, .. }) if processor == "vector_lane")
    );
}

#[test]
fn mlir_module_ref_from_mlir_records_single_module_and_functions() {
    let path = "tests/2d_mesh/processors/vector_lane.mlir";
    let module = MlirModule::from_mlir(path).unwrap();
    assert_eq!(module.path.as_deref(), Some(path));
    assert_eq!(module.module_name.as_deref(), Some("processor"));
    for prefix in ["vec_max_", "vec_div_"] {
        assert!(
            module
                .functions
                .iter()
                .any(|function| function.name.starts_with(prefix))
        );
    }
}

#[test]
fn mlir_func_ref_from_mlir_extracts_symbols_tensors_and_bindings() {
    let module = MlirModule::from_mlir("tests/2d_mesh/processors/matrix_lane_ss.mlir").unwrap();
    let function = module
        .functions
        .iter()
        .find(|function| function.name.starts_with("matmul_"))
        .unwrap();
    let details = function.mlir_details.as_ref().unwrap();
    assert_eq!(function.symbols, Sym::from_names(["M", "K", "N"]));
    assert!(details.tensor_args.is_empty());
    assert_eq!(details.memref_args, ["A", "B", "C"]);
    assert!(details.output_tensors.is_empty());
    assert_eq!(details.source_memrefs, ["A", "B"]);
    assert_eq!(details.target_memrefs, ["C"]);
    assert_eq!(details.mem_region_bindings.len(), 3);
    assert!(!details.linalg_ops.is_empty());
    assert!(details.tensor_symbol_bindings.is_empty());
    assert_eq!(details.memref_symbol_bindings.len(), 3);
    for (binding, (name, symbols)) in details.memref_symbol_bindings.iter().zip([
        ("A", Sym::from_names(["M", "K"])),
        ("B", Sym::from_names(["K", "N"])),
        ("C", Sym::from_names(["M", "N"])),
    ]) {
        assert_eq!(binding.memref, name);
        assert_eq!(binding.symbols, symbols);
    }
}

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

    let schedule = Schedule::Sequential {
        schedules: vec![
            Schedule::Func {
                func: add_func,
                scenarios: None,
            },
            Schedule::Parallel {
                schedules: vec![Schedule::Func {
                    func: mul_func,
                    scenarios: None,
                }],
                scenarios: Some(vec![PerfScenario {
                    constraints: ConstraintExpr::True,
                    time_cost: TimeCost::Expression(Expr::Const(40)),
                }]),
            },
        ],
        scenarios: Some(vec![PerfScenario {
            constraints: ConstraintExpr::True,
            time_cost: TimeCost::Expression(Expr::Const(150)),
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
    assert!(
        value["Sequential"]["schedules"][0]["Func"]
            .get("processor")
            .is_none()
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
