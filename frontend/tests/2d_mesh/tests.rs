use mlar_frontend::Connection;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

use mlar_rust::arch::EndpointIndex;
use mlar_rust::{
    AdlExportError, Architecture, Axis, Expr, MemoryDefinition, MemoryTechnology, PerfScenario,
    ProcessorTarget, Resource, Schedule, Sym, architecture_to_mlir, evaluate,
    generate_evaluator_binary,
};

#[path = "../../../tests/2d_mesh/arch.rs"]
mod core_fixture;

fn processor_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/2d_mesh/processors")
}

fn load() -> mlar_rust::Architecture {
    mlar_frontend::archs::load_arch(processor_dir())
        .expect("redesigned 2D mesh package should load")
}

fn load_example_schedule(name: &str) -> Schedule {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/2d_mesh/schedules")
        .join(name);
    let json = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {path:?}: {error}"));
    serde_json::from_str(&json).unwrap_or_else(|error| panic!("failed to parse {path:?}: {error}"))
}

fn node_scenarios(schedule: &Schedule) -> &[PerfScenario] {
    match schedule {
        Schedule::Func {
            scenarios: Some(scenarios),
            ..
        }
        | Schedule::PlacedFunc {
            scenarios: Some(scenarios),
            ..
        }
        | Schedule::Sequential {
            scenarios: Some(scenarios),
            ..
        }
        | Schedule::Parallel {
            scenarios: Some(scenarios),
            ..
        } => scenarios,
        _ => panic!("expected an evaluated schedule node"),
    }
}

fn build_imperative() -> Architecture {
    mlar_frontend::ArchitectureBuilder::new("system")
        .axis("dram_channel", 8)
        .axis("x", 8)
        .axis("y", 8)
        .memory_definition(MemoryDefinition::new("DRAM", 1_610_612_736, 8192))
        .memory_definition(
            MemoryDefinition::new("L1_R", 1_398_784, 16)
                .with_banking(16)
                .with_technology(MemoryTechnology::new("rram", 1)),
        )
        .memory_definition(
            MemoryDefinition::new("L1_S", 1_398_784, 16)
                .with_banking(16)
                .with_technology(MemoryTechnology::new("sram", 0)),
        )
        .place_memory("DRAM", ["dram_channel"])
        .place_memory("L1_R", ["x", "y"])
        .place_memory("L1_S", ["x", "y"])
        .resource(Resource::exclusive("noc0"))
        .resource(Resource::exclusive("noc1"))
        .resource(
            Resource::exclusive("matrix_lane").indexed(vec![Axis::new("x", 8), Axis::new("y", 8)]),
        )
        .processor_source_dir(processor_dir())
        .processors([
            "matrix_lane",
            "matrix_lane_ss",
            "matrix_lane_sr",
            "matrix_lane_rs",
            "matrix_lane_rr",
            "vector_lane",
            "dram_l1_noc0",
            "l1_l1_noc0",
            "l1_dram_noc1",
        ])
        .connect(
            "matrix_lane",
            Connection::parse_named(
                ["x", "y"],
                [("data", "L1_S[x, y]")],
                [("result", "L1_S[x, y]")],
            )
            .unwrap()
            .with_resources(["matrix_lane"]),
        )
        .connect(
            "matrix_lane_ss",
            Connection::parse_named(
                ["x", "y"],
                [("lhs", "L1_S[x, y]"), ("rhs", "L1_S[x, y]")],
                [("result", "L1_S[x, y]")],
            )
            .unwrap()
            .with_resources(["matrix_lane"]),
        )
        .connect(
            "matrix_lane_sr",
            Connection::parse_named(
                ["x", "y"],
                [("lhs", "L1_S[x, y]"), ("rhs", "L1_R[x, y]")],
                [("result", "L1_S[x, y]")],
            )
            .unwrap()
            .with_resources(["matrix_lane"]),
        )
        .connect(
            "matrix_lane_rs",
            Connection::parse_named(
                ["x", "y"],
                [("lhs", "L1_R[x, y]"), ("rhs", "L1_S[x, y]")],
                [("result", "L1_S[x, y]")],
            )
            .unwrap()
            .with_resources(["matrix_lane"]),
        )
        .connect(
            "matrix_lane_rr",
            Connection::parse_named(
                ["x", "y"],
                [("lhs", "L1_R[x, y]"), ("rhs", "L1_R[x, y]")],
                [("result", "L1_S[x, y]")],
            )
            .unwrap()
            .with_resources(["matrix_lane"]),
        )
        .connect(
            "vector_lane",
            Connection::parse_named(
                ["x", "y"],
                [("data", "L1_S[x, y]")],
                [("result", "L1_S[x, y]")],
            )
            .unwrap(),
        )
        .connect(
            "dram_l1_noc0",
            Connection::parse_named(
                [],
                [("src", "DRAM[:]")],
                [("dst_s", "L1_S[:, :]"), ("dst_r", "L1_R[:, :]")],
            )
            .unwrap()
            .with_resources(["noc0"]),
        )
        .connect(
            "l1_l1_noc0",
            Connection::parse_named([], [("src", "L1_S[:, :]")], [("dst", "L1_S[:, :]")])
                .unwrap()
                .with_resources(["noc0"]),
        )
        .connect(
            "l1_dram_noc1",
            Connection::parse_named([], [("src", "L1_S[:, :]")], [("dst", "DRAM[:]")])
                .unwrap()
                .with_resources(["noc1"]),
        )
        .build()
        .expect("imperative 2D mesh should build")
}

#[test]
fn native_core_fixture_matches_frontend_model_export_and_schedules() {
    let core = core_fixture::scaled_mesh_torus();
    let lowered = load();
    let native_core = core_fixture::single_core();
    let mut core_contract = serde_json::to_value(&core).unwrap();
    let mut lowered_contract = serde_json::to_value(&lowered).unwrap();
    normalize_processor_contracts(&mut core_contract);
    normalize_processor_contracts(&mut lowered_contract);
    assert_eq!(
        core_contract, lowered_contract,
        "native and discovered function contracts differ"
    );
    mlar_rust::architecture_to_mlir_unchecked(&core).unwrap();
    mlar_rust::architecture_to_mlir_unchecked(&lowered).unwrap();
    for name in [
        "core_vector_two_ops.json",
        "core_parallel_vector.json",
        "core_nested_parallel_sequential.json",
        "core_matmul.json",
        "system_data_roundtrip.json",
    ] {
        let schedule = load_example_schedule(name);
        let input = if name.starts_with("core_") {
            &native_core
        } else {
            &core
        };
        let core_result = evaluate(&schedule, input).unwrap();
        let lowered_result = evaluate(&schedule, &lowered).unwrap();
        assert_eq!(
            serde_json::to_value(core_result).unwrap(),
            serde_json::to_value(lowered_result).unwrap(),
            "{name}: core and frontend schedule evaluation differ",
        );
    }
}

fn normalize_processor_contracts(value: &mut serde_json::Value) {
    for definition in value["processor_definitions"].as_array_mut().unwrap() {
        definition.as_object_mut().unwrap().remove("source");
        let functions = definition["functions"].as_array_mut().unwrap();
        for operation in functions.iter_mut() {
            normalize_memref_names(&mut operation["func"]["mlir_details"]);
        }
        functions.sort_by_key(|operation| operation["func"]["name"].as_str().unwrap().to_string());
    }
}

fn normalize_memref_names(details: &mut serde_json::Value) {
    let Some(object) = details.as_object_mut() else {
        return;
    };
    let names = object
        .get("memref_args")
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
        .map(|(index, name)| (name.as_str().unwrap().to_string(), format!("arg{index}")))
        .collect::<BTreeMap<_, _>>();
    fn rename(value: &mut serde_json::Value, names: &BTreeMap<String, String>) {
        match value {
            serde_json::Value::String(name) => {
                if let Some(canonical) = names.get(name) {
                    *name = canonical.clone();
                }
            }
            serde_json::Value::Array(values) => {
                for value in values {
                    rename(value, names);
                }
            }
            serde_json::Value::Object(fields) => {
                for value in fields.values_mut() {
                    rename(value, names);
                }
            }
            _ => {}
        }
    }
    rename(details, &names);
}

#[test]
fn loads_the_heterogeneous_2d_mesh_architecture() {
    let architecture = load();
    assert_eq!(architecture.name(), "system");

    let dram = architecture.memory("DRAM").expect("DRAM array");
    let l1_s = architecture.memory("L1_S").expect("L1_S array");
    let l1_r = architecture.memory("L1_R").expect("L1_R array");
    assert_eq!(dram.instances(), 8);
    assert_eq!(l1_s.instances(), 64);
    assert_eq!(l1_r.instances(), 64);
    assert_eq!(
        architecture.memory_definition(dram).unwrap().capacity,
        1_610_612_736
    );
    let l1_definition = architecture.memory_definition(l1_s).unwrap();
    assert_eq!(l1_definition.capacity, 1_398_784);
    assert_eq!(l1_definition.word_size, 16);
    assert_eq!(l1_definition.banking.as_ref().unwrap().banks, 16);

    // The mesh-wide selection is an endpoint on the movers, not a named alias.
    let mesh_wide = architecture
        .processor_array("dram_l1_noc0")
        .expect("dram_l1_noc0 array")
        .connection()
        .outputs
        .first()
        .expect("one output")
        .clone();
    assert_eq!(mesh_wide.endpoint.memory, "L1_S");
    assert_eq!(
        mesh_wide.endpoint.indices,
        [EndpointIndex::All, EndpointIndex::All]
    );

    assert_eq!(architecture.processor_definitions().len(), 9);
    assert_eq!(architecture.processors().len(), 9);
    for processor in architecture.processors() {
        let expected_instances = match processor.definition_name() {
            "matrix_lane" | "matrix_lane_ss" | "matrix_lane_sr" | "matrix_lane_rs"
            | "matrix_lane_rr" | "vector_lane" => 64,
            "dram_l1_noc0" | "l1_l1_noc0" | "l1_dram_noc1" => 1,
            other => panic!("unexpected processor {other}"),
        };
        assert_eq!(
            processor.instances(&architecture).len(),
            expected_instances,
            "{}",
            processor.name()
        );
    }

    let noc0_users = architecture
        .processors()
        .iter()
        .filter(|processor| {
            processor
                .resources()
                .iter()
                .any(|resource| resource.name() == "noc0")
        })
        .map(|processor| processor.definition_name())
        .collect::<Vec<_>>();
    assert_eq!(noc0_users, ["dram_l1_noc0", "l1_l1_noc0"]);
}

#[test]
fn resolved_sources_preserve_the_processor_catalog() {
    let architecture = load();
    let mut functions = architecture
        .processor_definitions()
        .iter()
        .flat_map(|definition| definition.operations())
        .map(|function| function.func.name.as_str())
        .collect::<Vec<_>>();
    functions.sort();
    let mut expected = vec![
        "matmul_f16",
        "matmul_f16",
        "matmul_f16",
        "matmul_f16",
        "batch_matmul_f16",
        "batch_matmul_f16",
        "batch_matmul_f16",
        "batch_matmul_f16",
        "vec_vsum_f16",
        "vec_vmax_f16",
        "vec_max1_f16",
        "elementwise_add_f16",
        "elementwise_mul_f16",
        "vec_max_f16",
        "vec_exp_f16",
        "vec_sum_f16",
        "vec_add_f16",
        "vec_mul_f16",
        "vec_div_f16",
        "vec_sub_f16",
        "vec_powf_f16",
        "vec_cmpf_ogt_f16",
        "vec_select_f16",
        "vec_log_f16",
        "dram_to_l1_S_f16",
        "dram_to_l1_S_bcst",
        "dram_to_l1_R_f16",
        "dram_to_l1_R_bcst",
        "l1_gather",
        "l1_to_dram_f16",
    ];
    expected.sort();
    assert_eq!(functions, expected);
}

#[test]
fn shared_matmul_source_specializes_into_four_capabilities() {
    let authored = mlar_rust::mlir::MlirModule::from_mlir(
        processor_dir()
            .join("matrix_ops.mlir")
            .display()
            .to_string(),
    )
    .unwrap();
    assert_eq!(
        authored
            .functions
            .iter()
            .map(|function| function.name.as_str())
            .collect::<Vec<_>>(),
        ["matmul_f16", "batch_matmul_f16"]
    );

    let architecture = load();
    for (name, lhs_kind, rhs_kind, scenarios) in [
        ("matrix_lane_ss", None, None, 2),
        ("matrix_lane_sr", None, Some(1), 1),
        ("matrix_lane_rs", Some(1), None, 1),
        ("matrix_lane_rr", Some(1), Some(1), 1),
    ] {
        let definition = architecture.processor_definition(name).unwrap();
        let function = definition.get_function("matmul_f16").unwrap();
        assert_eq!(function.perf.scenarios.len(), scenarios);
        let types = function
            .func
            .mlir_details
            .as_ref()
            .unwrap()
            .memref_arg_types
            .iter()
            .map(|(_, ty)| ty.as_str())
            .collect::<Vec<_>>();
        let expected = |kind| match kind {
            Some(kind) => format!("memref<?x?xf16,{kind}>"),
            None => "memref<?x?xf16>".to_string(),
        };
        assert_eq!(
            types,
            [expected(lhs_kind), expected(rhs_kind), expected(None)]
        );
        assert!(
            architecture
                .processor_array(name)
                .unwrap()
                .resources()
                .iter()
                .any(|resource| resource.name() == "matrix_lane")
        );
    }
}

#[test]
fn emitted_matmul_capabilities_retain_shared_source_provenance() {
    let output =
        std::env::temp_dir().join(format!("mlar-matrix-specialization-{}", std::process::id()));
    if output.exists() {
        std::fs::remove_dir_all(&output).unwrap();
    }
    mlar_frontend::emit_processor_sources(processor_dir(), &output).unwrap();
    let manifest = std::fs::read_to_string(output.join("manifest.yaml")).unwrap();
    assert_eq!(manifest.matches("path: matrix_ops.mlir").count(), 8);
    assert!(
        std::fs::read_to_string(output.join("matrix_lane_sr.mlir"))
            .unwrap()
            .contains("%arg1: memref<?x?xf16, 1>")
    );
    assert!(
        std::fs::read_to_string(output.join("matrix_lane_rs.mlir"))
            .unwrap()
            .contains("%arg0: memref<?x?xf16, 1>")
    );
    std::fs::remove_dir_all(output).unwrap();
}

#[test]
fn declarative_and_imperative_memory_resolution_agree() {
    let declarative = load();
    let imperative = build_imperative();
    assert_eq!(
        serde_json::to_value(&declarative).unwrap(),
        serde_json::to_value(&imperative).unwrap(),
        "declarative and imperative canonical architectures differ"
    );

    let declarative_mlir =
        architecture_to_mlir(&declarative).expect("declarative 2D mesh should export");
    let imperative_mlir =
        architecture_to_mlir(&imperative).expect("imperative 2D mesh should export");
    assert_eq!(
        declarative_mlir, imperative_mlir,
        "declarative and imperative exports differ"
    );

    assert_eq!(
        adl_contract(&declarative_mlir),
        adl_contract(&imperative_mlir)
    );
}

#[test]
fn schedule_uses_the_restored_processor_performance_models() {
    let architecture = load();
    let function = architecture
        .processor_definition("matrix_lane_ss")
        .unwrap()
        .get_function("matmul_f16")
        .expect("SS matmul function")
        .func
        .clone();
    let evaluated = evaluate(
        &Schedule::PlacedFunc {
            func: function,
            target: ProcessorTarget::array("matrix_lane_ss"),
            scenarios: None,
        },
        &architecture,
    )
    .expect("schedule should evaluate");
    let Schedule::PlacedFunc {
        scenarios: Some(scenarios),
        ..
    } = evaluated
    else {
        panic!("expected evaluated function");
    };
    let cost = scenarios[0].time_cost.to_expr().substitute(&[
        (Sym::new("M"), Expr::Const(64)),
        (Sym::new("N"), Expr::Const(128)),
        (Sym::new("K"), Expr::Const(32)),
    ]);
    assert!(cost.eval_const().is_some());
}

#[test]
fn evaluates_main_schedule_examples_on_the_redesigned_architecture() {
    let architecture = load();
    let bindings = [
        (Sym::new("BM"), Expr::Const(32)),
        (Sym::new("BN"), Expr::Const(32)),
    ];

    let vector = evaluate(
        &load_example_schedule("core_vector_two_ops.json"),
        &architecture,
    )
    .unwrap();
    assert_eq!(
        node_scenarios(&vector)[0]
            .time_cost
            .to_expr()
            .substitute(&bindings)
            .eval_const(),
        Some(4)
    );

    let parallel = evaluate(
        &load_example_schedule("core_parallel_vector.json"),
        &architecture,
    )
    .unwrap();
    assert_eq!(
        node_scenarios(&parallel)[0]
            .time_cost
            .to_expr()
            .substitute(&bindings)
            .eval_const(),
        Some(147)
    );

    let nested = evaluate(
        &load_example_schedule("core_nested_parallel_sequential.json"),
        &architecture,
    )
    .unwrap();
    assert_eq!(
        node_scenarios(&nested)[0]
            .time_cost
            .to_expr()
            .substitute(&bindings)
            .eval_const(),
        Some(149)
    );

    let matmul = evaluate(&load_example_schedule("core_matmul.json"), &architecture).unwrap();
    assert_eq!(node_scenarios(&matmul).len(), 2);
    for scenario in node_scenarios(&matmul) {
        let mut symbols = scenario.time_cost.to_expr().free_symbols();
        symbols.extend(scenario.constraints.free_symbols());
        assert!(!symbols.contains(&Sym::new("M")));
        assert!(!symbols.contains(&Sym::new("N")));
        assert!(!symbols.contains(&Sym::new("K")));
        assert!(symbols.contains(&Sym::new("BM")));
        assert!(symbols.contains(&Sym::new("BN")));
        assert!(symbols.contains(&Sym::new("BK")));
    }

    let roundtrip = evaluate(
        &load_example_schedule("system_data_roundtrip.json"),
        &architecture,
    )
    .unwrap();
    assert_eq!(
        node_scenarios(&roundtrip)[0]
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
fn missing_type_is_a_specific_export_error() {
    let architecture = load().with_processor_type("matrix_lane", None).unwrap();
    let error = architecture_to_mlir(&architecture).expect_err("untyped export must fail");
    assert!(matches!(error, AdlExportError::MissingProcessorType { .. }));
}

// Called by the Loom build to produce tests/2d_mesh/bin/eval_system.
#[test]
fn test_generate_system_evaluator_binary() {
    let architecture = load();
    let output_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/2d_mesh/bin");
    let binary = generate_evaluator_binary(&architecture, "eval_system", &output_dir)
        .expect("system evaluator binary should build");
    assert!(binary.is_file(), "no binary at {binary:?}");

    let function = architecture
        .processor_definition("matrix_lane_ss")
        .unwrap()
        .get_function("matmul_f16")
        .expect("SS matmul function")
        .func
        .clone();
    let schedule = Schedule::PlacedFunc {
        func: function,
        target: ProcessorTarget::array("matrix_lane_ss"),
        scenarios: None,
    };

    let mut child = Command::new(&binary)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("generated evaluator should run");
    child
        .stdin
        .take()
        .expect("stdin was piped")
        .write_all(serde_json::to_string(&schedule).unwrap().as_bytes())
        .expect("evaluator should accept a schedule");
    let output = child.wait_with_output().expect("evaluator should exit");
    assert!(
        output.status.success(),
        "evaluator failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // The embedded architecture must evaluate exactly as the in-process one does.
    let from_binary: Schedule =
        serde_json::from_slice(&output.stdout).expect("evaluator should emit a Schedule");
    let in_process = evaluate(&schedule, &architecture).expect("schedule should evaluate");
    assert_eq!(
        serde_json::to_value(&from_binary).unwrap(),
        serde_json::to_value(&in_process).unwrap(),
        "generated binary disagrees with the library evaluator"
    );
}

#[derive(Debug, PartialEq, Eq)]
struct AdlContract {
    root_module: String,
    dimension_sizes: Vec<u64>,
    memories: Vec<MemoryNode>,
    resources: Vec<String>,
    processors: Vec<ProcessorContract>,
    functions: Vec<FunctionContract>,
    compose_count: usize,
    scale_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum MemoryNode {
    Bank {
        block_size: u64,
        blocks: u64,
    },
    Array {
        dimensions: Vec<u64>,
        element: Box<MemoryNode>,
    },
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ProcessorContract {
    name: String,
    kind: String,
    input: MemoryNode,
    output: MemoryNode,
    resources: Vec<String>,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct FunctionContract {
    name: String,
    signature: String,
    symbols: Vec<String>,
    shapes: Vec<String>,
    bindings: Vec<String>,
    operation: String,
}

fn adl_contract(mlir: &str) -> AdlContract {
    let root_module = mlir
        .lines()
        .find_map(|line| line.trim().strip_prefix("module @arch_"))
        .and_then(|line| line.split_whitespace().next())
        .expect("architecture module")
        .to_string();
    let mut dimensions = BTreeMap::new();
    let mut memory_ssa = BTreeMap::new();
    let mut memory_symbols = BTreeMap::new();
    let mut memories = Vec::new();
    let mut resource_ssa = BTreeMap::new();
    let mut resources = Vec::new();
    let mut processors = Vec::new();

    for line in mlir.lines().map(str::trim) {
        if line.contains(" = adl.spatial_dim ") {
            dimensions.insert(ssa_result(line), trailing_u64(line));
        } else if line.contains(" = adl.memory.bank ") {
            let block_size = field_u64(line, "bsize = ");
            let blocks = field_u64(line, "nblk = ");
            let node = MemoryNode::Bank { block_size, blocks };
            memory_symbols.insert(quoted_name(line), node.clone());
            memory_ssa.insert(ssa_result(line), node);
        } else if line.contains(" = adl.memory.array ") {
            let dimensions_list = bracket_contents(line)
                .split(',')
                .filter(|value| !value.trim().is_empty())
                .map(|value| dimensions[value.trim()])
                .collect::<Vec<_>>();
            let element_ssa = line
                .split_once(" of ")
                .expect("memory array element")
                .1
                .split_whitespace()
                .next()
                .unwrap();
            let node = MemoryNode::Array {
                dimensions: dimensions_list,
                element: Box::new(memory_ssa[element_ssa].clone()),
            };
            memory_symbols.insert(quoted_name(line), node.clone());
            memory_ssa.insert(ssa_result(line), node.clone());
            memories.push(node);
        } else if line.contains(" = adl.resource.") {
            let name = quoted_name(line);
            resource_ssa.insert(ssa_result(line), name.clone());
            resources.push(name);
        } else if line.contains(" = adl.processor.") {
            let kind = line
                .split_once("adl.processor.")
                .unwrap()
                .1
                .split_whitespace()
                .next()
                .unwrap()
                .to_string();
            let name = line
                .split('@')
                .nth(1)
                .unwrap()
                .split(',')
                .next()
                .unwrap()
                .to_string();
            let route = line.split_once("from ").unwrap().1;
            let input_ssa = route.split_whitespace().next().unwrap();
            let output_ssa = route
                .split_once(" to ")
                .unwrap()
                .1
                .split([',', ' '])
                .find(|value| !value.is_empty())
                .unwrap();
            let processor_resources = line
                .split_once("with [")
                .map(|(_, values)| {
                    values
                        .trim_end_matches(']')
                        .split(',')
                        .filter(|value| !value.trim().is_empty())
                        .map(|value| resource_ssa[value.trim()].clone())
                        .collect()
                })
                .unwrap_or_default();
            processors.push(ProcessorContract {
                name,
                kind,
                input: memory_ssa[input_ssa].clone(),
                output: memory_ssa[output_ssa].clone(),
                resources: processor_resources,
            });
        }
    }
    let mut dimension_sizes = dimensions.values().copied().collect::<Vec<_>>();
    dimension_sizes.sort_unstable();
    memories.sort();
    resources.sort();
    processors.sort();
    let mut functions = function_contracts(mlir, &memory_symbols);
    functions.sort();
    AdlContract {
        root_module,
        dimension_sizes,
        memories,
        resources,
        processors,
        functions,
        compose_count: mlir.matches("adl.arch.compose").count(),
        scale_count: mlir.matches("adl.arch.scale").count(),
    }
}

/// Memory level symbols are axis-derived, so the oracle compares the memory
/// each binding *names structurally* rather than how the exporter spells it.
fn canonical_memory_symbols(text: &str, symbols: &BTreeMap<String, MemoryNode>) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find("@mem_") {
        out.push_str(&rest[..at]);
        rest = &rest[at + 1..];
        let end = rest
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(rest.len());
        let (symbol, tail) = rest.split_at(end);
        out.push('@');
        out.push_str(&match symbols.get(symbol) {
            Some(node) => memory_shape_token(node),
            None => symbol.to_string(),
        });
        rest = tail;
    }
    out.push_str(rest);
    out
}

fn memory_shape_token(node: &MemoryNode) -> String {
    match node {
        MemoryNode::Bank { block_size, blocks } => format!("bank({block_size},{blocks})"),
        MemoryNode::Array {
            dimensions,
            element,
        } => format!("array({dimensions:?},{})", memory_shape_token(element)),
    }
}

fn function_contracts(
    mlir: &str,
    memory_symbols: &BTreeMap<String, MemoryNode>,
) -> Vec<FunctionContract> {
    let mut contracts = Vec::new();
    let mut cursor = 0;
    while let Some(relative) = mlir[cursor..].find("func.func @") {
        let start = cursor + relative;
        let opening = mlir[start..]
            .find('{')
            .map(|offset| start + offset)
            .unwrap();
        let closing = matching_brace(mlir, opening);
        let header = &mlir[start..opening];
        let body = &mlir[opening + 1..closing];
        let name = header
            .split('@')
            .nth(1)
            .unwrap()
            .split('(')
            .next()
            .unwrap()
            .to_string();
        let mut shapes = selected_lines(body, "loom.bind_shape");
        let mut bindings = selected_lines(body, "loom.bind_mem")
            .iter()
            .map(|line| canonical_memory_symbols(line, memory_symbols))
            .collect::<Vec<_>>();
        shapes.sort();
        bindings.sort();
        let operation_start = ["linalg.", "loom.copy", "loom.gather"]
            .iter()
            .filter_map(|operation| body.find(operation))
            .min()
            .expect("supported function operation");
        let operation_end = body[operation_start..]
            .rfind("return")
            .map(|offset| operation_start + offset)
            .expect("function return");
        let mut symbols = selected_lines(body, "loom.sym");
        symbols.sort();
        contracts.push(FunctionContract {
            name,
            signature: no_whitespace(header),
            symbols,
            shapes,
            bindings,
            operation: no_whitespace(&canonical_memory_symbols(
                &body[operation_start..operation_end],
                memory_symbols,
            )),
        });
        cursor = closing + 1;
    }
    contracts
}

fn selected_lines(text: &str, needle: &str) -> Vec<String> {
    text.lines()
        .filter(|line| line.contains(needle))
        .map(no_whitespace)
        .collect()
}

fn matching_brace(text: &str, opening: usize) -> usize {
    let mut depth = 0;
    for (offset, byte) in text.as_bytes()[opening..].iter().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return opening + offset;
                }
            }
            _ => {}
        }
    }
    panic!("unbalanced MLIR braces");
}

fn no_whitespace(text: &str) -> String {
    text.chars()
        .filter(|character| !character.is_whitespace())
        .collect()
}

fn ssa_result(line: &str) -> &str {
    line.split_whitespace().next().unwrap()
}

fn trailing_u64(line: &str) -> u64 {
    line.rsplit_once(',').unwrap().1.trim().parse().unwrap()
}

fn field_u64(line: &str, field: &str) -> u64 {
    line.split_once(field)
        .unwrap()
        .1
        .split(|character: char| !character.is_ascii_digit())
        .next()
        .unwrap()
        .parse()
        .unwrap()
}

fn quoted_name(line: &str) -> String {
    line.split('"').nth(1).unwrap().to_string()
}

fn bracket_contents(line: &str) -> &str {
    line.split_once('[').unwrap().1.split_once(']').unwrap().0
}
