use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

use mlar_rust::arch::{EndpointIndex, ProcessorYaml};
use mlar_rust::{
    AdlExportError, Architecture, Connection, Expr, MemoryAlias, MemoryDefinition, MemoryEndpoint,
    Resource, Schedule, Sym, architecture_to_mlir, evaluate, generate_evaluator_binary,
};

fn processor_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/2d_mesh/processors")
}

fn load() -> mlar_rust::Architecture {
    mlar_rust::archs::load_arch(processor_dir()).expect("redesigned 2D mesh package should load")
}

fn load_processor_definition(name: &str) -> mlar_rust::ProcessorDefinition {
    let path = processor_dir().join(format!("{name}.yaml"));
    ProcessorYaml::from_file(&path)
        .and_then(|yaml| yaml.build_definition(&path))
        .unwrap_or_else(|error| panic!("processor '{name}' should load: {error}"))
}

fn connection(domain: &[&str], inputs: &[&str], outputs: &[&str]) -> Connection {
    Connection::new(
        domain.iter().copied(),
        inputs
            .iter()
            .map(|endpoint| MemoryEndpoint::parse(endpoint).unwrap())
            .collect(),
        outputs
            .iter()
            .map(|endpoint| MemoryEndpoint::parse(endpoint).unwrap())
            .collect(),
    )
}

fn build_imperative() -> Architecture {
    Architecture::builder("system")
        .axis("dram_channel", 8)
        .axis("x", 8)
        .axis("y", 8)
        .memory_definition(MemoryDefinition::new(
            "DRAM",
            ["dram_channel"],
            1_610_612_736,
            8192,
        ))
        .memory_definition(MemoryDefinition::new("L1", ["x", "y"], 1_398_784, 16).with_banking(16))
        .memory_alias(MemoryAlias::new(
            "all_l1",
            MemoryEndpoint::parse("L1[:, :]").unwrap(),
        ))
        .place_memory("DRAM", ["dram_channel"])
        .place_memory("L1", ["x", "y"])
        .resource(Resource::exclusive("noc0"))
        .resource(Resource::exclusive("noc1"))
        .processor_definition(load_processor_definition("matrix_lane"))
        .processor_definition(load_processor_definition("vector_lane"))
        .processor_definition(load_processor_definition("dram_l1_noc0"))
        .processor_definition(load_processor_definition("l1_l1_noc0"))
        .processor_definition(load_processor_definition("l1_dram_noc1"))
        .connect(
            "matrix_lane",
            "matrix_lane",
            connection(&["x", "y"], &["L1[x, y]"], &["L1[x, y]"]),
        )
        .connect(
            "vector_lane",
            "vector_lane",
            connection(&["x", "y"], &["L1[x, y]"], &["L1[x, y]"]),
        )
        .connect(
            "dram_l1_noc0",
            "dram_l1_noc0",
            connection(&[], &["DRAM[:]"], &["all_l1"]).with_resources(["noc0"]),
        )
        .connect(
            "l1_l1_noc0",
            "l1_l1_noc0",
            connection(&[], &["all_l1"], &["all_l1"]).with_resources(["noc0"]),
        )
        .connect(
            "l1_dram_noc1",
            "l1_dram_noc1",
            connection(&[], &["all_l1"], &["DRAM[:]"]).with_resources(["noc1"]),
        )
        .build()
        .expect("imperative 2D mesh should build")
}

#[test]
fn recreates_the_pre_redesign_2d_mesh_architecture() {
    let architecture = load();
    assert_eq!(architecture.name(), "system");

    let dram = architecture.memory("DRAM").expect("DRAM array");
    let l1 = architecture.memory("L1").expect("L1 array");
    assert_eq!(dram.instances(), 8);
    assert_eq!(l1.instances(), 64);
    assert_eq!(
        architecture.memory_definition(dram).unwrap().capacity,
        1_610_612_736
    );
    let l1_definition = architecture.memory_definition(l1).unwrap();
    assert_eq!(l1_definition.capacity, 1_398_784);
    assert_eq!(l1_definition.word_size, 16);
    assert_eq!(l1_definition.banking.as_ref().unwrap().banks, 16);

    let all_l1 = architecture
        .memory_alias("all_l1")
        .expect("mesh-wide L1 alias");
    assert_eq!(all_l1.endpoint.memory, "L1");
    assert_eq!(
        all_l1.endpoint.indices,
        [EndpointIndex::All, EndpointIndex::All]
    );

    assert_eq!(architecture.processor_definitions().len(), 5);
    assert_eq!(architecture.processors().len(), 5);
    for processor in architecture.processors() {
        let expected_instances = match processor.definition_name() {
            "matrix_lane" | "vector_lane" => 64,
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
fn compact_sources_preserve_the_full_golden_processor_catalog() {
    let architecture = load();
    let functions = architecture
        .processor_definitions()
        .iter()
        .flat_map(|definition| definition.operations())
        .map(|function| function.func.name.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        functions,
        [
            "matmul_SS_f16",
            "matmul_SR_f16",
            "matmul_RS_f16",
            "matmul_RR_f16",
            "batch_matmul_SS_f16",
            "batch_matmul_SR_f16",
            "batch_matmul_RS_f16",
            "batch_matmul_RR_f16",
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
        ]
    );
}

#[test]
fn declarative_imperative_and_pre_redesign_golden_agree() {
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

    let golden = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/2d_mesh/golden/test_golden_ref.mlir"),
    )
    .expect("pre-redesign MLIR golden");

    let golden_contract = adl_contract(&golden);
    assert_eq!(
        adl_contract(&declarative_mlir),
        golden_contract,
        "declarative export differs from the pre-redesign contract"
    );
    assert_eq!(
        adl_contract(&imperative_mlir),
        adl_contract(&golden),
        "imperative export differs from the pre-redesign contract"
    );
}

#[test]
fn schedule_uses_the_restored_processor_performance_models() {
    let architecture = load();
    let function = architecture
        .get_function("matmul_SS_f16")
        .expect("golden matmul function")
        .func
        .clone();
    let evaluated = evaluate(
        &Schedule::Func {
            func: function,
            scenarios: None,
        },
        &architecture,
    )
    .expect("schedule should evaluate");
    let Schedule::Func {
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
fn missing_type_is_a_specific_export_error() {
    let architecture = load().with_processor_type("matrix_lane", None).unwrap();
    let error = architecture_to_mlir(&architecture).expect_err("untyped export must fail");
    assert!(matches!(error, AdlExportError::MissingProcessorType { .. }));
}

/// The Loom monorepo builds its evaluator through this test — `scripts/build-mlar.sh`
/// runs it by name, and `loom/loom_utils/mlar/core.py` then invokes the result at
/// `tests/2d_mesh/bin/eval_system`, feeding it a Schedule on stdin.
#[test]
fn test_generate_system_evaluator_binary() {
    let architecture = load();
    let output_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/2d_mesh/bin");
    let binary = generate_evaluator_binary(&architecture, "eval_system", &output_dir)
        .expect("system evaluator binary should build");
    assert!(binary.is_file(), "no binary at {binary:?}");

    let function = architecture
        .get_function("matmul_SS_f16")
        .expect("golden matmul function")
        .func
        .clone();
    let schedule = Schedule::Func {
        func: function,
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
        name: String,
        dimensions: Vec<u64>,
        element: Box<MemoryNode>,
    },
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ProcessorContract {
    name: String,
    kind: String,
    input: String,
    output: String,
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
            memory_ssa.insert(ssa_result(line), node);
        } else if line.contains(" = adl.memory.array ") {
            let name = quoted_name(line);
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
                name,
                dimensions: dimensions_list,
                element: Box::new(memory_ssa[element_ssa].clone()),
            };
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
                input: memory_name(&memory_ssa[input_ssa]),
                output: memory_name(&memory_ssa[output_ssa]),
                resources: processor_resources,
            });
        }
    }
    let mut dimension_sizes = dimensions.values().copied().collect::<Vec<_>>();
    dimension_sizes.sort_unstable();
    memories.sort();
    resources.sort();
    processors.sort();
    let mut functions = function_contracts(mlir);
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

fn function_contracts(mlir: &str) -> Vec<FunctionContract> {
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
        let mut bindings = selected_lines(body, "loom.bind_mem");
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
            operation: no_whitespace(&body[operation_start..operation_end]),
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

fn memory_name(memory: &MemoryNode) -> String {
    match memory {
        MemoryNode::Bank { .. } => "<bank>".into(),
        MemoryNode::Array { name, .. } => name.clone(),
    }
}
