use mlar_frontend::Connection;
use mlar_frontend::ProcessorDefinition;
use mlar_frontend::selection::MemoryEndpoint;
use std::path::{Path, PathBuf};

use mlar_rust::arch::EndpointIndex;

use mlar_rust::{
    AdlExportError, Architecture, MemoryDefinition, ProcessorType, Schedule, architecture_to_mlir,
    evaluate,
};

/// Examples the current `adl.*` dialect can lower and validate.
const LOWERABLE: &[&str] = &[
    "single-core",
    "mesh-torus",
    "heterogeneous-lanes",
    "dual-noc-mesh",
    "shared-link-mesh",
];

#[path = "support/cache_hierarchy.rs"]
mod imperative_cache_hierarchy;
#[path = "support/dual_noc_mesh.rs"]
mod imperative_dual_noc_mesh;
#[path = "support/shared_link_mesh.rs"]
mod imperative_shared_link_mesh;

#[allow(dead_code)]
#[path = "../../examples/cache_hierarchy/main.rs"]
mod core_cache_hierarchy;
#[allow(dead_code)]
#[path = "../../examples/dual_noc_mesh/main.rs"]
mod core_dual_noc_mesh;
#[allow(dead_code)]
#[path = "../../examples/mesh_torus/main.rs"]
mod core_mesh_torus;
#[allow(dead_code)]
#[path = "../../examples/shared_link_mesh/main.rs"]
mod core_shared_link_mesh;
#[allow(dead_code)]
#[path = "../../examples/single_core/main.rs"]
mod core_single_core;

fn example_dir(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples/declarative")
        .join(name)
}

#[test]
fn all_architecture_examples_load_and_export() {
    for name in [
        "single-core",
        "cache-hierarchy",
        "mesh-torus",
        "heterogeneous-lanes",
        "dual-noc-mesh",
        "shared-link-mesh",
    ] {
        let architecture = mlar_frontend::archs::load_arch(example_dir(name))
            .unwrap_or_else(|error| panic!("example '{name}' should load: {error}"));
        if LOWERABLE.contains(&name) {
            let mlir = architecture_to_mlir(&architecture)
                .unwrap_or_else(|error| panic!("example '{name}' should export: {error}"));
            assert!(mlir.starts_with("module @arch_system {"));
        }

        let function = architecture
            .processor_definitions()
            .first()
            .and_then(|definition| definition.operations().first())
            .expect("example should contain a function")
            .func
            .clone();
        evaluate(
            &Schedule::Func {
                func: function,
                scenarios: None,
            },
            &architecture,
        )
        .expect("example function should evaluate");
    }
}

#[test]
fn heterogeneous_templates_derive_spaces_from_connected_operand_positions() {
    let architecture = mlar_frontend::load_arch(example_dir("heterogeneous-lanes")).unwrap();
    let matrix = architecture
        .processor_definitions()
        .iter()
        .find(|definition| definition.name() == "matrix_lane")
        .unwrap()
        .source();
    assert!(matrix.contains("%lhs: memref<?x?xf16, 0>"));
    assert!(matrix.contains("%rhs: memref<?x?xf16, 1>"));
    assert!(matrix.contains("loom.bind_mem %lhs, @input_0"));
    assert!(matrix.contains("loom.bind_mem %rhs, @input_1"));

    let vector = architecture
        .processor_definitions()
        .iter()
        .find(|definition| definition.name() == "vector_lane")
        .unwrap()
        .source();
    assert!(vector.contains("%lhs: memref<?xf16, 1>"));
    assert!(vector.contains("%rhs: memref<?xf16, 0>"));
}

#[test]
fn hierarchy_normalizes_to_flat_memory_and_reports_unsupported_adl_slice() {
    let architecture = mlar_frontend::load_arch(example_dir("cache-hierarchy")).unwrap();
    assert_eq!(architecture.memory("L1").unwrap().rank(), 2);
    assert!(matches!(
        architecture_to_mlir(&architecture),
        Err(AdlExportError::UnsupportedMemorySelection { .. })
    ));
}

#[test]
fn sibling_memories_share_one_scale_via_the_enclosing_composition() {
    let architecture = mlar_frontend::ArchitectureBuilder::new("siblings")
        .axis("x", 2)
        .memory_definition(MemoryDefinition::new("sram", 1024, 16))
        .memory_definition(MemoryDefinition::new("rram", 1024, 16))
        .place_memory("sram", ["x"])
        .place_memory("rram", ["x"])
        .processor_definition(
            ProcessorDefinition::new("lane", "", Vec::new()).with_type(ProcessorType::Compute),
        )
        .connect(
            "lane",
            Connection::new(
                ["x"],
                vec![MemoryEndpoint::parse("sram[x]").unwrap()],
                vec![MemoryEndpoint::parse("rram[x]").unwrap()],
            ),
        )
        .build()
        .expect("sibling architecture is valid in the runtime model");

    let mlir = architecture_to_mlir(&architecture).expect("sibling memories should export");
    let sram = mlir
        .lines()
        .find(|line| line.contains("adl.memory.array \"mem_sram\""))
        .unwrap()
        .trim()
        .split(" = ")
        .next()
        .unwrap();
    let rram = mlir
        .lines()
        .find(|line| line.contains("adl.memory.array \"mem_rram\""))
        .unwrap()
        .trim()
        .split(" = ")
        .next()
        .unwrap();
    let scale = mlir
        .lines()
        .find(|line| line.contains("adl.arch.scale \"arch_x\""))
        .unwrap();
    let root = mlir
        .lines()
        .find(|line| line.contains("adl.arch.compose \"arch_siblings\""))
        .unwrap();
    assert!(!scale.contains("mem_region"));
    assert!(root.contains(&format!("mem[{sram}, {rram}]")), "{root}");
}

#[test]
fn frontend_builder_matches_declarative_packages() {
    assert_imperative_matches(
        "dual-noc-mesh",
        imperative_dual_noc_mesh::build().expect("imperative dual-NoC mesh should build"),
    );
    assert_imperative_matches(
        "cache-hierarchy",
        imperative_cache_hierarchy::build().expect("imperative cache hierarchy should build"),
    );
    assert_imperative_matches(
        "shared-link-mesh",
        imperative_shared_link_mesh::build().expect("imperative shared-link mesh should build"),
    );
}

#[test]
fn core_single_core_matches_frontend() {
    assert_core_matches("single-core", core_single_core::build().unwrap());
}

#[test]
fn core_cache_hierarchy_matches_frontend() {
    assert_core_matches("cache-hierarchy", core_cache_hierarchy::build().unwrap());
}

#[test]
fn core_mesh_torus_matches_frontend() {
    assert_core_matches("mesh-torus", core_mesh_torus::build().unwrap());
}

#[test]
fn core_dual_noc_mesh_remains_a_valid_native_example() {
    core_dual_noc_mesh::build().unwrap().validate().unwrap();
}

#[test]
fn core_shared_link_mesh_matches_frontend() {
    assert_core_matches("shared-link-mesh", core_shared_link_mesh::build().unwrap());
}

fn assert_core_matches(name: &str, core: Architecture) {
    let lowered = mlar_frontend::load_arch(example_dir(name)).unwrap();
    let mut core_contract = serde_json::to_value(&core).unwrap();
    let mut lowered_contract = serde_json::to_value(&lowered).unwrap();
    normalize_processor_contracts(&mut core_contract);
    normalize_processor_contracts(&mut lowered_contract);
    assert_eq!(
        core_contract, lowered_contract,
        "{name}: architecture, function interfaces, or performance differ"
    );
    if LOWERABLE.contains(&name) {
        mlar_rust::architecture_to_mlir_unchecked(&core).unwrap();
        mlar_rust::architecture_to_mlir_unchecked(&lowered).unwrap();
        if mlar_rust::mlir_validators_available() {
            architecture_to_mlir(&core).unwrap();
            architecture_to_mlir(&lowered).unwrap();
        }
    } else {
        for architecture in [&core, &lowered] {
            assert!(matches!(
                mlar_rust::architecture_to_mlir_unchecked(architecture),
                Err(AdlExportError::UnsupportedMemorySelection { .. })
            ));
        }
    }
}

fn normalize_processor_contracts(value: &mut serde_json::Value) {
    let definitions = value["processor_definitions"].as_array_mut().unwrap();
    for definition in definitions {
        definition.as_object_mut().unwrap().remove("source");
        let functions = definition["functions"].as_array_mut().unwrap();
        for operation in functions.iter_mut() {
            let function = &mut operation["func"];
            function["symbols"]
                .as_array_mut()
                .unwrap()
                .sort_by_key(|symbol| symbol.as_str().unwrap().to_string());
            if let Some(details) = function["mlir_details"].as_object_mut() {
                details.remove("linalg_ops");
                details.remove("operations");
                details.remove("copy_ops");
                details.remove("gather_ops");
                canonicalize_memref_names(details);
            }
        }
        functions.sort_by_key(|operation| operation["func"]["name"].as_str().unwrap().to_string());
    }
}

fn canonicalize_memref_names(details: &mut serde_json::Map<String, serde_json::Value>) {
    let names = details["memref_args"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(index, name)| {
            (
                name.as_str().unwrap().to_string(),
                format!("operand_{index}"),
            )
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    let rename = |value: &mut serde_json::Value| {
        if let Some(name) = value.as_str()
            && let Some(canonical) = names.get(name)
        {
            *value = canonical.clone().into();
        }
    };
    for field in ["memref_args", "source_memrefs", "target_memrefs"] {
        for name in details[field].as_array_mut().unwrap() {
            rename(name);
        }
    }
    for entry in details["memref_arg_types"].as_array_mut().unwrap() {
        rename(&mut entry.as_array_mut().unwrap()[0]);
    }
    for field in ["memref_symbol_bindings", "mem_region_bindings"] {
        for entry in details[field].as_array_mut().unwrap() {
            rename(&mut entry.as_object_mut().unwrap()["memref"]);
        }
    }
}

// Named placements may share one definition.
#[test]
fn one_definition_can_back_several_named_placements() {
    let architecture = mlar_frontend::archs::load_arch(example_dir("shared-link-mesh"))
        .expect("shared-link-mesh should load");

    let links = architecture
        .processors()
        .iter()
        .filter(|processor| processor.definition_name() == "link_dma")
        .map(|processor| processor.name())
        .collect::<Vec<_>>();
    assert_eq!(
        links,
        ["east_link", "west_link", "north_link", "south_link"]
    );

    assert_eq!(
        architecture.processor_definitions().len(),
        2,
        "the four link placements must share one registered definition"
    );
}

fn assert_imperative_matches(name: &str, imperative: Architecture) {
    let declarative = mlar_frontend::archs::load_arch(example_dir(name))
        .unwrap_or_else(|error| panic!("declarative example '{name}' should load: {error}"));
    let declarative_json = serde_json::to_value(&declarative).unwrap();
    let imperative_json = serde_json::to_value(&imperative).unwrap();
    if declarative_json != imperative_json {
        panic!(
            "{name} canonical architectures differ first at {}",
            first_json_difference(&declarative_json, &imperative_json, "$")
                .unwrap_or_else(|| "an unknown location".into())
        );
    }
    if LOWERABLE.contains(&name) {
        assert_eq!(
            architecture_to_mlir(&declarative).unwrap(),
            architecture_to_mlir(&imperative).unwrap(),
            "{name} exports differ"
        );
    }
}

fn first_json_difference(
    left: &serde_json::Value,
    right: &serde_json::Value,
    path: &str,
) -> Option<String> {
    match (left, right) {
        (serde_json::Value::Array(left), serde_json::Value::Array(right)) => {
            if left.len() != right.len() {
                return Some(format!("{path}.length: {} != {}", left.len(), right.len()));
            }
            left.iter()
                .zip(right)
                .enumerate()
                .find_map(|(index, (left, right))| {
                    first_json_difference(left, right, &format!("{path}[{index}]"))
                })
        }
        (serde_json::Value::Object(left), serde_json::Value::Object(right)) => {
            if left.keys().collect::<Vec<_>>() != right.keys().collect::<Vec<_>>() {
                return Some(format!("{path}.keys differ"));
            }
            left.iter().find_map(|(key, left)| {
                first_json_difference(left, &right[key], &format!("{path}.{key}"))
            })
        }
        _ if left != right => Some(format!("{path}: {left} != {right}")),
        _ => None,
    }
}

#[test]
fn examples_use_the_canonical_model() {
    let architecture = mlar_frontend::archs::load_arch(example_dir("dual-noc-mesh"))
        .expect("dual-NoC example should load");
    assert!(!architecture.memories().is_empty());
    assert!(!architecture.processor_definitions().is_empty());
    assert!(!architecture.processors().is_empty());
    assert!(
        architecture
            .processors()
            .iter()
            .all(|processor| !processor.instances(&architecture).is_empty())
    );
}

#[test]
fn dual_noc_connects_system_movers_to_the_mesh_wide_l1_region() {
    let architecture = mlar_frontend::archs::load_arch(example_dir("dual-noc-mesh"))
        .expect("dual-NoC example should load");
    let noc_processors = architecture
        .processors()
        .iter()
        .filter(|processor| {
            matches!(
                processor.definition_name(),
                "dram_l1_noc0" | "l1_l1_noc0" | "l1_dram_noc1"
            )
        })
        .collect::<Vec<_>>();

    assert_eq!(noc_processors.len(), 3);
    assert!(
        noc_processors
            .iter()
            .all(|processor| processor.axes().is_empty())
    );
    assert!(noc_processors.iter().all(|processor| {
        processor
            .connection()
            .inputs
            .iter()
            .chain(&processor.connection().outputs)
            .any(|endpoint| {
                matches!(endpoint.memory.as_str(), "L1_S" | "L1_R")
                    && endpoint.indices.len() == 2
                    && endpoint
                        .indices
                        .iter()
                        .all(|index| matches!(index, EndpointIndex::All))
            })
    }));

    let mlir = architecture_to_mlir(&architecture).expect("dual-NoC should export");
    assert!(!mlir.contains("adl.arch.scale \"arch_dram_l1_noc0\""));
    assert!(!mlir.contains("adl.arch.scale \"arch_l1_l1_noc0\""));
    assert!(!mlir.contains("adl.arch.scale \"arch_l1_dram_noc1\""));
}

#[test]
fn mesh_torus_retains_queryable_wraparound_links() {
    let architecture = mlar_frontend::archs::load_arch(example_dir("mesh-torus"))
        .expect("mesh-torus example should load");
    let torus = architecture
        .networks()
        .iter()
        .find(|network| network.name == "l1_torus")
        .expect("explicit L1 torus");
    assert_eq!(torus.edges().len(), 4 * 4 * 4);
    let route = torus
        .shortest_route(&[3, 1], &[0, 1])
        .expect("east wraparound route");
    assert_eq!(route.len(), 1);
    assert_eq!(route[0].link, "east");
    assert_eq!(route[0].resource_indices, [3, 1]);
}

#[test]
fn examples_match_pre_redesign_adl_contracts() {
    let contracts = [
        (
            "single-core",
            &["{bsize = 64, nblk = 1024}"][..],
            &["vector_add"][..],
            1,
            0,
            1,
        ),
        // cache-hierarchy has no ADL contract: see
        // `multi_region_levels_are_reported_as_unlowerable`.
        (
            "mesh-torus",
            &["{bsize = 4096, nblk = 65536}", "{bsize = 64, nblk = 512}"][..],
            &["matmul", "load_l1", "load_l1_broadcast", "writeback_dram"][..],
            3,
            1,
            3,
        ),
        (
            "dual-noc-mesh",
            &["{bsize = 8192, nblk = 196608}", "{bsize = 16, nblk = 5464}"][..],
            &[
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
            ][..],
            5,
            1,
            5,
        ),
    ];

    for (name, banks, functions, processors, scales, arrays) in contracts {
        let architecture = mlar_frontend::archs::load_arch(example_dir(name)).unwrap();
        let mlir = architecture_to_mlir(&architecture).unwrap();
        for bank in banks {
            assert!(
                mlir.contains(bank),
                "{name} is missing bank geometry {bank}"
            );
        }
        for function in functions {
            assert!(
                mlir.contains(&format!("func.func @{function}")),
                "{name} is missing function {function}"
            );
        }
        assert_eq!(mlir.matches("adl.processor.").count(), processors, "{name}");
        assert_eq!(mlir.matches("adl.arch.scale").count(), scales, "{name}");
        assert_eq!(mlir.matches("adl.memory.array").count(), arrays, "{name}");
    }

    let dual = architecture_to_mlir(
        &mlar_frontend::archs::load_arch(example_dir("dual-noc-mesh")).unwrap(),
    )
    .unwrap();
    let noc0_load = processor_line(&dual, "@proc_dram_l1_noc0");
    let noc0_gather = processor_line(&dual, "@proc_l1_l1_noc0");
    assert_eq!(resource_clause(noc0_load), resource_clause(noc0_gather));
    assert!(dual.contains("area: [%bcst_x, %bcst_y]"));
    assert!(dual.contains("dst_mem_space @mem_L1_R"));
    assert!(dual.contains("loom.gather"));

    let mesh =
        architecture_to_mlir(&mlar_frontend::archs::load_arch(example_dir("mesh-torus")).unwrap())
            .unwrap();
    assert!(mesh.contains("area: [%bcst_x, %bcst_y]"));
}

fn processor_line<'a>(mlir: &'a str, module: &str) -> &'a str {
    mlir.lines()
        .find(|line| line.contains("adl.processor.") && line.contains(module))
        .unwrap_or_else(|| panic!("missing processor {module}"))
}

fn resource_clause(line: &str) -> &str {
    line.split_once("with [")
        .map(|(_, resources)| resources.trim_end_matches(']'))
        .expect("processor should have resources")
}
