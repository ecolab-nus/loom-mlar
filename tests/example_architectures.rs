use std::path::{Path, PathBuf};

use mlar_rust::{Architecture, Schedule, architecture_to_mlir, evaluate};

#[allow(dead_code)]
#[path = "../examples/imperative_cache_hierarchy.rs"]
mod imperative_cache_hierarchy;
#[allow(dead_code)]
#[path = "../examples/imperative_dual_noc_mesh.rs"]
mod imperative_dual_noc_mesh;

fn example_dir(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples/architectures")
        .join(name)
}

#[test]
fn all_architecture_examples_load_and_export() {
    for name in [
        "single-core",
        "cache-hierarchy",
        "mesh-torus",
        "dual-noc-mesh",
    ] {
        let architecture = mlar_rust::archs::load_arch(example_dir(name))
            .unwrap_or_else(|error| panic!("example '{name}' should load: {error}"));
        let mlir = architecture_to_mlir(&architecture)
            .unwrap_or_else(|error| panic!("example '{name}' should export: {error}"));
        assert!(mlir.starts_with("module @arch_"));

        let function = architecture
            .processor_definitions
            .first()
            .and_then(|definition| definition.functions.first())
            .expect("example should contain a function")
            .func
            .clone();
        evaluate(
            &Schedule::Func {
                func: function,
                processor: None,
                scenarios: None,
            },
            &architecture,
        )
        .expect("example function should evaluate");
    }
}

#[test]
fn imperative_examples_match_their_declarative_packages() {
    assert_imperative_matches(
        "dual-noc-mesh",
        imperative_dual_noc_mesh::build().expect("imperative dual-NoC mesh should build"),
    );
    assert_imperative_matches(
        "cache-hierarchy",
        imperative_cache_hierarchy::build().expect("imperative cache hierarchy should build"),
    );
}

fn assert_imperative_matches(name: &str, imperative: Architecture) {
    let declarative = mlar_rust::archs::load_arch(example_dir(name))
        .unwrap_or_else(|error| panic!("declarative example '{name}' should load: {error}"));
    assert_eq!(
        serde_json::to_value(&declarative).unwrap(),
        serde_json::to_value(&imperative).unwrap(),
        "{name} canonical architectures differ"
    );
    assert_eq!(
        architecture_to_mlir(&declarative).unwrap(),
        architecture_to_mlir(&imperative).unwrap(),
        "{name} exports differ"
    );
}

#[test]
fn examples_use_the_canonical_model() {
    let architecture = mlar_rust::archs::load_arch(example_dir("dual-noc-mesh"))
        .expect("dual-NoC example should load");
    assert!(!architecture.memories.is_empty());
    assert!(!architecture.processor_definitions.is_empty());
    assert!(!architecture.processors.is_empty());
    assert!(
        architecture
            .processors
            .iter()
            .all(|processor| !processor.relation.instances.is_empty())
    );
}

#[test]
fn dual_noc_connects_system_movers_to_the_mesh_wide_l1_region() {
    let architecture = mlar_rust::archs::load_arch(example_dir("dual-noc-mesh"))
        .expect("dual-NoC example should load");
    let noc_processors = architecture
        .processors
        .iter()
        .filter(|processor| {
            matches!(
                processor.definition.as_str(),
                "dram_l1_noc0" | "l1_l1_noc0" | "l1_dram_noc1"
            )
        })
        .collect::<Vec<_>>();

    assert_eq!(noc_processors.len(), 3);
    assert!(
        noc_processors
            .iter()
            .all(|processor| processor.relation.domain.is_empty())
    );
    assert!(noc_processors.iter().all(|processor| {
        processor
            .connection
            .inputs
            .iter()
            .chain(&processor.connection.outputs)
            .any(|endpoint| endpoint.memory == "all_l1")
    }));

    let mlir = architecture_to_mlir(&architecture).expect("dual-NoC should export");
    assert!(!mlir.contains("adl.arch.scale \"arch_dram_l1_noc0\""));
    assert!(!mlir.contains("adl.arch.scale \"arch_l1_l1_noc0\""));
    assert!(!mlir.contains("adl.arch.scale \"arch_l1_dram_noc1\""));
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
        (
            "cache-hierarchy",
            &[
                "{bsize = 4096, nblk = 32768}",
                "{bsize = 64, nblk = 8192}",
                "{bsize = 64, nblk = 512}",
            ][..],
            &["elementwise_add", "load_l1", "writeback_l2", "load_l2"][..],
            4,
            2,
            6,
        ),
        (
            "mesh-torus",
            &["{bsize = 4096, nblk = 65536}", "{bsize = 64, nblk = 512}"][..],
            &["matmul", "load_l1", "writeback_dram"][..],
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
                "elementwise_add_f16",
                "vec_exp_f16",
                "dram_to_l1_S_f16",
                "dram_to_l1_S_bcst",
                "dram_to_l1_R_f16",
                "dram_to_l1_R_bcst",
                "l1_gather",
                "l1_to_dram_f16",
            ][..],
            5,
            1,
            3,
        ),
    ];

    for (name, banks, functions, processors, scales, arrays) in contracts {
        let architecture = mlar_rust::archs::load_arch(example_dir(name)).unwrap();
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

    let dual =
        architecture_to_mlir(&mlar_rust::archs::load_arch(example_dir("dual-noc-mesh")).unwrap())
            .unwrap();
    let noc0_load = processor_line(&dual, "@proc_dram_l1_noc0");
    let noc0_gather = processor_line(&dual, "@proc_l1_l1_noc0");
    assert_eq!(resource_clause(noc0_load), resource_clause(noc0_gather));
    assert!(dual.contains("area: [%bcst_x, %bcst_y]"));
    assert!(dual.contains("dst_mem_space @mem_array_L1 : 1"));
    assert!(dual.contains("loom.gather"));
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
