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
    "dual-noc-mesh",
    "shared-link-mesh",
];

#[allow(dead_code)]
#[path = "../examples/imperative/cache_hierarchy.rs"]
mod imperative_cache_hierarchy;
#[allow(dead_code)]
#[path = "../examples/imperative/dual_noc_mesh.rs"]
mod imperative_dual_noc_mesh;
#[allow(dead_code)]
#[path = "../examples/imperative/shared_link_mesh.rs"]
mod imperative_shared_link_mesh;

#[allow(dead_code)]
#[path = "../../examples/cache_hierarchy.rs"]
mod core_cache_hierarchy;
#[allow(dead_code)]
#[path = "../../examples/dual_noc_mesh.rs"]
mod core_dual_noc_mesh;
#[allow(dead_code)]
#[path = "../../examples/mesh_torus.rs"]
mod core_mesh_torus;
#[allow(dead_code)]
#[path = "../../examples/shared_link_mesh.rs"]
mod core_shared_link_mesh;
#[allow(dead_code)]
#[path = "../../examples/single_core.rs"]
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
fn hierarchy_normalizes_to_flat_memory_and_reports_unsupported_adl_slice() {
    let architecture = mlar_frontend::load_arch(example_dir("cache-hierarchy")).unwrap();
    assert_eq!(architecture.memory("L1").unwrap().rank(), 2);
    assert!(matches!(
        architecture_to_mlir(&architecture),
        Err(AdlExportError::UnsupportedMemorySelection { .. })
    ));
}

// Sibling memories cannot share the dialect's single region slot.
#[test]
fn sibling_memories_on_one_domain_still_exceed_a_single_scale() {
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

    match architecture_to_mlir(&architecture) {
        Err(AdlExportError::MultipleMemoryRegions { scope, count }) => {
            assert_eq!(scope, "x");
            assert_eq!(count, 2);
        }
        other => panic!("expected a multi-region rejection, got {other:?}"),
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
fn core_dual_noc_mesh_matches_frontend() {
    assert_core_matches("dual-noc-mesh", core_dual_noc_mesh::build().unwrap());
}

#[test]
fn core_shared_link_mesh_matches_frontend() {
    assert_core_matches("shared-link-mesh", core_shared_link_mesh::build().unwrap());
}

fn assert_core_matches(name: &str, core: Architecture) {
    let lowered = mlar_frontend::load_arch(example_dir(name)).unwrap();
    assert_eq!(
        serde_json::to_value(&core).unwrap(),
        serde_json::to_value(&lowered).unwrap(),
        "{name}: core construction differs from frontend lowering"
    );
    if LOWERABLE.contains(&name) {
        assert_eq!(
            mlar_rust::architecture_to_mlir_unchecked(&core).unwrap(),
            mlar_rust::architecture_to_mlir_unchecked(&lowered).unwrap(),
            "{name}: ADL exports differ"
        );
        if mlar_rust::mlir_validators_available() {
            architecture_to_mlir(&core).unwrap();
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
    assert_eq!(
        serde_json::to_value(&declarative).unwrap(),
        serde_json::to_value(&imperative).unwrap(),
        "{name} canonical architectures differ"
    );
    if LOWERABLE.contains(&name) {
        assert_eq!(
            architecture_to_mlir(&declarative).unwrap(),
            architecture_to_mlir(&imperative).unwrap(),
            "{name} exports differ"
        );
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
                endpoint.memory == "L1"
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
            3,
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
    assert!(dual.contains("dst_mem_space @mem_L1 : 1"));
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
