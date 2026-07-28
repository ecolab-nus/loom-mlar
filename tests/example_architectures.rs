use std::path::{Path, PathBuf};

use mlar_rust::{Expr, Schedule, Sym, architecture_to_mlir, evaluate};

fn example_dir(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples/architectures")
        .join(name)
}

#[test]
fn all_yaml_architecture_examples_load_and_export() {
    for name in [
        "single-core",
        "cache-hierarchy",
        "mesh-torus",
        "dual-noc-mesh",
    ] {
        let dir = example_dir(name);
        let arch = mlar_rust::archs::load_arch(&dir)
            .unwrap_or_else(|error| panic!("example '{name}' should load: {error}"));
        let mlir = architecture_to_mlir(&arch)
            .unwrap_or_else(|| panic!("example '{name}' should export concrete platform MLIR"));
        assert!(mlir.starts_with("module @arch_"));
    }
}

#[test]
fn examples_cover_hierarchy_l2_and_networks() {
    let hierarchy = mlar_rust::archs::load_arch(example_dir("cache-hierarchy"))
        .expect("cache hierarchy should load");
    assert_eq!(
        hierarchy
            .get_memory_region("DRAM")
            .and_then(|memory| memory.total_size_bytes()),
        Some(256 * 1024 * 1024)
    );
    assert_eq!(
        hierarchy
            .get_memory_region("L2")
            .and_then(|memory| memory.total_size_bytes()),
        Some(2 * 1024 * 1024)
    );
    assert_eq!(
        hierarchy
            .get_memory_region("L1")
            .and_then(|memory| memory.total_size_bytes()),
        Some(256 * 1024)
    );
    assert_eq!(hierarchy.children.len(), 1);
    assert_eq!(hierarchy.children[0].name, "clusters");
    assert_eq!(hierarchy.children[0].children.len(), 1);
    assert_eq!(hierarchy.children[0].children[0].name, "cores");
    assert_eq!(hierarchy.processors_recursive().len(), 4);
    let hierarchy_mlir = architecture_to_mlir(&hierarchy).expect("cache hierarchy should export");
    assert!(hierarchy_mlir.contains("adl.memory.array \"mem_array_L2\""));
    assert!(hierarchy_mlir.contains("adl.memory.array \"mem_array_array_L1\""));
    assert!(hierarchy_mlir.contains("dst_mem_space @mem_array_L2"));
    assert!(hierarchy_mlir.contains("dst_mem_space @mem_array_L1"));

    let mesh = mlar_rust::archs::load_arch(example_dir("mesh-torus")).expect("mesh should load");
    assert_eq!(mesh.networks.len(), 1);
    assert_eq!(mesh.networks[0].mesh_links().len(), 2);
    assert_eq!(mesh.networks[0].bandwidth().eval_const(), Some(64));
    assert_eq!(mesh.networks[0].io().link_bandwidth.eval_const(), Some(32));
    assert_eq!(mesh.networks[0].io().map.apply(&[2, 3]), vec![2, 3]);
    assert_eq!(mesh.children[0].dims().len(), 2);
}

#[test]
fn single_core_example_evaluates_constrained_scenarios() {
    let arch =
        mlar_rust::archs::load_arch(example_dir("single-core")).expect("single core should load");
    let func = arch
        .get_processor("vector_lane")
        .expect("vector lane should exist")
        .functionality
        .functions[0]
        .clone();
    let schedule = Schedule::Func {
        func,
        processor: None,
        scenarios: None,
    };

    let evaluated = evaluate(&schedule, &arch).expect("schedule should evaluate");
    let Schedule::Func {
        scenarios: Some(scenarios),
        ..
    } = evaluated
    else {
        panic!("evaluation should populate function scenarios");
    };
    assert_eq!(scenarios.len(), 2);

    let at = |scenario: usize, l: i64| {
        scenarios[scenario]
            .time_cost
            .to_expr()
            .substitute(&[(Sym::new("L"), Expr::Const(l))])
            .eval_const()
    };
    assert_eq!(at(0, 1024), Some(34));
    assert_eq!(at(1, 1025), Some(34));
}

#[test]
fn dual_noc_example_matches_the_loom_platform_contract() {
    let arch = mlar_rust::archs::load_arch(example_dir("dual-noc-mesh"))
        .expect("dual-NoC mesh should load");

    assert_eq!(arch.children.len(), 1);
    assert_eq!(arch.children[0].dims().len(), 2);

    let noc0_load = arch
        .get_data_mover("dram_l1_noc0")
        .expect("NoC0 load mover should exist");
    assert!(
        noc0_load
            .resources
            .iter()
            .any(|resource| resource.id().as_str() == "noc0")
    );
    assert!(noc0_load.get_function("dram_to_l1_S_f16").is_some());
    assert!(noc0_load.get_function("dram_to_l1_S_bcst").is_some());

    let noc0_gather = arch
        .get_data_mover("l1_l1_noc0")
        .expect("NoC0 gather mover should exist");
    assert!(
        noc0_gather
            .resources
            .iter()
            .any(|resource| resource.id().as_str() == "noc0")
    );
    assert!(noc0_gather.get_function("l1_gather").is_some());

    let noc1_store = arch
        .get_data_mover("l1_dram_noc1")
        .expect("NoC1 store mover should exist");
    assert!(
        noc1_store
            .resources
            .iter()
            .any(|resource| resource.id().as_str() == "noc1")
    );
    assert!(noc1_store.get_function("l1_to_dram_f16").is_some());

    let matmul = arch
        .get_processor("matrix_lane")
        .and_then(|processor| processor.get_function("matmul_SS_f16"))
        .expect("matrix lane should expose the local-memory matmul")
        .func
        .clone();
    let evaluated = evaluate(
        &Schedule::Func {
            func: matmul,
            processor: None,
            scenarios: None,
        },
        &arch,
    )
    .expect("dual-NoC matmul should evaluate symbolically");
    let Schedule::Func {
        scenarios: Some(scenarios),
        ..
    } = evaluated
    else {
        panic!("evaluation should populate matmul scenarios");
    };
    assert_eq!(scenarios.len(), 2);

    let mlir = architecture_to_mlir(&arch).expect("dual-NoC mesh should export");
    assert_eq!(mlir.matches("adl.arch.scale").count(), 1);
    assert!(mlir.contains("adl.processor.dmover @proc_dram_l1_noc0"));
    assert!(mlir.contains("adl.processor.dmover @proc_l1_dram_noc1"));
}
