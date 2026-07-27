use std::path::{Path, PathBuf};

use mlar_rust::{Expr, Schedule, Sym, architecture_to_mlir, evaluate};

fn example_dir(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples/architectures")
        .join(name)
}

#[test]
fn all_yaml_architecture_examples_load_and_export() {
    for name in ["single-core", "cache-hierarchy", "mesh-torus"] {
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
    assert_eq!(hierarchy.processors_recursive().len(), 4);

    let mesh =
        mlar_rust::archs::load_arch(example_dir("mesh-torus")).expect("mesh should load");
    assert_eq!(mesh.networks.len(), 1);
    assert_eq!(mesh.networks[0].mesh_links().len(), 2);
    assert_eq!(mesh.networks[0].bandwidth().eval_const(), Some(64));
    assert_eq!(
        mesh.networks[0].io().link_bandwidth.eval_const(),
        Some(32)
    );
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
