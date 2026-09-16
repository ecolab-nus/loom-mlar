#[allow(dead_code)]
#[path = "../examples/staged_heterogeneous_mesh.rs"]
mod example;

use mlar_rust::architecture_to_mlir;

#[test]
fn staged_mesh_exports_one_scale_and_all_physical_memories() {
    let architecture = example::build().unwrap();
    let mlir = architecture_to_mlir(&architecture).unwrap();

    assert_eq!(mlir.matches("adl.arch.scale").count(), 1);
    let scale = mlir
        .lines()
        .find(|line| line.contains("adl.arch.scale \"arch_tile\""))
        .unwrap();
    assert!(!scale.contains("mem_region"));

    let element = mlir
        .lines()
        .find(|line| line.contains("adl.arch.compose \"arch_tile_element\""))
        .unwrap();
    let memory_handles = |symbol: &str| {
        let line = mlir
            .lines()
            .find(|line| line.contains(&format!("adl.memory.array \"{symbol}\"")))
            .unwrap()
            .trim();
        let aggregate = line.split(" = ").next().unwrap();
        let instance = line.split(" of ").nth(1).unwrap();
        (aggregate, instance)
    };
    let handles =
        ["GCRAM", "RRAM", "STAGE", "OUTPUT"].map(|name| memory_handles(&format!("mem_{name}")));
    let leaves = handles.map(|(_, instance)| instance);
    for leaf in leaves {
        assert!(element.contains(leaf), "{element}");
    }
    let root = mlir
        .lines()
        .find(|line| line.contains("adl.arch.compose \"arch_staged_heterogeneous_mesh\""))
        .unwrap();
    let aggregates = handles.map(|(aggregate, _)| aggregate);
    for aggregate in aggregates {
        assert!(root.contains(aggregate), "{root}");
    }

    assert_eq!(
        mlir.matches("loom.bind_mem %A, @mem_STAGE_instance")
            .count(),
        1
    );
    assert_eq!(
        mlir.matches("loom.bind_mem %B, @mem_STAGE_instance")
            .count(),
        1
    );
    assert!(mlir.contains("loom.bind_mem %C, @mem_OUTPUT_instance"));
    assert!(mlir.contains("src_mem_space @mem_GCRAM_instance dst_mem_space @mem_STAGE_instance"));
    assert!(mlir.contains("src_mem_space @mem_RRAM_instance dst_mem_space @mem_STAGE_instance"));
}

#[test]
fn capacities_force_tiled_staging_for_the_acceptance_workload() {
    let architecture = example::build().unwrap();
    for (memory, capacity) in [
        ("GCRAM", example::GCRAM_CAPACITY),
        ("RRAM", example::RRAM_CAPACITY),
        ("STAGE", example::STAGE_CAPACITY),
        ("OUTPUT", example::OUTPUT_CAPACITY),
    ] {
        let array = architecture.memory(memory).unwrap();
        assert_eq!(
            architecture.memory_definition(array).unwrap().capacity,
            capacity
        );
    }

    let full_inputs = 2 * (512 * 256 + 256 * 512);
    assert!(full_inputs > example::STAGE_CAPACITY);
    assert_eq!(example::staging_bytes(128, 64, 128), 32 * 1024);
    assert!(example::staging_bytes(128, 64, 128) <= example::STAGE_CAPACITY);
    assert!(512 * 256 * 2 <= example::GCRAM_CAPACITY);
    assert!(256 * 512 * 2 <= example::RRAM_CAPACITY);
    assert_eq!(512 * 512 * 4, example::OUTPUT_CAPACITY);
}

#[test]
fn processor_routes_keep_stage_and_output_physically_distinct() {
    let architecture = example::build().unwrap();
    let matrix = architecture.processor_array("matrix_lane").unwrap();
    assert_eq!(
        matrix
            .connection()
            .inputs
            .iter()
            .map(|endpoint| endpoint.memory.as_str())
            .collect::<Vec<_>>(),
        ["STAGE", "STAGE"]
    );
    assert_eq!(matrix.connection().outputs[0].memory, "OUTPUT");
    assert_eq!(matrix.connection().resources, ["stage_port", "matrix"]);
}
