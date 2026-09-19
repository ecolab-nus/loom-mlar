#[allow(dead_code)]
#[path = "../examples/staged_heterogeneous_accelerator/main.rs"]
mod example;

use mlar_rust::architecture_to_mlir;

#[test]
fn staged_accelerator_exports_one_scale_and_all_physical_memories() {
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
        ["DRAM", "RRAM", "SRAM", "STAGE"].map(|name| memory_handles(&format!("mem_{name}")));
    for (_, leaf) in &handles[1..] {
        assert!(element.contains(leaf), "{element}");
    }
    assert!(!element.contains(handles[0].1), "{element}");
    let root = mlir
        .lines()
        .find(|line| line.contains("adl.arch.compose \"arch_staged_heterogeneous_accelerator\""))
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
    assert!(mlir.contains("loom.bind_mem %C, @mem_SRAM_instance"));
    assert!(mlir.contains("src_mem_space @mem_SRAM_instance dst_mem_space @mem_STAGE_instance"));
    assert!(mlir.contains("src_mem_space @mem_RRAM_instance dst_mem_space @mem_STAGE_instance"));
    assert!(mlir.contains("src_mem_space @mem_DRAM dst_mem_space @mem_SRAM"));
    assert!(mlir.contains("src_mem_space @mem_SRAM dst_mem_space @mem_DRAM"));
    assert!(mlir.contains("src_mem_space @mem_DRAM dst_mem_space @mem_RRAM"));
    assert!(mlir.contains("src_mem_space @mem_RRAM dst_mem_space @mem_DRAM"));
    assert!(!mlir.contains("src_mem_space @mem_DRAM dst_mem_space @mem_STAGE"));
    assert!(!mlir.contains("src_mem_space @mem_STAGE dst_mem_space @mem_DRAM"));
}

#[test]
fn staged_accelerator_preserves_memory_capacities() {
    let architecture = example::build().unwrap();
    for (memory, capacity) in [
        ("DRAM", example::DRAM_CAPACITY),
        ("RRAM", example::RRAM_CAPACITY),
        ("SRAM", example::SRAM_CAPACITY),
        ("STAGE", example::STAGE_CAPACITY),
    ] {
        let array = architecture.memory(memory).unwrap();
        assert_eq!(
            architecture.memory_definition(array).unwrap().capacity,
            capacity
        );
    }
}

#[test]
fn processor_routes_stage_inputs_and_return_output_to_sram() {
    let architecture = example::build().unwrap();
    let matrix = architecture.processor_array("matrix_lane").unwrap();
    assert_eq!(
        matrix
            .connection()
            .inputs
            .iter()
            .map(|port| port.endpoint.memory.as_str())
            .collect::<Vec<_>>(),
        ["STAGE", "STAGE"]
    );
    assert_eq!(matrix.connection().outputs[0].endpoint.memory, "SRAM");
    assert_eq!(matrix.connection().resources, ["stage_port", "matrix"]);
}
