//! Tenstorrent-style mesh shared by the wormhole and blackhole fixtures: one L1
//! per core, DRAM channels, NoC0 loads/gathers, NoC1 stores. Each fixture
//! supplies its grid, L1 size and perf YAMLs; processor MLIR is shared.

use std::fs;
use std::path::Path;

use mlar_rust::{
    Architecture, Axis, Connection, EndpointIndex, MemoryDefinition, MemoryDomain,
    MemoryEndpoint, ProcessorDefinition, ProcessorType, Resource, architecture_to_mlir,
    generate_evaluator_binary, mlir_validators_available,
};

pub struct Perf {
    pub matrix_lane: &'static str,
    pub matrix_lane_ss: &'static str,
    pub vector_lane: &'static str,
    pub dram_l1_noc0: &'static str,
    pub l1_l1_noc0: &'static str,
    pub l1_dram_noc1: &'static str,
}

pub struct Chip {
    pub x: u64,
    pub y: u64,
    pub dram_channels: u64,
    pub l1_bytes: u64,
    pub perf: Perf,
}

fn processor(name: &str, mlir: &str, perf: &str, kind: ProcessorType) -> ProcessorDefinition {
    ProcessorDefinition::from_mlir_source_with_perf_yaml(name, mlir, perf)
        .unwrap_or_else(|e| panic!("{name}: {e}"))
        .with_type(kind)
}

fn all2() -> MemoryEndpoint {
    MemoryEndpoint::new("L1", vec![EndpointIndex::All, EndpointIndex::All])
}

pub fn mesh(chip: &Chip) -> Architecture {
    let p = &chip.perf;
    Architecture::builder("system")
        .axis("dram_channel", chip.dram_channels)
        .axis("x", chip.x)
        .axis("y", chip.y)
        .memory_definition(MemoryDefinition::new("DRAM", 1_610_612_736, 8_192))
        .memory_definition(MemoryDefinition::new("L1", chip.l1_bytes, 16).with_banking(16))
        .place_memory("DRAM", MemoryDomain::DRAM, ["dram_channel"])
        .place_memory("L1", MemoryDomain::L1, ["x", "y"])
        .resource(Resource::exclusive("noc0"))
        .resource(Resource::exclusive("noc1"))
        .resource(
            Resource::exclusive("matrix_lane")
                .indexed(vec![Axis::new("x", chip.x), Axis::new("y", chip.y)]),
        )
        .processor_definition(processor(
            "matrix_lane",
            include_str!("processors/matrix_lane.mlir"),
            p.matrix_lane,
            ProcessorType::Compute,
        ))
        .processor_definition(processor(
            "matrix_lane_ss",
            include_str!("processors/matrix_lane_ss.mlir"),
            p.matrix_lane_ss,
            ProcessorType::Compute,
        ))
        .processor_definition(
            processor(
                "vector_lane",
                include_str!("processors/vector_lane.mlir"),
                p.vector_lane,
                ProcessorType::Compute,
            )
            .with_resources(vec![Resource::exclusive("vector_lane")]),
        )
        .processor_definition(processor(
            "dram_l1_noc0",
            include_str!("processors/dram_l1_noc0.mlir"),
            p.dram_l1_noc0,
            ProcessorType::DataMover,
        ))
        .processor_definition(processor(
            "l1_l1_noc0",
            include_str!("processors/l1_l1_noc0.mlir"),
            p.l1_l1_noc0,
            ProcessorType::DataMover,
        ))
        .processor_definition(processor(
            "l1_dram_noc1",
            include_str!("processors/l1_dram_noc1.mlir"),
            p.l1_dram_noc1,
            ProcessorType::DataMover,
        ))
        .connect(
            "matrix_lane",
            Connection::new(["x", "y"])
                .input("data", "L1")
                .output("result", "L1")
                .with_resources(["matrix_lane"]),
        )
        .connect(
            "matrix_lane_ss",
            Connection::new(["x", "y"])
                .input("lhs", "L1")
                .input("rhs", "L1")
                .output("result", "L1")
                .with_resources(["matrix_lane"]),
        )
        .connect(
            "vector_lane",
            Connection::new(["x", "y"]).input("data", "L1").output("result", "L1"),
        )
        .connect(
            "dram_l1_noc0",
            Connection::new(Vec::<String>::new())
                .input("src", MemoryEndpoint::new("DRAM", vec![EndpointIndex::All]))
                .output("dst", all2())
                .with_resources(["noc0"]),
        )
        .connect(
            "l1_l1_noc0",
            Connection::new(Vec::<String>::new())
                .input("src", all2())
                .output("dst", all2())
                .with_resources(["noc0"]),
        )
        .connect(
            "l1_dram_noc1",
            Connection::new(Vec::<String>::new())
                .input("src", all2())
                .output("dst", MemoryEndpoint::new("DRAM", vec![EndpointIndex::All]))
                .with_resources(["noc1"]),
        )
        .build()
        .expect("TT mesh should build")
}

/// Write `<dir>/2d_mesh_torus.mlir`; fail if it changed from the checked-in copy.
pub fn export_hw_spec(chip: &Chip, dir: &str) {
    assert!(
        mlir_validators_available(),
        "adl-opt and loom-opt are required to export a checked hw_spec"
    );
    let mlir = architecture_to_mlir(&mesh(chip)).expect("MLIR export and validation");
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(dir).join("2d_mesh_torus.mlir");
    let previous = fs::read_to_string(&path).ok();
    fs::write(&path, &mlir).expect("write hw_spec");
    if let Some(previous) = previous {
        assert_eq!(previous, mlir, "{} changed; rerun to accept", path.display());
    }
}

pub fn generate_evaluator(chip: &Chip, dir: &str) {
    let out = Path::new(env!("CARGO_MANIFEST_DIR")).join(dir).join("bin");
    let binary = generate_evaluator_binary(&mesh(chip), "eval_system", &out)
        .expect("evaluator generation");
    assert!(binary.exists(), "missing {binary:?}");
}
