use mlar_frontend::Connection;
use std::error::Error;
use std::path::{Path, PathBuf};

use mlar_rust::{Architecture, MemoryDefinition, MemoryTechnology, Resource};

type ExampleResult<T> = Result<T, Box<dyn Error>>;

fn architecture_dir(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples/declarative")
        .join(name)
}

pub fn build() -> ExampleResult<Architecture> {
    Ok(mlar_frontend::ArchitectureBuilder::new("dual_noc_system")
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
        .processor_source_dir(architecture_dir("dual-noc-mesh"))
        .processors([
            "matrix_lane",
            "vector_lane",
            "dram_l1_noc0",
            "l1_l1_noc0",
            "l1_dram_noc1",
        ])
        .connect(
            "matrix_lane",
            Connection::parse_named(
                ["x", "y"],
                [("L1_S", "L1_S[x, y]"), ("L1_R", "L1_R[x, y]")],
                [("L1_S", "L1_S[x, y]")],
            )?,
        )
        .connect(
            "vector_lane",
            Connection::parse_named(
                ["x", "y"],
                [("data", "L1_S[x, y]")],
                [("result", "L1_S[x, y]")],
            )?,
        )
        .connect(
            "dram_l1_noc0",
            Connection::parse_named(
                [],
                [("src", "DRAM[:]")],
                [("s", "L1_S[:, :]"), ("r", "L1_R[:, :]")],
            )?
            .with_resources(["noc0"]),
        )
        .connect(
            "l1_l1_noc0",
            Connection::parse_named([], [("src", "L1_S[:, :]")], [("dst", "L1_R[:, :]")])?
                .with_resources(["noc0"]),
        )
        .connect(
            "l1_dram_noc1",
            Connection::parse_named([], [("src", "L1_S[:, :]")], [("dst", "DRAM[:]")])?
                .with_resources(["noc1"]),
        )
        .build()?)
}
