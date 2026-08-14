use std::error::Error;
use std::path::{Path, PathBuf};

use mlar_rust::{
    Architecture, Connection, MemoryAlias, MemoryDefinition, MemoryEndpoint, Resource,
    architecture_to_mlir,
};

type ExampleResult<T> = Result<T, Box<dyn Error>>;

fn architecture_dir(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples/declarative")
        .join(name)
}

pub fn build() -> ExampleResult<Architecture> {
    Ok(Architecture::builder("dual_noc_system")
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
            MemoryEndpoint::parse("L1[:, :]")?,
        ))
        .place_memory("DRAM", ["dram_channel"])
        .place_memory("L1", ["x", "y"])
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
            Connection::parse(["x", "y"], ["L1[x, y]"], ["L1[x, y]"])?,
        )
        .connect(
            "vector_lane",
            Connection::parse(["x", "y"], ["L1[x, y]"], ["L1[x, y]"])?,
        )
        .connect(
            "dram_l1_noc0",
            Connection::parse([], ["DRAM[:]"], ["all_l1"])?.with_resources(["noc0"]),
        )
        .connect(
            "l1_l1_noc0",
            Connection::parse([], ["all_l1"], ["all_l1"])?.with_resources(["noc0"]),
        )
        .connect(
            "l1_dram_noc1",
            Connection::parse([], ["all_l1"], ["DRAM[:]"])?.with_resources(["noc1"]),
        )
        .build()?)
}

fn main() -> ExampleResult<()> {
    let architecture = build()?;
    print!("{}", architecture_to_mlir(&architecture)?);
    Ok(())
}
