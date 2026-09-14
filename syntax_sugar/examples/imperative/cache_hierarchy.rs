use mlar_syntax_sugar::Connection;
use std::error::Error;
use std::path::{Path, PathBuf};

use mlar_rust::{Architecture, MemoryDefinition, Resource, architecture_to_mlir};

type ExampleResult<T> = Result<T, Box<dyn Error>>;

fn architecture_dir(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples/declarative")
        .join(name)
}

pub fn build() -> ExampleResult<Architecture> {
    Ok(mlar_syntax_sugar::ArchitectureBuilder::new("cache_system")
        .axis("cluster", 2)
        .axis("core", 4)
        .axis("dram_channel", 2)
        .memory_definition(MemoryDefinition::new("DRAM", 134_217_728, 4096))
        .memory_definition(MemoryDefinition::new("L1", 262_144, 64).with_banking(8))
        .memory_definition(MemoryDefinition::new("L2", 2_097_152, 64).with_banking(4))
        .place_memory("DRAM", ["dram_channel"])
        .place_memory_levels(
            "L1",
            "L1",
            vec![vec!["cluster".into()], vec!["core".into()]],
        )
        .place_memory("L2", ["cluster"])
        .resource(Resource::exclusive("memory_fabric"))
        .resource(Resource::exclusive("l2_fabric"))
        .processor_source_dir(architecture_dir("cache-hierarchy"))
        .processors(["core_lane", "dram_l2_dma", "l2_l1_dma", "l1_l2_dma"])
        .connect(
            "core_lane",
            Connection::parse(
                ["cluster", "core"],
                ["L1[cluster][core]"],
                ["L1[cluster][core]"],
            )?,
        )
        .connect(
            "dram_l2_dma",
            Connection::parse([], ["DRAM[:]"], ["L2[:]"])?.with_resources(["memory_fabric"]),
        )
        .connect(
            "l2_l1_dma",
            Connection::parse(["cluster"], ["L2[cluster]"], ["L1[cluster]"])?
                .with_resources(["memory_fabric", "l2_fabric"]),
        )
        .connect(
            "l1_l2_dma",
            Connection::parse(["cluster"], ["L1[cluster]"], ["L2[cluster]"])?
                .with_resources(["memory_fabric", "l2_fabric"]),
        )
        .build()?)
}

// Partial L1 selections currently prevent ADL export.
fn main() -> ExampleResult<()> {
    let architecture = build()?;
    print!("{}", architecture_to_mlir(&architecture)?);
    Ok(())
}
