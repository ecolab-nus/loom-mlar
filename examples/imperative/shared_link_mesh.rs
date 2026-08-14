//! One definition placed under four names with `connect_as`.

use std::error::Error;
use std::path::{Path, PathBuf};

use mlar_rust::{Architecture, Connection, MemoryDefinition, Resource, architecture_to_mlir};

type ExampleResult<T> = Result<T, Box<dyn Error>>;

fn architecture_dir(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples/declarative")
        .join(name)
}

pub fn build() -> ExampleResult<Architecture> {
    Ok(Architecture::builder("link_system")
        .axis("x", 4)
        .axis("y", 4)
        .memory_definition(MemoryDefinition::new("L1", ["x", "y"], 262_144, 64).with_banking(8))
        .place_memory("L1", ["x", "y"])
        .resource(Resource::exclusive("x_links"))
        .resource(Resource::exclusive("y_links"))
        .processor_source_dir(architecture_dir("shared-link-mesh"))
        .processors(["lane", "link_dma"])
        .connect(
            "lane",
            Connection::parse(["x", "y"], ["L1[x, y]"], ["L1[x, y]"])?,
        )
        .connect_as(
            "east_link",
            "link_dma",
            Connection::parse(["x", "y"], ["L1[x, y]"], ["L1[(x + 1) mod 4, y]"])?
                .with_resources(["x_links"]),
        )
        .connect_as(
            "west_link",
            "link_dma",
            Connection::parse(["x", "y"], ["L1[x, y]"], ["L1[(x + 3) mod 4, y]"])?
                .with_resources(["x_links"]),
        )
        .connect_as(
            "north_link",
            "link_dma",
            Connection::parse(["x", "y"], ["L1[x, y]"], ["L1[x, (y + 1) mod 4]"])?
                .with_resources(["y_links"]),
        )
        .connect_as(
            "south_link",
            "link_dma",
            Connection::parse(["x", "y"], ["L1[x, y]"], ["L1[x, (y + 3) mod 4]"])?
                .with_resources(["y_links"]),
        )
        .build()?)
}

fn main() -> ExampleResult<()> {
    let architecture = build()?;
    println!("architecture {}", architecture.name());
    for processor in architecture.processors() {
        println!(
            "  {} uses definition {} across {} instances",
            processor.name(),
            processor.definition_name(),
            processor.instances(&architecture).len()
        );
    }
    println!(
        "exported {} bytes",
        architecture_to_mlir(&architecture)?.len()
    );
    Ok(())
}
