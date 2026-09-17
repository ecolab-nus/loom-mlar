use mlar_rust::{
    Architecture, Connection, EndpointIndex, MemoryDefinition, MemoryEndpoint, Resource,
};
use std::error::Error;

mod processors;

pub fn build() -> Result<Architecture, Box<dyn Error>> {
    Ok(Architecture::builder("dual_noc_system")
        .axis("dram_channel", 8)
        .axis("x", 8)
        .axis("y", 8)
        .memory_definition(MemoryDefinition::new("DRAM", 1_610_612_736, 8_192))
        .memory_definition(MemoryDefinition::new("L1", 1_398_784, 16).with_banking(16))
        .place_memory("DRAM", ["dram_channel"])
        .place_memory("L1", ["x", "y"])
        .resource(Resource::exclusive("noc0"))
        .resource(Resource::exclusive("noc1"))
        .processor_definition(processors::matrix_lane()?)
        .processor_definition(processors::vector_lane()?)
        .processor_definition(processors::dram_l1_noc0()?)
        .processor_definition(processors::l1_l1_noc0()?)
        .processor_definition(processors::l1_dram_noc1()?)
        .connect(
            "matrix_lane",
            Connection::new(["x", "y"])
                .input("lhs", "L1")
                .input("rhs", "L1")
                .output("result", "L1"),
        )
        .connect(
            "vector_lane",
            Connection::new(["x", "y"])
                .input("input", "L1")
                .output("result", "L1"),
        )
        .connect(
            "dram_l1_noc0",
            Connection::new(Vec::<String>::new())
                .input("src", MemoryEndpoint::new("DRAM", vec![EndpointIndex::All]))
                .output(
                    "dst",
                    MemoryEndpoint::new("L1", vec![EndpointIndex::All, EndpointIndex::All]),
                )
                .with_resources(["noc0"]),
        )
        .connect(
            "l1_l1_noc0",
            Connection::new(Vec::<String>::new())
                .input(
                    "src",
                    MemoryEndpoint::new("L1", vec![EndpointIndex::All, EndpointIndex::All]),
                )
                .output(
                    "dst",
                    MemoryEndpoint::new("L1", vec![EndpointIndex::All, EndpointIndex::All]),
                )
                .with_resources(["noc0"]),
        )
        .connect(
            "l1_dram_noc1",
            Connection::new(Vec::<String>::new())
                .input(
                    "src",
                    MemoryEndpoint::new("L1", vec![EndpointIndex::All, EndpointIndex::All]),
                )
                .output("dst", MemoryEndpoint::new("DRAM", vec![EndpointIndex::All]))
                .with_resources(["noc1"]),
        )
        .build()?)
}

fn main() -> Result<(), Box<dyn Error>> {
    let architecture = build()?;
    print!("{}", mlar_rust::architecture_to_mlir(&architecture)?);
    Ok(())
}
