use mlar_rust::{
    Architecture, Axis, Connection, EndpointIndex, MemoryDefinition, MemoryEndpoint, Resource,
};
use std::error::Error;

mod processors;

pub const DRAM_CAPACITY: u64 = 1024 * 1024 * 1024;
pub const SRAM_CAPACITY: u64 = 16 * 1024 * 1024;
pub const RRAM_CAPACITY: u64 = 16 * 1024 * 1024;
pub const STAGE_CAPACITY: u64 = 64 * 1024;

fn all(name: &str, rank: usize) -> MemoryEndpoint {
    MemoryEndpoint::new(name, vec![EndpointIndex::All; rank])
}

pub fn build() -> Result<Architecture, Box<dyn Error>> {
    Ok(Architecture::builder("staged_heterogeneous_accelerator")
        .axis("dram_channel", 8)
        .axis("x", 2)
        .axis("y", 2)
        .memory_definition(MemoryDefinition::new("DRAM", DRAM_CAPACITY, 8_192))
        .memory_definition(MemoryDefinition::new("SRAM", SRAM_CAPACITY, 16))
        .memory_definition(MemoryDefinition::new("RRAM", RRAM_CAPACITY, 16))
        .memory_definition(MemoryDefinition::new("STAGE", STAGE_CAPACITY, 16))
        .place_memory("DRAM", mlar_rust::MemoryDomain::DRAM, ["dram_channel"])
        .place_memory("SRAM", mlar_rust::MemoryDomain::L1, ["x", "y"])
        .place_memory("RRAM", mlar_rust::MemoryDomain::L1, ["x", "y"])
        .place_memory("STAGE", mlar_rust::MemoryDomain::L1, ["x", "y"])
        .resource(Resource::exclusive("noc0"))
        .resource(Resource::exclusive("noc1"))
        .resource(
            Resource::exclusive("stage_port").indexed(vec![Axis::new("x", 2), Axis::new("y", 2)]),
        )
        .resource(Resource::exclusive("matrix").indexed(vec![Axis::new("x", 2), Axis::new("y", 2)]))
        .processor_definition(processors::dram_to_sram()?)
        .processor_definition(processors::sram_to_dram()?)
        .processor_definition(processors::dram_to_rram()?)
        .processor_definition(processors::rram_to_dram()?)
        .processor_definition(processors::sram_to_stage()?)
        .processor_definition(processors::rram_to_stage()?)
        .processor_definition(processors::matrix_lane()?)
        .connect(
            "dram_to_sram",
            Connection::new(Vec::<String>::new())
                .input("src", all("DRAM", 1))
                .output("dst", all("SRAM", 2))
                .with_resources(["noc0"]),
        )
        .connect(
            "sram_to_dram",
            Connection::new(Vec::<String>::new())
                .input("src", all("SRAM", 2))
                .output("dst", all("DRAM", 1))
                .with_resources(["noc1"]),
        )
        .connect(
            "dram_to_rram",
            Connection::new(Vec::<String>::new())
                .input("src", all("DRAM", 1))
                .output("dst", all("RRAM", 2))
                .with_resources(["noc0"]),
        )
        .connect(
            "rram_to_dram",
            Connection::new(Vec::<String>::new())
                .input("src", all("RRAM", 2))
                .output("dst", all("DRAM", 1))
                .with_resources(["noc1"]),
        )
        .connect(
            "sram_to_stage",
            Connection::new(["x", "y"])
                .input("sram", "SRAM")
                .output("stage", "STAGE")
                .with_resources(["noc0", "stage_port"]),
        )
        .connect(
            "rram_to_stage",
            Connection::new(["x", "y"])
                .input("rram", "RRAM")
                .output("stage", "STAGE")
                .with_resources(["noc1", "stage_port"]),
        )
        .connect(
            "matrix_lane",
            Connection::new(["x", "y"])
                .input("stage_a", "STAGE")
                .input("stage_b", "STAGE")
                .output("result", "SRAM")
                .with_resources(["stage_port", "matrix"]),
        )
        .build()?)
}

fn main() -> Result<(), Box<dyn Error>> {
    print!("{}", mlar_rust::architecture_to_mlir(&build()?)?);
    Ok(())
}
