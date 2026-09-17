use mlar_rust::{
    Architecture, Axis, Connection, MemoryDefinition, MemoryTechnology, Resource, Scope,
};
use std::error::Error;

mod processors;

pub const GCRAM_CAPACITY: u64 = 16 * 1024 * 1024;
pub const RRAM_CAPACITY: u64 = 16 * 1024 * 1024;
pub const STAGE_CAPACITY: u64 = 64 * 1024;
pub const OUTPUT_CAPACITY: u64 = 1024 * 1024;

pub fn build() -> Result<Architecture, Box<dyn Error>> {
    Ok(Architecture::builder("staged_heterogeneous_accelerator")
        .axis("x", 2)
        .axis("y", 2)
        .memory_definition(
            MemoryDefinition::new("GCRAM", GCRAM_CAPACITY, 16)
                .with_technology(MemoryTechnology::new("gcram", 0)),
        )
        .memory_definition(
            MemoryDefinition::new("RRAM", RRAM_CAPACITY, 16)
                .with_technology(MemoryTechnology::new("rram", 1)),
        )
        .memory_definition(
            MemoryDefinition::new("STAGE", STAGE_CAPACITY, 16)
                .with_technology(MemoryTechnology::new("sram", 2)),
        )
        .memory_definition(
            MemoryDefinition::new("OUTPUT", OUTPUT_CAPACITY, 16)
                .with_technology(MemoryTechnology::new("sram", 2)),
        )
        .place_memory("GCRAM", ["x", "y"])
        .place_memory("RRAM", ["x", "y"])
        .place_memory("STAGE", ["x", "y"])
        .place_memory("OUTPUT", ["x", "y"])
        .resource(Resource::exclusive("noc0").indexed(vec![Axis::new("x", 2), Axis::new("y", 2)]))
        .resource(Resource::exclusive("noc1").indexed(vec![Axis::new("x", 2), Axis::new("y", 2)]))
        .resource(
            Resource::exclusive("stage_port").indexed(vec![Axis::new("x", 2), Axis::new("y", 2)]),
        )
        .resource(Resource::exclusive("matrix").indexed(vec![Axis::new("x", 2), Axis::new("y", 2)]))
        .processor_definition(processors::gcram_to_stage()?)
        .processor_definition(processors::rram_to_stage()?)
        .processor_definition(processors::matrix_lane()?)
        .connect(
            "gcram_to_stage",
            Connection::new(["x", "y"])
                .input("gcram", "GCRAM")
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
                .output("result", "OUTPUT")
                .with_resources(["stage_port", "matrix"]),
        )
        .scope(
            Scope::new("tile", ["x", "y"])
                .with_memories(["GCRAM", "RRAM", "STAGE", "OUTPUT"])
                .with_processors(["gcram_to_stage", "rram_to_stage", "matrix_lane"])
                .with_resources(["noc0", "noc1", "stage_port", "matrix"]),
        )
        .build()?)
}

fn main() -> Result<(), Box<dyn Error>> {
    print!("{}", mlar_rust::architecture_to_mlir(&build()?)?);
    Ok(())
}
