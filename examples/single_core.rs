use mlar_rust::{Architecture, Connection, MemoryDefinition, MemoryEndpoint};
use std::error::Error;

#[path = "single_core/processors.rs"]
mod processors;

pub fn build() -> Result<Architecture, Box<dyn Error>> {
    Ok(Architecture::builder("single_core")
        .memory_definition(MemoryDefinition::new("L1", 262_144, 64).with_banking(4))
        .place_memory("L1", Vec::<String>::new())
        .processor_definition(processors::vector_lane()?)
        .connect(
            "vector_lane",
            Connection::new(
                Vec::<String>::new(),
                vec![MemoryEndpoint::new("L1", vec![])],
                vec![MemoryEndpoint::new("L1", vec![])],
            ),
        )
        .build()?)
}

fn main() -> Result<(), Box<dyn Error>> {
    let architecture = build()?;
    print!("{}", mlar_rust::architecture_to_mlir(&architecture)?);
    Ok(())
}
