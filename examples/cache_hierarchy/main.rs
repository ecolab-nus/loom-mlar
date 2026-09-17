use mlar_rust::{
    AffineExpr, Architecture, Connection, EndpointIndex, MemoryDefinition, MemoryEndpoint, Resource,
};
use std::error::Error;

mod processors;

pub fn build() -> Result<Architecture, Box<dyn Error>> {
    Ok(Architecture::builder("cache_system")
        .axis("cluster", 2)
        .axis("core", 4)
        .axis("dram_channel", 2)
        .memory_definition(MemoryDefinition::new("DRAM", 134_217_728, 4_096))
        .memory_definition(MemoryDefinition::new("L1", 262_144, 64).with_banking(8))
        .memory_definition(MemoryDefinition::new("L2", 2_097_152, 64).with_banking(4))
        .place_memory("DRAM", ["dram_channel"])
        .place_memory("L1", ["cluster", "core"])
        .place_memory("L2", ["cluster"])
        .resource(Resource::exclusive("memory_fabric"))
        .resource(Resource::exclusive("l2_fabric"))
        .processor_definition(processors::core_lane()?)
        .processor_definition(processors::dram_l2_dma()?)
        .processor_definition(processors::l2_l1_dma()?)
        .processor_definition(processors::l1_l2_dma()?)
        .connect(
            "core_lane",
            Connection::new(
                ["cluster", "core"],
                vec![MemoryEndpoint::new(
                    "L1",
                    vec![
                        EndpointIndex::Expression(AffineExpr::variable("cluster")),
                        EndpointIndex::Expression(AffineExpr::variable("core")),
                    ],
                )],
                vec![MemoryEndpoint::new(
                    "L1",
                    vec![
                        EndpointIndex::Expression(AffineExpr::variable("cluster")),
                        EndpointIndex::Expression(AffineExpr::variable("core")),
                    ],
                )],
            ),
        )
        .connect(
            "dram_l2_dma",
            Connection::new(
                Vec::<String>::new(),
                vec![MemoryEndpoint::new("DRAM", vec![EndpointIndex::All])],
                vec![MemoryEndpoint::new("L2", vec![EndpointIndex::All])],
            )
            .with_resources(["memory_fabric"]),
        )
        .connect(
            "l2_l1_dma",
            Connection::new(
                ["cluster"],
                vec![MemoryEndpoint::new(
                    "L2",
                    vec![EndpointIndex::Expression(AffineExpr::variable("cluster"))],
                )],
                vec![MemoryEndpoint::new(
                    "L1",
                    vec![
                        EndpointIndex::Expression(AffineExpr::variable("cluster")),
                        EndpointIndex::All,
                    ],
                )],
            )
            .with_resources(["memory_fabric", "l2_fabric"]),
        )
        .connect(
            "l1_l2_dma",
            Connection::new(
                ["cluster"],
                vec![MemoryEndpoint::new(
                    "L1",
                    vec![
                        EndpointIndex::Expression(AffineExpr::variable("cluster")),
                        EndpointIndex::All,
                    ],
                )],
                vec![MemoryEndpoint::new(
                    "L2",
                    vec![EndpointIndex::Expression(AffineExpr::variable("cluster"))],
                )],
            )
            .with_resources(["memory_fabric", "l2_fabric"]),
        )
        .build()?)
}

fn main() -> Result<(), Box<dyn Error>> {
    let architecture = build()?;
    println!("{}", serde_json::to_string_pretty(&architecture)?);
    Ok(())
}
