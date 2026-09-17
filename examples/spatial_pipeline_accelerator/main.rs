use mlar_rust::{
    AffineExpr, Architecture, Axis, Connection, EndpointIndex, MemoryDefinition, MemoryEndpoint,
    Resource, Scope,
};
use std::error::Error;

mod processors;

fn local(memory: &str) -> MemoryEndpoint {
    MemoryEndpoint::new(
        memory,
        vec![EndpointIndex::Expression(AffineExpr::variable("pipeline"))],
    )
}

pub fn build() -> Result<Architecture, Box<dyn Error>> {
    let indexed = |name| Resource::exclusive(name).indexed(vec![Axis::new("pipeline", 2)]);
    Ok(Architecture::builder("spatial_pipeline_accelerator")
        .axis("dram_channel", 2)
        .axis("pipeline", 2)
        .memory_definition(MemoryDefinition::new("DRAM", 256 * 1024 * 1024, 4096))
        .memory_definition(MemoryDefinition::new("INPUT", 2 * 1024 * 1024, 64).with_banking(8))
        .memory_definition(MemoryDefinition::new("MATMUL_OUT", 1024 * 1024, 64).with_banking(8))
        .memory_definition(MemoryDefinition::new("ACTIVATION_OUT", 1024 * 1024, 64).with_banking(8))
        .memory_definition(MemoryDefinition::new("OUTPUT", 256 * 1024, 32).with_banking(4))
        .place_memory("DRAM", ["dram_channel"])
        .place_memory("INPUT", ["pipeline"])
        .place_memory("MATMUL_OUT", ["pipeline"])
        .place_memory("ACTIVATION_OUT", ["pipeline"])
        .place_memory("OUTPUT", ["pipeline"])
        .resource(indexed("ingress_link"))
        .resource(indexed("matrix_stage"))
        .resource(indexed("activation_stage"))
        .resource(indexed("reduction_stage"))
        .resource(indexed("egress_link"))
        .processor_definition(processors::ingress_dma()?)
        .processor_definition(processors::matrix_stage()?)
        .processor_definition(processors::activation_stage()?)
        .processor_definition(processors::reduction_stage()?)
        .processor_definition(processors::egress_dma()?)
        .connect(
            "ingress_dma",
            Connection::new(
                ["pipeline"],
                vec![MemoryEndpoint::new("DRAM", vec![EndpointIndex::All])],
                vec![local("INPUT")],
            )
            .with_resources(["ingress_link"]),
        )
        .connect(
            "matrix_stage",
            Connection::new(
                ["pipeline"],
                vec![local("INPUT"), local("INPUT")],
                vec![local("MATMUL_OUT")],
            )
            .with_resources(["matrix_stage"]),
        )
        .connect(
            "activation_stage",
            Connection::new(
                ["pipeline"],
                vec![local("MATMUL_OUT")],
                vec![local("ACTIVATION_OUT")],
            )
            .with_resources(["activation_stage"]),
        )
        .connect(
            "reduction_stage",
            Connection::new(
                ["pipeline"],
                vec![local("ACTIVATION_OUT")],
                vec![local("OUTPUT")],
            )
            .with_resources(["reduction_stage"]),
        )
        .connect(
            "egress_dma",
            Connection::new(
                ["pipeline"],
                vec![local("OUTPUT")],
                vec![MemoryEndpoint::new("DRAM", vec![EndpointIndex::All])],
            )
            .with_resources(["egress_link"]),
        )
        .scope(
            Scope::new("pipeline", ["pipeline"])
                .with_memories(["INPUT", "MATMUL_OUT", "ACTIVATION_OUT", "OUTPUT"])
                .with_processors([
                    "ingress_dma",
                    "matrix_stage",
                    "activation_stage",
                    "reduction_stage",
                    "egress_dma",
                ])
                .with_resources([
                    "ingress_link",
                    "matrix_stage",
                    "activation_stage",
                    "reduction_stage",
                    "egress_link",
                ]),
        )
        .build()?)
}

fn main() -> Result<(), Box<dyn Error>> {
    print!("{}", mlar_rust::architecture_to_mlir(&build()?)?);
    Ok(())
}
