use mlar_rust::{
    AffineExpr, Architecture, Connection, EndpointIndex, MemoryDefinition, MemoryEndpoint, Resource,
};
use std::error::Error;

#[path = "shared_link_mesh/processors.rs"]
mod processors;

pub fn build() -> Result<Architecture, Box<dyn Error>> {
    Ok(Architecture::builder("link_system")
        .axis("x", 4)
        .axis("y", 4)
        .memory_definition(MemoryDefinition::new("L1", 262_144, 64).with_banking(8))
        .place_memory("L1", ["x", "y"])
        .resource(Resource::exclusive("x_links"))
        .resource(Resource::exclusive("y_links"))
        .processor_definition(processors::lane()?)
        .processor_definition(processors::link_dma()?)
        .connect(
            "lane",
            Connection::new(
                ["x", "y"],
                vec![MemoryEndpoint::new(
                    "L1",
                    vec![
                        EndpointIndex::Expression(AffineExpr::variable("x")),
                        EndpointIndex::Expression(AffineExpr::variable("y")),
                    ],
                )],
                vec![MemoryEndpoint::new(
                    "L1",
                    vec![
                        EndpointIndex::Expression(AffineExpr::variable("x")),
                        EndpointIndex::Expression(AffineExpr::variable("y")),
                    ],
                )],
            ),
        )
        .connect_as(
            "east_link",
            "link_dma",
            Connection::new(
                ["x", "y"],
                vec![MemoryEndpoint::new(
                    "L1",
                    vec![
                        EndpointIndex::Expression(AffineExpr::variable("x")),
                        EndpointIndex::Expression(AffineExpr::variable("y")),
                    ],
                )],
                vec![MemoryEndpoint::new(
                    "L1",
                    vec![
                        EndpointIndex::Expression(AffineExpr::modulo(
                            AffineExpr::add(AffineExpr::variable("x"), AffineExpr::constant(1)),
                            4,
                        )),
                        EndpointIndex::Expression(AffineExpr::variable("y")),
                    ],
                )],
            )
            .with_resources(["x_links"]),
        )
        .connect_as(
            "west_link",
            "link_dma",
            Connection::new(
                ["x", "y"],
                vec![MemoryEndpoint::new(
                    "L1",
                    vec![
                        EndpointIndex::Expression(AffineExpr::variable("x")),
                        EndpointIndex::Expression(AffineExpr::variable("y")),
                    ],
                )],
                vec![MemoryEndpoint::new(
                    "L1",
                    vec![
                        EndpointIndex::Expression(AffineExpr::modulo(
                            AffineExpr::add(AffineExpr::variable("x"), AffineExpr::constant(3)),
                            4,
                        )),
                        EndpointIndex::Expression(AffineExpr::variable("y")),
                    ],
                )],
            )
            .with_resources(["x_links"]),
        )
        .connect_as(
            "north_link",
            "link_dma",
            Connection::new(
                ["x", "y"],
                vec![MemoryEndpoint::new(
                    "L1",
                    vec![
                        EndpointIndex::Expression(AffineExpr::variable("x")),
                        EndpointIndex::Expression(AffineExpr::variable("y")),
                    ],
                )],
                vec![MemoryEndpoint::new(
                    "L1",
                    vec![
                        EndpointIndex::Expression(AffineExpr::variable("x")),
                        EndpointIndex::Expression(AffineExpr::modulo(
                            AffineExpr::add(AffineExpr::variable("y"), AffineExpr::constant(1)),
                            4,
                        )),
                    ],
                )],
            )
            .with_resources(["y_links"]),
        )
        .connect_as(
            "south_link",
            "link_dma",
            Connection::new(
                ["x", "y"],
                vec![MemoryEndpoint::new(
                    "L1",
                    vec![
                        EndpointIndex::Expression(AffineExpr::variable("x")),
                        EndpointIndex::Expression(AffineExpr::variable("y")),
                    ],
                )],
                vec![MemoryEndpoint::new(
                    "L1",
                    vec![
                        EndpointIndex::Expression(AffineExpr::variable("x")),
                        EndpointIndex::Expression(AffineExpr::modulo(
                            AffineExpr::add(AffineExpr::variable("y"), AffineExpr::constant(3)),
                            4,
                        )),
                    ],
                )],
            )
            .with_resources(["y_links"]),
        )
        .build()?)
}

fn main() -> Result<(), Box<dyn Error>> {
    let architecture = build()?;
    print!("{}", mlar_rust::architecture_to_mlir(&architecture)?);
    Ok(())
}
