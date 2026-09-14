use mlar_rust::{
    AffineExpr, AffineMap, Architecture, Axis, Connection, EndpointIndex, Expr, MemoryDefinition,
    MemoryEndpoint, NetworkInterface, NetworkLink, NetworkTopology, Resource, Scope,
};
use std::error::Error;

#[path = "mesh_torus/processors.rs"]
mod processors;

pub fn build() -> Result<Architecture, Box<dyn Error>> {
    let axes = vec![Axis::new("x", 4), Axis::new("y", 4)];
    let network = NetworkTopology::new("l1_torus", axes.clone())
        .with_parameters(["torus_link_bandwidth"])
        .with_link(
            NetworkLink::new(
                "east",
                AffineMap::new(
                    &axes,
                    &axes,
                    vec![
                        AffineExpr::modulo(
                            AffineExpr::add(AffineExpr::variable("x"), AffineExpr::constant(1)),
                            4,
                        ),
                        AffineExpr::variable("y"),
                    ],
                )?,
                Expr::sym("torus_link_bandwidth"),
            )
            .with_resource("x_links"),
        )
        .with_link(
            NetworkLink::new(
                "west",
                AffineMap::new(
                    &axes,
                    &axes,
                    vec![
                        AffineExpr::modulo(
                            AffineExpr::add(AffineExpr::variable("x"), AffineExpr::constant(3)),
                            4,
                        ),
                        AffineExpr::variable("y"),
                    ],
                )?,
                Expr::sym("torus_link_bandwidth"),
            )
            .with_resource("x_links"),
        )
        .with_link(
            NetworkLink::new(
                "north",
                AffineMap::new(
                    &axes,
                    &axes,
                    vec![
                        AffineExpr::variable("x"),
                        AffineExpr::modulo(
                            AffineExpr::add(AffineExpr::variable("y"), AffineExpr::constant(1)),
                            4,
                        ),
                    ],
                )?,
                Expr::sym("torus_link_bandwidth"),
            )
            .with_resource("y_links"),
        )
        .with_link(
            NetworkLink::new(
                "south",
                AffineMap::new(
                    &axes,
                    &axes,
                    vec![
                        AffineExpr::variable("x"),
                        AffineExpr::modulo(
                            AffineExpr::add(AffineExpr::variable("y"), AffineExpr::constant(3)),
                            4,
                        ),
                    ],
                )?,
                Expr::sym("torus_link_bandwidth"),
            )
            .with_resource("y_links"),
        )
        .with_interface(NetworkInterface::new(
            "l1",
            MemoryEndpoint::new("L1", vec![EndpointIndex::All, EndpointIndex::All]),
        ))
        .with_resource(Resource::exclusive("x_links").indexed(axes.clone()))
        .with_resource(Resource::exclusive("y_links").indexed(axes.clone()));
    Ok(Architecture::builder("torus_system")
        .axis("dram_channel", 4)
        .axis("x", 4)
        .axis("y", 4)
        .memory_definition(MemoryDefinition::new("DRAM", 268_435_456, 4_096))
        .memory_definition(MemoryDefinition::new("L1", 262_144, 64).with_banking(8))
        .place_memory("DRAM", ["dram_channel"])
        .place_memory("L1", ["x", "y"])
        .resource(Resource::exclusive("noc_ingress"))
        .resource(Resource::exclusive("noc_egress"))
        .resource(Resource::exclusive("l1_torus_x"))
        .resource(Resource::exclusive("l1_torus_y"))
        .processor_definition(processors::matrix_lane()?)
        .processor_definition(processors::dram_l1_dma()?)
        .processor_definition(processors::l1_dram_dma()?)
        .connect(
            "matrix_lane",
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
        .connect(
            "dram_l1_dma",
            Connection::new(
                Vec::<String>::new(),
                vec![MemoryEndpoint::new("DRAM", vec![EndpointIndex::All])],
                vec![MemoryEndpoint::new(
                    "L1",
                    vec![EndpointIndex::All, EndpointIndex::All],
                )],
            )
            .with_resources(["noc_ingress"]),
        )
        .connect(
            "l1_dram_dma",
            Connection::new(
                Vec::<String>::new(),
                vec![MemoryEndpoint::new(
                    "L1",
                    vec![EndpointIndex::All, EndpointIndex::All],
                )],
                vec![MemoryEndpoint::new("DRAM", vec![EndpointIndex::All])],
            )
            .with_resources(["noc_egress"]),
        )
        .network(network)
        .scope(
            Scope::new("mesh", ["x", "y"])
                .with_memories(["L1"])
                .with_processors(["matrix_lane"])
                .with_networks(["l1_torus"]),
        )
        .build()?)
}

fn main() -> Result<(), Box<dyn Error>> {
    let architecture = build()?;
    print!("{}", mlar_rust::architecture_to_mlir(&architecture)?);
    Ok(())
}
