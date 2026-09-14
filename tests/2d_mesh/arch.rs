use mlar_rust::{
    AffineExpr, Architecture, Connection, EndpointIndex, MemoryDefinition, MemoryEndpoint, Resource,
};

#[path = "processors/mod.rs"]
mod processors;

pub fn scaled_mesh_torus() -> Architecture {
    Architecture::builder("system")
        .axis("dram_channel", 8)
        .axis("x", 8)
        .axis("y", 8)
        .memory_definition(MemoryDefinition::new("DRAM", 1_610_612_736, 8_192))
        .memory_definition(MemoryDefinition::new("L1", 1_398_784, 16).with_banking(16))
        .place_memory("DRAM", ["dram_channel"])
        .place_memory("L1", ["x", "y"])
        .resource(Resource::exclusive("noc0"))
        .resource(Resource::exclusive("noc1"))
        .processor_definition(processors::matrix_lane().unwrap())
        .processor_definition(processors::vector_lane().unwrap())
        .processor_definition(processors::dram_l1_noc0().unwrap())
        .processor_definition(processors::l1_l1_noc0().unwrap())
        .processor_definition(processors::l1_dram_noc1().unwrap())
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
            "vector_lane",
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
            "dram_l1_noc0",
            Connection::new(
                Vec::<String>::new(),
                vec![MemoryEndpoint::new("DRAM", vec![EndpointIndex::All])],
                vec![MemoryEndpoint::new(
                    "L1",
                    vec![EndpointIndex::All, EndpointIndex::All],
                )],
            )
            .with_resources(["noc0"]),
        )
        .connect(
            "l1_l1_noc0",
            Connection::new(
                Vec::<String>::new(),
                vec![MemoryEndpoint::new(
                    "L1",
                    vec![EndpointIndex::All, EndpointIndex::All],
                )],
                vec![MemoryEndpoint::new(
                    "L1",
                    vec![EndpointIndex::All, EndpointIndex::All],
                )],
            )
            .with_resources(["noc0"]),
        )
        .connect(
            "l1_dram_noc1",
            Connection::new(
                Vec::<String>::new(),
                vec![MemoryEndpoint::new(
                    "L1",
                    vec![EndpointIndex::All, EndpointIndex::All],
                )],
                vec![MemoryEndpoint::new("DRAM", vec![EndpointIndex::All])],
            )
            .with_resources(["noc1"]),
        )
        .build()
        .expect("native mesh should build")
}

pub fn single_core() -> Architecture {
    Architecture::builder("core")
        .memory_definition(MemoryDefinition::new("L1", 1_398_784, 16).with_banking(16))
        .place_memory("L1", Vec::<String>::new())
        .processor_definition(processors::matrix_lane().unwrap())
        .processor_definition(processors::vector_lane().unwrap())
        .connect(
            "matrix_lane",
            Connection::new(
                Vec::<String>::new(),
                vec![MemoryEndpoint::new("L1", vec![])],
                vec![MemoryEndpoint::new("L1", vec![])],
            ),
        )
        .connect(
            "vector_lane",
            Connection::new(
                Vec::<String>::new(),
                vec![MemoryEndpoint::new("L1", vec![])],
                vec![MemoryEndpoint::new("L1", vec![])],
            ),
        )
        .build()
        .expect("native core should build")
}
