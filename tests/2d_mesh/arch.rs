use mlar_rust::{
    Architecture, Axis, Connection, EndpointIndex, MemoryDefinition, MemoryEndpoint,
    MemoryTechnology, Resource,
};

#[path = "processors/mod.rs"]
mod processors;

pub fn scaled_mesh_torus() -> Architecture {
    Architecture::builder("system")
        .axis("dram_channel", 8)
        .axis("x", 8)
        .axis("y", 8)
        .memory_definition(MemoryDefinition::new("DRAM", 1_610_612_736, 8_192))
        .memory_definition(
            MemoryDefinition::new("L1_R", 1_398_784, 16)
                .with_banking(16)
                .with_technology(MemoryTechnology::new("rram", 1)),
        )
        .memory_definition(
            MemoryDefinition::new("L1_S", 1_398_784, 16)
                .with_banking(16)
                .with_technology(MemoryTechnology::new("sram", 0)),
        )
        .place_memory("DRAM", ["dram_channel"])
        .place_memory("L1_R", ["x", "y"])
        .place_memory("L1_S", ["x", "y"])
        .resource(Resource::exclusive("noc0"))
        .resource(Resource::exclusive("noc1"))
        .resource(
            Resource::exclusive("matrix_lane").indexed(vec![Axis::new("x", 8), Axis::new("y", 8)]),
        )
        .processor_definition(processors::matrix_lane().unwrap())
        .processor_definition(processors::matrix_lane_ss().unwrap())
        .processor_definition(processors::matrix_lane_sr().unwrap())
        .processor_definition(processors::matrix_lane_rs().unwrap())
        .processor_definition(processors::matrix_lane_rr().unwrap())
        .processor_definition(processors::vector_lane().unwrap())
        .processor_definition(processors::dram_l1_noc0().unwrap())
        .processor_definition(processors::l1_l1_noc0().unwrap())
        .processor_definition(processors::l1_dram_noc1().unwrap())
        .connect(
            "matrix_lane",
            Connection::new(["x", "y"])
                .input("data", "L1_S")
                .output("result", "L1_S")
                .with_resources(["matrix_lane"]),
        )
        .connect(
            "matrix_lane_ss",
            Connection::new(["x", "y"])
                .input("lhs", "L1_S")
                .input("rhs", "L1_S")
                .output("result", "L1_S")
                .with_resources(["matrix_lane"]),
        )
        .connect(
            "matrix_lane_sr",
            Connection::new(["x", "y"])
                .input("lhs", "L1_S")
                .input("rhs", "L1_R")
                .output("result", "L1_S")
                .with_resources(["matrix_lane"]),
        )
        .connect(
            "matrix_lane_rs",
            Connection::new(["x", "y"])
                .input("lhs", "L1_R")
                .input("rhs", "L1_S")
                .output("result", "L1_S")
                .with_resources(["matrix_lane"]),
        )
        .connect(
            "matrix_lane_rr",
            Connection::new(["x", "y"])
                .input("lhs", "L1_R")
                .input("rhs", "L1_R")
                .output("result", "L1_S")
                .with_resources(["matrix_lane"]),
        )
        .connect(
            "vector_lane",
            Connection::new(["x", "y"])
                .input("data", "L1_S")
                .output("result", "L1_S"),
        )
        .connect(
            "dram_l1_noc0",
            Connection::new(Vec::<String>::new())
                .input("src", MemoryEndpoint::new("DRAM", vec![EndpointIndex::All]))
                .output(
                    "dst_s",
                    MemoryEndpoint::new("L1_S", vec![EndpointIndex::All, EndpointIndex::All]),
                )
                .output(
                    "dst_r",
                    MemoryEndpoint::new("L1_R", vec![EndpointIndex::All, EndpointIndex::All]),
                )
                .with_resources(["noc0"]),
        )
        .connect(
            "l1_l1_noc0",
            Connection::new(Vec::<String>::new())
                .input(
                    "src",
                    MemoryEndpoint::new("L1_S", vec![EndpointIndex::All, EndpointIndex::All]),
                )
                .output(
                    "dst",
                    MemoryEndpoint::new("L1_S", vec![EndpointIndex::All, EndpointIndex::All]),
                )
                .with_resources(["noc0"]),
        )
        .connect(
            "l1_dram_noc1",
            Connection::new(Vec::<String>::new())
                .input(
                    "src",
                    MemoryEndpoint::new("L1_S", vec![EndpointIndex::All, EndpointIndex::All]),
                )
                .output("dst", MemoryEndpoint::new("DRAM", vec![EndpointIndex::All]))
                .with_resources(["noc1"]),
        )
        .build()
        .expect("native mesh should build")
}

pub fn single_core() -> Architecture {
    Architecture::builder("core")
        .memory_definition(
            MemoryDefinition::new("L1_R", 1_398_784, 16)
                .with_banking(16)
                .with_technology(MemoryTechnology::new("rram", 1)),
        )
        .memory_definition(
            MemoryDefinition::new("L1_S", 1_398_784, 16)
                .with_banking(16)
                .with_technology(MemoryTechnology::new("sram", 0)),
        )
        .place_memory("L1_R", Vec::<String>::new())
        .place_memory("L1_S", Vec::<String>::new())
        .resource(Resource::exclusive("matrix_lane"))
        .processor_definition(processors::matrix_lane().unwrap())
        .processor_definition(processors::matrix_lane_ss().unwrap())
        .processor_definition(processors::matrix_lane_sr().unwrap())
        .processor_definition(processors::matrix_lane_rs().unwrap())
        .processor_definition(processors::matrix_lane_rr().unwrap())
        .processor_definition(processors::vector_lane().unwrap())
        .connect(
            "matrix_lane",
            Connection::new(Vec::<String>::new())
                .input("data", "L1_S")
                .output("result", "L1_S")
                .with_resources(["matrix_lane"]),
        )
        .connect(
            "matrix_lane_ss",
            Connection::new(Vec::<String>::new())
                .input("lhs", "L1_S")
                .input("rhs", "L1_S")
                .output("result", "L1_S")
                .with_resources(["matrix_lane"]),
        )
        .connect(
            "matrix_lane_sr",
            Connection::new(Vec::<String>::new())
                .input("lhs", "L1_S")
                .input("rhs", "L1_R")
                .output("result", "L1_S")
                .with_resources(["matrix_lane"]),
        )
        .connect(
            "matrix_lane_rs",
            Connection::new(Vec::<String>::new())
                .input("lhs", "L1_R")
                .input("rhs", "L1_S")
                .output("result", "L1_S")
                .with_resources(["matrix_lane"]),
        )
        .connect(
            "matrix_lane_rr",
            Connection::new(Vec::<String>::new())
                .input("lhs", "L1_R")
                .input("rhs", "L1_R")
                .output("result", "L1_S")
                .with_resources(["matrix_lane"]),
        )
        .connect(
            "vector_lane",
            Connection::new(Vec::<String>::new())
                .input("data", "L1_S")
                .output("result", "L1_S"),
        )
        .build()
        .expect("native core should build")
}
