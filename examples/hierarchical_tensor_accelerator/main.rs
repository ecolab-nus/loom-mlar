use mlar_rust::{
    AffineExpr, Architecture, Axis, Connection, EndpointIndex, MemoryDefinition, MemoryEndpoint,
    Resource, Scope,
};
use std::error::Error;

mod processors;

fn selected(memory: &str, indices: Vec<EndpointIndex>) -> MemoryEndpoint {
    MemoryEndpoint::new(memory, indices)
}

pub fn build() -> Result<Architecture, Box<dyn Error>> {
    let cluster = EndpointIndex::Expression(AffineExpr::variable("cluster"));
    Ok(Architecture::builder("hierarchical_tensor_accelerator")
        .axis("cluster", 2)
        .axis("dram_channel", 2)
        .axis("pe", 4)
        .memory_definition(MemoryDefinition::new("DRAM", 512 * 1024 * 1024, 4096))
        .memory_definition(
            MemoryDefinition::new("CLUSTER_SRAM", 4 * 1024 * 1024, 64).with_banking(8),
        )
        .memory_definition(MemoryDefinition::new("PE_SRAM", 256 * 1024, 32).with_banking(4))
        .place_memory("DRAM", mlar_rust::MemoryDomain::DRAM, ["dram_channel"])
        .place_memory("CLUSTER_SRAM", mlar_rust::MemoryDomain::L1, ["cluster"])
        .place_memory("PE_SRAM", mlar_rust::MemoryDomain::L1, ["cluster", "pe"])
        .resource(Resource::exclusive("dram_fabric"))
        .resource(Resource::exclusive("cluster_fabric").indexed(vec![Axis::new("cluster", 2)]))
        .resource(
            Resource::exclusive("pe_sram_port")
                .indexed(vec![Axis::new("cluster", 2), Axis::new("pe", 4)]),
        )
        .processor_definition(processors::tensor_engine()?)
        .processor_definition(processors::dram_cluster_dma()?)
        .processor_definition(processors::cluster_pe_dma()?)
        .processor_definition(processors::pe_cluster_dma()?)
        .processor_definition(processors::cluster_dram_dma()?)
        .connect(
            "tensor_engine",
            Connection::new(["cluster", "pe"])
                .input("lhs", "PE_SRAM")
                .input("rhs", "PE_SRAM")
                .output("result", "PE_SRAM")
                .with_resources(["pe_sram_port"]),
        )
        .connect(
            "dram_cluster_dma",
            Connection::new(Vec::<String>::new())
                .input("src", selected("DRAM", vec![EndpointIndex::All]))
                .output("dst", selected("CLUSTER_SRAM", vec![EndpointIndex::All]))
                .with_resources(["dram_fabric"]),
        )
        .connect(
            "cluster_pe_dma",
            Connection::new(["cluster"])
                .input("src", selected("CLUSTER_SRAM", vec![cluster.clone()]))
                .output(
                    "dst",
                    selected("PE_SRAM", vec![cluster.clone(), EndpointIndex::All]),
                )
                .with_resources(["cluster_fabric"]),
        )
        .connect(
            "pe_cluster_dma",
            Connection::new(["cluster"])
                .input(
                    "src",
                    selected("PE_SRAM", vec![cluster.clone(), EndpointIndex::All]),
                )
                .output("dst", selected("CLUSTER_SRAM", vec![cluster.clone()]))
                .with_resources(["cluster_fabric"]),
        )
        .connect(
            "cluster_dram_dma",
            Connection::new(Vec::<String>::new())
                .input("src", selected("CLUSTER_SRAM", vec![EndpointIndex::All]))
                .output("dst", selected("DRAM", vec![EndpointIndex::All]))
                .with_resources(["dram_fabric"]),
        )
        .scope(
            Scope::new("cluster", ["cluster"])
                .with_memories(["CLUSTER_SRAM"])
                .with_processors(["cluster_pe_dma", "pe_cluster_dma"])
                .with_resources(["cluster_fabric"]),
        )
        .scope(
            Scope::new("pe", ["cluster", "pe"])
                .with_parent("cluster")
                .with_memories(["PE_SRAM"])
                .with_processors(["tensor_engine"])
                .with_resources(["pe_sram_port"]),
        )
        .build()?)
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("{}", serde_json::to_string_pretty(&build()?)?);
    Ok(())
}
