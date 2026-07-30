mod imperative_support;

use mlar_rust::{
    Architecture, MemoryCatalog, MemoryDefinition, MemoryEndpoint, NamedMemoryRegion,
    ResourceArray, architecture_to_mlir,
};

use imperative_support::{ExampleResult, architecture_dir, connection, processor_definition};

pub fn build() -> ExampleResult<Architecture> {
    let directory = architecture_dir("cache-hierarchy");
    let catalog = MemoryCatalog {
        definitions: vec![
            MemoryDefinition::new("DRAM", ["dram_channel"], 134_217_728, 4096),
            MemoryDefinition::new("L1", ["cluster", "core"], 262_144, 64).with_banking(8),
            MemoryDefinition::new("L2", ["cluster"], 2_097_152, 64).with_banking(4),
        ],
        regions: vec![
            NamedMemoryRegion::new("l1_cluster", MemoryEndpoint::parse("L1[cluster, :]")?),
            NamedMemoryRegion::new("l2_clusters", MemoryEndpoint::parse("L2[:]")?),
        ],
    };

    Ok(Architecture::builder("cache_system")
        .dimension("cluster", 2)
        .dimension("core", 4)
        .dimension("dram_channel", 2)
        .memory_catalog(catalog)
        .place_memory("DRAM", ["dram_channel"])
        .place_memory("L1", ["cluster", "core"])
        .place_memory("L2", ["cluster"])
        .resource(ResourceArray::exclusive("memory_fabric"))
        .resource(ResourceArray::exclusive("l2_fabric"))
        .processor_definition(processor_definition(&directory, "core_lane")?)
        .processor_definition(processor_definition(&directory, "dram_l2_dma")?)
        .processor_definition(processor_definition(&directory, "l2_l1_dma")?)
        .processor_definition(processor_definition(&directory, "l1_l2_dma")?)
        .connect(
            "core_lane",
            connection(&["L1[cluster, core]"], &["L1[cluster, core]"])?,
        )
        .connect(
            "dram_l2_dma",
            connection(&["DRAM[:]"], &["l2_clusters"])?.with_resources(["memory_fabric"]),
        )
        .connect(
            "l2_l1_dma",
            connection(&["L2[cluster]"], &["l1_cluster"])?
                .with_resources(["memory_fabric", "l2_fabric"]),
        )
        .connect(
            "l1_l2_dma",
            connection(&["l1_cluster"], &["L2[cluster]"])?
                .with_resources(["memory_fabric", "l2_fabric"]),
        )
        .build()?)
}

fn main() -> ExampleResult<()> {
    let architecture = build()?;
    print!("{}", architecture_to_mlir(&architecture)?);
    Ok(())
}
