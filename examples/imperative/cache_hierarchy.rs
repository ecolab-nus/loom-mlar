mod support;

use mlar_rust::{
    Architecture, MemoryAlias, MemoryDefinition, MemoryEndpoint, Resource, architecture_to_mlir,
};

use support::{ExampleResult, architecture_dir, connection};

pub fn build() -> ExampleResult<Architecture> {
    Ok(Architecture::builder("cache_system")
        .axis("cluster", 2)
        .axis("core", 4)
        .axis("dram_channel", 2)
        .memory_definition(MemoryDefinition::new(
            "DRAM",
            ["dram_channel"],
            134_217_728,
            4096,
        ))
        .memory_definition(
            MemoryDefinition::new("L1", ["cluster", "core"], 262_144, 64).with_banking(8),
        )
        .memory_definition(MemoryDefinition::new("L2", ["cluster"], 2_097_152, 64).with_banking(4))
        .memory_alias(MemoryAlias::new(
            "l1_cluster",
            MemoryEndpoint::parse("L1[cluster, :]")?,
        ))
        .memory_alias(MemoryAlias::new(
            "l2_clusters",
            MemoryEndpoint::parse("L2[:]")?,
        ))
        .place_memory("DRAM", ["dram_channel"])
        .place_memory("L1", ["cluster", "core"])
        .place_memory("L2", ["cluster"])
        .resource(Resource::exclusive("memory_fabric"))
        .resource(Resource::exclusive("l2_fabric"))
        .processor_source_dir(architecture_dir("cache-hierarchy"))
        .processors(["core_lane", "dram_l2_dma", "l2_l1_dma", "l1_l2_dma"])
        .connect(
            "core_lane",
            connection(
                ["cluster", "core"],
                ["L1[cluster, core]"],
                ["L1[cluster, core]"],
            )?,
        )
        .connect(
            "dram_l2_dma",
            connection([], ["DRAM[:]"], ["l2_clusters"])?.with_resources(["memory_fabric"]),
        )
        .connect(
            "l2_l1_dma",
            connection(["cluster"], ["L2[cluster]"], ["l1_cluster"])?
                .with_resources(["memory_fabric", "l2_fabric"]),
        )
        .connect(
            "l1_l2_dma",
            connection(["cluster"], ["l1_cluster"], ["L2[cluster]"])?
                .with_resources(["memory_fabric", "l2_fabric"]),
        )
        .build()?)
}

/// Two levels — cores owning L1 inside clusters owning L2 — so the export
/// carries one `adl.arch.scale` per level, each with a single `mem_region`.
/// loom-dataflow requires exactly one scale per module, so this module is
/// well-formed MLIR that its explorer still will not accept.
fn main() -> ExampleResult<()> {
    let architecture = build()?;
    print!("{}", architecture_to_mlir(&architecture)?);
    Ok(())
}
