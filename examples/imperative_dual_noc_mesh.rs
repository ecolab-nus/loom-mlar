mod imperative_support;

use mlar_rust::{
    Architecture, MemoryCatalog, MemoryDefinition, MemoryEndpoint, NamedMemoryRegion,
    ResourceArray, architecture_to_mlir,
};

use imperative_support::{ExampleResult, architecture_dir, connection, processor_definition};

pub fn build() -> ExampleResult<Architecture> {
    let directory = architecture_dir("dual-noc-mesh");
    let catalog = MemoryCatalog {
        definitions: vec![
            MemoryDefinition::new("DRAM", ["dram_channel"], 1_610_612_736, 8192),
            MemoryDefinition::new("L1", ["x", "y"], 1_398_784, 16).with_banking(16),
        ],
        regions: vec![NamedMemoryRegion::new(
            "all_l1",
            MemoryEndpoint::parse("L1[:, :]")?,
        )],
    };

    Ok(Architecture::builder("dual_noc_system")
        .dimension("dram_channel", 8)
        .dimension("x", 8)
        .dimension("y", 8)
        .memory_catalog(catalog)
        .place_memory("DRAM", ["dram_channel"])
        .place_memory("L1", ["x", "y"])
        .resource(ResourceArray::exclusive("noc0"))
        .resource(ResourceArray::exclusive("noc1"))
        .processor_definition(processor_definition(&directory, "matrix_lane")?)
        .processor_definition(processor_definition(&directory, "vector_lane")?)
        .processor_definition(processor_definition(&directory, "dram_l1_noc0")?)
        .processor_definition(processor_definition(&directory, "l1_l1_noc0")?)
        .processor_definition(processor_definition(&directory, "l1_dram_noc1")?)
        .connect("matrix_lane", connection(&["L1[x, y]"], &["L1[x, y]"])?)
        .connect("vector_lane", connection(&["L1[x, y]"], &["L1[x, y]"])?)
        .connect(
            "dram_l1_noc0",
            connection(&["DRAM[:]"], &["all_l1"])?.with_resources(["noc0"]),
        )
        .connect(
            "l1_l1_noc0",
            connection(&["all_l1"], &["all_l1"])?.with_resources(["noc0"]),
        )
        .connect(
            "l1_dram_noc1",
            connection(&["all_l1"], &["DRAM[:]"])?.with_resources(["noc1"]),
        )
        .build()?)
}

fn main() -> ExampleResult<()> {
    let architecture = build()?;
    print!("{}", architecture_to_mlir(&architecture)?);
    Ok(())
}
