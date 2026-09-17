use mlar_rust::{Expr, FuncPerfModel, ProcessorDefinition, ProcessorType, Resource};

fn copy_definition(
    name: &str,
    source: &str,
    function: &'static str,
    latency: i64,
    throughput: i64,
) -> Result<ProcessorDefinition, String> {
    Ok(ProcessorDefinition::from_mlir_source(
        name,
        source,
        [(
            function,
            FuncPerfModel::builder()
                .symbols(["L"])
                .simple_time_cost(
                    Expr::Const(latency),
                    Expr::parse("2 * L").map_err(|error| error.to_string())?,
                    Expr::Const(throughput),
                )
                .build(),
        )],
    )?
    .with_type(ProcessorType::DataMover))
}

pub fn tensor_engine() -> Result<ProcessorDefinition, String> {
    Ok(ProcessorDefinition::from_mlir_source(
        "tensor_engine",
        include_str!("tensor_engine.mlir"),
        [(
            "tensor_matmul_f16",
            FuncPerfModel::builder()
                .symbols(["K", "M", "N"])
                .simple_time_cost(
                    Expr::Const(6),
                    Expr::parse("2 * M * N * K").map_err(|error| error.to_string())?,
                    Expr::Const(256),
                )
                .build(),
        )],
    )?
    .with_type(ProcessorType::Compute)
    .with_resources(vec![Resource::exclusive("tensor_pipeline")]))
}

pub fn dram_cluster_dma() -> Result<ProcessorDefinition, String> {
    copy_definition(
        "dram_cluster_dma",
        include_str!("dram_cluster_dma.mlir"),
        "load_cluster_f16",
        60,
        64,
    )
}

pub fn cluster_pe_dma() -> Result<ProcessorDefinition, String> {
    copy_definition(
        "cluster_pe_dma",
        include_str!("cluster_pe_dma.mlir"),
        "distribute_tile_f16",
        10,
        128,
    )
}

pub fn pe_cluster_dma() -> Result<ProcessorDefinition, String> {
    copy_definition(
        "pe_cluster_dma",
        include_str!("pe_cluster_dma.mlir"),
        "collect_tile_f16",
        10,
        128,
    )
}

pub fn cluster_dram_dma() -> Result<ProcessorDefinition, String> {
    copy_definition(
        "cluster_dram_dma",
        include_str!("cluster_dram_dma.mlir"),
        "store_result_f16",
        60,
        64,
    )
}
