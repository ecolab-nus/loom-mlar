use mlar_rust::{Expr, FuncPerfModel, PerfScenario, ProcessorDefinition, ProcessorType};

fn definition(
    name: &str,
    source: &str,
    function: &'static str,
    symbols: impl IntoIterator<Item = &'static str>,
    expression: &str,
    kind: ProcessorType,
) -> Result<ProcessorDefinition, String> {
    Ok(ProcessorDefinition::from_mlir_source(
        name,
        source,
        [(
            function,
            FuncPerfModel::builder()
                .symbols(symbols)
                .scenario(PerfScenario::new(
                    Expr::parse(expression).map_err(|error| error.to_string())?,
                ))
                .build(),
        )],
    )?
    .with_type(kind))
}

pub fn ingress_dma() -> Result<ProcessorDefinition, String> {
    definition(
        "ingress_dma",
        include_str!("ingress_dma.mlir"),
        "load_inputs_f16",
        ["L"],
        "30 + (2 * L + 63) / 64",
        ProcessorType::DataMover,
    )
}

pub fn matrix_stage() -> Result<ProcessorDefinition, String> {
    definition(
        "matrix_stage",
        include_str!("matrix_stage.mlir"),
        "pipeline_matmul_f16",
        ["M", "N", "K"],
        "6 + (M * N * K + 255) / 256",
        ProcessorType::Compute,
    )
}

pub fn activation_stage() -> Result<ProcessorDefinition, String> {
    definition(
        "activation_stage",
        include_str!("activation_stage.mlir"),
        "pipeline_exp_f16",
        ["L"],
        "3 + (L + 31) / 32",
        ProcessorType::Compute,
    )
}

pub fn reduction_stage() -> Result<ProcessorDefinition, String> {
    definition(
        "reduction_stage",
        include_str!("reduction_stage.mlir"),
        "pipeline_reduce_f16",
        ["P", "R"],
        "4 + (P * R + 63) / 64",
        ProcessorType::Compute,
    )
}

pub fn egress_dma() -> Result<ProcessorDefinition, String> {
    definition(
        "egress_dma",
        include_str!("egress_dma.mlir"),
        "store_output_f16",
        ["L"],
        "30 + (2 * L + 63) / 64",
        ProcessorType::DataMover,
    )
}
