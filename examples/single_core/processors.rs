use mlar_rust::{
    ConstraintExpr, Expr, FuncPerfModel, ProcessorDefinition, ProcessorType, Resource, TimeCost,
};

pub fn vector_lane() -> Result<ProcessorDefinition, String> {
    Ok(ProcessorDefinition::from_mlir_source(
        "vector_lane",
        include_str!("vector_lane.mlir"),
        [(
            "vector_add",
            FuncPerfModel::builder()
                .symbols(["L"])
                .scenario_with_constraints(
                    ConstraintExpr::parse("(L > 0) && (L <= 1024)")
                        .map_err(|error| error.to_string())?,
                    TimeCost::throughput(Expr::Const(2), Expr::sym("L"), Expr::Const(32)),
                )
                .scenario_with_constraints(
                    ConstraintExpr::parse("(L > 0) && (L > 1024)")
                        .map_err(|error| error.to_string())?,
                    Expr::parse("18 + L / 64").map_err(|error| error.to_string())?,
                )
                .build(),
        )],
    )?
    .with_type(ProcessorType::Compute)
    .with_resources(vec![
        Resource::exclusive("vector_pipeline"),
        Resource::exclusive("vector_lane"),
    ]))
}
