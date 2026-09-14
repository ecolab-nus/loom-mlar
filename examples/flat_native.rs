use mlar_rust::{
    Architecture, Connection, EndpointIndex, Expr, FuncPerfModel, MemoryDefinition, MemoryEndpoint,
    ProcessorDefinition, ProcessorType,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let source = r#"module @copy {
  func.func @copy(%src: memref<?xf16>, %dst: memref<?xf16>) {
    %L = loom.sym @L : index
    loom.bind_shape %src, [%L] : memref<?xf16>
    loom.bind_shape %dst, [%L] : memref<?xf16>
    loom.bind_mem %src, @source : memref<?xf16>
    loom.bind_mem %dst, @destination : memref<?xf16>
    memref.copy %src, %dst : memref<?xf16> to memref<?xf16>
    return
  }
}"#;
    let perf = FuncPerfModel::builder()
        .symbols(["L"])
        .simple_time_cost(Expr::Const(1), Expr::sym("L"), Expr::Const(32))
        .build();
    let processor = ProcessorDefinition::from_mlir_source("copy", source, [("copy", perf)])?
        .with_type(ProcessorType::DataMover);
    let architecture = Architecture::builder("flat")
        .axis("x", 2)
        .axis("y", 3)
        .memory_definition(MemoryDefinition::new("M", 1024, 16))
        .place_memory("M", ["x", "y"])
        .processor_definition(processor)
        .connect(
            "copy",
            Connection::new(
                Vec::<String>::new(),
                vec![MemoryEndpoint::new(
                    "M",
                    vec![EndpointIndex::All, EndpointIndex::All],
                )],
                vec![MemoryEndpoint::new(
                    "M",
                    vec![EndpointIndex::All, EndpointIndex::All],
                )],
            ),
        )
        .build()?;
    println!("{}", mlar_rust::architecture_to_mlir(&architecture)?);
    Ok(())
}
