use mlar_rust::arch::{
    EndpointIndex::{All, Expression},
    ResolvedEndpointIndex,
};
use mlar_rust::{
    AffineExpr, Architecture, Connection, MemoryDefinition, MemoryEndpoint, ProcessorDefinition,
};

fn build(
    indices: Vec<mlar_rust::arch::EndpointIndex>,
) -> Result<Architecture, mlar_rust::ArchitectureError> {
    Architecture::builder("flat")
        .axis("cluster", 2)
        .axis("core", 3)
        .memory_definition(MemoryDefinition::new("L1", 1024, 16).with_banking(2))
        .place_memory("L1", mlar_rust::MemoryDomain::L1, ["cluster", "core"])
        .processor_definition(ProcessorDefinition::new("lane", "", vec![]))
        .connect(
            "lane",
            Connection::new(["core"]).input(
                "input",
                MemoryEndpoint::new("L1", indices).with_bank(AffineExpr::constant(1)),
            ),
        )
        .build()
}

#[test]
fn full_rank_column_and_bank_select_every_parent() {
    let architecture = build(vec![All, Expression(AffineExpr::variable("core"))]).unwrap();
    let memory = architecture.memory("L1").unwrap();
    assert_eq!(
        memory.points().collect::<Vec<_>>(),
        vec![
            vec![0, 0],
            vec![0, 1],
            vec![0, 2],
            vec![1, 0],
            vec![1, 1],
            vec![1, 2]
        ]
    );
    let instances = architecture.connection_instances(&architecture.processors()[0]);
    assert_eq!(instances.len(), 3);
    assert_eq!(
        instances[2].inputs[0].indices,
        vec![ResolvedEndpointIndex::All, ResolvedEndpointIndex::Index(2)]
    );
    assert_eq!(instances[2].inputs[0].bank, Some(1));
    let json = serde_json::to_value(&architecture).unwrap();
    assert_eq!(json["memories"][0]["axes"][0]["name"], "cluster");
    assert!(json["memories"][0].get("levels").is_none());
    assert!(
        build(vec![All])
            .unwrap_err()
            .to_string()
            .contains("memory expects 2")
    );
    assert!(build(vec![All, All, All]).is_err());
}

#[test]
fn whole_selection_is_explicit_and_rank_zero_has_one_instance() {
    let architecture = Architecture::builder("scalar")
        .memory_definition(MemoryDefinition::new("M", 64, 16))
        .place_memory("M", mlar_rust::MemoryDomain::L1, Vec::<String>::new())
        .build()
        .unwrap();
    let memory = architecture.memory("M").unwrap();
    assert_eq!(memory.points().collect::<Vec<_>>(), vec![Vec::<u64>::new()]);
    assert!(MemoryEndpoint::whole(memory).indices.is_empty());
    let memory = build(vec![All, All]).unwrap();
    assert_eq!(
        MemoryEndpoint::whole(memory.memory("L1").unwrap()).indices,
        vec![All, All]
    );
}

#[test]
fn canonical_native_boundary_rejects_compact_source_and_old_shapes() {
    let mut json = serde_json::to_value(build(vec![All, All]).unwrap()).unwrap();
    json["memories"][0]["levels"] = serde_json::json!([[{"name":"cluster","extent":2}]]);
    assert!(serde_json::from_value::<Architecture>(json).is_err());
    let compact = ProcessorDefinition::new(
        "compact",
        "func @copy(in a: f16[L], out b: f16[L]) { loom.copy %a to %b }",
        vec![],
    );
    assert!(compact.validate().is_err());
}
