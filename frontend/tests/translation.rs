use mlar_frontend::{ArchitectureBuilder, Connection, ProcessorDefinition, parse_loom_source};
use mlar_rust::{FuncPerfModel, MemoryDefinition, OperationModel};

fn definition(source: &str) -> ProcessorDefinition {
    let module = parse_loom_source(source).unwrap();
    ProcessorDefinition::new(
        "broadcast",
        source,
        module
            .functions
            .into_iter()
            .map(|func| {
                OperationModel::new(
                    func,
                    FuncPerfModel::builder()
                        .symbols(["L"])
                        .simple_time_cost(1_i64.into(), mlar_rust::Expr::sym("L"), 32_i64.into())
                        .build(),
                )
            })
            .collect(),
    )
}

#[test]
fn hierarchy_and_flat_authoring_produce_identical_canonical_selections() {
    let make = |hierarchy| {
        let builder = ArchitectureBuilder::new("same")
            .axis("cluster", 2)
            .axis("core", 3)
            .memory_definition(MemoryDefinition::new("M", 1024, 16).with_banking(2))
            .processor_definition(ProcessorDefinition::new("lane", "", vec![]));
        let builder = if hierarchy {
            builder.place_memory_levels("M", "M", vec![vec!["cluster".into()], vec!["core".into()]])
        } else {
            builder.place_memory("M", ["cluster", "core"])
        };
        builder
            .connect(
                "lane",
                Connection::parse(
                    ["core"],
                    [if hierarchy {
                        "M[:][core].bank[1]"
                    } else {
                        "M[:, core].bank[1]"
                    }],
                    [],
                )
                .unwrap(),
            )
            .build()
            .unwrap()
    };
    assert_eq!(
        serde_json::to_value(make(true)).unwrap(),
        serde_json::to_value(make(false)).unwrap()
    );
    let direct = mlar_rust::Architecture::builder("same")
        .axis("cluster", 2)
        .axis("core", 3)
        .memory_definition(MemoryDefinition::new("M", 1024, 16).with_banking(2))
        .place_memory("M", ["cluster", "core"])
        .processor_definition(mlar_rust::ProcessorDefinition::new("lane", "", vec![]))
        .connect(
            "lane",
            mlar_rust::Connection::new(
                ["core"],
                vec![
                    mlar_rust::MemoryEndpoint::new(
                        "M",
                        vec![
                            mlar_rust::arch::EndpointIndex::All,
                            mlar_rust::arch::EndpointIndex::Expression(
                                mlar_rust::AffineExpr::variable("core"),
                            ),
                        ],
                    )
                    .with_bank(mlar_rust::AffineExpr::constant(1)),
                ],
                vec![],
            ),
        )
        .build()
        .unwrap();
    assert_eq!(
        serde_json::to_value(direct).unwrap(),
        serde_json::to_value(make(true)).unwrap()
    );
}

#[test]
fn differing_selected_extents_specialize_bodies_but_identical_contexts_reuse() {
    let source = "func @bcst(in src: f16[L], out dst: f16[L]) {\n loom.broadcast %src to %dst\n}";
    let architecture = ArchitectureBuilder::new("contexts")
        .axis("x", 2)
        .axis("y", 3)
        .memory_definition(MemoryDefinition::new("M", 1024, 16))
        .place_memory("M", ["x", "y"])
        .processor_definition(definition(source))
        .connect_as(
            "whole",
            "broadcast",
            Connection::parse([], ["M"], ["M"]).unwrap(),
        )
        .connect_as(
            "same",
            "broadcast",
            Connection::parse([], ["M"], ["M"]).unwrap(),
        )
        .connect_as(
            "column",
            "broadcast",
            Connection::parse(["x"], ["M[x, :]"], ["M[x, :]"]).unwrap(),
        )
        .build()
        .unwrap();
    assert_eq!(architecture.processor_definitions().len(), 2);
    assert_eq!(
        architecture.processors()[0].definition_name(),
        architecture.processors()[1].definition_name()
    );
    assert_ne!(
        architecture.processors()[0].definition_name(),
        architecture.processors()[2].definition_name()
    );
    let whole = architecture
        .processor_definition(architecture.processors()[0].definition_name())
        .unwrap();
    let column = architecture
        .processor_definition(architecture.processors()[2].definition_name())
        .unwrap();
    assert!(
        whole.source().contains("area: [2, 3]"),
        "{}",
        whole.source()
    );
    assert!(column.source().contains("area: [3]"), "{}", column.source());
    let path = std::env::temp_dir().join(format!("mlar-artifact-{}.json", std::process::id()));
    mlar_frontend::write_artifact(&architecture, &path).unwrap();
    let decoded: mlar_rust::Architecture =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    std::fs::remove_file(path).unwrap();
    assert_eq!(
        serde_json::to_value(decoded).unwrap(),
        serde_json::to_value(architecture).unwrap()
    );
}

#[test]
fn explicit_symbolic_extent_stays_symbolic_and_reuses_across_contexts() {
    let source = "func @bcst(in src: f16[L], out dst: f16[L]) {\n %k = loom.sym @k : index\n loom.broadcast %src to %dst extent: [k]\n}";
    let architecture = ArchitectureBuilder::new("symbolic")
        .axis("x", 2)
        .axis("y", 3)
        .memory_definition(MemoryDefinition::new("M", 1024, 16))
        .place_memory("M", ["x", "y"])
        .processor_definition(definition(source))
        .connect_as(
            "whole",
            "broadcast",
            Connection::parse([], ["M"], ["M"]).unwrap(),
        )
        .connect_as(
            "column",
            "broadcast",
            Connection::parse(["x"], ["M[x, :]"], ["M[x, :]"]).unwrap(),
        )
        .build()
        .unwrap();
    assert_eq!(architecture.processor_definitions().len(), 1);
    assert!(
        architecture.processor_definitions()[0]
            .source()
            .contains("area: [%k]")
    );
}

#[test]
fn inconsistent_authoring_metadata_is_rejected_before_lowering() {
    let source = "func @bcst(in src: f16[L], out dst: f16[L]) {\n loom.broadcast %src to %dst\n}";
    let mut function = parse_loom_source(source).unwrap().functions.remove(0);
    function.symbols.push(mlar_rust::Sym::new("undeclared"));
    let model = FuncPerfModel::builder()
        .symbols(["L"])
        .simple_time_cost(1_i64.into(), mlar_rust::Expr::sym("L"), 32_i64.into())
        .build();
    let error = ArchitectureBuilder::new("bad")
        .memory_definition(MemoryDefinition::new("M", 1024, 16))
        .place_memory("M", Vec::<String>::new())
        .processor_definition(ProcessorDefinition::new(
            "broadcast",
            source,
            vec![OperationModel::new(function, model)],
        ))
        .connect("broadcast", Connection::parse([], ["M"], ["M"]).unwrap())
        .build()
        .unwrap_err();
    assert!(error.to_string().contains("metadata disagrees"));
}
