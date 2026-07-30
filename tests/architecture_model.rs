use std::path::Path;

use mlar_rust::{
    AdlExportError, Architecture, Banking, ChipYaml, ConnectionSpec, EndpointIndex, MemoryCatalog,
    MemoryDefinition, MemoryEndpoint, NamedMemoryRegion, ProcessorSelectionError,
    ProcessorSelector, ProcessorType, ProcessorYaml, architecture_to_mlir, parse_loom_source,
};

#[test]
fn chip_yaml_uses_memories_and_processor_entries() {
    ChipYaml::from_yaml_str(
        r#"
name: syntax
memories:
  L1: [x]
processor:
  lane.yaml: []
"#,
    )
    .expect("current chip syntax");

    assert!(
        ChipYaml::from_yaml_str(
            r#"
name: obsolete
placement:
  L1: [x]
"#
        )
        .is_err()
    );
}

fn fixture_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/indexed-affine")
}

#[test]
fn descriptive_and_imperative_architectures_are_canonical_equivalents() {
    let dir = fixture_dir();
    let descriptive = mlar_rust::archs::load_arch(&dir).expect("fixture should load");
    let matrix = ProcessorYaml::from_file(dir.join("matrix_lane.yaml"))
        .and_then(|yaml| yaml.build_definition(dir.join("matrix_lane.yaml")))
        .expect("matrix definition");
    let dma = ProcessorYaml::from_file(dir.join("dma.yaml"))
        .and_then(|yaml| yaml.build_definition(dir.join("dma.yaml")))
        .expect("DMA definition");

    let catalog = MemoryCatalog {
        definitions: vec![
            MemoryDefinition::new("DRAM", ["channel"], 1_073_741_824, 64),
            MemoryDefinition::new("L1", ["row", "column"], 65_536, 16).with_banking(8),
            MemoryDefinition::new("L2", ["row", "column"], 1_048_576, 64).with_banking(8),
        ],
        regions: vec![NamedMemoryRegion::new(
            "L1_all",
            MemoryEndpoint::parse("L1[:, :]").unwrap(),
        )],
    };
    let connection = |inputs: &[&str], outputs: &[&str]| {
        ConnectionSpec::new(
            inputs
                .iter()
                .map(|endpoint| MemoryEndpoint::parse(endpoint).unwrap())
                .collect(),
            outputs
                .iter()
                .map(|endpoint| MemoryEndpoint::parse(endpoint).unwrap())
                .collect(),
        )
    };
    let imperative = Architecture::builder("mesh_system")
        .dimension("channel", 2)
        .dimension("lx", 2)
        .dimension("ly", 2)
        .dimension("x", 4)
        .dimension("y", 4)
        .memory_catalog(catalog)
        .place_memory("DRAM", ["channel"])
        .place_memory("L1", ["x", "y"])
        .place_memory("L2", ["lx", "ly"])
        .processor_definition(matrix)
        .processor_definition(dma)
        .connect(
            "matrix_lane",
            connection(&["L1[x, y]", "L1[x, y]"], &["L1[x, y]"]),
        )
        .connect(
            "dma",
            connection(&["L1[x, y]"], &["L2[x floordiv 2, y floordiv 2]"]),
        )
        .connect("dma", connection(&["L1[x, y]"], &["L1[(x + 1) mod 4, y]"]))
        .connect(
            "dma",
            connection(&["DRAM[x mod 2]"], &["L1[x, y].bank[(x + y) mod 8]"]),
        )
        .build()
        .expect("imperative architecture");

    assert_eq!(
        serde_json::to_value(&descriptive).unwrap(),
        serde_json::to_value(&imperative).unwrap()
    );
    assert_eq!(
        architecture_to_mlir(&descriptive).unwrap(),
        architecture_to_mlir(&imperative).unwrap()
    );
}

#[test]
fn endpoint_parser_handles_full_affine_subset_and_bank_selection() {
    let endpoint = MemoryEndpoint::parse("L1[(x + 3) floordiv 2, y ceildiv 2].bank[(x + y) mod 8]")
        .expect("endpoint should parse");
    assert_eq!(endpoint.indices.len(), 2);
    assert!(endpoint.bank.is_some());
    assert_eq!(
        MemoryEndpoint::parse("L1[:, :]")
            .unwrap()
            .indices
            .iter()
            .filter(|index| matches!(index, EndpointIndex::All))
            .count(),
        2
    );
}

#[test]
fn non_modular_out_of_bounds_points_are_dropped() {
    let dir = fixture_dir();
    let dma = ProcessorYaml::from_file(dir.join("dma.yaml"))
        .and_then(|yaml| yaml.build_definition(dir.join("dma.yaml")))
        .expect("DMA definition");
    let architecture = Architecture::builder("drop_test")
        .dimension("x", 4)
        .memory_definition(MemoryDefinition::new("L1", ["i"], 1024, 16))
        .place_memory("L1", ["x"])
        .processor_definition(dma)
        .connect(
            "dma",
            ConnectionSpec::new(
                vec![MemoryEndpoint::parse("L1[x]").unwrap()],
                vec![MemoryEndpoint::parse("L1[x + 1]").unwrap()],
            ),
        )
        .build()
        .expect("architecture should build");
    assert_eq!(architecture.processors[0].relation.instances.len(), 3);

    let dma = architecture.processor_array("dma").unwrap();
    assert_eq!(dma.select_all().len(), 3);
    assert!(
        dma.select([ProcessorSelector::Index(3)])
            .unwrap()
            .is_empty(),
        "an in-domain coordinate can be absent from a sparse relation"
    );
    assert!(matches!(
        dma.select([ProcessorSelector::Index(4)]),
        Err(ProcessorSelectionError::OutOfBounds { .. })
    ));
}

#[test]
fn processor_array_selection_is_uniform_for_all_subset_and_point_queries() {
    use ProcessorSelector::{All, Index};

    let architecture = mlar_rust::archs::load_arch(fixture_dir()).unwrap();
    let lanes = architecture.processor_array("matrix_lane").unwrap();

    let all = lanes.select([All, All]).unwrap();
    assert_eq!(all.len(), 16);
    assert_eq!(
        all.free_domain()
            .map(|dimension| dimension.name.as_str())
            .collect::<Vec<_>>(),
        ["x", "y"]
    );

    let row = lanes.select([Index(2), All]).unwrap();
    assert_eq!(row.len(), 4);
    assert_eq!(
        row.free_domain()
            .map(|dimension| dimension.name.as_str())
            .collect::<Vec<_>>(),
        ["y"]
    );
    assert!(
        row.instances()
            .all(|instance| instance.variables.get("x") == Some(&2))
    );

    let column = lanes.select([All, Index(3)]).unwrap();
    assert_eq!(column.len(), 4);
    assert!(
        column
            .instances()
            .all(|instance| instance.variables.get("y") == Some(&3))
    );

    let point = lanes.select([Index(2), Index(3)]).unwrap();
    assert_eq!(point.len(), 1);
    assert_eq!(point.free_domain().count(), 0);
    assert_eq!(point.array().name, "matrix_lane");
    assert_eq!(point.into_iter().next().unwrap().variables["x"], 2);

    assert!(matches!(
        lanes.select([All]),
        Err(ProcessorSelectionError::RankMismatch {
            expected: 2,
            actual: 1
        })
    ));
}

#[test]
fn memory_and_endpoint_validation_is_strict() {
    assert!(
        MemoryDefinition {
            name: "bad_word".into(),
            indices: Vec::new(),
            capacity: 65,
            word_size: 16,
            banking: None,
        }
        .validate()
        .is_err()
    );
    assert!(
        MemoryDefinition {
            name: "bad_banks".into(),
            indices: Vec::new(),
            capacity: 64,
            word_size: 16,
            banking: Some(Banking::new(8)),
        }
        .validate()
        .is_err()
    );

    let error = Architecture::builder("arity")
        .dimension("x", 2)
        .memory_definition(MemoryDefinition::new("L1", ["i"], 1024, 16))
        .place_memory("L1", ["x"])
        .processor_definition(
            ProcessorYaml::from_file(fixture_dir().join("dma.yaml"))
                .and_then(|yaml| yaml.build_definition(fixture_dir().join("dma.yaml")))
                .unwrap(),
        )
        .connect(
            "dma",
            ConnectionSpec::new(vec![MemoryEndpoint::parse("L1").unwrap()], vec![]),
        )
        .build()
        .expect_err("missing endpoint index must fail");
    assert!(error.to_string().contains("expects 1"));
}

#[test]
fn compact_loom_records_symbolic_shapes_and_roles() {
    let source = r#"
func @add {
  params: [M, N]
  ins:
    lhs: !loom.buffer<MxNxf16>
    rhs: !loom.buffer<MxNxf16>
  outs:
    out: !loom.buffer<MxNxf16>
  linalg.add ins(%lhs, %rhs) outs(%out)
}
"#;
    let module = parse_loom_source(source).expect("compact source should parse");
    let details = module.functions[0].mlir_details.as_ref().unwrap();
    assert_eq!(module.functions[0].symbols.len(), 2);
    assert_eq!(details.source_memrefs, ["lhs", "rhs"]);
    assert_eq!(details.target_memrefs, ["out"]);
    assert_eq!(details.memref_symbol_bindings.len(), 3);
}

#[test]
fn incompatible_type_hints_fail_the_whole_export() {
    let mut architecture = mlar_rust::archs::load_arch(fixture_dir()).unwrap();
    let dma = architecture
        .processor_definitions
        .iter_mut()
        .find(|definition| definition.name == "dma")
        .unwrap();
    dma.processor_type = Some(ProcessorType::Compute);
    assert!(matches!(
        architecture_to_mlir(&architecture),
        Err(AdlExportError::ComputeContainsMovement { .. })
    ));

    let mut architecture = mlar_rust::archs::load_arch(fixture_dir()).unwrap();
    let matrix = architecture
        .processor_definitions
        .iter_mut()
        .find(|definition| definition.name == "matrix_lane")
        .unwrap();
    matrix.processor_type = Some(ProcessorType::DataMover);
    assert!(matches!(
        architecture_to_mlir(&architecture),
        Err(AdlExportError::DataMoverContainsCompute { .. })
    ));
}

#[test]
fn canonical_architecture_json_round_trips_for_abi_use() {
    let architecture = mlar_rust::archs::load_arch(fixture_dir()).unwrap();
    let json = serde_json::to_string(&architecture).unwrap();
    let decoded: Architecture = serde_json::from_str(&json).unwrap();
    assert_eq!(
        serde_json::to_value(decoded).unwrap(),
        serde_json::to_value(architecture).unwrap()
    );
}
