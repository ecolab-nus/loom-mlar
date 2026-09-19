use mlar_frontend::Connection;
use mlar_frontend::ProcessorDefinition;
use mlar_frontend::selection::MemoryEndpoint;
use std::path::Path;

use mlar_frontend::{ChipYaml, ProcessorYaml};
use mlar_rust::arch::{EndpointIndex, ProcessorSelectionError};
use mlar_rust::{
    AdlExportError, Architecture, Axis, Banking, MemoryDefinition, MemoryTechnology,
    ProcessorSelector, ProcessorType, ResolvedEndpointIndex, architecture_to_mlir,
};

#[test]
fn chip_yaml_uses_named_processor_placements() {
    ChipYaml::from_yaml_str(
        r#"
name: syntax
memories:
  L1: [x]
processors:
  lane:
    definition: lane.yaml
    domain: [x]
    inputs: {data: "L1[x]"}
    outputs: {result: "L1[x]"}
"#,
    )
    .expect("current chip syntax");

    assert!(
        ChipYaml::from_yaml_str(
            r#"
name: obsolete
memories:
  L1: [x]
processor:
  lane.yaml:
    - inputs: ["L1[x]"]
      outputs: ["L1[x]"]
"#
        )
        .is_err()
    );
}

fn fixture_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/indexed-affine")
}

#[test]
fn named_operand_ports_bind_distinct_connected_technologies() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/typed-memory");
    let architecture = mlar_frontend::archs::load_arch(&dir).expect("typed-memory fixture");
    assert_eq!(
        architecture
            .memory_definition(architecture.memory("L1_gcram").unwrap())
            .unwrap()
            .capacity,
        65_536
    );
    assert_eq!(
        architecture
            .memory_definition(architecture.memory("L1_rram").unwrap())
            .unwrap()
            .capacity,
        262_144
    );

    let mlir = architecture_to_mlir(&architecture).expect("typed memories should export");
    assert!(mlir.contains("loom.bind_mem %lhs, @mem_L1_gcram"));
    assert!(mlir.contains("loom.bind_mem %rhs, @mem_L1_rram"));
    assert!(mlir.contains("%rhs: memref<?xf16, 1>"));

    let definition = ProcessorYaml::from_file(dir.join("mixed_lane.yaml"))
        .and_then(|yaml| yaml.build_definition(dir.join("mixed_lane.yaml")))
        .unwrap();
    let named = mlar_frontend::ArchitectureBuilder::new("named")
        .memory_definition(
            MemoryDefinition::new("cache_a", 1024, 16)
                .with_technology(MemoryTechnology::new("gcram", 0)),
        )
        .memory_definition(
            MemoryDefinition::new("cache_b", 1024, 16)
                .with_technology(MemoryTechnology::new("gcram", 0)),
        )
        .place_memory("cache_a", std::iter::empty::<&str>())
        .place_memory("cache_b", std::iter::empty::<&str>())
        .processor_definition(definition)
        .connect(
            "mixed_lane",
            Connection::parse_named(
                [],
                [("rram", "cache_a"), ("gcram", "cache_b")],
                [("result", "cache_a")],
            )
            .unwrap(),
        )
        .build()
        .expect("named ports do not depend on technology uniqueness");
    let mlir = architecture_to_mlir(&named).unwrap();
    assert!(mlir.contains("loom.bind_mem %lhs, @mem_cache_b"));
    assert!(mlir.contains("loom.bind_mem %rhs, @mem_cache_a"));
}

#[test]
fn one_native_source_specializes_per_connection() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/native-specialization");
    let architecture = mlar_frontend::load_arch(dir).unwrap();
    let sr = architecture.processor_definition("lane").unwrap();
    let rs = architecture.processor_definition("lane__1").unwrap();
    assert!(sr.source().contains("memref<?xf16, 1>"));
    assert!(rs.source().contains("%arg0: memref<?xf16, 1>"));
    assert_ne!(sr.source(), rs.source());
    assert_eq!(
        serde_json::to_value(&sr.operations()[0].perf).unwrap(),
        serde_json::to_value(&rs.operations()[0].perf).unwrap()
    );
}

#[test]
fn descriptive_and_imperative_architectures_are_canonical_equivalents() {
    let dir = fixture_dir();
    let descriptive = mlar_frontend::archs::load_arch(&dir).expect("fixture should load");
    let matrix = ProcessorYaml::from_file(dir.join("matrix_lane.yaml"))
        .and_then(|yaml| yaml.build_definition(dir.join("matrix_lane.yaml")))
        .expect("matrix definition");
    let dma = ProcessorYaml::from_file(dir.join("dma.yaml"))
        .and_then(|yaml| yaml.build_definition(dir.join("dma.yaml")))
        .expect("DMA definition");
    assert_eq!(
        dma.name(),
        "dma",
        "definition name defaults to the file stem"
    );

    let connection = |domain: &[&str], inputs: &[&str], outputs: &[&str]| {
        let input_names = if inputs.len() == 2 {
            vec!["lhs", "rhs"]
        } else {
            vec!["src"]
        };
        let output_names = if inputs.len() == 2 {
            vec!["result"]
        } else {
            vec!["dst"]
        };
        Connection::parse_named(
            domain.iter().copied(),
            input_names.into_iter().zip(inputs.iter().copied()),
            output_names.into_iter().zip(outputs.iter().copied()),
        )
        .unwrap()
    };
    let imperative = mlar_frontend::ArchitectureBuilder::new("mesh_system")
        .axis("channel", 2)
        .axis("lx", 2)
        .axis("ly", 2)
        .axis("x", 4)
        .axis("y", 4)
        .memory_definition(MemoryDefinition::new("DRAM", 1_073_741_824, 64))
        .memory_definition(MemoryDefinition::new("L1", 65_536, 16).with_banking(8))
        .memory_definition(MemoryDefinition::new("L2", 1_048_576, 64).with_banking(8))
        .place_memory("DRAM", ["channel"])
        .place_memory("L1", ["x", "y"])
        .place_memory("L2", ["lx", "ly"])
        .processor_definition(matrix)
        .processor_definition(dma)
        .connect(
            "matrix_lane",
            connection(&["x", "y"], &["L1[x, y]", "L1[x, y]"], &["L1[x, y]"]),
        )
        .connect_as(
            "l1_to_l2",
            "dma",
            connection(
                &["x", "y"],
                &["L1[x, y]"],
                &["L2[x floordiv 2, y floordiv 2]"],
            ),
        )
        .connect_as(
            "east_dma",
            "dma",
            connection(&["x", "y"], &["L1[x, y]"], &["L1[(x + 1) mod 4, y]"]),
        )
        .connect_as(
            "dram_to_l1",
            "dma",
            connection(
                &["x", "y"],
                &["DRAM[x mod 2]"],
                &["L1[x, y].bank[(x + y) mod 8]"],
            ),
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
    assert_eq!(endpoint.indices.len(), 1);
    assert_eq!(endpoint.indices[0].len(), 2);
    assert!(endpoint.bank.is_some());
    assert_eq!(
        MemoryEndpoint::parse("L1[:, :]")
            .unwrap()
            .indices
            .iter()
            .flatten()
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
    let architecture = mlar_frontend::ArchitectureBuilder::new("drop_test")
        .axis("x", 4)
        .memory_definition(MemoryDefinition::new("L1", 1024, 16))
        .place_memory("L1", ["x"])
        .processor_definition(dma)
        .connect(
            "dma",
            Connection::new(["x"])
                .input("src", MemoryEndpoint::parse("L1[x]").unwrap())
                .output("dst", MemoryEndpoint::parse("L1[x + 1]").unwrap()),
        )
        .build()
        .expect("architecture should build");
    assert_eq!(
        architecture.processors()[0].instances(&architecture).len(),
        3
    );

    let dma = architecture.processor_array("dma").unwrap();
    assert_eq!(dma.select_all(&architecture).len(), 3);
    assert!(
        dma.select(&architecture, [ProcessorSelector::Index(3)])
            .unwrap()
            .is_empty(),
        "an in-domain coordinate can be absent from a sparse relation"
    );
    assert!(matches!(
        dma.select(&architecture, [ProcessorSelector::Index(4)]),
        Err(ProcessorSelectionError::OutOfBounds { .. })
    ));
}

#[test]
fn processor_array_selection_is_uniform_for_all_subset_and_point_queries() {
    use ProcessorSelector::{All, Index};

    let architecture = mlar_frontend::archs::load_arch(fixture_dir()).unwrap();
    let lanes = architecture.processor_array("matrix_lane").unwrap();

    let all = lanes.select(&architecture, [All, All]).unwrap();
    assert_eq!(all.len(), 16);
    assert_eq!(
        all.free_domain().map(Axis::name).collect::<Vec<_>>(),
        ["x", "y"]
    );

    let row = lanes.select(&architecture, [Index(2), All]).unwrap();
    assert_eq!(row.len(), 4);
    assert_eq!(row.free_domain().map(Axis::name).collect::<Vec<_>>(), ["y"]);
    assert!(
        row.instances()
            .all(|instance| instance.variables.get("x") == Some(&2))
    );

    let column = lanes.select(&architecture, [All, Index(3)]).unwrap();
    assert_eq!(column.len(), 4);
    assert!(
        column
            .instances()
            .all(|instance| instance.variables.get("y") == Some(&3))
    );

    let point = lanes.select(&architecture, [Index(2), Index(3)]).unwrap();
    assert_eq!(point.len(), 1);
    assert_eq!(point.free_domain().count(), 0);
    assert_eq!(point.array().name(), "matrix_lane");
    assert_eq!(point.into_iter().next().unwrap().variables["x"], 2);

    assert!(matches!(
        lanes.select(&architecture, [All]),
        Err(ProcessorSelectionError::RankMismatch {
            expected: 2,
            actual: 1
        })
    ));
}

#[test]
fn definition_placements_and_memory_points_enumerate_in_declaration_order() {
    let architecture = mlar_frontend::archs::load_arch(fixture_dir()).unwrap();

    assert_eq!(
        architecture
            .processors_of("dma")
            .map(|processor| processor.name())
            .collect::<Vec<_>>(),
        ["l1_to_l2", "east_dma", "dram_to_l1"]
    );
    assert_eq!(architecture.processors_of("matrix_lane").count(), 1);
    assert_eq!(architecture.processors_of("absent").count(), 0);

    let l1 = architecture.memory("L1").unwrap();
    let points = l1.points().collect::<Vec<_>>();
    assert_eq!(points.len() as u64, l1.instances());
    assert_eq!(points[0], [0, 0]);
    assert_eq!(points[1], [0, 1]);
    assert_eq!(points[4], [1, 0]);
    assert_eq!(points.last().unwrap(), &[3, 3]);

    let shared = mlar_frontend::ArchitectureBuilder::new("shared_definition")
        .axis("x", 2)
        .memory_definition(MemoryDefinition::new("L1", 4096, 64))
        .place_memory_as("l1_a", "L1", ["x"])
        .place_memory_as("l1_b", "L1", ["x"])
        .build()
        .unwrap();
    assert_eq!(
        shared
            .memories_of("L1")
            .map(|memory| memory.name())
            .collect::<Vec<_>>(),
        ["l1_a", "l1_b"]
    );
    assert_eq!(shared.memories_of("l1_a").count(), 0);

    let scalar = mlar_frontend::ArchitectureBuilder::new("scalar")
        .memory_definition(MemoryDefinition::new("regs", 256, 4))
        .place_memory("regs", Vec::<String>::new())
        .build()
        .unwrap();
    assert_eq!(
        scalar.memory("regs").unwrap().points().collect::<Vec<_>>(),
        [Vec::<u64>::new()]
    );
}

#[test]
fn memory_and_endpoint_validation_is_strict() {
    assert!(
        MemoryDefinition {
            name: "bad_word".into(),
            capacity: 65,
            word_size: 16,
            technology: None,
            banking: None,
        }
        .validate()
        .is_err()
    );
    assert!(
        MemoryDefinition {
            name: "bad_banks".into(),
            capacity: 64,
            word_size: 16,
            technology: None,
            banking: Some(Banking::new(8)),
        }
        .validate()
        .is_err()
    );

    let error = mlar_frontend::ArchitectureBuilder::new("arity")
        .axis("x", 2)
        .memory_definition(MemoryDefinition::new("L1", 1024, 16))
        .place_memory("L1", ["x"])
        .processor_definition(
            ProcessorYaml::from_file(fixture_dir().join("dma.yaml"))
                .and_then(|yaml| yaml.build_definition(fixture_dir().join("dma.yaml")))
                .unwrap(),
        )
        .connect(
            "dma",
            Connection::new(["x"]).input("src", MemoryEndpoint::parse("L1[x, x]").unwrap()),
        )
        .build()
        .expect_err("wrong group arity must fail");
    assert!(error.to_string().contains("expects 1"));
}

#[test]
fn templates_record_symbolic_shapes_and_roles() {
    let path = fixture_dir().join("matrix_lane.yaml");
    let definition = ProcessorYaml::from_file(&path)
        .and_then(|yaml| yaml.build_definition(&path))
        .expect("template definition should build");
    let function = &definition.functions()[0].func;
    let details = function.mlir_details.as_ref().unwrap();
    assert_eq!(function.symbols.len(), 3);
    assert_eq!(details.source_memrefs, ["lhs", "rhs"]);
    assert_eq!(details.target_memrefs, ["out"]);
    assert_eq!(details.memref_symbol_bindings.len(), 3);
}

#[test]
fn incompatible_type_hints_fail_the_whole_export() {
    let architecture = mlar_frontend::archs::load_arch(fixture_dir())
        .unwrap()
        .with_processor_type("dma", Some(ProcessorType::Compute))
        .unwrap();
    assert!(matches!(
        architecture_to_mlir(&architecture),
        Err(AdlExportError::ComputeContainsMovement { .. })
    ));

    let architecture = mlar_frontend::archs::load_arch(fixture_dir())
        .unwrap()
        .with_processor_type("matrix_lane", Some(ProcessorType::DataMover))
        .unwrap();
    assert!(matches!(
        architecture_to_mlir(&architecture),
        Err(AdlExportError::DataMoverContainsCompute { .. })
    ));
}

#[test]
fn canonical_architecture_json_round_trips_for_abi_use() {
    let architecture = mlar_frontend::archs::load_arch(fixture_dir()).unwrap();
    let json = serde_json::to_string(&architecture).unwrap();
    let decoded: Architecture = serde_json::from_str(&json).unwrap();
    assert_eq!(
        serde_json::to_value(decoded).unwrap(),
        serde_json::to_value(architecture).unwrap()
    );
}

#[test]
fn canonical_architecture_json_omits_instances_and_rejects_inconsistent_axes() {
    let architecture = mlar_frontend::archs::load_arch(fixture_dir()).unwrap();
    let mut json = serde_json::to_value(&architecture).unwrap();
    assert!(json["processors"][0].get("instances").is_none());
    json["processors"][0]["axes"] = serde_json::json!([]);
    let error = serde_json::from_value::<Architecture>(json).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("axes inconsistent with its connection")
    );
}

#[test]
fn canonical_architecture_rejects_processor_source_model_drift() {
    let architecture = mlar_frontend::archs::load_arch(fixture_dir()).unwrap();
    let mut json = serde_json::to_value(&architecture).unwrap();
    json["processor_definitions"][0]["functions"][0]["func"]["name"] =
        serde_json::json!("not_in_source");
    let error = serde_json::from_value::<Architecture>(json).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("source functions disagree with its operation models")
    );
}

#[test]
fn memory_selections_target_placed_memory_names() {
    mlar_frontend::ArchitectureBuilder::new("renamed")
        .axis("x", 2)
        .axis("y", 2)
        .memory_definition(MemoryDefinition::new("L1", 1024, 16))
        .place_memory_as("local_l1", "L1", ["x", "y"])
        .processor_definition(ProcessorDefinition::new("lane", "", Vec::new()))
        .connect(
            "lane",
            Connection::parse(["y"], ["local_l1[:, y]"], []).unwrap(),
        )
        .build()
        .expect("slice resolves against the placed name");
}

#[test]
fn connection_domain_order_and_resolved_regions_are_explicit() {
    let architecture = mlar_frontend::ArchitectureBuilder::new("regions")
        .axis("x", 2)
        .axis("y", 3)
        .memory_definition(MemoryDefinition::new("L1", 1024, 16))
        .place_memory_levels("L1", "L1", vec![vec!["x".into()], vec!["y".into()]])
        .processor_definition(ProcessorDefinition::new("lane", "", Vec::new()))
        .connect_as(
            "replicated_lane",
            "lane",
            Connection::new(["y", "x"]).input("input", MemoryEndpoint::parse("L1[x]").unwrap()),
        )
        .build()
        .expect("explicit domain may include replication axes");

    let lanes = architecture.processor_array("replicated_lane").unwrap();
    assert_eq!(
        lanes.axes().iter().map(Axis::name).collect::<Vec<_>>(),
        ["y", "x"]
    );
    let instance = lanes
        .instances(&architecture)
        .into_iter()
        .find(|instance| instance.variables["y"] == 2 && instance.variables["x"] == 1)
        .unwrap();
    assert_eq!(
        instance.inputs[0].indices,
        [ResolvedEndpointIndex::Index(1), ResolvedEndpointIndex::All]
    );
}

#[test]
fn endpoint_variables_must_be_declared_in_the_connection_domain() {
    let error = mlar_frontend::ArchitectureBuilder::new("domain")
        .axis("x", 2)
        .memory_definition(MemoryDefinition::new("L1", 1024, 16))
        .place_memory("L1", ["x"])
        .processor_definition(ProcessorDefinition::new("lane", "", Vec::new()))
        .connect(
            "lane",
            Connection::new(std::iter::empty::<&str>())
                .input("input", MemoryEndpoint::parse("L1[x]").unwrap()),
        )
        .build()
        .unwrap_err();
    assert!(error.to_string().contains("not in its declared domain"));
}

// Builder load failures are deferred until `build`.
#[test]
fn processor_load_failures_surface_at_build() {
    let missing_dir = mlar_frontend::ArchitectureBuilder::new("no_dir")
        .processor("lane")
        .build()
        .unwrap_err()
        .to_string();
    assert!(
        missing_dir.contains("processor_source_dir"),
        "expected a source-directory hint, got: {missing_dir}"
    );

    let missing_file = mlar_frontend::ArchitectureBuilder::new("no_file")
        .processor_source_dir(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/2d_mesh/processors"),
        )
        .processor("not_a_processor")
        .build()
        .unwrap_err()
        .to_string();
    assert!(
        missing_file.contains("not_a_processor"),
        "expected the failing name, got: {missing_file}"
    );
}

/// One level per bracket group, emitted inner to outer, each symbol named for
/// the axes that index it — so a symbol never encodes its depth.
#[test]
fn nested_memory_levels_emit_one_axis_named_array_each() {
    let architecture = mlar_frontend::ArchitectureBuilder::new("levels")
        .axis("cluster", 2)
        .axis("core", 4)
        .memory_definition(MemoryDefinition::new("L1", 1024, 16))
        .place_memory_levels(
            "L1",
            "L1",
            vec![vec!["cluster".into()], vec!["core".into()]],
        )
        .processor_definition(ProcessorDefinition::new("lane", "", Vec::new()))
        .connect(
            "lane",
            Connection::new(["cluster", "core"])
                .input("input", MemoryEndpoint::parse("L1[cluster][core]").unwrap())
                .output(
                    "result",
                    MemoryEndpoint::parse("L1[cluster][core]").unwrap(),
                ),
        )
        .build()
        .expect("nested levels build")
        .with_processor_type("lane", Some(ProcessorType::Compute))
        .expect("lane is a compute processor");

    let l1 = architecture.memory("L1").expect("L1");
    assert_eq!(l1.rank(), 2);

    let mlir = mlar_rust::architecture_to_mlir_unchecked(&architecture).expect("levels export");
    assert_eq!(
        mlir.lines()
            .filter(|line| line.contains("adl.memory.array"))
            .count(),
        1
    );
    assert!(mlir.contains(r#"adl.memory.array "mem_L1", ["#));
    // The leaf is one instance, so the bank carries the full-rank symbol.
    assert!(mlir.contains(r#"adl.memory.bank "mem_L1_instance""#));
}

/// A flat array allows dimension slices even when ADL has no slice handle.
#[test]
fn flat_dimension_slices_are_valid_but_not_lowerable_as_whole_levels() {
    let architecture = mlar_frontend::ArchitectureBuilder::new("slice")
        .axis("cluster", 2)
        .axis("core", 4)
        .memory_definition(MemoryDefinition::new("L1", 1024, 16))
        .place_memory("L1", ["cluster", "core"])
        .processor_definition(
            ProcessorDefinition::new("dma", "", Vec::new()).with_type(ProcessorType::DataMover),
        )
        .connect(
            "dma",
            Connection::parse(["cluster"], ["L1[cluster, :]"], []).unwrap(),
        )
        .build()
        .expect("slices do not require hierarchy boundaries");
    assert!(matches!(
        mlar_rust::architecture_to_mlir_unchecked(&architecture),
        Err(AdlExportError::UnsupportedMemorySelection { .. })
    ));
}

#[test]
fn a_level_may_not_repeat_an_axis() {
    let error = mlar_frontend::ArchitectureBuilder::new("repeat")
        .axis("x", 2)
        .memory_definition(MemoryDefinition::new("L1", 1024, 16))
        .place_memory_levels("L1", "L1", vec![vec!["x".into()], vec!["x".into()]])
        .build()
        .expect_err("an axis may index a memory only once");

    assert!(error.to_string().contains("repeats an axis"));
}
