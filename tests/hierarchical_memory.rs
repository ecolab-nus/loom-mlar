use mlar_rust::arch::{ArchitectureError, EndpointIndex};
use mlar_rust::{
    AdlExportError, Architecture, Connection, MemoryDefinition, MemoryEndpoint, NetworkInterface,
    NetworkTopology, ProcessorDefinition, ProcessorType, ResolvedEndpointIndex,
    architecture_to_mlir_unchecked,
};

fn architecture(endpoints: &[&str]) -> Result<Architecture, ArchitectureError> {
    Architecture::builder("hierarchical")
        .axis("x", 2)
        .axis("y", 3)
        .axis("core", 4)
        .memory_definition(MemoryDefinition::new("L1", 1024, 16).with_banking(2))
        .place_memory_levels(
            "L1",
            "L1",
            vec![vec!["x".into(), "y".into()], vec!["core".into()]],
        )
        .processor_definition(
            ProcessorDefinition::new("lane", "", Vec::new()).with_type(ProcessorType::Compute),
        )
        .connect(
            "lane",
            Connection::parse(
                ["x", "y", "core"],
                endpoints.iter().copied(),
                endpoints.iter().copied(),
            )
            .unwrap(),
        )
        .build()
}

#[test]
fn parser_preserves_groups_and_rejects_malformed_traversals() {
    let endpoint = MemoryEndpoint::parse(" L1 [x, :] [core mod 4] .bank[1] ").unwrap();
    assert_eq!(endpoint.indices.len(), 2);
    assert_eq!(endpoint.indices[0].len(), 2);
    assert_eq!(endpoint.indices[0][1], EndpointIndex::All);
    assert_eq!(endpoint.indices[1].len(), 1);
    assert_eq!(
        endpoint.variables().into_iter().collect::<Vec<_>>(),
        ["core", "x"]
    );
    assert!(MemoryEndpoint::parse("L1").unwrap().indices.is_empty());
    for invalid in [
        "L1[]",
        "L1[x,]",
        "L1[x][",
        "L1[[x]]",
        "L1[x]]",
        "L1[x]junk",
        "L1[x].[core]",
        "L1[x].bank[]",
        "L1[x].bank[1][core]",
        "L1[x].bank[1].bank[0]",
    ] {
        assert!(MemoryEndpoint::parse(invalid).is_err(), "{invalid}");
    }
}

#[test]
fn resolution_preserves_subtrees_and_traverses_all_selected_parents() {
    use ResolvedEndpointIndex::{All, Index};
    let architecture = architecture(&[
        "L1",
        "L1[x, y]",
        "L1[x, y][core]",
        "L1[:, y][core]",
        "L1[x, :][core]",
        "L1[:, :][core]",
        "L1[x, y][:]",
        "L1[:, y][core].bank[1]",
    ])
    .unwrap();
    let instances = architecture.processors()[0].instances(&architecture);
    assert_eq!(instances.len(), 24);
    let instance = instances
        .iter()
        .find(|instance| {
            instance.variables["x"] == 1
                && instance.variables["y"] == 2
                && instance.variables["core"] == 3
        })
        .unwrap();
    let expected = [
        vec![],
        vec![vec![Index(1), Index(2)]],
        vec![vec![Index(1), Index(2)], vec![Index(3)]],
        vec![vec![All, Index(2)], vec![Index(3)]],
        vec![vec![Index(1), All], vec![Index(3)]],
        vec![vec![All, All], vec![Index(3)]],
        vec![vec![Index(1), Index(2)], vec![All]],
        vec![vec![All, Index(2)], vec![Index(3)]],
    ];
    for (location, expected) in instance.inputs.iter().zip(expected) {
        assert_eq!(location.indices, expected);
    }
    assert_eq!(instance.inputs[7].bank, Some(1));
}

#[test]
fn each_group_matches_one_level_and_banks_belong_to_leaves() {
    for (endpoint, expected) in [
        ("L1[x]", "level expects 2"),
        ("L1[x, y, core]", "level expects 2"),
        ("L1[x, y][core, x]", "level expects 1"),
        ("L1[x, y][core][0]", "placed memory has 2 levels"),
        ("L1.bank[0]", "index every memory level"),
        ("L1[x, y].bank[0]", "index every memory level"),
        ("L1[x, y][core].bank[2]", "out of bounds"),
    ] {
        let error = architecture(&[endpoint]).unwrap_err().to_string();
        assert!(error.contains(expected), "{endpoint}: {error}");
    }
}

#[test]
fn deeper_affine_indices_still_filter_bounds_and_wrap() {
    let architecture = architecture(&["L1[:, y][core + 1]"]).unwrap();
    let instances = architecture.processors()[0].instances(&architecture);
    assert_eq!(instances.len(), 18);
    assert!(
        instances
            .iter()
            .all(|instance| instance.variables["core"] < 3)
    );

    let architecture = self::architecture(&["L1[:, y][(core + 1) mod 4]"]).unwrap();
    let instances = architecture.processors()[0].instances(&architecture);
    assert_eq!(instances.len(), 24);
    let wrapped = instances
        .iter()
        .find(|instance| instance.variables["core"] == 3)
        .unwrap();
    assert_eq!(
        wrapped.inputs[0].indices[1],
        [ResolvedEndpointIndex::Index(0)]
    );
}

#[test]
fn adl_lowers_subtrees_and_rejects_unrepresentable_slices() {
    for (endpoint, symbol) in [
        ("L1", "mem_L1"),
        ("L1[:, :]", "mem_L1"),
        ("L1[:, :][:]", "mem_L1"),
        ("L1[x, y]", "mem_L1__x_y"),
        ("L1[x, y][:]", "mem_L1__x_y"),
        ("L1[x, y][core]", "mem_L1__x_y_core"),
    ] {
        let architecture = architecture(&[endpoint]).unwrap();
        let mlir = architecture_to_mlir_unchecked(&architecture).unwrap();
        let processor = mlir
            .lines()
            .find(|line| line.contains("= adl.processor."))
            .unwrap();
        let memory_line = mlir
            .lines()
            .find(|line| line.contains(&format!("\"{symbol}\"")))
            .unwrap();
        let ssa = memory_line.trim().split(" = ").next().unwrap();
        assert!(
            processor.trim().ends_with(&format!("from {ssa} to {ssa}")),
            "{endpoint}: {processor}"
        );
    }
    for endpoint in ["L1[:, y]", "L1[x, :]", "L1[:, :][core]", "L1[x, :][core]"] {
        let architecture = architecture(&[endpoint]).unwrap();
        assert!(
            matches!(
                architecture_to_mlir_unchecked(&architecture),
                Err(AdlExportError::UnsupportedMemorySelection { .. })
            ),
            "{endpoint}"
        );
    }
}

#[test]
fn hierarchical_endpoints_round_trip_and_revalidate() {
    let architecture = architecture(&["L1[:, y][core]", "L1[x, y]"]).unwrap();
    let mut json = serde_json::to_value(&architecture).unwrap();
    let decoded: Architecture = serde_json::from_value(json.clone()).unwrap();
    assert_eq!(serde_json::to_value(&decoded).unwrap(), json);
    assert_eq!(
        decoded.processors()[0].instances(&decoded),
        architecture.processors()[0].instances(&architecture)
    );
    json["processors"][0]["connection"]["inputs"][0]["indices"][0] = serde_json::json!(["all"]);
    let error = serde_json::from_value::<Architecture>(json)
        .unwrap_err()
        .to_string();
    assert!(error.contains("level expects 2"), "{error}");
}

#[test]
fn network_interfaces_use_the_same_hierarchical_validation() {
    let builder = |endpoint| {
        Architecture::builder("network")
            .axis("x", 2)
            .axis("core", 4)
            .memory_definition(MemoryDefinition::new("L1", 1024, 16))
            .place_memory_levels("L1", "L1", vec![vec!["x".into()], vec!["core".into()]])
            .network(
                NetworkTopology::new("noc", vec![]).with_interface(NetworkInterface::new(
                    "port",
                    MemoryEndpoint::parse(endpoint).unwrap(),
                )),
            )
    };
    let architecture = builder("L1[:][1]").build().unwrap();
    let mut json = serde_json::to_value(&architecture).unwrap();
    json["networks"][0]["interfaces"][0]["endpoint"]["indices"] =
        serde_json::json!([["all", "all"]]);
    let error = serde_json::from_value::<Architecture>(json)
        .unwrap_err()
        .to_string();
    assert!(error.contains("level expects 1"), "{error}");
    let error = builder("L1[0, 1]").build().unwrap_err().to_string();
    assert!(error.contains("level expects 1"), "{error}");
}
