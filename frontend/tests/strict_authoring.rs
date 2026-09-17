use mlar_frontend::Connection;
use mlar_frontend::ProcessorDefinition;
use mlar_frontend::selection::MemoryEndpoint;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use mlar_frontend::{ChipYaml, PerformanceYaml, ProcessorYaml};
use mlar_rust::{
    AffineExpr, AffineMap, Architecture, Axis, Expr, FuncPerfModel, MemoryDefinition, MlirFunc,
    NetworkInterface, NetworkLink, NetworkTopology, OperationModel, Sym,
};

struct Package(PathBuf);
impl Package {
    fn new(chip: &str, memory: &str) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "mlar-strict-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(path.join("chip.yaml"), chip).unwrap();
        std::fs::write(path.join("memory.yaml"), memory).unwrap();
        Self(path)
    }
}
impl Drop for Package {
    fn drop(&mut self) {
        std::fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn nested_authoring_objects_reject_unknown_fields() {
    for extra in [
        "memories:\n  L1:\n    definition: L1\n    axse: [x]\n",
        "processors:\n  lane:\n    definition: lane.yaml\n    domian: [x]\n",
        "resources:\n  - name: lock\n    capcity: 2\n",
        "networks:\n  - name: noc\n    dimensions: [x]\n    interafces: []\n",
        "networks:\n  - name: noc\n    dimensions: [x]\n    links:\n      - name: east\n        map: '[x] -> [x]: (x)'\n        bandwidth: '1'\n        latnecy: '2'\n",
        "networks:\n  - name: noc\n    dimensions: [x]\n    interfaces:\n      - name: port\n        endpoint: 'L1[x]'\n        injection_bandwith: '1'\n",
        "scopes:\n  - name: mesh\n    procesors: []\n",
    ] {
        assert!(
            ChipYaml::from_yaml_str(&format!("name: test\n{extra}")).is_err(),
            "{extra}"
        );
    }
    assert!(ProcessorYaml::from_yaml_str("functions: {}\nperformnace: {}\n").is_err());
    let package = Package::new(
        "name: test\nmemories: {L1: null}\n",
        "memories:\n  L1:\n    capacity: 1024\n    word_size: 16\n    banking: {banks: 2, interleaving: 64}\n",
    );
    assert!(mlar_frontend::archs::load_arch(&package.0).is_err());
}

#[test]
fn duplicate_mapping_names_and_bindings_are_rejected() {
    for extra in [
        "dimensions:\n  x: 2\n  x: 4\n",
        "memories:\n  L1: [x]\n  L1: [y]\n",
        "processors:\n  lane: {definition: a.yaml}\n  lane: {definition: b.yaml}\n",
    ] {
        assert!(
            ChipYaml::from_yaml_str(&format!("name: test\n{extra}")).is_err(),
            "{extra}"
        );
    }
    assert!(PerformanceYaml::from_yaml_str("f: []\nf: []\n").is_err());
    let package = Package::new("name: test\nparameters: [X]\n", "memories: {}\n");
    let chip = ChipYaml::from_file(package.0.join("chip.yaml")).unwrap();
    let error = chip
        .build_with_bindings(&package.0, [("X", 2), ("X", 4)])
        .unwrap_err();
    assert!(error.to_string().contains("duplicate binding"));
}

#[test]
fn performance_symbols_require_an_interface_or_explicit_declaration() {
    let function = MlirFunc::with_symbols("copy", Sym::from_names(["BYTES"]));
    for field in ["constraint", "latency", "volume", "throughput"] {
        let value = if field == "constraint" {
            "BYETS > 0"
        } else {
            "BYETS"
        };
        let mut fields = std::collections::BTreeMap::from([
            ("constraint", "BYTES > 0"),
            ("latency", "0"),
            ("volume", "BYTES"),
            ("throughput", "1"),
        ]);
        fields.insert(field, value);
        let yaml = format!(
            "copy:\n  - constraint: '{}'\n    latency: '{}'\n    volume: '{}'\n    throughput: '{}'\n",
            fields["constraint"], fields["latency"], fields["volume"], fields["throughput"]
        );
        let spec = PerformanceYaml::from_yaml_str(&yaml).unwrap();
        let error = spec.model_for_func(&function).unwrap_err().to_string();
        assert!(error.contains("BYETS"), "{field}: {error}");
    }
    let spec = PerformanceYaml::from_yaml_str(
        "copy:\n  - latency: '0'\n    volume: BYTES\n    throughput: bandwidth\n",
    )
    .unwrap();
    assert!(spec.model_for_func(&function).is_err());
    let declared = MlirFunc::with_symbols("copy", Sym::from_names(["BYTES", "bandwidth"]));
    spec.model_for_func(&declared).unwrap();
    for names in [vec!["bandwidth", "bandwidth"], vec!["invalid name"]] {
        let invalid = MlirFunc::with_symbols("copy", Sym::from_names(names));
        assert!(spec.model_for_func(&invalid).is_err());
    }

    let model = FuncPerfModel::builder()
        .simple_time_cost(Expr::Const(0), Expr::sym("BYETS"), Expr::Const(1))
        .build();
    assert!(
        OperationModel::new(function.clone(), model)
            .validate()
            .is_err()
    );
    let model = FuncPerfModel::builder()
        .symbols(["bandwidth"])
        .simple_time_cost(Expr::Const(0), Expr::sym("BYTES"), Expr::sym("bandwidth"))
        .build();
    OperationModel::new(function, model).validate().unwrap();
}

fn network() -> NetworkTopology {
    let axes = vec![Axis::new("x", 2)];
    NetworkTopology::new("noc", axes.clone()).with_link(NetworkLink::new(
        "east",
        AffineMap::identity(&axes),
        Expr::sym("bandwidth"),
    ))
}

#[test]
fn network_symbols_and_interface_axes_are_checked() {
    let builder = || {
        mlar_frontend::ArchitectureBuilder::new("network")
            .axis("x", 2)
            .memory_definition(MemoryDefinition::new("L1", 1024, 16))
            .place_memory("L1", ["x"])
    };
    let error = builder()
        .network(network())
        .build()
        .unwrap_err()
        .to_string();
    assert!(error.contains("undeclared symbol 'bandwidth'"), "{error}");
    let declared = network().with_parameters(["bandwidth"]);
    let architecture = builder().network(declared.clone()).build().unwrap();
    let json = serde_json::to_string(&architecture).unwrap();
    serde_json::from_str::<Architecture>(&json).unwrap();
    let error = builder()
        .network(
            declared.with_interface(NetworkInterface::new(
                "port",
                MemoryEndpoint::parse("L1[xx]")
                    .unwrap()
                    .lower(&[vec!["x".into()]])
                    .unwrap(),
            )),
        )
        .build()
        .unwrap_err()
        .to_string();
    assert!(error.contains("undeclared axis 'xx'"), "{error}");

    let mut json = serde_json::to_value(&architecture).unwrap();
    json["networks"][0]["links"][0]["map"]["expressions"][0] =
        serde_json::json!({"variable": "xx"});
    assert!(serde_json::from_value::<Architecture>(json).is_err());
}

#[test]
fn canonical_objects_reject_unknown_fields_and_invalid_affine_nodes() {
    let architecture = mlar_frontend::ArchitectureBuilder::new("canonical")
        .axis("x", 2)
        .memory_definition(MemoryDefinition::new("L1", 1024, 16))
        .place_memory("L1", ["x"])
        .processor_definition(ProcessorDefinition::new("lane", "", vec![]))
        .connect(
            "lane",
            Connection::parse_named(["x"], [("data", "L1[x]")], [("result", "L1[x]")]).unwrap(),
        )
        .build()
        .unwrap();
    let original = serde_json::to_value(&architecture).unwrap();
    for path in [
        "",
        "/axes/0",
        "/memories/0",
        "/memory_definitions/0",
        "/processors/0",
        "/processors/0/connection",
        "/processors/0/connection/inputs/0",
    ] {
        let mut json = original.clone();
        json.pointer_mut(path)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert("typo".into(), serde_json::json!(1));
        let error = serde_json::from_value::<Architecture>(json)
            .unwrap_err()
            .to_string();
        assert!(error.contains("unknown field"), "{path}: {error}");
    }
    let mut json = original;
    json["processors"][0]["connection"]["inputs"][0]["endpoint"]["indices"][0] =
        serde_json::json!({"expression": {"mod": [{"variable": "x"}, 0]}});
    let error = serde_json::from_value::<Architecture>(json)
        .unwrap_err()
        .to_string();
    assert!(error.contains("divisors must be positive"), "{error}");
    assert!(
        AffineMap::new(
            &[Axis::new("x", 2)],
            &[Axis::new("x", 2)],
            vec![AffineExpr::modulo(AffineExpr::variable("x"), 0)]
        )
        .is_err()
    );
}

#[test]
fn malformed_expressions_unknown_references_and_bad_groups_fail() {
    let function = MlirFunc::with_symbols("copy", Sym::from_names(["BYTES"]));
    let spec = PerformanceYaml::from_yaml_str(
        "copy:\n  - latency: '0'\n    volume: 'BYTES *'\n    throughput: '1'\n",
    )
    .unwrap();
    assert!(spec.model_for_func(&function).is_err());
    for endpoint in ["Missing[x]", "L1[xx]", "L1[x, x]", "L1[x][x]"] {
        let result = mlar_frontend::ArchitectureBuilder::new("bad")
            .axis("x", 2)
            .memory_definition(MemoryDefinition::new("L1", 1024, 16))
            .place_memory("L1", ["x"])
            .processor_definition(ProcessorDefinition::new("lane", "", vec![]))
            .connect("lane", Connection::parse(["x"], [endpoint], []).unwrap())
            .build();
        assert!(result.is_err(), "{endpoint}");
    }
}
